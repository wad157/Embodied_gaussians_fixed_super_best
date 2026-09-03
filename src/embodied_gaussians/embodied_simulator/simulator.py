# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Literal
import os
from pathlib import Path
import sys

import numpy as np
import pysegreduce
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import warp as wp
import warp.sim


def _configure_conda_cuda_headers() -> None:
    """Expose Conda's target-specific CUDA headers to torch JIT extensions."""
    candidate = Path(sys.prefix) / "targets/x86_64-linux/include"
    if not (candidate / "cuda_runtime_api.h").exists():
        return
    current = os.environ.get("CPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if str(candidate) not in entries:
        os.environ["CPATH"] = os.pathsep.join([str(candidate), *entries])


_configure_conda_cuda_headers()

from gsplat.rendering import rasterization
from embodied_gaussians.physics_simulator.simulator import (
    PhysicsRolloutAuxiliaryState,
    Simulator,
    clone_state,
    copy_control,
    copy_state,
)

from embodied_gaussians.embodied_simulator import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState

from embodied_gaussians.embodied_simulator.appearance_optimizer import AppearanceOptimizer
from embodied_gaussians.embodied_simulator.trajectory_appearance import (
    TrajectoryAppearanceResult,
    TrajectoryAppearanceSettings,
    refine_trajectory_gaussian_appearance,
)
from embodied_gaussians.embodied_simulator.frames import Frames
from embodied_gaussians.embodied_simulator.visual_forces import VisualForces, VisualForcesSettings
from embodied_gaussians.physics_simulator.visual_tissue_residual_mapping import (
    TetrahedralGaussianVisualResidualMapper,
    VisualTissueResidualMappingResult,
    equal_camera_visual_loss,
)
from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (
    FlowDepthParticleStateUpdate,
)
from embodied_gaussians.embodied_simulator.warp import (
    accumulate_soft_particle_force_spread_kernel,
    apply_forces_kernel,
    apply_soft_particle_forces_kernel,
    clamp_soft_particle_forces_kernel,
    copy_soft_particle_force_spread_kernel,
    normalize_soft_particle_force_spread_kernel,
    scatter_soft_gaussian_forces_kernel,
    seed_soft_particle_force_spread_kernel,
    update_gaussians_transforms_kernel,
    update_soft_gaussians_transforms_kernel,
    update_visual_forces_kernel,
)


def equal_camera_weighted_loss(
    pixel_loss: torch.Tensor,
    loss_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize image loss inside each camera before averaging cameras."""
    if pixel_loss.ndim != 3:
        raise ValueError(
            f"Expected pixel loss shape (C,H,W), got {pixel_loss.shape}"
        )
    if loss_weights is None:
        camera_losses = pixel_loss.mean(dim=(1, 2))
        active_cameras = torch.ones(
            len(camera_losses),
            dtype=torch.bool,
            device=camera_losses.device,
        )
    else:
        if loss_weights.shape != pixel_loss.shape:
            raise ValueError(
                f"Loss weight shape {loss_weights.shape} does not match "
                f"pixel loss {pixel_loss.shape}"
            )
        camera_weight_sums = loss_weights.sum(dim=(1, 2))
        active_cameras = camera_weight_sums > 0.0
        if not bool(active_cameras.any().item()):
            raise ValueError(
                "Visual-force loss weights contain no active camera pixels"
            )
        camera_losses = (
            (pixel_loss * loss_weights).sum(dim=(1, 2))
            / torch.clamp(camera_weight_sums, min=1.0)
        )
    return (
        camera_losses[active_cameras].mean(),
        camera_losses,
        active_cameras,
    )


def equal_camera_masked_dssim(
    rendered_colors: torch.Tensor,
    target_colors: torch.Tensor,
    pixel_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable masked DSSIM aligned with the formal 11x11 metric."""

    x = rendered_colors.permute(0, 3, 1, 2)
    y = target_colors.permute(0, 3, 1, 2)
    coordinates = torch.arange(-5, 6, device=x.device, dtype=x.dtype)
    gaussian = torch.exp(-0.5 * (coordinates / 1.5).square())
    gaussian = gaussian / gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11)
    mu_x = F.conv2d(x, kernel, padding=5, groups=3)
    mu_y = F.conv2d(y, kernel, padding=5, groups=3)
    sigma_x = F.conv2d(x.square(), kernel, padding=5, groups=3) - mu_x.square()
    sigma_y = F.conv2d(y.square(), kernel, padding=5, groups=3) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, padding=5, groups=3) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = (
        (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        / (
            (mu_x.square() + mu_y.square() + c1)
            * (sigma_x + sigma_y + c2)
        ).clamp_min(1.0e-12)
    ).mean(dim=1)
    # Match the formal scorer's erosion: a pixel is valid only if every pixel
    # in its 11x11 neighborhood is valid.
    valid = 1.0 - F.max_pool2d(
        1.0 - (pixel_weights > 0.0).to(dtype=x.dtype)[:, None],
        kernel_size=11,
        stride=1,
        padding=5,
    )[:, 0]
    weights = pixel_weights * valid
    sums = weights.sum(dim=(1, 2))
    active = sums > 0.0
    if not bool(active.any().item()):
        raise ValueError("DSSIM has no valid camera pixels")
    camera_losses = (
        ((1.0 - ssim) * 0.5 * weights).sum(dim=(1, 2))
        / sums.clamp_min(1.0)
    )
    return camera_losses[active].mean(), camera_losses


@dataclass
class EmbodiedGaussianState:
    # 这是一个“完整快照”：
    # - physics_state: 刚体/关节等物理状态
    # - physics_control: 当前控制输入
    # - gaussian_state: 当前高斯位姿与外观状态
    #
    # 回放、reset、保存初始状态时都依赖这个结构。
    physics_state: warp.sim.State
    physics_control: warp.sim.Control
    gaussian_state: GaussianState


@dataclass
class EmbodiedGaussianRolloutState:
    """A dynamics snapshot plus stateful contact/material auxiliaries."""

    embodied_state: EmbodiedGaussianState
    # XPBD ping-pongs state_0/state_1 at every substep.  state_1 is not merely
    # disposable output: after a shadow rollout its stale candidate contents
    # become the next live output buffer.  Snapshot both buffers so a rejected
    # counterfactual is a complete transaction rather than a one-buffer reset.
    secondary_physics_state: warp.sim.State
    auxiliary_state: PhysicsRolloutAuxiliaryState


@dataclass
class SoftForceScatterResult:
    particle_forces: torch.Tensor
    force_budget_before_total_clamp: torch.Tensor
    total_force_vector_before_total_clamp: torch.Tensor
    global_scale: torch.Tensor


@dataclass(frozen=True)
class TissueVisualAlignmentResult:
    loss: float
    camera_losses: tuple[float, ...]
    camera_weight_sums: tuple[float, ...]
    camera_active_pixel_counts: tuple[int, ...]
    camera_mask_coverage_fractions: tuple[float, ...]


class EmbodiedGaussiansSimulator(Simulator):
    def __init__(
        self,
        builder: EmbodiedGaussiansBuilder,
        device: str = "cuda",
        require_grad=False,
    ):
        # 先初始化底层 physics simulator，再把 Gaussian/visual-force 相关模块挂上来。
        super().__init__(builder, device=device, requires_grad=require_grad)
        self.gaussian_model = builder.gaussian_model
        self.gaussian_state = builder.gaussian_state
        self.bodies_affected_by_visual_forces = builder.bodies_affected_by_visual_forces
        # visual_forces 保存“视觉优化过程中临时更新后的 Gaussian 位姿”
        # 以及由此推回来的 per-Gaussian forces / moments。
        self.visual_forces = VisualForces(
            self.gaussian_model,
            self.gaussian_state,
            bodies_affected_by_visual_forces=self.bodies_affected_by_visual_forces,
        )
        # appearance_optimizer 控制颜色、透明度、尺度等外观参数是否也参与视觉拟合。
        self.appearance_optimizer = AppearanceOptimizer(self.gaussian_state)
        particle_count = self.model.particle_count
        self._soft_particle_forces = torch.zeros(
            (particle_count, 3), device=self.gaussian_model.device, dtype=torch.float32
        )
        self._soft_particle_force_spread_a = torch.zeros_like(
            self._soft_particle_forces
        )
        self._soft_particle_force_spread_b = torch.zeros_like(
            self._soft_particle_forces
        )
        self._soft_particle_force_spread_weight_a = torch.zeros(
            particle_count, device=self.gaussian_model.device, dtype=torch.float32
        )
        self._soft_particle_force_spread_weight_b = torch.zeros_like(
            self._soft_particle_force_spread_weight_a
        )
        self._soft_particle_weight_sums = torch.zeros(
            particle_count, device=self.gaussian_model.device, dtype=torch.float32
        )
        if self.gaussian_model.num_soft_gaussians > 0:
            self._soft_particle_weight_sums.scatter_add_(
                0,
                self.gaussian_model.soft_gaussian_particle_indices.long().reshape(-1),
                self.gaussian_model.soft_gaussian_barycentric_weights.reshape(-1),
            )
        self._soft_force_global_scale = torch.ones(
            1, device=self.gaussian_model.device, dtype=torch.float32
        )
        # Kept after each visual solve so the GUI can display the actual
        # post-clamp particle forces even after physics clears state_0.particle_f.
        self.last_visual_force_loss: torch.Tensor | None = None
        self.last_visual_force_camera_losses: dict[str, float | None] = {}
        self.last_soft_force_scatter: SoftForceScatterResult | None = None
        self.last_soft_visual_force_metrics: dict[str, float | int] | None = None
        self.last_visual_tissue_residual_metrics: dict[str, object] | None = None
        # Learned render appearance is intentionally separate from the fixed
        # appearance used by geometry/material image losses.  Leaving these
        # as None preserves all legacy modes bit-for-bit.
        self._visual_geometry_colors_logits: torch.Tensor | None = None
        self._visual_geometry_opacities_logits: torch.Tensor | None = None

    def set_visual_geometry_appearance_reference(
        self,
        colors_logits: torch.Tensor | None,
        opacities_logits: torch.Tensor | None,
    ) -> None:
        if colors_logits is None or opacities_logits is None:
            self._visual_geometry_colors_logits = None
            self._visual_geometry_opacities_logits = None
            return
        if colors_logits.shape != self.gaussian_state.colors_logits.shape:
            raise ValueError("Geometry color reference shape mismatch")
        if opacities_logits.shape != self.gaussian_state.opacities_logits.shape:
            raise ValueError("Geometry opacity reference shape mismatch")
        self._visual_geometry_colors_logits = colors_logits.detach().clone()
        self._visual_geometry_opacities_logits = (
            opacities_logits.detach().clone()
        )

    def _geometry_soft_appearance(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        colors_logits = self._visual_geometry_colors_logits
        opacities_logits = self._visual_geometry_opacities_logits
        if colors_logits is None or opacities_logits is None:
            colors_logits = self.gaussian_state.colors_logits
            opacities_logits = self.gaussian_state.opacities_logits
        return (
            colors_logits.detach()[ids].sigmoid(),
            opacities_logits.detach()[ids].sigmoid(),
        )

    def scatter_soft_gaussian_forces(
        self,
        gaussian_displacements: torch.Tensor,
        kp: float,
        max_gaussian_force: float = 0.0,
        max_particle_force: float = 0.0,
        max_total_force: float = 0.0,
        max_particle_acceleration: float = 0.0,
        spread_layers: int = 0,
        particle_f=None,
        gaussian_opacities: torch.Tensor | None = None,
    ) -> SoftForceScatterResult:
        """Convert soft Gaussian displacement targets into bounded particle forces.

        This is deliberately independent of rendering.  Only sparse soft
        bindings participate, so rigid PSM and static ground Gaussians cannot
        write particle forces through this path.
        """
        if self.gaussian_model.num_soft_gaussians == 0:
            raise ValueError("Soft force scatter requires soft Gaussian bindings")
        if gaussian_displacements.shape != self.gaussian_model.means.shape:
            raise ValueError(
                "gaussian_displacements must match the full Gaussian means shape"
            )
        if gaussian_displacements.device != self.gaussian_model.device:
            raise ValueError("gaussian_displacements is on the wrong device")
        if gaussian_displacements.dtype != torch.float32:
            raise ValueError("gaussian_displacements must use float32")
        if kp < 0.0:
            raise ValueError("kp must be non-negative")
        if min(
            max_gaussian_force,
            max_particle_force,
            max_total_force,
            max_particle_acceleration,
        ) < 0.0:
            raise ValueError("force limits must be non-negative")
        if spread_layers < 0 or spread_layers > 4:
            raise ValueError("soft force spread_layers must be between 0 and 4")
        if gaussian_opacities is None:
            gaussian_opacities = self.gaussian_state.opacities.detach()
        if gaussian_opacities.shape != self.gaussian_model.opacities.shape:
            raise ValueError("gaussian_opacities has the wrong shape")
        if particle_f is None:
            particle_f = self.state_0.particle_f

        self._soft_particle_forces.zero_()
        wp.launch(
            kernel=scatter_soft_gaussian_forces_kernel,
            dim=self.gaussian_model.num_soft_gaussians,  # type: ignore
            inputs=[
                self.gaussian_model.soft_gaussian_ids,
                self.gaussian_model.soft_gaussian_particle_indices,
                self.gaussian_model.soft_gaussian_barycentric_weights,
                self._soft_particle_weight_sums,
                self.model.particle_inv_mass,
                gaussian_opacities.contiguous(),
                gaussian_displacements.contiguous(),
                float(kp),
                float(max_gaussian_force),
            ],
            outputs=[self._soft_particle_forces],
            device=self.model.device,
        )
        if spread_layers:
            wp.launch(
                kernel=seed_soft_particle_force_spread_kernel,
                dim=self.model.particle_count,
                inputs=[self._soft_particle_forces],
                outputs=[
                    self._soft_particle_force_spread_a,
                    self._soft_particle_force_spread_weight_a,
                ],
                device=self.model.device,
            )
            spread_forces = self._soft_particle_force_spread_a
            spread_weights = self._soft_particle_force_spread_weight_a
            next_forces = self._soft_particle_force_spread_b
            next_weights = self._soft_particle_force_spread_weight_b
            for _ in range(spread_layers):
                next_forces.zero_()
                next_weights.zero_()
                wp.launch(
                    kernel=accumulate_soft_particle_force_spread_kernel,
                    dim=self.model.tet_count,
                    inputs=[
                        self.model.tet_indices,
                        self.model.particle_inv_mass,
                        spread_forces,
                        spread_weights,
                    ],
                    outputs=[next_forces, next_weights],
                    device=self.model.device,
                )
                wp.launch(
                    kernel=normalize_soft_particle_force_spread_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self._soft_particle_forces,
                        next_forces,
                        next_weights,
                    ],
                    device=self.model.device,
                )
                spread_forces, next_forces = next_forces, spread_forces
                spread_weights, next_weights = next_weights, spread_weights
            wp.launch(
                kernel=copy_soft_particle_force_spread_kernel,
                dim=self.model.particle_count,
                inputs=[spread_forces],
                outputs=[self._soft_particle_forces],
                device=self.model.device,
            )
        wp.launch(
            kernel=clamp_soft_particle_forces_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_inv_mass,
                float(max_particle_force),
                float(max_particle_acceleration),
            ],
            outputs=[self._soft_particle_forces],
            device=self.model.device,
        )

        particle_force_norms = torch.linalg.vector_norm(
            self._soft_particle_forces, dim=1
        )
        force_budget = particle_force_norms.sum()
        total_force_vector = self._soft_particle_forces.sum(dim=0)
        if max_total_force > 0.0:
            scale = torch.clamp(
                torch.as_tensor(
                    max_total_force,
                    device=self.gaussian_model.device,
                    dtype=torch.float32,
                )
                / torch.clamp(force_budget, min=1.0e-12),
                max=1.0,
            )
            self._soft_force_global_scale.copy_(scale.reshape(1))
        else:
            self._soft_force_global_scale.fill_(1.0)
        wp.launch(
            kernel=apply_soft_particle_forces_kernel,
            dim=self.model.particle_count,
            inputs=[
                self._soft_force_global_scale,
                self.model.particle_inv_mass,
                self._soft_particle_forces,
            ],
            outputs=[particle_f],
            device=self.model.device,
        )
        return SoftForceScatterResult(
            particle_forces=self._soft_particle_forces,
            force_budget_before_total_clamp=force_budget,
            total_force_vector_before_total_clamp=total_force_vector,
            global_scale=self._soft_force_global_scale,
        )

    def get_specific_environment_state(self, env_ind: int):
        # 从 batched simulator 中抽出某一个环境的单独状态。
        # 这在多环境并行时很有用，比如想单独保存/恢复其中一个 env。
        with torch.no_grad():
            sim = self
            s = wp.to_torch(sim.state_0).reshape(self.num_envs(), -1, 7)[env_ind]
            c = wp.to_torch(sim.control).reshape(self.num_envs(), -1)[env_ind]
            g = sim.gaussian_state.reshape((self.num_envs(), -1, 7)).slice(env_ind).clone()
            s = wp.from_torch(s)
            c = wp.from_torch(c)
            return EmbodiedGaussianState(
                physics_state=s, physics_control=c, gaussian_state=g
            )
    
    def set_specific_environment_state(self, env_ind: int, state: EmbodiedGaussianState):
        # 与上面的 get_specific_environment_state 对应，
        # 把单个环境的状态写回 batched simulator 的指定槽位。
        sim = self
        with torch.no_grad():
            wp.to_torch(sim.state_0).reshape(self.num_envs(), -1, 7)[env_ind] = wp.to_torch(state.physics_state)
            wp.to_torch(sim.control).reshape(self.num_envs(), -1)[env_ind] = wp.to_torch(state.physics_control)
            g = sim.gaussian_state.reshape((self.num_envs(), -1, 7)).slice(env_ind)
            g.copy(state.gaussian_state)

    def embodied_gaussian_state(self):
        # 返回“当前状态对象本身”。
        # 注意 physics_state / control 这里不是深拷贝，gaussian_state 才 clone 了一份。
        s = self.state_0
        c = self.control
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(
            physics_state=s, physics_control=c, gaussian_state=g
        )

    def clone_embodied_gaussian_state(self):
        # 返回一个可安全保存的完整深拷贝。
        # 这通常用于：
        # - 启动时保存 first_state
        # - Reset 时恢复
        # - 中间做回滚
        s = self.clone_state()
        c = self.clone_control()
        g = self.gaussian_state.clone()
        return EmbodiedGaussianState(
            physics_state=s, physics_control=c, gaussian_state=g
        )

    def copy_embodied_gaussian_state(self, state: EmbodiedGaussianState):
        # 用外部快照完整覆盖当前 simulator 状态。
        self.set_state(state.physics_state)
        self.set_control(state.physics_control)
        self.gaussian_state.copy(state.gaussian_state)

    def clone_embodied_gaussian_rollout_state(
        self,
    ) -> EmbodiedGaussianRolloutState:
        """Clone everything needed to replay the next physics steps exactly."""
        return EmbodiedGaussianRolloutState(
            embodied_state=self.clone_embodied_gaussian_state(),
            secondary_physics_state=clone_state(self.state_1),
            auxiliary_state=self.clone_rollout_auxiliary_state(),
        )

    def copy_embodied_gaussian_rollout_state(
        self, state: EmbodiedGaussianRolloutState
    ) -> None:
        self.copy_embodied_gaussian_state(state.embodied_state)
        copy_state(self.state_1, state.secondary_physics_state)
        self.restore_rollout_auxiliary_state(state.auxiliary_state)

    def render_visual_forces(
        self,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
    ):
        # 这个函数专门渲染“视觉优化后的 Gaussian 位姿”，
        # 也就是 self.visual_forces.means / quats，而不是原始 gaussian_state。
        #
        # 它主要服务于调试：
        # 你可以直观看到“视觉力想把高斯推到哪里去”。
        num_images = X_CWs.shape[0]
        with torch.no_grad():
            # gsplat 当前要求 backgrounds 形状与 image batch 对齐。
            backgrounds = background.view(1, 3).expand(num_images, 3).contiguous()
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=X_CWs,
                Ks=Ks,
                width=int(width),
                height=int(height),
                camera_model="pinhole",
                render_mode="RGB",
                packed=False,
                backgrounds=backgrounds,
            )
        return render_colors, render_alphas, info

    def render_gaussians(
        self,
        gaussian_state: GaussianState,
        X_CWs: torch.Tensor,
        Ks: torch.Tensor,
        width: float,
        height: float,
        background: torch.Tensor,
        near_plane=0.01,
        far_plane=3.0,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        **kwargs,
    ):
        # 普通 Gaussian 渲染入口。
        # 这里渲染的是传入的 gaussian_state，通常是当前真实场景状态，
        # 而不是视觉优化过程中的临时状态。
        return render_gaussians(
            gaussian_state,
            X_CWs,
            Ks,
            width,
            height,
            background,
            near_plane,
            far_plane,
            render_mode,
            **kwargs,
        )

    def compute_visual_forces(
        self, settings: VisualForcesSettings, frames: Frames, dt: float
    ):
        # 对外暴露的视觉力计算入口。
        self._compute_visual_forces(settings, frames, dt)

    @staticmethod
    def _scaled_tissue_visual_inputs(
        mapper: TetrahedralGaussianVisualResidualMapper,
        frames: Frames,
        *,
        observations_are_bgr: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        loss_weights = frames.loss_weights_gpu
        if loss_weights is None or float(loss_weights.sum().item()) <= 0.0:
            raise ValueError(
                "Visual tissue alignment requires active pixel weights"
            )
        target_colors = frames.colors_gpu
        if observations_are_bgr:
            target_colors = target_colors.flip(-1)
        image_scale = mapper.settings.image_scale
        render_width = int(frames.width)
        render_height = int(frames.height)
        render_Ks = frames.Ks_gpu
        if image_scale < 1.0:
            render_width = max(1, int(round(render_width * image_scale)))
            render_height = max(1, int(round(render_height * image_scale)))
            scale_x = render_width / float(frames.width)
            scale_y = render_height / float(frames.height)
            target_colors = torch.nn.functional.interpolate(
                target_colors.permute(0, 3, 1, 2),
                size=(render_height, render_width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            loss_weights = torch.nn.functional.interpolate(
                loss_weights[:, None],
                size=(render_height, render_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            render_Ks = frames.Ks_gpu.clone()
            render_Ks[:, 0, :] *= scale_x
            render_Ks[:, 1, :] *= scale_y
        return (
            target_colors,
            loss_weights,
            render_Ks,
            render_width,
            render_height,
        )

    def evaluate_visual_tissue_alignment(
        self,
        mapper: TetrahedralGaussianVisualResidualMapper,
        frames: Frames,
        *,
        observations_are_bgr: bool = False,
    ) -> TissueVisualAlignmentResult:
        """Render the exactly re-skinned tissue and evaluate the RGB loss."""
        (
            target_colors,
            loss_weights,
            render_Ks,
            render_width,
            render_height,
        ) = self._scaled_tissue_visual_inputs(
            mapper,
            frames,
            observations_are_bgr=observations_are_bgr,
        )
        ids = mapper.soft_gaussian_ids
        geometry_colors, geometry_opacities = self._geometry_soft_appearance(
            ids
        )
        with torch.no_grad():
            rendered, _alphas, _info = rasterization(
                means=self.gaussian_state.means.detach()[ids],
                quats=self.gaussian_state.quats.detach()[ids],
                scales=self.gaussian_state.scales.detach()[ids],
                colors=geometry_colors,
                opacities=geometry_opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=render_Ks,
                width=render_width,
                height=render_height,
                camera_model="pinhole",
                render_mode="RGB",
            )
            loss, camera_losses = equal_camera_visual_loss(
                rendered,
                target_colors,
                loss_weights,
                robust_loss_beta=mapper.settings.robust_loss_beta,
                affine_color_calibration=(
                    mapper.settings.photometric_affine_calibration
                ),
                affine_gain_minimum=mapper.settings.photometric_gain_minimum,
                affine_gain_maximum=mapper.settings.photometric_gain_maximum,
                affine_bias_limit=mapper.settings.photometric_bias_limit,
            )
            weight_sums = loss_weights.sum(dim=(1, 2))
            active_pixel_counts = torch.count_nonzero(
                loss_weights > 0.0, dim=(1, 2)
            )
            pixels_per_camera = loss_weights.shape[1] * loss_weights.shape[2]
        return TissueVisualAlignmentResult(
            loss=float(loss.item()),
            camera_losses=tuple(
                float(value) for value in camera_losses.cpu().tolist()
            ),
            camera_weight_sums=tuple(
                float(value) for value in weight_sums.cpu().tolist()
            ),
            camera_active_pixel_counts=tuple(
                int(value) for value in active_pixel_counts.cpu().tolist()
            ),
            camera_mask_coverage_fractions=tuple(
                float(value)
                for value in (
                    active_pixel_counts.to(dtype=torch.float32)
                    / float(pixels_per_camera)
                )
                .cpu()
                .tolist()
            ),
        )

    def solve_visual_tissue_residual(
        self,
        mapper: TetrahedralGaussianVisualResidualMapper,
        frames: Frames,
        *,
        previous_residual: torch.Tensor | None = None,
        dynamic_exclusion_mask: torch.Tensor | None = None,
        observations_are_bgr: bool = False,
        trajectory_particle_indices: torch.Tensor | None = None,
        trajectory_particle_weights: torch.Tensor | None = None,
        trajectory_confidence: torch.Tensor | None = None,
        trajectory_valid_mask: torch.Tensor | None = None,
        trajectory_hold_weight: float = 0.0,
    ) -> VisualTissueResidualMappingResult:
        """Solve a paper-style visual residual without mutating physics.

        The existing visual-force path optimizes independent Gaussian means and
        converts their displacement into force.  This experimental path instead
        lets ``mapper`` optimize physical-particle residuals; the bound Gaussian
        means are reconstructed inside the differentiable render closure.  The
        returned corrected state remains separate until a caller explicitly
        accepts it after contact and trajectory-level validation.
        """
        (
            target_colors,
            loss_weights,
            render_Ks,
            render_width,
            render_height,
        ) = self._scaled_tissue_visual_inputs(
            mapper,
            frames,
            observations_are_bgr=observations_are_bgr,
        )

        soft_gaussian_ids = mapper.soft_gaussian_ids
        packed_quats = self.gaussian_state.quats.detach()[soft_gaussian_ids]
        packed_scales = self.gaussian_state.scales.detach()[soft_gaussian_ids]
        packed_colors, packed_opacities = self._geometry_soft_appearance(
            soft_gaussian_ids
        )

        def render_colors(
            means: torch.Tensor,
            covariances: torch.Tensor | None = None,
        ) -> torch.Tensor:
            colors, _alphas, _info = rasterization(
                means=means,
                quats=packed_quats,
                scales=packed_scales,
                covars=covariances,
                colors=packed_colors,
                opacities=packed_opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=render_Ks,
                width=render_width,
                height=render_height,
                camera_model="pinhole",
                render_mode="RGB",
            )
            return colors

        return mapper.solve(
            physical_positions=wp.to_torch(self.state_0.particle_q),
            base_gaussian_means=self.gaussian_state.means[soft_gaussian_ids],
            target_colors=target_colors,
            pixel_weights=loss_weights,
            render_colors=render_colors,
            previous_residual=previous_residual,
            dynamic_exclusion_mask=dynamic_exclusion_mask,
            trajectory_particle_indices=trajectory_particle_indices,
            trajectory_particle_weights=trajectory_particle_weights,
            trajectory_confidence=trajectory_confidence,
            trajectory_valid_mask=trajectory_valid_mask,
            trajectory_hold_weight=trajectory_hold_weight,
        )

    def refine_trajectory_gaussian_appearance(
        self,
        mapper: TetrahedralGaussianVisualResidualMapper,
        frames: Frames,
        *,
        reference_colors_logits: torch.Tensor,
        reference_opacities_logits: torch.Tensor,
        settings: TrajectoryAppearanceSettings,
        observations_are_bgr: bool = False,
        camera_indices: tuple[int, ...] | None = None,
        loss_weights_override: torch.Tensor | None = None,
        image_scale: float | None = None,
        dssim_weight: float = 0.0,
        apply_result: bool = True,
    ) -> TrajectoryAppearanceResult:
        """Refine tissue color/opacity after the physical trajectory update.

        Position, rotation and scale remain detached because runtime skinning
        owns those quantities.  Only the soft-tissue Gaussian rows can be
        written back; robot and other scene Gaussians are immutable here.
        """

        if dssim_weight < 0.0:
            raise ValueError("DSSIM weight must be non-negative")
        selected = (
            tuple(range(len(frames.X_CWs_opencv_gpu)))
            if camera_indices is None
            else tuple(int(value) for value in camera_indices)
        )
        if not selected:
            raise ValueError("Appearance refinement requires a camera")
        target_colors = frames.colors_gpu[list(selected)]
        if observations_are_bgr:
            target_colors = target_colors.flip(-1)
        loss_weights = (
            frames.loss_weights_gpu[list(selected)]
            if loss_weights_override is None
            else loss_weights_override
        )
        if loss_weights is None:
            raise ValueError("Appearance refinement requires loss weights")
        loss_weights = loss_weights.to(
            device=target_colors.device, dtype=target_colors.dtype
        )
        if loss_weights.shape != target_colors.shape[:3]:
            raise ValueError("Appearance loss weights have the wrong shape")
        render_width = int(frames.width)
        render_height = int(frames.height)
        render_Ks = frames.Ks_gpu[list(selected)].clone()
        viewmats = frames.X_CWs_opencv_gpu[list(selected)]
        scale = mapper.settings.image_scale if image_scale is None else image_scale
        if not 0.0 < scale <= 1.0:
            raise ValueError("Appearance image scale must lie in (0, 1]")
        if scale < 1.0:
            render_width = max(1, int(round(render_width * scale)))
            render_height = max(1, int(round(render_height * scale)))
            scale_x = render_width / float(frames.width)
            scale_y = render_height / float(frames.height)
            target_colors = F.interpolate(
                target_colors.permute(0, 3, 1, 2),
                size=(render_height, render_width),
                mode="area",
            ).permute(0, 2, 3, 1)
            loss_weights = F.interpolate(
                loss_weights[:, None],
                size=(render_height, render_width),
                mode="nearest",
            )[:, 0]
            render_Ks[:, 0, :] *= scale_x
            render_Ks[:, 1, :] *= scale_y
        ids = mapper.soft_gaussian_ids
        means = self.gaussian_state.means.detach()[ids]
        quats = self.gaussian_state.quats.detach()[ids]
        scales = self.gaussian_state.scales.detach()[ids]
        current_colors = self.gaussian_state.colors_logits.detach()[ids]
        current_opacities = self.gaussian_state.opacities_logits.detach()[ids]
        if reference_colors_logits.shape == self.gaussian_state.colors_logits.shape:
            reference_colors_logits = reference_colors_logits[ids]
        if (
            reference_opacities_logits.shape
            == self.gaussian_state.opacities_logits.shape
        ):
            reference_opacities_logits = reference_opacities_logits[ids]

        def visual_loss(
            colors: torch.Tensor,
            opacities: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            rendered, _alphas, _info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                colors=colors,
                opacities=opacities,
                viewmats=viewmats,
                Ks=render_Ks,
                width=render_width,
                height=render_height,
                camera_model="pinhole",
                render_mode="RGB",
            )
            mse, camera_mse = equal_camera_visual_loss(
                rendered,
                target_colors,
                loss_weights,
                # Geometry fitting deliberately profiles exposure so a color
                # mismatch cannot move tissue.  Appearance fitting has the
                # opposite job: minimize the raw RGB MSE used by PSNR, without
                # explaining that mismatch away through an affine transform.
                robust_loss_beta=0.0,
                affine_color_calibration=False,
            )
            if dssim_weight <= 0.0:
                return mse, camera_mse
            dssim, camera_dssim = equal_camera_masked_dssim(
                rendered, target_colors, loss_weights
            )
            return (
                mse + dssim_weight * dssim,
                camera_mse + dssim_weight * camera_dssim,
            )

        result = refine_trajectory_gaussian_appearance(
            current_colors_logits=current_colors,
            current_opacities_logits=current_opacities,
            reference_colors_logits=reference_colors_logits,
            reference_opacities_logits=reference_opacities_logits,
            visual_loss=visual_loss,
            settings=settings,
        )
        if result.accepted and apply_result:
            with torch.no_grad():
                self.gaussian_state.colors_logits.index_copy_(
                    0, ids, result.colors_logits
                )
                self.gaussian_state.opacities_logits.index_copy_(
                    0, ids, result.opacities_logits
                )
        return result

    def apply_flow_depth_particle_state_update(
        self,
        result: FlowDepthParticleStateUpdate,
        *,
        mapper: TetrahedralGaussianVisualResidualMapper,
        maximum_anchor_regression_m: float = 2.0e-4,
        maximum_penetration_m: float | None = None,
        penetration_tolerance_m: float = 1.0e-4,
        maximum_backtracks: int = 8,
        backtrack_factor: float = 0.5,
        maximum_local_inversion_projection_passes: int = 8,
        enforce_inversion_gate: bool = True,
        enforce_low_volume_gate: bool = True,
        enforce_anchor_gate: bool = True,
        enforce_penetration_gate: bool = True,
    ) -> bool:
        """Apply a flow/depth q/qd candidate with configurable safety gates.

        The observer computes its candidate outside the simulator.  This method
        preserves fixed particles and locally removes only corrections incident
        to a newly inverted tetrahedron.  Low-volume, anchor and penetration
        gates remain available to conservative callers, but trajectory mode can
        monitor them without letting one local warning suppress the complete
        dense visual update.  The normal material/contact projector still runs
        on the following physics step.
        """

        result.validate()
        if maximum_anchor_regression_m < 0.0:
            raise ValueError("Maximum anchor regression must be non-negative")
        if maximum_penetration_m is not None and maximum_penetration_m < 0.0:
            raise ValueError("Maximum penetration must be non-negative")
        if penetration_tolerance_m < 0.0:
            raise ValueError("Penetration tolerance must be non-negative")
        if maximum_backtracks < 0:
            raise ValueError("Maximum backtracks must be non-negative")
        if maximum_local_inversion_projection_passes < 0:
            raise ValueError(
                "Maximum local inversion projection passes must be non-negative"
            )
        if not 0.0 < backtrack_factor < 1.0:
            raise ValueError("Backtrack factor must stay in (0, 1)")

        particle_q = wp.to_torch(self.state_0.particle_q)
        particle_qd = wp.to_torch(self.state_0.particle_qd)
        if result.corrected_positions.shape != tuple(particle_q.shape):
            raise ValueError("Flow-depth position candidate has the wrong shape")
        if result.corrected_velocities.shape != tuple(particle_qd.shape):
            raise ValueError("Flow-depth velocity candidate has the wrong shape")
        requested_q = torch.as_tensor(
            result.corrected_positions,
            device=particle_q.device,
            dtype=particle_q.dtype,
        )
        requested_qd = torch.as_tensor(
            result.corrected_velocities,
            device=particle_qd.device,
            dtype=particle_qd.dtype,
        )
        original_q = particle_q.detach().clone()
        original_qd = particle_qd.detach().clone()
        fixed_mask = mapper.fixed_mask.to(
            device=particle_q.device, dtype=torch.bool
        )
        finite = bool(
            torch.isfinite(requested_q).all().item()
            and torch.isfinite(requested_qd).all().item()
        )
        fixed_positions_preserved = bool(
            torch.equal(requested_q[fixed_mask], original_q[fixed_mask])
        )
        fixed_velocities_preserved = bool(
            torch.equal(requested_qd[fixed_mask], original_qd[fixed_mask])
        )
        baseline_quality = mapper.physical_quality_metrics(original_q)
        baseline_contact = self.triangle_skin_contact_metrics() or {}
        rejection_reasons: list[str] = []
        if not finite:
            rejection_reasons.append("non_finite")
        if not fixed_positions_preserved:
            rejection_reasons.append("fixed_position_changed")
        if not fixed_velocities_preserved:
            rejection_reasons.append("fixed_velocity_changed")
        candidate_q = original_q
        candidate_qd = original_qd
        candidate_quality = baseline_quality
        candidate_contact: dict = {}
        accepted_scale = 0.0
        backtrack_count = 0
        backtrack_records: list[dict] = []
        local_projection_passes = 0
        locally_suppressed_mask = torch.zeros_like(fixed_mask)
        if not rejection_reasons:
            requested_position_delta = requested_q - original_q
            requested_velocity_delta = requested_qd - original_qd
            # Do not globally shrink hundreds of valid observations because a
            # few local surface nodes would invert their incident tetrahedra.
            # Suppress only those incident nodes, repeat to a fixed point, and
            # leave the rest of the visual correction at full amplitude.
            tet_indices = mapper.tet_indices.to(
                device=particle_q.device, dtype=torch.long
            )
            rest_volumes = mapper.rest_volumes.to(
                device=particle_q.device, dtype=particle_q.dtype
            )

            def volume_ratios(positions: torch.Tensor) -> torch.Tensor:
                points = positions[tet_indices]
                volumes = torch.linalg.det(
                    torch.stack(
                        (
                            points[:, 1] - points[:, 0],
                            points[:, 2] - points[:, 0],
                            points[:, 3] - points[:, 0],
                        ),
                        dim=-1,
                    )
                ) / 6.0
                return volumes / rest_volumes

            baseline_volume_ratio = volume_ratios(original_q)
            for local_projection_passes in range(
                maximum_local_inversion_projection_passes + 1
            ):
                projected_ratio = volume_ratios(
                    original_q + requested_position_delta
                )
                newly_inverted = (
                    ~torch.isfinite(projected_ratio)
                    | (
                        (baseline_volume_ratio > 0.0)
                        & (projected_ratio <= 0.0)
                    )
                )
                if not bool(newly_inverted.any().item()):
                    break
                if (
                    local_projection_passes
                    == maximum_local_inversion_projection_passes
                ):
                    break
                violating_particles = torch.unique(
                    tet_indices[newly_inverted].reshape(-1)
                )
                new_particles = violating_particles[
                    ~locally_suppressed_mask[violating_particles]
                ]
                if not len(new_particles):
                    break
                locally_suppressed_mask[new_particles] = True
                requested_position_delta = requested_position_delta.clone()
                requested_velocity_delta = requested_velocity_delta.clone()
                requested_position_delta[new_particles] = 0.0
                requested_velocity_delta[new_particles] = 0.0
            for backtrack_count in range(maximum_backtracks + 1):
                scale = backtrack_factor**backtrack_count
                candidate_q = original_q + scale * requested_position_delta
                candidate_qd = original_qd + scale * requested_velocity_delta
                candidate_quality = mapper.physical_quality_metrics(candidate_q)
                reasons: list[str] = []
                if enforce_inversion_gate and (
                    int(candidate_quality["inverted_tetrahedra"])
                    > int(baseline_quality["inverted_tetrahedra"])
                ):
                    reasons.append("new_tetrahedron_inversion")
                if enforce_low_volume_gate and (
                    int(candidate_quality["tetrahedra_below_volume_floor"])
                    > int(baseline_quality["tetrahedra_below_volume_floor"])
                ):
                    reasons.append("new_low_volume_tetrahedron")
                candidate_contact = {}
                if not reasons:
                    with torch.no_grad():
                        particle_q.copy_(candidate_q)
                        particle_qd.copy_(candidate_qd)
                    self.update_gaussian_transforms()
                    candidate_contact = (
                        self.triangle_skin_contact_metrics() or {}
                    )
                    baseline_anchor = float(
                        baseline_contact.get(
                            "persistent_grip_anchor_error_maximum_m", 0.0
                        )
                    )
                    candidate_anchor = float(
                        candidate_contact.get(
                            "persistent_grip_anchor_error_maximum_m", 0.0
                        )
                    )
                    if enforce_anchor_gate and (
                        not np.isfinite(candidate_anchor)
                        or candidate_anchor
                        > baseline_anchor + maximum_anchor_regression_m
                    ):
                        reasons.append("grip_anchor_regressed")
                    if (
                        enforce_penetration_gate
                        and maximum_penetration_m is not None
                    ):
                        baseline_penetration = float(
                            baseline_contact.get("maximum_penetration_m", 0.0)
                        )
                        candidate_penetration = float(
                            candidate_contact.get("maximum_penetration_m", 0.0)
                        )
                        allowed_penetration = max(
                            float(maximum_penetration_m),
                            baseline_penetration + penetration_tolerance_m,
                        )
                        if (
                            not np.isfinite(candidate_penetration)
                            or candidate_penetration > allowed_penetration
                        ):
                            reasons.append("penetration_regressed")
                backtrack_records.append(
                    {
                        "scale": float(scale),
                        "reasons": tuple(reasons),
                        "inverted_tetrahedra": int(
                            candidate_quality["inverted_tetrahedra"]
                        ),
                        "tetrahedra_below_volume_floor": int(
                            candidate_quality[
                                "tetrahedra_below_volume_floor"
                            ]
                        ),
                        "minimum_volume_ratio": float(
                            candidate_quality["minimum_volume_ratio"]
                        ),
                    }
                )
                if not reasons:
                    accepted_scale = float(scale)
                    break
                with torch.no_grad():
                    particle_q.copy_(original_q)
                    particle_qd.copy_(original_qd)
                self.update_gaussian_transforms()
            if accepted_scale == 0.0:
                rejection_reasons.extend(backtrack_records[-1]["reasons"])

        accepted = not rejection_reasons
        if not accepted:
            with torch.no_grad():
                particle_q.copy_(original_q)
                particle_qd.copy_(original_qd)
            self.update_gaussian_transforms()
        position_norm = torch.linalg.vector_norm(
            candidate_q - original_q, dim=1
        )
        velocity_norm = torch.linalg.vector_norm(
            candidate_qd - original_qd, dim=1
        )
        self.last_flow_depth_particle_update_metrics = {
            "accepted": accepted,
            "rejection_reason": "+".join(rejection_reasons),
            "accepted_scale": accepted_scale,
            "backtrack_count": backtrack_count,
            "backtrack_records": tuple(backtrack_records),
            "local_inversion_projection_passes": int(
                local_projection_passes
            ),
            "locally_suppressed_particles": int(
                torch.count_nonzero(locally_suppressed_mask).item()
            ),
            "enforced_gates": {
                "new_tetrahedron_inversion": bool(enforce_inversion_gate),
                "new_low_volume_tetrahedron": bool(enforce_low_volume_gate),
                "grip_anchor_regression": bool(enforce_anchor_gate),
                "penetration_regression": bool(enforce_penetration_gate),
            },
            "valid_tracks": int(np.count_nonzero(result.track_valid)),
            "updated_particles": int(
                torch.count_nonzero(position_norm > 0.0).item()
            ),
            "velocity_updated_particles": int(
                torch.count_nonzero(velocity_norm > 0.0).item()
            ),
            "maximum_position_correction_m": float(
                position_norm.max().item()
            ),
            "requested_maximum_position_correction_m": float(
                np.linalg.norm(result.position_correction, axis=1).max(
                    initial=0.0
                )
            ),
            "rms_position_correction_m": float(
                torch.sqrt(torch.mean((candidate_q - original_q).square())).item()
            ),
            "maximum_velocity_correction_m_s": float(
                velocity_norm.max().item()
            ),
            "requested_maximum_velocity_correction_m_s": float(
                np.linalg.norm(result.velocity_correction, axis=1).max(
                    initial=0.0
                )
            ),
            "baseline_physical_quality": baseline_quality,
            "candidate_physical_quality": candidate_quality,
            "baseline_contact": baseline_contact,
            "candidate_contact": candidate_contact,
            "mode_2_gaussian_update_called": accepted,
        }
        return accepted

    def apply_visual_tissue_residual(
        self,
        result: VisualTissueResidualMappingResult,
        *,
        require_visual_improvement: bool = True,
        mapper: TetrahedralGaussianVisualResidualMapper | None = None,
        frames: Frames | None = None,
        observations_are_bgr: bool = False,
        observation_dt_s: float | None = None,
        velocity_correction_gain: float = 0.0,
        maximum_velocity_correction_m_s: float = 0.0,
        dynamic_exclusion_mask: torch.Tensor | None = None,
    ) -> bool:
        """Accept a bounded position/velocity visual state correction.

        The optional velocity term is the beta update of an alpha-beta state
        observer: ``qd += beta * position_innovation / observation_dt``.  It is
        capped per particle and never touches fixed or live kinematic-control
        nodes.  Because it is written before snapshotting a candidate, the
        existing no-vision persistence and cross-frame RGB gates validate its
        future effect instead of admitting an unchecked impulse.
        """
        if velocity_correction_gain < 0.0:
            raise ValueError("Velocity correction gain must be non-negative")
        velocity_update_requested = velocity_correction_gain > 0.0
        if velocity_update_requested:
            if mapper is None:
                raise ValueError("Velocity correction requires the residual mapper")
            if observation_dt_s is None or observation_dt_s <= 0.0:
                raise ValueError("Velocity correction requires a positive observation dt")
            if maximum_velocity_correction_m_s <= 0.0:
                raise ValueError("Velocity correction cap must be positive")
        finite = bool(
            torch.isfinite(result.corrected_positions).all().item()
            and torch.isfinite(result.residual).all().item()
        )
        volume_safe = result.newly_inverted_tetrahedra == 0
        if (mapper is None) != (frames is None):
            raise ValueError(
                "Exact residual validation requires both mapper and frames"
            )
        exact_alignment = None
        exact_final_visual_loss = result.final_visual_loss
        exact_final_camera_visual_losses = result.final_camera_visual_losses
        original_positions = None
        # Validate the exact state that would actually be displayed and roll it
        # back if the differentiable candidate does not survive runtime
        # skinning/rasterization.
        if finite and volume_safe and mapper is not None and frames is not None:
            particle_q = wp.to_torch(self.state_0.particle_q)
            original_positions = particle_q.detach().clone()
            with torch.no_grad():
                particle_q.copy_(result.corrected_positions)
            self.update_gaussian_transforms()
            exact_alignment = self.evaluate_visual_tissue_alignment(
                mapper,
                frames,
                observations_are_bgr=observations_are_bgr,
            )
            exact_final_visual_loss = exact_alignment.loss
            exact_final_camera_visual_losses = exact_alignment.camera_losses
        visual_improved = bool(
            exact_final_visual_loss < result.initial_visual_loss
        )
        camera_visual_safe = all(
            weight_sum <= 0.0
            or final_loss
            <= initial_loss + max(1.0e-6, 5.0e-3 * initial_loss)
            for initial_loss, final_loss, weight_sum in zip(
                result.initial_camera_visual_losses,
                exact_final_camera_visual_losses,
                result.camera_weight_sums,
            )
        )
        accepted = bool(
            finite
            and volume_safe
            and (
                (visual_improved and camera_visual_safe)
                or not require_visual_improvement
            )
        )
        if not finite:
            rejection_reason = "non_finite"
        elif not volume_safe:
            rejection_reason = "new_tetrahedron_inversion"
        elif require_visual_improvement and not visual_improved:
            rejection_reason = "visual_loss_not_improved"
        elif require_visual_improvement and not camera_visual_safe:
            rejection_reason = "camera_visual_loss_regressed"
        else:
            rejection_reason = ""

        if accepted and original_positions is None:
            particle_q = wp.to_torch(self.state_0.particle_q)
            if particle_q.shape != result.corrected_positions.shape:
                raise ValueError(
                    "Visual residual particle count does not match simulation"
                )
            with torch.no_grad():
                particle_q.copy_(result.corrected_positions)
            self.update_gaussian_transforms()
        elif not accepted and original_positions is not None:
            with torch.no_grad():
                wp.to_torch(self.state_0.particle_q).copy_(original_positions)
            self.update_gaussian_transforms()

        velocity_correction = torch.zeros_like(result.residual)
        velocity_updated_particles = 0
        if accepted and velocity_update_requested:
            assert mapper is not None
            assert observation_dt_s is not None
            velocity_correction = (
                result.residual
                * (float(velocity_correction_gain) / float(observation_dt_s))
            )
            correction_norm = torch.linalg.vector_norm(
                velocity_correction, dim=1, keepdim=True
            )
            velocity_correction = velocity_correction * torch.clamp(
                float(maximum_velocity_correction_m_s)
                / correction_norm.clamp_min(1.0e-12),
                max=1.0,
            )
            exclusion = mapper.fixed_mask
            if dynamic_exclusion_mask is not None:
                dynamic_exclusion = dynamic_exclusion_mask.detach().to(
                    device=exclusion.device, dtype=torch.bool
                )
                if dynamic_exclusion.shape != exclusion.shape:
                    raise ValueError(
                        "Velocity dynamic exclusion mask has the wrong shape"
                    )
                exclusion = exclusion | dynamic_exclusion
            velocity_correction = velocity_correction.masked_fill(
                exclusion[:, None], 0.0
            )
            velocity_updated_particles = int(
                torch.count_nonzero(
                    torch.linalg.vector_norm(velocity_correction, dim=1)
                    > 0.0
                ).item()
            )
            with torch.no_grad():
                wp.to_torch(self.state_0.particle_qd).add_(
                    velocity_correction
                )

        reduction = (
            (result.initial_visual_loss - exact_final_visual_loss)
            / max(result.initial_visual_loss, 1.0e-12)
        )
        self.last_visual_tissue_residual_metrics = {
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "initial_visual_loss": result.initial_visual_loss,
            "final_visual_loss": result.final_visual_loss,
            "exact_final_visual_loss": exact_final_visual_loss,
            "initial_camera_visual_losses": (
                result.initial_camera_visual_losses
            ),
            "final_camera_visual_losses": (
                result.final_camera_visual_losses
            ),
            "exact_final_camera_visual_losses": (
                exact_final_camera_visual_losses
            ),
            "camera_weight_sums": result.camera_weight_sums,
            "camera_active_pixel_counts": (
                result.camera_active_pixel_counts
            ),
            "camera_mask_coverage_fractions": (
                result.camera_mask_coverage_fractions
            ),
            "visual_loss_reduction_fraction": reduction,
            "maximum_residual_m": result.maximum_residual_m,
            "rms_residual_m": result.rms_residual_m,
            "initial_minimum_volume_ratio": (
                result.initial_minimum_volume_ratio
            ),
            "minimum_volume_ratio": result.minimum_volume_ratio,
            "inverted_tetrahedra": result.inverted_tetrahedra,
            "initial_inverted_tetrahedra": (
                result.initial_inverted_tetrahedra
            ),
            "newly_inverted_tetrahedra": result.newly_inverted_tetrahedra,
            "dynamically_excluded_particles": (
                result.dynamically_excluded_particles
            ),
            "locally_frozen_particles": result.locally_frozen_particles,
            "local_volume_projection_passes": (
                result.local_volume_projection_passes
            ),
            "backtrack_count": result.backtrack_count,
            "velocity_correction_enabled": velocity_update_requested,
            "velocity_correction_gain": float(velocity_correction_gain),
            "velocity_observation_dt_s": (
                None if observation_dt_s is None else float(observation_dt_s)
            ),
            "velocity_correction_maximum_m_s": float(
                torch.linalg.vector_norm(velocity_correction, dim=1).max().item()
            ),
            "velocity_correction_rms_m_s": float(
                torch.sqrt(torch.mean(velocity_correction.square())).item()
            ),
            "velocity_updated_particles": velocity_updated_particles,
        }
        return accepted

    def clear_visual_tissue_residual_metrics(self) -> None:
        self.last_visual_tissue_residual_metrics = None

    def apply_soft_visual_forces(
        self, settings: VisualForcesSettings
    ) -> SoftForceScatterResult | None:
        """Scatter the optimizer's soft Gaussian mean targets to particles."""
        if not settings.enable_soft_particle_forces:
            self.last_soft_visual_force_metrics = None
            return None
        displacements = (
            self.visual_forces.means.detach() - self.gaussian_state.means.detach()
        )
        result = self.scatter_soft_gaussian_forces(
            displacements,
            kp=settings.kp,
            max_gaussian_force=settings.soft_max_gaussian_force,
            max_particle_force=settings.soft_max_particle_force,
            max_total_force=settings.soft_max_total_force,
            max_particle_acceleration=settings.soft_max_particle_acceleration,
            spread_layers=settings.soft_force_spread_layers,
        )
        with torch.no_grad():
            particle_force_norms = torch.linalg.vector_norm(
                result.particle_forces, dim=1
            )
            particle_inverse_mass = wp.to_torch(
                self.model.particle_inv_mass
            ).detach()
            particle_acceleration_norms = (
                particle_force_norms * particle_inverse_mass
            )
            global_scale = result.global_scale.reshape(())
            diagnostics = torch.stack(
                (
                    torch.linalg.vector_norm(displacements, dim=1).max(),
                    result.force_budget_before_total_clamp.reshape(()),
                    result.force_budget_before_total_clamp.reshape(())
                    * global_scale,
                    particle_force_norms.max() * global_scale,
                    particle_acceleration_norms.max() * global_scale,
                    global_scale,
                    (particle_force_norms > 1.0e-12).sum().float(),
                )
            ).detach().cpu()
        (
            target_delta_max_m,
            force_budget_before_n,
            applied_force_budget_n,
            maximum_particle_force_n,
            maximum_particle_acceleration_m_s2,
            scale,
            active_particles,
        ) = diagnostics.tolist()
        self.last_soft_visual_force_metrics = {
            "target_delta_max_m": float(target_delta_max_m),
            "force_budget_before_n": float(force_budget_before_n),
            "applied_force_budget_n": float(applied_force_budget_n),
            "maximum_particle_force_n": float(maximum_particle_force_n),
            "maximum_particle_acceleration_m_s2": float(
                maximum_particle_acceleration_m_s2
            ),
            "global_scale": float(scale),
            "active_particles": int(active_particles),
        }
        return result

    def reapply_last_soft_visual_forces(self, particle_f=None) -> bool:
        """Hold the latest bounded soft force between image optimizations."""
        scatter = self.last_soft_force_scatter
        if scatter is None:
            return False
        if particle_f is None:
            particle_f = self.state_0.particle_f
        wp.launch(
            kernel=apply_soft_particle_forces_kernel,
            dim=self.model.particle_count,
            inputs=[
                scatter.global_scale,
                self.model.particle_inv_mass,
                scatter.particle_forces,
            ],
            outputs=[particle_f],
            device=self.model.device,
        )
        return True

    def clear_soft_visual_force_cache(self) -> None:
        """Discard held force when resetting or disabling visual feedback."""
        self.last_soft_force_scatter = None
        self.last_soft_visual_force_metrics = None
        self._soft_particle_forces.zero_()
        self._soft_force_global_scale.fill_(1.0)

    def update_gaussian_transforms(self):
        # 根据当前 body_q 更新每个 Gaussian 在世界坐标系下的位姿。
        # 也就是把“高斯绑定在哪个 body 上”的信息真正转换成当前渲染状态。
        update_gaussian_transforms(
            self.gaussian_model,
            self.state_0.body_q,
            self.gaussian_state,
            particle_q=self.state_0.particle_q,
        )

    def _compute_visual_forces(
        self, settings: VisualForcesSettings, frames: Frames, dt: float
    ):
        # 这是视觉力的核心流程：
        # 1. 用当前 gaussian_state 初始化一份可优化的临时 Gaussian 位姿
        # 2. 对着真实观测 frames 做若干步渲染误差优化
        # 3. 把“优化前后位姿差”转成 per-Gaussian 力/力矩
        # 4. 聚合成 per-body 总力
        # 5. 施加到物理系统的 body_f 上
        if settings.iterations <= 0:
            self.last_visual_force_loss = None
            self.last_visual_force_camera_losses = {}
            self.last_soft_force_scatter = None
            self.last_soft_visual_force_metrics = None
            return
        loss_weights = frames.loss_weights_gpu
        if settings.enable_soft_particle_forces and settings.require_loss_weights_for_soft:
            if loss_weights is None:
                raise ValueError(
                    "Soft visual forces require per-pixel loss weights; provide "
                    "visible_tissue & ~dilated_tool confidence weights first"
                )
            if float(loss_weights.sum().item()) <= 0.0:
                raise ValueError("Soft visual-force loss weights contain no valid pixels")

        with torch.no_grad():
            # 每次都从“当前真实高斯状态”出发，而不是沿用上一次优化结果。
            self.visual_forces.means.copy_(self.gaussian_state.means)
            self.visual_forces.quats.copy_(self.gaussian_state.quats)

        if settings.reset_optimizer_each_step:
            self.visual_forces.optimizer.reset_internal_state()
        self.visual_forces.set_learnings_rates([settings.lr_means, settings.lr_quats])
        self.appearance_optimizer.set_learnings_rates(
            [settings.lr_color, settings.lr_opacity, settings.lr_scale]
        )

        for _ in range(settings.iterations):
            # 从所有观测相机视角渲染当前“可优化高斯状态”。
            render_colors, render_alphas, info = rasterization(
                means=self.visual_forces.means,
                quats=self.visual_forces.quats,
                scales=self.gaussian_state.scales,
                colors=self.gaussian_state.colors,
                opacities=self.gaussian_state.opacities,
                viewmats=frames.X_CWs_opencv_gpu,
                Ks=frames.Ks_gpu,
                width=int(frames.width),
                height=int(frames.height),
                camera_model="pinhole",
                render_mode="RGB",
            )

            # Dataset adapters may fix the observation channel order and use
            # a robust loss before turning the residual into pose updates.
            target_colors = frames.colors_gpu
            if settings.observations_are_bgr:
                target_colors = target_colors.flip(-1)
            if settings.robust_loss_beta > 0.0:
                pixel_loss = torch.nn.functional.smooth_l1_loss(
                    render_colors,
                    target_colors,
                    beta=settings.robust_loss_beta,
                    reduction="none",
                )
            else:
                pixel_loss = torch.nn.functional.mse_loss(
                    render_colors, target_colors, reduction="none"
                )
            pixel_loss = pixel_loss.mean(dim=-1)
            # Normalize each camera independently before averaging. Otherwise a
            # larger tissue mask would dominate the gradient, and adding the
            # right camera would also change the effective force scale.
            loss, camera_losses, active_cameras = equal_camera_weighted_loss(
                pixel_loss,
                loss_weights,
            )
            # ideas: add a loss that pushes the colors back to their orignal values or to some sort of ema colors
            # ideas: allow the gaussians to jitter a bit while anchoring them to the original positions

            self.visual_forces.zero_grad()
            self.appearance_optimizer.zero_grad()
            loss.backward()
            self.visual_forces.step()
            self.appearance_optimizer.step()

        self.last_visual_force_loss = loss.detach()
        self.last_visual_force_camera_losses = {
            name: (
                float(camera_losses[index].detach().item())
                if bool(active_cameras[index].item())
                else None
            )
            for index, name in enumerate(frames.names)
        }

        # 把“原始 Gaussian 状态”和“视觉优化后的 Gaussian 状态”做比较，
        # 转成每个 Gaussian 对应的力与力矩。
        wp.launch(
            kernel=update_visual_forces_kernel,
            dim=self.gaussian_model.num_gaussians,  # type: ignore
            inputs=[
                settings.kp,
                self.gaussian_state.means.detach(),
                self.gaussian_state.quats.detach(),
                self.gaussian_state.opacities.detach(),
                self.visual_forces.means.detach(),
                self.visual_forces.quats.detach(),
                self.gaussian_model.body_ids.detach(),
                self.state_0.body_q,
                self.visual_forces.forces.detach(),
                self.visual_forces.moments.detach(),
            ],
        )

        # 上一步得到的是 per-Gaussian 力。
        # 这里按 body 分段求和，得到真正能施加到刚体上的总力。
        pysegreduce.reduce_vec3f(
            self.visual_forces.forces.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_forces.data_ptr(),
            0,
        ) # Replace this with segmented reduce when it is implemented in warp

        # 力矩也做同样的按 body 聚合。
        pysegreduce.reduce_vec3f(
            self.visual_forces.moments.data_ptr(),
            self.visual_forces._start_inds.data_ptr(),
            self.visual_forces._end_inds.data_ptr(),
            len(self.visual_forces._start_inds),
            self.visual_forces._total_moments.data_ptr(),
            0,
        )

        # 最后把聚合后的总力 / 总力矩真正写入 body_f，
        # 供后续 physics step 使用。
        wp.launch(
            kernel=apply_forces_kernel,
            dim=self.visual_forces._num_bodies,  # type: ignore
            inputs=[
                dt,
                self.visual_forces._total_forces,
                self.visual_forces._total_moments,
                self.visual_forces._body_ids,
                self.visual_forces._gaussian_counts,
                self.visual_forces._apply_physics_forces,
                int(settings.normalize_forces_by_gaussian_count),
                settings.max_force,
                settings.max_moment,
                self.state_0.body_f,
            ],
        )
        self.last_soft_force_scatter = self.apply_soft_visual_forces(settings)


def render_gaussians(
    gaussian_state: GaussianState,
    X_CWs: torch.Tensor,
    Ks: torch.Tensor,
    width: float,
    height: float,
    background: torch.Tensor,
    near_plane=0.01,
    far_plane=3.0,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    **kwargs,
):
    # 这是一个模块级通用渲染函数，既可以被 simulator 调，也可以被 viewer 直接调。
    # 输入是一份 GaussianState 和一批相机参数，输出渲染颜色、alpha 和底层 meta 信息。
    num_images = X_CWs.shape[0]
    with torch.no_grad():
        # 与 render_visual_forces 一样，background 要扩成与图像 batch 数量一致。
        backgrounds = background.view(1, 3).expand(num_images, 3).contiguous()
        render_colors, render_alphas, info = rasterization(
            means=gaussian_state.means,
            quats=gaussian_state.quats,
            scales=gaussian_state.scales,
            colors=gaussian_state.colors,
            opacities=gaussian_state.opacities,
            viewmats=X_CWs,
            Ks=Ks,
            width=int(width),
            height=int(height),
            camera_model="pinhole",
            render_mode=render_mode,
            packed=False,
            backgrounds=backgrounds,
            near_plane=near_plane,
            far_plane=far_plane,
            **kwargs,
        )
    return render_colors, render_alphas, info


def update_gaussian_transforms(
    model: GaussianModel,
    body_q,
    out_state: GaussianState,
    particle_q=None,
):
    # 把“高斯在各自刚体局部坐标系里的初始位姿”
    # 变换成“当前世界坐标系里的高斯位姿”。
    #
    # 这是 physics 和 gaussian 渲染之间最关键的同步点之一。
    if model.num_gaussians == 0:
        return
    if model.num_body_gaussians > 0:
        if body_q is None:
            raise ValueError("Rigid Gaussian transforms require body_q")
        wp.launch(
            kernel=update_gaussians_transforms_kernel,
            dim=model.num_gaussians,  # type: ignore
            inputs=[
                model.means,
                model.quats,
                model.body_ids,
                body_q,
            ],
            outputs=[
                out_state.means,
                out_state.quats,
            ],
        )
    if model.num_soft_gaussians == 0:
        return
    if particle_q is None:
        raise ValueError("Soft Gaussian skinning requires particle_q")
    wp.launch(
        kernel=update_soft_gaussians_transforms_kernel,
        dim=model.num_soft_gaussians,  # type: ignore
        inputs=[
            model.soft_gaussian_ids,
            model.soft_gaussian_particle_indices,
            model.soft_gaussian_tet_ids,
            model.soft_gaussian_barycentric_weights,
            model.soft_gaussian_rest_offsets,
            model.soft_gaussian_binding_modes,
            model.soft_gaussian_face_particle_indices,
            model.soft_gaussian_rest_face_frames,
            model.soft_gaussian_visual_vertex_particle_indices,
            model.soft_gaussian_visual_vertex_weights,
            model.soft_gaussian_visual_vertex_rest_offsets,
            model.soft_gaussian_visual_vertex_rest_physical_frames,
            model.soft_gaussian_rest_visual_face_frames,
            model.soft_gaussian_rest_visual_face_poses,
            model.quats.detach(),
            model.scales.detach(),
            particle_q,
            model.soft_tet_rest_poses,
        ],
        outputs=[
            out_state.means,
            out_state.quats,
            # Appearance optimization owns this leaf tensor, while runtime
            # skinning is a non-autograd state update.  The detached view
            # shares storage and keeps Warp's array interface valid.
            out_state.scale_log.detach(),
        ],
    )


def copy_embodied_gaussian_state(
    dest: EmbodiedGaussianState, src: EmbodiedGaussianState
):
    # 深拷贝一个完整快照到另一个快照对象中。
    copy_state(dest.physics_state, src.physics_state)
    copy_control(dest.physics_control, src.physics_control)
    dest.gaussian_state.copy(src.gaussian_state)
