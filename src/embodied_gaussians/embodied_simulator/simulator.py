# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from typing import Literal
import pysegreduce
from dataclasses import dataclass
import torch
import warp as wp
import warp.sim
from gsplat.rendering import rasterization
from embodied_gaussians.physics_simulator.simulator import Simulator, copy_control, copy_state

from embodied_gaussians.embodied_simulator import EmbodiedGaussiansBuilder
from embodied_gaussians.embodied_simulator.gaussians import GaussianModel, GaussianState

from embodied_gaussians.embodied_simulator.appearance_optimizer import AppearanceOptimizer
from embodied_gaussians.embodied_simulator.frames import Frames
from embodied_gaussians.embodied_simulator.visual_forces import VisualForces, VisualForcesSettings
from embodied_gaussians.embodied_simulator.warp import (
    apply_forces_kernel,
    update_gaussians_transforms_kernel,
    update_visual_forces_kernel,
)


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

    def update_gaussian_transforms(self):
        # 根据当前 body_q 更新每个 Gaussian 在世界坐标系下的位姿。
        # 也就是把“高斯绑定在哪个 body 上”的信息真正转换成当前渲染状态。
        update_gaussian_transforms(
            self.gaussian_model, self.state_0.body_q, self.gaussian_state
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
                loss = torch.nn.functional.smooth_l1_loss(
                    render_colors,
                    target_colors,
                    beta=settings.robust_loss_beta,
                )
            else:
                loss = torch.nn.functional.mse_loss(render_colors, target_colors)
            # ideas: add a loss that pushes the colors back to their orignal values or to some sort of ema colors
            # ideas: allow the gaussians to jitter a bit while anchoring them to the original positions

            self.visual_forces.zero_grad()
            self.appearance_optimizer.zero_grad()
            loss.backward()
            self.visual_forces.step()
            self.appearance_optimizer.step()

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


def update_gaussian_transforms(model: GaussianModel, body_q, out_state: GaussianState):
    # 把“高斯在各自刚体局部坐标系里的初始位姿”
    # 变换成“当前世界坐标系里的高斯位姿”。
    #
    # 这是 physics 和 gaussian 渲染之间最关键的同步点之一。
    if model.num_gaussians == 0:
        return
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


def copy_embodied_gaussian_state(
    dest: EmbodiedGaussianState, src: EmbodiedGaussianState
):
    # 深拷贝一个完整快照到另一个快照对象中。
    copy_state(dest.physics_state, src.physics_state)
    copy_control(dest.physics_control, src.physics_control)
    dest.gaussian_state.copy(src.gaussian_state)
