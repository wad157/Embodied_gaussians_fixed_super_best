# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trio
import warp as wp

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent

# 优先导入当前仓库源码，避免运行到系统里旧安装的 embodied_gaussians。
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(current_dir))

from embodied_environments.super_embodied.super_embodied import (  # noqa: E402
    PSM_ARTICULATION_INDEX,
    apply_psm_lnd_pose,
    build_environment,
    expand_psm_q7_to_urdf_order,
    load_mimic_config,
    urdf_actuated_joint_order,
)
from embodied_gaussians import DatasetManager, EmbodiedGaussiansEnvironment  # noqa: E402
from embodied_gaussians.utils.indexing import scalar_index  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 SUPER grasp5 离线回放 demo。")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "SUPER 离线数据目录。目录内需要包含 robots.json、cameras.json、"
            "videos/stereo_left.mp4、videos/stereo_right.mp4 和对应 json。"
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="离线回放速度。默认按视频约 30 FPS 播放。")
    parser.add_argument(
        "--monitor-psm-base-q",
        action="store_true",
        help="打印 PSM 基座 body_q，诊断机械臂基座是否真的在物理仿真中漂移。",
    )
    parser.add_argument(
        "--monitor-tissue-q",
        action="store_true",
        help="打印 tissue 刚体 body_q，诊断组织是否被物理或视觉力拉走。",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="body_q 监控打印间隔，单位秒。默认 0.5 秒。",
    )
    parser.add_argument(
        "--visual-force-iterations",
        type=int,
        default=0,
        help="visual forces 每步优化迭代次数。初次 GUI 接入默认关闭。",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="stereo_left",
        help="逗号分隔的离线相机名。默认只保留 stereo_left，避免右侧相机画面干扰。",
    )
    parser.add_argument(
        "--camera-go-zoom",
        type=float,
        default=0.9,
        help=(
            "Go To Camera 视角缩放系数。1.0 表示保持纵横比并完整包含相机画面；"
            "默认 0.9，在完整画面外额外保留约 10% 边距。"
        ),
    )
    parser.add_argument(
        "--psm-roll-offset-deg",
        type=float,
        default=0.0,
        help=(
            "手动给 PSM roll 关节增加一个角度偏移，单位 degree。"
            "roll 是器械沿长杆轴线的自旋，用于临时对齐视频中的腕部/夹爪朝向；"
            "只在运行时生效，不修改 robots.json。"
        ),
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_arg: Path | None) -> Path:
    default_dataset = repo_root / "data/super/grasp5_offline_demo"
    dataset_from_env = os.environ.get("EMBODIED_GAUSSIANS_SUPER_DATASET")
    dataset_path = dataset_arg
    if dataset_path is None and dataset_from_env:
        dataset_path = Path(dataset_from_env).expanduser()
    if dataset_path is None:
        dataset_path = default_dataset
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    return dataset_path


class SuperPlaybackControls:
    """SUPER 离线回放控制器。

    它做两件事：
    1. 按当前时间戳从 DatasetManager 取视频帧；
    2. 按同一个时间戳从 robots.json 取 q7，展开成完整 PSM q_full。

    原 PushT demo 可以直接把 q 塞给 Panda。SUPER 不行，因为 PSM 的完整
    URDF 有 mimic 关节，所以这里多了一步 q7 -> q_full。
    """

    def __init__(
        self,
        environment: EmbodiedGaussiansEnvironment,
        dataset_manager: DatasetManager,
        fps: int,
        monitor_psm_base_q: bool = False,
        monitor_tissue_q: bool = False,
        monitor_interval: float = 0.5,
        psm_roll_offset_deg: float = 0.0,
    ):
        self.current_timestep = 0.0
        self.playing = False
        self.environment = environment
        self.dataset_manager = dataset_manager
        self.fps = fps
        self.first_state = environment.sim.clone_embodied_gaussian_state()
        self.monitor_psm_base_q = monitor_psm_base_q
        self.monitor_tissue_q = monitor_tissue_q
        self.monitor_interval = monitor_interval
        self.psm_roll_offset_deg = float(psm_roll_offset_deg)
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self._initial_base_q: np.ndarray | None = None
        self._initial_tissue_q: np.ndarray | None = None
        self._warned_missing_base_id = False
        self._warned_missing_tissue_id = False

        # mimic_cfg records how 7 active joints derive mimic joints.
        # joint_order records the joint order expected by the simulator.
        self.mimic_cfg = load_mimic_config()
        self.joint_order = urdf_actuated_joint_order(
            repo_root / "data/super/psm_robot/psm.urdf"
        )
        self.roll_joint_index = self.joint_order.index("roll")
        # PSM is fully driven by offline q, needs re-anchoring after each physics step.
        # _last_q_full stores the most recent q from go_to_timestep,
        # so run_physics can pull PSM back to the correct pose after each step.
        self._last_q_full: torch.Tensor = self.q_full_at(0.0)
        self._last_state_index = self.state_index_at(0.0)

    def state_index_at(self, timestep: float) -> int:
        robot_data = self.dataset_manager.robots["PSM1"]
        return scalar_index(
            robot_data.state_index_look_up.value(timestep), len(robot_data.states)
        )

    def q_full_at(self, timestep: float) -> torch.Tensor:
        robot_data = self.dataset_manager.robots["PSM1"]
        # DatasetManager / OfflineCamera 都使用 scalar_index 处理 Drake lookup
        # 返回数组的问题。这里同样用它取 robots.json 的离线状态索引。
        state_index = self.state_index_at(timestep)
        q7 = robot_data.states[state_index]["q"]
        q_full = expand_psm_q7_to_urdf_order(
            q7,
            mimic_cfg=self.mimic_cfg,
            joint_order=self.joint_order,
        )
        # 手动 roll 偏移：roll 是器械绕长杆轴线的自旋自由度。
        # 这个 offset 只加在运行时的 q_full 上，不写回 robots.json。
        q_full[self.roll_joint_index] += np.deg2rad(self.psm_roll_offset_deg)
        return torch.from_numpy(q_full).float()

    def reset(self):
        self.current_timestep = 0.0
        self.environment.sim.copy_embodied_gaussian_state(self.first_state)
        self.environment.sim.eval_ik()
        self.go_to_timestep(0.0)
        self._initial_base_q = self.current_psm_base_q()
        self._initial_tissue_q = self.current_tissue_q()
        self._last_base_monitor_time = -float("inf")
        self._last_tissue_monitor_time = -float("inf")
        self.maybe_print_psm_base_q(force=True)
        self.maybe_print_tissue_q(force=True)

    def go_to_timestep(self, timestep: float):
        self.current_timestep = timestep
        q_full = self.q_full_at(timestep)

        # set_robot_q 直接把当前机器人姿态放到离线帧位置；
        # set_robot_desired_q 同步控制目标，避免下一次 physics step 又把它拉回旧目标。
        self.environment.set_robot_q(PSM_ARTICULATION_INDEX, q_full)
        self.environment.set_robot_desired_q(PSM_ARTICULATION_INDEX, q_full)
        self._last_q_full = q_full
        self._last_state_index = self.state_index_at(timestep)
        apply_psm_lnd_pose(self.environment, self._last_state_index)
        self.dataset_manager.update_frames(timestep)

    def psm_base_body_id(self) -> int | None:
        """返回第 0 个环境里的 PSM 基座 body id。

        body_q 是按 body id 索引的。SUPER 场景构建时已经把
        PSM1_psm_base_link 的 id 存到 environment.super_psm_base_body_ids。
        """
        body_ids = getattr(self.environment, "super_psm_base_body_ids", [])
        if not body_ids:
            return None
        return int(body_ids[0])

    def current_psm_base_q(self) -> np.ndarray | None:
        """读取当前 PSM 基座 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。前三个数是基座在
        world 坐标系的位置，后四个数是姿态四元数。
        """
        body_id = self.psm_base_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def tissue_body_id(self) -> int | None:
        """返回第 0 个环境里的 tissue body id。"""
        body_ids = getattr(self.environment, "super_tissue_body_ids", [])
        if body_ids:
            return int(body_ids[0])
        body_id = getattr(self.environment, "super_tissue_body_id", None)
        if body_id is None:
            return None
        return int(body_id)

    def current_tissue_q(self) -> np.ndarray | None:
        """读取当前 tissue 刚体 body_q。

        返回值是 7 维数组：[x, y, z, qx, qy, qz, qw]。
        """
        body_id = self.tissue_body_id()
        if body_id is None:
            return None
        body_q = wp.to_torch(self.environment.sim.state_0.body_q)
        return body_q[body_id].detach().cpu().numpy().copy()

    def format_pose_delta(self, q: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        p = q[:3]
        quat = q[3:]
        p0 = q0[:3]
        quat0 = q0[3:]
        dpos = float(np.linalg.norm(p - p0))

        quat_norm = np.linalg.norm(quat)
        quat0_norm = np.linalg.norm(quat0)
        if quat_norm > 0.0 and quat0_norm > 0.0:
            quat_dot = float(abs(np.dot(quat / quat_norm, quat0 / quat0_norm)))
            dangle_deg = float(np.degrees(2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))))
        else:
            dangle_deg = float("nan")
        return p, quat, dpos, dangle_deg

    def maybe_print_psm_base_q(self, force: bool = False):
        if not self.monitor_psm_base_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_base_monitor_time < self.monitor_interval:
            return
        self._last_base_monitor_time = current_time

        body_id = self.psm_base_body_id()
        q = self.current_psm_base_q()
        if body_id is None or q is None:
            if not self._warned_missing_base_id:
                print("[PSM base monitor] 没找到 PSM1_psm_base_link 的 body id，无法监控基座。")
                self._warned_missing_base_id = True
            return

        if self._initial_base_q is None:
            self._initial_base_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_base_q)

        print(
            "[PSM base monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def maybe_print_tissue_q(self, force: bool = False):
        if not self.monitor_tissue_q:
            return

        current_time = self.environment.time()
        if not force and current_time - self._last_tissue_monitor_time < self.monitor_interval:
            return
        self._last_tissue_monitor_time = current_time

        body_id = self.tissue_body_id()
        q = self.current_tissue_q()
        if body_id is None or q is None:
            if not self._warned_missing_tissue_id:
                print("[tissue monitor] 没找到 tissue 的 body id，无法监控组织。")
                self._warned_missing_tissue_id = True
            return

        if self._initial_tissue_q is None:
            self._initial_tissue_q = q.copy()

        p, quat, dpos, dangle_deg = self.format_pose_delta(q, self._initial_tissue_q)

        print(
            "[tissue monitor] "
            f"sim_t={current_time:.4f}s body_id={body_id} "
            f"p=({p[0]:+.6f}, {p[1]:+.6f}, {p[2]:+.6f}) "
            f"q_xyzw=({quat[0]:+.6f}, {quat[1]:+.6f}, {quat[2]:+.6f}, {quat[3]:+.6f}) "
            f"dpos_from_reset={dpos:.9f}m dangle_from_reset={dangle_deg:.6f}deg"
        )

    def draw(self):
        # marsoom/pyglet 需要真实显示环境；放到 draw 里导入，可以让
        # `python examples/example_embodied_super_offline.py --help` 这类非 GUI
        # 操作在无显示环境里也能正常运行。
        from marsoom import imgui

        imgui.begin("SUPER Playback")
        imgui.text(f"Current timestep: {self.current_timestep:.2f}")
        imgui.text(f"Playing: {self.playing}")
        _, self.fps = imgui.slider_int("FPS", self.fps, 1, 120)
        changed_roll, roll_offset_deg = imgui.slider_float(
            "PSM roll offset deg",
            self.psm_roll_offset_deg,
            -180.0,
            180.0,
        )
        if changed_roll:
            self.psm_roll_offset_deg = float(roll_offset_deg)
            self.go_to_timestep(self.current_timestep)
        if imgui.button("Play"):
            self.playing = True
        imgui.same_line()
        if imgui.button("Pause"):
            self.playing = False
        imgui.same_line()
        if imgui.button("Reset"):
            self.reset()
        imgui.end()

    async def run_physics(self):
        dt = self.environment.dt()
        while True:
            self.environment.step()
            # PSM 完全由离线 q 驱动，不参与真实物理。每步物理后重新锚定
            # PSM 关节和 body 到最近一次 go_to_timestep 设定的位置，
            # 防止 XPBD 数值误差累积导致 PSM base 漂移甚至 NaN。
            self.environment.set_robot_q(PSM_ARTICULATION_INDEX, self._last_q_full)
            # LND modified-DH poses drive the visible tool links. The dVRK
            # articulation remains as the body/mesh carrier only.
            apply_psm_lnd_pose(self.environment, self._last_state_index)
            self.maybe_print_psm_base_q()
            self.maybe_print_tissue_q()
            await trio.sleep(dt)

    async def run(self):
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self.run_physics)
            while True:
                if self.playing:
                    self.current_timestep += 1 / self.fps
                    self.go_to_timestep(self.current_timestep)
                await trio.sleep(1 / self.fps)


async def main(
    dataset_path: Path,
    fps: int,
    monitor_psm_base_q: bool,
    monitor_tissue_q: bool,
    monitor_interval: float,
    visual_force_iterations: int,
    camera_names: list[str],
    camera_go_zoom: float,
    psm_roll_offset_deg: float,
):
    from embodied_gaussians.vis import EmbodiedGUI

    environment = build_environment()
    environment.visual_forces_settings.iterations = visual_force_iterations
    print(
        "[example_embodied_super_offline] visual_force_iterations="
        f"{environment.visual_forces_settings.iterations}"
    )

    dataset_manager = DatasetManager(dataset_path)
    if hasattr(dataset_manager, 'keep_only_cameras'):
        dataset_manager.keep_only_cameras(camera_names)
    print(f"[example_embodied_super_offline] enabled_cameras={getattr(dataset_manager.frames, 'names', 'N/A')}")
    environment.frames = dataset_manager.frames

    playback_controls = SuperPlaybackControls(
        environment,
        dataset_manager,
        fps,
        monitor_psm_base_q=monitor_psm_base_q,
        monitor_tissue_q=monitor_tissue_q,
        monitor_interval=monitor_interval,
        psm_roll_offset_deg=psm_roll_offset_deg,
    )
    playback_controls.reset()

    from pyglet.math import Vec3

    visualizer = EmbodiedGUI()
    visualizer.set_environment(environment)
    visualizer.viewer_3d.camera_go_zoom = camera_go_zoom
    # 默认自由视角改成侧视角：从世界 -X 方向看向手术区域，
    # 方便一进 demo 就检查 tissue 高度、z=0 ground plane、相机视锥和 PSM 的相对位置。
    # 这只影响启动后的 3D viewer 初始视角；点击 Go To Camera 仍然会切到真实离线相机视角。
    visualizer.viewer_3d._camera_pos = Vec3(-0.25, -0.02, 0.06)
    visualizer.viewer_3d._camera_front = Vec3(0.995, 0.0, -0.100)
    visualizer.viewer_3d._camera_up = Vec3(0.0, 0.0, 1.0)
    visualizer.viewer_3d.update_view_matrix()
    visualizer.callbacks_render.append(playback_controls.draw)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(playback_controls.run)
        await visualizer.run()
        nursery.cancel_scope.cancel()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset)
    print(f"[example_embodied_super_offline] 使用数据集目录: {dataset_path}")
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    wp.config.quiet = True
    wp.init()
    trio.run(
        main,
        dataset_path,
        args.fps,
        args.monitor_psm_base_q,
        args.monitor_tissue_q,
        args.monitor_interval,
        args.visual_force_iterations,
        camera_names,
        args.camera_go_zoom,
        args.psm_roll_offset_deg,
    )
