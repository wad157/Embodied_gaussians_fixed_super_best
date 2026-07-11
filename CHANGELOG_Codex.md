# Codex 改动记录

本文档记录我在 `embodied_gaussians4` 仓库内做过的具体代码改动，尽量细化到文件、函数和行号，方便你后续自己继续改数据集。

更新约定：

- 只记录我实际动过或新建过的文件。
- 每次我继续修改这个仓库时，会同步更新这份文档。
- 行号以当前工作区文件为准；后续如果文件继续变动，行号可能会漂移。

## 当前已记录改动

### 2026-07-07 Xvfb/noVNC demo 运行修复

#### `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py`

- 新增 `_image_for_xvfb_texture(...)`：
  - 对 `torch.Tensor` 相机图像先 `detach().cpu().numpy()`
  - 统一转成 contiguous `uint8` numpy 图像
  - 让 marsoom 走 `Texture.copy_from_host()`，避开 Xvfb 下不可用的 CUDA/OpenGL interop

- `render_cameras()`：
  - 保留 `CameraWireframeWithImage` 和相机贴图显示
  - 把 `frames.colors_gpu[i]` 转成 CPU 图像后再 `camera.update_image(...)`
  - 更新后调用 `camera.set_texture_id(camera.texture.id)`，避免 marsoom texture resize 后 ImageQuad 仍绑定旧 texture id

- `render_virtual_camerawireframes()`：
  - 同样保留虚拟相机 wireframe 和图像贴图
  - 把 `cameras.rendered_images[j, i]` 转成 CPU 图像后上传
  - 同步 ImageQuad 的 texture id

作用：

- 修复在服务器 `Xvfb :12 + x11vnc + noVNC` 下运行 demo 时，`glBindTexture(...)` 抛出 `GLException: Invalid operation` 的问题。
- 这不是关闭 `draw_cameras` / `draw_virtual_cameras`，而是保留显示功能并把图像上传路径换成 Xvfb 可用的 CPU 上传。
- `Warp CUDA error 304` 仍可能打印，这是 Xvfb 不支持 CUDA/OpenGL interop 的环境警告；当前相机图像路径已经不依赖这条 interop 路径。

#### `src/embodied_gaussians/utils/indexing.py`

- 新增 `scalar_index(value, upper_bound=None)`：
  - 兼容 Drake `PiecewisePolynomial.value(...)` 返回数组而不是纯标量的情况
  - 支持 clamp 到合法索引范围

#### 索引调用点

- `src/embodied_gaussians/embodied_simulator/offline_camera.py`
- `src/embodied_gaussians/dataset/dataset_manager.py`
- `src/embodied_gaussians/physics_simulator/loader.py`
- `src/embodied_gaussians/embodied_simulator/loader.py`

作用：

- 修复 `TypeError: only 0-dimensional arrays can be converted to Python scalars`。

#### 运行脚本

- `scripts/start_display_browser.sh`
  - 启动 `Xvfb :12`
  - 启动 `x11vnc :5912`
  - 启动 `websockify/noVNC :6082`

- `scripts/run_demo_on_display.sh`
  - 设置 conda env、CUDA include/lib、`DISPLAY=:12`
  - 运行 `examples/example_embodied_pusht_offline.py`

- `scripts/run_demo_browser_12.sh`
  - 串联启动显示栈并运行 demo

- `scripts/prewarm_gsplat.py`
  - 用于提前触发 gsplat CUDA extension 编译/加载

### 0. `embodied_gaussians_fixed` 中的紧急修复

#### `src/embodied_gaussians/embodied_simulator/simulator.py`

- 第 255 到 269 行：
  - 新增 `backgrounds = background.view(1, 3).expand(num_images, 3).contiguous()`
  - 在 `rasterization(...)` 中新增 `packed=False`
  - 把背景参数从 `background.reshape(1, 3).repeat(num_images, 1)` 改成 `backgrounds`

作用：

- 修复 `Gaussian Render` 路径下 `gsplat` / `rasterization` 的参数兼容问题。
- 这部分和之前修过的 `visual forces render` 逻辑保持一致。

#### `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py`

- 第 463 到 467 行：
  - 把 `ids = meta["gaussian_ids"]` 改成 `ids = meta.get("gaussian_ids")`
  - 当 `gaussian_ids` 缺失时，回退为 `torch.arange(...)`

- 第 486 行：
  - 把 `conics=meta["conics"]` 改成 `conics=meta["conics"].reshape(-1, 3)`

作用：

- 修复点击 `Gaussian Render` 后，旧代码路径仍按老版 `meta` 格式取值而导致的崩溃。
- 这也是为什么之前 `visual forces` 分支能部分工作，但 `Gaussian Render` 一点就窗口消失。

#### `examples/example_embodied_pusht_offline.py`

- 文件开头新增：
  - `import sys`
  - `current_dir = Path(__file__).resolve().parent`
  - `repo_root = current_dir.parent`
  - `sys.path.insert(0, str(repo_root / "src"))`
  - `sys.path.insert(0, str(current_dir))`

作用：

- 修复“在 `embodied_gaussians_fixed` 目录里运行，但 Python 实际仍导入 `embodied_gaussians4/src` 已安装源码”的问题。
- 之前你日志里出现的报错路径全是 `/home/hsieh/data0/wad/embodied_gaussians4/src/...`，说明补丁根本没生效。
- 加了这段后，`fixed` 目录下的源码会被优先导入，前面的 `Gaussian Render` 修复才真正会进入运行路径。

### 1. [examples/example_embodied_pusht_offline.py](/home/hsieh/data0/wad/embodied_gaussians4/examples/example_embodied_pusht_offline.py)

具体改动位置：

- 第 3 到 4 行：新增 `argparse`、`os` 导入。
  作用：支持通过命令行参数和环境变量选择数据集目录。

- 第 15 到 31 行：新增 `parse_args()`。
  作用：
  - 新增 `--dataset` 参数。
  - 让启动时可以直接写：
    `python examples/example_embodied_pusht_offline.py --dataset /你的数据集目录`

- 第 34 到 50 行：新增 `resolve_dataset_path()`。
  作用：
  - 优先读取命令行参数 `--dataset`
  - 如果没传，再读环境变量 `EMBODIED_GAUSSIANS_DATASET`
  - 都没有时，退回默认路径：
    `examples/embodied_environments/pusht_embodied/sample_demos/0`
  - 把相对路径统一转成绝对路径

- 第 55 到 115 行附近：为 `PlaybackControls` 增加中文解释注释。
  涉及方法：
  - `__init__`
  - `reset`
  - `go_to_timestep`
  - `draw`
  - `run_physics`
  - `run`
  作用：解释“物理循环”和“离线数据回放循环”如何配合。

- 第 124 行：把 `main()` 改成 `main(dataset_path: Path)`。
  作用：不再在函数内部写死数据集目录，而是从外部传入。

- 第 131 到 148 行：重写数据集读取前的注释说明。
  作用：
  - 写清默认数据集目录
  - 写清 `DatasetManager` 需要的目录结构
  - 写清以后如何不改源码切换数据集

- 第 148 行：把
  `DatasetManager(current_dir / Path(...sample_demos/0))`
  改成
  `DatasetManager(dataset_path)`
  作用：真正实现“外部指定数据集目录”。

- 第 149 到 168 行附近：补充 `environment.frames`、GUI 回放流程相关中文注释。

- 第 174 到 181 行：修改脚本入口。
  具体变化：
  - 新增 `args = parse_args()`
  - 新增 `dataset_path = resolve_dataset_path(args.dataset)`
  - 新增启动时打印当前数据集目录
  - 把 `trio.run(main)` 改成 `trio.run(main, dataset_path)`

当前使用方式：

```bash
cd /home/hsieh/data0/wad/embodied_gaussians4
python examples/example_embodied_pusht_offline.py
python examples/example_embodied_pusht_offline.py --dataset /你的数据集目录
```

---

### 2. [scripts/run_example_embodied_pusht_offline.sh](/home/hsieh/data0/wad/embodied_gaussians4/scripts/run_example_embodied_pusht_offline.sh)

这是新建文件。

具体内容：

- 第 1 到 2 行：新增 `bash` 脚本头和 `set -euo pipefail`
- 第 4 到 10 行：新增中文用法说明
- 第 12 行：计算仓库根目录 `ROOT_DIR`
- 第 13 行：设置默认数据集目录 `sample_demos/0`
- 第 14 行：读取第一个位置参数作为可选数据集目录
- 第 18 到 20 行：通过环境变量 `EMBODIED_GAUSSIANS_DATASET` 启动 Python 示例

作用：

- 不改源码即可切换数据集
- 把常用启动方式固定下来

使用方式：

```bash
cd /home/hsieh/data0/wad/embodied_gaussians4
bash scripts/run_example_embodied_pusht_offline.sh
bash scripts/run_example_embodied_pusht_offline.sh /你的数据集目录
```

---

### 3. [scripts/convert_hf_simulation_to_offline_demo.py](/home/hsieh/data0/wad/embodied_gaussians4/scripts/convert_hf_simulation_to_offline_demo.py)

这是新建文件。

作用：

- 把 Hugging Face `anon211/embodied_gaussians` 一类仿真数据转换成 `example_embodied_pusht_offline.py` 可读的离线目录结构。

当前状态：

- 这是为之前 `simulated/fall_1` 适配实验加入的转换脚本。
- 目前你已经决定暂时回到原始数据集，所以它保留但不在默认运行路径上。
- 围绕 `bodies_eval.json` 的进一步扩展没有收尾，当前不建议把它当最终流程。

---

### 4. [examples/example_pusht_datacollector.py](/home/hsieh/data0/wad/embodied_gaussians4/examples/example_pusht_datacollector.py)

这份文件的主要改动是补中文注释，帮助理解“PushT 数据是怎么采出来的”。

具体改动位置：

- 第 25 行：`Settings.path` 注释改成中文，说明这是数据集保存根目录。
- 第 27 行：`Settings.physics` 注释改成中文。

- 第 35 到 42 行：在 `CollectGUI.__init__()` 中新增中文注释。
  作用：解释 `0/ 1/ 2/` 这类 demo 子目录编号如何推导。

- 第 49 到 57 行：为 `PushTEnvironment.build()` 和 `Saver(...)` 增加说明。
  作用：解释“仿真环境负责产生轨迹，Saver 负责写盘”。

- 第 69 到 72 行：在 `physics_loop()` 中补注释。
  作用：解释每个物理步会推进仿真并在录制时缓存 state/control。

- 第 90 到 94 行：在 `draw_2d_viewer()` 中补注释。
  作用：解释 2D 视图中 T-block、目标、pusher 的含义。

- 第 122 行：补注释说明鼠标拖动如何控制 pusher 目标位置。

- 第 133 行：为 `reset()` 补注释。

- 第 139 行：为 `new_demo()` 补注释。

- 第 144 到 145 行：为 `save_demo()` 补注释。
  作用：说明保存出来的目录结构就是 `datainspector` 读取的格式。

- 第 157 到 159 行：为 `update()` 补注释。
  作用：说明任务完成后自动保存并重置。

- 第 166 到 167 行：为 `set_joints()` 补注释。

- 第 241 到 243 行：补充 `2D Viewer` / `3D Viewer` 的说明。

- 第 257 到 258 行：在 `main()` 前补充整文件用途说明。

---

### 5. [examples/example_pusht_datainspector.py](/home/hsieh/data0/wad/embodied_gaussians4/examples/example_pusht_datainspector.py)

这份文件的主要改动也是补中文注释，帮助理解“离线 demo 是怎么读回来的”。

具体改动位置：

- 第 21 行：`Settings.path` 注释改成中文，说明这里应是含多个数字子目录的根目录。

- 第 28 到 30 行：在 `InspectGUI.__init__()` 中补注释。
  作用：解释这里是“把离线 state/control 灌回仿真器”，不是重新跑策略。

- 第 36 行：补“播放进度和播放状态”说明。

- 第 42 到 44 行：补 `demo_paths` 扫描逻辑说明。

- 第 51 到 54 行：为 `load_demo()` 补注释。
  作用：解释切换 demo 时的 3 个步骤。

- 第 63 到 68 行：为 `go_to_index()` 补注释。
  作用：说明它非常适合检查你整理的数据集是否还能被 Loader 正确读取。

- 第 78 行：在 `start_playing()` 中补注释，说明按固定帧率自增索引实现回放。

- 第 102 行：为空格键播放/暂停补注释。

- 第 126 到 128 行：为手动拖进度条补注释。

- 第 149 行：补充左侧 demo 列表用途说明。

- 第 161 行：补充 3D Viewer 用途说明。

- 第 169 到 171 行：在 `main()` 前补充整文件用途说明。

---

### 6. [examples/embodied_environments/pusht_embodied/pusht_embodied.py](/home/hsieh/data0/wad/embodied_gaussians4/examples/embodied_environments/pusht_embodied/pusht_embodied.py)

这份文件的主要改动是补中文注释，帮助你区分“场景定义”和“离线数据集”。

具体改动位置：

- 第 14 到 16 行：在 `Q_START` 上方补注释。
  作用：说明机器人初始姿态和数据集不一致时，这里也要一起调整。

- 第 29 到 32 行：补充资源文件路径说明。
  涉及：
  - `ground_plane.json`
  - `extrinsics.json`
  - `objects/*.json`

- 第 40 到 46 行：为 `get_body()` 补注释。
  作用：说明对象几何/高斯描述文件从哪里读，以及换物体时优先看这里。

- 第 54 到 61 行：为 `build_environment()` 补注释。
  作用：强调“环境里有什么”和“数据集记录了什么”是两件事。

- 第 68 到 70 行：为 `add_renderable_articulation_from_urdf(...)` 补注释。

- 第 81 到 82 行：为 `add_articulation_from_urdf(...)` 的无高斯分支补注释。

- 第 93 行：说明这里加入的是主要操作物体 T-block。

- 第 96 行：说明地面也作为可视化对象加入。

- 第 101 行：说明多环境模式会复制整套场景。

- 第 104 到 106 行：为 `gravity_factor` 处理补注释。
  作用：提醒你换场景后若行为异常，这里值得检查。

- 第 109 行：说明机器人控制目标初始化和 `Q_START` 对齐。

- 第 116 到 121 行：为 `add_static_cameras()` 补注释。
  作用：明确这是仿真虚拟相机，不是离线数据目录里的 `cameras.json`。

- 第 130 到 135 行：为 `camera_data` 结构补注释。
  作用：说明如果这里与离线数据集相机位姿不一致，visual forces 会严重错位。

- 第 221 到 225 行：为 `add_gripper_camera()` 补注释。
  作用：说明手眼相机是否存在、`body_id` 和安装位姿要不要匹配。

---

### 7. [examples/sim_environments/pusht.py](/home/hsieh/data0/wad/embodied_gaussians4/examples/sim_environments/pusht.py)

这份文件也主要是补中文注释，帮助你理解 PushT 仿真内部状态定义。

具体改动位置：

- 第 13 到 14 行：为 `TBLOCK_ID` 补注释。

- 第 23 行：为 `PushTEnvironmentActions.allocate()` 补注释。

- 第 39 到 41 行：为 `PushTEnvironmentObservations.allocate()` 补注释。

- 第 57 到 64 行：为 `PushTEnvironment.build()` 补注释。
  作用：说明 datacollector / datainspector 保存和读取的就是这套内部状态定义。

- 第 94 行：为 `Simulator` 初始化补注释。

- 第 109 行：为目标位姿 `_X_ET` 补注释。

- 第 111 行：把原来的 `warm start cuda kernels` 英文注释改成中文解释。

- 第 118 到 120 行：为 `reset()` 补注释。

- 第 133 行：为 `act()` 补注释。

- 第 137 到 140 行：为 `step()` 补注释。

- 第 158 到 159 行：为 `_update_state()` 补注释。

- 第 173 到 175 行：为 `_update_task()` 补注释。

- 第 196 行：为 `_reset_pusher()` 补注释。

- 第 204 到 205 行：为 `_randomize_tblock()` 补注释。

- 第 218 行：为 `set_target()` 补注释。

- 第 223 到 228 行：为 `add_tblock_shape()` 补注释。
  作用：强调 T-block 几何本身也是硬编码的，换场景不只是换数据集。

- 第 254 到 256 行：为 `synchronize_state_kernel()` 补注释。

- 第 277 到 281 行：为 `get_reward_and_success_kernel()` 补注释。

- 第 306 行：为 `randomize_states_kernel()` 补注释。

---

### 8. [src/embodied_gaussians/embodied_simulator/simulator.py](/home/hsieh/data0/wad/embodied_gaussians4/src/embodied_gaussians/embodied_simulator/simulator.py)

这里是功能性修改，不只是注释。

具体改动位置：

- 第 101 行：新增
  `backgrounds = background.view(1, 3).expand(num_images, 3).contiguous()`

- 第 114 到 115 行：在 `rasterization(...)` 调用中：
  - 新增 `packed=False`
  - 把原先
    `backgrounds=background.reshape(1, 3).repeat(num_images, 1)`
    改成
    `backgrounds=backgrounds`

改动目的：

- 调整传给 `rasterization` 的背景张量形状和内存布局
- 兼容当前 `gsplat` / 渲染调用方式，减少运行时兼容性问题

---

### 9. [src/embodied_gaussians/embodied_visualizer/embodied_viewer.py](/home/hsieh/data0/wad/embodied_gaussians4/src/embodied_gaussians/embodied_visualizer/embodied_viewer.py)

这里也是功能性修改，不只是注释。

具体改动位置：

- 第 310 到 312 行：
  把
  `pyglet.math.Mat4(frames.X_WCs_cpu[i].T.flatten().numpy())`
  类似的单参数构造方式改成
  `pyglet.math.Mat4(*frames.X_WCs_cpu[i].T.flatten().numpy().tolist())`
  作用：兼容当前 `pyglet` 的 `Mat4` 构造参数格式。

- 第 348 行：
  把
  `pyglet.math.Mat4(X_WC.T.flatten())`
  改成
  `pyglet.math.Mat4(*X_WC.T.flatten().tolist())`
  作用同上。

- 第 397 到 402 行：
  把
  `ids = meta["gaussian_ids"]`
  改成
  `ids = meta.get("gaussian_ids")`
  并在缺失时回退为 `torch.arange(...)`
  作用：兼容 `meta` 里没有 `gaussian_ids` 的情况，避免直接报错。

- 第 420 行：
  把
  `conics=meta["conics"]`
  改成
  `conics=meta["conics"].reshape(-1, 3)`
  作用：兼容当前渲染器对 `conics` 输入形状的要求。

---

### 10. [imgui.ini](/home/hsieh/data0/wad/embodied_gaussians4/imgui.ini)

这是 GUI 布局状态文件改动，不属于核心逻辑修改。

具体改动位置：

- 第 3 行：窗口总宽从 `1280` 变成 `1278`
- 第 18 行：`3D Viewer` 宽度从 `730` 变成 `728`
- 第 112 行：`Playback` 窗口位置从 `1030` 变成 `1028`
- 第 124 行：`DockSpace` 尺寸从 `1280,720` 变成 `1278,720`

作用：

- 只是界面停靠布局被运行时更新了
- 不属于数据集接口或仿真逻辑的核心修改

---

## 当前改动的总体分类

### A. 纯注释型改动

- `examples/example_pusht_datacollector.py`
- `examples/example_pusht_datainspector.py`
- `examples/embodied_environments/pusht_embodied/pusht_embodied.py`
- `examples/sim_environments/pusht.py`
- `examples/example_embodied_pusht_offline.py` 中的大部分中文注释

### B. 功能型改动

- `examples/example_embodied_pusht_offline.py`
  - 新增 `--dataset`
  - 新增环境变量数据集入口
  - 默认回到 `sample_demos/0`
- `scripts/run_example_embodied_pusht_offline.sh`
  - 新增启动脚本
- `scripts/convert_hf_simulation_to_offline_demo.py`
  - 新增转换脚本
- `src/embodied_gaussians/embodied_simulator/simulator.py`
  - 调整 `rasterization` 参数
- `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py`
  - 修复 `Mat4` 构造、`gaussian_ids` 缺失、`conics` 形状

### C. 运行状态文件改动

- `imgui.ini`

---

## 2026-05-14 注释补充

### [src/embodied_gaussians/embodied_simulator/simulator.py](/home/hsieh/data0/wad/embodied_gaussians_fixed/src/embodied_gaussians/embodied_simulator/simulator.py)

本次新增的是“结构解释型”中文注释，没有改逻辑。

主要补充位置：

- `EmbodiedGaussianState`
  - 解释它是 physics state / control / gaussian state 的完整快照

- `EmbodiedGaussiansSimulator.__init__`
  - 解释 physics simulator、visual_forces、appearance_optimizer 各自负责什么

- `get_specific_environment_state` / `set_specific_environment_state`
  - 解释多环境 batched state 如何抽取和写回

- `embodied_gaussian_state` / `clone_embodied_gaussian_state` / `copy_embodied_gaussian_state`
  - 解释浅状态、深拷贝状态、整体覆盖之间的区别

- `render_visual_forces`
  - 解释它渲染的是“视觉优化后的临时 Gaussian 位姿”

- `render_gaussians`
  - 解释它渲染的是普通场景 GaussianState

- `update_gaussian_transforms`
  - 解释 physics body 到 Gaussian 世界位姿同步的作用

- `_compute_visual_forces`
  - 重点补充了视觉力完整流程：
    1. 从真实 GaussianState 初始化临时可优化副本
    2. 对观测图像做渲染误差优化
    3. 把位姿差转成 per-Gaussian 力/力矩
    4. 聚合成 per-body 总力
    5. 写回物理系统

- 模块级 `render_gaussians`
  - 解释它是通用渲染入口

- `copy_embodied_gaussian_state`
  - 解释它是完整快照复制

### [src/embodied_gaussians/embodied_simulator/visual_forces.py](/home/hsieh/data0/wad/embodied_gaussians_fixed/src/embodied_gaussians/embodied_simulator/visual_forces.py)

本次新增的是“机制解释型”中文注释，没有改逻辑。

主要补充位置：

- `VisualForcesSettings`
  - 解释 `iterations`
  - 解释 `lr_means / lr_quats / lr_color / lr_opacity / lr_scale`
  - 解释 `kp`

- `VisualForces` 类定义
  - 解释它维护的是视觉优化过程中的临时变量，而不是场景真值

- `__init__`
  - 解释可优化的 `means / quats`、输出的 `forces / moments`
  - 解释 `bodies_affected_by_visual_forces` 如何转成 mask
  - 解释为什么要预计算 body 对应的 Gaussian 连续段
  - 解释 Warp 版 Adam 的用途

- `set_learnings_rates`
  - 解释每步如何动态更新学习率

- `zero_grad`
  - 解释是反传前清梯度

- `step`
  - 解释为什么先把“不允许受视觉力影响的 Gaussian”梯度清零，再更新

- `_initialize`
  - 解释 `start_inds / end_inds / body_ids`
  - 解释这是为后续 per-Gaussian -> per-body 聚合做准备
  - 解释 `_total_forces / _total_moments` 的意义

---

## 维护说明

如果后续我继续改这个仓库，我会继续按下面格式追加：

- 文件路径
- 具体行号
- 改动前后差异要点
- 改动目的
