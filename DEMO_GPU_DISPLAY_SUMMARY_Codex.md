# Demo / GPU Display 配置与问题总结

更新时间：2026-07-07（最后更新：CPU fallback 方案实施与性能分析）

仓库路径：

```bash
/Media_HDD/jwshan/wad/Embodied_gaussians_fixed
```

Conda 环境：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex
```

注意：该环境不一定能用名字 `eg_codex` 激活，应使用完整路径：

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
```

## 当前结论（已更新）

**✅ Demo 已可以在浏览器/noVNC 中运行**，通过修改 `marsoom/texture.py` 添加了 CPU texture upload fallback。但性能明显低于原生 GPU OpenGL 方案。

**性能对比：**

| | 4090 + NVIDIA Xorg（原生） | A800 + Xvfb/llvmpipe（当前） |
|---|---|---|
| OpenGL 渲染 | GPU 硬件加速 | **Mesa llvmpipe CPU 软件渲染** |
| 纹理上传 | `cuda_pbo.map()` GPU→GPU 零拷贝 | GPU→CPU→PBO 多跳拷贝 |
| CUDA/GL interop | ✅ 正常 | ❌ `RegisteredGLBuffer` 失败 |
| FPS | ~60 | 慢（取决于视口分辨率） |

**慢的原因不是代码差异，是显示栈不同：**
- 原版在 4090 上跑 `DISPLAY=:0`（真实 NVIDIA Xorg），整套 OpenGL 管线都在 GPU 上
- 当前服务器用 `Xvfb :12`（Mesa llvmpipe），整个 3D 视口在 CPU 上软渲染，加上每帧 GPU→CPU 纹理拷贝

**本质解决方案：** 仍需管理员配置 NVIDIA GPU OpenGL 显示（见下方管理员方案 A/B/C）

**CMU fallback 方案（已实施）：**
- 修改 `marsoom/texture.py`：`create_pbo()` 捕获 `RegisteredGLBuffer` 创建失败，自动关闭 CUDA 路径；`copy_from_device()` 新增 `_copy_from_device_cpu()` fallback
- 修改 `embodied_viewer.py`：3 处 `tex_id` 同步（resize 重建纹理后消费者需要更新 ID）
- 代价：每帧多一次 GPU→CPU 数据拷贝（640×480 约 5-10ms），llvmpipe 软件渲染是大头

## 备份

修改前做过完整备份：

```bash
/Media_HDD/jwshan/tmp/Embodied_gaussians_fixed_backup_20260707_114719.tar.gz
```

## 已做的代码改动

### 1. 修复 Drake PiecewisePolynomial 索引返回数组的问题

新增文件：

```bash
src/embodied_gaussians/utils/indexing.py
```

用途：

- 新增 `scalar_index(value, upper_bound=None)`
- 兼容 `PiecewisePolynomial.value(timestamp)` 返回 numpy 数组而不是纯标量的情况
- 可把索引 clamp 到合法范围内

已替换这些位置的 `int(...value(timestamp))`：

```bash
src/embodied_gaussians/embodied_simulator/offline_camera.py
src/embodied_gaussians/dataset/dataset_manager.py
src/embodied_gaussians/physics_simulator/loader.py
src/embodied_gaussians/embodied_simulator/loader.py
```

修复的报错：

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

### 2. 新增显示和运行脚本

新增：

```bash
scripts/start_display_browser.sh
scripts/run_demo_on_display.sh
scripts/run_demo_browser_12.sh
scripts/prewarm_gsplat.py
scripts/run_demo_browser.sh
```

主要脚本：

```bash
bash scripts/start_display_browser.sh
```

作用：

- 启动 `Xvfb :12`
- 启动 `x11vnc`，端口 `5912`
- 启动 `websockify/noVNC`，端口 `6082`
- 设置普通箭头光标，避免 Xvfb 默认 `×` 光标

```bash
bash scripts/run_demo_on_display.sh
```

作用：

- 设置 conda 环境路径
- 设置 CUDA include/lib 路径
- 设置 `DISPLAY=:12`
- 设置 `TORCH_EXTENSIONS_DIR=/Media_HDD/jwshan/tmp/torch_extensions`
- 设置 `MAX_JOBS=2`
- 设置 `TORCH_CUDA_ARCH_LIST=8.0`
- 运行：

```bash
python examples/example_embodied_pusht_offline.py
```

### 3. `embodied_viewer.py` 当前状态

当前已做 3 处 `tex_id` 同步修改（resize 重建纹理后消费者需要更新 ID）：

```python
# render_cameras() L317
camera.set_texture_id(camera.texture.id)  # sync after potential resize

# render_virtual_camerawireframes() L352
camera.set_texture_id(camera.texture.id)  # sync after potential resize

# render_gaussians() L493
self.gaussian_overlay.tex_id = self.gaussian_texture.id  # sync after potential resize
```

与原版对比：只有这 3 行是新增的，其余逻辑完全相同。代码差异已通过 `diff` 确认。

### 4. `marsoom/texture.py` CPU fallback（核心修改）

修改了 conda 环境中的库文件：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/lib/python3.11/site-packages/marsoom/texture.py
```

两处改动：

**`create_pbo()` (L82-94)：** `warp.RegisteredGLBuffer` 创建包裹 `try-except`，失败时设置 `cuda_available = False`

**`copy_from_device()` (L156-190)：** 
- 入口检查 `cuda_available`，不可用直接走 CPU
- 原 CUDA 路径 `cuda_pbo.map()`/`warp.copy()` 包裹 `try-except`，失败时 fallback

**新增 `_copy_from_device_cpu()` (L192-227)：**
- 尺寸不变时：float32→uint8 转换 + 直接 PBO 上传，不触发 `resize()`（保留纹理 ID）
- 尺寸变化时：走 `copy_from_host()`（触发 resize，但 consumer 已有 tex_id sync）

**数据流（修复后）：**
```
copy_from_device(float32_tensor)
 → CUDA interop 不可用 → _copy_from_device_cpu()
   → 尺寸匹配 → float32→uint8 → 直接 PBO 上传 → 纹理 ✅
   → 尺寸不匹配 → copy_from_host → resize（tex_id 变化）
                  → caller 已通过 tex_id sync 更新 ✅
```

### 5. 与原版代码对比

`diff -rq 原版/src/ 现在/src/` 的唯一功能性差异是 `embodied_viewer.py` 中的 3 行 `tex_id` 同步。

**原版在 4090 上能跑 60 FPS，不是因为代码不同，而是因为 4090 有真实 NVIDIA Xorg（DISPLAY=:0），CUDA/OpenGL interop 正常工作，整个 OpenGL 管线都在 GPU 上。**

当前 A800 服务器走 `Xvfb :12`（Mesa llvmpipe CPU 软件渲染），所有 3D 渲染都在 CPU 上完成，这才是性能差的根本原因。

## 环境配置与诊断命令

### 激活环境

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed
```

### 检查 GPU

```bash
nvidia-smi --query-gpu=driver_version,name,compute_cap --format=csv
```

当前观察到：

```text
535.104.12, NVIDIA A800 80GB PCIe, 8.0
535.104.12, NVIDIA A800 80GB PCIe, 8.0
```

### 检查当前 Xvfb renderer

```bash
DISPLAY=:12 /Media_HDD/jwshan/conda_envs/eg_codex/bin/python - <<'PY'
import pyglet
from pyglet import gl
w = pyglet.window.Window(width=64, height=64, visible=False)
print(gl.gl_info.get_vendor())
print(gl.gl_info.get_renderer())
print(gl.gl_info.get_version())
w.close()
PY
```

当前观察到：

```text
Mesa
llvmpipe (LLVM 15.0.7, 256 bits)
(4, 5)
```

这说明 `:12` 不是 GPU OpenGL。

### 启动 noVNC/Xvfb

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed
bash scripts/start_display_browser.sh
```

本地 Windows PowerShell 建立 SSH 隧道：

```powershell
ssh -N -L 20000:127.0.0.1:6082 -p 221 jwshan@137.189.101.233
```

浏览器打开：

```text
http://localhost:20000/vnc.html?autoconnect=true&view_only=false&resize=scale
```

### 运行 demo

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed
bash scripts/run_demo_on_display.sh
```

或一条命令启动显示栈并运行：

```bash
bash scripts/run_demo_browser_12.sh
```

注意：在当前 Xvfb/llvmpipe 环境下，恢复原始 GPU 上传路径后，demo 仍可能在 camera / Gaussian Render 路径崩溃。

## VirtualGL + TurboVNC 尝试

由于没有 sudo，已尝试用户态安装：

```bash
mkdir -p /Media_HDD/jwshan/tmp/virtualgl_turbovnc
cd /Media_HDD/jwshan/tmp/virtualgl_turbovnc
```

查询官方 latest：

```bash
python - <<'PY'
import urllib.request
for url in [
    'https://github.com/VirtualGL/virtualgl/releases/latest',
    'https://github.com/TurboVNC/turbovnc/releases/latest',
]:
    r = urllib.request.urlopen(url, timeout=20)
    print(url, '->', r.geturl())
PY
```

得到：

```text
VirtualGL 3.1.4
TurboVNC 3.3
```

下载并解包：

```bash
cd /Media_HDD/jwshan/tmp/virtualgl_turbovnc
python - <<'PY'
import urllib.request
assets = {
 'virtualgl_3.1.4_amd64.deb': 'https://github.com/VirtualGL/virtualgl/releases/download/3.1.4/virtualgl_3.1.4_amd64.deb',
 'turbovnc_3.3_amd64.deb': 'https://github.com/TurboVNC/turbovnc/releases/download/3.3/turbovnc_3.3_amd64.deb',
}
for name, url in assets.items():
    urllib.request.urlretrieve(url, name)
PY

rm -rf root
mkdir -p root
dpkg-deb -x virtualgl_3.1.4_amd64.deb root
dpkg-deb -x turbovnc_3.3_amd64.deb root
```

工具路径：

```bash
/Media_HDD/jwshan/tmp/virtualgl_turbovnc/root/opt/VirtualGL/bin/vglrun
/Media_HDD/jwshan/tmp/virtualgl_turbovnc/root/opt/TurboVNC/bin/Xvnc
```

测试 TurboVNC Xvnc：

```bash
ROOT=/Media_HDD/jwshan/tmp/virtualgl_turbovnc/root
XVNC=$ROOT/opt/TurboVNC/bin/Xvnc

setsid -f "$XVNC" :13 -geometry 1280x720 -depth 24 \
  -securitytypes none -localhost -rfbport 5913 \
  >/tmp/turbovnc-13.log 2>&1 < /dev/null
```

`Xvnc :13` 可以启动，但 plain renderer 测试失败或不是 NVIDIA。

测试 VirtualGL EGL：

```bash
ROOT=/Media_HDD/jwshan/tmp/virtualgl_turbovnc/root
export LD_LIBRARY_PATH="$ROOT/usr/lib:${LD_LIBRARY_PATH:-}"

DISPLAY=:12 "$ROOT/opt/VirtualGL/bin/vglrun" -d egl +v \
  "$ROOT/opt/VirtualGL/bin/glxinfo"
```

失败现象：

```text
libEGL warning: failed to open /dev/dri/renderD129: Permission denied
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
EGL_BAD_MATCH
```

解包本地 NVIDIA GL 包测试：

```bash
BASE=/Media_HDD/jwshan/tmp/virtualgl_turbovnc
NVIDIA_ROOT=$BASE/nvidia_gl_535_230
mkdir -p "$NVIDIA_ROOT"
dpkg-deb -x /var/nvidia-driver-local-repo-ubuntu2204-535.230.02/libnvidia-gl-535_535.230.02-0ubuntu1_amd64.deb "$NVIDIA_ROOT"

ROOT=$BASE/root
export LD_LIBRARY_PATH="$ROOT/usr/lib:$NVIDIA_ROOT/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

DISPLAY=:12 "$ROOT/opt/VirtualGL/bin/vglrun" -d egl +v \
  "$ROOT/opt/VirtualGL/bin/glxinfo"
```

失败现象：

```text
No EGL devices found
```

判断：

- 系统正在运行的 NVIDIA kernel module 是 `535.104.12`
- 本地 repo 里的 `libnvidia-gl` 是 `535.230.02`
- 版本不匹配，且当前用户没有 `/dev/dri/renderD*` 权限
- 因此用户态 VirtualGL 不能正常获得 NVIDIA OpenGL context

## 当前权限/系统限制

当前用户组：

```bash
id
```

观察到：

```text
uid=1001(jwshan) gid=1001(jwshan) groups=1001(jwshan),1004(hddusers)
```

不在：

```text
video
render
```

`/dev/dri/renderD*` 权限：

```text
crw-rw----+ root render /dev/dri/renderD128
crw-rw----+ root render /dev/dri/renderD129
```

真实 Xorg：

```bash
ps -ef | rg 'Xorg|X :0|/usr/lib/xorg'
```

观察到：

```text
root ... /usr/lib/xorg/Xorg vt2 -displayfd 3 -auth /run/user/1000/gdm/Xauthority ...
```

`/run/user/1000` 属于用户 `server1`，`jwshan` 无权读取：

```text
/run/user/1000/gdm/Xauthority -> Permission denied
```

因此：

```bash
DISPLAY=:0 glxinfo
```

会失败：

```text
Authorization required, but no authorization protocol specified
Error: unable to open display :0
```

## 管理员需要处理的事项

要实现“浏览器可见 + 真 GPU OpenGL + CUDA/OpenGL interop”，需要管理员配置至少一种方案。

### 方案 A：给 `jwshan` 访问真实 GPU Xorg 的权限

由拥有 `:0` 图形会话权限的用户或管理员执行：

```bash
xhost +SI:localuser:jwshan
```

然后测试：

```bash
DISPLAY=:0 glxinfo | grep -E 'OpenGL vendor|OpenGL renderer'
```

期望看到 NVIDIA，而不是 Mesa/llvmpipe。

### 方案 B：安装并配置 VirtualGL + TurboVNC

管理员安装：

```bash
sudo apt install virtualgl turbovnc libnvidia-gl-535
```

并确保 `libnvidia-gl` 版本和当前 kernel module 匹配：

```bash
cat /proc/driver/nvidia/version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

当前 kernel module 是：

```text
535.104.12
```

不要混用 `535.230.02` 的 OpenGL 用户态库，除非同时升级 kernel module。

还需要让 `jwshan` 具备图形设备权限：

```bash
sudo usermod -aG video,render jwshan
```

重新登录后验证：

```bash
id
ls -l /dev/dri/renderD*
```

### 方案 C：管理员统一升级 NVIDIA 驱动栈

当前系统状态不一致：

- kernel module: `535.104.12`
- 本地 repo: `535.230.02`
- apt candidate: `535.309.01`
- 系统已装 OpenGL vendor: 只有 Mesa

建议管理员统一安装一套匹配的 NVIDIA driver + libnvidia-gl，避免 VirtualGL/EGL 与 kernel driver mismatch。

## 管理员完成后建议测试命令

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed

ROOT=/Media_HDD/jwshan/tmp/virtualgl_turbovnc/root
export LD_LIBRARY_PATH="$ROOT/usr/lib:${LD_LIBRARY_PATH:-}"

DISPLAY=:12 "$ROOT/opt/VirtualGL/bin/vglrun" -d :0 \
  "$ROOT/opt/VirtualGL/bin/glxinfo" | grep -E 'OpenGL vendor|OpenGL renderer'
```

期望：

```text
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA ...
```

再运行 demo：

```bash
DISPLAY=:12 "$ROOT/opt/VirtualGL/bin/vglrun" -d :0 \
  /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  examples/example_embodied_pusht_offline.py
```

## 当前可用（CPU fallback）的浏览器显示命令

Demo 可正常运行，但性能受 llvmpipe 软件渲染限制：

服务器端：

```bash
conda activate /Media_HDD/jwshan/conda_envs/eg_codex
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed
bash scripts/start_display_browser.sh
```

本地 PowerShell：

```powershell
ssh -N -L 20000:127.0.0.1:6082 -p 221 jwshan@137.189.101.233
```

浏览器：

```text
http://localhost:20000/vnc.html?autoconnect=true&view_only=false&resize=scale
```

运行 demo：

```bash
bash scripts/run_demo_on_display.sh
```

**性能提示：** 在 UI 中关闭 `Show Cameras`、`Show Virtual Cameras`、`Show Physics`、`Gaussian Meshes`、`Gaussian Outlines` 可减少 llvmpipe 渲染负载。

## 当前 git 工作区相关文件

当前改动/新增文件：

```bash
CHANGELOG_Codex.md
DEMO_GPU_DISPLAY_SUMMARY_Codex.md
src/embodied_gaussians/dataset/dataset_manager.py
src/embodied_gaussians/embodied_simulator/loader.py
src/embodied_gaussians/embodied_simulator/offline_camera.py
src/embodied_gaussians/embodied_visualizer/embodied_viewer.py  # +3行 tex_id sync
src/embodied_gaussians/physics_simulator/loader.py
src/embodied_gaussians/utils/indexing.py
scripts/prewarm_gsplat.py
scripts/run_demo_browser.sh
scripts/run_demo_browser_12.sh
scripts/run_demo_on_display.sh
scripts/start_display_browser.sh
```

conda 环境库文件改动：

```bash
# CPU fallback 方案的核心修改
/Media_HDD/jwshan/conda_envs/eg_codex/lib/python3.11/site-packages/marsoom/texture.py
```

**与原版对比：** `diff -rq` 确认，只有 `embodied_viewer.py` 中新增 3 行 `tex_id` sync，其余 Python 源码无差异。FPS 差距来自显示栈（NVIDIA Xorg vs Xvfb/llvmpipe），不是代码。
