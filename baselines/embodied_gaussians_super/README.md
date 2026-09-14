# Embodied Gaussians EG-Soft baseline on SUPER

该目录将 [RAI Embodied Gaussians](https://github.com/rai-opensource/embodied_gaussians)
接入 SUPER `grasp5/grasp3/grasp1` 的统一正式协议。上游固定到提交
`c97ec671f97af25985e0af8844c0aac8d8119b97`。

## 方法边界

公开仓库只提供刚体实现，并明确没有发布论文中的 shape-matching 软体代码。因此这里的正式方法名为
`Embodied Gaussians EG-Soft paper reconstruction`，不能写成“运行了未公开的官方 soft code”。
软体部分只复现论文式 (4)–(5) 与 Algorithm 2：有向粒子、局部 shape matching、粒子碰撞、
Gaussian 最近粒子刚性绑定，以及 RGB 优化位移产生视觉力。实现与 SIM baseline 共用经过审计的
`embodied_gaussians_fixed_super_best_sim/baselines/embodied_gaussians_sim/run_soft.py`，SUPER
适配器不调用本项目的四面体 XPBD、AllTracker、视觉 3D 状态修正或在线材料辨识。

论文没有公开柔软物体的数值 `k_S` 和邻域构造。为避免在 SUPER 上调参，固定使用
`k_S=1`（完整执行式 (5) 投影）和静止态 Delaunay 几何邻接；三套数据完全相同。

## 固定论文/公开代码参数

| 参数 | 固定值 |
|---|---:|
| particle radius | 6 mm（公开 `SimpleBodyBuilder` 默认值，位于论文 4–7 mm 范围） |
| ordinary-object particle mass | 0.1 kg |
| frame rate / physical step | 30 fps / 1/30 s |
| substeps / Jacobi passes | 20 / 4 |
| velocity damping | 0.9 per frame |
| gravity | -9.80665 m/s²，世界 z 轴 |
| particle / Gaussian initialization | 80 / 250 iterations |
| online visual optimization | 5 Adam iterations |
| position / rotation / color / opacity LR | 1e-3 / 1e-4 / 5e-4 / 5e-4 |
| visual-force Kp / deadzone | 60 / 2 mm |
| online width | 640 px |

不存在 grasp5、grasp3、grasp1 各自的物理参数或拟合参数。

## SUPER 输入适配

- 相机严格只有 `stereo_left` 和 `stereo_right`，没有五视角或虚拟视角。
- SUPER 的重建留出相位是 `frame % 8 == 0`，所以 frame 0 必须保持未见。初始化使用第一个合法
  训练帧 left frame 1 及其最近时间戳 right frame，而不是读取 frame 0。
- 初始化深度从两张 RGB 用 FoundationStereo 直接计算；固定 32 iterations、非 hierarchical。
  不使用 35–250 mm 数据集范围、不使用 LR consistency、不使用 RAFT、GT depth 或 GT 3D；
  只保留正视差以及公开 EG builder 的 2 m `max_depth`。
- 在线训练只在合法前 80% 非留出帧读取时间同步双目 RGB 和组织 mask。最后 20% 不读取 RGB、
  depth、mask 或轨迹。
- 器械输入是没有图像修正的原始 LND/FK link poses。器械表面点只进入论文球碰撞；不使用
  本项目的三角形接触、持久抓取、位置边界或人工轨迹。
- Future 继续输入记录的真实器械运动，作为已知控制；组织状态保持严格开放环。

## 轨迹与评分

不运行 Shape of Motion。EG 自身已经具有持久物理粒子坐标：冻结 rollout 后，用 frame-0
评测像素和 EG 自身渲染的 alpha/depth 选中最近 Gaussian，并把该查询点固定到 Gaussian 的
父粒子坐标系，随后直接读取原生 PBD 轨迹。若查询像素没有达到统一 `1/255` alpha 阈值，
确定性选择图像中最近的可见 Gaussian，并使用该 Gaussian 自身深度；不读取 GT depth/3D，
所有 fallback 都写入 metadata。若查询帧中连一个可见模型 Gaussian 都不存在，则不存在可辩护的
模型深度：该次预测轨迹写为 NaN，2D/3D 指标为 N/A、覆盖率为0；不会读取 GT，也不会选择图像外
Gaussian 来伪造查询绑定。

评分与 EndoGaussian、EH-SurGS 完全共用：

- protocol：`joint_reconstruction_7to1_future_80to20`；
- Reconstruction：前 80% 内 `frame % 8 == 0`；
- Future：最后 20%，开放环；
- 10 个持久 query、绝对 2D/3D error，不做尺度、ICP 或刚体对齐；
- stereo-left 0.5 倍分辨率 PSNR/SSIM/LPIPS，使用同一冻结器械 mask 和 CUDA 视频解码器。

## 运行

已有环境：`/Media_HDD/jwshan/conda_envs/eg_codex`。单次运行：

SUPER 适配器复用 SIM 仓库中已经审计的 shape-matching 方程实现；复现时需把
[`Embodied_gaussians_fixed_super_best_sim`](https://github.com/wad157/Embodied_gaussians_fixed_super_best_sim/tree/f60b1a6ac54d1168c4010bd1e873a7ab5adab1bb)
以目录名 `embodied_gaussians_fixed_super_best_sim` 放在本仓库同级，固定到提交 `f60b1a6`。

```bash
EG_SUPER_GPU_ID=1 bash scripts/run_embodied_gaussians_super_baseline_once.sh \
  grasp5 repeat_01 0 outputs/embodied_gaussians_super_joint_v1/grasp5/repeat_01
```

三数据集各三个真实 seed，顺序运行并在最后取算术均值和总体标准差：

```bash
EG_SUPER_GPU_ID=1 bash scripts/run_embodied_gaussians_super_three_datasets_three_repeats.sh \
  outputs/embodied_gaussians_super_joint_v1
```

最终汇总写入 `outputs/embodied_gaussians_super_joint_v1/summary/summary.{md,json,csv}`；逐次
目录保存输入/参数审计、初始化 body、原始 PBD rollout、查询轨迹、渲染对和完整指标。

## 正式结果

grasp5、grasp3、grasp1 均实际运行 seeds 0/1/2。表中为算术均值 ± 总体标准差，不使用 smoke、
不挑选最好一次，也不丢弃失败 seed：

| 数据集 | 分区 | 3D mean / RMSE (mm) ↓ | 2D mean / RMSE (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| grasp5 | Reconstruction 7:1 | 41148.397 ± 19541.061 / 44615.404 ± 19895.769 | 87510.372 ± 113921.409 / 485497.950 ± 660396.982 | 9.006 ± 0.008 | 0.0062 ± 0.0005 | 0.8421 ± 0.0002 |
| grasp5 | Future 80:20 | 55991.091 ± 40324.853 / 56429.869 ± 40105.699 | 18685.905 ± 22026.995 / 48793.754 ± 64523.957 | 9.023 ± 0.000 | 0.0025 ± 0.0000 | 0.8380 ± 0.0000 |
| grasp3 | Reconstruction 7:1 | 22242.982 ± 11815.215 / 25730.193 ± 13531.498 | 8198.319 ± 5197.794 / 10199.639 ± 7837.416 | 9.050 ± 0.015 | 0.0049 ± 0.0005 | 0.8374 ± 0.0001 |
| grasp3 | Future 80:20 | 46620.254 ± 25912.195 / 46695.757 ± 25976.670 | 4561.292 ± 1566.874 / 4570.859 ± 1570.665 | 9.031 ± 0.000 | 0.0025 ± 0.0000 | 0.8392 ± 0.0000 |
| grasp1 | Reconstruction 7:1 | N/A / N/A | N/A / N/A | 9.109 ± 0.001 | 0.0030 ± 0.0004 | 0.8470 ± 0.0001 |
| grasp1 | Future 80:20 | N/A / N/A | N/A / N/A | 9.169 ± 0.000 | 0.0025 ± 0.0000 | 0.8400 ± 0.0000 |

### grasp1 N/A 的含义

- seed 0 与 seed 2 的10个查询均由 EG raster 直接覆盖，最大初始重投影误差分别为
  `0.000477 px` 和 `0.000857 px`。
- seed 1 的112个 Gaussian 在查询帧均不位于图像内，其中91个虽在相机前方，但其投影仍全部在
  视野外；所以10个查询没有模型 alpha/depth，2D/3D轨迹覆盖率均为0。
- seed 1 的 rendering 仍完整有效，并正常参与三次 PSNR/SSIM/LPIPS 聚合。
- 聚合器要求全部请求运行都有有限轨迹值；只要其中一次未定义，三次轨迹 mean/std 就整体报告
  N/A。不能只平均 seed 0/2，因为这等价于事后选择成功运行。

该 N/A 是论文原参数 collision-only baseline 离开视野的测评结果，不是缺失实验，也没有通过
额外夹持、数据集专用刚度/阻尼、GT depth/3D、图像外 Gaussian 或后验对齐进行修补。九次运行
均通过协议、完整日程、观测隔离和单次 SHA-256 校验。完整聚合文件位于
[`outputs/embodied_gaussians_super_joint_v1/summary/summary.{md,json,csv}`](../../outputs/embodied_gaussians_super_joint_v1/summary/summary.md)。
