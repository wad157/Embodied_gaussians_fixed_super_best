# SUPER grasp1 / grasp3 / grasp5 新协议三次评估

本文记录 grasp1、grasp3、grasp5 在统一 Joint 协议下的最新三次评估。
表中均为 mean ± sample standard deviation，运行次数 n=3。

## 协议

- 每种方法对完整序列执行一次连续 rollout。
- 前 80% 为训练前缀，其中每 8 帧留出 1 帧进行 Reconstruction 计分；留出帧不能执行视觉或材料更新。
- 后 20% 为严格开放环 Future；视觉修正与材料更新全部冻结。
- 三种方法为 Pure PBD、PBD + trajectory、PBD + trajectory + online stiffness。
- 在线刚度版本使用 direct H2/H3 与接触后持续边应变，更新全局 distance stiffness 和速度阻尼。
- 正式配置关闭 post-trajectory RGB residual。

| 数据集 | 完整帧 | 前 80% 训练 | 后 20% Future |
|---|---:|---:|---:|
| grasp1 | 0–4204 | 0–3363 | 3364–4204 |
| grasp3 | 0–2061 | 0–1648 | 1649–2061 |
| grasp5 | 0–1439 | 0–1151 | 1152–1439 |

## grasp1

### Reconstruction：前 80% 内 7:1 留出

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.8082 ± 0.0254 | 30.4708 ± 0.4799 | 17.2561 ± 0.0030 | 0.778103 ± 0.000606 | 0.487804 ± 0.000166 |
| PBD + trajectory | 1.2269 ± 0.0109 | 21.6063 ± 0.1099 | **17.2580 ± 0.0030** | 0.783854 ± 0.000284 | 0.483591 ± 0.000365 |
| **PBD + trajectory + online stiffness** | **1.1384 ± 0.0000** | **19.6570 ± 0.0000** | 17.2521 ± 0.0000 | **0.784502 ± 0.000000** | **0.483429 ± 0.000000** |

### Future：后 20% 开放环

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 2.0660 ± 0.1108 | 37.6246 ± 2.1441 | 16.9611 ± 0.0214 | 0.754402 ± 0.003369 | 0.485364 ± 0.000885 |
| PBD + trajectory | 2.0387 ± 0.0710 | 36.6171 ± 1.2949 | 16.9739 ± 0.0005 | 0.756207 ± 0.001644 | 0.482693 ± 0.001478 |
| **PBD + trajectory + online stiffness** | **1.9369 ± 0.0000** | **31.7454 ± 0.0000** | **16.9975 ± 0.0000** | **0.762827 ± 0.000000** | **0.480510 ± 0.000000** |

在线刚度相对 trajectory 的 Reconstruction 3D/2D 误差降低 7.21% / 9.02%；
Future 3D/2D 误差降低 4.99% / 13.30%，Future 三项渲染指标也全部改善。

## grasp3

### Reconstruction：前 80% 内 7:1 留出

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 2.1188 ± 0.0000 | 33.9941 ± 0.0000 | 16.5203 ± 0.0000 | 0.745028 ± 0.000000 | 0.507461 ± 0.000000 |
| **PBD + trajectory** | **0.8002 ± 0.0016** | **11.6526 ± 0.0618** | **16.6310 ± 0.0007** | **0.756546 ± 0.000058** | **0.500856 ± 0.000130** |
| PBD + trajectory + online stiffness | 0.9145 ± 0.0162 | 12.6347 ± 0.1071 | 16.6095 ± 0.0051 | 0.755553 ± 0.000196 | 0.502689 ± 0.000368 |

### Future：后 20% 开放环

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.8196 ± 0.0000 | 25.1578 ± 0.0000 | **16.7987 ± 0.0000** | **0.733477 ± 0.000000** | 0.505239 ± 0.000000 |
| **PBD + trajectory** | **1.3208 ± 0.0886** | 23.7806 ± 1.9826 | 16.5869 ± 0.0132 | 0.724790 ± 0.001865 | 0.513582 ± 0.000625 |
| PBD + trajectory + online stiffness | 1.3860 ± 0.0839 | **20.3308 ± 0.4244** | 16.7617 ± 0.0024 | 0.732978 ± 0.000685 | **0.505007 ± 0.000862** |

在线刚度相对 trajectory 将 Future 2D 误差降低 14.51%，并改善全部 Future 渲染指标；
但 Reconstruction 3D/2D 分别退化 14.29% / 8.43%，Future 3D 退化 4.94%。

## grasp5

### Reconstruction：前 80% 内 7:1 留出

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.5035 ± 0.0248 | 26.5160 ± 0.2966 | **22.7108 ± 0.0074** | 0.785567 ± 0.000073 | **0.472204 ± 0.000515** |
| PBD + trajectory | 0.7371 ± 0.0290 | 10.2642 ± 0.4899 | 22.5635 ± 0.0499 | 0.789939 ± 0.001038 | 0.475201 ± 0.001353 |
| **PBD + trajectory + online stiffness** | **0.6971 ± 0.0324** | **9.5021 ± 0.3898** | 22.6290 ± 0.0125 | **0.790696 ± 0.000408** | 0.474278 ± 0.000975 |

### Future：后 20% 开放环

| 方法 | 3D Tracking (mm) ↓ | 2D Tracking (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Pure PBD | 1.6954 ± 0.0970 | 28.1901 ± 0.9342 | **21.8616 ± 0.0356** | **0.750567 ± 0.001622** | 0.478282 ± 0.001050 |
| PBD + trajectory | 1.3238 ± 0.2349 | 17.1282 ± 2.4896 | 21.2649 ± 0.0361 | 0.743987 ± 0.002942 | 0.480492 ± 0.006259 |
| **PBD + trajectory + online stiffness** | **1.1734 ± 0.0665** | **14.5807 ± 1.3649** | 21.5574 ± 0.1066 | 0.746897 ± 0.002003 | **0.478710 ± 0.005939** |

在线刚度相对 trajectory 的 Reconstruction 3D/2D 误差降低 5.43% / 7.42%；
Future 3D/2D 误差降低 11.36% / 14.87%，Future 三项渲染指标全部改善。

## 结论与原始统计

- 三个数据集的 Future 2D 与 Future 渲染指标均因在线刚度更新而改善。
- grasp1 与 grasp5 的 Reconstruction/Future 3D、2D 均优于 trajectory 基线。
- grasp3 的 Future 2D 和渲染受益，但 Reconstruction 与 Future 3D 退化，表明当前策略
  仍存在跨场景累计材料过修正。
- 三个数据集的协议验证均通过。
- 精确均值、标准差、三次单独数值与协议字段见
  [grasp1 summary.json](outputs/super_joint_h2direct_persistent_three_datasets_three_repeats_20260911_v1/grasp1/summary/summary.json)、
  [grasp3 summary.json](outputs/super_joint_h2direct_persistent_three_datasets_three_repeats_20260911_v1/grasp3/summary/summary.json) 和
  [grasp5 summary.json](outputs/super_joint_h2direct_persistent_three_datasets_three_repeats_20260911_v1/grasp5/summary/summary.json)。
