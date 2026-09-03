# SUPER 物理重建与未来预测评估协议

## 1. 这次究竟评估什么

同一个模型做两张彼此独立的“考卷”。

1. **重建与重放（reconstruction & resimulation）**：按 EH-SurGS 的 `7:1` 思路，完整视频
   每 8 帧留出 1 帧。留出帧的图像不能参与该帧的视觉残差或材料更新，只用于测量；其余 7 帧
   可以用于在线对齐和刚度估计。
2. **未来预测（future prediction）**：frame `0..1151`（前 80%）用于视觉残差和材料估计；到
   frame `1152` 后立即冻结观测更新，只把真实器械运动作为控制输入，开放环预测
   frame `1152..1439`（后 20%）。

两张考卷都报告 2D/3D 点轨迹误差和渲染 PSNR、SSIM、LPIPS。这里不做逐帧刚体对齐，因为对齐会
把相机、器械或组织整体漂移从误差里抹掉。

## 2. 真值轨迹

### 2.1 2D 真值

人工标注文件是
`data/super/evaluation_v1/manual_tissue_tracks_10/annotations.json`。当前验收结果为：

- 149 个采样帧全部 `human_verified=true`；
- 每帧 10 点，共 1490 条人工观测；
- 坐标是完整 `1920×1080` rectified stereo-left 像素，左上角为原点。

自动 LK 建议只帮助移动光标；只有按 ENTER 确认后的坐标才是真值。

### 2.2 由严格双目深度得到 3D 真值

每个可见点的像素为 `(u,v)`，严格双目深度为 `Z`，冻结内参为

\[
K=\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}.
\]

反投影到左相机 OpenCV 坐标系：

\[
X=\frac{(u-c_x)Z}{f_x},\qquad
Y=\frac{(v-c_y)Z}{f_y},\qquad
\mathbf p_C=[X,Y,Z]^T.
\]

再用冻结外参变换到 SUPER 世界坐标：

\[
\tilde{\mathbf p}_W=T_{WC}^{opencv}\tilde{\mathbf p}_C.
\]

用于计分的 `Z` 不是普通稠密深度，而是同时通过以下条件的
`NNNNNN-depth_high_confidence.npy`：

1. FoundationStereo 给出有限正深度；
2. 翻转左右图再次估计后，左右一致性误差不超过 `1.5 px`；
3. RAFT-Stereo 也给出有效深度；
4. 两个模型的深度相差不超过 `3 mm`。

只允许在标注像素周围最多 `2 px` 内寻找有限的严格深度。不做时间插值，不用普通稠密深度补洞。
因此某点没有可靠深度时，它仍进入 2D 评分，但明确排除于 3D 评分，同时报告 3D 覆盖率。

## 3. 估计轨迹

frame 0 是 TAP 查询帧。每个人工点在查询帧绑定一个最近的**组织 soft Gaussian 持久身份**：

\[
g_i=\arg\min_{g\in\mathcal G_{tissue}}
\|\boldsymbol\mu_g^0-\mathbf p_{i,GT}^0\|_2.
\]

若该查询点没有严格 3D 深度，则退化为最近的 2D 投影 Gaussian。10 个点强制绑定不同身份。之后
不再查看未来人工坐标，而是跟随相同 Gaussian 的物理运动：

\[
\hat{\mathbf p}_i^t=\boldsymbol\mu_{g_i}^t,
\qquad
\hat{\mathbf u}_i^t=\pi(K,T_{CW},\hat{\mathbf p}_i^t).
\]

这对应 Embodied Gaussians / Liang 等方法里“查询点绑定最近的持久物理表示并随模拟状态运动”的
思想，也对应 PhysTwin 用持久质量节点轨迹评价动力学，而不是每帧重新找最近点。每帧重新最近邻
会偷偷换身份，误差会显得虚假地小，本协议禁止这样做。

## 4. 点跟踪指标

对所有人工可见且预测有限的样本集合 `V`，2D 平均误差为

\[
E_{2D}=\frac1{|V|}\sum_{(t,i)\in V}
\|\hat{\mathbf u}_i^t-\mathbf u_{i,GT}^t\|_2\quad[px].
\]

同时报告 RMSE、median、P90、P95 和 maximum。为了能与
Tracking Everything in Robotic-Assisted Surgery / TAP-Vid 口径对照，还报告

\[
\delta_{avg}=\frac15\sum_{\tau\in\{1,2,4,8,16\}}
\frac1{|V|}\sum_{(t,i)\in V}
\mathbb 1[e_{t,i}^{2D}<\tau].
\]

由于当前人工标注全部是 visible，正式结果不会伪造 occlusion prediction，所以不把 AJ/OA 冒充为
已评估能力。

严格 3D 有效集合记作 `V_3D`：

\[
E_{3D}=\frac1{|V_{3D}|}\sum_{(t,i)\in V_{3D}}
\|\hat{\mathbf p}_{i,C}^t-\mathbf p_{i,C,GT}^t\|_2\quad[mm].
\]

同样报告 RMSE、median、P90、P95、maximum，以及
`|V_3D|/|V|` 覆盖率。绝对误差不经过 ICP、尺度或刚体对齐。

## 5. 渲染指标

为了复用 EH-SurGS 的器械掩码协议，真值器械 mask 内的像素同时从预测图和真值图中置零；其余像素
计分。固定使用 stereo-left 和 `960×540`（原分辨率的 0.5 倍），每种方法必须使用相同分辨率与
mask。

### PSNR

\[
MSE=\frac1{3|M|}\sum_{p\in M}\|\hat I(p)-I(p)\|_2^2,
\qquad
PSNR=-10\log_{10}(MSE).
\]

图像归一化到 `[0,1]`，PSNR 越高越好。

### SSIM

\[
SSIM(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
\]

使用 `11×11, σ=1.5` Gaussian window，最终只在完整有效窗口中平均；越接近 1 越好。

### LPIPS

\[
LPIPS(x,y)=\sum_l\frac1{H_lW_l}\sum_{h,w}
\|w_l\odot(\hat f_l^x(h,w)-\hat f_l^y(h,w))\|_2^2.
\]

使用官方 `lpips==0.1.4`、`net=alex`、v0.1 权重；越低越好。不得用普通 AlexNet feature MSE
冒充 LPIPS。

## 6. 防止未来信息泄漏

评测器的读取通道不写模拟状态。每个留出/未来测试帧到来前：

1. 禁止该帧视觉 residual；
2. 禁止该帧触发新的刚度候选；
3. 取消尚未完成、可能跨入测试帧的候选验证；
4. 保留此前已提交的材料场和物理状态；
5. 继续输入器械位姿，因为它是 PhysTwin / Liang 开放环定义里的已知 control sequence；
6. 先物理滚动，再只读记录持久 Gaussian 位置和渲染，不回写人工点或图像。

## 7. 运行

完整双协议：

```bash
bash scripts/run_super_tissue_evaluation.sh \
  outputs/super_tissue_evaluation_YYYYMMDD
```

关键中间产物：

```text
data/super/evaluation_v1/manual_tissue_tracks_10/
├── annotations.json
├── stereo_depth_v1/
├── ground_truth_2d3d_v1.npz
└── ground_truth_2d3d_v1.json

outputs/super_tissue_evaluation_YYYYMMDD/
├── reconstruction_7to1/
│   ├── predicted_tracks.npz
│   ├── render_pairs/
│   ├── evaluation_results.json
│   └── evaluation_results.md
└── future_80to20/
    └── ...
```

CPU 指标与协议测试：

```bash
PYTHONPATH=src:scripts python scripts/test_super_tissue_evaluation.py
```

## 8. 与参考论文口径的关系

- [Liang et al., arXiv:2309.11656](https://arxiv.org/abs/2309.11656)：未来控制开放环滚动、未来
  gap / keypoint error，以及查询点绑定最近持久物理表示；
- [PhysTwin, arXiv:2503.17973](https://arxiv.org/abs/2503.17973)：重建重放、未来预测、人工点
  tracking error 和 PSNR/SSIM/LPIPS 的联合评价；
- [Tracking Everything in Robotic-Assisted Surgery, arXiv:2409.19821](https://arxiv.org/abs/2409.19821)：
  人工可见性标注及 1/2/4/8/16 px 阈值位置准确率；
- [EH-SurGS, arXiv:2501.01101](https://arxiv.org/abs/2501.01101)：7:1 重建划分、器械 mask、
  PSNR/SSIM/LPIPS。

本项目不是复现上述任一论文；这里只迁移它们的评测思想，并明确记录本项目自己的物理表示、采样
频率、双目置信门和开放环边界。
