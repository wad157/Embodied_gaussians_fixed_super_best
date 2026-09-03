# CoTracker、GauSTAR 与 XPBD 协同跟踪方案（V1.1 因果稀疏深度版）

> 版本：V1.1，2026-08-28
> 状态：固定范围绑定、训练段因果 3D flow、q/qd 更新、
> 物理安全回溯和 mode-2 Gaussian 同步已接入正式离线循环；
> 尚未完成三组全序列指标验证。

## 1. 目标与当前状态

目标不是让 RGB loss 独自搜索组织的大位移，而是先通过 **CoTracker 二维对应 + 双目深度** 得到可靠的三维表面运动，再把每条轨迹绑定到初始三维位置附近的一组物理表面粒子，按固定距离权重进行 XPBD 状态校正，最后通过现有 mode-2 三角面绑定带动 Gaussian。

一句话流程：

```text
XPBD预测 → CoTracker二维轨迹 → 双目反投影 → 3D表面流
        → 初始6 mm物理表面范围固定绑定 → 距离加权校正位置/速度
        → 物理安全投影 → mode-2带动Gaussian
        → 在线刚度更新（仅完整方法开启）
```

本文是下一阶段的固定算法设计，不表示指标已经提升。
当前已完成 CoTracker3 运动可视化、frame-0 固定范围绑定、
训练段 575 对因果 flow-depth 观测、innovation 散射、模拟器安全
提交及离线循环接入；正式结论必须由完整三组实验给出。

V1 明确不做：

- 不选择最大权重物理粒子或最大权重 Gaussian；
- 不使用局部仿射 patch、复杂三角面 Jacobian 或每帧重新匹配；
- 不直接写 Gaussian 位置；
- 不启用 GauSTAR 的解除绑定和重新网格化。

V1 固定的绑定参数为：

| 参数 | 固定值 |
|---|---:|
| 初始物理表面半径 \(R\) | 6 mm |
| 粒子不足时的备用半径 | 8 mm |
| 最少可动表面粒子 | 3 |
| 距离权重宽度 \(\sigma\) | 3 mm |
| 粒子编号和范围权重 | 轨迹生命期内固定 |

视觉增益 \(\gamma\)、速度增益 \(\beta\)、柔顺度 \(\alpha\) 和 maximum step 属于状态更新调参，不改变上述固定绑定定义；先通过诊断确定数值，再锁定正式评测配置。

---

## 2. 各模块的职责

| 模块 | 主要职责 |
|---|---|
| CoTracker3 | 建立跨帧二维材料点对应，解决大位移搜索 |
| GauSTAR tracking | 提供“二维运动 + 深度 → 三维 surface scene flow”思路 |
| 双目深度 | 把二维目标提升到三维，并提供左右目一致性检查 |
| 范围加权绑定 | 把一条轨迹固定绑定到附近一组物理表面粒子 |
| XPBD | 保持距离、形状、体积、接触和固定区域等物理一致性 |
| 三角面绑定 | 将物理粒子运动传给高分辨率表面和 Gaussian |
| 渲染评分 | 计算 PSNR/SSIM/LPIPS，不反向修改正式三组的状态 |
| 在线刚度 | 用接受后的真实形变证据改善未来开放环预测 |

参考：

- [CoTracker3](https://cotracker3.github.io/)
- [GauSTAR](https://eth-ait.github.io/GauSTAR/)
- [GauSTAR 论文](https://arxiv.org/abs/2501.10283)
- [TrackerSplat](https://arxiv.org/abs/2604.02586)

---

## 3. 二维轨迹与物理表面范围的固定对应

### 3.1 用首帧深度确定轨迹的三维中心

对于轨迹 \(j\) 的初始像素 \(u^0_j\)，使用首帧深度反投影，并变换到模拟使用的 table/world 坐标：

\[
X^0_j
=T_{Wc}\!\left[D_0(u^0_j)K_c^{-1}\tilde u^0_j\right].
\]

只接受组织 mask 内、深度有效且与当前渲染表面深度一致的初始点。
grasp5 V1 初始化还固定使用
`masks_left/tool_dilated/000000-tool-dilated.png` 排除器械及其安全边界，
避免把器械纹理轨迹错误绑定成组织材料点。

### 3.2 在初始表面建立固定范围

令 \(\mathcal S\) 为当前物理组织的表面粒子集合。V1 在初始位置周围寻找：

\[
\mathcal N_j
=\left\{
p\in\mathcal S:\|q_p^0-X_j^0\|_2\le R
\right\},
\qquad R=6\ {\rm mm}.
\]

如果范围内少于 3 个可动物理表面粒子，将 \(R\) 扩大到 \(8\ {\rm mm}\)；仍不足 3 个则丢弃该轨迹。固定粒子可以参与表面位置计算，但逆质量为零，不能被视觉写回移动。

使用初始距离计算高斯权重：

\[
\widetilde w_{jp}
=\exp\!\left(
-\frac{\|q_p^0-X_j^0\|_2^2}{2\sigma^2}
\right),
\qquad \sigma=3\ {\rm mm},
\]

\[
w_{jp}
=\frac{\widetilde w_{jp}}
{\sum_{k\in\mathcal N_j}\widetilde w_{jk}+\varepsilon}.
\]

因此轨迹代表的物理表面范围中心为：

\[
\boxed{
Y_j(q)=\sum_{p\in\mathcal N_j}w_{jp}q_p
}.
\]

轨迹建立后永久保存 `particle_ids`、`particle_weights`、`binding_radius` 和初始三维中心。粒子集合与权重不能逐帧重新搜索，否则材料对应会在组织表面滑动。

### 3.3 左右目共享同一个物理范围

同一个物理范围中心 \(Y_j\) 分别投影到左右目作为 query：

\[
u_j^L=\pi_L(Y_j),\qquad u_j^R=\pi_R(Y_j).
\]

左右目轨迹共享同一个 `material_track_id`、粒子编号和固定权重。只有一侧可靠时保留单目二维约束；两侧都可靠时融合三维运动。

### 3.4 二维目标

若跟踪点在时刻 \(t\) 的像素为 \(u^c_{j,t}\)，CoTracker 给出下一时刻对应点：

\[
\hat u^c_{j,t+1}=\operatorname{Track}_c(u^c_{j,t}),
\qquad c\in\{L,R\}.
\]

### 3.5 深度反投影

利用相机内参 \(K_c\)、相机到世界坐标变换 \(T_{Wc}\) 和深度 \(D^c\)，得到：

\[
\hat X^c_{j,t}
=T_{Wc}\!\left[D^c_t(u^c_{j,t})K_c^{-1}\tilde u^c_{j,t}\right].
\]

因此单个相机观察到的三维表面流为：

\[
F^{obs,c}_{j,t}
=\hat X^c_{j,t+1}-\hat X^c_{j,t}.
\]

左右目结果根据置信度融合：

\[
F^{obs}_{j,t}
=\frac{\sum_c w^c_{j,t}F^{obs,c}_{j,t}}
       {\sum_c w^c_{j,t}+\varepsilon}.
\]

第一版以三维增量为主，绝对深度位置只作为弱防漂移约束，以降低双目深度逐帧偏移的影响。

### 3.6 稀疏深度之间的因果保持

现有 FoundationStereo 训练段深度约每 10 个视频帧一张，
CoTracker 每 2 帧给出一次轨迹。为了不用未来深度倒插值，对轨迹
(j) 仅前向保持最近一次直接深度帧 τ 的样本：

\[
D_{j,t}=D_{j,\tau},\qquad
c^{depth}_{j,t}=c^{depth}_{j,\tau}
\exp\!\left(-\frac{t-\tau}{\tau_d}\right),
\quad \tau\le t.
\]

当新深度帧到来时立即更新 (D_{j,t})；中间帧仍通过新的
CoTracker 像素坐标更新 (x/y) 方向的三维流，只是深度置信度随
样本年龄衰减。V1.1 使用 τd=10 帧、最大保持年龄 12 帧；
该策略绝不把后一张深度传回更早时刻。

---

## 4. 只修正 PBD 没有解释的运动

XPBD 已经预测了一部分真实运动，不能再把完整观测位移重复加一次。先计算物理预测的表面流：

\[
F^{pbd}_{j,t}
=\sum_{p\in\mathcal N_j}w_{jp}
\left(q^-_{p,t+1}-q^+_{p,t}\right)
=Y_j(q^-_{t+1})-Y_j(q^+_t).
\]

真正需要视觉更正的是创新量：

\[
r_{j,t}=F^{obs}_{j,t}-F^{pbd}_{j,t}.
\]

- \(r\approx0\)：PBD 已经解释了运动，不应强行移动粒子；
- \(r\) 较大且置信度高：PBD 状态需要校正；
- \(r\) 持续存在：在状态校正后可作为材料参数错误的证据。

---

## 5. 按固定范围权重校正物理粒子

相机只能观测表面。V1 不求复杂三角面 Jacobian，而是把每个创新量 \(r_j\) 作为一个范围位置约束，按固定权重和粒子逆质量分配：

\[
\Delta q_{jp}
=\gamma
\frac{c_jm_p^{-1}w_{jp}}
{\sum_{k\in\mathcal N_j}m_k^{-1}w_{jk}^{2}+\alpha_j}
r_j,
\qquad p\in\mathcal N_j.
\]

其中：

- \(c_j\) 是轨迹、深度、mask 和可见性的联合置信度；
- \(m_p^{-1}=0\) 的 anchor/fixed 粒子不会移动；
- \(\gamma\) 是视觉状态增益；
- \(\alpha_j\) 是视觉约束柔顺度；
- 单步粒子修正仍受 maximum step 限制。

多条轨迹同时作用于一个粒子时，先累积候选校正，再按联合置信度求平均；随后统一执行 XPBD 距离、体积、形状、anchor 和穿透安全投影。V1 不把范围内每个粒子当成互不相关的孤立更新。

接受安全投影后的最终校正 \(\Delta q\) 必须同时更新位置和速度：

\[
q^+_{t+1}=q^-_{t+1}+\Delta q,
\]

\[
\dot q^+_{t+1}=\dot q^-_{t+1}
+\beta\frac{\Delta q}{\Delta t}.
\]

只更新位置会保留错误旧速度，下一物理步可能把组织再次拉回错误方向。

---

## 6. Gaussian 怎样跟随

Gaussian 不直接跟随 CoTracker 像素，而是继续服从现有物理—三角面绑定：

\[
q\longrightarrow v(q)\longrightarrow
\{\mu_g,R_g,S_g\}.
\]

三角面顶点更新后：

- Gaussian 中心 \(\mu_g\) 跟随三角面重心；
- 旋转 \(R_g\) 跟随三角面局部坐标系；
- 尺度 \(S_g\) 根据三角面边长、面积和面内形变更新。

因此无需给每个 Gaussian 单独添加一套不受物理约束的位移，也不选择范围内“权重最大的 Gaussian”。物理粒子只更新一次，随后调用现有 `update_gaussian_transforms()` 完成 Gaussian 同步，避免重复叠加视觉运动。

---

## 7. 置信度、平滑与安全门

综合置信度建议写成：

\[
w_j=w_{track}\,w_{cycle}\,w_{stereo}\,w_{depth}\,
w_{mask}\,w_{tool}\,w_{visible}.
\]

主要过滤条件包括 CoTracker 可见性、前后向一致性、左右目重投影/极线一致性、深度突变、组织 mask、器械遮挡及渲染可见性。

V1 的空间连续性来自 6 mm 固定范围权重和后续 XPBD 约束，不再额外加入局部仿射 patch 或第二套网格流平滑。器械接触附近仍通过 tool mask、深度边界和物理安全门避免跨区域错误传播。每次状态写回前保留：

- 不新增翻转四面体；
- 不增加低体积四面体；
- volume、penetration 和 anchor 不越界；
- 无效或低置信度轨迹不能驱动物理状态。

GauSTAR 的拓扑解除绑定和重新网格化不进入 V1，因为 grasp5 没有明确组织撕裂，而重网格会破坏现有四面体、接触、刚度及 Gaussian 绑定关系。

---

## 8. 轨迹视觉修正与在线刚度的顺序

### 8.1 轨迹视觉修正是正式视觉组的唯一状态更新

正式三组评测中，第二、三组都只用本文的 flow-depth innovation 修改物理状态：

\[
q^+_{t+1}=q^-_{t+1}+\Delta q_{flow},\qquad
\dot q^+_{t+1}=\dot q^-_{t+1}
+\beta\frac{\Delta q_{flow}}{\Delta t}.
\]

旧的 RGB residual 优化器不作为独立正式方法，也默认不叠加到这三组中；RGB
图像只用于计算 PSNR/SSIM/LPIPS。这样第二、三组之间唯一差异就是刚度更新，
轨迹视觉修正本身的收益也能直接相对 Pure PBD 归因。

### 8.2 刚度在状态可靠后更新

先证明 flow-depth 状态校正有效，再恢复在线刚度，避免同时修改状态和材料后无法判断收益来源。刚度可在 log 空间小步更新：

\[
\log k_i^{new}=
\operatorname{clip}\!\left(
\log k_i+\eta s_i,
\log k_{min},\log k_{max}
\right),
\]

其中 \(s_i\) 由接受后的视觉创新量、边应变和接触区域共同决定。刚度更新必须提升未来开放环预测，而不能只增加提交次数。

---

## 9. 实施和消融顺序

1. **A：绑定诊断**——为 CoTracker 轨迹生成固定粒子范围，检查范围粒子数、权重和、初始三维距离和左右目共享 ID；不修改模拟状态。
2. **B：观测诊断**——生成 CoTracker + 深度三维表面流，比较 \(F^{obs}\)、\(F^{pbd}\) 和创新量；不修改模拟状态。
3. **C：状态校正**——`PBD + flow-depth range weighting`，刚度冻结，验证位置/速度更新和物理安全。
4. **D：材料估计**——在完全相同的轨迹视觉修正上加入在线刚度，要求未来
   80/20 明确优于不更新刚度的轨迹视觉组。

### 9.1 2026-08-28 当前实现记录

已完成：

- `flow_depth_particle_observer.py`：固定范围绑定、二维轨迹与深度反投影、
  (F^{obs}-F^{pbd}) innovation、范围加权位置/速度候选更新；
- `build_super_cotracker_range_bindings.py`：grasp5 frame-0 非侵入式绑定诊断；
- `diagnose_super_cotracker_flow_depth_observations.py`：已有深度帧上的非侵入式
  3D flow 诊断；
- `build_super_cotracker_causal_sparse_depth_observations.py`：利用现有
  FoundationStereo 稀疏深度和因果置信度衰减，构造训练段
  15 Hz 轨迹观测；
- `apply_flow_depth_particle_state_update()`：四面体、fixed、anchor 与 penetration
  门控后的原子 q/qd 提交，并调用现有 mode-2 Gaussian 更新；
- `example_embodied_super_offline.py --visual-feedback-mode trajectory`：已在
  正式离线帧循环中按目标帧消费观测，缓存源帧 q+，并遵守
  reconstruction 7-1 与 future 80/20 观测边界；
- CPU 门已验证：固定粒子不动、只增加 innovation 而非完整观测 flow、位置
  maximum step 生效、接受的位置修正同步更新速度。

grasp5 正式初始化结果：483 条 CoTracker 候选中，膨胀器械 mask 排除 17 条，
剩余 466 条全部绑定成功；每条轨迹包含 6--22 个物理表面粒子，中位数为 11。

训练段现有 117 帧可直接使用的 FoundationStereo 深度锚点。
因果前向保持后，`0..1150` 共形成 575 个 CoTracker 采样对，
232599 个请求中接受 230844 个，接受率 99.25%；有效观测位移
中位数为 0.033 mm，95% 分位为 0.469 mm。`1152` 之后的观测根本
没有写入文件，因此 future 20% 无法被视觉更新读取。

真实 GPU 循环小测中，`0→2` 使用 464 条有效轨迹更新 611 个
物理粒子；0.5 mm 候选因低体积四面体触发安全回溯，以 0.5
比例接受，实际最大位移 0.25 mm，无翻转或低体积四面体留在提交状态。

正式只比较以下三组，不再扩展成五组：

| 方法 | 用途 |
|---|---|
| Pure PBD | 物理基线 |
| PBD + CoTracker/depth trajectory correction | 验证固定范围轨迹视觉状态修正 |
| PBD + CoTracker/depth trajectory correction + stiffness | 完整方法，只额外开启在线刚度 |

第二组和第三组必须使用相同的 CoTracker 轨迹、深度、固定粒子范围、位置/速度
增益和物理安全门；第三组唯一额外变量是在线刚度更新。旧 RGB residual 不作为
独立正式对照组，避免把评测扩成五组并混淆刚度收益来源。

---

## 10. 因果评估与最终指标

### 重建 7-1

留出测试帧只能评分，不能用该帧的 RGB、CoTracker 目标、深度或人工轨迹修改状态和刚度。

### 未来预测 80-20

前 80%（grasp5 为 `0..1151`）允许估计状态与材料；从 `1152` 开始冻结：

- CoTracker/光流观测；
- RGB 和深度更新；
- 视觉 residual；
- 在线刚度更新。

后 20% 只使用训练段最终状态、刚度和允许的器械控制输入开放环预测。

最终报告：

\[
E_{2D}=\frac{1}{N}\sum_i\|\hat u_i-u_i^{gt}\|_2,
\qquad
E_{3D}=\frac{1}{N}\sum_i\|\hat X_i-X_i^{gt}\|_2,
\]

以及 `PSNR ↑`、`SSIM ↑`、`LPIPS ↓`。同时记录有效轨迹数量、三维观测接受率、粒子修正次数/幅度、刚度提交次数/幅度、FPS 和物理安全事件。

---

## 11. 最终结论

V1 不是“用 CoTracker 直接拖动 Gaussian”，也不是“一个轨迹只拉一个粒子”，而是：

> CoTracker 找二维对应，深度把它升维成三维运动；每条轨迹固定绑定初始位置周围 6 mm 的物理表面粒子并按 \(\sigma=3\) mm 的距离权重校正；XPBD 保证物理安全，mode-2 带动 Gaussian；完整组再单独开启在线刚度以提升未来预测。

其中最关键的公式是：

\[
\boxed{r=F^{obs}-F^{pbd}}
\]

以及：

\[
\boxed{
Y_j(q)=\sum_{p\in\mathcal N_j}w_{jp}q_p
}
\]

即只校正物理模型没有解释的部分，并把一条轨迹的证据连续分配给固定范围内的一组表面粒子，而不是把完整视觉运动重复叠加到 PBD 上或只更新一个孤立粒子。
