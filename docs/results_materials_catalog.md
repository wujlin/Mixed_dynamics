# 结果素材索引（用于 Essay 写作）

更新时间：2025-12-21

> 目标：把 Note01–04（理论/模拟）与 Note07（经验）产出的**关键结论、关键数值、关键图表**统一归档，作为 Results 写作的“单一事实来源”，避免口径漂移。

---

## 0. 统一口径（写作应以此为准）

### 0.1 理论/模拟（Note01–04）

- 理论临界点基准（对称域）：$\chi\approx 9.5962$，$r_c\approx 0.753279$（$\phi=0.54,\theta=0.46,k=50,n_m=10,n_w=5$）。
- 对称系统的分岔指标：使用 $|Q|$ 或“对齐后的 signed $Q$”，不能用 $\langle Q\rangle$（会被 $\pm$ 分支抵消）。
- 有限尺寸估计 $r_c$：优先用 Binder cumulant 交点（消灭 susceptibility 伪峰/双峰）。

权威图文报告：`docs/theory_validation_report_note01-04.md`

### 0.2 经验验证（Note07，按 PI 收敛）

- 叙事结构：Two‑Tier（结构性 H2 vs 动力学 H1/H4），并停止新增经验迭代。
- 主口径（用于论文主结果）：`freq=4H`，`segment=W`。
- 统计单元：段（segment）是相关/回归的样本点；段内由窗口（4H windows）聚合得到。

权威收敛稿（含 Table 1、主图与附录）：`Essay/note07_empirical_closure.md`

---

## 1. 论文图表（Essay/figures）与来源映射

> 说明：论文正文引用应优先使用 `Essay/figures/` 下的“定稿图”。对应的 notebook 输出图主要位于 `outputs/figs/`。

### Figure 1：理论分岔与有效势

- `Essay/figures/fig1a_bifurcation.png`（理论分岔）
- `Essay/figures/fig1b_potential.png`（有效势 $V(q)$）
- 来源：Note01（`notebooks/01_Theory_and_Potential.ipynb`），详见 `docs/theory_validation_report_note01-04.md`

### Figure 2：网络验证与有限尺寸稳健性

- `Essay/figures/fig2a_network_validation.png`（网络 ABM 分岔验证，含对称/非对称对照）
- `Essay/figures/fig2b_binder_u4.png`（Binder cumulant 稳健定位临界区）
- 来源：Note02（`notebooks/02_Network_Topology.ipynb`）

### Figure 3：临界慢化（CSD）

- `Essay/figures/fig3_csd.png`（弛豫时间/自相关随 $r\to r_c$ 上升）
- 来源：Note03（`notebooks/03_Critical_Slowing_Down.ipynb`）

### Figure 4：结构效应相图（阈值/密度/生态/耦合）

- `Essay/figures/fig4a_chi_rc_landscape.png`（$\chi$ 与 $r_c$ 的阈值相图）
- `Essay/figures/fig4b_k_effect.png`（信息密度 $k$ 的效应：理论 vs ABM）
- `Essay/figures/fig4c_media_ratio.png`（媒体生态 $n_w/n_m$ 的效应：理论 vs ABM）
- `Essay/figures/fig4d_beta_effect.png`（局部耦合 $\beta$ 造成 $r_c(\beta)$ 迁移 + 分支偏置诊断）
- 来源：Note04（`notebooks/04_Sensitivity_Chi_Landscape.ipynb`）

### Figure 5：经验验证（按 PI 收敛）

- `Essay/figures/fig5a_h1_all.png`（H1：All 上 Activity→Jump 支持）
- `Essay/figures/fig5b_h2_batch3_density.png`（H2：Batch3 上显著，但密度为混杂）
- `Essay/figures/fig5c_h4_event_batch3.png`（H4：Placebo 不显著/受连续性限制，作为局限性）
- 来源：Note07（`notebooks/07_Empirical_Validation.ipynb`）；最终叙事见 `Essay/note07_empirical_closure.md`

---

## 2. 可直接写进 Results 的“关键数值”

### 2.1 理论/模拟（Note01–04）

- 对称域理论临界点：$r_c\approx 0.7533$（基准参数见上）。
- Binder crossing（对称域，bootstrap 95% CI）在大 $N$ 对（1000–2000）上与理论几乎重合：见 `docs/theory_validation_report_note01-04.md` 的“有限尺寸效应”小节。

### 2.2 经验（Note07）

> 以 `Essay/note07_empirical_closure.md` 的 Table 1 为准。

- H1（All=master+batch3）：Pearson $r=0.241$, $p=0.00798$。
- H2（Batch3）：Pearson $r=0.265$, $p=0.00434$；但控制段内窗口数（密度）后 partial $r=0.078$, $p=0.411$（提示混杂）。
- H4：4H 下 Placebo 不显著；1H/2H 提频因连续块不足导致 eligible=0（不可评估）——作为“可观测边界”的负结果写入附录。

---

## 3. 结果文档（写作素材）分组

### 3.1 理论/模拟（Note01–04）

- 图文版报告（主入口）：`docs/theory_validation_report_note01-04.md`
- notebook：`notebooks/01_Theory_and_Potential.ipynb`，`notebooks/02_Network_Topology.ipynb`，`notebooks/03_Critical_Slowing_Down.ipynb`，`notebooks/04_Sensitivity_Chi_Landscape.ipynb`

### 3.2 经验（Note07）

- 收敛稿（主入口）：`Essay/note07_empirical_closure.md`
- 汇总报告（面向仓库读者）：`docs/note07_empirical_validation_report.md`
- 实时报告（已冻结，避免口径漂移）：`docs/empirical_validation_live.md`

