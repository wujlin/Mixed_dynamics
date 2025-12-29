# 结果素材索引（用于 Essay 写作）

更新时间：2025-12-28

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

### Figure 1：研究路线图（Roadmap）

- `Essay/figures/fig0_framework.pdf`（框架图：Theory → ABM → Empirical）

### Figure 2：理论分岔与有效势

- `Essay/figures/fig1a_bifurcation.pdf`（理论分岔，基准参数）
- `Essay/figures/fig1b_potential.pdf`（有效势 $V_{\mathrm{eff}}(q)$ 形态演化）
- 来源：Note01（`notebooks/01_Theory_and_Potential.ipynb`），详见 `docs/theory_validation_report_note01-04.md`

### Figure 3：网络验证与有限尺寸稳健性

- `Essay/figures/fig2a_network_validation.pdf`（网络 ABM 分岔验证，ER vs BA + 95% CI + 理论分支）
- `Essay/figures/fig2b_binder_u4.pdf`（Binder cumulant $U_4$ 稳健定位临界区）
- `Essay/figures/fig2c_activity.pdf`（稳态 activity $a(r)$：对称 vs activity-coupled asymmetric）
- 来源：Note02（`notebooks/02_Network_Topology.ipynb`）

### Figure 4：临界慢化（CSD）

- `Essay/figures/fig3a_csd_scaling.pdf`（确定性 ODE：$\tau \propto (r_c-r)^{-1}$ 的 log-log 标度）
- `Essay/figures/fig3b_csd_sde_vs_abm.pdf`（SDE vs ABM：time-aligned short-lag autocorrelation 随 $r\to r_c$ 上升）
- `Essay/figures/fig3c_csd_timeseries.pdf`（时间序列示例：远离临界 vs 近临界的弛豫差异）
- 来源：Note03（`notebooks/03_Critical_Slowing_Down.ipynb`）

### Figure 5：结构效应相图（阈值/密度/生态/耦合）

- `Essay/figures/fig4a_chi_rc_landscape.pdf`（阈值参数景观：理论 $r_c(\phi,\theta)$；浅色遮罩为无相变区（$\chi\le 2$），白色留白为无效域（$\phi\le\theta$））
- `Essay/figures/fig4b_k_effect.pdf`（信息密度 $k$ 的效应：$\chi(k)$ 与 $r_c(k)$；ABM 含 95\% CI）
- `Essay/figures/fig4c_media_ratio.pdf`（媒体生态 $n_w/n_m$ 的效应：理论 vs ABM（95\% CI））
- `Essay/figures/fig4d_beta_effect.pdf`（局部耦合 $\beta$：$r_c(\beta)$ 迁移（ABM 95\% CI）；灰点线为理论 $\beta=0$ 参考）
- 来源：Note04（`notebooks/04_Sensitivity_Chi_Landscape.ipynb`）

### Figure 6：经验验证（按 PI 收敛）

- `Essay/figures/fig5a_activity_jump.pdf`（Activity–jump association：pooled dataset 上支持）
- `Essay/figures/fig5b_media_volatility_density.pdf`（Media-dominance–volatility association：high-density subset 上更强，但受密度混杂影响；图内不嵌统计量）
- `Essay/figures/fig5c_h4_event_batch3.pdf`（Early-warning 事件对齐：$AC1(|Q|)$ 与 $\mathrm{Var}(|Q|)$；当前不进正文主结果，仅在 Discussion/补充材料说明）
- 来源：Note07（`notebooks/07_Empirical_Validation.ipynb`）；最终叙事见 `Essay/note07_empirical_closure.md`

---

## 2. 可直接写进 Results 的“关键数值”

### 2.1 理论/模拟（Note01–04）

- 对称域理论临界点：$r_c\approx 0.7533$（基准参数见上）。
- Binder crossing（对称域，bootstrap 95% CI）在大 $N$ 对（1000–2000）上与理论几乎重合：见 `docs/theory_validation_report_note01-04.md` 的“有限尺寸效应”小节。

### 2.2 经验（Note07）

> 以 `Essay/note07_empirical_closure.md` 的 Table 1 为准。

- Activity–jump association（pooled dataset）：Pearson $r=0.241$, $p=0.00798$。
- Media-dominance–volatility association（high-density subset）：Pearson $r=0.265$, $p=0.00434$；但控制段内窗口数（密度）后 partial $r=0.078$, $p=0.411$（提示混杂）。
- 更苛刻的 early-warning 相关检验（原 H3/H4 口径）当前不作为正文主结果，建议仅在 Discussion 限制与补充材料中说明其对观测密度/连续性的要求。

---

## 3. 结果文档（写作素材）分组

### 3.1 理论/模拟（Note01–04）

- 图文版报告（主入口）：`docs/theory_validation_report_note01-04.md`
- notebook：`notebooks/01_Theory_and_Potential.ipynb`，`notebooks/02_Network_Topology.ipynb`，`notebooks/03_Critical_Slowing_Down.ipynb`，`notebooks/04_Sensitivity_Chi_Landscape.ipynb`

### 3.2 经验（Note07）

- 收敛稿（主入口）：`Essay/note07_empirical_closure.md`
- 汇总报告（面向仓库读者）：`docs/note07_empirical_validation_report.md`
- 实时报告（已冻结，避免口径漂移）：`docs/empirical_validation_live.md`
