# Note07 经验验证实时报告（已冻结）

更新时间：2025-12-21

> 本文件曾用于 Note07 迭代阶段的“随跑随更”记录。  
> 按 PI 最终决策：**停止新增经验迭代，进入论文写作**。为避免口径漂移，本文件改为“冻结版快照”，论文写作请以收敛稿为准：
>
> - 收敛稿（Table 1 + 主图 + 1H/2H 失败附录）：`Essay/note07_empirical_closure.md`
> - 结果素材总索引（论文图映射/关键数值汇总）：`docs/results_materials_catalog.md`

---

## 1. 最终口径（写作使用）

- 聚合频率：`freq=4H`
- 段内统计：`segment=W`（周；H1/H2 的统计单元）
- 变量映射：
  - 极化：$Q=X_H-X_L$
  - 活跃度：$a=X_H+X_L=1-X_M$
  - 媒体生态代理：
    $$
    r_{\text{proxy}}=\frac{n_{\text{wemedia}}}{n_{\text{wemedia}}+n_{\text{mainstream}}+n_{\text{government}}}
    $$

---

## 2. 最终结果快照（PI 收敛叙事）

> 下表为论文叙事的“主结论入口”，以 `Essay/note07_empirical_closure.md` 的 Table 1 为准。

| Hypothesis | Batch1 (single topic, strict) | Batch3 (expanded) | All (master+batch3) |
|---|---|---|---|
| **H1** Activity → Jump | Pearson r = −0.349 (p = 0.293)\* | Pearson r = 0.159 (p = 0.090) | **Pearson r = 0.241 (p = 0.00798)** |
| **H2** $r_{\text{proxy}}$ → Volatility | Pearson r = 0.098 (p = 0.774) | **Pearson r = 0.265 (p = 0.00434)** | Pearson r = 0.065 (p = 0.484) |
| **H4** CSD (AC1/Var ↑ before jumps) | events = 2；placebo p(AC1)=0.525；p(Var)=0.537 | events = 13；placebo p(AC1)=0.571；p(Var)=0.905 | events = 16；placebo p(AC1)=0.359；p(Var)=0.843 |

\* Batch1 的单话题结果对 `min_posts_public/segment` 较敏感，因此不作为主结果。

**写作一句话版本**

1) **H1 支持（All）**：活跃度越高，段内 jump 强度越大（All 显著）。  
2) **H2 支持但受混杂（Batch3）**：$r_{\text{proxy}}$ 与波动性显著正相关，但控制“段内密度/可用窗口数”后不显著，需要如实披露。  
3) **H4 不定论**：4H 下 placebo 不显著；1H/2H 提频受连续块不足影响（eligible=0），用于界定理论在真实数据中的可观测边界。

---

## 3. 论文用图（定稿）

- H1（All）：`Essay/figures/fig5a_h1_all.png`
- H2（Batch3，密度分组主图）：`Essay/figures/fig5b_h2_batch3_density.png`
- H4（Batch3，对齐图）：`Essay/figures/fig5c_h4_event_batch3.png`

Notebook/脚本生成的对应图（用于溯源）：

- 基础时序：`outputs/figs/empirical/fig7a_*_basic_4h.png`
- H2 密度主图：`outputs/figs/empirical/fig7b_h2_scatter_batch3_density_4h.png`
- H4 事件对齐：`outputs/figs/empirical/fig7c_h4_eventstudy_*_4h.png`

---

## 4. 最小复现命令（参考）

完整复现命令请直接复制收敛稿中的“复现命令”段落：`Essay/note07_empirical_closure.md`。

