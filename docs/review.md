Review 意见：
关于 "medium/neutral" (点1)：
建议简化：确实有点啰嗦。既然你后面说了 (M) serves as a moderate buffer，前面直接用 moderate ((M)) 最统一，也最符合物理直觉（温和派）。或者只用 neutral ((M))，但 moderate 更有“中间态”的感觉。
推荐：high ((H)), moderate ((M)), and low ((L))。简单有力。
关于 "decouple symmetry breaking..." (点2)：
确实有点硬：这对物理学家很友好，但对一般读者有点抽象。
建议改写：把重点放在“坐标变换”的直观意义上。
推荐：These coordinates separate the directional bias (q) from the total intensity (a), allowing us to recover the original fractions via \rho_H=(a+q)/2 and \rho_L=(a-q)/2.
这样既保留了精确性（能还原回去），又解释了为什么要这么定义（分离方向和强度）。
关于 (Q,A) vs (q,a) (点3)：
建议推后：这里是 Methods 的开篇，讲的是一般理论框架。突然引入仿真/经验符号会打断逻辑流。
策略：保持这里纯净，讲清楚物理定义。等到 Empirical Validation 或 Results 章节，再说“我们在数据中用 $Q$ 和 $A$ 来估计理论上的 $q$ 和 $a$”。这样层次分明。
总结与行动建议
你需要做的是：
修改状态定义：把 medium/neutral 统一为 moderate。
柔化最后一句：用 separate directional bias from total intensity 替换 decouple symmetry breaking。
保持纯净：这里不提 $Q/A$。



是的，我们用到的标注模型是qwen3-8B。

第 2 段：阈值更新规则
Review 意见：
这一段写得非常扎实，但我发现了一个可以微调的地方，让它读起来更“顺滑”：
关于 following threshold-response abstractions...：
这个引用放在这里有点打断“更新规则”的描述流。
建议移到开头：Following threshold-response abstractions in social dynamics [Granovetter1978, Watts2002], individuals update arousal by...
或者移到这一句结束：...yields moderate arousal (M). This mechanism aligns with classic threshold models [Granovetter1978, Watts2002].
推荐前者：开宗明义，告诉读者“我不是瞎编的，我是有传承的”，然后再讲细节。
关于 we next specify...：
这就很棒，完美的承上启下。✅
小结：这段内容无需大改，只需调整引用的位置，让“机制描述”一气呵成。
总结与行动建议
你需要做的是：
确认模型名：如果 Qwen3-8B 没问题，就不动；有问题请回复修正。
调整引用位置：把 [Granovetter1978, Watts2002] 提到本句或本段开头，或者单独成句。

1. 关于 "mainstream-removal ratio" (r) 的直觉性
Review 意见：
问题：当前的 $r$ 定义稍微有点绕。你把它定义为“主流媒体移除比例”，但公式里写的是 $(1-r)$（主流）和 $r$（自媒体）。
$r=0$ → $1\cdot n_m p^{\text{main}}$（全是主流），符合“移除率为 0”。
$r=1$ → $1\cdot n_w p^{\text{we}}$（全是自媒体），符合“移除率为 1”。
判断：直觉是一致的，且符合我们“调节 $r$ 看看主流媒体消失会发生什么”的叙事目标。
建议：保留这个定义，但可以在最后一句强化一下解释：...removes mainstream and leaves only We-media feedback. 改为 ...removes mainstream influence, leaving the system driven solely by We-media.
2. 关于 $p^{\text{we}}(q,a)=\rho_H$ 的解释
Review 意见：
问题：直接扔出 $p^{\text{we}}=\rho_H=\frac{a+q}{2}$ 确实有点突兀，读者可能会问“为什么要耦合 $a$？”。
解释需求：这个公式的物理意义是——自媒体的风险信号直接等于当前群体中 High Arousal 的比例（因为只有 H 会发声/传播风险）。
建议：在公式前加一句直白的机制解释。
改写：In simulations, we consider a realistic asymmetric variant where We-media simply amplifies the voices of the highly aroused subpopulation: $p^{\text{we}}(q,a)=\rho_H=\frac{a+q}{2}$.
这样就把“耦合 $a$”这一数学操作变成了“自媒体只听那个大嗓门（H）”的直观社会学解释。
3. 关于符号密度和 $(n_m, n_w)$
Review 意见：
问题：确实有点密。而且 $n_m$ 和 $n_w$ 容易被误解为具体的人数，其实它们更像是“权重系数”或“媒体生态基数”。
建议：简化对 $n_m, n_w$ 的描述，把它变得更“人话”。
改写：...where $n_m$ and $n_w$ represent the baseline supply strengths of mainstream and We-media sources, respectively.
这样读者就知道这是两个常数，代表媒体生态的初始配置。
总结与行动建议
你需要做的是：
强化 $p^{\text{we}}$ 解释：加上 amplifies the voices of the highly aroused subpopulation。
简化 $n_m/n_w$ 描述：用 baseline supply strengths 这种词。
微调结尾：让 $r$ 的物理意义更清晰。

---

## Note07：H2 采样密度去噪（rarefaction）检查

背景：H2（$r_{\text{proxy}}$ vs $\mathrm{std}(Q)$）在 batch3 上有显著相关，但控制段内密度（`n_windows_aq`）后显著性下降，怀疑小样本噪声与缺口导致伪相关。

### 方案（物理去噪 / Rarefaction）

在每个 `4H` 时间窗内，对 **public** 帖子无放回抽样固定样本量 `N` 来计算 $(Q,a)$：

- 若窗口 public 帖子数 `< max(min_posts_public, N)`：该窗口 `a/Q` 置为 NaN（相当于丢弃）。
- 否则：从 public 中抽样 `N` 条，计算 $X_H,X_M,X_L$，进而 $Q=X_H-X_L,\ a=X_H+X_L$。

实现位置：

- `src/empirical/time_series.py`：`TimeSeriesConfig.rarefy_public_n / rarefy_seed`，`aggregate_time_series()` 新增列 `n_public_used`。
- `scripts/run_note7_empirical.py`：新增 CLI `--rarefy-public-n / --rarefy-seed` 并传递到 `TimeSeriesConfig`。

推荐对照方式（避免“窗口集合不一致”的混淆）：

1) baseline：`--min-posts-public N` 且 `--rarefy-public-n 0`（用全量 public 计算 Q）
2) rarefaction：同样 `--min-posts-public N` 且 `--rarefy-public-n N`（用固定 N 计算 Q）

### batch3 smoke（事实记录）

在 `batch3`、`freq=4h`、`segment=W` 下：

- baseline（`min_posts_public=5`，无 rarefaction）：H2 Pearson r=0.265 (p=0.00434)；H2(partial ctrl n_windows_aq) r=0.078 (p=0.411)
- baseline（`min_posts_public=10`，无 rarefaction）：H2 Pearson r=0.159 (p=0.238)；partial r=0.056 (p=0.678)
- rarefaction（`rarefy_public_n=10`，等价要求每窗>=10）：H2 Pearson r=-0.152 (p=0.259)；partial r=-0.082 (p=0.546)
- rarefaction（`rarefy_public_n=50`）：有效窗口骤降（`valid windows` 仅 126），段数不足导致 H1/H2 无法评估（nan）。

初步结论（仅针对“去噪是否能让 H2 更稳健”）：

- 当提高窗口内最小 public 数量或强制 rarefaction 后，H2 的显著性无法维持，且会因为缺口增多导致统计单元骤减；这支持“原始 H2 受采样密度/缺口噪声影响较大”的解释。

### 方案A：放宽时间粒度（4H→12H/24H）

动机：增大每窗帖子数以降低 small-\(N\) 噪声，检验 H2 是否能在控制密度后仍然成立。

（batch3，`segment=W` for 12H；24H 为满足段内最小窗口数要求，使用 `segment=14D`）

- `freq=12H`：H2 Pearson r=0.429 (p=6.03e-07)；H2(partial ctrl n_windows_aq) r=0.386 (p=8.79e-06)
- `freq=24H`：H2 Pearson r=0.386 (p=0.0015)；H2(partial ctrl n_windows_aq) r=0.455 (p=1.43e-04)

结论：在更粗的时间粒度下，H2 在 batch3 上对“段内密度控制”表现为稳健（partial 仍显著），说明 4H 下的显著性损失更可能来自 small-\(N\)/缺口噪声而非真实机制缺失。
