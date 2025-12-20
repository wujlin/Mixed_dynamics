主题：Re: Note07 经验验证 —— 按 Two-Tier 建议复跑后的结果与下一步决策点

PI 您好，

我们已按您邮件里建议的 **Two-Tier Validation** 把经验验证拆成两条主线，并在统一口径下复跑了关键实验。下面汇总最新结果（含一处重要“不可评估”结论），并请您确认下一步该如何收敛写作与实验资源。

---

## A) H2/H3（结构性生态效应）：用更大的混合池换取 r_proxy 方差

### A1. master + batch3（含 all）按周分段（freq=4H, segment=W）

我们在 `freq=4H`、按周分段（`segment=W`）下对 `master`、`batch3` 及其合并 `all` 进行了段内统计检验（H1/H2 同时输出，H4 仅做可评估性诊断）。

关键结果（段数足够）：

- `batch3`：**H2 显著为正**  
  - Pearson：r=0.265，p=0.0043  
  - Spearman：r=0.244，p=0.0090  
  - 但控制段内样本量 `n_windows_aq` 后的部分相关不显著：r=0.078，p=0.411  
  解释：H2 的“显著相关”可能部分被段内密度混杂；我们建议把“密度控制前/后”都写进稳健性与局限性。

- `all`：H2 不显著（Pearson r=0.065，p=0.484），但 Spearman 有边缘趋势（r=0.161，p=0.078）。  
  同时 `all` 上 **H1 为正且显著**（Pearson r=0.241，p=0.008）。

- `master`：由于数据稀疏，严格 H4 口径下连续块太短，H4 不可评估（eligible=0）。这更像功效/连续性不足，而非否定 H4。

复现命令：

```bash
/home/wujlin/miniconda3/envs/emotion/bin/python scripts/run_note7_empirical.py \
  --datasets master,batch3 \
  --freq 4H \
  --min-posts-public 5 \
  --time-start 2019-01-01 \
  --segment W \
  --event-on-eligible both \
  --roll-win 12 --pre 24 \
  --no-plots
```

---

## B) H1/H4（动力学信号）：坚持单词条，但尝试提频

我们对 `batch1`（#新冠后遗症# 单词条）尝试了 `freq=2H/1H`：

- `freq=2H`：可找到团簇，但 **严格 block-aware 的 H4 在该频率下不可评估**（eligible=0，events=0）。  
  原因：提频会引入更多缺口，连续块长度不足以支撑 `roll_win≈48h` + `pre≈96h` 的窗口要求。

- `freq=1H`：在默认团簇参数（roll_days=14, q=0.9, min_days=10）下难以形成满足最短长度的团簇（clusters=0）。  
  为避免“调参”，我们改用 4H 团簇窗口作为 time_start/time_end 边界，在该窗口内跑 1H；H1 不显著，H4 仍不可评估（连续块不足）。

这意味着：在当前数据稠密度下，“把 freq 提高到 1H/2H”并不能增强 H4，反而让 H4 更难评估；H4 若要做得严谨，仍更适合保留在 4H 口径，或换更连续、更高密度的数据源再提频。

---

## 我们想请您确认的决策点（用于收敛写作）

1) H2 的经验主证据是否可以以 `batch3` 为主（因为 H2 在 batch3 上显著）？  
   - `all` 混合后 H2 变弱，可能是“异质性/混杂”导致；我们可以把 `all` 作为外部复现或稳健性对照，而不强求它显著。

2) H4 的经验验证是否可以在正文中降级（exploratory / limitation）？  
   - 在 master 稀疏、batch1 提频失败的情况下，当前最稳健的说法是“严格口径下功效不足/连续性不足，难以从现有经验数据中稳定捕捉 CSD 预警信号”。

3) 若您仍希望把 H4 做成主结果：  
   - 我们需要明确投入方向：换更连续的经验数据源（更高密度、更少缺口），或接受更弱的连续性假设（例如允许小缺口并做插补/敏感性分析）。请您指示优先级。

谢谢！我们可以在您确认后，把经验验证部分按“主结果（H2/H1）+ 局限性（H4）”的结构快速收敛成论文可用段落。

