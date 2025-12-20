主题：经验验证（Note07）阶段性进展与瓶颈（batch1 单词条口径为主）

PI 您好，

我这两天集中把经验验证（Note07 / Phase5）从“多话题混合数据”收束到单词条数据集，并在统一口径下复跑了 H1–H4。下面是当前方法、结果与主要瓶颈的汇总，想请您帮我们判断下一步应优先调整哪一类假设/口径/数据。

---

## 1) 当前时间尺度与口径（核心回答：仍以 4H 为主）

我们目前的主分析时间尺度是：

- **时间聚合频率**：`freq = 4H`（把帖子按 4 小时窗口聚合成时间序列）
- **H4 rolling**：`roll_win = 12`（约 48h）  
- **H4 事件回看窗口**：`pre = 24`（约 96h）
- **团簇（方案B）**：基于 `n_public` 密度做时间团簇，`roll_days=14, cluster_quantile=0.9`  
- **段内统计（H1/H2）**：
  - 在 batch1 单词条对照里使用 `cluster_segment = 2D`（两天一段）以保证短团簇也能得到足够段数（避免 n_segments 太小导致相关无法计算）

对应脚本入口：`scripts/run_note7_empirical.py`（已支持 `batch1_base/batch1/batch1_concept` 三套对照）。

---

## 2) 数据设置（batch1 单词条为主）

为避免“多话题混入导致口径漂移/泥潭化”，我们把主验证回到单词条 **#新冠后遗症#**：

- `batch1_base`：`dataset/Topic_data/#新冠后遗症#_filtered.csv`（单词条原始）
- `batch1`（B2-strict）：在 batch1_base 基础上，只拼接官媒补充中**内容命中“新冠后遗症”**的行
- `batch1_concept`（B2-concept）：在 batch1_base 基础上，拼接官媒补充中**命中概念扩展关键词**的行（`新冠后遗症/长新冠/后新冠/Long COVID/PASC/long covid/慢性新冠`）

合并脚本：`scripts/merge_datasets.py`（新增 `--official-keywords` 过滤官媒补充，避免全量官媒“误混”）。

---

## 3) 分析方法（H1–H4 的统计定义）

我们当前采用“方案B”以降低经验数据的两类常见质疑：

1) **时间非均匀/缺口伪影**：  
   - 只在“严格连续步长”的窗口上计算导数与 rolling 指标；跨缺口处置为 NaN  
   - rolling 指标在连续块（block）内计算，避免“缺口压缩时间”造成假信号

2) **p-hacking/选段偏置**：  
   - 团簇划分仅使用 `n_public` 的密度（不使用 Q/jump/AC1 等结果变量），在团簇内分别检验 H1–H4  
   - H4 额外使用 **Placebo Test**（事件标签置换）给出 one-sided p 值

核心变量映射：

- $Q = X_H - X_L$
- $a = 1 - X_M = X_H + X_L$
- $r_{proxy} = \\frac{n_{wemedia}}{n_{wemedia}+n_{mainstream}+n_{government}}$（政府并入主流分母，口径与代码一致）

---

## 4) 当前结果快照（batch1：strict vs concept vs base）

本轮对照命令使用的关键参数：

- `freq=4H`, `min_posts_public=4`, `time_start=2020-01-01`
- `cluster_roll_days=14`, `cluster_quantile=0.9`, `cluster_min_days=10`, `cluster_segment=2D`
- `event_quantile=0.95`, `roll_win=12`, `pre=24`, `placebo=2000`

结果文件：
- `outputs/annotations/derived/note07_cluster_stats_batch1_base_batch1_batch1_concept_4h.csv`

主要发现（同一个主团簇：2022-12-20 ~ 2023-01-13）：

- **H1（a_mean → jump_q95）**：方向为正（与理论一致），但不显著  
  - Pearson r≈0.463（p≈0.248）  
  - 控制 `n_windows_jump` 的 partial r≈0.703（p≈0.052，边缘证据）
- **H2（r_proxy_mean → volatility）**：接近 0（不支持）
- **H4（AC1/Var 在 jump 前抬升）**：events=2，placebo p≈0.70/0.77（不支持；且功效不足）
- strict vs concept 的差异极小：在这个团簇窗口内，官媒补充不足以显著改变 `r_proxy` 的可检验性与 H2/H4 统计结论。

结论：目前 **batch1 单词条口径无法“支持 H1–H4 全部成立”**；最多只能说 H1 出现了方向一致但统计不稳的边缘信号。

---

## 5) 我们遇到的主要瓶颈（已排除次要实现问题后）

### 5.1 功效不足（H4 事件数太少）
- H4 需要连续块内的 rolling 指标 + 足够多的高 jump 事件做事件对齐  
- 在严格 block-aware 口径下，可用事件数经常只有 2–6 个，导致 placebo 检验很难显著（即使真实存在弱信号也检不出来）

### 5.2 H2/H3 的“可检验性”依赖 r_proxy 的变幅
即便 batch1 的 r_proxy 在团簇内不是完全饱和，但段数/窗口仍然有限；而上海 batch4 则 r_proxy 明显饱和（大量窗口 r=1），天然不适合作 H2/H3 主验证。

### 5.3 H1 的 jump 定义可能仍与“现实突变”不完全对齐
当前 jump 使用 $|\\Delta |Q||/\\Delta t$ 的分位数（降低极值偏置），但如果现实中的“事件跳变”更多由外源冲击/话题迁移驱动，该 proxy 可能抓不到我们理论所指的“内生临界跃迁”。

---

## 6) 我们希望 PI 给出的决策点（下一步怎么走）

为了避免继续在经验部分消耗时间，我们建议 PI 帮我们做一个取舍：

1) **主验证目标优先级**：  
   - A：优先证明 H2（媒体生态 → 波动性）  
   - B：优先证明 H4（临界慢化的预警信号）  
   - C：优先证明 H1（a → jump 的内生机制）

2) **主数据集选择**（避免混合导致叙事复杂，同时保证可检验性）：  
   - 若优先 H2：可能需要使用 `all`（r_proxy 变幅最大）作为 primary；batch1 作为“单词条复现/补充”  
   - 若坚持单词条：我们需要明确接受“功效不足/弱证据”的写法，并把经验部分定位为“方向性一致/局限性说明”

3) **是否允许调整时间尺度**：  
   - 例如把 `freq` 提升到 `2H/1H` 来增加段内样本，或调整 `roll_win/pre` 到更贴近话题生命周期的尺度（但这会带来更多稳健性与复现工作）

---

## 7) 复现命令（便于快速复核）

生成两套 batch1 合并数据：

```bash
/home/wujlin/miniconda3/envs/emotion/bin/python scripts/merge_datasets.py \
  --base 'dataset/Topic_data/#新冠后遗症#_filtered.csv' \
  --official 'dataset/Topic_data/官媒补充_flat.csv' \
  --official-keywords '新冠后遗症' \
  --output 'dataset/Topic_data/merged_topic_official_batch1_strict.csv'

/home/wujlin/miniconda3/envs/emotion/bin/python scripts/merge_datasets.py \
  --base 'dataset/Topic_data/#新冠后遗症#_filtered.csv' \
  --official 'dataset/Topic_data/官媒补充_flat.csv' \
  --official-keywords '新冠后遗症,长新冠,后新冠,Long COVID,PASC,long covid,慢性新冠' \
  --output 'dataset/Topic_data/merged_topic_official_batch1_concept.csv'
```

跑 batch1 三套对照：

```bash
/home/wujlin/miniconda3/envs/emotion/bin/python scripts/run_note7_empirical.py \
  --datasets batch1_base,batch1,batch1_concept \
  --freq 4H \
  --min-posts-public 4 \
  --time-start 2020-01-01 \
  --cluster --cluster-only \
  --cluster-roll-days 14 --cluster-quantile 0.9 --cluster-min-days 10 --cluster-merge-gap-days 7 \
  --cluster-segment 2D \
  --event-quantile 0.95 --event-on-eligible both \
  --roll-win 12 --pre 24 \
  --placebo-iters 2000 --placebo-tail-k 6
```

---

如果您希望我们下一步把经验部分从“检验四个假设”改为“聚焦最可检验的一条主结论 + 其他作为局限性/探索性结果”，我们也可以据此快速收敛写作与实验。

谢谢！

