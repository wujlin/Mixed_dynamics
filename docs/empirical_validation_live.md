# 经验分析实时报告（Note07 / Phase5）

> 用途：给 PI/通讯作者快速同步当前经验验证进展，并作为“随跑随更”的单一事实来源（避免口径漂移）。

更新时间：2025-12-20（手动更新）

## 1. 统一口径（本报告当前快照）

**核心变量（与理论映射）**

- 序参量：$Q = X_H - X_L$
- 活跃度：$a = 1 - X_M = X_H + X_L$
- 正反馈占优代理：
  $$
  r_{\text{proxy}}=\frac{n_{\text{wemedia}}}{n_{\text{wemedia}}+n_{\text{mainstream}}+n_{\text{government}}}
  $$

**当前快照使用的统计设置**

- 聚合频率：`freq=4H`
- 默认时间窗：建议显式传入 `--time-start/--time-end`（脚本默认不截断）
- 团簇识别（方案B）：`roll_days=14, cluster_quantile=0.9`
- 段内统计：`segment=W`（master/batch3/all），`segment=2D`（batch4 与 batch1 单词条对照；避免团簇太短导致 n_segments<5 被跳过）
- H4（事件对齐 + placebo）：`roll_win=12 (~48h), pre=24 (~96h)`；placebo 为事件标签置换（one-sided p）

数据源与缓存入口：

- 团簇统计（master/batch3/all）：`outputs/annotations/derived/note07_cluster_stats_4h.csv`
- 团簇统计（batch4）：`outputs/annotations/derived/note07_cluster_stats_batch4_4h.csv`
- 时间序列缓存：`outputs/annotations/derived/time_series_{master,batch3,all,batch4}_4h.csv`

## 2. 可检验性诊断（r_proxy 变幅）

目的：在不使用结果变量（jump/AC1/Var）的前提下，评估 H2/H3 是否“有统计可辨识性”（避免把功效不足误判成假设失败）。

指标来自 `outputs/annotations/derived/time_series_*_4h.csv`（按 `r_proxy` 非空窗口统计）：

| 数据集 | 时间跨度（缓存） | r_proxy 有效窗口 | mean | median | std | p10 | p90 | pct(r=1) | 计数(public/wemedia/mainstream/gov) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| master | 2023-01-01 ~ 2024-08-30 | 745 | 0.451 | 0.333 | 0.449 | 0.000 | 1.000 | 36.24% | 2625 / 650 / 1300 / 2 |
| batch3 | 2020-03-18 ~ 2025-07-11 | 3354 | 0.849 | 1.000 | 0.281 | 0.500 | 1.000 | 70.36% | 56341 / 8268 / 2186 / 746 |
| all | 2019-01-06 ~ 2025-07-11 | 5638 | 0.562 | 0.667 | 0.440 | 0.000 | 1.000 | 43.42% | 61328 / 10439 / 10614 / 763 |
| batch4（上海） | 2022-01-01 ~ 2022-05-25 | 421 | 0.961 | 1.000 | 0.141 | 0.889 | 1.000 | 87.41% | 30028 / 1513 / 52 / 11 |

**解读要点（当前事实）**

- `batch4（上海）` 的 `r_proxy` **高度饱和（接近 1）**，绝大多数窗口 `r_proxy=1`，因此 **H2/H3 的功效先天不足**；即使补齐更多 UID，除非官方叙事账号密度结构性上升，否则很难改变这一点。
- `all` 的 `r_proxy` 变幅最大（std≈0.44，p10=0），更适合作为 H2/H3 的“主验证口径”候选；`batch3` 更偏“高 r_proxy 场景”。

## 3. H1–H4 当前快照（团簇内，方案B）

说明：

- H1：`corr(a_mean, jump_q95)`（段内相关；jump 为段内 `|d|Q|/dt|` 的高分位）
- H2：`corr(r_proxy_mean, volatility)`（段内相关；volatility 为段内 `std(Q)`）
- H4：事件对齐（pre 窗口）+ placebo（事件置换），报告 one-sided p

### 3.1 master / batch3 / all（来自 `note07_cluster_stats_4h.csv`）

| 数据集-团簇 | 时间范围 | n_windows | n_segments | H1 Pearson r (p) | H2 Pearson r (p) | H4 events | H4 used(ac/var) | H4 placebo p(ac1/var) |
|---|---|---:|---:|---|---|---:|---|---|
| master-c0 | 2022-12-12 ~ 2023-01-26 | 266 | 6 | 0.025 (0.962) | 0.121 (0.820) | 6 | 0/0 | NA / NA（严格口径下不可评估） |
| batch3-c0 | 2023-01-21 ~ 2023-03-22 | 313 | 8 | -0.637 (0.089) | 0.064 (0.879) | 12 | 7/7 | 0.465 / 0.923 |
| batch3-c1 | 2023-05-09 ~ 2023-06-21 | 262 | 7 | -0.441 (0.322) | -0.332 (0.467) | 11 | 7/7 | 0.406 / 0.759 |
| all-c0 | 2023-01-28 ~ 2023-04-06 | 411 | 10 | -0.092 (0.801) | 0.471 (0.169) | 14 | 7/7 | 0.547 / 0.763 |
| all-c1 | 2023-05-05 ~ 2023-06-24 | 302 | 8 | -0.360 (0.382) | -0.218 (0.604) | 12 | 8/8 | 0.110 / 0.745 |

### 3.2 batch4（上海）/ cluster0（来自 `note07_cluster_stats_batch4_4h.csv`）

| 数据集-团簇 | 时间范围 | n_windows | n_segments | H1 Pearson r (p) | H2 Pearson r (p) | H4 events | placebo iters | H4 placebo p(ac1/var) |
|---|---|---:|---:|---|---|---:|---:|---|
| batch4-c0 | 2022-05-12 ~ 2022-05-25 | 79 | 5 | -0.078 (0.900) | 0.543 (0.344) | 4 | 200 | 0.095 / 0.249 |

### 3.3 batch1（单词条：#新冠后遗症#，方案B2 严格/概念扩展对照）

目的：把经验分析从“多话题混合”收束为单词条，并分别测试两套官媒补充口径：
- `batch1_base`：仅 `dataset/Topic_data/#新冠后遗症#_filtered.csv`
- `batch1`（strict）：+ 官媒补充（content 命中“新冠后遗症”）
- `batch1_concept`：+ 官媒补充（content 命中“新冠后遗症/长新冠/后新冠/Long COVID/PASC/long covid/慢性新冠”）

当前快照（`time_start=2020-01-01, min_posts_public=4, cluster_segment=2D`；来自 `note07_cluster_stats_batch1_base_batch1_batch1_concept_4h.csv`）：

| 数据集-团簇 | 时间范围 | n_windows | n_segments | H1 Pearson r (p) | H2 Pearson r (p) | H4 events | H4 placebo p(ac1/var) |
|---|---|---:|---:|---|---|---:|---|
| batch1_base-c0 | 2022-12-20 ~ 2023-01-13 | 141 | 8 | 0.463 (0.248) | -0.019 (0.964) | 2 | 0.699 / 0.783 |
| batch1-c0 | 2022-12-20 ~ 2023-01-13 | 143 | 8 | 0.463 (0.248) | 0.002 (0.997) | 2 | 0.712 / 0.769 |
| batch1_concept-c0 | 2022-12-20 ~ 2023-01-13 | 143 | 8 | 0.463 (0.248) | 0.002 (0.997) | 2 | 0.712 / 0.769 |

解读要点（当前事实）：
- batch1 的 H1 方向与理论一致（a_mean 与 jump_q95 正相关），但样本段数仍少，统计不显著；控制 `n_windows_jump` 的部分相关 `r≈0.703, p≈0.052` 属于“边缘证据”。  
- strict vs concept 的差异非常小：官媒补充在该团簇窗口内不足以显著改变 r_proxy 与 H2/H4 的可检验性（需要更多团簇/更长窗口/更高密度）。  

## 4. 已排除的次要问题（避免反复“怀疑实现”）

- `verified_type==0` 误判：已修正为“黄V/自媒体（wemedia）”，避免把个人认证大 V 当作 public。
- `user_meta` 的 error 行污染：加载时跳过 `error` 行并过滤 `"nan"`，避免覆盖 `user_type`。
- `segment=2D` 分段伪问题：已改为使用 `.dt.floor()` 全局对齐固定长度段，避免 pandas `to_period('2D')` 产生“按天滑动标号”导致段内样本不足。

## 5. 当前主要瓶颈（需要 PI 决策/指令的部分）

1) **H2/H3 的可辨识性依赖 r_proxy 的变幅**：上海 batch4 当前 `r_proxy` 近饱和（大量窗口=1），属于“高 r 场景”，不适合作为 H2/H3 的主验证数据。  
2) **H4 对连续性与事件数敏感**：严格 block-aware 口径会显著减少“可用事件”，导致 placebo p 值不稳定或不可评估（如 master-c0）。  
3) **H1 当前在多个团簇内呈负相关趋势或接近 0**：可能意味着（a→jump）机制在经验数据中不成立，或 jump 定义仍不够贴近“突变”。

## 6. 下一步（建议作为“预先声明”的分析协议）

为避免“参数挑选/p-hacking”质疑，建议先确定一个 **Primary dataset + 固定口径**，再做外部复现：

- Primary（建议二选一）  
  - 选项1：`all` 做主验证（H2/H3 优先），`batch4` 仅作 H4 案例；  
  - 选项2：`batch3` 做主验证（高 r 场景），再用 `master`/`all` 做外部复现。

确定后再做：

- 在 Primary 上做小规模稳健性栅格：`cluster_quantile ∈ {0.85,0.9,0.95}` × `roll_days ∈ {7,14,21}` × `event_quantile ∈ {0.9,0.95}`  
- 明确 H4 的“可用事件门槛”（例如 `events_used >= 8` 才允许下结论；否则标注为功效不足）。

## 7. 更新本报告的方法（最小流程）

1) 跑脚本更新缓存与统计：
```bash
/home/wujlin/miniconda3/envs/emotion/bin/python scripts/run_note7_empirical.py \
  --datasets master,batch3,all \
  --freq 4H \
  --min-posts-public 5 \
  --time-start 2023-01-01 \
  --cluster --cluster-only \
  --cluster-roll-days 14 --cluster-quantile 0.9 --cluster-min-days 21 --cluster-merge-gap-days 7 \
  --cluster-segment W \
  --event-quantile 0.95 \
  --roll-win 12 --pre 24 \
  --placebo-iters 5000 --placebo-tail-k 6
```

2) 更新可检验性表：从 `time_series_*_4h.csv` 重新统计 `r_proxy`（mean/median/std/pct==1）。

3) 在本文件顶部更新“更新时间”，并把第 2、3 节表格替换为最新快照。
