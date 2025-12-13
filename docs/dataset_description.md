# 数据使用说明文档 (Dataset Documentation)

## 1. 数据集概述
本项目使用 **Weibo Long-COVID 话题数据** 来验证集体情绪的混合反馈模型。
数据集包含从 2020年到 2024年（重点集中在 2020-2022）的微博内容、发布时间、用户信息及情绪/风险标注。

- **总样本量**: 17,604 条（去重后）
- **覆盖率**: 100% 已标注
- **核心文件**: `outputs/annotations/master/long_covid_annotations_master.jsonl`

## 2. 目录结构详解

数据目录位于 `project_root/outputs/annotations/`，结构如下：

```
outputs/annotations/
├── master/                  <-- 【主数据区】分析请认准这里
│   └── long_covid_annotations_master.jsonl  (与 merged_topic_official.csv 完全对齐，17,604 条，含 mid)
│
├── batches/                 <-- 【原始批次区】历史归档，只读
│   ├── batch_01_filtered_rules/  (第一轮: 规则标注, 无mid, ~6k)
│   └── batch_02_official_llm/    (第二轮: LLM标注, 含全量, ~17k)
│
├── derived/                 <-- 【分析产物区】可随时重跑脚本生成
│   ├── time_series_1h.csv       (1小时粒度聚合)
│   └── time_series_10m.csv      (10分钟粒度聚合)
│
├── intermediate/            <-- 【中间文件区】
│   └── to_annotate_batch2.csv   (批次2产生的待标注清单)
│
└── legacy/                  <-- 【遗留文件区】历史/中间文件归档，只读
    ├── long_covid_annotations_master_full_23545.jsonl  (清理前 master 备份，含无 mid 记录)
    └── long_covid_annotations_no_mid_5941.jsonl        (仅无 mid 的遗留记录)
```

## 3. 关键文件说明

### 3.1 主标注文件 (`master/*.jsonl`)
**格式**: JSONL (每行一个 JSON 对象)
**字段说明**:
- `mid`: 微博唯一ID (字符串)
- `text`: 微博正文（清洗后，建议用于文本分析/对齐回退）
- `original_text`: 微博原文（用于追溯）
- `emotion_class`: 情绪分类 (`H`: 高唤醒, `M`: 中性, `L`: 低唤醒)
- `risk_class`: 风险分类 (`risk`: 风险信息, `norisk`: 非风险)
- `reasoning`: LLM 标注理由（部分批次包含）

**说明**：
- `publish_time` / `user_name` / `verify_typ` 等用户与时间信息来自原始合并数据 `dataset/Topic_data/merged_topic_official.csv`，不在 master 标注文件中重复存储。

### 3.2 衍生时间序列 (`derived/*.csv`)
用于假设检验和绘图的聚合数据。
**关键字段**:
- `time_window`: 时间窗口起始点
- `n_posts`: 该窗口内帖子总数
- `n_public`: 公众帖子数
- `X_H`, `X_M`, `X_L`: 三种情绪类别的占比 (0~1)
- `a`: Activity (活跃度/非中立度) = $1 - X_M = X_H + X_L$
- `Q`: Order Parameter (极化度) = $X_H - X_L$
- `n_mainstream`: 主流媒体(蓝V)发帖数
- `n_wemedia`: 自媒体(黄V)发帖数
- `p_risk_mainstream`: 主流媒体的风险报道比例
- `p_risk_wemedia`: 自媒体的风险报道比例
- `n_government`: 政府/机构账号发帖数

可进一步派生（默认不落盘）：
- `r_proxy` = $n_{wemedia} / (n_{mainstream} + n_{wemedia})$
- `volatility`: $Q$ 的滚动波动率等

## 4. 数据处理流水线
本数据由 `dataset/Topic_data` 下的原始 CSV 经过以下步骤生成：

1.  **扁平化与合并**:
    - `flatten_official_media.py`: 将嵌套 JSON 转为 CSV。
    - `merge_datasets.py`: 结合 `#新冠后遗症#_filtered.csv` 和官媒补充数据，生成 `merged_topic_official.csv` (17,604 条)。

2.  **筛选与标注**:
    - `extract_new_samples.py`: 筛选出去重后的待标注样本。
    - `run_new_annotation.py`: 调用 Qwen/GPT 进行 LLM 标注。

3.  **合并与规范化**:
    - `merge_new_annotations.py`: 将新批次合并入 `master` 库。

4.  **聚合**:
    - `run_phase5_preprocessing.py`: 映射用户类型，按时间窗口聚合生成 CSV。

## 5. 常见问题
- **Q: 为什么 merged CSV 有 3万行？**
  - A: 因为微博内容包含大量换行符，导致 `wc -l` 统计虚高。实际有效数据是 17,604 条。
- **Q: 之前的 6k 条规则标注去哪了？**
  - A: 被新的 LLM 标注覆盖了。因为新批次是对全量数据进行的重新标注，质量更高且标准统一。
  - 旧数据仍在 `batches/batch_01_filtered_rules` 以备查验。
