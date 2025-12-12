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
│   └── long_covid_annotations_master.jsonl  (最新全量标注结果)
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
└── legacy/                  <-- 【遗留文件区】旧版残留，勿用
```

## 3. 关键文件说明

### 3.1 主标注文件 (`master/*.jsonl`)
**格式**: JSONL (每行一个 JSON 对象)
**字段说明**:
- `mid`: 微博唯一ID (字符串)
- `content`: 微博正文 (清洗后)
- `original_text`: 微博原文
- `publish_time`: 发布时间
- `user_name`: 用户昵称
- `verify_typ`: 认证类型 (e.g., "蓝V认证", "没有认证")
- `emotion_class`: 情绪分类 (`H`: 高唤醒, `M`: 中性, `L`: 低唤醒)
- `risk_class`: 风险分类 (`risk`: 风险信息, `norisk`: 非风险)
- `reasoning`: LLM 标注理由 (仅批次2包含)

### 3.2 衍生时间序列 (`derived/*.csv`)
用于假设检验和绘图的聚合数据。
**关键字段**:
- `time_window`: 时间窗口起始点
- `n_posts`: 该窗口内帖子总数
- `X_H`, `X_M`, `X_L`: 三种情绪类别的占比 (0~1)
- `a`: Activity (活跃度/非中立度) = $1 - X_M = X_H + X_L$
- `Q`: Order Parameter (极化度) = $X_H - X_L$
- `n_mainstream`: 主流媒体(蓝V)发帖数
- `n_wemedia`: 自媒体(黄V)发帖数
- `r_proxy`: 媒体控制代理变量 = $n_{wemedia} / (n_{mainstream} + n_{wemedia})$
- `p_risk_mainstream`: 主流媒体的风险报道比例
- `volatility`: (后续计算) $Q$ 的波动率

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
