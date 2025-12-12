## 代码与数据结构概览（emotion_dynamics）

### 目录结构与核心模块
- `src/`
  - `theory.py`：理论计算 χ/rc/GL 势能。
  - `sde_solver.py`：SDE 数值模拟。
  - `network_sim.py`：网络主体仿真（BA/ER，对称/非对称）。
  - `empirical/`（Phase 5）
    - `llm_annotator.py`：LLM 标注器（OpenAI/兼容端点）。
    - `user_mapper.py`：用户类型映射（蓝V媒体/黄V自媒体/红V+无认证公众/政府/其他，自动加载官媒清单）。
    - `time_series.py`：时间序列聚合与指标（X_H/X_M/X_L, a, Q, p_risk_mainstream/wemedia, r_proxy 等）。
    - `data_loader.py`：话题 CSV 加载与用户映射。
    - `hypothesis_test.py`：突变/交互效应/临界慢化工具函数。
    - `classifier.py`：情绪/风险分类器骨架（Transformers）。
    - `text_preprocessor.py`：微博文本清洗。
- `scripts/`
  - `run_phase5_preprocessing.py`：合并标注+聚合时间序列（支持 mid 或 content 对齐）。
  - 数据合并与标注辅助：`flatten_official_media.py`（扁平官媒 JSON）、`merge_datasets.py`（主数据+官媒 CSV 合并）、`extract_new_samples.py`（筛未标注样本）、`run_new_annotation.py`（批量标注）、`merge_new_annotations.py`（合并新旧标注）、`test_llm_connection.py`（连通性测试）。
  - 分析脚本：`analyze_time_series.py`、`analyze_active_periods.py`（如需扩展）、`verify_hypotheses.py`（假设验证）。
- `notebooks/`
  - `06_Active_Period_Analysis.py`：活跃期分析（1h/4h 聚合、相关/回归/分布）。
  - 其他 notebook 覆盖理论、网络、CSD、标注流程等。
- `docs/`
  - `vllm_qwen_setup.md`：远程 vLLM 部署与代理排查指南。
  - `code_data_structure.md`：本文件（代码结构概览）。
  - `dataset_description.md`：数据集与标注说明文档（详细）。
  - `progress.md`：阶段进度清单。

### 数据与中间产物
- 原始话题数据：`dataset/Topic_data/#新冠后遗症#_filtered.csv` 等（多话题 CSV）。
- 官媒补充：`dataset/Topic_data/新增官媒数据/*.json`（需扁平化）、`dataset/Topic_data/官媒补充_flat.csv`。
- 合并数据：`dataset/Topic_data/merged_topic_official.csv`（主数据+官媒）。

### 标注与分析数据 (`outputs/annotations/`)
**新结构 (2025.12)**:
- `master/`: **主版本数据区**。唯一事实来源。
  - `long_covid_annotations_master.jsonl`: 全量标注文件。
- `batches/`: **原始批次归档**。
  - `batch_01_filtered_rules/`: 第一轮规则标注。
  - `batch_02_official_llm/`: 第二轮 LLM 标注。
- `derived/`: **衍生分析数据**。
  - `time_series_1h.csv`: 1小时聚合。
  - `time_series_10m.csv`: 10分钟聚合。
- `intermediate/`: **中间过程文件**。
  - `to_annotate_batch2.csv`: 中间产生的待标注名单。

### 配置与代理
- LLM 服务：远程 vLLM（示例 `http://10.13.12.164:7890/v1`，api_key=abc123），脚本内已内置代理设置并清除 no_proxy。
- 代理：默认 `socks5://127.0.0.1:1080`，可通过环境变量覆盖。
