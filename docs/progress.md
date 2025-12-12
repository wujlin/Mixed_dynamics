## 开发进度与对照（Phase 5.2 相关）

### 当前状态概览
- LLM 标注：`annotated_intent_rule_v3.jsonl` 已生成（风险/情绪规则版，代理问题已解决，连接测试通过）。
- 预处理：`run_phase5_preprocessing.py` 已运行，生成 `outputs/annotations/v3/time_series_1h.csv`（覆盖率 98.85%，用户类型覆盖率 100%）。
- 活跃期分析：`notebooks/06_Active_Period_Analysis.py` 可运行，含分布/回归；1h/4h 聚合完成，结果已输出。
- 数据补充：官媒 JSON 已整理脚本；待标注样本提取、合并标注、再预处理的流水线脚本已齐全（未执行完最后两步，需开启 LLM 后继续）。

### 对照 DEVELOPMENT.md（Phase 5.2）
- 5.2.1 用户类型识别：`user_mapper.py` 已实现，自动加载官媒清单，支持媒体/政府关键词与自定义名单。
- 5.2.2 时间序列聚合：`time_series.py` 已输出 X_H/X_M/X_L、a、Q、p_risk_mainstream/p_risk_wemedia、n_posts 等；`run_phase5_preprocessing.py` 可生成 1h/4h/1d。
- 5.2.3 数据清洗与质量：预处理脚本支持去空、时间解析；缺失窗口以 NaN 标记；若需更少缺失可调整 min_posts/freq。
验收标准：时间序列生成 OK；用户类型覆盖率已达 100%；连续性可通过降低阈值或增大窗口进一步改善。

### 待办与建议
1) 扁平官媒数据 + 合并数据（已提供脚本，需执行）  
   ```
   python scripts/flatten_official_media.py --input-dir dataset/Topic_data/新增官媒数据 --output dataset/Topic_data/官媒补充_flat.csv
   python scripts/merge_datasets.py --base "dataset/Topic_data/#新冠后遗症#_filtered.csv" --official dataset/Topic_data/官媒补充_flat.csv --output dataset/Topic_data/merged_topic_official.csv
   ```
2) 提取未标注样本、跑 LLM 标注（需开启服务），合并标注  
   ```
   python scripts/extract_new_samples.py --merged dataset/Topic_data/merged_topic_official.csv --annotated outputs/annotations/v3/annotated_intent_rule_v3.jsonl --output outputs/annotations/v3/to_annotate.csv
   # 开启 LLM 后：
   python scripts/run_new_annotation.py --input outputs/annotations/v3/to_annotate.csv --output outputs/annotations/v3/new_official_ann.jsonl --base-url http://10.13.12.164:7890/v1 --api-key abc123 --model Qwen/Qwen3-8B
   python scripts/merge_new_annotations.py --base outputs/annotations/v3/annotated_intent_rule_v3.jsonl --new outputs/annotations/v3/new_official_ann.jsonl --output outputs/annotations/v3/annotated_intent_rule_v3_plus.jsonl
   ```
3) 重跑预处理/分析（可试 1h/4h/1d 或降低 min_posts）  
   ```
   python scripts/run_phase5_preprocessing.py --dataset dataset/Topic_data/merged_topic_official.csv --annotations outputs/annotations/v3/annotated_intent_rule_v3_plus.jsonl --freq 1h --output outputs/annotations/v3/time_series_1h_plus.csv
   ```
   然后运行 `python notebooks/06_Active_Period_Analysis.py`，查看 r_proxy 分布、回归结果是否改善。

### 代理与连接提醒
- 脚本 `test_llm_connection.py`、`run_new_annotation.py` 已内置代理设置并清除 no_proxy，默认 `socks5://127.0.0.1:1080`，可用环境变量覆盖。
- 服务端需监听 0.0.0.0:7890 且放行端口；如连通失败，先用 `test_llm_connection.py` 验证。 
