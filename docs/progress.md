## 开发进度与对照（Phase 5 完成）

### 1. 总体状态
- **Phase 1-4 (理论/模拟)**: ✅ 已完成
- **Phase 5 (经验验证)**: 🔄 已扩展（新增 Batch3 标注；需在 Note07 复核 H1-H4）
- **文档更新**: 🔄 进行中（对齐 master vs batch3 口径）
 
### 2. Phase 5 详细成果
对应 `DEVELOPMENT.md` 的 Phase 5 任务：

- **5.1 文本分析**: 
  - ✅ LLM 标注流水线已跑通 (`src/empirical/llm_annotator.py`)。
  - ✅ 核心对照集：完成 17,604 条全量标注 (覆盖率 100%)。
    - 产出: `outputs/annotations/master/long_covid_annotations_master.jsonl`。
  - ✅ 扩展集（Batch3）：完成 73,456 条标注（与核心对照集 mid 不重叠）。
    - 产出: `outputs/annotations/batches/batch_03_expanded/new_batch3.jsonl`。
  - ✅ master 已清理为“可对齐主数据”：仅保留含 `mid` 的 17,604 条；清理前备份与无 `mid` 遗留已归档到 `outputs/annotations/legacy/`。

- **5.2 数据预处理**:
  - ✅ 用户类型映射 (`user_mapper.py`) 覆盖率 100%。
  - ✅ 数据重构完成：建立了 `master`/`batches`/`derived` 清晰目录结构。
  - ✅ 核心对照集聚合产出: `outputs/annotations/derived/time_series_1h.csv`。
  - 🔄 扩展集与合并口径的聚合：由 `notebooks/07_Empirical_Validation.ipynb` 统一生成与对照（避免口径漂移）。

- **5.3/5.4 假设检验 (验证结果)**:
  - 🔄 结果需以 `notebooks/07_Empirical_Validation.ipynb` 的复核输出为准（将同时报告：核心对照集 / 扩展集 / 合并样本 三套结果）。

- **5.5 精细化分析**:
  - ✅ 尝试了 10分钟窗口 (`time_series_10m.csv`)。
  - 备注：该结论需要在新增扩展集后复核；后续将把“窗口敏感性”作为稳健性分析的一部分。

### 3. 下一步建议
优先完成经验验证口径统一与复核（Note07），再进入论文撰写阶段：
- 先给出 H1-H4 在三套口径下的一致/不一致结论与稳健性说明；
- 再客观讨论 H4（CSD）在经验数据中可能难以显现的原因（外生冲击/平台机制/观测噪声等）。
 
