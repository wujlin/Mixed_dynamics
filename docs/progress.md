## 开发进度与对照（Phase 5 收敛）

### 1. 总体状态
- **Phase 1-4 (理论/模拟)**: ✅ 已完成
- **Phase 5 (经验验证)**: ✅ 已收敛（按 PI Two‑Tier 叙事停止新增迭代）
- **写作推进**: 🔄 进行中（Essay 主稿逐段打磨）
 
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
  - ✅ 聚合与检验口径已统一收敛：以 `scripts/run_note7_empirical.py` 生成的 `time_series_*_4h.csv` 与 `note07_cluster_stats_*_4h.csv` 为准（避免 notebook 口径漂移）。

- **5.3/5.4 假设检验 (验证结果)**:
  - ✅ 已按 PI 决策收敛：主结果聚焦 H1（All=master+batch3）与 H2（batch3）；H4 作为“受数据连续性/功效限制的不定论”写入附录。
  - 权威收敛稿：`Essay/note07_empirical_closure.md`

- **5.5 精细化分析**:
  - ✅ 尝试了 10分钟窗口 (`time_series_10m.csv`)。
  - 备注：更细粒度（1H/2H）在真实数据中会遇到连续块不足（eligible=0），已作为“可观测边界”的负结果写入附录，不再继续迭代。

### 3. 下一步建议
进入论文写作阶段（不再新增经验迭代）：
- 结果素材总索引：`docs/results_materials_catalog.md`
- 理论/模拟图文报告：`docs/theory_validation_report_note01-04.md`
- 经验收敛稿（可直接引用）：`Essay/note07_empirical_closure.md`
 
