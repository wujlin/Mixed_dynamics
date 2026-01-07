# Source Data（Nature Communications）

Nature Communications 通常鼓励/要求为主要图表提供 **Source Data**（可复现图中曲线/点/统计量的表格数据），以便编辑与审稿人核查，也方便后续复现。

本目录用于集中放置这些源数据文件（建议与主文图号一一对应）。

## 推荐的文件命名

- `SourceData_Fig1.xlsx`
- `SourceData_Fig2.xlsx`
- `SourceData_Fig3.xlsx`
- `SourceData_Fig4.xlsx`
- `SourceData_Fig5.xlsx`
- 若某图数据较多，可拆分：
  - `SourceData_Fig4a.csv`
  - `SourceData_Fig4b.csv`

## 每个文件建议包含

- **README sheet / 首行注释**：说明数据来源、生成脚本路径（仓库内相对路径）、关键参数（如窗口大小、seed 数、bootstrap 次数）。
- **列名清晰**：例如 `r`, `Q_mean`, `Q_ci_low`, `Q_ci_high`，并注明单位/含义。
- **与图一致**：主文图中展示的每条曲线/点都能在表中定位。

## 提交时的最小集

优先保证主文 Figures 的 Source Data（补充图可选，或放到 SI/仓库）。

如果你希望我来生成这些 Source Data，需要你指定：
- 哪些图必须提供（NC 通常看主文图即可）
- 当前数据文件/中间结果存放位置（例如 `outputs/` / `data/` / `notebooks/` 产物）

