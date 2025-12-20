# Emotion Dynamics（Mixed Dynamics）

基于统计物理的混合反馈模型，用于研究集体情绪的 **相变/极化**、**临界慢化（CSD）** 与 **脆弱性相图**。项目覆盖：解析理论（Mean-field/GL）、SDE 数值模拟、网络 ABM，以及基于微博数据的经验验证（H1–H4）。

## 效果图（部分）

<p align="center">
  <img src="outputs/figs/fig1/fig1_bifurcation.png" width="48%" />
  <img src="outputs/figs/fig4/fig4a_chi_rc_landscape.png" width="48%" />
</p>
<p align="center">
  <img src="outputs/figs/fig4/fig4e_beta_rc_maxslope.png" width="48%" />
  <img src="outputs/figs/empirical/fig7c_h4_eventstudy_all_4h.png" width="48%" />
</p>

## 快速开始

### 1) 环境

- Python 3.9+
- 安装依赖：`pip install -r requirements.txt`

### 2) 理论与仿真（Note01–Note04）

- Notebooks：`notebooks/01_Theory_and_Potential.ipynb`、`notebooks/02_Network_Topology.ipynb`、`notebooks/03_Critical_Slowing_Down.ipynb`、`notebooks/04_Sensitivity_Chi_Landscape.ipynb`
- 阶段性报告（图文版）：`docs/theory_validation_report_note01-04.md`

### 3) 经验验证（Note07）

- Notebook：`notebooks/07_Empirical_Validation.ipynb`
- 命令行复现入口：`scripts/run_note7_empirical.py`
- 阶段性报告：`docs/note07_empirical_validation_report.md`

## 经验代理口径（与代码一致）

- 极化方向：$Q=X_H-X_L$
- 活跃度：$a=X_H+X_L=1-X_M$
- 控制参数代理（媒体生态）：
  $$
  r_{\\text{proxy}}=\\frac{n_{\\text{wemedia}}}{n_{\\text{wemedia}}+n_{\\text{mainstream}}+n_{\\text{government}}}
  $$
  其中 `government` 视为“官方叙事”，并入主流分母。

## 文档索引

- 代码与数据结构：`docs/code_data_structure.md`
- 数据集说明：`docs/dataset_description.md`
- 理论验证报告（Note01–04）：`docs/theory_validation_report_note01-04.md`
- 经验验证报告（Note07）：`docs/note07_empirical_validation_report.md`
- 工作站 Qwen/vLLM 记录：`docs/vllm_qwen_setup.md`

## 数据与合规说明

- 大体量原始数据与标注数据默认不入库（见 `.gitignore`）；论文复现所需的关键派生产物与脚本会在仓库中维护。
- 微博爬虫需要本地 cookie（放在 `secrets/`，不入库）。详见 `scripts/fetch_user_meta_weibo.py`；若历史 `user_meta` 口径有误，可用 `scripts/fix_user_meta_csv.py` 离线修正。
