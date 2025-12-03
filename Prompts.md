# Codex CLI Development Prompts

## 项目背景

这是一个情绪动力学研究项目，使用统计物理方法研究集体情绪的相变现象。项目已完成理论模拟部分（Phase 1-4），现在需要进行经验数据验证（Phase 5）。

**核心理论洞察**：中立者（温和派）的消失是情绪相变的驱动力。当 Activity (a = 1 - X_M) 升高，系统更容易突变。

---

## Prompt 1: 启动经验验证流程

```
请阅读 DEVELOPMENT.md 中的 Phase 5 部分，了解经验数据验证的完整计划。

项目当前状态：
- src/empirical/ 目录已创建，包含以下模块的框架代码：
  - llm_annotator.py: LLM 辅助标注器
  - user_mapper.py: 用户类型映射
  - time_series.py: 时间序列聚合
  - hypothesis_test.py: 假设检验
  - classifier.py: 情绪分类器

- 数据位于 dataset/Topic_data/#新冠后遗症#.csv
- 数据字段包括: mid, user_name, verify_typ, publish_time, content, forward_num, comment_num, like_num 等

请按以下顺序执行 Phase 5.1 的任务：

1. 首先读取数据，了解数据结构和分布
2. 实现数据加载和预处理函数
3. 测试 LLM 标注器（如果有 API key）或设计替代方案
4. 每完成一步，报告进度和结果
```

---

## Prompt 2: 数据预处理与用户映射

```
基于 DEVELOPMENT.md Phase 5.2，执行数据预处理：

任务：
1. 使用 src/empirical/user_mapper.py 将 verify_typ 映射到模型角色：
   - 蓝V媒体 → mainstream
   - 黄V → wemedia  
   - 红V/无认证 → public

2. 读取 dataset/Topic_data/ 中的数据，添加 user_type 列

3. 使用 src/empirical/time_series.py 聚合时间序列：
   - 时间窗口: 1小时
   - 计算: X_H, X_M, X_L, a, Q, r_proxy
   
4. 保存处理后的数据到 dataset/processed/

5. 输出基础统计信息：
   - 用户类型分布
   - 时间范围
   - a 和 r_proxy 的分布

注意：如果情绪分类（H/M/L）尚未完成，先跳过情绪相关计算，只做用户类型映射和媒体统计。
```

---

## Prompt 3: 假设检验

```
基于 DEVELOPMENT.md Phase 5.3，执行假设检验：

前提：需要先完成情绪分类，生成 emotion_class 列

核心假设验证：
- H1: corr(a, jump_score) > 0.3, p < 0.05
- H2: corr(r_proxy, volatility) > 0
- H3: 高r高a组 vs 低r低a组波动差异显著
- H4: 突变前检测到 AC1↑, Var↑

使用 src/empirical/hypothesis_test.py 中的函数：
1. run_full_hypothesis_test(df_ts) - 完整检验
2. test_r_a_interaction(df_ts) - 交互效应检验

输出：
1. 每个假设的检验结果（统计量、p值、是否支持）
2. 关键可视化图表
3. 结果解读和下一步建议
```

---

## Prompt 4: 创建分析 Notebook

```
创建 notebooks/05_Empirical_Validation.ipynb，包含以下 cells：

Cell 1: 导入和配置
- 导入 src/empirical 模块
- 设置路径和参数

Cell 2: 数据加载
- 读取原始数据
- 用户类型映射
- 基础统计

Cell 3: 时间序列聚合
- 调用 aggregate_time_series()
- 计算 r_proxy
- 可视化时间序列

Cell 4: 假设检验
- H1-H4 依次检验
- 输出统计结果

Cell 5: 可视化
- 时间序列多面板图
- a vs jump_score 散点图
- 临界慢化信号图

Cell 6: 结果总结
- 哪些假设得到支持
- 结果的理论意义
```

---

## 关键文件路径

```
项目根目录: emotion_dynamics/
├── DEVELOPMENT.md          # 开发计划（必读）
├── src/empirical/          # 经验验证模块
│   ├── llm_annotator.py
│   ├── user_mapper.py
│   ├── time_series.py
│   ├── hypothesis_test.py
│   └── classifier.py
├── dataset/
│   ├── Topic_data/#新冠后遗症#.csv  # 原始数据
│   ├── annotations/        # 标注数据
│   └── processed/          # 处理后数据
└── notebooks/              # 分析笔记本
```

---

## 核心公式提醒

```python
# Activity (中立者缺失度)
a = X_H + X_L = 1 - X_M

# r_proxy (自媒体主导程度)
r_proxy = n_wemedia / (n_mainstream + n_wemedia)

# 理论预测
# 高 r_proxy + 高 a → 最脆弱 → 最可能突变
```

---

## 注意事项

1. 数据文件名包含 # 符号，读取时需要注意路径处理
2. 情绪分类是关键步骤，如无 LLM API，可考虑：
   - 使用词典方法（dataset/Lexicon/）作为临时方案
   - 或先完成用户类型映射和媒体统计
3. 所有输出图表保存到 outputs/figs/
4. 中间数据保存到 dataset/processed/

