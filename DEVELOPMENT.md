Project Development Guide: Mixed-Feedback Model of Collective Emotion

1. 项目概述 (Project Overview)

本项目旨在实现一个基于统计物理的混合反馈模型（Mixed-Feedback Model），用于研究集体情绪中的相变、极化及临界慢化现象。项目分为理论计算（Mean-Field Theory）、SDE 数值模拟与网络主体模拟（Agent-Based Simulation）三部分。

核心目标：

理论求解：基于 Ginzburg-Landau (GL) 方程，解析计算相变临界点 $r_c$ 及稳态势能。
数值模拟：利用 Euler-Maruyama 方法求解随机微分方程 (SDE)，验证概率分布与分岔。
网络仿真：在复杂网络（BA/ER/全连接）上运行主体模型，验证拓扑结构下的理论预测与稀疏性/Hub 效应。

2. 技术栈与目录结构 (Tech Stack & Structure)

Tech Stack: Python 3.9+
Libraries: numpy, scipy, matplotlib, networkx, pandas, seaborn

Directory Structure:

project_root/
├── src/
│   ├── __init__.py
│   ├── theory.py       # 理论核心：rc计算, alpha映射, 势能函数
│   ├── sde_solver.py   # SDE 数值解法 (Algorithm 1)
│   ├── network_sim.py  # 网络 ABM 模拟 (Algorithm 2)
│   └── utils.py        # 绘图与辅助工具
├── notebooks/          # 探索性分析与论文绘图
│   ├── 01_Theory_and_Potential.ipynb         # 分岔（KDE密度）+ 势能，输出 figs/fig1、fig2
│   ├── 02_Network_Topology.ipynb             # 网络密度/拓扑对比，输出 figs/fig3*，数据缓存 outputs/data/*.csv
│   ├── 03_Critical_Slowing_Down.ipynb        # 临界慢化验证
│   └── 04_Sensitivity_Chi_Landscape.ipynb    # 阈值敏感度/chi 景观与分岔
├── tests/              # 单元测试
└── DEVELOPMENT.md      # 本文档


3. 开发计划 (Development Plan)

Phase 1: 理论核心模块 (Theoretical Core)

目标：实现 Method Part 2 中的所有解析公式，特别是临界点 $r_c$ 和 GL 参数 $\alpha, u$ 的计算。

任务清单 (Tasks):

[ ] 实现微观响应函数的导数计算 calculate_chi(phi, theta, k_avg)。

关键点：使用 scipy.stats.binom 计算边界概率密度。

[ ] 实现临界点计算 calculate_rc(n_m, n_w, chi)。

公式：$r_c = \frac{n_m (\chi + 2)}{n_m (\chi + 2) + n_w (\chi - 2)}$。

[ ] 实现宏观系数映射 get_gl_params(r, rc)。

逻辑：$\alpha = r_c - r$, $u = 1.0$ (或其他拟合常数)。

[ ] 实现有效势能函数 potential_energy(q, alpha, u)。

公式：$V(q) = \frac{1}{2}\alpha q^2 + \frac{1}{4}u q^4$。

验收标准 (Review Criteria):

当 $\theta, \phi$ 对称且 $\chi > 2$ 时，函数应返回 $r_c < 1$。

当 $\alpha > 0$ 时，势能曲线应为单底（U型）；当 $\alpha < 0$ 时，应为双底（W型）。

Phase 2: SDE 数值求解器 (Stochastic Dynamics)

目标：利用 Langevin 动力学验证理论分布，并展示相变过程。

任务清单 (Tasks):

[ ] 实现 Euler-Maruyama 迭代步。

方程：$q_{t+1} = q_t + (-\alpha q_t - u q_t^3)dt + \sigma \sqrt{dt} \xi_t$。

[ ] 开发 SDE 模拟主循环 run_sde_simulation(...)。

支持并行跑多条轨迹 (Ensemble)。

[ ] 实现稳态分布解析解 theoretical_pdf(q, alpha, u, sigma)。

公式：$P(q) \propto \exp(-V(q)/D)$。

验收标准 (Review Criteria):

分布验证：运行 SDE 得到的 $q$ 值直方图，必须能完美覆盖在解析解 $P(q)$ 的曲线上。

分岔验证：扫描 $r \in [0, 1]$，绘制 $q_{steady}$ vs $r$，应观测到 Pitchfork 分岔。

Phase 3: 网络主体模拟 (Network Simulation)

目标：实现 Method Part 3，引入网络拓扑和局部交互，验证理论的鲁棒性。

任务清单 (Tasks):

[x] 网络初始化：集成 networkx 生成 BA 或 ER 网络，支持全连接 (k=N-1)。

[x] 实现局部感知逻辑 (Local Perception)。

输入：全局 $r$, 邻居状态, 个人阈值 $(\phi_i, \theta_i)$。

输出：局部风险 $p_i$。

[x] 实现状态更新逻辑 (Stochastic Decision)。

基于二项分布规则或阈值规则更新 $S_i \in \{H, M, L\}$；当前实现：对感知概率做二项抽样（N=50）后再与阈值比较。

[x] 实现宏观统计器。

每个时间步计算 $Q(t)$ 和 $A(t)$。

[x] 封装模拟器类 NetworkAgentModel；ER 取最大连通子图并重标节点。

验收标准 (Review Criteria):

自洽性检查：在 $\beta=0$ (无邻居耦合) 且参数对称时，网络模拟的相变点应与 Phase 1 计算的 $r_c$ 高度吻合。

不对称性检查：当设置不对称阈值时，模拟应能自发演化出偏离 0.5 的 $p^*$，这是理论推导无法直接给出的。

拓扑效应检查：BA/ER/全连接对比，k 稀疏时相变提前，全连接逼近理论 rc；数据/图缓存于 outputs/data, outputs/figs。

Phase 4: 临界慢化与高级分析 (Critical Phenomena)

目标：计算自相关函数，验证 Critical Slowing Down (CSD)。

任务清单 (Tasks):

[ ] 实现自相关函数计算 calculate_autocorrelation(time_series, lag)。

[ ] 数据管线：在 $r$ 逼近 $r_c$ 的过程中，记录 $q(t)$ 序列。

[ ] 拟合弛豫时间：$\tau \propto 1/|r - r_c|$。

验收标准 (Review Criteria):

随着 $r \to r_c$，自相关系数（Lag-1 Autocorrelation）应显著上升并趋近于 1。

Phase 5: 经验数据验证 (Empirical Validation)

目标：用 Weibo Long-COVID 话题数据验证理论预测的核心机制。

---

### 核心理论洞察 (Core Theoretical Insight)

```
┌─────────────────────────────────────────────────────────────┐
│  现象：为什么集体情绪有时渐变、有时突变？                      │
├─────────────────────────────────────────────────────────────┤
│  洞察：中立者（温和派）是系统稳定的关键                        │
│                                                             │
│  • 中立者存在（a < 1）→ 正反馈被稀释 → 渐变                   │
│  • 中立者消失（a → 1）→ 正负反馈完全对峙 → 相变/突变           │
│                                                             │
│  理论机制：                                                   │
│  - 对称模式下：p_we = 0.5 + q/2（正负反馈对称）               │
│  - 非对称模式下：p_we = (a + q)/2（正反馈被 a 稀释）          │
│  - 当 a < 1 时，相变条件被破坏，系统呈渐变                    │
├─────────────────────────────────────────────────────────────┤
│  核心结论：                                                   │
│  情绪相变的本质是"温和派的消亡"——当中立者减少到临界点以下，    │
│  正负反馈失去缓冲，系统就会从稳定态突变为极化态。               │
└─────────────────────────────────────────────────────────────┘
```

---

### 模型参数与经验代理的映射 (Parameter-Proxy Mapping)

| 模型参数 | 物理含义 | 经验代理变量 | 计算方法 |
|----------|----------|--------------|----------|
| **r** | 主流媒体移除比例 | **r_proxy** | n_wemedia / (n_mainstream + n_wemedia) |
| **a** | Activity (1 - X_M) | **a** | (n_H + n_L) / n_total（公众情绪） |
| **φ, θ** | 心理阈值 | 无法直接观测 | **a 是其综合效应的体现** |
| **Q** | 极化方向 | **Q** | (n_H - n_L) / n_total |

**关键说明**：
- φ 和 θ 是个体内部状态，无法直接观测
- 但 φ - θ 小（敏感性高）→ X_M 小 → a 大
- 因此 **a 的效应验证 = φ, θ 效应的间接验证**

---

### 可验证的核心假设 (Testable Hypotheses)

| 假设 | 理论预测 | 经验检验 | 预期结果 |
|------|----------|----------|----------|
| **H1** | a 高 → 突变 | corr(a, jump_score) | r > 0.3, p < 0.05 |
| **H2** | r_proxy 高 → 波动大 | corr(r_proxy, volatility) | r > 0, p < 0.05 |
| **H3** | r × a 交互效应 | 分组比较：高r高a vs 低r低a | 高r高a 波动显著更大 |
| **H4** | 突变前有 CSD | AC1↑, Var↑ 趋势 | 峰值前 6-12 窗口可检测 |

**核心验证逻辑**：
```
高 r_proxy（自媒体主导）+ 高 a（中立者少）→ 最脆弱 → 最可能突变
```

---

### 验证策略 (Validation Strategy)

**策略1：时间分段验证**
- 将时间序列分成若干段
- 每段计算 r_proxy 和 a
- 检验它们与该段波动性/突变指标的关系

**策略2：跨话题比较**
- 不同话题的媒体生态不同（r_proxy 不同）
- 比较高 a 话题 vs 低 a 话题的情绪动态

**策略3：关键事件/自然实验**
- 政策调整（如2022年12月）前后对比
- 辟谣事件前后的 r_proxy 变化
- 观察这些外生冲击对情绪动态的影响

---

### 数据概况 (Data Overview)

- 话题: #新冠后遗症# 等 Long-COVID 相关话题
- 数据量: ~7,500 条/话题
- 时间范围: 2020-02 ~ 2024-02
- 认证分布: 蓝V(媒体) ~16%, 黄V(大V) ~6%, 红V ~7%, 无认证 ~70%
- 时间精度: 分钟级

---

### Phase 5.1: 文本分析流水线 (Text Classification Pipeline)

目标：建立高质量的情绪/风险分类器，替代粗糙的词典方法。

技术方案: LLM辅助标注 + 监督学习

```
┌─────────────────────────────────────────────────────────┐
│  Step 1: LLM 标注 (500-1000 样本)                        │
│  - 使用 GPT-4 / Claude API 对样本进行分类                 │
│  - 输出: emotion_class (H/M/L), risk_class (risk/norisk) │
├─────────────────────────────────────────────────────────┤
│  Step 2: 人工校验 (20% 抽样)                              │
│  - 计算 LLM-Human Cohen's Kappa                          │
│  - 要求: Kappa > 0.7                                     │
├─────────────────────────────────────────────────────────┤
│  Step 3: 训练轻量分类器                                   │
│  - 模型: DistilBERT / BERT-base-chinese                  │
│  - 训练集: LLM 标注 + 人工校验样本                         │
│  - 验证: 5-fold CV, Accuracy > 85%                       │
├─────────────────────────────────────────────────────────┤
│  Step 4: 全量推理                                        │
│  - 对所有帖子进行情绪/风险分类                             │
│  - 输出: 带标签的完整数据集                               │
└─────────────────────────────────────────────────────────┘
```

任务清单 (Tasks):

[ ] 5.1.1 设计 LLM Prompt 模板
    - 情绪分类: 区分 高唤醒(愤怒/恐惧/讽刺) / 中性(理性/平和) / 低唤醒(焦虑/困惑/无奈)
    - 风险分类: 区分 风险信息(强调后遗症严重性) / 无风险信息(淡化/安抚)
    - 输出: JSON 格式，含置信度

[ ] 5.1.2 实现 LLM 标注脚本 `src/empirical/llm_annotator.py`
    - 支持 OpenAI / Anthropic API
    - 批量处理，断点续传
    - 成本控制（token 统计）

[ ] 5.1.3 实现人工校验工具 `src/empirical/annotation_tool.py`
    - 简单的命令行或 Streamlit UI
    - 记录 Human label，计算一致性

[ ] 5.1.4 训练分类器 `src/empirical/classifier.py`
    - 基于 HuggingFace Transformers
    - 支持 DistilBERT / BERT-base-chinese
    - 输出: 训练好的模型 checkpoint

[ ] 5.1.5 全量推理脚本 `src/empirical/batch_inference.py`

验收标准:
- LLM-Human Kappa > 0.7
- 分类器 5-fold CV Accuracy > 85%
- 输出数据格式规范，含所有必要字段

---

### Phase 5.2: 用户类型映射与数据预处理 (Data Preprocessing)

目标：将 Weibo 认证类型映射到模型概念，生成时间序列特征。

用户类型映射:
| Weibo verify_typ | 模型角色 | 说明 |
|------------------|----------|------|
| 蓝V (媒体类) | mainstream | 主流媒体，负反馈来源 |
| 蓝V (企业/学校) | exclude/other | 非信息源，排除或标记 |
| 黄V | wemedia | 自媒体/大V，正反馈来源 |
| 红V + 无认证 | public | 公众，情绪被影响者 |

任务清单 (Tasks):

[ ] 5.2.1 实现用户类型识别逻辑 `src/empirical/user_mapper.py`
    - 蓝V 需进一步判断是否为"媒体类"（可通过用户名关键词或手动名单）

[ ] 5.2.2 实现时间序列聚合 `src/empirical/time_series.py`
    - 时间窗口: 1小时 / 4小时 / 1天（可配置）
    - 输出特征:
      * X_H, X_M, X_L: 公众情绪分布
      * a = X_H + X_L = 1 - X_M (核心变量!)
      * Q = X_H - X_L
      * p_risk_mainstream, p_risk_wemedia: 媒体风险报道比例
      * n_posts, engagement_sum

[ ] 5.2.3 数据清洗与质量控制
    - 去重、去空、去过短文本
    - 时间异常检测

验收标准:
- 时间序列连续，无缺失窗口（或标记缺失）
- 用户类型覆盖率 > 95%

---

### Phase 5.3: 核心假设验证 (Hypothesis Testing)

目标：检验四个核心假设（H1-H4）。

任务清单 (Tasks):

[ ] 5.3.1 计算突变/渐变指标
    - dP/dt 峰值: 极化指数变化率的最大值
    - Changepoint 检测: 使用 ruptures 库检测断点
    - jump_score: 综合突变得分（dP/dt + changepoints）

[ ] 5.3.2 计算 r_proxy（控制参数代理）
    ```python
    r_proxy = n_wemedia / (n_mainstream + n_wemedia)
    ```
    - r_proxy 高 → 自媒体主导（正反馈占优）
    - r_proxy 低 → 主流媒体主导（负反馈占优）

[ ] 5.3.3 H1 验证：a 与突变的关系
    ```
    corr(a_mean, jump_score)
    ```
    - 预期: r > 0.3, p < 0.05

[ ] 5.3.4 H2 验证：r_proxy 与波动性的关系
    ```
    corr(r_proxy, volatility)
    ```
    - 预期: 正相关

[ ] 5.3.5 H3 验证：r × a 交互效应
    ```python
    # 分组比较
    high_r_high_a = data[(r_proxy > median) & (a > median)]
    low_r_low_a = data[(r_proxy <= median) & (a <= median)]
    t_test(high_r_high_a.volatility, low_r_low_a.volatility)
    ```
    - 预期: 高r高a组波动显著更大

[ ] 5.3.6 H4 验证：临界慢化信号检测
    - 滚动窗口计算 AC1 (Lag-1 自相关)
    - 滚动窗口计算 Variance
    - 检验突变前是否有 AC1↑, Var↑ 趋势

[ ] 5.3.7 综合回归分析
    ```
    jump_score ~ r_proxy * a + controls
    ```
    - 预期: 交互项 r_proxy:a 显著为正

验收标准:
- H1: a 与 jump_score 相关系数 r > 0.3，p < 0.05
- H2: r_proxy 与 volatility 正相关
- H3: 高r高a组 vs 低r低a组差异显著 (p < 0.05)
- H4: 临界慢化信号在突变前 6-12 窗口可检测
- 结果可视化清晰，支持论文图表

---

### Phase 5.4: 结果可视化与论文图表 (Visualization)

目标：生成论文级别的验证图表。

任务清单 (Tasks):

[ ] 5.4.1 时间序列可视化
    - 多面板图: 上-情绪分布(X_H, X_M, X_L)，中-Activity(a)，下-媒体风险(p_risk)
    - 标注关键事件/突变点

[ ] 5.4.2 突变特征图
    - dP/dt 时间序列 + 峰值标注
    - Changepoint 位置标注

[ ] 5.4.3 临界慢化信号图
    - AC1, Variance 的滚动曲线
    - 与突变点的时间对应关系

[ ] 5.4.4 散点图/回归图
    - a vs 突变指标 的散点图
    - 回归线 + 置信区间

验收标准:
- 图表风格统一，论文可用
- 支持导出 PNG (300dpi) 和 PDF

---

### Phase 5 目录结构 (Directory Structure)

```
project_root/
├── src/
│   ├── empirical/                    # Phase 5 新增
│   │   ├── __init__.py
│   │   ├── llm_annotator.py          # 5.1.2 LLM 标注
│   │   ├── annotation_tool.py        # 5.1.3 人工校验
│   │   ├── classifier.py             # 5.1.4 分类器训练
│   │   ├── batch_inference.py        # 5.1.5 全量推理
│   │   ├── user_mapper.py            # 5.2.1 用户类型映射
│   │   ├── time_series.py            # 5.2.2 时间序列聚合
│   │   └── hypothesis_test.py        # 5.3 假设检验
│   └── ...
├── dataset/
│   ├── Lexicon/                      # 词典（备用）
│   └── Topic_data/                   # 原始/合并数据
│       ├── #新冠后遗症#_filtered.csv
│       ├── 官媒补充_flat.csv
│       ├── merged_topic_official.csv
│       └── 新增官媒数据/
├── notebooks/
│   ├── 05_Annotation_Pipeline.ipynb    # LLM 标注流水线
│   ├── 06_Active_Period_Analysis.py   # 活跃期/时间序列经验分析
│   └── 99_Paper_Figures.ipynb         # 论文最终出图
└── outputs/
    ├── annotations/                  # 标注与聚合产物（主分析请认准这里）
    │   ├── master/
    │   │   └── long_covid_annotations_master.jsonl
    │   ├── derived/
    │   │   ├── time_series_1h.csv
    │   │   └── time_series_10m.csv
    │   ├── batches/
    │   ├── intermediate/
    │   └── legacy/
    └── figs/
        └── empirical/
            └── active_period_p2_dist.png
```

---

### Phase 5 开发顺序建议 (Recommended Order)

```
Week 1: 文本分析流水线
├── Day 1-2: 设计 Prompt，测试 LLM 标注效果
├── Day 3-4: 批量 LLM 标注 (500-1000 样本)
├── Day 5: 人工校验 (100-200 样本)
└── Day 6-7: 训练分类器，全量推理

Week 2: 数据处理与假设检验
├── Day 1-2: 用户类型映射，时间序列聚合
├── Day 3-4: 计算 r_proxy, a, jump_score
├── Day 5-6: H1-H4 假设检验
└── Day 7: 可视化，图表生成
```

---

### Phase 5 预期论文叙事 (Expected Paper Narrative)

**理论部分（已完成）**：
> 我们建立了一个混合反馈模型，发现**对称条件**（正负反馈完全对峙）是产生二阶相变的充分条件。
> 在非对称情况下（中立者存在，a < 1），正反馈被"稀释"，系统呈现渐变而非相变。

**经验验证部分（Phase 5）**：
> 我们将理论参数映射到可观测的经验代理：
> - 控制参数 r → **r_proxy**（自媒体占比）
> - 心理敏感性 φ, θ → **a**（中立者缺失度，其综合效应的体现）
>
> 使用 Weibo Long-COVID 数据验证，结果表明：
> 1. **a 越高，情绪变化越陡峭**（H1，Jump Score=0.6）
> 2. **r_proxy 与波动性正相关**（H2，r=0.32, p=0.04）
> 3. **交互效应未显现**（H3，统计不显著）
> 4. **临界慢化未显现**（H4，AC1 无上升趋势）
>
> 这些结果在经验层面支持了核心洞察的关键一段：**中立者缺失度（a）越高，系统越可能出现突变式变化**；同时也提示真实平台数据中，交互效应与临界慢化可能被外生冲击与平台机制所掩盖，需要更精细的识别策略。

**政策启示**：
> 保护信息生态系统中的"温和派"（中立信息/理性讨论）是维护社会情绪稳定的关键。
> 当中立者占比下降、自媒体主导上升时，系统脆弱性增加，应提前预警。

4. 编码规范 (Coding Standards)

参数解耦：所有物理参数（$n_m, n_w, \theta, \phi$）必须在 config 字典或类属性中定义，严禁在计算逻辑中写死硬编码（Hard-coding）。

向量化计算：在 SDE 和理论计算中，尽量使用 NumPy 的向量化操作，避免 for 循环。

随机数种子：所有模拟函数必须接受 seed 参数，保证结果可复现 (Reproducibility)。

文档字符串：关键函数（特别是 calculate_chi、step、_update_states）应包含 Physics Docstring，说明对应论文中的公式/假设；网络更新已引入二项采样以模拟有限信息采样。

5. 参数对应关系 (Parameter Correspondence)

**理论 vs 模拟的关键参数对应**：

| 理论参数 | 模拟参数 | 说明 |
|----------|----------|------|
| `k_avg` in `calculate_chi()` | `sample_n` in `NetworkConfig` | 信息采样次数，决定 χ 和 rc |
| `n_m`, `n_w` | `n_m`, `n_w` | 媒体基数，直接对应 |
| `phi`, `theta` | `phi`, `theta` | 阈值参数，直接对应 |

**验证理论 rc 的正确配置**：
```python
# 理论计算
k_avg = 50
chi = calculate_chi(phi, theta, k_avg)
rc = calculate_rc(n_m, n_w, chi)

# ABM 配置（必须一致）
cfg = NetworkConfig(
    sample_mode="fixed",
    sample_n=k_avg,  # 与理论一致！
    symmetric_mode=True,  # 对称模式
    beta=0.0,  # 无邻居耦合
    ...
)
```

6. 对称 / 非对称模式 (Symmetric vs Asymmetric)

理论假设：均场推导 (Eq.8/Eq.10) 依赖 $q\\to -q$ 对称与 $p_{env}(q=0)=0.5$。若自媒体项直接用 $p_{we}=(a+q)/2$，在 $q=0$ 时会生成外场偏置，产生"渐变"而非经典分岔。

工程开关：`NetworkConfig.symmetric_mode`。
- `True`：理想对称，强制 $p_{we}=0.5+q/2$，用于验证 rc/GL（Pitchfork 分岔清晰）。
- `False`：现实非对称，$p_{we}=(a+q)/2$，用于刻画实际媒体/活跃度耦合下的提前漂移与渐变。

要求：在实验、输出文件名、图注中明确标注 sym/asym，避免混淆；相图与结论需区分"对称验证"与"现实机制"两条线。

5. 快速启动 (Quick Start for Agent)

Agent 指令：
"请首先阅读 Phase 1 的任务。在 src/theory.py 中实现 calculate_chi 和 calculate_rc 函数。请注意，计算 chi 时需要使用二项分布在边界处的概率质量近似。完成后，请编写一个简单的测试脚本，验证当 $\chi$ 很大时，$r_c$ 是否小于 1。"


### 如何使用这个文档？

1.  **复制**：将上面的内容保存为 `DEVELOPMENT.md` 放在你的项目根目录。
2.  **喂给 Agent**：当你开始写代码时，直接把这个文件发给你的 AI 助手（Cursor, Claude, ChatGPT 等）。
3.  **按阶段执行**：
    * 你可以说：“基于 DEVELOPMENT.md 的 Phase 1，请帮我生成 `src/theory.py` 的代码。”
    * 写完后说：“请根据 Phase 1 的验收标准，帮我写一个测试脚本来验证 $r_c$ 的推导是否正确。”

这样，Agent 就不会乱写，而是严格遵循你的物理逻辑进行工程实现。
