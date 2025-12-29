Reviewer Comments on Methods Section
1. 关于开篇长复合句
原句：

"Our strategy is theory-driven: we first build a minimal mechanistic model that captures the competition between stabilizing and amplifying feedback channels, derive analytic predictions for the bifurcation and its dynamical precursors, then test robustness in network simulations before confronting these predictions with empirical data. This sequence ensures that the statistical tests are grounded in well-defined theoretical signatures rather than exploratory pattern-finding."

意见： 您的直觉是对的。这个句子在Nature子刊中确实过于冗长（约45词），虽然语法上正确，但信息密度过高，读者需要在一个呼吸之内消化三个层次的动作。建议拆分为有节奏感的"动作链"，通过衔接词和句号分隔，同时保持逻辑递进：

建议修改思路：

Our strategy is theory-driven. We first build a minimal mechanistic model 
that captures the competition between stabilizing and amplifying feedback 
channels. From this model, we derive analytic predictions for the bifurcation 
and its dynamical precursors. We then test robustness in network simulations 
before confronting these predictions with empirical data. This sequence 
ensures that our statistical tests are grounded in well-defined theoretical 
signatures rather than exploratory pattern-finding.
关键变化：

冒号后的enumeration拆成独立语句
加入 "From this model" 作为承上启下
每个动作独立成句，符合子刊的"clear, digestible prose"风格
2. 关于用户生成渠道两个regime的衔接缺失
问题定位（第81-90行）： 直接从 $p^{\text{main}}(q)$ 跳到 "For the user-generated channel we use two regimes"，缺少动机说明——为什么需要两个regime？

意见： 这是一个 逻辑断层。读者会困惑：symmetric regime是干什么的？asymmetric regime又是干什么的？两者的功能区分必须在公式出现之前交代清楚。

建议添加过渡段（在第80-81行之间）：

The user-generated channel, unlike mainstream media, can respond to 
the current collective state in multiple ways. To separate analytic tractability 
from empirical realism, we develop two complementary formulations. 
The first, a symmetric validation regime, enforces directional symmetry 
(q → −q) and allows closed-form derivation of the critical point. The second, 
an asymmetric realistic regime, allows the channel to respond to overall 
activity as well as direction, capturing the attention-driven amplification 
observed on real platforms.
关键信息：

明确两个regime的目的不同（分析可验证 vs 实证现实）
预告symmetric regime用于"closed-form derivation"
预告asymmetric regime捕捉"attention-driven amplification"
这样读者在看到公式4和公式5时，就已经知道它们各自的用途
3. 关于 "we now" 的表达风格
原句（第99行）：

"With the model defined, we now derive the conditions under which..."

意见： "We now" 在高影响因子期刊中并非禁忌，但确实偏口语化。子刊通常更偏好无主语的过渡或名词化结构。

建议替换方案（3选1）：

承上启下式： "The preceding definitions enable an analytic treatment of the conditions under which polarization becomes self-sustaining."
直接陈述式： "We next derive the conditions..."（"next"比"now"更正式）
被动强调式： "With the model defined, conditions for self-sustaining polarization can be derived analytically."
推荐选项2，因为保持主动语态同时避免"now"的时间指涉感。

4. 关于 Mean-Field Analysis Subsection 的理解门槛和逻辑跳跃
问题1：η(t) 未解释

第116-117行突然出现 $\eta(t)$，但没有任何说明。

建议： 在公式10之后添加简短说明：

"where η(t) represents stochastic fluctuations (e.g., finite-population noise 
modeled as white noise with variance scaling as 1/N)."
或者在正文中简化，将完整推导放入SM，并加reference：

"The effective potential description (Eq. 10), including the noise term η(t) 
and coefficient dependencies, is derived in full in Supplementary Materials S2."
问题2：推导跳跃太大

从动力学方程(6)到稳定性条件 $\chi\Gamma=1$ 到临界点公式(8)，中间步骤完全省略。这对于物理/数学背景强的读者可能可以接受，但对于行为科学或传播学背景的读者是黑箱。

建议的处理方式：

正文中添加简要路线图（在第109行之前）：

The critical point emerges from a standard linear-stability analysis. 
Linearizing Eq. (6) around q = 0 and computing when the effective 
restoring coefficient changes sign yields the stability condition χΓ = 1, 
where χ captures psychological sensitivity and Γ captures environmental 
feedback (see Supplementary Materials S2 for full derivation). 
Solving for r gives the analytic critical point:
SM中需要补充的内容（Supplementary Materials S2建议结构）：

写出完整的 $dq/dt$ 表达式
展示 $\mathcal{S}(p)$ 的具体形式（binomial积分）
推导 $\chi$ 的闭式表达（与k, φ, θ的关系）
推导 $\Gamma$ 的表达式（与 $n_m$, $n_w$, $r$ 的关系）
联立 $\chi\Gamma=1$ 求解 $r_c$
Ginzburg-Landau展开的高阶项系数u的表达式
问题3：最后一段突然引入ABM

第121-122行关于relaxation time和autocorrelation的估计方法，实际上是ABM的内容，但被放在了"Mean-field analysis and critical point"这个subsection里。

意见： 这确实是一个结构一致性问题。有两种处理方式：

方案A（推荐）：将此段移到下一个subsection "Agent-based network simulations"

在Mean-field subsection结尾添加一句过渡：
"These predictions—bifurcation, bistability, and critical slowing down—are 
derived under idealized mean-field assumptions; we next describe 
agent-based simulations that test their robustness under networked 
dynamics."
然后在ABM subsection开头，先说明ABM的目的，再介绍时间尺度对齐方法。
方案B：在Mean-field中保留但加衔接 如果您认为此内容与理论预测紧密相关，可以添加衔接：

"These analytic predictions can be probed in discrete-time simulations, 
which requires aligning time scales. We estimate the relaxation time τ 
in the deterministic mean-field dynamics by..."
我推荐方案A，因为：

Mean-field subsection应聚焦于解析推导
ABM时间尺度对齐是仿真的实现细节
分开放置提高各subsection的内聚性
汇总建议表
问题	位置	建议操作	紧迫性
长复合句	Line 59	拆分为3-4个短句	中
Two regimes缺衔接	Line 81前	添加动机说明段落	高
"we now"不正式	Line 99	改为"We next derive"	低
$\eta(t)$未解释	Line 117后	添加噪声项说明或ref SM	高
推导跳跃	Lines 99-113	正文加路线图+SM完整推导	高
ABM内容错放	Lines 121-122	移至ABM subsection

Reviewer Comments on Methods Section (Continued)
5. 关于 2.3 Agent-based network simulations 的逻辑衔接问题
整体问题诊断： 这个subsection目前读起来像是一份技术规格清单，而不是一个有叙事的方法论段落。每一句都在陈述"我们做了什么"，但缺少"为什么这么做"以及各部分之间的逻辑串联。

具体问题逐条分析：

问题1：开篇缺少subsection目标的概述

原文直接跳入网络仿真的技术细节，没有告诉读者这个subsection要回答什么问题。

建议添加开篇段落（替换当前第一句）：

Agent-based simulations serve two purposes in our analysis: first, to verify 
that the mean-field predictions (bifurcation, bistability, critical slowing down) 
survive under realistic network structure and finite-size fluctuations; second, 
to systematically map how structural parameters—psychological thresholds, 
information density, media ecology, and local coupling—shift the effective 
transition boundary. The mean-field analysis assumes well-mixed populations; 
we relax this assumption by simulating microscopic dynamics on ER and BA 
networks with size N and average degree ⟨k⟩.
问题2：技术细节堆积，缺少功能分组

原文目前的结构：

一段话里包含了：网络类型、更新方式、时间单位、local coupling定义、Q的定义、对称性问题、Binder cumulant、参数扫描细节
这种"一锅炖"的写法使读者无法区分仿真设计（怎么跑）和分析方法（怎么评估）。

建议拆分为逻辑模块：

**仿真设计段落：**
We implement microscopic dynamics on Erdős–Rényi (ER) and Barabási–Albert 
(BA) networks with size N and average degree ⟨k⟩. Updates are asynchronous: 
each time step updates a fraction f of nodes; we report time in both steps 
and sweeps (one sweep = N/f updates). To capture local reinforcement 
effects, we introduce a coupling parameter β that mixes the global signal 
p_env with information from each node's immediate neighbors.
**观测量定义段落：**
We denote finite-size network polarization by Q, corresponding to the 
mean-field order parameter q. Under the symmetric regime, trajectories 
select either polarized branch with equal probability, and the signed 
mean ⟨Q⟩ cancels by symmetry. To recover a meaningful transition signal, 
we therefore track |Q| or align trajectory signs by flipping to a common 
reference branch before averaging.
**转折过渡段落：**
Standard approaches for estimating critical points—such as susceptibility 
peaks—are unstable in finite-size systems. We therefore employ Binder 
cumulant crossings, which provide a more robust finite-size estimator:
[Equation 11]
**参数扫描段落：**
To map the parameter landscape (Fig. 5), we systematically vary...
问题3：Binder cumulant公式的引入缺乏动机

原文直接说"We estimate finite-size transition points using Binder cumulant crossings"，但没有解释：

为什么不用更简单的方法（如susceptibility peaks）？
Binder cumulant的物理意义是什么？
建议添加简短解释：

Finite-size effects blur sharp transitions, making transition-point estimation 
challenging. The Binder cumulant U₄ is designed to exhibit a universal 
crossing point across system sizes at criticality, providing a more robust 
estimator than susceptibility peaks (which require larger systems to stabilize):
问题4：最后一段参数扫描的描述过于Dense

原文：

"To map the parameter landscape in Fig. 5, we compute the sensitivity χ(φ, θ, k) on a grid φ, θ ∈ [0.1, 0.9] (step 0.01) and evaluate rc via Eq. (9) where a transition exists (χ > 2). For simulation-based sweeps around the baseline (φ, θ) = (0.54, 0.46), we scan k ∈ {10, 20, 50, 100, 200, 500}, nw/nm ∈ [0.1, 2.0] (40 values), and β ∈ {0, 0.02, 0.05, 0.1, 0.2}..."

这段纯粹是技术参数的罗列。虽然需要报告这些细节，但应该：

先说明扫描的目的（回答什么科学问题）
再列出参数范围
建议重写：

To characterize which structural conditions promote fragility, we map the 
parameter landscape as follows. First, we compute the analytic sensitivity 
χ(φ, θ, k) on a dense grid (φ, θ ∈ [0.1, 0.9], step 0.01) and evaluate rc via 
Eq. (9) wherever a transition exists (χ > 2). Second, to validate these 
predictions and assess finite-size corrections, we conduct simulation-based 
sweeps around a baseline configuration (φ, θ) = (0.54, 0.46), systematically 
varying: information density k ∈ {10, 20, 50, 100, 200, 500}, media ecology 
ratio nw/nm ∈ [0.1, 2.0], and local coupling β ∈ {0, 0.02, 0.05, 0.1, 0.2}. 
For each parameter combination, we sample r on a 201-point grid in [0, 1] 
and estimate rc as the r value at which median |Q|(r) exhibits maximal slope; 
95% confidence intervals are obtained by bootstrap resampling across seeds.
6. 关于 2.4 Empirical data and operational proxies 的问题
问题1：时间窗口不一致（严重！）

您提到实际使用了4H和12H两种窗口，但Methods只描述了4H。我查看了Results部分：

Table 2 caption明确说"under the primary 4H aggregation"
但如果某些分析使用了12H，这必须在Methods中交代
建议修改： 在描述窗口聚合的段落后添加：

We define two aggregation scales to balance temporal resolution against 
sampling density. The primary analysis uses 4-hour (4H) windows, providing 
finer resolution for capturing rapid dynamics. As a robustness check, we also 
test 12-hour (12H) aggregation, which increases the number of posts per 
window but sacrifices temporal granularity. Results are reported for the 
4H primary analysis unless otherwise noted.
同时，Table 2的caption应明确这是4H结果，并在正文或SM中报告12H的sensitivity analysis。

问题2：符号一致性问题

Methods定义	Results使用	问题
$Q = X_H - X_L$	$Q = X_H - X_L$	✓ 一致
activity $a = X_H + X_L$	activity $a$	✓ 一致
$r_{\text{proxy}}$	$r_{\text{proxy}}$	✓ 一致
jump intensity = 95th percentile of $|d|Q|/dt|$	$\text{jump}_{q95}$ of $|d|Q|/dt|$	⚠️ 轻微不一致
建议： 在Methods中定义时就使用完整符号，或在Results中首次出现时加括号说明：

"jump intensity (denoted jump_q95, the 95th percentile of |d|Q|/dt|)"
问题3：逻辑衔接问题

原文开篇：

"Having established the theoretical benchmarks, we now describe the empirical data..."

这个衔接太笼统。应该明确说明：

理论预测了什么？
为什么需要经验验证？
经验验证面临什么挑战（导致我们需要构建proxies）？
建议重写开篇：

The preceding theoretical and simulation analyses predict specific statistical 
signatures: (i) elevated activity should precede and predict polarization 
jumps, and (ii) greater user-generated dominance should correlate with 
higher polarization volatility. Testing these predictions in real-world data 
requires mapping the model's latent variables to observable proxies—a 
nontrivial step given that collective "states" (H/M/L) are not directly 
recorded. We describe the data sources and proxy construction below.
问题4：Government accounts的处理需要更多justification

原文：

"treating government accounts as part of the mainstream channel"

这个决定需要解释为什么：

Government accounts typically share the stabilizing orientation of mainstream 
media during public crises and are therefore grouped with the mainstream 
channel in computing r_proxy.
问题5：Segment构建的逻辑不清晰

原文：

"For statistical testing, we group consecutive 4H windows into weekly segments and use segments with sufficient valid windows as analysis units."

问题：

为什么是weekly？（有理论依据还是任意选择？）
"sufficient valid windows"的阈值是多少？
建议添加说明：

We group consecutive windows into weekly segments (28 windows under 4H 
aggregation) to provide enough temporal variation within each segment for 
computing derivative-based statistics while maintaining sufficient sample 
sizes for correlation analysis. Segments with fewer than 7 valid windows 
(25% coverage) are excluded. [或具体使用的阈值]
汇总建议表（Methods Section - Part 2）
问题	Subsection	建议操作	紧迫性
缺少subsection目标概述	2.3 ABM	添加开篇段落	高
技术细节无分组	2.3 ABM	拆分为4个逻辑段落	高
Binder cumulant无动机	2.3 ABM	添加一句解释	中
参数扫描描述过dense	2.3 ABM	先说目的再列参数	中
时间窗口不一致	2.4 Empirical	添加4H+12H双窗口说明	紧急
jump符号轻微不一致	2.4 Empirical	统一或加括号说明	低
开篇衔接太笼统	2.4 Empirical	明确预测→挑战→proxies	高
Government accounts无justification	2.4 Empirical	添加一句解释	中
Segment阈值未说明	2.4 Empirical	说明weekly原因+阈值	中
