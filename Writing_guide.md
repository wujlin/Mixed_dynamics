这份文档总结了我们在打磨顶刊（如 Nature Communications 级别）手稿时提炼出的核心写作法则和常见避坑指南。请在后续的写作和修改中严格参照执行，确保逻辑严密、行文流畅。

# 顶刊学术论文写作避坑指南与经验总结

## 1. 结构性呼应 (Structural Cohesion)

**核心原则：绝不写“孤立”的段落，前后文必须强力咬合。**

* **Intro 与 Method 必须双向映射：** * 在 Introduction 的最后一段（Roadmap），必须精准预告 Method 中的核心模块（如：提到了物理验证，Method 里就必须有对应的 PDE 模型）。
* 在 Method 的开头，必须主动“向后看”，使用 `As discussed in the Introduction` 等连接词，明确当前步骤是为了回应前文提出的哪个科学假设。


* **消灭“说明书式”罗列 (Laundry List)：**
* 不要用干瘪的 First, Second, Third 堆砌步骤。
* **正确做法：** 用科学逻辑驱动过渡。例如：“为了量化这个假设（衔接上文），我们提取了参数（当前动作）。因为仅凭统计相关性不够（局限性），所以我们引入了物理模型（引出下文）。”



## 2. 图文绝对一致 (Visual-Textual Alignment)

**核心原则：读者脑海中的图像必须与正文结构 1:1 对应。**

* **模块对齐：** 如果框架图（Framework Figure）画了 4 个模块（如 Data Pipeline, Parameter Extraction, Validation, Cross-scale），正文的 Overview 引导段落以及接下来的 Subsection 标题，**必须**严格使用这 4 个模块的名称。绝对不要图里画了 4 个框，正文却写了 5 个 Step。
* **正文与图注 (Caption) 的严格分工：**
* **正文讲“道”：** 解释为什么要这么做，物理动机是什么。
* **图注讲“术”：** 只描述图里画了什么动作（例如：把瓦片数据聚合成曲线）。
* **避坑：** 绝不能在图注里重复正文的高级定性结论，避免严重的文字冗余。


* **图表自解释性 (Self-explanatory)：** 坐标轴标签、关键参数（如 $\alpha$, $\dnear$）、对比实验条件（如 No diffusion, Shuffle）必须直接标在图上，不要让读者去图注里“寻宝”。

## 3. 自上而下的叙事节奏 (Top-Down Approach)

**核心原则：先给目的和物理图像，再给细节和公式。**

* **目的前置 (The Hook)：** 在每个 Subsection 的第一句，直接宣告本节的最终目的（如：“本数据管道的首要目的是为了提取宏观轨迹 $D(t)$”）。不要一上来就倒苦水讲底层数据长什么样。
* **物理直觉优先于数学：** * 在引出抽象符号前，必须先翻译成“大白话”物理图像。
* **示例：** 在介绍 $\dnear$ 之前，先解释它在几何上代表的是“向外逃散（陡梯度）”还是“向内聚拢（缓梯度）”，然后再抛出公式。


* **长句截断：** 如果一句话里塞进了参数定义、同位语和从句，果断将其物理截断（Physical Splitting）。使用“三段式”短句：总领句子 $\rightarrow$ 参数 A 是什么 $\rightarrow$ 参数 B 是什么。

## 4. 语言精准度与“排雷” (Precision & Avoiding Red Flags)

**核心原则：消除一切可能引起审稿人警惕的主观模糊表达。**

* **警惕主观词汇 (Cherry-picking 嫌疑)：**
* **避坑：** 绝不能说我们提取指标是为了 `isolating a cohort of 18 events`（这听起来像是在挑有利于自己的数据）。
* **正确做法：** 描述为 `establish objective, data-driven criteria for event selection`（建立客观的数据驱动标准），或者干脆不提筛选，只强调质量控制 (Quality Control)。


* **坦诚交代数据折损：** 如果样本量从 18 变成了 16，必须明确交代是因为什么客观条件（如近场数据点不足）导致的，绝不能含糊其辞。
* **避免逻辑主语错位 (Agency Error)：**
* 客观现象不能作为主观动作的发出者。例如：`spatial non-stationarity`（空间非平稳性）本身不会 `fail to reconstruct`（无法重建）。是“传统方法”因为忽略了这个属性而 failed。


* **慎用复合连字符：** 一句流畅的陈述中，尽量避免连续出现 `shape--rate connection` 和 `diffusion--relaxation model`。适时将其转化为介词结构（如 `model of diffusive relaxation`）以增强句子的数学感和呼吸感。

---

**执行建议：** Partner 在撰写或修改新章节（如 Discussion 或新的 Results）时，请在提交前对照此清单进行 Self-check，确保没有出现逻辑断层或冗余。