# 写作踩坑清单（避免重复踩坑）

本文件用于记录我们在本项目论文写作（理论 + 仿真 + 经验验证）中反复踩过的坑，目的是：
- 降低“可复现性不清/口径漂移/定义缺失”带来的编辑与审稿风险
- 让每一段文字都能被读者在第一次阅读时评估（定义清楚、逻辑可追、证据可核）
- **培养科学叙事思维，避免技术报告式写作**

建议用法：每写完一个段落或一张图，就对照文末的 Checklist 自检一次。

---

## 0. 高水平期刊写作的底层思维

### 0.1 问题驱动 vs 验证驱动（最核心的思维转换）

**症状**：文章读起来像"我们做了A，验证了B，又验证了C"，而不是"我们发现了X，这意味着Y"。

**诊断方法**：
- 数一数文章中"validate/confirm/test"出现了多少次
- Results的每一节是在"证明模型work"还是在"回答一个科学问题"
- 读者读完能记住的是"你的方法"还是"你的发现"

**纠正**：
| 验证驱动（❌） | 问题驱动（✓） |
|---|---|
| "We validate the prediction" | "We find that X" |
| "Network simulations confirm" | "The mechanism persists under" |
| "We test whether..." | "We show that..." / "X reveals..." |
| Results按方法排序 | Results按科学问题递进 |

### 0.2 叙事主线 vs 技术清单

**症状**：每一节都是独立的技术展示，读者读完不知道"so what"。

**诊断方法**：
- 删除所有section标题后，文章是否仍有逻辑流？
- 每节开头是否在回答"为什么读者应该关心这一节"？
- 每节结尾是否有take-home message（而不是下一节的预告）？

**叙事主线模板**：
```
问题：什么结构条件决定集体脆弱性？
↓
机制：媒体生态平衡决定稳定性边界（3.1）
↓
深化：什么因素移动这个边界？（3.2 Parameter Landscape）
↓
确认：这个机制是否普遍？（3.3 Network Robustness）
↓
延伸：能否提前检测？（3.4 CSD）
↓
验证：数据支持吗？（3.5 Empirical）
```

### 0.3 信息层次与重点突出

**症状**：核心发现被埋在技术细节中，读者找不到重点。

**原则**：
- **核心发现**：在Abstract、Intro末尾、Results关键位置明确陈述
- **支撑细节**：在正文中简述，细节移到Methods或SI
- **技术债务**：完全移到SI（代码路径、API参数、具体grid）

**自检**：读者只看Abstract、每节第一句、每节最后一句，能否理解核心贡献？

---

## 0.5 各部分的常见问题模式

### Title

**常见问题**：
- 太长（>15词）
- 使用defensive语言（"A framework for..."、"Towards..."）
- 没有点明核心发现或机制

**好的Title特征**：
- 10-15词
- 点明核心机制或发现
- Assertive语气

**示例对比**：
| ❌ 有问题 | ✓ 改进 |
|---|---|
| "A Theoretical Framework for Understanding Phase Transitions in Collective Emotion Motivated by Media Dynamics" | "Channel balance controls collective emotional resilience in mixed media ecologies" |
| "Towards Early Warning of Collective Polarization" | "Elevated activity signals approaching collective instability" |

### Abstract

**常见问题**：
- 没有明确的问题陈述
- 核心发现被压缩成从句
- 使用验证语言（"We validate..."）

**Abstract结构模板**：
```
1. 现象/问题（1-2句）
2. Gap/为什么难（1句）
3. 我们的approach（1句）
4. 核心发现1（1-2句）——最重要的
5. 核心发现2（1句）
6. 实证支持（1句）
7. Take-home/implications（1句）
```

### Introduction

**常见问题**：
- P1没有stakes（为什么读者应该关心）
- 文献综述太散，没有聚焦到一个clear gap
- 预告Results时使用"验证清单"式写法
- "To address this gap"等defensive开头

**Introduction结构模板**：
```
P1: 现象 + 挑战 + Stakes
P2: 文献1（cascade/threshold models）→ 它们的局限
P3: 文献2（echo chamber/contagion）→ 聚焦到gap
P4: 我们的approach（assertive开头："We formalize..."）
P5: 核心发现预告（用发现语言，不是验证清单）
P6: 全文roadmap（可选）
```

### Results

**常见问题**：
- 按方法顺序排列，而不是按科学问题递进
- 每节开头是技术描述，而不是问题陈述
- 每节结尾是下一节的预告（显式过渡），而不是take-home
- Robustness check和核心发现混在一起

**Results排序原则**：
1. 核心机制/核心发现放前面
2. "什么因素影响核心机制"紧随其后
3. Robustness check作为独立section（或整合到相关section末尾）
4. 实证验证放最后

**过渡句处理**：
| ❌ 显式过渡 | ✓ 隐式过渡 |
|---|---|
| "We next test whether..." | （直接开始下一节，读者自然想知道） |
| "We return to this in Section X" | （删除，或改写为take-home） |
| "Having established X, we now..." | （删除，用下一节开头的问题自然引入） |

### Discussion

**常见问题**：
- 第一段重述"我们做了什么"（重复Introduction）
- 与prior work的比较占太多篇幅
- Practical implications太薄
- Limitations是"清单"而不是"指向future directions"
- Conclusion太平淡，没有升华

**Discussion结构模板**：
```
P1: 核心发现的interpretation（不是summary）
P2: 与prior work的关键区别（压缩，只说最重要的）
P3: Theoretical significance（深化，不是重复）
P4: Practical implications（具体，可操作）
P5: Limitations → Future directions（配对，不是清单）
P6: Conclusion（升华，big picture）
```

### Methods

**常见问题**：
- 把Results的发现放在Methods里（如$r_c$公式的推导结果）
- 第一句是Results预告（"We test whether..."）
- 技术细节过多（应该在SI）

**Methods原则**：
- Methods说"怎么做"，Results说"发现什么"
- 开头给strategy overview，帮助读者理解各subsection的关系
- 具体参数grid、代码细节移到SI

### Supplementary Materials

**常见问题**：
- Section编号与正文不对应
- 标题使用defensive语言（"Additional diagnostics"）
- Figure captions重复正文的解释
- 引入新的命名convention（如Dataset A/B/C）

**SI原则**：
- Section编号与正文Results一一对应
- 标题用描述性语言，不用"Additional"
- Figure captions假设读者已读过正文
- 命名与正文保持一致

---

## 0.7 直接对应 npj Complexity 编辑意见的"硬约束"

编辑团队的典型质疑点（我们必须在正文中主动回答）：
1) **主流媒体数据也来自社交媒体**：必须明确它同样来自 Weibo，只是账号类型不同（来自账号 registry/规则识别）。  
2) **主流媒体/官媒体量不清**：必须报告各类账号的样本量（至少在用于检验的标注语料中）。  
3) **经验指标定义不清**：像 "Emotion High""risk"等必须给操作性定义（公式/计数规则/窗口定义）。  
4) **图表难评估**：轴标签、单位、图例、误差条/置信区间、字体大小必须可读。  
5) **模型规则/方程不清**：仿真更新规则、均值场方程（或关键动力学方程）必须写在 Methods/Model 中，而不是只靠代码或 notebook。

---

## 1) 操作性定义不足（最高优先级）

### 1.1 账号类型（account types）必须可复现

**症状**：文中出现 “mainstream / We-media / public”，但没说怎么分。  
**风险**：读者无法复现 `r_proxy`，直接被认定为“结果不可评估”。  
**标准写法（模板）**：
- 输入字段：`verify_typ`（蓝V/黄V/红V/无认证）+ `user_name`（账号名）
- Registry：`data/config/official_media_list.txt`（官媒/官方叙事账号名列表）
- 规则（建议在论文中用 3–5 行写清）：
  - `official narrative`：蓝V组织号，且（在 registry 中）或（命中媒体/政府关键词，如“日报/新闻/发布/卫健委”）
  - `We-media`：黄V（影响者/自媒体）
  - `public`：无认证或个人认证（红V + 无认证）
  - 例外：蓝V但不属于 `official narrative` 的组织号归入 `other`，并**从 `r_proxy` 分母中排除**（必须明确）
- 对应代码位置（便于审稿人/复现者核对）：`src/empirical/user_mapper.py`

**必须报告的体量（至少 1 次）**：
- 在“用于经验检验的标注语料”中，三类账号的帖子数（public / wemedia / official narrative）以及 `other` 的排除量。
- 如果论文中还会引用“全量爬取语料（未标注）”，可以额外报全量的量级，但要明确它们是否用于检验。

### 1.2 指标必须给“计算口径”

**症状**：图里写 “Emotion High”“risk”，正文没有公式或计数定义。  
**风险**：编辑会认为是“简单指标 + 弱相关 + 无法评估”，拒稿概率高。  
**标准写法（模板）**：
- 帖子级标签：`emotion_class ∈ {H,M,L}`，`risk_class ∈ {risk,norisk}`
- 窗口聚合（4H 举例）：
  - 公共用户 arousal 计数：$(n_H,n_M,n_L)$
  - $n_{pub}=n_H+n_M+n_L$（不足阈值则该窗口对 $Q,a$ 记为缺失）
  - $X_H=n_H/n_{pub}$，$X_M=n_M/n_{pub}$，$X_L=n_L/n_{pub}$
  - $Q=X_H-X_L$，$a=X_H+X_L$
- 媒体构成：
  - $n_{wemedia}$：窗口内 We-media 帖子数
  - $n_{official}$：窗口内 official narrative 帖子数
  - $r_{proxy}=n_{wemedia}/(n_{wemedia}+n_{official})$

### 1.3 “段”（segment）到底是什么必须讲清楚

**症状**：Results 里直接说 segment-level correlation，但读者不知道 segment 的时长与构造。  
**标准写法**：
- 先定义 window（例如 4H）  
- 再定义 segment（例如：由连续 4H windows 聚合成 1 周）  
- 说明段内统计：段均值/段标准差/段内分位数等

---

## 2) 口径漂移/不一致（会被质疑“p-hacking”）

**症状**：同一个量在不同地方用不同定义（最常见：`r_proxy` 分母是否包含政府账号、是否包含“蓝V但非官媒”的组织号、是否只算媒体账号）。  
**风险**：审稿人会认为你在挑口径以得到显著性。  
**解决**：
- 正文只保留 **一个主口径**（Primary definition），其余作为 robustness/appendix，并解释“为什么不是主口径”
- 给一个“口径表”（可以 1 个短表放 Appendix）：每个 proxy 的定义、分母、排除项、最小窗口阈值等

---

## 3) 逻辑跳跃（读者读起来“别扭”的根因）

**症状**：一句话突然引入阈值模型，下一句直接谈 mainstream/We-media，再下一句直接说“channel competition/控制参数”。  
**解决模板（四步桥接）**：
1) 现实观察：不同来源信息对情绪的影响方向可能相反  
2) 机制抽象：个体将风险线索聚合成“感知风险”，并通过阈值映射到 arousal 状态  
3) 系统反馈：两类来源对系统状态的响应不同（负反馈 vs 正反馈）  
4) 控制变量：用一个参数（$r$）连续调节两类反馈的权重，从而得到可解析的临界点与可检验预测

写作自检：如果读者问“为什么这句能推出下一句”，你能否用 1 句话补出桥接？

---

## 4) 证据强度与措辞不匹配（结论先行）

**症状**：
- 用 “confirm/validate/strongly support” 描述弱相关或样本很小的结果  
- 先写“结论”再找数据解释（典型是硬编码 print）  
**解决**：
- 用分层措辞：
  - 强：“supported / robust” （需要稳定、可复现、对照/稳健性通过）
  - 中：“consistent with / suggestive” （效应存在但受混杂/窗口影响）
  - 弱：明确写 “inconclusive / underpowered / limited by data continuity”
- 最少报告：`n` + effect size（相关系数/回归系数）+ CI（或误差条）+ p-value（如适用）

---

## 5) 图表不可评估（最常见的编辑“直接拒稿点”）

每张图必须满足：
- 轴标签：变量名 + 单位/时间尺度（4H? 1H? weekly?）
- 图例：每条线/点代表什么（数据集/账号类型/参数）
- 可读性：字体大小、线宽、颜色对比；避免图例压住数据
- 不确定性：误差条/阴影带/置信区间，或在图注说明为何没有（例如确定性理论曲线）
- 图注要闭环：读者只看图和图注就能知道“算的是什么、发现了什么”

---

## 6) “工程细节”不要污染正文

**症状**：正文出现内部命名（master/batch3）、缓存文件名、脚本参数串。  
**解决**：
- 正文用读者友好的命名：single-topic / multi-topic / pooled
- 参数表与运行细节放 Supplement/Appendix（或仓库文档），正文只给关键参数与口径

---

## 7) 最终自检 Checklist（可复制粘贴到每次改稿的 TODO）

### Narrative / Structure（叙事与结构）
- [ ] Title：10-15词，assertive，点明核心机制或发现
- [ ] Abstract：有明确问题陈述，核心发现突出，无验证语言
- [ ] Introduction：有stakes，gap聚焦，发现预告（非验证清单）
- [ ] Results：按科学问题递进排列，非按方法顺序
- [ ] 过渡句：删除所有"We next..."、"We return to..."等显式过渡
- [ ] Discussion：开头是interpretation（非summary），有具体implications
- [ ] Keywords：反映核心贡献，非supplementary内容

### Data / Operationalization
- [ ] 明确说明：主流媒体数据也来自 Weibo（不是外部媒体库）
- [ ] 账号分类给出输入字段、规则、排除项（并报告体量）
- [ ] `Q,a,r_proxy` 给公式与窗口定义；说明最小窗口阈值与缺失处理
- [ ] “段/事件/跳变”的定义明确且一致

### Model / Theory / Simulation
- [ ] 仿真更新规则在正文中可读（不是只在代码里）
- [ ] 均值场动力学方程/关键方程给出并解释参数含义
- [ ] 相变判据/临界点的推导关键步骤可追溯

### Evidence / Claims
- [ ] 结论措辞与证据强度匹配（supported vs suggestive vs inconclusive）
- [ ] 关键统计报告 `n + effect size + CI + p`
- [ ] 混杂因素（如 sampling density）明确说明并给控制结果

### Figures
- [ ] 轴/图例/单位/字体可读
- [ ] 有误差条/CI 或有合理解释
- [ ] 图注能独立解释“算什么 + 看到什么”
- [ ] Figure captions描述性，不重复正文解释

### References
- [ ] 数量适当（Nature子刊通常30-50篇）
- [ ] 覆盖核心领域：opinion dynamics, polarization, hybrid media
- [ ] 方法学引用完整：网络模型、统计方法
- [ ] 无unused条目（bib文件与正文一致）

### Supplementary Materials
- [ ] Section编号与正文Results对应
- [ ] 标题不使用"Additional"等defensive语言
- [ ] 命名与正文一致（无Dataset A/B/C等新convention）
- [ ] 有Parameter Table列出所有参数及baseline值

---

## 8) 词汇替换速查表

### 验证语言 → 发现语言
| ❌ 避免 | ✓ 使用 |
|---|---|
| validate | identify / reveal / show |
| confirm | find / demonstrate |
| test whether | show that / find that |
| is consistent with predictions | reveals / indicates |
| we next test | （删除，直接开始下一节） |

### Defensive语言 → Assertive语言
| ❌ 避免 | ✓ 使用 |
|---|---|
| To address this gap | We formalize / We show |
| A theoretical framework for | （直接说核心发现） |
| We attempt to | We |
| may potentially | can / does |

### 结构词替换
| ❌ 过度使用 | ✓ 替代方案 |
|---|---|
| First, Second, Third | 用段落主题句自然分隔 |
| In this section, we | （删除，直接说内容） |
| As mentioned above | （删除或用具体引用） |
