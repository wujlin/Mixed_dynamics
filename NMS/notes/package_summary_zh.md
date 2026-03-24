# NMS 投稿包中文摘要

## 1. 这套稿件是什么

这是一套投给 **New Media & Society** 的投稿包，主题不是你当前仓库里那篇复杂系统相变论文，而是另一篇关于 **抖音与快手平台比较** 的论文。

主标题是：

`Platformed Recognition: Issue Visibility, Affective Evaluation, and Boundary-Making across Douyin and Kuaishou`

从标题页可见：

- 稿件类型：`Original Article`
- 目标长度：约 `8,080` 词
- 摘要：`150` 词
- 审稿材料中包含 `7` 个表、`4` 个图

## 2. 文章核心问题

文章想回答的问题是：

- 短视频平台不仅传播身份相关话题，也会塑造“哪些议题更容易被看见”
- 这些被看见的议题会被用户以怎样的情感方式评价
- 这种情感评价又如何进一步推动评论中的身份边界取向

论文把这个机制概括为一条顺序链条：

- `platform regime -> issue visibility -> affective evaluation -> identity-boundary orientation`

也就是说，平台差异首先体现在“上游的议题生态”，而不是直接体现在最终态度表达上。

## 3. 数据与方法

主文稿和附录里给出的数据规模是：

- 抖音评论：`634,289`
- 快手评论：`518,763`
- 全部清洗后语料合计约：`1.15 million comments`
- 用于聚类的月度分层样本：`115,305`
- 人工标注子样本：`10,000`

使用的方法包括：

- topic clustering
- semantic-network analysis
- supervised large-scale labeling
- multinomial logistic regression
- mediation analysis

附录里还给了模型验证信息：

- affective-evaluation classifier：`83.62%` accuracy，`F1 = .8303`
- boundary-making classifier：`83.26%` accuracy，`F1 = .8317`

## 4. 论文提出的 5 个假设

根据主文稿和表格文件，假设是：

- `H1`：抖音和快手在议题结构与话语可见性上显著不同
- `H2`：议题结构会显著预测情感评价
- `H3`：情感评价会显著预测身份边界取向
- `H4`：情感评价在议题结构与身份边界取向之间起中介作用
- `H5`：平台之间的差异更多体现在上游议题生态，而不是下游情感中介机制本身

## 5. 主文稿的结构

从匿名主文稿提取出的章节结构是：

- `1. Introduction`
- `2. Platform Visibility, Affective Publics, and Identity-Boundary Work`
- `3. Theoretical Framework and Hypotheses`
- `4. Data and Methods`
- `5. Findings`
- `5.1 Issue structures and platform ecologies`
- `5.2 Semantic networks and discursive organization`
- `5.3 Multinomial logit and mediation results`
- `6. Discussion`
- `7. Conclusion`

## 6. 主要发现

### 6.1 上游议题生态确实不同

抖音的议题结构更分散、更均衡，主要包括：

- stigma and respect：`32.79%`
- tradition and cultural distinctiveness：`29.81%`
- commonality and shared belonging：`19.62%`
- identity and pride：`17.77%`

快手则更集中，尤其集中在一个大类上：

- performance and revival：`68.40%`
- Han identity and race：`18.70%`
- conflict and confrontation：`12.90%`

这意味着：

- 抖音的议题入口更多样
- 快手的议题更集中，也更容易向本质化身份和冲突表达聚拢

### 6.2 语义网络结构也不同

主文稿中给出的可恢复网络指标显示：

- Douyin comment network：`90` nodes，`192` edges，平均路径 `3.41`，聚类系数 `0.360`
- Kuaishou comment network：`471` nodes，`602` edges，平均路径 `5.63`，聚类系数 `0.280`

作者的解释是：

- 抖音语义网络更紧凑、更连贯
- 快手网络更大、更分散，也更容易形成割裂的语义块

### 6.3 下游情感中介机制在两平台上都成立

主文稿认为，真正稳定的机制不是“平台直接决定边界取向”，而是：

- 平台先塑造议题结构
- 议题结构再影响情感评价
- 情感评价再影响 integrative / neutral / conflictual 的边界取向

文中强调：

- affective evaluation 是两平台上都比较稳定、且影响最强的预测变量
- 平台差异主要发生在上游 issue ecology，而不是 affect 到 boundary orientation 的基本方向上

## 7. 各个提交文件分别在做什么

- `NMS_Anonymous_Manuscript.docx`
  - 匿名审稿版主文稿
- `NMS_Title_Page.docx`
  - 作者、单位、通讯方式、基金、伦理、数据可得性等信息
- `NMS_Cover_Letter.docx`
  - 给 `New Media & Society` 编辑的投稿信
- `NMS_Tables.docx`
  - 可编辑的 7 个表
- `NMS_Supplementary_Appendix.docx`
  - 方法、复现性和可恢复模型输出的补充说明
- `NMS_Submission_Compliance_Check.docx`
  - 一份投稿合规检查表，列出哪些地方已经满足，哪些还待确认

## 8. 当前最值得注意的点

这套投稿包已经比较完整，但 compliance check 里明确写了几类仍需最终确认的事项：

- 摘要是否正好 `150` 词
- title page 里是否还有占位符未替换
- funding / ethics / consent / data availability 是否在所有文件中完全一致
- ORCID 是否已经最终填入
- AI 辅助披露是否决定保留，并保持所有文件一致
- 是否能补上更完整的 coefficient-level model output

这些细节我已经另外整理到：

- [submission_open_items.md](/Users/jinlin/Desktop/Project/Complex_dynamics/Mixed_dynamics/NMS/notes/submission_open_items.md)
