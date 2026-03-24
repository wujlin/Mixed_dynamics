# Submission Open Items

下面这些不是我推测的，而是来自 `NMS_Submission_Compliance_Check.docx` 里明确列出的仍需确认项。

## 明确待检查或待补充的项目

- 摘要长度是否最终精确到 `150` 词
- 审稿匿名稿的文件元数据是否已经清理
- 标题页里是否还有占位符没有替换成最终作者信息
- cover letter 里是否还要加入系统要求的额外 disclosure wording
- funding / ethics / consent / data availability / permissions 这些表述，是否已经在 title page、cover letter 和投稿系统字段中完全一致
- Sage Harvard 的作者-年份引用格式，是否已做最后一轮交叉检查
- 图件是否已准备好 standalone `300 dpi` 文件
- data availability statement 是否已经选定最终版本
- ethics / consent 的最终措辞是否已经定稿
- ORCID 是否已经由提交作者在系统和标题页中同时填写
- AI / third-party writing assistance disclosure 是否决定保留，以及是否在所有文件中统一
- 如果完整的 coefficient-level model output 可以恢复，是否要替换掉当前只保留部分统计量的版本
- 标题页中的通讯邮箱是否有拼写或版本不一致问题
  - `NMS_Title_Page` 中出现 `jwu@conncet.hkust-gz.edu.cn`
  - `NMS_Cover_Letter` 中出现 `Jwu923@connect.hkust-gz.edu.cn`
- 正文结果段落与表格中的 OR 数值是否完全一致
  - 正文 5.3 写到 Douyin / Kuaishou 的 affective evaluation OR 为 `2.61 / 0.61` 与 `2.32 / 0.77`
  - `Table 7` 中对应 retained odds ratios 为 `6.47 / 0.29` 与 `6.68 / 0.30`
  - 需要确认是不同模型口径，还是正文或表格仍有一处未同步

## 当前状态上特别容易遗漏的点

从合规表里看，最容易在真正提交前漏掉的是：

- `Partially meets`
  - title page
  - data availability
  - ethics and consent
- `Conditionally meets`
  - abstract 150 词
  - statements and declarations
  - reference style
  - figure permissions / originality
  - complete coefficient-level model output
- `Pending`
  - ORCID
  - AI disclosure
- `Potential inconsistency`
  - corresponding email spelling / version
  - main-text OR vs Table 7 OR

## 建议的临提交前复核顺序

1. 先锁定最终标题页信息
2. 再统一 funding / ethics / consent / data availability 的文字
3. 再检查 abstract 词数、参考文献格式、匿名元数据，以及邮箱/作者信息一致性
4. 再核对正文结果数值与表格数值是否完全同步
5. 最后检查 figure 文件、ORCID、AI disclosure 和模型输出附件
