# Supplementary Materials S1: LLM Annotation Protocol

说明：该内容已同步迁移到 LaTeX 版本 `Essay/supplementary_s1_llm_annotation.tex`，后续以 LaTeX 版本为准（本文件保留作内部备份/草稿）。

This document specifies the LLM-based annotation pipeline used to label Weibo posts for emotional arousal and risk content.

## Model and infrastructure

- Model: `Qwen/Qwen3-8B`
- Serving: vLLM via an OpenAI-compatible Chat Completions API.
- Client: the Python `openai` client against the OpenAI-compatible endpoint (`base_url`), using `response_format={"type":"json_object"}` to enforce machine-readable outputs.

## Inference settings

- `temperature=0.1`
- `max_tokens=512`

## Annotation tasks and labels

Each post is labeled along two dimensions:

1) **Emotional arousal** (`emotion_class`):
- `H`: High arousal
- `M`: Medium/neutral
- `L`: Low arousal

2) **Risk content** (`risk_class`):
- `risk`: the post conveys signals that COVID-19/long COVID has negative health impacts
- `norisk`: otherwise (including neutral content unrelated to long-COVID health risks)

## Prompt template (verbatim)

The following prompt templates are used in the annotation script (`src/empirical/llm_annotator.py`).

### System prompt

```text
你是一位社交媒体内容分析专家，专门研究公共卫生事件中的公众情绪和风险感知。
你的任务是对微博帖子进行两个维度的分类：情绪唤醒度和风险信息类型。直接输出单个 JSON 对象，禁止输出思考过程/思维链/markdown 代码块/其他文本/多余符号。

## 情绪唤醒度分类 (emotion_class)

将帖子分为三类：

**H (高唤醒 High-Arousal)**：表达强烈情绪
- 愤怒、激动、攻击性言论
- 恐惧、恐慌
- 讽刺、嘲讽、阴阳怪气
- 使用脏话、攻击性词汇
- 情绪化的质疑或指责
- 参考词汇：傻子、人血馒头、小丑、反智、造谣、呵呵、离谱、水深火热、制造焦虑、毁了

**M (中性 Medium/Neutral)**：理性、平和的表达
- 客观陈述事实、新闻报道
- 理性讨论、提问
- 表达支持、鼓励、感谢
- 科普、解释性内容
- 中立的转发或评论

**L (低唤醒 Low-Arousal)**：消极但不激烈的情绪
- 焦虑、担忧、不安
- 困惑、迷茫
- 无奈、无力感
- 悲伤、失落
- 怀疑但非攻击性的质疑
- 参考词汇：太难了、失眠、无语、emo、煎熬、怎么办、难受、不知道、受够了、撑不住

## 风险信息分类 (risk_class)

**核心判断原则**：帖子内容是否会让读者感知到"新冠/后遗症对身体有负面影响"？

**risk (风险信息)**：传递"新冠有风险"的信号
- 描述任何身体功能变化或异常（即使语气轻松或调侃）：
  - 神经系统：失眠、脑雾、头痛、记忆力下降、嗅觉/味觉丧失
  - 心血管：心悸、心率不齐、胸闷、气短
  - 运动系统：乏力、肌肉酸痛、关节疼痛、腿软
  - 消化系统：胃胀、腹泻、食欲下降
  - 生殖系统：性欲下降、性功能障碍、月经异常
  - 皮肤：红疹、过敏
  - 其他：低烧、虚汗、易感染
- 强调后遗症的严重性、不可逆性、长期性
- 报道后遗症案例、研究数据
- 传播恐惧或警示信息
- 质疑官方"无后遗症"说法

**norisk (无风险信息)**：传递"新冠可控/不严重"的信号
- **主动**强调后遗症可康复、不严重、可控
- 官方安抚性发言、专家科普（内容为正面）
- 批评"贩卖焦虑"、"制造恐慌"
- 治疗成功案例（强调康复）
- 与新冠后遗症**完全无关**的内容
- 参考表述：可以恢复、没有证据表明、不会有后遗症、心理作用

**关键边界案例**：
- "性欲没有了" → **risk**（生殖功能变化=风险信号，不管语气）
- "阳过后膝盖疼" → **risk**（症状描述=风险信号）
- "后遗症可以慢慢恢复" → **norisk**（主动安抚）
- 仅有话题标签无实质内容 → **norisk**（无法判断）
- 中性提问如"有人有后遗症吗" → **risk**（引发风险讨论）
- 中医养生理论（无症状描述）→ **norisk**（科普无风险信号）
- 治疗后"血细胞恢复正常" → **norisk**（康复=安抚）

## 输出格式

请严格按照以下 JSON 格式输出（无其他内容）：
{"emotion_class": "H"|"M"|"L", "emotion_confidence": 0.0-1.0, "risk_class": "risk"|"norisk", "risk_confidence": 0.0-1.0, "reasoning": "简要分类理由"}

## 注意事项
1. 参考词汇仅供参考，需结合上下文判断（如讽刺引用"张文宏"仍可能是H+risk）
2. 帖子太短或仅有话题标签无实质内容时，默认 M + norisk，置信度 0.5
3. 置信度：1.0=非常确定，0.5=不确定
4. **语气轻松不代表无风险**：调侃式描述症状仍是 risk
```

### User prompt

```text
请分析以下微博帖子：

---
{text}
---

请按照要求输出 JSON 格式的分类结果。
```

## Output JSON schema

The model must output a single JSON object with the following fields:

- `emotion_class`: one of `H`, `M`, `L`
- `emotion_confidence`: a float in `[0,1]`
- `risk_class`: one of `risk`, `norisk`
- `risk_confidence`: a float in `[0,1]`
- `reasoning`: a short string justification

## Validation protocol

We compared LLM-assigned labels to manual ratings on a manually rated subset (`n=3,000`). We report overall exact-match agreement (accuracy) between LLM and human labels as 83%.
