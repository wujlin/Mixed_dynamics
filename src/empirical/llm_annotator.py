"""
LLM 辅助标注器 (LLM-Assisted Annotator)

使用 GPT-4 / Claude API 对 Weibo 帖子进行情绪和风险分类标注。

标注任务：
1. emotion_class: 高唤醒(H) / 中性(M) / 低唤醒(L)
2. risk_class: 风险信息(risk) / 无风险信息(norisk)

使用方法：
    annotator = LLMAnnotator(provider="openai", api_key="...")
    result = annotator.annotate(text)
    # result: {"emotion": "H", "risk": "risk", "confidence": 0.9, "reasoning": "..."}
"""

from __future__ import annotations

import json
import time
import logging
from typing import Literal, Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path

# ================== 网络自动配置开始 ==================
# 若处于需代理的环境，自动清理干扰并设置本地隧道
import os

# 清除 no_proxy，避免直连失败
os.environ.pop("no_proxy", None)
os.environ.pop("NO_PROXY", None)

# 指定 socks5 代理（如需修改，请替换端口/地址）
PROXY_URL = "socks5://127.0.0.1:1080"
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL
os.environ["ALL_PROXY"] = PROXY_URL
# ================== 网络自动配置结束 ==================

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """单条标注结果"""
    text: str
    emotion_class: Literal["H", "M", "L"]  # 高唤醒/中性/低唤醒
    risk_class: Literal["risk", "norisk"]  # 风险/无风险
    emotion_confidence: float  # 情绪分类置信度 [0, 1]
    risk_confidence: float  # 风险分类置信度 [0, 1]
    reasoning: str  # LLM 推理过程
    raw_response: str  # 原始响应（调试用）
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Prompt 模板设计
# ============================================================

SYSTEM_PROMPT = """你是一位社交媒体内容分析专家，专门研究公共卫生事件中的公众情绪和风险感知。
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
"""

USER_PROMPT_TEMPLATE = """请分析以下微博帖子：

---
{text}
---

请按照要求输出 JSON 格式的分类结果。"""


class LLMAnnotator:
    """
    LLM 辅助标注器
    
    支持 OpenAI 和 Anthropic API。
    
    Parameters
    ----------
    provider : str
        API 提供商，"openai" 或 "anthropic"
    api_key : str
        API 密钥
    model : str, optional
        模型名称，默认根据 provider 自动选择
    max_retries : int
        最大重试次数
    retry_delay : float
        重试间隔（秒）
    """
    
    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: Optional[str] = None,  # 支持自托管/OpenAI 兼容端点（如本地 Qwen）
    ):
        self.provider = provider
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        
        # 默认模型
        if model is None:
            self.model = "gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307"
        else:
            self.model = model
        
        # 初始化客户端
        self._client = None
        self._init_client()
        
        # 统计
        self.total_tokens = 0
        self.total_requests = 0
    
    def _init_client(self):
        """初始化 API 客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                if self.base_url:
                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("请安装 anthropic: pip install anthropic")
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")
    
    def _call_openai(self, text: str, max_tokens: int = 512) -> tuple[str, int]:
        """调用 OpenAI API"""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # 低温度，更确定的输出
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return content, tokens
    
    def _call_anthropic(self, text: str) -> tuple[str, int]:
        """调用 Anthropic API"""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
        )
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return content, tokens
    
    def _parse_response(self, response: str, text: str) -> AnnotationResult:
        """解析 LLM 响应"""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            return AnnotationResult(
                text=text,
                emotion_class=data.get("emotion_class", "M"),
                risk_class=data.get("risk_class", "norisk"),
                emotion_confidence=float(data.get("emotion_confidence", 0.5)),
                risk_confidence=float(data.get("risk_confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"解析响应失败: {e}, response={response[:100]}")
            # 返回默认值
            return AnnotationResult(
                text=text,
                emotion_class="M",
                risk_class="norisk",
                emotion_confidence=0.0,
                risk_confidence=0.0,
                reasoning=f"解析失败: {e}",
                raw_response=response,
            )

    @staticmethod
    def _extract_json(resp: str) -> str:
        """
        从响应中提取 JSON，处理 <think>、markdown 代码块等杂质。
        """
        import re
        # 去除 <think>...</think>
        resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.S)
        # 优先提取 ```json ... ``` 块
        m = re.search(r"```json(.*?)```", resp, flags=re.S | re.I)
        if m:
            return m.group(1).strip()
        # 次选提取第一个 {...}
        m = re.search(r"\{.*\}", resp, flags=re.S)
        if m:
            return m.group(0)
        # 若没有，返回原文，交由 json 解析报错
        return resp.strip()
    
    def annotate(self, text: str, max_tokens: int = 512) -> AnnotationResult:
        """
        对单条文本进行标注
        
        Parameters
        ----------
        text : str
            待标注的文本
        max_tokens : int
            输出最大 token 数，避免被思维链占满
            
        Returns
        -------
        AnnotationResult
            标注结果
        """
        # 预处理：去除过长文本
        text = text[:500] if len(text) > 500 else text
        
        # 跳过过短文本
        if len(text.strip()) < 5:
            return AnnotationResult(
                text=text,
                emotion_class="M",
                risk_class="norisk",
                emotion_confidence=0.0,
                risk_confidence=0.0,
                reasoning="文本过短",
                raw_response="",
            )
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response, tokens = self._call_openai(text, max_tokens=max_tokens)
                else:
                    response, tokens = self._call_anthropic(text)
                
                self.total_tokens += tokens
                self.total_requests += 1
                
                return self._parse_response(response, text)
                
            except Exception as e:
                logger.warning(f"API 调用失败 (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        # 全部重试失败
        return AnnotationResult(
            text=text,
            emotion_class="M",
            risk_class="norisk",
            emotion_confidence=0.0,
            risk_confidence=0.0,
            reasoning="API 调用失败",
            raw_response="",
        )
    
    def annotate_batch(
        self,
        texts: List[str],
        output_path: Optional[Path] = None,
        progress_fn=None,
    ) -> List[AnnotationResult]:
        """
        批量标注
        
        Parameters
        ----------
        texts : List[str]
            待标注文本列表
        output_path : Path, optional
            输出路径，每条结果会追加写入（断点续传）
        progress_fn : callable, optional
            进度回调函数 progress_fn(current, total)
            
        Returns
        -------
        List[AnnotationResult]
            标注结果列表
        """
        results = []
        
        # 如果有输出文件，检查已处理的数量（断点续传）
        start_idx = 0
        if output_path and output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                start_idx = sum(1 for _ in f)
            logger.info(f"断点续传: 从第 {start_idx} 条开始")
        
        # 打开输出文件（追加模式）
        out_file = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = open(output_path, "a", encoding="utf-8")
        
        try:
            for i, text in enumerate(texts):
                if i < start_idx:
                    continue  # 跳过已处理的
                
                result = self.annotate(text)
                results.append(result)
                
                # 写入文件
                if out_file:
                    out_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    out_file.flush()
                
                # 进度回调
                if progress_fn:
                    progress_fn(i + 1, len(texts))
                
                # 避免 rate limit
                time.sleep(0.1)
                
        finally:
            if out_file:
                out_file.close()
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self._estimate_cost(),
        }
    
    def _estimate_cost(self) -> float:
        """估算成本（USD）"""
        # 粗略估算，实际价格请参考官方定价
        if self.provider == "openai":
            if "gpt-4o-mini" in self.model:
                return self.total_tokens * 0.00015 / 1000  # $0.15/1M input
            elif "gpt-4o" in self.model:
                return self.total_tokens * 0.005 / 1000  # $5/1M input
        elif self.provider == "anthropic":
            if "haiku" in self.model:
                return self.total_tokens * 0.00025 / 1000  # $0.25/1M input
            elif "sonnet" in self.model:
                return self.total_tokens * 0.003 / 1000  # $3/1M input
        return 0.0


# ============================================================
# 便捷函数
# ============================================================

def annotate_sample(
    texts: List[str],
    provider: str = "openai",
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    sample_size: int = 100,
) -> List[AnnotationResult]:
    """
    便捷函数：对样本进行标注
    
    Parameters
    ----------
    texts : List[str]
        全部文本
    provider : str
        API 提供商
    api_key : str
        API 密钥（如果为 None，从环境变量读取）
    output_path : str
        输出路径
    sample_size : int
        采样数量
        
    Returns
    -------
    List[AnnotationResult]
        标注结果
    """
    import random
    import os
    
    # 获取 API key
    if api_key is None:
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(f"请设置 API key: {provider.upper()}_API_KEY")
    
    # 采样
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
    
    # 标注
    annotator = LLMAnnotator(provider=provider, api_key=api_key)
    
    def progress(cur, total):
        if cur % 10 == 0:
            print(f"Progress: {cur}/{total}")
    
    results = annotator.annotate_batch(
        texts,
        output_path=Path(output_path) if output_path else None,
        progress_fn=progress,
    )
    
    # 打印统计
    stats = annotator.get_stats()
    print(f"\n=== 标注完成 ===")
    print(f"总请求数: {stats['total_requests']}")
    print(f"总 tokens: {stats['total_tokens']}")
    print(f"估算成本: ${stats['estimated_cost_usd']:.4f}")
    
    return results
