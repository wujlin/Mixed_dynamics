"""
文本预处理模块 (Text Preprocessor)

对微博文本进行清洗，去除噪音，保留核心内容，提升标注质量。

使用方法：
    from src.empirical import preprocess_weibo_text
    clean_text = preprocess_weibo_text(raw_text)
"""

from __future__ import annotations

import re
from typing import Optional


def preprocess_weibo_text(
    text: str,
    max_length: int = 500,
    keep_hashtags: bool = True,
    dedupe_hashtags: bool = True,
) -> str:
    """
    清洗微博文本，去除噪音，保留核心内容。
    
    Parameters
    ----------
    text : str
        原始微博文本
    max_length : int
        最大保留长度（超出部分截断）
    keep_hashtags : bool
        是否保留话题标签
    dedupe_hashtags : bool
        是否去重话题标签
    
    Returns
    -------
    str
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    original = text
    
    # 1. 移除 URL 和链接占位符
    # 微博链接格式：O网页链接、O+中文标题、http(s)://...
    text = re.sub(r'O网页链接', '', text)
    text = re.sub(r'O[^\s#@]{2,50}', '', text)  # O+标题形式
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'//\S+', '', text)  # 省略协议的链接
    
    # 2. 移除视频/图片标记
    # L用户微博视频、L+用户名+的微博视频
    text = re.sub(r'L\S+的微博视频', '', text)
    text = re.sub(r'L\S+微博视频', '', text)
    
    # 3. 处理转发前缀（保留内容，移除转发标记）
    # //@用户名: 或 //@用户名：
    text = re.sub(r'//@[^:：]+[:：]\s*', '', text)
    
    # 4. 移除"转发原文：" + 后续被删除提示
    text = re.sub(r'转发原文[：:]\s*(抱歉，此微博已被作者删除。查看帮助[：:]?)?', '', text)
    text = re.sub(r'抱歉，作者已设置仅展示半年内微博，此微博已不可见。', '', text)
    text = re.sub(r'抱歉，此微博已被作者删除。', '', text)
    
    # 5. 移除位置信息
    # 2北京、2绵阳·xxx
    text = re.sub(r'\s*2[^\s#@]{1,20}(·[^\s#@]{1,20})?', '', text)
    
    # 6. 移除@提及（保留文本语义）
    text = re.sub(r'@\S+', '', text)
    
    # 7. 处理话题标签
    if keep_hashtags:
        # 提取所有话题标签
        hashtags = re.findall(r'#[^#]+#', text)
        
        if dedupe_hashtags:
            # 去重，保持顺序
            seen = set()
            unique_hashtags = []
            for tag in hashtags:
                if tag not in seen:
                    seen.add(tag)
                    unique_hashtags.append(tag)
            hashtags = unique_hashtags
        
        # 从文本中移除所有话题标签
        text_without_tags = re.sub(r'#[^#]+#', '', text)
        
        # 限制话题标签数量（最多保留3个最相关的）
        relevant_tags = [t for t in hashtags if '新冠' in t or '后遗症' in t or '疫情' in t]
        other_tags = [t for t in hashtags if t not in relevant_tags]
        
        # 优先保留相关标签，最多3个
        final_tags = (relevant_tags + other_tags)[:3]
        
        # 重组：标签在前，正文在后
        text = ' '.join(final_tags) + ' ' + text_without_tags.strip()
    else:
        text = re.sub(r'#[^#]+#', '', text)
    
    # 8. 清理多余空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 9. 移除展开提示
    text = re.sub(r'\s*展开c?\s*$', '', text)
    
    # 10. 截断过长文本（保留开头，更可能包含核心信息）
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # 11. 如果清洗后为空，返回原文（避免丢失数据）
    if not text.strip():
        return original[:max_length] if len(original) > max_length else original
    
    return text


def extract_core_content(text: str) -> str:
    """
    提取核心内容（用于快速预览）。
    去除所有标签、链接，只保留纯文本。
    """
    if not text:
        return ""
    
    # 移除话题标签
    text = re.sub(r'#[^#]+#', '', text)
    # 移除链接
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'O\S+', '', text)
    # 移除@提及
    text = re.sub(r'@\S+', '', text)
    # 清理空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def is_valid_for_annotation(text: str, min_length: int = 5) -> bool:
    """
    判断文本是否适合标注。
    
    排除：
    - 仅有话题标签
    - 仅有被删除提示
    - 内容过短
    """
    if not text:
        return False
    
    core = extract_core_content(text)
    
    # 内容过短
    if len(core) < min_length:
        return False
    
    # 仅有被删除提示
    if '已被作者删除' in text or '此微博已不可见' in text:
        if len(core) < 20:  # 除了删除提示外内容很少
            return False
    
    return True


# 批量处理
def preprocess_batch(texts: list[str], **kwargs) -> list[str]:
    """批量预处理文本"""
    return [preprocess_weibo_text(t, **kwargs) for t in texts]




