"""
经验数据验证模块 (Empirical Validation Module)

Phase 5: 用 Weibo Long-COVID 数据验证理论预测
- 文本分类（情绪 H/M/L，风险 risk/norisk）
- 用户类型映射（mainstream/wemedia/public）
- 时间序列聚合（X_H, X_M, X_L, a, Q）
- 假设检验（a 与突变指标的关系）

核心假设：
- H1: Activity (a = 1 - X_M) 越高，情绪变化越陡峭（突变特征）
- H2: Activity 低时，情绪呈渐变过渡
- H3: 突变前应出现临界慢化信号（AC1↑, Var↑）
"""

# 延迟导入，避免循环依赖和不必要的加载
def __getattr__(name):
    if name == "LLMAnnotator":
        from .llm_annotator import LLMAnnotator
        return LLMAnnotator
    elif name == "EmotionClassifier":
        from .classifier import EmotionClassifier
        return EmotionClassifier
    elif name == "aggregate_time_series":
        from .time_series import aggregate_time_series
        return aggregate_time_series
    elif name == "calculate_jump_metrics":
        from .hypothesis_test import calculate_jump_metrics
        return calculate_jump_metrics
    elif name == "UserTypeMapper":
        from .user_mapper import UserTypeMapper
        return UserTypeMapper
    elif name == "load_topic_dataset":
        from .data_loader import load_topic_dataset
        return load_topic_dataset
    elif name == "preprocess_weibo_text":
        from .text_preprocessor import preprocess_weibo_text
        return preprocess_weibo_text
    elif name == "is_valid_for_annotation":
        from .text_preprocessor import is_valid_for_annotation
        return is_valid_for_annotation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LLMAnnotator",
    "EmotionClassifier", 
    "aggregate_time_series",
    "calculate_jump_metrics",
    "UserTypeMapper",
    "load_topic_dataset",
    "preprocess_weibo_text",
    "is_valid_for_annotation",
]

