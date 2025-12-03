"""
时间序列聚合模块 (Time Series Aggregation)

将帖子级数据聚合为时间序列特征，用于验证理论预测。

核心输出特征：
- X_H, X_M, X_L: 公众情绪分布
- a = X_H + X_L = 1 - X_M: Activity（中立者缺失度）
- Q = X_H - X_L: 极化方向
- p_risk_mainstream, p_risk_wemedia: 媒体风险报道比例

使用方法：
    df_ts = aggregate_time_series(df, time_col='publish_time', freq='1H')
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal, Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class TimeSeriesConfig:
    """时间序列配置"""
    time_col: str = "publish_time"
    emotion_col: str = "emotion_class"
    risk_col: str = "risk_class"
    user_type_col: str = "user_type"
    freq: str = "1H"  # 时间窗口：1H, 4H, 1D
    min_posts: int = 5  # 最小帖子数（低于此数的窗口标记为缺失）


def aggregate_time_series(
    df: pd.DataFrame,
    config: Optional[TimeSeriesConfig] = None,
) -> pd.DataFrame:
    """
    将帖子数据聚合为时间序列
    
    Parameters
    ----------
    df : pd.DataFrame
        帖子数据，必须包含以下列：
        - time_col: 发布时间
        - emotion_col: 情绪分类 (H/M/L)
        - user_type_col: 用户类型 (mainstream/wemedia/public/...)
        - risk_col (可选): 风险分类 (risk/norisk)
    config : TimeSeriesConfig
        配置参数
        
    Returns
    -------
    pd.DataFrame
        时间序列数据，包含以下列：
        - time_window: 时间窗口
        - X_H, X_M, X_L: 公众情绪分布
        - a: Activity = 1 - X_M
        - Q: Polarization = X_H - X_L
        - p_risk_mainstream: 主流媒体风险比例
        - p_risk_wemedia: 自媒体风险比例
        - n_posts: 总帖子数
        - n_public: 公众帖子数
        - n_mainstream: 主流媒体帖子数
        - n_wemedia: 自媒体帖子数
    """
    if config is None:
        config = TimeSeriesConfig()
    
    # 复制并预处理
    df = df.copy()
    df[config.time_col] = pd.to_datetime(df[config.time_col])
    
    # 创建时间窗口
    df["time_window"] = df[config.time_col].dt.floor(config.freq)
    
    # 按时间窗口聚合
    results = []
    
    for window, group in df.groupby("time_window"):
        row = {"time_window": window}
        
        # 总帖子数
        row["n_posts"] = len(group)
        
        # 公众情绪分布（仅计算 public 用户）
        public_posts = group[group[config.user_type_col] == "public"]
        row["n_public"] = len(public_posts)
        
        if len(public_posts) >= config.min_posts:
            emotions = public_posts[config.emotion_col].value_counts(normalize=True)
            row["X_H"] = emotions.get("H", 0.0)
            row["X_M"] = emotions.get("M", 0.0)
            row["X_L"] = emotions.get("L", 0.0)
        else:
            row["X_H"] = np.nan
            row["X_M"] = np.nan
            row["X_L"] = np.nan
        
        # Activity 和 Polarization
        if not np.isnan(row.get("X_H", np.nan)):
            row["a"] = row["X_H"] + row["X_L"]  # Activity = 1 - X_M
            row["Q"] = row["X_H"] - row["X_L"]  # Polarization direction
        else:
            row["a"] = np.nan
            row["Q"] = np.nan
        
        # 主流媒体风险比例
        mainstream = group[group[config.user_type_col] == "mainstream"]
        row["n_mainstream"] = len(mainstream)
        if len(mainstream) > 0 and config.risk_col in group.columns:
            row["p_risk_mainstream"] = (mainstream[config.risk_col] == "risk").mean()
        else:
            row["p_risk_mainstream"] = np.nan
        
        # 自媒体风险比例
        wemedia = group[group[config.user_type_col] == "wemedia"]
        row["n_wemedia"] = len(wemedia)
        if len(wemedia) > 0 and config.risk_col in group.columns:
            row["p_risk_wemedia"] = (wemedia[config.risk_col] == "risk").mean()
        else:
            row["p_risk_wemedia"] = np.nan
        
        # 政府账号
        gov = group[group[config.user_type_col] == "government"]
        row["n_government"] = len(gov)
        
        results.append(row)
    
    df_ts = pd.DataFrame(results)
    df_ts = df_ts.sort_values("time_window").reset_index(drop=True)
    
    return df_ts


def calculate_rolling_stats(
    df_ts: pd.DataFrame,
    window_size: int = 6,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算滚动窗口统计量（用于临界慢化检测）
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    window_size : int
        滚动窗口大小
    columns : List[str], optional
        要计算的列，默认 ["a", "Q"]
        
    Returns
    -------
    pd.DataFrame
        带滚动统计量的时间序列
    """
    df_ts = df_ts.copy()
    
    if columns is None:
        columns = ["a", "Q"]
    
    for col in columns:
        if col in df_ts.columns:
            # 滚动均值
            df_ts[f"{col}_rolling_mean"] = df_ts[col].rolling(window_size, min_periods=1).mean()
            # 滚动方差
            df_ts[f"{col}_rolling_var"] = df_ts[col].rolling(window_size, min_periods=1).var()
            # 滚动标准差
            df_ts[f"{col}_rolling_std"] = df_ts[col].rolling(window_size, min_periods=1).std()
    
    return df_ts


def calculate_autocorrelation(
    series: pd.Series,
    lag: int = 1,
) -> float:
    """
    计算自相关系数（Lag-k）
    
    Parameters
    ----------
    series : pd.Series
        时间序列
    lag : int
        滞后步数
        
    Returns
    -------
    float
        自相关系数
    """
    series = series.dropna()
    if len(series) <= lag:
        return np.nan
    
    x = series.values
    x = x - x.mean()
    
    x1 = x[:-lag]
    x2 = x[lag:]
    
    denom = np.sqrt(np.sum(x1**2) * np.sum(x2**2))
    if denom == 0:
        return 0.0
    
    return float(np.sum(x1 * x2) / denom)


def calculate_rolling_ac1(
    df_ts: pd.DataFrame,
    column: str = "Q",
    window_size: int = 12,
) -> pd.DataFrame:
    """
    计算滚动 Lag-1 自相关（临界慢化信号）
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    column : str
        要计算的列
    window_size : int
        滚动窗口大小
        
    Returns
    -------
    pd.DataFrame
        带 AC1 的时间序列
    """
    df_ts = df_ts.copy()
    
    ac1_values = []
    series = df_ts[column].values
    
    for i in range(len(series)):
        if i < window_size - 1:
            ac1_values.append(np.nan)
        else:
            window = series[i - window_size + 1 : i + 1]
            window = window[~np.isnan(window)]
            if len(window) > 2:
                ac1 = calculate_autocorrelation(pd.Series(window), lag=1)
                ac1_values.append(ac1)
            else:
                ac1_values.append(np.nan)
    
    df_ts[f"{column}_rolling_ac1"] = ac1_values
    
    return df_ts


def calculate_r_proxy(df_ts: pd.DataFrame) -> pd.Series:
    """
    计算 r 的代理变量（主流媒体相对缺失程度）
    
    r_proxy = n_wemedia / (n_mainstream + n_wemedia)
    
    r_proxy 高 → 自媒体主导（正反馈占优）→ 类似高 r
    r_proxy 低 → 主流媒体主导（负反馈占优）→ 类似低 r
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
        
    Returns
    -------
    pd.Series
        r_proxy 序列
    """
    total_media = df_ts["n_mainstream"] + df_ts["n_wemedia"]
    # 避免除零
    r_proxy = df_ts["n_wemedia"] / total_media.replace(0, np.nan)
    return r_proxy


def calculate_feedback_balance(df_ts: pd.DataFrame) -> pd.Series:
    """
    计算正负反馈的平衡度
    
    当 p_risk_mainstream ≈ p_risk_wemedia 时，反馈接近对称
    
    balance = |p_risk_mainstream - p_risk_wemedia|
    
    balance 小 → 接近对称 → 可能产生相变
    balance 大 → 不对称 → 渐变
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
        
    Returns
    -------
    pd.Series
        feedback_balance 序列
    """
    if "p_risk_mainstream" in df_ts.columns and "p_risk_wemedia" in df_ts.columns:
        balance = (df_ts["p_risk_mainstream"] - df_ts["p_risk_wemedia"]).abs()
        return balance
    return pd.Series([np.nan] * len(df_ts))


def get_topic_summary(df_ts: pd.DataFrame, topic_name: str = "") -> Dict[str, Any]:
    """
    获取话题级别的汇总统计
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    topic_name : str
        话题名称
        
    Returns
    -------
    Dict[str, Any]
        汇总统计
    """
    valid_a = df_ts["a"].dropna()
    valid_Q = df_ts["Q"].dropna()
    
    # 计算 r_proxy
    r_proxy = calculate_r_proxy(df_ts)
    valid_r = r_proxy.dropna()
    
    summary = {
        "topic_name": topic_name,
        "n_windows": len(df_ts),
        "n_valid_windows": len(valid_a),
        "total_posts": df_ts["n_posts"].sum(),
        "duration_hours": (df_ts["time_window"].max() - df_ts["time_window"].min()).total_seconds() / 3600,
        
        # Activity 统计（a = 1 - X_M，中立者缺失度）
        "a_mean": valid_a.mean() if len(valid_a) > 0 else np.nan,
        "a_max": valid_a.max() if len(valid_a) > 0 else np.nan,
        "a_min": valid_a.min() if len(valid_a) > 0 else np.nan,
        "a_std": valid_a.std() if len(valid_a) > 0 else np.nan,
        
        # r_proxy 统计（主流媒体缺失程度）
        "r_proxy_mean": valid_r.mean() if len(valid_r) > 0 else np.nan,
        "r_proxy_max": valid_r.max() if len(valid_r) > 0 else np.nan,
        "r_proxy_std": valid_r.std() if len(valid_r) > 0 else np.nan,
        
        # Polarization 统计
        "Q_mean": valid_Q.mean() if len(valid_Q) > 0 else np.nan,
        "Q_abs_max": valid_Q.abs().max() if len(valid_Q) > 0 else np.nan,
        "Q_std": valid_Q.std() if len(valid_Q) > 0 else np.nan,
        
        # 情绪分布均值
        "X_H_mean": df_ts["X_H"].mean(),
        "X_M_mean": df_ts["X_M"].mean(),
        "X_L_mean": df_ts["X_L"].mean(),
        
        # 媒体分布
        "n_mainstream_total": df_ts["n_mainstream"].sum(),
        "n_wemedia_total": df_ts["n_wemedia"].sum(),
        "n_public_total": df_ts["n_public"].sum(),
    }
    
    return summary

