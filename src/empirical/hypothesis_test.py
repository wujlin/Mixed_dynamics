"""
假设检验模块 (Hypothesis Testing)

验证核心假设：Activity (a) 越高，情绪变化越陡峭（突变特征）。

主要功能：
1. 计算突变指标（dP/dt 峰值、断点检测）
2. 临界慢化信号检测（AC1、Variance 趋势）
3. 相关性检验（a vs 突变指标）

使用方法：
    metrics = calculate_jump_metrics(df_ts)
    csd_signals = detect_csd_signals(df_ts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class JumpMetrics:
    """突变指标"""
    dP_dt_max: float  # 极化变化率峰值
    dP_dt_max_time: pd.Timestamp  # 峰值发生时间
    da_dt_max: float  # Activity 变化率峰值
    n_changepoints: int  # 断点数量
    changepoint_times: List[pd.Timestamp]  # 断点时间
    jump_score: float  # 综合突变得分 [0, 1]


@dataclass  
class CSDSignals:
    """临界慢化信号"""
    ac1_trend: float  # AC1 趋势（正 = 上升）
    var_trend: float  # 方差趋势（正 = 上升）
    ac1_before_peak: float  # 峰值前的 AC1
    var_before_peak: float  # 峰值前的方差
    csd_detected: bool  # 是否检测到 CSD


def calculate_derivative(series: pd.Series, dt: float = 1.0) -> pd.Series:
    """
    计算时间序列的导数（变化率）
    
    Parameters
    ----------
    series : pd.Series
        时间序列
    dt : float
        时间间隔
        
    Returns
    -------
    pd.Series
        导数序列
    """
    return series.diff() / dt


def calculate_jump_metrics(
    df_ts: pd.DataFrame,
    Q_col: str = "Q",
    a_col: str = "a",
    time_col: str = "time_window",
    changepoint_penalty: float = 3.0,
) -> JumpMetrics:
    """
    计算突变指标
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    Q_col : str
        极化列名
    a_col : str
        Activity 列名
    time_col : str
        时间列名
    changepoint_penalty : float
        断点检测惩罚参数（越大断点越少）
        
    Returns
    -------
    JumpMetrics
        突变指标
    """
    df = df_ts.copy().dropna(subset=[Q_col, a_col])
    
    if len(df) < 3:
        return JumpMetrics(
            dP_dt_max=0.0,
            dP_dt_max_time=pd.NaT,
            da_dt_max=0.0,
            n_changepoints=0,
            changepoint_times=[],
            jump_score=0.0,
        )
    
    # 1. 计算变化率
    dQ_dt = calculate_derivative(df[Q_col].abs())  # 使用 |Q| 的变化率
    da_dt = calculate_derivative(df[a_col])
    
    # 2. 找峰值
    dQ_dt_abs = dQ_dt.abs()
    dP_dt_max = dQ_dt_abs.max() if len(dQ_dt_abs.dropna()) > 0 else 0.0
    
    if dP_dt_max > 0:
        peak_idx = dQ_dt_abs.idxmax()
        dP_dt_max_time = df.loc[peak_idx, time_col] if peak_idx in df.index else pd.NaT
    else:
        dP_dt_max_time = pd.NaT
    
    da_dt_max = da_dt.abs().max() if len(da_dt.dropna()) > 0 else 0.0
    
    # 3. 断点检测
    changepoint_times = []
    n_changepoints = 0
    
    try:
        import ruptures as rpt
        
        signal = df[Q_col].abs().values
        signal = signal[~np.isnan(signal)]
        
        if len(signal) > 10:
            algo = rpt.Pelt(model="rbf").fit(signal)
            change_points = algo.predict(pen=changepoint_penalty)
            # 排除最后一个点（总是被检测为断点）
            change_points = [cp for cp in change_points if cp < len(signal)]
            n_changepoints = len(change_points)
            
            # 映射回时间
            valid_times = df[time_col].iloc[:len(signal)]
            changepoint_times = [valid_times.iloc[cp] for cp in change_points if cp < len(valid_times)]
    except ImportError:
        # ruptures 未安装，跳过断点检测
        pass
    except Exception:
        pass
    
    # 4. 综合突变得分
    # 归一化各指标到 [0, 1]，然后加权
    dP_dt_norm = min(dP_dt_max / 0.5, 1.0)  # 假设 0.5 为较大变化率
    n_cp_norm = min(n_changepoints / 3.0, 1.0)  # 假设 3 个断点为较多
    
    jump_score = 0.6 * dP_dt_norm + 0.4 * n_cp_norm
    
    return JumpMetrics(
        dP_dt_max=dP_dt_max,
        dP_dt_max_time=dP_dt_max_time,
        da_dt_max=da_dt_max,
        n_changepoints=n_changepoints,
        changepoint_times=changepoint_times,
        jump_score=jump_score,
    )


def detect_csd_signals(
    df_ts: pd.DataFrame,
    Q_col: str = "Q",
    window_size: int = 6,
    peak_time: Optional[pd.Timestamp] = None,
    lookback: int = 12,
) -> CSDSignals:
    """
    检测临界慢化信号
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    Q_col : str
        极化列名
    window_size : int
        滚动窗口大小
    peak_time : pd.Timestamp, optional
        峰值时间（用于检测峰值前的信号）
    lookback : int
        峰值前回看的时间步数
        
    Returns
    -------
    CSDSignals
        临界慢化信号
    """
    df = df_ts.copy()
    series = df[Q_col].dropna()
    
    if len(series) < window_size + 2:
        return CSDSignals(
            ac1_trend=0.0,
            var_trend=0.0,
            ac1_before_peak=np.nan,
            var_before_peak=np.nan,
            csd_detected=False,
        )
    
    # 计算滚动 AC1 和方差
    from .time_series import calculate_rolling_ac1, calculate_rolling_stats
    
    df = calculate_rolling_ac1(df, column=Q_col, window_size=window_size)
    df = calculate_rolling_stats(df, window_size=window_size, columns=[Q_col])
    
    ac1_col = f"{Q_col}_rolling_ac1"
    var_col = f"{Q_col}_rolling_var"
    
    # 计算趋势（线性回归斜率）
    def calc_trend(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) < 3:
            return 0.0
        x = np.arange(len(valid))
        slope, _, _, _, _ = stats.linregress(x, valid.values)
        return slope
    
    ac1_trend = calc_trend(df[ac1_col]) if ac1_col in df.columns else 0.0
    var_trend = calc_trend(df[var_col]) if var_col in df.columns else 0.0
    
    # 峰值前的信号
    ac1_before_peak = np.nan
    var_before_peak = np.nan
    
    if peak_time is not None and "time_window" in df.columns:
        peak_idx = df[df["time_window"] <= peak_time].index
        if len(peak_idx) > lookback:
            before_peak = df.loc[peak_idx[-lookback:]]
            ac1_before_peak = before_peak[ac1_col].mean() if ac1_col in before_peak.columns else np.nan
            var_before_peak = before_peak[var_col].mean() if var_col in before_peak.columns else np.nan
    
    # 判断是否检测到 CSD（AC1 和 Var 都呈上升趋势）
    csd_detected = (ac1_trend > 0.01) and (var_trend > 0)
    
    return CSDSignals(
        ac1_trend=ac1_trend,
        var_trend=var_trend,
        ac1_before_peak=ac1_before_peak,
        var_before_peak=var_before_peak,
        csd_detected=csd_detected,
    )


def test_a_jump_correlation(
    topic_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    检验 Activity (a) 与突变指标的相关性
    
    Parameters
    ----------
    topic_data : List[Dict]
        每个话题的汇总数据，需包含：
        - a_mean: Activity 均值
        - a_max: Activity 最大值
        - jump_score: 突变得分
        
    Returns
    -------
    Dict[str, Any]
        相关性检验结果
    """
    if len(topic_data) < 3:
        return {
            "n_topics": len(topic_data),
            "error": "样本量不足（至少需要 3 个话题）",
        }
    
    df = pd.DataFrame(topic_data)
    
    results = {
        "n_topics": len(df),
    }
    
    # a_mean vs jump_score
    if "a_mean" in df.columns and "jump_score" in df.columns:
        valid = df[["a_mean", "jump_score"]].dropna()
        if len(valid) >= 3:
            r, p = stats.pearsonr(valid["a_mean"], valid["jump_score"])
            results["a_mean_vs_jump"] = {
                "correlation": r,
                "p_value": p,
                "significant": p < 0.05,
            }
    
    # a_max vs jump_score
    if "a_max" in df.columns and "jump_score" in df.columns:
        valid = df[["a_max", "jump_score"]].dropna()
        if len(valid) >= 3:
            r, p = stats.pearsonr(valid["a_max"], valid["jump_score"])
            results["a_max_vs_jump"] = {
                "correlation": r,
                "p_value": p,
                "significant": p < 0.05,
            }
    
    return results


def test_r_a_interaction(
    df_ts: pd.DataFrame,
    volatility_col: str = "Q",
) -> Dict[str, Any]:
    """
    检验 r_proxy 和 a 的交互效应
    
    理论预测：r_proxy 高 + a 高 → 最脆弱，波动最大
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    volatility_col : str
        波动性度量列
        
    Returns
    -------
    Dict[str, Any]
        交互效应检验结果
    """
    from .time_series import calculate_r_proxy
    
    df = df_ts.copy()
    df["r_proxy"] = calculate_r_proxy(df)
    
    # 计算波动性（滚动标准差）
    df["volatility"] = df[volatility_col].rolling(6, min_periods=1).std()
    
    # 去除缺失值
    valid = df[["r_proxy", "a", "volatility"]].dropna()
    
    if len(valid) < 10:
        return {"error": "数据量不足", "n_samples": len(valid)}
    
    results = {"n_samples": len(valid)}
    
    # 1. r_proxy 与波动性的相关
    r_vol, p_vol = stats.pearsonr(valid["r_proxy"], valid["volatility"])
    results["r_proxy_vs_volatility"] = {
        "correlation": r_vol,
        "p_value": p_vol,
        "significant": p_vol < 0.05,
    }
    
    # 2. a 与波动性的相关
    a_vol, p_a = stats.pearsonr(valid["a"], valid["volatility"])
    results["a_vs_volatility"] = {
        "correlation": a_vol,
        "p_value": p_a,
        "significant": p_a < 0.05,
    }
    
    # 3. 交互效应（简化版：分组比较）
    r_median = valid["r_proxy"].median()
    a_median = valid["a"].median()
    
    # 四象限分组
    high_r_high_a = valid[(valid["r_proxy"] > r_median) & (valid["a"] > a_median)]["volatility"]
    low_r_low_a = valid[(valid["r_proxy"] <= r_median) & (valid["a"] <= a_median)]["volatility"]
    
    if len(high_r_high_a) > 2 and len(low_r_low_a) > 2:
        t_stat, p_interaction = stats.ttest_ind(high_r_high_a, low_r_low_a)
        results["interaction_effect"] = {
            "high_r_high_a_volatility": high_r_high_a.mean(),
            "low_r_low_a_volatility": low_r_low_a.mean(),
            "t_statistic": t_stat,
            "p_value": p_interaction,
            "significant": p_interaction < 0.05,
            "theory_supported": high_r_high_a.mean() > low_r_low_a.mean(),
        }
    
    return results


def run_full_hypothesis_test(
    df_ts: pd.DataFrame,
    topic_name: str = "",
) -> Dict[str, Any]:
    """
    运行完整的假设检验流程
    
    检验假设：
    - H1: Activity (a) 高 → 突变
    - H2: r_proxy 高 → 波动大
    - H3: r_proxy × a 交互效应
    - H4: 突变前有临界慢化信号
    
    Parameters
    ----------
    df_ts : pd.DataFrame
        时间序列数据
    topic_name : str
        话题名称
        
    Returns
    -------
    Dict[str, Any]
        完整的检验结果
    """
    from .time_series import get_topic_summary, calculate_r_proxy
    
    # 1. 基础统计
    summary = get_topic_summary(df_ts, topic_name)
    
    # 2. 突变指标
    jump_metrics = calculate_jump_metrics(df_ts)
    
    # 3. 临界慢化信号
    csd_signals = detect_csd_signals(
        df_ts, 
        peak_time=jump_metrics.dP_dt_max_time,
    )
    
    # 4. r_proxy 和 a 的交互效应
    interaction_results = test_r_a_interaction(df_ts)
    
    # 5. 汇总结果
    results = {
        "topic_name": topic_name,
        "summary": summary,
        "jump_metrics": {
            "dP_dt_max": jump_metrics.dP_dt_max,
            "dP_dt_max_time": str(jump_metrics.dP_dt_max_time),
            "da_dt_max": jump_metrics.da_dt_max,
            "n_changepoints": jump_metrics.n_changepoints,
            "jump_score": jump_metrics.jump_score,
        },
        "csd_signals": {
            "ac1_trend": csd_signals.ac1_trend,
            "var_trend": csd_signals.var_trend,
            "csd_detected": csd_signals.csd_detected,
        },
        "parameter_effects": interaction_results,
        "hypothesis_support": {
            "H1_high_a_high_jump": summary["a_mean"] > 0.5 and jump_metrics.jump_score > 0.3,
            "H2_high_r_high_volatility": interaction_results.get("r_proxy_vs_volatility", {}).get("correlation", 0) > 0,
            "H3_interaction_effect": interaction_results.get("interaction_effect", {}).get("theory_supported", False),
            "H4_csd_before_jump": csd_signals.csd_detected,
        },
    }
    
    return results

