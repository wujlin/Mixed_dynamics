"""
通用工具函数。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def calculate_autocorrelation(time_series: ArrayLike, lag: int = 1, demean: bool = True) -> float:
    """
    计算单序列的自相关系数（lag-k）。

    参数：
    - time_series: 一维数组
    - lag: 滞后步数（正整数）
    - demean: 是否先去均值

    返回：
    - 自相关系数，范围 [-1, 1]
    """
    x = np.asarray(time_series, dtype=float)
    if x.ndim != 1:
        raise ValueError("time_series 必须为一维数组")
    if lag <= 0 or lag >= x.size:
        raise ValueError("lag 必须在 1 和 len(time_series)-1 之间")
    if demean:
        x = x - x.mean()
    x1 = x[:-lag]
    x2 = x[lag:]
    denom = np.sqrt(np.sum(x1**2) * np.sum(x2**2))
    if denom == 0:
        return 0.0
    return float(np.sum(x1 * x2) / denom)
