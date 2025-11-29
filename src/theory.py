"""
理论核心模块：计算心理敏感度 chi、临界点 rc、GL 参数以及有效势能。

主要基于 DEVELOPMENT.md/Methods.md 中的 Phase 1 公式：
- chi: 由微观二项分布在阈值边界处的概率质量给出响应函数斜率（Eq. 5 思路）。
- rc: 稳定性判据 chi * Gamma = 1 导出的解析式 (Eq. 8)。
- GL 参数: alpha = rc - r, u>0 控制饱和。
- 势能: V(q) = 0.5*alpha*q^2 + 0.25*u*q^4 (Eq. 10)。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom


def calculate_chi(phi: float, theta: float, k_avg: int) -> float:
    """
    计算心理敏感度 χ = dS/dp|_{p=0.5}，使用二项分布在阈值边界的概率质量近似。

    物理背景（Methods Eq. 5）：在对称阈值假设下（phi = 1 - theta），微小的风险扰动
    会促使位于阈值附近的个体跨越阈值，因此响应斜率与边界概率密度成正比。
    这里用 binomial(n=k_avg, p=0.5) 在两个阈值处的 pmf 之和，再乘以 k_avg
    作为放大量纲，得到可达 χ>2 的敏感区间。

    参数：
    - phi: 高阈值（推荐 0.5-1 区间，且满足 phi ≈ 1 - theta）。
    - theta: 低阈值。
    - k_avg: 平均接触次数/平均度，决定二项分布试验数。

    返回：
    - chi: 响应斜率近似值（标量）。
    """
    if k_avg <= 0:
        raise ValueError("k_avg 必须为正整数")

    n = int(round(k_avg))
    p0 = 0.5
    k_high = int(round(phi * n))
    k_low = int(round(theta * n))

    pmf_high = binom.pmf(k_high, n, p0)
    pmf_low = binom.pmf(k_low, n, p0)

    # 对称时 pmf_high ≈ pmf_low，乘以 n 放大到宏观敏感度尺度。
    chi = n * (pmf_high + pmf_low)
    return float(chi)


def calculate_rc(n_m: float, n_w: float, chi: ArrayLike) -> np.ndarray:
    """
    计算临界移除比例 r_c，公式来源：Eq. (8) r_c = 1 + (n_w/n_m) * (2 - chi) / (2 + chi)。

    参数：
    - n_m: 主流媒体基数 n_m > 0
    - n_w: 自媒体基数 n_w > 0
    - chi: 心理敏感度，可为标量或数组

    返回：
    - r_c 与 chi 同形状的 ndarray
    """
    if n_m <= 0 or n_w <= 0:
        raise ValueError("n_m 与 n_w 必须为正数")

    chi_arr = np.asarray(chi, dtype=float)
    if np.any(np.isclose(chi_arr, -2.0)):
        raise ValueError("chi 不能等于 -2，会导致分母为零")
    rc = 1.0 + (n_w / n_m) * (2.0 - chi_arr) / (2.0 + chi_arr)
    return rc


def get_gl_params(r: ArrayLike, rc: float, u: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    映射控制参数到 GL 系数。

    参数：
    - r: 当前移除比例，可为标量或数组
    - rc: 临界点
    - u: 饱和值系数，默认 1.0

    返回：
    - alpha 数组（形状同 r）
    - u（回传便于调用链统一）
    """
    alpha = np.asarray(rc - np.asarray(r, dtype=float), dtype=float)
    return alpha, float(u)


def potential_energy(q: ArrayLike, alpha: float, u: float = 1.0) -> np.ndarray:
    """
    计算有效势能 V(q) = 0.5*alpha*q^2 + 0.25*u*q^4。

    参数：
    - q: 极化方向变量，可为标量或数组
    - alpha: 二阶系数（由 r_c - r 决定）
    - u: 四阶系数，需为正以保证势能有界

    返回：
    - V(q) 与 q 同形状的 ndarray
    """
    q_arr = np.asarray(q, dtype=float)
    return 0.5 * alpha * q_arr**2 + 0.25 * u * q_arr**4
