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
    - k_avg: 信息采样次数（每个个体每步采样的信号数量）。
              **重要**：此参数必须与 ABM 中的 NetworkConfig.sample_n 一致！
              - 当 ABM 使用 sample_mode="fixed" 时，k_avg = sample_n
              - 当 ABM 使用 sample_mode="degree" 时，k_avg ≈ avg_degree

    返回：
    - chi: 响应斜率近似值（标量）。

    示例：
        >>> chi = calculate_chi(phi=0.54, theta=0.46, k_avg=50)
        >>> print(f"chi = {chi:.3f}")  # 约 9.6
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
    计算临界移除比例 r_c。

    推导过程（对称假设 p_main=(1-q)/2, p_we=0.5+q/2）：
    1. 反馈梯度 Γ = ∂p_env/∂q|_{q=0} = [r·n_w - (1-r)·n_m] / {2·[(1-r)·n_m + r·n_w]}
    2. 稳定性条件 χ·Γ = 1 解出 r_c
    3. 正确公式：r_c = n_m(χ+2) / [n_m(χ+2) + n_w(χ-2)]

    参数：
    - n_m: 主流媒体基数 n_m > 0
    - n_w: 自媒体基数 n_w > 0
    - chi: 心理敏感度，可为标量或数组

    返回：
    - r_c 与 chi 同形状的 ndarray

    注意：
    - 当 χ < 2 时，分母中 (χ-2) < 0，可能导致 r_c > 1（无相变）或 r_c < 0
    - 当 χ = 2 时，r_c = 1（临界边界）
    - 当 χ > 2 时，0 < r_c < 1（存在相变）
    """
    if n_m <= 0 or n_w <= 0:
        raise ValueError("n_m 与 n_w 必须为正数")

    chi_arr = np.asarray(chi, dtype=float)
    numerator = n_m * (chi_arr + 2.0)
    denominator = n_m * (chi_arr + 2.0) + n_w * (chi_arr - 2.0)

    # 处理分母为零的情况（当 n_m(χ+2) = -n_w(χ-2) 时）
    with np.errstate(divide='ignore', invalid='ignore'):
        rc = numerator / denominator
        # 分母为零时设为 inf（无相变）
        rc = np.where(np.isclose(denominator, 0.0), np.inf, rc)

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
