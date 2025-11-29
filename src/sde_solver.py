"""
SDE 求解器：用于验证 GL 势能下的随机动力学 (Phase 2)。

核心方程（Methods Eq. 9 近似形式）:
    dq = (-alpha*q - u*q^3) dt + sigma * sqrt(dt) * N(0,1)

其中漂移项对应 V'(q) = alpha*q + u*q^3，噪声项方差 sigma^2。
稳态分布：P(q) ∝ exp(-V(q)/D)，其中 D = sigma^2 / 2。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .theory import potential_energy


@dataclass
class SDEConfig:
    alpha: float
    u: float = 1.0
    sigma: float = 0.2
    dt: float = 1e-2
    steps: int = 10_000
    n_trajectories: int = 1
    seed: Optional[int] = None


def euler_maruyama_step(q: np.ndarray, alpha: float, u: float, sigma: float, dt: float, rng: np.random.Generator) -> np.ndarray:
    """
    Euler-Maruyama 单步更新。

    方程：q_{t+1} = q_t + (-alpha*q_t - u*q_t^3) dt + sigma * sqrt(dt) * xi_t
    其中 xi_t ~ N(0,1)。
    """
    noise = rng.standard_normal(q.shape, dtype=float)
    drift = (-alpha * q - u * q**3) * dt
    diffusion = sigma * np.sqrt(dt) * noise
    return q + drift + diffusion


def run_sde_simulation(
    config: SDEConfig,
    q0: ArrayLike = 0.0,
    record_interval: int = 1,
    log_interval: Optional[int] = None,
    progress_fn: Optional[Callable[[int, int, float, np.ndarray], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    运行 Langevin 动力学，支持多条轨迹并返回时间序列。

    参数：
    - config: SDEConfig，包含 alpha/u/sigma/dt/steps/n_trajectories/seed
    - q0: 初值标量或数组；若标量则广播到所有轨迹
    - record_interval: 记录步长（>1 时做子采样以减小存储）
    - log_interval: 日志步长（步数，None 表示不输出）；应选择为记录间隔的整数倍以减少开销
    - progress_fn: 可选回调，签名 progress_fn(step:int, total:int, t:float, q:np.ndarray)
                  可用于自定义进度条/日志（例如 tqdm 手动更新）

    返回：
    - t: 记录的时间戳数组 (shape [T,])
    - q_traj: 轨迹数组 (shape [T, n_trajectories])
    """
    rng = np.random.default_rng(config.seed)
    n_traj = int(config.n_trajectories)

    q = np.broadcast_to(np.asarray(q0, dtype=float), (n_traj,)).copy()
    times = []
    records = []

    for step in range(config.steps):
        t = step * config.dt
        if step % record_interval == 0:
            times.append(t)
            records.append(q.copy())
        q = euler_maruyama_step(q, config.alpha, config.u, config.sigma, config.dt, rng)
        if log_interval and progress_fn and (step + 1) % log_interval == 0:
            progress_fn(step + 1, config.steps, t + config.dt, q)

    # 记录最后一步
    times.append(config.steps * config.dt)
    records.append(q.copy())

    t_arr = np.asarray(times, dtype=float)
    q_arr = np.stack(records, axis=0)  # [T, n_traj]
    return t_arr, q_arr


def theoretical_pdf(q: ArrayLike, alpha: float, u: float, sigma: float) -> np.ndarray:
    """
    解析稳态概率密度 P(q) ∝ exp(-V(q)/D)，D = sigma^2/2。
    返回归一化后的密度（使用数值积分近似归一化）。
    """
    q_arr = np.asarray(q, dtype=float)
    V = potential_energy(q_arr, alpha=alpha, u=u)
    D = sigma**2 / 2.0
    unnorm = np.exp(-V / D)

    # 数值归一化：使用梯形法
    # 若 q 非等距，np.trapezoid 会按采样点处理
    norm = np.trapezoid(unnorm, q_arr)
    if norm <= 0:
        raise ValueError("归一化因子非正，检查输入参数或 q 范围")
    return unnorm / norm
