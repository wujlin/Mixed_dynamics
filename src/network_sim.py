"""
网络主体仿真 (Phase 3)

实现要点：
- 支持 BA/ER 网络生成（依赖 networkx）。
- 本地感知：结合全局媒体信号 p_env 与邻居高唤醒比例（系数 beta）。
- 状态更新：阈值规则 S_i ∈ {H, M, L}，由 p_i 与个体阈值 (phi_i, theta_i) 决定。
- 统计器：每步计算 Q(t), A(t)。

当 beta=0 且阈值对称时，模型退化为均匀混合近似，可用于对照理论 rc。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

STATE_HIGH = 1
STATE_MEDIUM = 0
STATE_LOW = -1


@dataclass
class NetworkConfig:
    n: int
    avg_degree: float
    model: Literal["ba", "er"] = "ba"
    beta: float = 0.0  # 邻居高唤醒影响系数
    update_rate: float = 0.1  # 异步更新比例，降低同步震荡
    init_state: Literal["random", "medium"] = "random"  # 初始状态配置
    sample_mode: Literal["degree", "fixed"] = "degree"  # 采样模式：按度或固定样本数
    sample_n: int = 50  # 当 sample_mode="fixed" 时使用
    # 对称模式：True 用于理论验证（p_env 在 q=0 时固定 0.5），False 为现实非对称（p_we 依赖 a）
    symmetric_mode: bool = False
    r: float = 0.5  # 移除比例
    n_m: float = 10.0
    n_w: float = 5.0
    phi: float = 0.6
    theta: float = 0.4
    seed: Optional[int] = None


def generate_network(cfg: NetworkConfig) -> nx.Graph:
    rng = np.random.default_rng(cfg.seed)
    if cfg.model == "ba":
        m = max(1, int(round(cfg.avg_degree / 2)))
        g = nx.barabasi_albert_graph(cfg.n, m, seed=cfg.seed)
    elif cfg.model == "er":
        p = cfg.avg_degree / max(cfg.n - 1, 1)
        g = nx.erdos_renyi_graph(cfg.n, p, seed=cfg.seed)
        if not nx.is_connected(g):
            # 保证连通性：若不连通则取最大连通子图，并重标节点为 0..n-1
            largest = max(nx.connected_components(g), key=len)
            g = g.subgraph(largest).copy()
            g = nx.convert_node_labels_to_integers(g, first_label=0)
        else:
            g = nx.convert_node_labels_to_integers(g, first_label=0)
    else:
        raise ValueError("Unsupported model, choose 'ba' or 'er'")

    # 预存邻接列表，便于快速访问
    nx.set_node_attributes(g, {i: list(g.neighbors(i)) for i in g.nodes}, "neighbors")
    return g


class NetworkAgentModel:
    def __init__(self, cfg: NetworkConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.g = generate_network(cfg)
        self.n = self.g.number_of_nodes()
        # 预存每个节点的度数，便于方案 B 按实际度数设置信号采样次数
        self.degrees = np.array([deg for _, deg in self.g.degree()], dtype=int)
        # 阈值可扩展为个体异质：此处简单用全局常数
        self.phi = np.full(self.n, cfg.phi, dtype=float)
        self.theta = np.full(self.n, cfg.theta, dtype=float)
        # 初始状态：支持随机三分或全员中立，便于线性稳定性验证
        if cfg.init_state == "random":
            probs = np.array([1 / 3, 1 / 3, 1 / 3])
            self.state = self.rng.choice(
                [STATE_LOW, STATE_MEDIUM, STATE_HIGH], size=self.n, p=probs
            )
        elif cfg.init_state == "medium":
            self.state = np.full(self.n, STATE_MEDIUM, dtype=int)
        else:
            raise ValueError("init_state 仅支持 'random' 或 'medium'")

    def _macro_stats(self) -> Tuple[float, float, int, int]:
        n_high = int(np.sum(self.state == STATE_HIGH))
        n_low = int(np.sum(self.state == STATE_LOW))
        q = (n_high - n_low) / self.n
        a = (n_high + n_low) / self.n
        return q, a, n_high, n_low

    def _global_env(self, q: float, a: float) -> float:
        # 按 Methods Eq. (1)(2)(3)
        p_main = (1 - q) / 2.0
        if self.cfg.symmetric_mode:
            # 理想对称：q=0 时 p_we=0.5，完全镜像 mainstream，用于验证理论 rc
            p_we = 0.5 + q / 2.0
        else:
            # 现实非对称：p_we 随活动度与极化走，初始 a<1 时 p_we<0.5，体现冷却偏置
            p_we = (a + q) / 2.0
        num = (1 - self.cfg.r) * self.cfg.n_m * p_main + self.cfg.r * self.cfg.n_w * p_we
        denom = (1 - self.cfg.r) * self.cfg.n_m + self.cfg.r * self.cfg.n_w
        return num / denom

    def _local_perception(self, p_env: float) -> np.ndarray:
        # 邻居高唤醒数量
        neighbor_high = np.zeros(self.n, dtype=float)
        for node in self.g.nodes:
            nbrs = self.g.nodes[node]["neighbors"]
            if not nbrs:
                continue
            neighbor_high[node] = np.sum(self.state[nbrs] == STATE_HIGH)

        # p_i = (global_num + beta * k_high) / (global_den + beta * k_i)
        # 简化：直接用 p_env 乘全局权重再加局部项
        base_num = p_env * ((1 - self.cfg.r) * self.cfg.n_m + self.cfg.r * self.cfg.n_w)
        base_den = (1 - self.cfg.r) * self.cfg.n_m + self.cfg.r * self.cfg.n_w

        local_num = self.cfg.beta * neighbor_high
        local_den = self.cfg.beta * self.degrees

        denom = base_den + local_den
        # 避免除零：孤立点只受全局影响
        denom_safe = np.where(denom == 0, base_den, denom)
        num = base_num + local_num
        p_i = num / denom_safe
        return np.clip(p_i, 0.0, 1.0)

    def _update_states(self, p_i: np.ndarray) -> None:
        # 方案 A：按度采样；方案 B：固定采样数（用于均匀混合对照）
        if self.cfg.sample_mode == "degree":
            N_SAMPLES = np.maximum(self.degrees, 1)
        elif self.cfg.sample_mode == "fixed":
            N_SAMPLES = np.full(self.n, max(1, int(self.cfg.sample_n)), dtype=int)
        else:
            raise ValueError("sample_mode 仅支持 'degree' 或 'fixed'")

        signal_counts = self.rng.binomial(n=N_SAMPLES, p=p_i)
        perceived_risk = signal_counts / N_SAMPLES
        new_state = np.where(
            perceived_risk >= self.phi,
            STATE_HIGH,
            np.where(perceived_risk <= self.theta, STATE_LOW, STATE_MEDIUM),
        )
        # 引入异步/惰性更新，避免全局同步导致的周期震荡
        update_mask = self.rng.random(self.n) < self.cfg.update_rate
        self.state = np.where(update_mask, new_state, self.state)

    def step(self) -> Tuple[float, float]:
        q, a, _, _ = self._macro_stats()
        p_env = self._global_env(q, a)
        p_i = self._local_perception(p_env)
        self._update_states(p_i)
        q_new, a_new, _, _ = self._macro_stats()
        return q_new, a_new

    def run(self, steps: int, record_interval: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = []
        q_records = []
        a_records = []
        for step in range(steps):
            if step % record_interval == 0:
                q, a, _, _ = self._macro_stats()
                times.append(step)
                q_records.append(q)
                a_records.append(a)
            self.step()
        # 记录最后一步
        q, a, _, _ = self._macro_stats()
        times.append(steps)
        q_records.append(q)
        a_records.append(a)
        return (
            np.asarray(times, dtype=float),
            np.asarray(q_records, dtype=float),
            np.asarray(a_records, dtype=float),
        )
