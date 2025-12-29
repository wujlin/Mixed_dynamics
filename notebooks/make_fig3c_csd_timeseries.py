"""
Fig 3c：CSD 时间序列示例（远离临界 vs 近临界）——统一为 PDF/衬线字体/无网格。

来源：复用 notebooks/03_Critical_Slowing_Down.ipynb 的 demo 参数（sigma=0.15, dt=1e-2）。

运行：
  PYTHONDONTWRITEBYTECODE=1 /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig3c_csd_timeseries.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import add_panel_label, apply_paper_style  # noqa: E402
from src.sde_solver import SDEConfig, run_sde_simulation  # noqa: E402
from src.theory import get_gl_params  # noqa: E402


@dataclass(frozen=True)
class Fig3cConfig:
    sde_stats_path: Path = ROOT / "outputs" / "data" / "csd_sde_r_q_stats.npz"
    fig_size: Tuple[float, float] = (6.5, 2.45)
    # demo 参数：与 notebook 保持一致
    sigma: float = 0.15
    dt: float = 1e-2
    steps: int = 4000
    record_interval: int = 1
    n_plot: int = 1000
    seed0: int = 0
    # 选择 r：一个远离临界点，一个略低于 r_c
    r_safe: float = 0.0
    dr_critical: float = 0.02


def _simulate_q_series(rc: float, r: float, *, cfg: Fig3cConfig, seed: int) -> np.ndarray:
    alpha, u = get_gl_params(r=r, rc=rc)
    sde_cfg = SDEConfig(
        alpha=float(alpha),
        u=float(u),
        sigma=float(cfg.sigma),
        dt=float(cfg.dt),
        steps=int(cfg.steps),
        n_trajectories=1,
        seed=int(seed),
    )
    _, traj = run_sde_simulation(sde_cfg, q0=0.0, record_interval=int(cfg.record_interval))
    return traj[:, 0].astype(float, copy=False)


def main() -> None:
    cfg = Fig3cConfig()
    if not cfg.sde_stats_path.exists():
        raise FileNotFoundError(f"未找到 SDE 统计缓存：{cfg.sde_stats_path}")

    stats = np.load(cfg.sde_stats_path, allow_pickle=False)
    rc = float(stats["rc"])

    r_safe = float(cfg.r_safe)
    r_critical = float(max(rc - float(cfg.dr_critical), 0.0))

    q_safe = _simulate_q_series(rc, r_safe, cfg=cfg, seed=cfg.seed0)
    q_crit = _simulate_q_series(rc, r_critical, cfg=cfg, seed=cfg.seed0 + 1)

    n_plot = int(min(cfg.n_plot, q_safe.size, q_crit.size))
    x = np.arange(n_plot, dtype=int)
    q_safe = q_safe[:n_plot]
    q_crit = q_crit[:n_plot]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=cfg.fig_size, sharey=True)

    # 参考线：0（不进 legend）
    for ax in axes:
        ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=1)

    axes[0].plot(x, q_safe, lw=1.4, alpha=0.95, color="#0072B2", zorder=2)
    axes[1].plot(x, q_crit, lw=1.4, alpha=0.95, color="#D55E00", zorder=2)

    axes[0].set_title(rf"Stable ($r={r_safe:.2f}$)")  # 使用全局 titlesize (11pt)
    axes[1].set_title(rf"Near critical ($r={r_critical:.2f}$)")

    axes[0].set_xlabel("Time steps")
    axes[1].set_xlabel("Time steps")
    axes[0].set_ylabel(r"Polarization $q$")

    # 统一 y 轴范围：取两个子图数据的全局范围
    y_min = min(q_safe.min(), q_crit.min())
    y_max = max(q_safe.max(), q_crit.max())
    y_margin = (y_max - y_min) * 0.1
    for ax in axes:
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.tick_params(direction="in", top=True, right=True)

    add_panel_label(axes[0], "c")

    # 统一边距：与 Fig2c 保持一致的全宽排版
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.26, top=0.86, wspace=0.10)

    out_pdf = ROOT / "Essay" / "figures" / "fig3c_csd_timeseries.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig3" / "fig3c_csd_timeseries_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()

