"""
Fig 1a：理论分岔图（对称域）——均场 pitchfork + 理论 r_c（基准参数）。

基准参数（与正文/后续图一致）：
- n_m=10, n_w=5
- thresholds: phi=0.54, theta=0.46
- information density: k=50

绘图说明：
- 仅画理论稳态分支（示意），不叠加数值仿真散点（仿真验证在 Fig2）。
- r_c 用淡灰点线标注（不进 legend，可在 caption 说明）。
- 风格统一到 Fig2–5：Times New Roman、无网格、线宽字号一致、PDF 导出。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig1a_bifurcation.py
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

from src import theory  # noqa: E402
from src.plot_style import FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig1aConfig:
    n_m: float = 10.0
    n_w: float = 5.0
    phi: float = 0.54
    theta: float = 0.46
    k: int = 50
    u: float = 1.0
    fig_size: Tuple[float, float] = FIGSIZE_HALF


def main() -> None:
    cfg = Fig1aConfig()
    chi = theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=int(cfg.k))
    rc = float(theory.calculate_rc(n_m=cfg.n_m, n_w=cfg.n_w, chi=chi))

    r = np.linspace(0.0, 1.0, 401)
    q_branch = np.sqrt(np.maximum(r - rc, 0.0) / float(cfg.u))
    q0 = np.zeros_like(r)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    color = "black"
    ax.plot(r, q0, color=color, lw=2.4)
    ax.plot(r, q_branch, color=color, lw=2.4)
    ax.plot(r, -q_branch, color=color, lw=2.4)

    ax.axvline(rc, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Order parameter $q$")
    ax.set_xlim(0.0, 1.0)
    # y 轴对称给一点余量
    ymax = float(np.max(q_branch)) if np.max(q_branch) > 0 else 1.0
    ax.set_ylim(-1.05 * ymax, 1.05 * ymax)
    ax.tick_params(direction="in", top=True, right=True)

    add_panel_label(ax, "a")
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.22, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig1a_bifurcation.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig1" / "fig1a_bifurcation_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"chi={chi:.3f} rc={rc:.3f}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
