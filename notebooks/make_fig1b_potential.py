"""
Fig 1b：有效势 V_eff(q) 随控制参数 r 的形态演化（对称域，基准参数）。

基准参数（与正文/后续图一致）：
- n_m=10, n_w=5
- thresholds: phi=0.54, theta=0.46
- information density: k=50

绘图说明：
- 使用 GL 势能：V(q)=0.5*alpha*q^2+0.25*u*q^4，其中 alpha=rc-r
- 选择三个代表性的 r：r<rc（单井）、r=rc（临界）、r>rc（双井）
- 额外加入 r=1 以强化双井可见性，并用 inset 放大底部以展示势阱结构
- 风格统一到 Fig2–5：Times New Roman、无网格、线宽字号一致、PDF 导出

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig1b_potential.py
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402
from src.plot_style import FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig1bConfig:
    n_m: float = 10.0
    n_w: float = 5.0
    phi: float = 0.54
    theta: float = 0.46
    k: int = 50
    u: float = 1.0
    q_lim: float = 2.0
    y_min: float = -0.03
    zoom_q_lim: float = 0.6
    zoom_v_lim: Tuple[float, float] = (-0.02, 0.06)
    fig_size: Tuple[float, float] = FIGSIZE_HALF


def main() -> None:
    cfg = Fig1bConfig()
    chi = theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=int(cfg.k))
    rc = float(theory.calculate_rc(n_m=cfg.n_m, n_w=cfg.n_w, chi=chi))

    r_stable = max(0.0, rc - 0.12)
    r_critical = rc
    r_polar = min(1.0, rc + 0.12)
    r_extreme = 1.0

    q = np.linspace(-float(cfg.q_lim), float(cfg.q_lim), 601)
    alpha_s, _ = theory.get_gl_params(r_stable, rc=rc, u=cfg.u)
    alpha_c, _ = theory.get_gl_params(r_critical, rc=rc, u=cfg.u)
    alpha_p, _ = theory.get_gl_params(r_polar, rc=rc, u=cfg.u)
    alpha_x, _ = theory.get_gl_params(r_extreme, rc=rc, u=cfg.u)

    v_s = theory.potential_energy(q, alpha=float(alpha_s), u=float(cfg.u))
    v_c = theory.potential_energy(q, alpha=float(alpha_c), u=float(cfg.u))
    v_p = theory.potential_energy(q, alpha=float(alpha_p), u=float(cfg.u))
    v_x = theory.potential_energy(q, alpha=float(alpha_x), u=float(cfg.u))

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    ax.plot(q, v_s, color="#0072B2", lw=2.4, label=rf"$r={r_stable:.2f}$")
    ax.plot(q, v_c, color="#E69F00", lw=2.4, label=rf"$r=r_c={r_critical:.2f}$")
    ax.plot(q, v_p, color="#D55E00", lw=2.4, label=rf"$r={r_polar:.2f}$")
    ax.plot(q, v_x, color="#CC79A7", lw=2.4, ls="--", label=r"$r=1.00$")

    ax.set_xlabel(r"Polarization $q$")
    ax.set_ylabel(r"$V_{\mathrm{eff}}(q)$")  # 缩短，详细说明见 caption
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(-float(cfg.q_lim), float(cfg.q_lim))
    ax.set_ylim(float(cfg.y_min), float(np.max(v_s)))

    # 放大底部势阱：缩小 inset，并向中间移动，减少对主图曲线的遮挡
    axins = inset_axes(ax, width="36%", height="30%", loc="upper center", borderpad=1.2)
    axins.plot(q, v_s, color="#0072B2", lw=1.9)
    axins.plot(q, v_c, color="#E69F00", lw=1.9)
    axins.plot(q, v_p, color="#D55E00", lw=1.9)
    axins.plot(q, v_x, color="#CC79A7", lw=1.9, ls="--")
    axins.set_xlim(-float(cfg.zoom_q_lim), float(cfg.zoom_q_lim))
    axins.set_ylim(float(cfg.zoom_v_lim[0]), float(cfg.zoom_v_lim[1]))
    axins.tick_params(direction="in", top=True, right=True, labelsize=7)
    axins.set_xticks([-0.5, 0.0, 0.5])
    axins.set_yticks([-0.02, 0.0, 0.05])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        # 再下移一点点，并增加下边距避免被裁切
        bbox_to_anchor=(0.56, -0.015),
        frameon=False,
        ncol=2,
        handlelength=1.6,
        columnspacing=0.9,
        handletextpad=0.5,
    )

    add_panel_label(ax, "b")
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.38, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig1b_potential.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig1" / "fig1b_potential_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"chi={chi:.3f} rc={rc:.3f}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
