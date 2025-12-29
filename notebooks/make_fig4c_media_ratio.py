"""
Fig 4c：媒体生态 n_w/n_m 对临界点 r_c 的影响（theory vs ABM）——ABM 95% CI 误差棒。

PI 指示：
- 理论参考线（baseline ratio=0.5）用灰色点线，不进 legend，caption 说明。
- 无网格、Times、PDF。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig4c_media_ratio.py
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

from src.plot_style import FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig4cConfig:
    data_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_ratio_sweep_abm_phi54_theta46_k50_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_ratio40_v1.npz"
    )
    fig_size: Tuple[float, float] = FIGSIZE_HALF
    baseline_ratio: float = 0.5


def main() -> None:
    cfg = Fig4cConfig()
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"未找到 ratio-sweep 缓存：{cfg.data_path}")

    d = np.load(cfg.data_path, allow_pickle=False)
    ratio = d["ratio_list"].astype(float, copy=False)
    rc_theory = d["rc_theory"].astype(float, copy=False)
    rc_est = d["rc_est"].astype(float, copy=False)
    rc_lo = d["rc_ci_low"].astype(float, copy=False)
    rc_hi = d["rc_ci_high"].astype(float, copy=False)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    # baseline ratio 参考线（不进 legend）
    ax.axvline(cfg.baseline_ratio, color="gray", linestyle=":", alpha=0.6, linewidth=1.2, zorder=1)

    theory_color = "#0072B2"
    abm_color = "#D55E00"

    ax.plot(ratio, rc_theory, color=theory_color, label="Theory", zorder=2)
    yerr = np.vstack([rc_est - rc_lo, rc_hi - rc_est])
    ax.errorbar(
        ratio,
        rc_est,
        yerr=yerr,
        fmt="o",
        color=abm_color,
        ecolor=abm_color,
        elinewidth=1.4,
        capsize=3.0,
        markerfacecolor="white",
        markeredgecolor=abm_color,
        markeredgewidth=1.2,
        label="ABM (95% CI)",
        zorder=3,
    )

    ax.set_xlabel(r"$n_w/n_m$")
    ax.set_ylabel(r"$r_c$")
    ax.set_xlim(float(np.min(ratio)), float(np.max(ratio)))
    ax.set_ylim(0.40, 0.96)
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "c")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.34, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig4c_media_ratio.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig4" / "fig4c_media_ratio_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
