"""
Fig 4b：信息密度 k 的效应（theory vs ABM）——ABM 估计的 r_c 必须给出 95% CI。

设计要点（PI 指示）：
- 统一 Fig2/3 风格（Times New Roman、无网格、线宽字号一致、PDF）。
- 若使用双 y 轴：颜色区分清晰，legend 极简。
- 理论参考线（如 baseline r_c）不用；这里只展示 r_c(k) 的理论曲线与 ABM 估计。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig4b_k_effect.py
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
class Fig4bConfig:
    data_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_k_sweep_abm_phi54_theta46_nm10_nw5_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_k6_v1.npz"
    )
    fig_size: Tuple[float, float] = FIGSIZE_HALF


def main() -> None:
    cfg = Fig4bConfig()
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"未找到 k-sweep 缓存：{cfg.data_path}")

    d = np.load(cfg.data_path, allow_pickle=False)
    k_list = d["k_list"].astype(int, copy=False)
    chi_theory = d["chi_theory"].astype(float, copy=False)
    rc_theory = d["rc_theory"].astype(float, copy=False)
    rc_est = d["rc_est"].astype(float, copy=False)
    rc_lo = d["rc_ci_low"].astype(float, copy=False)
    rc_hi = d["rc_ci_high"].astype(float, copy=False)

    apply_paper_style()
    fig, ax_chi = plt.subplots(figsize=cfg.fig_size)
    ax_rc = ax_chi.twinx()

    # 配色（Okabe–Ito）
    chi_color = "#009E73"  # bluish green
    rc_color = "#0072B2"  # blue
    abm_color = "#D55E00"  # vermillion

    # χ(k)：左轴（仅理论）
    ax_chi.plot(k_list, chi_theory, color=chi_color, marker="o", zorder=3)

    # r_c(k)：右轴（理论 + ABM 95% CI）
    ax_rc.plot(k_list, rc_theory, color=rc_color, linestyle="--", marker="s", label="Theory", zorder=2)
    yerr = np.vstack([rc_est - rc_lo, rc_hi - rc_est])
    ax_rc.errorbar(
        k_list,
        rc_est,
        yerr=yerr,
        fmt="^",
        color=abm_color,
        ecolor=abm_color,
        elinewidth=1.4,
        capsize=3.0,
        markerfacecolor=abm_color,
        markeredgecolor=abm_color,
        label="ABM (95% CI)",
        zorder=4,
    )

    ax_chi.set_xlabel(r"Sample size $k$")
    ax_chi.set_ylabel(r"$\chi$")
    ax_rc.set_ylabel(r"$r_c$")
    ax_chi.tick_params(direction="in", top=True)
    ax_rc.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax_chi, "b")

    handles, labels = ax_rc.get_legend_handles_labels()
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

    # twin y-axis 需要给右侧 ylabel 预留空间，避免被裁剪
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.34, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig4b_k_effect.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig4" / "fig4b_k_effect_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
