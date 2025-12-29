"""
Fig 4d：局部耦合 β 对有效临界点 r_c 的影响（ABM max-slope）——两种 local_mode 对比 + 95% CI。

PI 指示：
- 理论参考线（beta=0 的 r_c）用灰色点线，不进 legend，caption 说明。
- ABM 点必须带 95% CI。
- Times / 无网格 / PDF。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig4d_beta_effect.py
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
class Fig4dConfig:
    high_only_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_beta_sweep_abm_phi54_theta46_nm10_nw5_k50_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_b5_lmhigh_only_v1.npz"
    )
    symmetric_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_beta_sweep_abm_phi54_theta46_nm10_nw5_k50_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_b5_lmsymmetric_v1.npz"
    )
    fig_size: Tuple[float, float] = FIGSIZE_HALF


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    d = np.load(path, allow_pickle=False)
    betas = d["betas"].astype(float, copy=False)
    rc = d["rc_est"].astype(float, copy=False)
    lo = d["rc_ci_low"].astype(float, copy=False)
    hi = d["rc_ci_high"].astype(float, copy=False)
    rc_ref = float(d["rc_ref"])
    return betas, rc, lo, hi, rc_ref


def main() -> None:
    cfg = Fig4dConfig()
    if not cfg.high_only_path.exists():
        raise FileNotFoundError(f"未找到 beta-sweep(high_only) 缓存：{cfg.high_only_path}")
    if not cfg.symmetric_path.exists():
        raise FileNotFoundError(f"未找到 beta-sweep(symmetric) 缓存：{cfg.symmetric_path}")

    betas_ho, rc_ho, lo_ho, hi_ho, rc_ref = _load(cfg.high_only_path)
    betas_sy, rc_sy, lo_sy, hi_sy, rc_ref2 = _load(cfg.symmetric_path)
    if abs(rc_ref - rc_ref2) > 1e-6:
        raise RuntimeError("两份 beta-sweep 缓存的 rc_ref 不一致")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    # 理论参考线：beta=0 的 r_c（不进 legend）
    ax.axhline(rc_ref, color="gray", linestyle=":", alpha=0.6, linewidth=1.2, zorder=1)

    # 配色（Okabe–Ito）
    c1 = "#0072B2"  # blue
    c2 = "#009E73"  # bluish green

    yerr_ho = np.vstack([rc_ho - lo_ho, hi_ho - rc_ho])
    ax.errorbar(
        betas_ho,
        rc_ho,
        yerr=yerr_ho,
        fmt="o-",
        color=c1,
        ecolor=c1,
        elinewidth=1.4,
        capsize=3.0,
        label="High-only local coupling",
        zorder=3,
    )

    yerr_sy = np.vstack([rc_sy - lo_sy, hi_sy - rc_sy])
    ax.errorbar(
        betas_sy,
        rc_sy,
        yerr=yerr_sy,
        fmt="s-",
        color=c2,
        ecolor=c2,
        elinewidth=1.4,
        capsize=3.0,
        label="Symmetric local coupling",
        zorder=3,
    )

    ax.set_xlabel(r"Local coupling $\beta$")
    ax.set_ylabel(r"Estimated $r_c$")
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(0.10, 0.80)
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "d")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=1,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.36, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig4d_beta_effect.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig4" / "fig4d_beta_effect_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
