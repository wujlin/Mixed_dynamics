"""
Fig 4a：参数景观（方案 B）——直接绘制理论临界点 r_c 的热图，并 mask 无相变区域。

设计要点（PI 决策 + 版式约束）：
- 单面板（半栏宽可读），不再在图内嵌套子图。
- r_c 是最终物理量；无相变区域（chi<=2）用浅米色填充；无效域（phi<=theta）留白并画边界线。
- baseline 参数点 (phi=0.54, theta=0.46) 必须标注。
- “No transition” 与 baseline 仅在 caption 中说明：热图类图表不放 legend，避免破坏色带连续性。
- 风格统一到 Fig2/3：Times New Roman、无网格、线宽字号一致、PDF 导出。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig4a_rc_landscape.py
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
class Fig4aConfig:
    data_path: Path = ROOT / "outputs" / "data" / "note4_sensitivity_data.npz"
    n_m: float = 10.0
    n_w: float = 5.0
    baseline_phi: float = 0.54
    baseline_theta: float = 0.46
    fig_size: Tuple[float, float] = FIGSIZE_HALF
    no_transition_color: str = "#FFF2E6"


def main() -> None:
    cfg = Fig4aConfig()
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"未找到 Note4 缓存：{cfg.data_path}")

    data = np.load(cfg.data_path, allow_pickle=False)
    phi = data["phi_range"].astype(float, copy=False)
    theta = data["theta_range"].astype(float, copy=False)
    chi_map = data["chi_map"].astype(float, copy=False)  # (phi, theta)，无效区域已为 NaN

    rc_map = theory.calculate_rc(n_m=float(cfg.n_m), n_w=float(cfg.n_w), chi=chi_map)
    # 有效域：阈值定义有效 & 存在相变（chi > 2）
    valid_transition = np.isfinite(chi_map) & (chi_map > 2.0)
    no_transition = np.isfinite(chi_map) & (chi_map <= 2.0)
    rc_valid = np.ma.masked_where(~valid_transition, rc_map)

    # colorbar 范围：理论上 rc ∈ [n_m/(n_m+n_w), 1)，给一点 padding
    vmin = float(cfg.n_m / (cfg.n_m + cfg.n_w)) - 0.02
    vmax = 1.0

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    # 主热图：用更浅的连续色带（避免过深的蓝色），无效域留白
    base_cmap = plt.get_cmap("YlGnBu")
    cmap = mpl.colors.ListedColormap(base_cmap(np.linspace(0.05, 0.85, 256)))
    cmap.set_bad(color="white")

    mesh = ax.pcolormesh(theta, phi, rc_valid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # 次层：无相变区域（chi<=2）用浅米色覆盖；无效域（phi<=theta）保持留白
    no_cmap = mpl.colors.ListedColormap([cfg.no_transition_color])
    no_cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    no_mask = np.ma.masked_where(~no_transition, np.ones_like(rc_map, dtype=float))
    ax.pcolormesh(theta, phi, no_mask, shading="auto", cmap=no_cmap, vmin=0.0, vmax=1.0)

    # 无效域边界：phi = theta（用淡灰点线，不进 legend）
    diag_min = float(max(np.min(theta), np.min(phi)))
    diag_max = float(min(np.max(theta), np.max(phi)))
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        color="gray",
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
        zorder=4,
    )

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$r_c$")

    # baseline 标注
    ax.plot(
        [cfg.baseline_theta],
        [cfg.baseline_phi],
        marker="*",
        markersize=12,
        markerfacecolor="black",
        markeredgecolor="white",
        markeredgewidth=0.9,
        linestyle="none",
        zorder=5,
    )

    ax.set_xlabel(r"Low threshold $\theta$")
    ax.set_ylabel(r"High threshold $\phi$")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "a")

    fig.tight_layout()

    out_pdf = ROOT / "Essay" / "figures" / "fig4a_chi_rc_landscape.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig4" / "fig4a_rc_landscape_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
