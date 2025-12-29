"""
Fig 2c：Activity dynamics（symmetric vs activity-coupled asymmetric）——论文版式统一导出 PDF。

目标（对应 reviewer 指出的问题）：
- 旧版 fig2c_activity_* 为 PNG，字体/网格/legend 风格与 Fig2a/b 不一致。
- 这里重新用统一的 paper style 输出矢量 PDF，并避免图例遮挡数据。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig2c_activity.py
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
from src.plot_style import add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig2cConfig:
    sym_path: Path = ROOT / "outputs" / "data" / "rq_a_scan_sym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    asym_path: Path = ROOT / "outputs" / "data" / "rq_a_scan_asym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    phi: float = 0.54
    theta: float = 0.46
    n_m: float = 10.0
    n_w: float = 5.0
    k_avg: int = 50
    fig_size: Tuple[float, float] = (6.5, 2.45)


def _load_activity(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    r = d["r_scan"].astype(float, copy=False)
    a = d["a_mean"].astype(float, copy=False)
    return r, a


def main() -> None:
    cfg = Fig2cConfig()
    if not cfg.sym_path.exists():
        raise FileNotFoundError(f"未找到 activity(sym) 缓存：{cfg.sym_path}")
    if not cfg.asym_path.exists():
        raise FileNotFoundError(f"未找到 activity(asym) 缓存：{cfg.asym_path}")

    r_sym, a_sym = _load_activity(cfg.sym_path)
    r_asym, a_asym = _load_activity(cfg.asym_path)

    chi = float(theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=int(cfg.k_avg)))
    rc = float(theory.calculate_rc(n_m=float(cfg.n_m), n_w=float(cfg.n_w), chi=chi))

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=cfg.fig_size, sharey=True)

    activity_color = "#009E73"  # Okabe–Ito bluish green
    rc_style = dict(color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    for ax, r, a, title in [
        (axes[0], r_sym, a_sym, "Symmetric"),
        (axes[1], r_asym, a_asym, "Activity-coupled"),
    ]:
        ax.plot(r, a, color=activity_color, marker="s", linewidth=2.4, markersize=4.0, zorder=3)
        ax.axvline(rc, **rc_style, zorder=1)
        ax.set_title(title)  # 使用全局 titlesize (11pt)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel(r"Activity $a$")
    axes[0].set_xlabel(r"Control parameter $r$")
    axes[1].set_xlabel(r"Control parameter $r$")

    # Panel label for the whole subfigure block
    add_panel_label(axes[0], "c", dx=-55.0)

    # 用图内最小注释替代 legend（避免遮挡 + 省空间）
    for ax in axes:
        ax.text(
            rc + 0.01,
            0.92,
            r"$r_c$",
            transform=ax.get_xaxis_transform(),
            color="gray",
            fontsize=float(mpl.rcParams.get("legend.fontsize", 9.0)),
        )

    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.26, top=0.86, wspace=0.18)

    out_pdf = ROOT / "Essay" / "figures" / "fig2c_activity.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig2" / "fig2c_activity_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
