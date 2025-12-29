"""
Fig 2b：Binder cumulant U4(r; N)（有限尺寸交点）——与 Fig2a 同风格输出 PDF。

要求：
- Times New Roman / Times-like 衬线字体
- 无网格、加粗线宽、期刊排版
- 理论 r_c 竖虚线：淡灰点线，不进 legend；caption 说明
- legend 仅包含不同 N（按从小到大）

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig2b_binder_u4.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig2bConfig:
    data_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "finite_size_binder_cross_sym_phi54_theta46_nm10_nw5_k50_N100-2000_initrandom_u10_ri5_steps2000_burn50_seeds8_r41_v4_cmmaxslope.npz"
    )
    xlim: tuple[float, float] = (0.60, 0.91)
    ylim: tuple[float, float] = (-0.12, 0.70)
    ci_alpha: float = 0.18

def main() -> None:
    cfg = Fig2bConfig()
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"未找到 Binder 缓存：{cfg.data_path}\n"
            "请先用 scripts/run_finite_size_binder.py 生成 outputs/data/*.npz"
        )

    data = np.load(cfg.data_path, allow_pickle=False)
    r = data["r_scan"].astype(float, copy=False)
    N_list = data["N_list"].astype(int).tolist()
    binder_mean = data["binder_mean_by_N"].astype(float, copy=False)
    binder_sem = data["binder_sem_by_N"].astype(float, copy=False)
    rc_theory = float(data["rc_theory"])

    # N 从小到大排序，保证 legend 顺序稳定
    order = np.argsort(N_list)
    N_sorted: List[int] = [int(N_list[i]) for i in order]
    binder_mean = binder_mean[order]
    binder_sem = binder_sem[order]

    apply_paper_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)

    # 理论 rc：淡灰点线，不进 legend
    ax.axvline(rc_theory, color="gray", linestyle=":", alpha=0.6, linewidth=1.2, zorder=1)

    # 曲线 + 误差带（SEM）
    for i, N in enumerate(N_sorted):
        line = ax.plot(r, binder_mean[i], linewidth=2.4, label=rf"$N={N}$", zorder=3)[0]
        color = line.get_color()
        ax.fill_between(
            r,
            binder_mean[i] - binder_sem[i],
            binder_mean[i] + binder_sem[i],
            color=color,
            alpha=cfg.ci_alpha,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlim(*cfg.xlim)
    ax.set_ylim(*cfg.ylim)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"$U_4$")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "b")
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib 多列 legend 默认按列填充；这里重排为“按行阅读”为主的顺序（N 从小到大）。
    ncol = 3
    rows = int(np.ceil(len(handles) / ncol)) if handles else 1
    grid: list[list[tuple[object, str] | None]] = [[None for _ in range(ncol)] for _ in range(rows)]
    for i, (h, lab) in enumerate(zip(handles, labels)):
        grid[i // ncol][i % ncol] = (h, lab)
    reordered: list[tuple[object, str]] = []
    for c in range(ncol):
        for r in range(rows):
            item = grid[r][c]
            if item is not None:
                reordered.append(item)
    handles = [h for h, _ in reordered]
    labels = [lab for _, lab in reordered]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=ncol,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    # 固定边距：与 Fig3a/b 对齐，避免并排时“视觉字号”不一致
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.36, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig2b_binder_u4.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig2" / "fig2b_binder_u4_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
