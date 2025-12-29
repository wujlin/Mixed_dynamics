"""
Fig 1c：对称/非对称两种机制下的 (q, a) 结构（方向-强度分解）。

目的：
- 将“极化方向 q”和“系统强度/活跃度 a”作为理论核心变量显式可视化；
- 对比对称验证域（pitchfork 的二阶相变）与现实非对称域（更平滑的 crossover）；
- 风格统一到 Fig2–5：Times New Roman（优先）/STIX（兜底）、无网格、线宽字号一致、PDF 导出。

数据来源：
- 使用已预计算的 r–q–a 扫描（ABM，seed 平均稳态）：
  outputs/data/rq_a_scan_sym_*.npz
  outputs/data/rq_a_scan_asym_*.npz

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig1c_sym_asym_qa.py
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
from src.plot_style import FIGSIZE_FULL, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig1cConfig:
    # 读取已有扫描数据（与 baseline 参数一致）
    sym_npz: Path = ROOT
    asym_npz: Path = ROOT

    # 理论参数（用于计算 rc 参考线）
    n_m: float = 10.0
    n_w: float = 5.0
    phi: float = 0.54
    theta: float = 0.46
    k: int = 50

    fig_size: Tuple[float, float] = (FIGSIZE_FULL[0], 3.3)


def _load_scan(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到扫描数据：{path}")
    data = np.load(path)
    required = {"r_scan", "abs_mean", "a_mean"}
    missing = required.difference(set(data.files))
    if missing:
        raise KeyError(f"{path} 缺少字段：{sorted(missing)}，现有：{data.files}")
    return {k: np.asarray(data[k], dtype=float) for k in data.files}


def main() -> None:
    cfg = Fig1cConfig(
        sym_npz=ROOT
        / "outputs"
        / "data"
        / "rq_a_scan_sym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz",
        asym_npz=ROOT
        / "outputs"
        / "data"
        / "rq_a_scan_asym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz",
    )

    sym = _load_scan(cfg.sym_npz)
    asym = _load_scan(cfg.asym_npz)

    # 理论参考：对称域解析 rc（k 与 sample_n 一致）
    chi = float(theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=int(cfg.k)))
    rc = float(theory.calculate_rc(n_m=cfg.n_m, n_w=cfg.n_w, chi=chi))

    apply_paper_style()
    fig, (ax_q, ax_a) = plt.subplots(1, 2, figsize=cfg.fig_size, sharex=True)

    # 颜色：与 Fig2–5 一致的蓝/橙搭配
    c_sym = "#0072B2"  # Okabe–Ito blue
    c_asym = "#D55E00"  # Okabe–Ito vermillion

    # |q|（用网络模拟的 |Q| 近似）：对称 vs 非对称
    ax_q.plot(sym["r_scan"], sym["abs_mean"], color=c_sym, lw=2.6, label="symmetric")
    ax_q.plot(
        asym["r_scan"],
        asym["abs_mean"],
        color=c_asym,
        lw=2.6,
        linestyle="--",
        label="asymmetric",
    )
    ax_q.axvline(rc, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_q.set_xlabel(r"Control parameter $r$")
    ax_q.set_ylabel(r"$|q|$")
    ax_q.set_xlim(0.0, 1.0)
    ax_q.set_ylim(-0.02, 1.02)
    ax_q.legend(frameon=True, framealpha=0.85, facecolor="white", edgecolor="none", loc="upper left")
    ax_q.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax_q, "c")

    # a：对称域近似阶跃（临界后“升温”），非对称域更平滑上升
    ax_a.plot(sym["r_scan"], sym["a_mean"], color=c_sym, lw=2.6)
    ax_a.plot(asym["r_scan"], asym["a_mean"], color=c_asym, lw=2.6, linestyle="--")
    ax_a.axvline(rc, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_a.set_xlabel(r"Control parameter $r$")
    ax_a.set_ylabel(r"Activity $a$")
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.64, 1.02)
    ax_a.tick_params(direction="in", top=True, right=True)

    # 并排子图需要更大左侧边距以容纳 panel 标签
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.26, top=0.96, wspace=0.35)

    out_pdf = ROOT / "Essay" / "figures" / "fig1c_sym_asym_qa.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig1" / "fig1c_sym_asym_qa_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"chi={chi:.3f} rc={rc:.3f}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
