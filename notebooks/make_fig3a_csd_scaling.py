"""
Fig 3a：Critical slowing down 的标度律（单面板 log-log 拟合）——与 Fig2 同风格输出 PDF。

决策（来自 review）：
- Fig3a 只保留 log-log 拟合面板（避免半栏内再嵌子图导致不可读）。
- 图内需显式标注临界指数（审稿人最关心）。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig3a_csd_scaling.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Fig3aConfig:
    data_path: Path = ROOT / "outputs" / "data" / "csd_sde_r_q_stats.npz"
    fig_size: Tuple[float, float] = (7.2, 4.4)
    n_bootstrap: int = 2000
    ci_level: float = 0.95


def _style_rcparams() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    times_paths = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    if any(p.exists() for p in times_paths):
        for p in times_paths:
            if p.exists():
                fm.fontManager.addfont(str(p))
        font_family = "Times New Roman"
        serif_fallback = ["Times New Roman"]
    else:
        font_family = "STIXGeneral"
        serif_fallback = ["STIXGeneral", "DejaVu Serif"]

    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": serif_fallback,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.grid": False,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def _fit_power_law(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    拟合 y ≈ A * x^slope（log-log 线性回归），返回 (slope, slope_err, A)。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("x/y 长度不一致")
    if x.size < 3:
        raise ValueError("拟合点数不足")

    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)

    # 线性回归标准误（无额外依赖，KISS）
    pred = slope * lx + intercept
    resid = ly - pred
    n = float(lx.size)
    sse = float(np.sum(resid**2))
    sxx = float(np.sum((lx - float(np.mean(lx))) ** 2))
    slope_err = float(np.sqrt((sse / (n - 2.0)) / sxx)) if sxx > 0 else float("nan")

    A = float(np.exp(intercept))
    return float(slope), slope_err, A


def main() -> None:
    cfg = Fig3aConfig()
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"未找到 CSD 缓存：{cfg.data_path}")

    data = np.load(cfg.data_path, allow_pickle=False)
    rc = float(data["rc"])
    r_det = data["r_vals_det"].astype(float, copy=False)
    tau = data["tau_measured"].astype(float, copy=False)

    x = rc - r_det
    # 去掉非正值/饱和段（tau_measured 在 very-near-rc 会被最大滞后截断）
    tau_cap = float(np.max(tau))
    mask = (x > 0.0) & (tau > 0.0) & (tau < 0.999 * tau_cap)
    x_fit = x[mask]
    tau_fit = tau[mask]
    if x_fit.size < 6:
        raise RuntimeError("有效拟合点过少：请检查 tau_measured 是否被截断或数据路径是否正确")

    slope, slope_err, A = _fit_power_law(x_fit, tau_fit)
    gamma = -slope
    gamma_err = slope_err

    _style_rcparams()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    # 数据点
    data_color = "#0072B2"  # Okabe–Ito blue
    ax.plot(x_fit, tau_fit, marker="o", linestyle="none", color=data_color, label="Deterministic ODE", zorder=3)

    # 拟合线
    x_line = np.logspace(np.log10(float(np.min(x_fit))), np.log10(float(np.max(x_fit))), 256)
    y_line = A * (x_line**slope)
    ax.plot(
        x_line,
        y_line,
        color="#D55E00",  # Okabe–Ito vermillion
        linewidth=2.6,
        label=rf"Fit: $\gamma={gamma:.3f}\pm{gamma_err:.3f}$",
        zorder=2,
    )

    # 参考斜率：gamma=1（仅作形状标尺，幅值按拟合中点对齐）
    x0 = float(np.median(x_fit))
    y0 = float(A * (x0**slope))
    y_ref = y0 * (x_line / x0) ** (-1.0)
    ax.plot(x_line, y_ref, color="black", linestyle=":", linewidth=2.0, label=r"Theory: $\gamma=1$", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_c - r$")
    ax.set_ylabel(r"Relaxation time $\tau$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    out_pdf = ROOT / "Essay" / "figures" / "fig3a_csd_scaling.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig3" / "fig3a_csd_scaling_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()

