"""
Fig 3b：CSD 对比（SDE vs ABM，time-aligned short-lag autocorrelation）——与 Fig2 同风格输出 PDF。

决策（来自 review）：
- ABM 扫描必须逼近 r_c（数据来自 scripts/run_csd_abm.py 的缓存，已覆盖至 r≈0.748）。
- 统一风格：Times New Roman / 无网格 / 线宽字号与 Fig2 对齐。
- Legend 保持简洁，仅保留 “SDE” 与 “ABM”；时间对齐细节写入论文 caption。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig3b_csd_sde_vs_abm.py
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
class Fig3bConfig:
    sde_path: Path = ROOT / "outputs" / "data" / "csd_sde_r_q_stats.npz"
    abm_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "csd_abm_wm_sym_phi54_theta46_nm10_nw5_k50_N1000_u10_steps20000_ri1_burn50_win5000_seeds64_r26_v1.npz"
    )
    fig_size: Tuple[float, float] = FIGSIZE_HALF
    band_alpha: float = 0.18
    # 取 time-aligned 的短滞后：以 “sweep” 为单位，默认对齐到 1 sweep（update_rate=0.1 时对应 lag_index≈10）
    target_lag_sweeps: float = 1.0
    n_bootstrap: int = 2000
    ci_level: float = 0.95


def _bootstrap_ci_mean(
    rng: np.random.Generator,
    samples: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    n = int(samples.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(samples[0])
        return v, v
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot = samples[idx].mean(axis=1)
    alpha = 1.0 - float(ci_level)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def main() -> None:
    cfg = Fig3bConfig()
    if not cfg.sde_path.exists():
        raise FileNotFoundError(f"未找到 SDE 缓存：{cfg.sde_path}")
    if not cfg.abm_path.exists():
        raise FileNotFoundError(f"未找到 ABM 缓存：{cfg.abm_path}")

    sde = np.load(cfg.sde_path, allow_pickle=False)
    r_sde = sde["r_vals"].astype(float, copy=False)
    ac_sde_samples = sde["ac_samples"].astype(float, copy=False)  # (nR, nTraj)
    ac_sde_mean = ac_sde_samples.mean(axis=1)
    rc = float(sde["rc"])

    abm = np.load(cfg.abm_path, allow_pickle=False)
    r_abm = abm["r_vals"].astype(float, copy=False)
    lags_sweeps = abm["lags_sweeps"].astype(float, copy=False)
    lag_i = int(np.argmin(np.abs(lags_sweeps - float(cfg.target_lag_sweeps))))
    lag_sweep_used = float(lags_sweeps[lag_i])
    ac_abm_samples = abm["ac"][:, :, lag_i].astype(float, copy=False)  # (nR, nSeeds)
    ac_abm_mean = np.nanmean(ac_abm_samples, axis=1)

    # 95% CI：对每个 r 点，bootstrap 估计均值的不确定性（与 Fig2 的口径一致）
    rng = np.random.default_rng(0)
    sde_ci = np.zeros((r_sde.size, 2), dtype=float)
    for i in range(r_sde.size):
        sde_ci[i, 0], sde_ci[i, 1] = _bootstrap_ci_mean(
            rng,
            ac_sde_samples[i],
            n_bootstrap=cfg.n_bootstrap,
            ci_level=cfg.ci_level,
        )
    abm_ci = np.zeros((r_abm.size, 2), dtype=float)
    for i in range(r_abm.size):
        abm_ci[i, 0], abm_ci[i, 1] = _bootstrap_ci_mean(
            rng,
            ac_abm_samples[i],
            n_bootstrap=cfg.n_bootstrap,
            ci_level=cfg.ci_level,
        )

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    # 参考线：rc（不进 legend）
    ax.axvline(rc, color="gray", linestyle=":", alpha=0.6, linewidth=1.2, zorder=1)
    ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=1)

    # 颜色（Okabe–Ito）
    sde_color = "#0072B2"
    abm_color = "#D55E00"

    # SDE：均值线 + 区间带
    ax.fill_between(r_sde, sde_ci[:, 0], sde_ci[:, 1], color=sde_color, alpha=cfg.band_alpha, linewidth=0, zorder=0)
    ax.plot(r_sde, ac_sde_mean, color=sde_color, label="SDE", zorder=3)

    # ABM：均值线 + 区间带
    ax.fill_between(r_abm, abm_ci[:, 0], abm_ci[:, 1], color=abm_color, alpha=cfg.band_alpha, linewidth=0, zorder=0)
    ax.plot(r_abm, ac_abm_mean, color=abm_color, linestyle="--", label="ABM", zorder=3)

    ax.set_xlim(0.0, min(1.0, rc + 0.01))
    ax.set_ylim(-0.10, 1.02)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Autocorrelation")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "b", dx=-55.0)
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
    # 固定版式边距：与 Fig3a 保持一致，避免并排时视觉尺寸不一致
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.34, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig3b_csd_sde_vs_abm.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig3" / "fig3b_csd_sde_vs_abm_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")
    print(f"Aligned lag (ABM): {lag_sweep_used:g} sweep (lags_sweeps={lags_sweeps.tolist()})")


if __name__ == "__main__":
    main()
