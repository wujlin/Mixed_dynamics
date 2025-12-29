"""
Supplementary S3：Critical slowing down（补充图统一风格导出 PDF）。

本脚本只读已有缓存（outputs/data/csd_*.npz），不重跑仿真。

输出：
  - Essay/figures_supp/s3_ews_ac_var.pdf
  - Essay/figures_supp/s3_abm_multilag_ac.pdf
以及对应预览 PNG（outputs/figs/supp/）。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_supp_s3_csd_extra.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import OKABE_ITO, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class S3Paths:
    sde_path: Path = ROOT / "outputs" / "data" / "csd_sde_r_q_stats.npz"
    abm_path: Path = (
        ROOT
        / "outputs"
        / "data"
        / "csd_abm_wm_sym_phi54_theta46_nm10_nw5_k50_N1000_u10_steps20000_ri1_burn50_win5000_seeds64_r26_v1.npz"
    )


def _out_dirs() -> Dict[str, Path]:
    out_pdf = ROOT / "Essay" / "figures_supp"
    out_png = ROOT / "outputs" / "figs" / "supp"
    out_pdf.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return {"pdf": out_pdf, "png": out_png}


def _bootstrap_ci_mean(
    rng: np.random.Generator,
    samples: np.ndarray,
    *,
    n_boot: int,
    ci_level: float = 0.95,
) -> Tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    n = int(samples.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(samples[0])
        return v, v
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = samples[idx].mean(axis=1)
    alpha = 1.0 - float(ci_level)
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def fig_ews_ac_var(paths: S3Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.sde_path.exists():
        raise FileNotFoundError(f"未找到 SDE 缓存：{paths.sde_path}")
    if not paths.abm_path.exists():
        raise FileNotFoundError(f"未找到 ABM 缓存：{paths.abm_path}")

    sde = np.load(paths.sde_path, allow_pickle=False)
    r_sde = sde["r_vals"].astype(float, copy=False)
    rc = float(sde["rc"])
    ac_sde_samples = sde["ac_samples"].astype(float, copy=False)
    var_sde_samples = sde["var_samples"].astype(float, copy=False)
    ac_sde_mean = np.nanmean(ac_sde_samples, axis=1)
    var_sde_mean = np.nanmean(var_sde_samples, axis=1)

    abm = np.load(paths.abm_path, allow_pickle=False)
    r_abm = abm["r_vals"].astype(float, copy=False)
    lags_sweeps = abm["lags_sweeps"].astype(float, copy=False)
    lag_i = int(np.argmin(np.abs(lags_sweeps - 1.0)))  # 对齐到 1 sweep
    ac_abm_samples = abm["ac"][:, :, lag_i].astype(float, copy=False)  # (nR, nSeeds)
    var_abm_samples = abm["var"].astype(float, copy=False)  # (nR, nSeeds)
    ac_abm_mean = np.nanmean(ac_abm_samples, axis=1)
    var_abm_mean = np.nanmean(var_abm_samples, axis=1)

    rng = np.random.default_rng(0)
    n_boot = 2000
    ci_sde_ac = np.zeros((r_sde.size, 2), dtype=float)
    ci_sde_var = np.zeros((r_sde.size, 2), dtype=float)
    for i in range(r_sde.size):
        ci_sde_ac[i, 0], ci_sde_ac[i, 1] = _bootstrap_ci_mean(rng, ac_sde_samples[i], n_boot=n_boot)
        ci_sde_var[i, 0], ci_sde_var[i, 1] = _bootstrap_ci_mean(rng, var_sde_samples[i], n_boot=n_boot)
    ci_abm_ac = np.zeros((r_abm.size, 2), dtype=float)
    ci_abm_var = np.zeros((r_abm.size, 2), dtype=float)
    for i in range(r_abm.size):
        ci_abm_ac[i, 0], ci_abm_ac[i, 1] = _bootstrap_ci_mean(rng, ac_abm_samples[i], n_boot=n_boot)
        ci_abm_var[i, 0], ci_abm_var[i, 1] = _bootstrap_ci_mean(rng, var_abm_samples[i], n_boot=n_boot)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    c_sde = OKABE_ITO["blue"]
    c_abm = OKABE_ITO["vermillion"]
    rc_style = dict(color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8)

    # (a) AC1
    ax = axes[0]
    ax.axvline(rc, **rc_style, zorder=1)
    ax.fill_between(r_sde, ci_sde_ac[:, 0], ci_sde_ac[:, 1], color=c_sde, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(r_sde, ac_sde_mean, color=c_sde, linewidth=2.2, label="SDE", zorder=3)
    ax.fill_between(r_abm, ci_abm_ac[:, 0], ci_abm_ac[:, 1], color=c_abm, alpha=0.16, linewidth=0, zorder=0)
    ax.plot(r_abm, ac_abm_mean, color=c_abm, linewidth=2.2, linestyle="--", label="ABM", zorder=3)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Autocorrelation (lag $\approx$ 1 sweep)")
    ax.set_xlim(0.0, min(1.0, rc + 0.01))
    ax.set_ylim(-0.10, 1.02)
    ax.tick_params(direction="in", top=True, right=True)

    # (b) Variance
    ax = axes[1]
    ax.axvline(rc, **rc_style, zorder=1)
    ax.fill_between(r_sde, ci_sde_var[:, 0], ci_sde_var[:, 1], color=c_sde, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(r_sde, var_sde_mean, color=c_sde, linewidth=2.2, zorder=3)
    ax.fill_between(r_abm, ci_abm_var[:, 0], ci_abm_var[:, 1], color=c_abm, alpha=0.16, linewidth=0, zorder=0)
    ax.plot(r_abm, var_abm_mean, color=c_abm, linewidth=2.2, linestyle="--", zorder=3)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Variance")
    ax.set_xlim(0.0, min(1.0, rc + 0.01))
    ax.tick_params(direction="in", top=True, right=True)

    handles, labels = axes[0].get_legend_handles_labels()
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
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.96, wspace=0.32)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_abm_multilag_ac(paths: S3Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.abm_path.exists():
        raise FileNotFoundError(f"未找到 ABM 缓存：{paths.abm_path}")
    d = np.load(paths.abm_path, allow_pickle=False)
    r = d["r_vals"].astype(float, copy=False)
    rc = float(d["rc"])
    lags = d["lags_sweeps"].astype(float, copy=False)
    ac = d["ac"].astype(float, copy=False)  # (nR, nSeeds, nLag)

    rng = np.random.default_rng(0)
    n_boot = 2000

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.0))

    ax.axvline(rc, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # 颜色：固定 4 条曲线
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"], OKABE_ITO["orange"]]
    for j, lag in enumerate(lags):
        samples = ac[:, :, j]
        mean = np.nanmean(samples, axis=1)
        ci = np.zeros((r.size, 2), dtype=float)
        for i in range(r.size):
            ci[i, 0], ci[i, 1] = _bootstrap_ci_mean(rng, samples[i], n_boot=n_boot)
        c = colors[j % len(colors)]
        ax.fill_between(r, ci[:, 0], ci[:, 1], color=c, alpha=0.12, linewidth=0, zorder=0)
        ax.plot(r, mean, color=c, linewidth=2.2, label=f"lag={lag:g} sweep", zorder=3)

    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Autocorrelation")
    ax.set_xlim(0.0, min(1.0, rc + 0.01))
    ax.set_ylim(-0.10, 1.02)
    ax.tick_params(direction="in", top=True, right=True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=2,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.36, top=0.96)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    paths = S3Paths()
    out = _out_dirs()

    fig_ews_ac_var(
        paths,
        out_pdf=out["pdf"] / "s3_ews_ac_var.pdf",
        out_png=out["png"] / "s3_ews_ac_var.png",
    )
    fig_abm_multilag_ac(
        paths,
        out_pdf=out["pdf"] / "s3_abm_multilag_ac.pdf",
        out_png=out["png"] / "s3_abm_multilag_ac.png",
    )

    print("[supp:S3] done")


if __name__ == "__main__":
    main()

