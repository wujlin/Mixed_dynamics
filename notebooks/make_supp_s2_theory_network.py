"""
Supplementary S2：Theory & Network robustness（补充图统一风格导出 PDF）。

本脚本只读已有缓存（outputs/data/*.npz），不重跑仿真。

输出：
  - Essay/figures_supp/s2_signed_vs_abs_q.pdf
  - Essay/figures_supp/s2_tau_ratio.pdf
  - Essay/figures_supp/s2_binder_fss.pdf
  - Essay/figures_supp/s2_susceptibility_fss.pdf
以及对应的预览 PNG（outputs/figs/supp/）。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_supp_s2_theory_network.py
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

from src import theory  # noqa: E402
from src.plot_style import OKABE_ITO, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class S2Paths:
    # q/a 扫描（用于说明 signed mean 在对称系统里会抵消）
    rq_a_sym: Path = ROOT / "outputs" / "data" / "rq_a_scan_sym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    rq_a_asym: Path = ROOT / "outputs" / "data" / "rq_a_scan_asym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    # τ_a/τ_q（用于支撑正文一行结论）
    tau_sweep: Path = ROOT / "outputs" / "data" / "appendix_tau_sweep_k50_N1000_beta0.0_u10_ri1_steps2000_burn20_seeds10_r15_v1.npz"
    # finite-size：Binder 交点（稳健 rc）
    binder_fss: Path = (
        ROOT
        / "outputs"
        / "data"
        / "finite_size_binder_cross_sym_phi54_theta46_nm10_nw5_k50_N100-2000_initrandom_u10_ri5_steps2000_burn50_seeds8_r41_v4_cmmaxslope.npz"
    )
    # finite-size：susceptibility peak（对照：更不稳健）
    chi_peak: Path = (
        ROOT
        / "outputs"
        / "data"
        / "finite_size_chi_peak_sym_phi54_theta46_nm10_nw5_k50_N100-2000_initrandom_u100_ri5_steps1200_burn50_seeds8_r31_v2.npz"
    )


def _out_dirs() -> Dict[str, Path]:
    out_pdf = ROOT / "Essay" / "figures_supp"
    out_png = ROOT / "outputs" / "figs" / "supp"
    out_pdf.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return {"pdf": out_pdf, "png": out_png}


def _compute_rc(*, phi: float = 0.54, theta: float = 0.46, k_avg: int = 50, n_m: float = 10.0, n_w: float = 5.0) -> float:
    chi = float(theory.calculate_chi(phi=float(phi), theta=float(theta), k_avg=int(k_avg)))
    return float(theory.calculate_rc(n_m=float(n_m), n_w=float(n_w), chi=chi))


def fig_signed_vs_abs_q(paths: S2Paths, *, out_pdf: Path, out_png: Path) -> None:
    for p in [paths.rq_a_sym, paths.rq_a_asym]:
        if not p.exists():
            raise FileNotFoundError(f"未找到缓存：{p}")

    sym = np.load(paths.rq_a_sym, allow_pickle=False)
    asym = np.load(paths.rq_a_asym, allow_pickle=False)
    r_sym = sym["r_scan"].astype(float, copy=False)
    signed_sym = sym["signed_mean"].astype(float, copy=False)
    abs_sym = sym["abs_mean"].astype(float, copy=False)
    r_asym = asym["r_scan"].astype(float, copy=False)
    signed_asym = asym["signed_mean"].astype(float, copy=False)
    abs_asym = asym["abs_mean"].astype(float, copy=False)

    rc = _compute_rc()

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)

    c_signed = OKABE_ITO["black"]
    c_abs = OKABE_ITO["blue"]
    rc_style = dict(color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8)

    for ax, r, s, a, title in [
        (axes[0], r_sym, signed_sym, abs_sym, "Symmetric"),
        (axes[1], r_asym, signed_asym, abs_asym, "Activity-coupled"),
    ]:
        ax.axhline(0.0, color=OKABE_ITO["gray"], linewidth=0.9, zorder=1)
        ax.axvline(rc, **rc_style, zorder=1)
        ax.plot(r, s, color=c_signed, linewidth=2.2, label=r"$\langle Q\rangle$ (signed)", zorder=3)
        ax.plot(r, a, color=c_abs, linewidth=2.2, label=r"$|\langle Q\rangle|$", zorder=3)
        ax.plot(r, -a, color=c_abs, linewidth=2.2, alpha=0.35, zorder=2)
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel(r"Polarization $Q$")
    for ax in axes:
        ax.set_xlabel(r"Control parameter $r$")

    # 统一 legend：放在图外下方，避免遮挡
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
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.32, top=0.90, wspace=0.18)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


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


def fig_tau_ratio(paths: S2Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.tau_sweep.exists():
        raise FileNotFoundError(f"未找到缓存：{paths.tau_sweep}")
    d = np.load(paths.tau_sweep, allow_pickle=False)
    r = d["r_scan"].astype(float, copy=False)
    tau_q_asym = d["tau_q_asym"].astype(float, copy=False)
    tau_a_asym = d["tau_a_asym"].astype(float, copy=False)
    tau_q_sym = d["tau_q_sym"].astype(float, copy=False)
    tau_a_sym = d["tau_a_sym"].astype(float, copy=False)

    # ratio：按 seed 先算，再聚合（更符合“同一随机性下两个变量比较”）
    # 防止 tau_q 为 0 或 NaN 引发 inf：统一转为 NaN 后再聚合
    ratio_asym = np.divide(
        tau_a_asym,
        tau_q_asym,
        out=np.full_like(tau_a_asym, np.nan, dtype=float),
        where=np.isfinite(tau_q_asym) & (tau_q_asym > 0),
    )
    ratio_sym = np.divide(
        tau_a_sym,
        tau_q_sym,
        out=np.full_like(tau_a_sym, np.nan, dtype=float),
        where=np.isfinite(tau_q_sym) & (tau_q_sym > 0),
    )

    rng = np.random.default_rng(0)
    n_boot = 2000
    ci_asym = np.zeros((r.size, 2), dtype=float)
    ci_sym = np.zeros((r.size, 2), dtype=float)
    def _nanmean_cols(x: np.ndarray) -> np.ndarray:
        out = np.full(x.shape[1], np.nan, dtype=float)
        for i in range(x.shape[1]):
            col = x[:, i]
            col = col[np.isfinite(col)]
            if col.size:
                out[i] = float(np.mean(col))
        return out

    mean_asym = _nanmean_cols(ratio_asym)
    mean_sym = _nanmean_cols(ratio_sym)
    for i in range(r.size):
        ci_asym[i, 0], ci_asym[i, 1] = _bootstrap_ci_mean(rng, ratio_asym[:, i], n_boot=n_boot)
        ci_sym[i, 0], ci_sym[i, 1] = _bootstrap_ci_mean(rng, ratio_sym[:, i], n_boot=n_boot)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))

    c_asym = OKABE_ITO["vermillion"]
    c_sym = OKABE_ITO["blue"]
    ax.axhline(1.0, color=OKABE_ITO["gray"], linewidth=1.0, zorder=1)
    ax.fill_between(r, ci_sym[:, 0], ci_sym[:, 1], color=c_sym, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(r, mean_sym, color=c_sym, linewidth=2.4, label="Symmetric", zorder=3)
    ax.fill_between(r, ci_asym[:, 0], ci_asym[:, 1], color=c_asym, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(r, mean_asym, color=c_asym, linewidth=2.4, linestyle="--", label="Asymmetric", zorder=3)

    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Relaxation ratio $\tau_a/\tau_q$")
    ax.set_xlim(float(np.min(r)), float(np.max(r)))
    ax.set_ylim(0.0, max(2.0, float(np.nanmax(ci_sym[:, 1]) * 1.05)))
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
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.33, top=0.96)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_binder_fss(paths: S2Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.binder_fss.exists():
        raise FileNotFoundError(f"未找到缓存：{paths.binder_fss}")
    d = np.load(paths.binder_fss, allow_pickle=False)
    r_scan = d["r_scan"].astype(float, copy=False)
    N_list = d["N_list"].astype(int, copy=False)
    u4_mean = d["binder_mean_by_N"].astype(float, copy=False)
    u4_sem = d["binder_sem_by_N"].astype(float, copy=False)
    rc_theory = float(d["rc_theory"])
    pair_mid = d["pair_N_mid"].astype(float, copy=False)
    rc_ci = d["rc_cross_ci"].astype(float, copy=False)  # (3, nPairs)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) U4 curves
    ax = axes[0]
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
    ]
    for i, N in enumerate(N_list):
        c = colors[i % len(colors)]
        ax.plot(r_scan, u4_mean[i], color=c, linewidth=2.0, label=f"N={int(N)}", zorder=3)
        ax.fill_between(r_scan, u4_mean[i] - u4_sem[i], u4_mean[i] + u4_sem[i], color=c, alpha=0.12, linewidth=0, zorder=1)
    ax.axvline(rc_theory, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Binder cumulant $U_4$")
    ax.set_xlim(float(r_scan[0]), float(r_scan[-1]))
    ax.set_ylim(-0.25, 0.80)
    ax.tick_params(direction="in", top=True, right=True)

    # (b) rc crossings with CI
    ax2 = axes[1]
    x = pair_mid
    y = rc_ci[1]
    yerr = np.vstack([y - rc_ci[0], rc_ci[2] - y])
    ax2.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        elinewidth=1.4,
        capsize=3.0,
        zorder=3,
    )
    ax2.axhline(rc_theory, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
    ax2.set_xlabel(r"Pair midpoint $\sqrt{N_i N_{i+1}}$")
    ax2.set_ylabel(r"Estimated $r_c$")
    ax2.tick_params(direction="in", top=True, right=True)
    ax2.set_xlim(float(np.min(x)) * 0.95, float(np.max(x)) * 1.05)
    ax2.set_ylim(float(np.min(rc_ci[0])) - 0.01, float(np.max(rc_ci[2])) + 0.01)

    # legend：只放左图（N 列表），放在图外下方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=3,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.96, wspace=0.35)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_susceptibility_fss(paths: S2Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.chi_peak.exists():
        raise FileNotFoundError(f"未找到缓存：{paths.chi_peak}")
    d = np.load(paths.chi_peak, allow_pickle=False)
    N_list = d["N_list"].astype(int, copy=False)
    r_scan = d["r_scan"].astype(float, copy=False)
    chi_mean = d["chi_mean_by_N"].astype(float, copy=False)
    chi_sem = d["chi_sem_by_N"].astype(float, copy=False)
    rc_peak = d["rc_peak_mean"].astype(float, copy=False)
    rc_peak_std = d["rc_peak_std"].astype(float, copy=False)

    rc_theory = _compute_rc()

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # (a) susceptibility curves
    ax = axes[0]
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
    ]
    for i, N in enumerate(N_list):
        c = colors[i % len(colors)]
        ax.plot(r_scan, chi_mean[i], color=c, linewidth=2.0, label=f"N={int(N)}", zorder=3)
        ax.fill_between(r_scan, chi_mean[i] - chi_sem[i], chi_mean[i] + chi_sem[i], color=c, alpha=0.12, linewidth=0, zorder=1)
    ax.axvline(rc_theory, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8, zorder=2)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Susceptibility $\chi_Q$")
    ax.tick_params(direction="in", top=True, right=True)

    # (b) peak-based rc estimates vs N
    ax2 = axes[1]
    x = N_list.astype(float)
    yerr = rc_peak_std
    ax2.errorbar(
        x,
        rc_peak,
        yerr=yerr,
        fmt="o-",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        elinewidth=1.4,
        capsize=3.0,
        zorder=3,
    )
    ax2.axhline(rc_theory, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"System size $N$")
    ax2.set_ylabel(r"$r_c$ by peak($\chi_Q$)")
    ax2.tick_params(direction="in", top=True, right=True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=3,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.96, wspace=0.35)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    paths = S2Paths()
    out = _out_dirs()

    fig_signed_vs_abs_q(
        paths,
        out_pdf=out["pdf"] / "s2_signed_vs_abs_q.pdf",
        out_png=out["png"] / "s2_signed_vs_abs_q.png",
    )
    fig_tau_ratio(
        paths,
        out_pdf=out["pdf"] / "s2_tau_ratio.pdf",
        out_png=out["png"] / "s2_tau_ratio.png",
    )
    fig_binder_fss(
        paths,
        out_pdf=out["pdf"] / "s2_binder_fss.pdf",
        out_png=out["png"] / "s2_binder_fss.png",
    )
    fig_susceptibility_fss(
        paths,
        out_pdf=out["pdf"] / "s2_susceptibility_fss.pdf",
        out_png=out["png"] / "s2_susceptibility_fss.png",
    )

    print("[supp:S2] done")


if __name__ == "__main__":
    main()
