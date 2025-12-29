"""
Supplementary S4：Parameter landscape（补充图统一风格导出 PDF）。

本脚本只读已有缓存（outputs/data/*.npz），不重跑仿真。

输出：
  - Essay/figures_supp/s4_symmetric_diagonal.pdf
  - Essay/figures_supp/s4_k_effect_split.pdf
  - Essay/figures_supp/s4_k500_finite_size.pdf
  - Essay/figures_supp/s4_beta_branch_bias.pdf
  - Essay/figures_supp/s4_beta_q_a.pdf
以及对应预览 PNG（outputs/figs/supp/）。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_supp_s4_parameter_extra.py
"""

from __future__ import annotations

import math
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
class S4Paths:
    # k-sweep（用于拆分展示 chi(k) 与 rc(k)）
    k_sweep: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_k_sweep_abm_phi54_theta46_nm10_nw5_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_k6_v1.npz"
    )
    # k=500 N-sweep（用于有限尺寸外推）
    k500_pattern: str = "outputs/data/note4_k500_N*_r074-080_steps12000_win400_seeds128_*.npz"
    # beta sweep：两种 local_mode（用于 branch bias 与 q/a 曲线）
    beta_high_only: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_beta_sweep_abm_phi54_theta46_nm10_nw5_k50_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_b5_lmhigh_only_v1.npz"
    )
    beta_symmetric: Path = (
        ROOT
        / "outputs"
        / "data"
        / "note4_beta_sweep_abm_phi54_theta46_nm10_nw5_k50_N400_deg50_er_u10_steps8000_ri10_burn50_win200_seeds128_r201_b5_lmsymmetric_v1.npz"
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


def fig_symmetric_diagonal(*, out_pdf: Path, out_png: Path) -> None:
    # 沿 phi+theta=1 做 1D 切片，用于说明“解析验证域”
    theta = np.linspace(0.10, 0.49, 40)
    phi = 1.0 - theta
    k_avg = 50
    n_m, n_w = 10.0, 5.0
    chi = np.array([theory.calculate_chi(phi=float(p), theta=float(t), k_avg=int(k_avg)) for p, t in zip(phi, theta)], dtype=float)
    rc = theory.calculate_rc(n_m=n_m, n_w=n_w, chi=chi).astype(float, copy=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharex=True)

    axes[0].plot(theta, chi, color=OKABE_ITO["bluish_green"], linewidth=2.2)
    axes[0].axhline(2.0, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8)
    axes[0].set_xlabel(r"Low threshold $\theta$")
    axes[0].set_ylabel(r"Psychological sensitivity $\chi$")
    axes[0].tick_params(direction="in", top=True, right=True)

    axes[1].plot(theta, rc, color=OKABE_ITO["blue"], linewidth=2.2)
    axes[1].set_xlabel(r"Low threshold $\theta$")
    axes[1].set_ylabel(r"Critical point $r_c$")
    axes[1].tick_params(direction="in", top=True, right=True)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.26, top=0.96, wspace=0.35)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_k_effect_split(paths: S4Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.k_sweep.exists():
        raise FileNotFoundError(f"未找到 k-sweep 缓存：{paths.k_sweep}")
    d = np.load(paths.k_sweep, allow_pickle=False)
    k_list = d["k_list"].astype(int, copy=False)
    chi_theory = d["chi_theory"].astype(float, copy=False)
    rc_theory = d["rc_theory"].astype(float, copy=False)
    rc_est = d["rc_est"].astype(float, copy=False)
    rc_lo = d["rc_ci_low"].astype(float, copy=False)
    rc_hi = d["rc_ci_high"].astype(float, copy=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))

    axes[0].plot(k_list, chi_theory, color=OKABE_ITO["bluish_green"], marker="o", linewidth=2.2)
    axes[0].set_xlabel(r"Sample size $k$")
    axes[0].set_ylabel(r"$\chi$")
    axes[0].tick_params(direction="in", top=True, right=True)

    axes[1].plot(k_list, rc_theory, color=OKABE_ITO["blue"], linestyle="--", marker="s", linewidth=2.0, label="Theory")
    yerr = np.vstack([rc_est - rc_lo, rc_hi - rc_est])
    axes[1].errorbar(
        k_list,
        rc_est,
        yerr=yerr,
        fmt="^",
        color=OKABE_ITO["vermillion"],
        ecolor=OKABE_ITO["vermillion"],
        elinewidth=1.4,
        capsize=3.0,
        label="ABM (95% CI)",
    )
    axes[1].set_xlabel(r"Sample size $k$")
    axes[1].set_ylabel(r"$r_c$")
    axes[1].tick_params(direction="in", top=True, right=True)

    handles, labels = axes[1].get_legend_handles_labels()
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
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.36, top=0.96, wspace=0.35)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _binom_pmf(n: int, k: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    logc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(logc + k * math.log(p) + (n - k) * math.log(1.0 - p))


def _chi_exact(phi: float, theta: float, n: int) -> float:
    k_high = int(math.ceil(phi * n))
    k_low = int(math.floor(theta * n))
    return float(n * (_binom_pmf(n - 1, k_high - 1) + _binom_pmf(n - 1, k_low)))


def _rc_from_chi(chi: float, n_m: float, n_w: float) -> float:
    return float(n_m * (chi + 2.0) / (n_m * (chi + 2.0) + n_w * (chi - 2.0)))


def fig_k500_finite_size(paths: S4Paths, *, out_pdf: Path, out_png: Path) -> None:
    pattern = paths.k500_pattern
    cands = sorted((ROOT / pattern).parent.glob(Path(pattern).name))
    if not cands:
        raise FileNotFoundError(f"未找到 k=500 N-sweep 缓存：{pattern}")

    points = []
    for p in cands:
        d = np.load(p, allow_pickle=False)
        k_list = d["k_list"].astype(int, copy=False) if "k_list" in d.files else np.asarray([500])
        if k_list.size != 1 or int(k_list[0]) != 500:
            continue
        points.append(
            {
                "n": int(d["n"]) if "n" in d.files else int(str(p.name).split("_")[2][1:]),
                "init": str(d.get("init_state", "")),
                "rc": float(d["rc_est"][0]),
                "lo": float(d["rc_ci_low"][0]),
                "hi": float(d["rc_ci_high"][0]),
                "phi": float(d.get("phi", 0.54)),
                "theta": float(d.get("theta", 0.46)),
                "n_m": float(d.get("n_m", 10.0)),
                "n_w": float(d.get("n_w", 5.0)),
                "path": p,
            }
        )

    if len(points) < 2:
        raise RuntimeError("k=500 有效 N 点数不足，无法做有限尺寸外推。")

    # 只用 random init 做外推（medium init 作为对照点）
    pts_random = [p for p in points if "random" in p["init"]]
    pts_medium = [p for p in points if "medium" in p["init"]]
    if len(pts_random) < 2:
        raise RuntimeError("k=500 random init 的 N 点数不足，无法拟合。")

    pts_random.sort(key=lambda x: x["n"])
    phi = float(pts_random[0]["phi"])
    theta = float(pts_random[0]["theta"])
    n_m = float(pts_random[0]["n_m"])
    n_w = float(pts_random[0]["n_w"])

    # exact discrete-threshold derivative
    chi_ex = _chi_exact(phi=phi, theta=theta, n=500)
    rc_exact = _rc_from_chi(chi_ex, n_m=n_m, n_w=n_w)
    # approx theory (src.theory)
    chi_approx = float(theory.calculate_chi(phi=phi, theta=theta, k_avg=500))
    rc_approx = float(theory.calculate_rc(n_m=n_m, n_w=n_w, chi=chi_approx))

    N = np.asarray([p["n"] for p in pts_random], dtype=float)
    y = np.asarray([p["rc"] for p in pts_random], dtype=float)
    lo = np.asarray([p["lo"] for p in pts_random], dtype=float)
    hi = np.asarray([p["hi"] for p in pts_random], dtype=float)
    se = (hi - lo) / 3.92
    w = 1.0 / (se**2)
    x = 1.0 / np.sqrt(N)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    X = np.vstack([np.ones_like(x[mask]), x[mask]]).T
    W = np.diag(w[mask])
    beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y[mask])
    cov = np.linalg.inv(X.T @ W @ X)
    rc_inf = float(beta[0])
    slope = float(beta[1])
    se_rc_inf = float(np.sqrt(cov[0, 0]))
    ci_lo, ci_hi = rc_inf - 1.96 * se_rc_inf, rc_inf + 1.96 * se_rc_inf

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.errorbar(
        x,
        y,
        yerr=[y - lo, hi - y],
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["blue"],
        elinewidth=1.4,
        capsize=3.0,
        label="ABM (random init)",
        zorder=3,
    )
    if pts_medium:
        xm = np.asarray([1.0 / math.sqrt(float(p["n"])) for p in pts_medium], dtype=float)
        ym = np.asarray([float(p["rc"]) for p in pts_medium], dtype=float)
        ax.plot(xm, ym, "s", color=OKABE_ITO["gray"], label="ABM (medium init)", zorder=3)

    xx = np.linspace(0.0, float(np.nanmax(x)) * 1.05, 200)
    yy = rc_inf + slope * xx
    ax.plot(xx, yy, "-", color=OKABE_ITO["blue"], alpha=0.65, label=r"fit: $r_c=r_\infty + c/\sqrt{N}$", zorder=2)

    ax.axhline(rc_exact, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.2, label=f"theory (exact) {rc_exact:.3f}", zorder=1)
    ax.axhline(rc_approx, color=OKABE_ITO["orange"], linestyle=":", linewidth=1.2, label=f"theory (approx) {rc_approx:.3f}", zorder=1)

    ax.set_xlabel(r"$1/\sqrt{N}$")
    ax.set_ylabel(r"Estimated $r_c$ (ABM, max-slope)")
    ax.tick_params(direction="in", top=True, right=True)

    ax.text(
        0.02,
        0.03,
        f"$r_\\infty={rc_inf:.4f}$\\n95% CI=[{ci_lo:.4f},{ci_hi:.4f}]",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="none"),
    )

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
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.96)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_beta_branch_bias(paths: S4Paths, *, out_pdf: Path, out_png: Path) -> None:
    for p in [paths.beta_high_only, paths.beta_symmetric]:
        if not p.exists():
            raise FileNotFoundError(f"未找到缓存：{p}")

    d1 = np.load(paths.beta_high_only, allow_pickle=False)
    d2 = np.load(paths.beta_symmetric, allow_pickle=False)
    betas = d1["betas"].astype(float, copy=False)
    p1 = d1["pos_branch_frac"].astype(float, copy=False)
    p2 = d2["pos_branch_frac"].astype(float, copy=False)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.6))

    ax.axhline(0.5, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.8)
    ax.plot(betas, p1, "o-", color=OKABE_ITO["blue"], linewidth=2.2, label="High-only local coupling")
    ax.plot(betas, p2, "s-", color=OKABE_ITO["bluish_green"], linewidth=2.2, label="Symmetric local coupling")
    ax.set_xlabel(r"Local coupling $\beta$")
    ax.set_ylabel(r"$P(Q>0\mid r=1)$")
    ax.set_xlim(float(np.min(betas)) - 0.005, float(np.max(betas)) + 0.005)
    ax.set_ylim(-0.02, 1.02)
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
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.34, top=0.96)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_beta_q_a(paths: S4Paths, *, out_pdf: Path, out_png: Path) -> None:
    if not paths.beta_high_only.exists():
        raise FileNotFoundError(f"未找到缓存：{paths.beta_high_only}")
    d = np.load(paths.beta_high_only, allow_pickle=False)
    betas = d["betas"].astype(float, copy=False)
    r_vals = d["r_vals"].astype(float, copy=False)
    q_abs = d["q_abs_mean"].astype(float, copy=False)  # (nBeta, nR, nSeeds)
    q_signed = d["q_mean"].astype(float, copy=False)
    a = d["a_mean"].astype(float, copy=False)

    q_abs_mean = np.nanmean(q_abs, axis=2)
    q_signed_mean = np.nanmean(q_signed, axis=2)
    a_mean = np.nanmean(a, axis=2)

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6), sharex=True)

    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"], OKABE_ITO["orange"], OKABE_ITO["reddish_purple"]]
    for i, beta in enumerate(betas):
        c = colors[i % len(colors)]
        axes[0].plot(r_vals, q_abs_mean[i], color=c, linewidth=2.0, label=fr"$\beta={beta:.2f}$")
        axes[1].plot(r_vals, q_signed_mean[i], color=c, linewidth=2.0)
        axes[2].plot(r_vals, a_mean[i], color=c, linewidth=2.0)

    for ax in axes:
        ax.axvline(float(theory.calculate_rc(10.0, 5.0, theory.calculate_chi(0.54, 0.46, 50))), color=OKABE_ITO["gray"], linestyle=":", linewidth=1.0, alpha=0.6)
        ax.tick_params(direction="in", top=True, right=True)
        ax.set_xlim(0.0, 1.0)

    axes[0].set_xlabel(r"$r$")
    axes[0].set_ylabel(r"$|Q|$")
    axes[1].set_xlabel(r"$r$")
    axes[1].set_ylabel(r"$Q$ (signed)")
    axes[2].set_xlabel(r"$r$")
    axes[2].set_ylabel(r"$a$")

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
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.36, top=0.96, wspace=0.35)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    paths = S4Paths()
    out = _out_dirs()

    fig_symmetric_diagonal(out_pdf=out["pdf"] / "s4_symmetric_diagonal.pdf", out_png=out["png"] / "s4_symmetric_diagonal.png")
    fig_k_effect_split(paths, out_pdf=out["pdf"] / "s4_k_effect_split.pdf", out_png=out["png"] / "s4_k_effect_split.png")
    fig_k500_finite_size(paths, out_pdf=out["pdf"] / "s4_k500_finite_size.pdf", out_png=out["png"] / "s4_k500_finite_size.png")
    fig_beta_branch_bias(paths, out_pdf=out["pdf"] / "s4_beta_branch_bias.pdf", out_png=out["png"] / "s4_beta_branch_bias.png")
    fig_beta_q_a(paths, out_pdf=out["pdf"] / "s4_beta_q_a.pdf", out_png=out["png"] / "s4_beta_q_a.png")

    print("[supp:S4] done")


if __name__ == "__main__":
    main()
