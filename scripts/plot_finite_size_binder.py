"""
从 finite_size_binder_cross_*.npz 生成可用于论文的可视化图。

默认输出到：outputs/figs/fig2/

图（默认生成 3 张）：
1) binder 曲线 U4(r;N)（均值 ± SEM）
2) 相邻 N 的 Binder 交点估计 r_c（bootstrap 95% CI）
3)（可选诊断）每个 seed 的交点散点分布

注意：
- 适配无显示环境：强制使用 Agg 后端。
- 若系统对 ~/.config/matplotlib 无写权限，会自动将 MPLCONFIGDIR 指向 /tmp。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def _configure_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # headless


def _crossing_position_in_window(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    center: float,
    window: float,
) -> float:
    diff = y1 - y2
    xs = []
    for i in range(len(x) - 1):
        if diff[i] == 0:
            xs.append(float(x[i]))
            continue
        if diff[i] * diff[i + 1] < 0:
            t = diff[i] / (diff[i] - diff[i + 1])
            xs.append(float(x[i] + t * (x[i + 1] - x[i])))
    if not xs:
        return float("nan")
    lo, hi = center - window, center + window
    xs = [v for v in xs if lo <= v <= hi]
    if not xs:
        return float("nan")
    return float(min(xs, key=lambda r: abs(r - center)))


def bootstrap_crossings(
    binder_seeds_by_N: np.ndarray,
    r_scan: np.ndarray,
    rc_center: float,
    cross_window: float,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
    - rc_boot: shape (nPairs, n_boot)
    - rc_ci:  shape (3, nPairs)  -> [q2.5, q50, q97.5]
    """
    rng = np.random.default_rng(int(seed))
    nN, nSeeds, _ = binder_seeds_by_N.shape
    nPairs = nN - 1
    rc_boot = np.full((nPairs, n_boot), np.nan, dtype=float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, nSeeds, size=nSeeds)
        u4_mean = np.nanmean(binder_seeds_by_N[:, idx, :], axis=1)
        for i in range(nPairs):
            rc_boot[i, b] = _crossing_position_in_window(
                r_scan,
                u4_mean[i],
                u4_mean[i + 1],
                center=rc_center,
                window=cross_window,
            )
    rc_ci = np.nanquantile(rc_boot, [0.025, 0.5, 0.975], axis=1).astype(float)
    return rc_boot, rc_ci


def bootstrap_from_seed_crossings(
    rc_cross_seeds_by_pair: np.ndarray,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接对每个 pair 的 seed-level 交点做 bootstrap（推荐，速度更快且与 run 脚本一致）。
    返回：
    - rc_boot: shape (nPairs, n_boot)
    - rc_ci:  shape (3, nPairs)  -> [q2.5, q50, q97.5]
    """
    rng = np.random.default_rng(int(seed))
    rc_seed = rc_cross_seeds_by_pair.astype(float)
    nPairs, nSeeds = rc_seed.shape
    rc_boot = np.full((nPairs, int(n_boot)), np.nan, dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, nSeeds, size=nSeeds)
        sample = rc_seed[:, idx]
        rc_boot[:, b] = np.nanmedian(sample, axis=1)
    rc_ci = np.nanquantile(rc_boot, [0.025, 0.5, 0.975], axis=1).astype(float)
    return rc_boot, rc_ci


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot finite-size Binder cumulant results from .npz")
    parser.add_argument(
        "--npz",
        type=str,
        default="",
        help="输入 .npz 路径（默认自动选择 outputs/data/ 下最新的 finite_size_binder_cross*.npz）",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/figs/fig2")
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fmt", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--cross-window",
        type=float,
        default=-1.0,
        help="（兼容旧数据）window 法交点窗口：默认读取 npz 内 cross_window，否则 0.05",
    )
    parser.add_argument("--diag-seed-scatter", action="store_true", help="额外输出每个 seed 的交点散点图（诊断用）")
    args = parser.parse_args()

    root = Path.cwd()
    npz_path: Path
    if args.npz:
        npz_path = Path(args.npz)
    else:
        cand = sorted((root / "outputs" / "data").glob("finite_size_binder_cross*.npz"))
        if not cand:
            raise SystemExit("未找到 finite_size_binder_cross*.npz，请用 --npz 显式指定。")
        npz_path = cand[-1]

    d = np.load(npz_path, allow_pickle=False)
    r_scan = d["r_scan"].astype(float)
    N_list = d["N_list"].astype(int)
    binder_mean = d["binder_mean_by_N"].astype(float)
    binder_sem = d["binder_sem_by_N"].astype(float)
    binder_seeds = d["binder_seeds_by_N"].astype(float)
    pair_mid = d["pair_N_mid"].astype(float)
    pair_labels = d["pair_labels"].astype(str)
    rc_theory = float(d["rc_theory"])
    confirms_method = str(d["cross_method"]) if "cross_method" in d.files else "window"

    cross_window = float(args.cross_window)
    if cross_window <= 0:
        cross_window = float(d["cross_window"]) if "cross_window" in d.files else 0.05

    # 若 npz 内已带 bootstrap CI，优先复用；否则在绘图脚本里算一次（成本很低）
    rc_ci = None
    if "rc_cross_ci" in d.files and d["rc_cross_ci"].size:
        rc_ci = d["rc_cross_ci"].astype(float)  # (3, nPairs)
    else:
        if "rc_cross_seeds_by_pair" in d.files and d["rc_cross_seeds_by_pair"].size:
            _, rc_ci = bootstrap_from_seed_crossings(
                rc_cross_seeds_by_pair=d["rc_cross_seeds_by_pair"].astype(float),
                n_boot=int(args.bootstrap),
                seed=int(args.bootstrap_seed),
            )
        else:
            # 兜底：老版本未保存 seed-level crossing，退回到 mean curve + window 法
            _, rc_ci = bootstrap_crossings(
                binder_seeds_by_N=binder_seeds,
                r_scan=r_scan,
                rc_center=rc_theory,
                cross_window=cross_window,
                n_boot=int(args.bootstrap),
                seed=int(args.bootstrap_seed),
            )

    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="deep")

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.strip() or npz_path.stem

    # -------------------------
    # Fig A: U4 curves
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, N in enumerate(N_list):
        line = ax.plot(r_scan, binder_mean[i], linewidth=2, label=f"N={int(N)}")[0]
        color = line.get_color()
        ax.fill_between(
            r_scan,
            binder_mean[i] - binder_sem[i],
            binder_mean[i] + binder_sem[i],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
    ax.axvline(rc_theory, color="gray", linestyle="--", linewidth=2, label=fr"Theory $r_c$={rc_theory:.3f}")
    ax.set_xlabel("Control Parameter $r$")
    ax.set_ylabel(r"$U_4=1-\langle Q^4\rangle/(3\langle Q^2\rangle^2)$")
    ax.set_title("Finite-Size Scaling: Binder cumulant $U_4(r;N)$")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # -------------------------
    # Fig B: rc crossings (bootstrap CI)
    # -------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    rc_med = rc_ci[1]
    rc_lo = rc_ci[0]
    rc_hi = rc_ci[2]
    yerr = np.vstack([rc_med - rc_lo, rc_hi - rc_med])

    ax2.errorbar(
        pair_mid,
        rc_med,
        yerr=yerr,
        fmt="o-",
        markersize=7,
        linewidth=2,
        capsize=3,
        label=f"Binder crossing (bootstrap 95% CI, {confirms_method})",
    )
    ax2.axhline(rc_theory, color="red", linestyle="--", linewidth=2, label=fr"Theory $r_c$={rc_theory:.3f}")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Pair size $\sqrt{N_i N_{i+1}}$")
    ax2.set_ylabel("Estimated $r_c$")
    ax2.set_title("Finite-Size Scaling: $r_c$ from Binder crossings")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # -------------------------
    # Fig C (optional): per-seed crossings scatter
    # -------------------------
    fig3 = None
    if args.diag_seed_scatter and "rc_cross_seeds_by_pair" in d.files:
        rc_seed = d["rc_cross_seeds_by_pair"].astype(float)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for i in range(rc_seed.shape[0]):
            x = np.full(rc_seed.shape[1], pair_mid[i])
            jitter = (np.arange(rc_seed.shape[1]) - rc_seed.shape[1] / 2) * 0.01 * pair_mid[i]
            y = rc_seed[i]
            ax3.scatter(x + jitter, y, s=25, alpha=0.75, label=pair_labels[i] if i == 0 else None)
        ax3.axhline(rc_theory, color="red", linestyle="--", linewidth=2, label=fr"Theory $r_c$={rc_theory:.3f}")
        ax3.set_xscale("log")
        ax3.set_xlabel(r"Pair size $\sqrt{N_i N_{i+1}}$")
        ax3.set_ylabel("Seed-level $r_c$ estimates")
        ax3.set_title("Diagnostics: seed-level Binder crossings")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        fig3.tight_layout()

    def save(fig, name: str) -> None:
        base = out_dir / f"{prefix}_{name}"
        if args.fmt in ("png", "both"):
            fig.savefig(base.with_suffix(".png"), dpi=int(args.dpi))
        if args.fmt in ("pdf", "both"):
            fig.savefig(base.with_suffix(".pdf"))

    save(fig, "u4_curves")
    save(fig2, "rc_crossings_bootstrap")
    if fig3 is not None:
        save(fig3, "rc_crossings_seed_scatter")

    print(f"[ok] npz: {npz_path}")
    print(f"[ok] out_dir: {out_dir}")
    print(f"[ok] cross_method: {confirms_method}")
    if confirms_method == "window":
        print(f"[ok] cross_window: ±{cross_window}")
    if "u4_low" in d.files and "u4_high" in d.files:
        print(f"[ok] u4_band: [{float(d['u4_low']):.3f}, {float(d['u4_high']):.3f}]")
    print(f"[ok] rc_theory: {rc_theory:.6f}")


if __name__ == "__main__":
    main()
