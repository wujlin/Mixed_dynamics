"""
从 Note04 的 k=500 N-sweep 缓存（.npz）生成有限尺寸外推图（Fig4b2）。

用途：
- 解释在固定 N 下，k=500 的 ABM 有效 r_c 可能偏离理论（有限尺寸效应）。
- 用 r_c(N)=r_∞ + c/sqrt(N) 做外推，检验 N→∞ 是否与理论一致。

输入（默认自动搜寻）：
- outputs/data/note4_k500_N*_*.npz

输出（默认）：
- outputs/figs/fig4/fig4b2_k500_finite_size.png

注意：
- 适配无显示环境：强制使用 Agg 后端。
- 若系统对 ~/.config/matplotlib 无写权限，会自动将 MPLCONFIGDIR 指向 /tmp。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(Path(os.getenv("TMPDIR", "/tmp")) / "matplotlib")
    import matplotlib

    matplotlib.use("Agg")  # headless


def _binom_pmf(n: int, k: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    logc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(logc + k * math.log(p) + (n - k) * math.log(1.0 - p))


def _chi_exact(phi: float, theta: float, n: int) -> float:
    """
    严格导数（与 ABM 判据一致）：
    - perceived_risk = X/n, X~Bin(n,p)
    - high: perceived_risk >= phi  => X >= ceil(phi*n)
    - low : perceived_risk <= theta => X <= floor(theta*n)
    令 S(p)=P(high)-P(low)，则：
      dS/dp|_{p=0.5} = n*[pmf_{n-1}(ceil(phi*n)-1)+pmf_{n-1}(floor(theta*n))]
    """
    k_high = int(math.ceil(phi * n))
    k_low = int(math.floor(theta * n))
    return float(n * (_binom_pmf(n - 1, k_high - 1) + _binom_pmf(n - 1, k_low)))


def _rc_from_chi(chi: float, n_m: float, n_w: float) -> float:
    return float(n_m * (chi + 2.0) / (n_m * (chi + 2.0) + n_w * (chi - 2.0)))


@dataclass(frozen=True)
class OnePoint:
    n: int
    init_state: str
    rc: float
    lo: float
    hi: float
    phi: float
    theta: float
    n_m: float
    n_w: float
    r_min: float
    r_max: float
    path: Path


def _load_one(path: Path, *, k_target: int) -> Optional[OnePoint]:
    try:
        d = np.load(path, allow_pickle=False)
        k_list = d["k_list"].astype(int)
        if k_list.size != 1 or int(k_list[0]) != int(k_target):
            d.close()
            return None
        n = int(d["n"])
        init_state = str(d.get("init_state", ""))
        rc = float(d["rc_est"][0])
        lo = float(d["rc_ci_low"][0])
        hi = float(d["rc_ci_high"][0])
        phi = float(d.get("phi", np.nan))
        theta = float(d.get("theta", np.nan))
        n_m = float(d.get("n_m", np.nan))
        n_w = float(d.get("n_w", np.nan))
        r_vals = d.get("r_vals", np.asarray([], dtype=float))
        r_min = float(r_vals[0]) if r_vals.size else float("nan")
        r_max = float(r_vals[-1]) if r_vals.size else float("nan")
        d.close()
        return OnePoint(
            n=n,
            init_state=init_state,
            rc=rc,
            lo=lo,
            hi=hi,
            phi=phi,
            theta=theta,
            n_m=n_m,
            n_w=n_w,
            r_min=r_min,
            r_max=r_max,
            path=path,
        )
    except Exception:
        return None


def _fit_rc_inf(points: List[OnePoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    N = np.asarray([p.n for p in points], dtype=float)
    y = np.asarray([p.rc for p in points], dtype=float)
    lo = np.asarray([p.lo for p in points], dtype=float)
    hi = np.asarray([p.hi for p in points], dtype=float)
    # 由 95%CI 近似标准差：CI ≈ ±1.96*se
    se = (hi - lo) / 3.92
    se = np.where(se <= 0, np.nan, se)
    w = 1.0 / (se**2)
    x = 1.0 / np.sqrt(N)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    if mask.sum() < 2:
        raise ValueError("有效 N 点数不足，无法拟合 rc_inf")

    X = np.vstack([np.ones_like(x[mask]), x[mask]]).T
    W = np.diag(w[mask])
    XtWX = X.T @ W @ X
    beta = np.linalg.inv(XtWX) @ (X.T @ W @ y[mask])
    cov = np.linalg.inv(XtWX)
    rc_inf = float(beta[0])
    slope = float(beta[1])
    se_rc_inf = float(np.sqrt(cov[0, 0]))
    return x, y, lo, hi, rc_inf, slope, se_rc_inf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot k=500 finite-size scaling from Note04 N-sweep .npz caches")
    parser.add_argument(
        "--pattern",
        type=str,
        default="outputs/data/note4_k500_N*_*.npz",
        help="输入缓存 glob（相对当前工作目录）",
    )
    parser.add_argument("--k", type=int, default=500, help="目标 k（默认 500）")
    parser.add_argument("--out", type=str, default="outputs/figs/fig4/fig4b2_k500_finite_size.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-approx-theory", action="store_true", help="不画 src.theory 的近似理论线（只保留 exact）")
    args = parser.parse_args()

    pattern = Path(args.pattern)
    if not pattern.is_absolute():
        pattern = ROOT / pattern
    cands = sorted(pattern.parent.glob(pattern.name))
    if not cands:
        raise SystemExit(f"未找到缓存：{args.pattern}")

    rows = [r for r in (_load_one(p, k_target=int(args.k)) for p in cands) if r is not None]
    if not rows:
        raise SystemExit(f"未找到 k={int(args.k)} 的有效缓存（检查 npz 内 k_list）")

    rows_random = sorted([r for r in rows if "random" in r.init_state], key=lambda x: x.n)
    rows_medium = sorted([r for r in rows if "medium" in r.init_state], key=lambda x: x.n)
    if len(rows_random) < 2:
        raise SystemExit("random init 的 N 点数不足（建议至少 2 个 N）")

    ref = rows_random[0]
    phi, theta = float(ref.phi), float(ref.theta)
    n_m, n_w = float(ref.n_m), float(ref.n_w)

    chi_ex = _chi_exact(phi, theta, int(args.k))
    rc_exact = _rc_from_chi(chi_ex, n_m, n_w)

    rc_approx = float("nan")
    if not args.no_approx_theory:
        try:
            from src.theory import calculate_chi, calculate_rc  # noqa: WPS433

            chi_approx = float(calculate_chi(phi=phi, theta=theta, k_avg=int(args.k)))
            rc_approx = float(calculate_rc(n_m=n_m, n_w=n_w, chi=chi_approx))
        except Exception:
            rc_approx = float("nan")

    x, y, lo, hi, rc_inf, slope, se_rc_inf = _fit_rc_inf(rows_random)
    ci_lo, ci_hi = rc_inf - 1.96 * se_rc_inf, rc_inf + 1.96 * se_rc_inf

    print("[k=500][N-sweep] files:")
    for r in rows_random:
        print("  ", r.path.name)
    for r in rows_medium:
        print("  ", r.path.name)
    print(f"[k={int(args.k)}][FSS] rc_inf={rc_inf:.6f} 95%CI=[{ci_lo:.6f},{ci_hi:.6f}]")
    print(f"[k={int(args.k)}][theory] rc_exact={rc_exact:.6f}" + (f", rc_approx={rc_approx:.6f}" if np.isfinite(rc_approx) else ""))

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", color="tab:blue", capsize=3, label="ABM (random init)")
    if rows_medium:
        xm = np.asarray([1.0 / math.sqrt(float(r.n)) for r in rows_medium], dtype=float)
        ym = np.asarray([float(r.rc) for r in rows_medium], dtype=float)
        ax.plot(xm, ym, "s", color="tab:gray", label="ABM (medium init)")

    xx = np.linspace(0.0, float(np.nanmax(x)) * 1.05, 200)
    yy = rc_inf + slope * xx
    ax.plot(xx, yy, "-", color="tab:blue", alpha=0.6, label=r"fit: $r_c=r_\infty + c/\sqrt{N}$")

    ax.axhline(rc_exact, color="tab:red", linestyle="--", linewidth=1.2, label=f"theory (exact) {rc_exact:.3f}")
    if np.isfinite(rc_approx):
        ax.axhline(rc_approx, color="tab:orange", linestyle=":", linewidth=1.2, label=f"theory (approx) {rc_approx:.3f}")

    ax.set_xlabel(r"$1/\sqrt{N}$")
    ax.set_ylabel(r"$r_c$ (ABM, max-slope)")
    ax.set_title(f"k={int(args.k)} finite-size scaling (local scan)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax.text(
        0.02,
        0.02,
        f"$r_\\infty={rc_inf:.4f}$\\n95%CI=[{ci_lo:.4f},{ci_hi:.4f}]\\nscan r∈[{ref.r_min:.2f},{ref.r_max:.2f}]",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    print("[done] Saved:", out_path)


if __name__ == "__main__":
    main()
