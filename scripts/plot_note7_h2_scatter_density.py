#!/usr/bin/env python3
"""
Plot Note07 H2 scatter with a density diagnostic overlay.

Goal (PI request):
  - Use Batch3 segments to plot r_proxy_mean vs volatility.
  - Mark high/low density to visualize the confounding effect of data density.

This script is intentionally "read-only" w.r.t. data: it only reads time_series_*.csv
and writes a single figure file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _segment_start_time(ts: pd.Series, segment: str) -> pd.Series:
    s = str(segment or "").strip()
    if not s:
        raise ValueError("segment 不能为空")
    return ts.dt.to_period(s).dt.to_timestamp()


def segment_metrics(df: pd.DataFrame, *, segment: str = "W") -> pd.DataFrame:
    x = df.dropna(subset=["time_window"]).copy()
    x["seg"] = _segment_start_time(x["time_window"], segment)
    rows: list[dict] = []
    for seg, g in x.groupby("seg"):
        g_aq = g.dropna(subset=["a", "Q"])
        if len(g_aq) < 10:
            continue
        # Keep the same segment filtering as `scripts/run_note7_empirical.py` so that
        # the scatter and the reported correlation use an identical sample set.
        g_jump = g.dropna(subset=["abs_dQ_abs_per_hour"])
        if len(g_jump) < 5:
            continue

        nw = float(g_aq.get("n_wemedia", pd.Series(dtype=float)).fillna(0).sum())
        nm = float(g_aq.get("n_mainstream", pd.Series(dtype=float)).fillna(0).sum())
        ng = float(g_aq.get("n_government", pd.Series(dtype=float)).fillna(0).sum())
        denom = nw + nm + ng
        r_proxy_mean = float(nw / denom) if denom > 0 else np.nan

        rows.append(
            {
                "seg": seg,
                "n_windows_aq": int(len(g_aq)),
                "n_windows_jump": int(len(g_jump)),
                "r_proxy_mean": r_proxy_mean,
                "volatility": float(g_aq["Q"].std()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("seg").reset_index(drop=True)


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[m], y[m])
    return float(r), float(p)


def partial_pearsonr(x: np.ndarray, y: np.ndarray, ctrl: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ctrl)
    if int(m.sum()) < 8:
        return float("nan"), float("nan")
    xv, yv, cv = x[m].astype(float), y[m].astype(float), ctrl[m].astype(float)
    X = np.column_stack([np.ones(len(cv)), cv])
    bx, *_ = np.linalg.lstsq(X, xv, rcond=None)
    by, *_ = np.linalg.lstsq(X, yv, rcond=None)
    xr = xv - X @ bx
    yr = yv - X @ by
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Note07 H2 scatter with density groups.")
    ap.add_argument(
        "--input",
        default="outputs/annotations/derived/time_series_batch3_4h.csv",
        help="Input time_series CSV (e.g., time_series_batch3_4h.csv)",
    )
    ap.add_argument("--freq", default="4h", help="Time aggregation frequency used to build the input CSV (default: 4h).")
    ap.add_argument("--segment", default="W", help="Segment granularity (default: W)")
    ap.add_argument(
        "--output",
        default="outputs/figs/empirical/fig7b_h2_scatter_batch3_density_4h.png",
        help="Output figure path (png)",
    )
    ap.add_argument(
        "--density-cut",
        default="median",
        choices=["median"],
        help="How to split high/low density (default: median split on n_windows_aq).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "time_window" not in df.columns:
        raise ValueError(f"缺少 time_window 列：{in_path}")
    df["time_window"] = pd.to_datetime(df["time_window"])

    # Add minimal window-level metrics required for segment filtering (jump windows).
    df = df.sort_values("time_window").reset_index(drop=True)
    step_hours = float(pd.to_timedelta(str(args.freq)).total_seconds() / 3600.0)
    df["dt_hours"] = df["time_window"].diff().dt.total_seconds() / 3600.0
    df.loc[df["dt_hours"] <= 0, "dt_hours"] = np.nan
    df["dt_ok"] = np.isclose(df["dt_hours"], step_hours)
    if len(df) > 0:
        df.loc[df.index[0], "dt_ok"] = False
    df.loc[~df["dt_ok"], "dt_hours"] = np.nan
    df["Q_abs"] = df["Q"].abs()
    df["dQ_abs_per_hour"] = df["Q_abs"].diff() / df["dt_hours"]
    df["abs_dQ_abs_per_hour"] = df["dQ_abs_per_hour"].abs()

    seg = segment_metrics(df, segment=args.segment)
    if seg.empty:
        raise SystemExit("无可用分段数据（可能数据过稀疏或阈值过严）。")

    cut = float(seg["n_windows_aq"].median())
    seg["density_group"] = np.where(seg["n_windows_aq"] >= cut, "High density", "Low density")

    x = seg["r_proxy_mean"].to_numpy(dtype=float)
    y = seg["volatility"].to_numpy(dtype=float)
    d = seg["n_windows_aq"].to_numpy(dtype=float)
    r_raw, p_raw = pearsonr(x, y)
    r_part, p_part = partial_pearsonr(x, y, d)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {"High density": "#1f77b4", "Low density": "#ff7f0e"}
    for grp, g in seg.groupby("density_group", sort=False):
        ax.scatter(
            g["r_proxy_mean"],
            g["volatility"],
            s=48,
            alpha=0.85,
            c=colors.get(str(grp), "#333333"),
            label=f"{grp} (n={len(g)})",
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title("H2 (Batch3): $r_{proxy}$ vs Volatility with Density Groups")
    ax.set_xlabel(r"$r_{proxy}$ (segment mean)")
    ax.set_ylabel(r"Volatility: $\mathrm{std}(Q)$ (segment)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, loc="best")

    text = (
        f"Pearson r={r_raw:.3f}, p={p_raw:.3g}\n"
        f"Partial r (ctrl n_windows)={r_part:.3f}, p={p_part:.3g}\n"
        f"Density split: median(n_windows_aq)={cut:.0f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#dddddd"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()
