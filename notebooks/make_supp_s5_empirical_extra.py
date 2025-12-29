"""
Supplementary S5：Empirical validation（补充图 + 表格素材）。

注意：
- 本脚本聚合已有派生 time series（outputs/annotations/derived/time_series_*_4h.csv）。
- 生成的 PDF 用于 supplementary.tex；预览 PNG 放在 outputs/figs/supp/。

输出：
  - Essay/figures_supp/s5_time_series_overview.pdf
  - Essay/figures_supp/s5_scatter_h1_h2.pdf
  - Essay/figures_supp/s5_eventstudy_h4.pdf

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_supp_s5_empirical_extra.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import OKABE_ITO, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class S5Config:
    freq: str = "4h"
    segment: str = "W"
    jump_q: float = 0.95
    n_boot: int = 2000
    seed: int = 0
    roll_win: int = 12
    pre: int = 24
    event_quantile: float = 0.95


def _out_dirs() -> Dict[str, Path]:
    out_pdf = ROOT / "Essay" / "figures_supp"
    out_png = ROOT / "outputs" / "figs" / "supp"
    out_pdf.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return {"pdf": out_pdf, "png": out_png}


def _freq_to_step_hours(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 3600.0)


_SEGMENT_FLOOR_UNITS = {"H": "h", "h": "h", "D": "D", "d": "D"}


def _segment_start_time(ts: pd.Series, segment: str) -> pd.Series:
    s = str(segment or "").strip()
    if not s:
        raise ValueError("segment 不能为空")
    m = re.fullmatch(r"(\\d+)\\s*([HhDd])", s)
    if m:
        n, unit = m.group(1), m.group(2)
        unit_norm = _SEGMENT_FLOOR_UNITS.get(unit, unit)
        return ts.dt.floor(f"{n}{unit_norm}")
    return ts.dt.to_period(s).dt.to_timestamp()


def add_window_metrics(ts: pd.DataFrame, *, freq: str, vol_win: int = 12) -> pd.DataFrame:
    df = ts.copy().sort_values("time_window").reset_index(drop=True)
    step_hours = _freq_to_step_hours(freq)

    df["Q_abs"] = df["Q"].abs()
    df["dt_hours"] = df["time_window"].diff().dt.total_seconds() / 3600.0
    df.loc[df["dt_hours"] <= 0, "dt_hours"] = np.nan

    df["dt_ok"] = np.isclose(df["dt_hours"], step_hours)
    if len(df) > 0:
        df.loc[df.index[0], "dt_ok"] = False
    df.loc[~df["dt_ok"], "dt_hours"] = np.nan

    df["dQ_abs_per_hour"] = df["Q_abs"].diff() / df["dt_hours"]
    df["abs_dQ_abs_per_hour"] = df["dQ_abs_per_hour"].abs()

    valid = df["Q"].notna()
    prev_valid = valid.shift(1, fill_value=False)
    df["is_break"] = (~valid) | (~prev_valid) | (~df["dt_ok"])
    df["block_id"] = df["is_break"].cumsum()

    df["Q_volatility"] = df["Q"].rolling(vol_win, min_periods=max(3, vol_win // 3)).std()
    return df


def segment_metrics_h1(df: pd.DataFrame, *, segment: str, jump_q: float) -> pd.DataFrame:
    x = df.dropna(subset=["time_window"]).copy()
    x["seg"] = _segment_start_time(x["time_window"], segment)
    rows: list[dict] = []
    for seg, g in x.groupby("seg"):
        g_aq = g.dropna(subset=["a", "Q"])
        if len(g_aq) < 10:
            continue
        g_jump = g.dropna(subset=["abs_dQ_abs_per_hour"])
        if len(g_jump) < 5:
            continue

        w = g_aq["n_public"].fillna(0).astype(float).values if "n_public" in g_aq.columns else None
        if w is not None and float(w.sum()) > 0:
            a_mean = float(np.average(g_aq["a"].values, weights=w))
        else:
            a_mean = float(g_aq["a"].mean())

        rows.append(
            {
                "seg": seg,
                "n_windows_aq": int(len(g_aq)),
                "n_windows_jump": int(len(g_jump)),
                "a_mean": a_mean,
                "jump_q95": float(np.nanpercentile(g_jump["abs_dQ_abs_per_hour"].values, 100.0 * float(jump_q))),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("seg").reset_index(drop=True) if not out.empty else out


def segment_metrics_h2(df: pd.DataFrame, *, segment: str) -> pd.DataFrame:
    x = df.dropna(subset=["time_window"]).copy()
    x["seg"] = _segment_start_time(x["time_window"], segment)
    rows: list[dict] = []
    for seg, g in x.groupby("seg"):
        g_aq = g.dropna(subset=["a", "Q"])
        if len(g_aq) < 10:
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
                "r_proxy_mean": r_proxy_mean,
                "volatility": float(g_aq["Q"].std()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("seg").reset_index(drop=True) if not out.empty else out


def _bootstrap_fit_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray, *, n_boot: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    coef = np.polyfit(x, y, deg=1)
    mean = coef[0] * grid + coef[1]
    rng = np.random.default_rng(int(seed))
    preds = np.empty((int(n_boot), grid.size), dtype=float)
    n = int(x.size)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        c = np.polyfit(x[idx], y[idx], deg=1)
        preds[b] = c[0] * grid + c[1]
    lo = np.percentile(preds, 2.5, axis=0)
    hi = np.percentile(preds, 97.5, axis=0)
    return mean, lo, hi


def rolling_ac1_window(w: np.ndarray) -> float:
    if np.isnan(w).any():
        return np.nan
    w = w - float(np.mean(w))
    x1 = w[:-1]
    x2 = w[1:]
    denom = float(np.sqrt(np.sum(x1**2) * np.sum(x2**2)))
    if denom == 0:
        return 0.0
    return float(np.sum(x1 * x2) / denom)


def add_block_rolling(df: pd.DataFrame, *, col: str, window: int, block_col: str = "block_id") -> pd.DataFrame:
    out = df.copy()
    out[f"{col}_rolling_var"] = np.nan
    out[f"{col}_rolling_ac1"] = np.nan
    for _, g in out.groupby(block_col):
        s = g[col]
        out.loc[g.index, f"{col}_rolling_var"] = s.rolling(int(window), min_periods=int(window)).var()
        out.loc[g.index, f"{col}_rolling_ac1"] = s.rolling(int(window), min_periods=int(window)).apply(rolling_ac1_window, raw=True)
    return out


def _eligible_indices_by_block(df: pd.DataFrame, *, col: str, pre: int, block_col: str = "block_id") -> set[int]:
    eligible: set[int] = set()
    if df.empty:
        return eligible
    for idx in range(int(pre), len(df)):
        if pd.isna(df.at[idx, col]):
            continue
        b = df.at[idx, block_col]
        w = df.loc[idx - int(pre) : idx, [col, block_col]]
        if w[block_col].nunique() != 1:
            continue
        if w[col].isna().any():
            continue
        eligible.add(int(idx))
    return eligible


def pick_jump_events(
    df: pd.DataFrame,
    *,
    freq: str,
    q_col: str,
    quantile: float,
    min_gap_windows: int,
    block_col: str,
    allowed_idx: set[int],
) -> Tuple[list[int], float]:
    x = df.dropna(subset=["time_window", q_col]).copy().sort_values("time_window")
    if not allowed_idx:
        return [], float("nan")
    x = x[x.index.isin(allowed_idx)]
    if len(x) < 30:
        return [], float("nan")
    thr = float(x[q_col].quantile(float(quantile)))
    cand = x[x[q_col] >= thr]
    step_hours = _freq_to_step_hours(freq)
    events: list[int] = []
    last_time = None
    last_block = None
    for idx, row in cand.iterrows():
        b = row.get(block_col, None)
        if last_time is not None and b == last_block:
            dt_hours = (row["time_window"] - last_time).total_seconds() / 3600.0
            if dt_hours < float(min_gap_windows) * step_hours:
                continue
        events.append(int(idx))
        last_time = row["time_window"]
        last_block = b
    return events, thr


def event_study(df: pd.DataFrame, event_idx: list[int], *, col: str, pre: int, block_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mats = []
    for idx in event_idx:
        if idx - int(pre) < 0:
            continue
        w = df.loc[idx - int(pre) : idx, [col, block_col]]
        if w[block_col].nunique() != 1:
            continue
        arr = w[col].values
        if np.isnan(arr).any():
            continue
        mats.append(arr)
    if not mats:
        return None
    mat = np.vstack(mats)
    mean = np.mean(mat, axis=0)
    lo = np.percentile(mat, 2.5, axis=0)
    hi = np.percentile(mat, 97.5, axis=0)
    return mean, lo, hi


def _load_ts(name: str, *, freq: str) -> pd.DataFrame:
    p = ROOT / "outputs" / "annotations" / "derived" / f"time_series_{name}_{freq}.csv"
    if not p.exists():
        raise FileNotFoundError(f"未找到 time series：{p}")
    ts = pd.read_csv(p)
    ts["time_window"] = pd.to_datetime(ts["time_window"])
    ts["dataset"] = str(name)
    return ts


def fig_time_series(cfg: S5Config, *, out_pdf: Path, out_png: Path) -> None:
    df = pd.concat([_load_ts(n, freq=cfg.freq) for n in ["master", "batch3", "all"]], ignore_index=True)

    apply_paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True)

    palette = {"master": OKABE_ITO["gray"], "batch3": OKABE_ITO["blue"], "all": OKABE_ITO["vermillion"]}
    for ax, col, ylabel in [
        (axes[0], "Q", r"Polarization $Q$"),
        (axes[1], "a", r"Activity $a$"),
        (axes[2], "r_proxy", r"$r_{\mathrm{proxy}}$"),
    ]:
        for name, g in df.groupby("dataset"):
            ax.plot(g["time_window"], g[col], color=palette[str(name)], linewidth=1.6, alpha=0.85, label=str(name))
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in", top=True, right=True)

    axes[2].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, ncol=3, handlelength=2.0)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.93, hspace=0.22)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_scatter_h1_h2(cfg: S5Config, *, out_pdf: Path, out_png: Path) -> None:
    datasets = ["master", "batch3", "all"]
    ts = {name: add_window_metrics(_load_ts(name, freq=cfg.freq), freq=cfg.freq) for name in datasets}
    seg_h1 = {name: segment_metrics_h1(ts[name], segment=cfg.segment, jump_q=cfg.jump_q) for name in datasets}
    seg_h2 = {name: segment_metrics_h2(ts[name], segment=cfg.segment) for name in datasets}

    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2))

    for j, name in enumerate(datasets):
        # H1
        ax = axes[0, j]
        g = seg_h1[name].copy()
        x = g["a_mean"].to_numpy(dtype=float)
        y = g["jump_q95"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        c = palette = {"master": OKABE_ITO["gray"], "batch3": OKABE_ITO["blue"], "all": OKABE_ITO["vermillion"]}[name]
        ax.scatter(x, y, s=26, alpha=0.85, c=c, edgecolors="white", linewidths=0.5, zorder=3)
        if x.size >= 5:
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            mean, lo, hi = _bootstrap_fit_band(x, y, grid, n_boot=cfg.n_boot, seed=cfg.seed + j)
            ax.plot(grid, mean, color=c, lw=2.0, zorder=4)
            ax.fill_between(grid, lo, hi, color=c, alpha=0.16, linewidth=0, zorder=2)
        ax.set_title(name)
        ax.set_xlabel(r"Activity $a$ (segment mean)")
        ax.set_ylabel(r"Jump intensity (q95)")
        ax.tick_params(direction="in", top=True, right=True)

        # H2
        ax = axes[1, j]
        g = seg_h2[name].copy()
        cut = float(g["n_windows_aq"].median())
        g["density"] = np.where(g["n_windows_aq"] >= cut, "High density", "Low density")
        colors = {"High density": OKABE_ITO["blue"], "Low density": OKABE_ITO["vermillion"]}
        linestyles = {"High density": "-", "Low density": "--"}
        for grp, gg in g.groupby("density"):
            x = gg["r_proxy_mean"].to_numpy(dtype=float)
            y = gg["volatility"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]
            if x.size < 3:
                continue
            c2 = colors[str(grp)]
            ax.scatter(x, y, s=26, alpha=0.80, c=c2, edgecolors="white", linewidths=0.5, zorder=3)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            mean, lo, hi = _bootstrap_fit_band(x, y, grid, n_boot=cfg.n_boot, seed=cfg.seed + 17 + j + (0 if grp == "High density" else 7))
            ax.plot(grid, mean, color=c2, lw=2.0, linestyle=linestyles[str(grp)], zorder=4)
            ax.fill_between(grid, lo, hi, color=c2, alpha=0.14, linewidth=0, zorder=2)
        ax.set_xlabel(r"$r_{\mathrm{proxy}}$ (segment mean)")
        ax.set_ylabel("Volatility")
        ax.tick_params(direction="in", top=True, right=True)

    # 全图 legend：密度分组
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="-", color=OKABE_ITO["blue"], markersize=6, lw=2.0, markeredgecolor="white", markeredgewidth=0.6),
        mpl.lines.Line2D([], [], marker="o", linestyle="--", color=OKABE_ITO["vermillion"], markersize=6, lw=2.0, markeredgecolor="white", markeredgewidth=0.6),
    ]
    fig.legend(handles, ["High density", "Low density"], loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, ncol=2, handlelength=2.0)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.94, wspace=0.32, hspace=0.38)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def fig_eventstudy_h4(cfg: S5Config, *, out_pdf: Path, out_png: Path) -> None:
    datasets = ["master", "batch3", "all"]
    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), sharex=True)

    for j, name in enumerate(datasets):
        ts = _load_ts(name, freq=cfg.freq)
        df = add_window_metrics(ts, freq=cfg.freq)
        df = add_block_rolling(df, col="Q_abs", window=int(cfg.roll_win), block_col="block_id")

        eligible_ac = _eligible_indices_by_block(df, col="Q_abs_rolling_ac1", pre=int(cfg.pre), block_col="block_id")
        eligible_var = _eligible_indices_by_block(df, col="Q_abs_rolling_var", pre=int(cfg.pre), block_col="block_id")
        eligible = eligible_ac & eligible_var

        events, _ = pick_jump_events(
            df,
            freq=cfg.freq,
            q_col="abs_dQ_abs_per_hour",
            quantile=float(cfg.event_quantile),
            min_gap_windows=max(3, int(cfg.roll_win) // 2),
            block_col="block_id",
            allowed_idx=eligible,
        )

        ac = event_study(df, events, col="Q_abs_rolling_ac1", pre=int(cfg.pre), block_col="block_id")
        var = event_study(df, events, col="Q_abs_rolling_var", pre=int(cfg.pre), block_col="block_id")

        xs = np.arange(-int(cfg.pre), 1)
        # AC1
        ax = axes[0, j]
        ax.axvline(0, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.0, alpha=0.8)
        if ac is not None:
            mean, lo, hi = ac
            ax.plot(xs, mean, color=OKABE_ITO["blue"], lw=2.0)
            ax.fill_between(xs, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, linewidth=0)
        ax.set_title(name)
        ax.set_ylabel("AC1(|Q|)")
        ax.tick_params(direction="in", top=True, right=True)

        # Var
        ax = axes[1, j]
        ax.axvline(0, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.0, alpha=0.8)
        if var is not None:
            mean, lo, hi = var
            ax.plot(xs, mean, color=OKABE_ITO["vermillion"], lw=2.0)
            ax.fill_between(xs, lo, hi, color=OKABE_ITO["vermillion"], alpha=0.18, linewidth=0)
        ax.set_xlabel("Hours before event (4H steps)")
        ax.set_ylabel("Var(|Q|)")
        ax.tick_params(direction="in", top=True, right=True)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.92, wspace=0.32, hspace=0.34)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main() -> None:
    cfg = S5Config()
    out = _out_dirs()

    fig_time_series(cfg, out_pdf=out["pdf"] / "s5_time_series_overview.pdf", out_png=out["png"] / "s5_time_series_overview.png")
    fig_scatter_h1_h2(cfg, out_pdf=out["pdf"] / "s5_scatter_h1_h2.pdf", out_png=out["png"] / "s5_scatter_h1_h2.png")
    fig_eventstudy_h4(cfg, out_pdf=out["pdf"] / "s5_eventstudy_h4.pdf", out_png=out["png"] / "s5_eventstudy_h4.png")

    print("[supp:S5] done")


if __name__ == "__main__":
    main()
