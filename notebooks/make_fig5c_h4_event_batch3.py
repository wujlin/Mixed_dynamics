"""
Fig 5c：经验验证 H4（Batch3）——事件对齐的 AC1(|Q|) 与 Var(|Q|)（无 placebo 线）。

口径（与 Note07 收敛稿一致）：
- 数据：outputs/annotations/derived/time_series_batch3_4h.csv（freq=4h）
- rolling：roll_win=12（约 48h）
- pre=24（约 96h）
- 事件：|d|Q|/dt|| 超过 95 分位（q=0.95），并在同一连续块内保持最小间隔
- 指标：block-aware rolling AC1(|Q|) 与 Var(|Q|)

可视化决策：
- 不画 placebo 基线；阴影带表示事件曲线在事件集合上的 95% CI
- Times New Roman、无网格、线宽字号对齐 Fig2–4
- 输出 PDF（矢量）+ 预览 PNG

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig5c_h4_event_batch3.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Fig5cConfig:
    input_csv: Path = ROOT / "outputs" / "annotations" / "derived" / "time_series_batch3_4h.csv"
    freq: str = "4h"
    roll_win: int = 12
    pre: int = 24
    event_quantile: float = 0.95
    fig_size: Tuple[float, float] = (7.6, 3.6)


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


def _freq_to_step_hours(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 3600.0)


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


def _eligible_indices_by_block(df: pd.DataFrame, *, col: str, pre: int, block_col: str = "block_id") -> dict[int, list[int]]:
    eligible: dict[int, list[int]] = {}
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
        eligible.setdefault(int(b), []).append(int(idx))
    return eligible


def pick_jump_events(
    df: pd.DataFrame,
    *,
    freq: str,
    q_col: str,
    quantile: float,
    min_gap_windows: int,
    block_col: str,
    allowed_idx: set[int] | None,
) -> tuple[list[int], float]:
    x = df.dropna(subset=["time_window", q_col]).copy().sort_values("time_window")
    if allowed_idx is not None:
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


def event_study(df: pd.DataFrame, event_idx: list[int], *, col: str, pre: int, block_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
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
    return mean, lo, hi, mat


def main() -> None:
    cfg = Fig5cConfig()
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{cfg.input_csv}")

    ts = pd.read_csv(cfg.input_csv)
    if "time_window" not in ts.columns:
        raise ValueError("CSV 缺少 time_window 列")
    ts["time_window"] = pd.to_datetime(ts["time_window"])

    df = add_window_metrics(ts, freq=cfg.freq)
    df = add_block_rolling(df, col="Q_abs", window=int(cfg.roll_win), block_col="block_id")

    eligible_ac = _eligible_indices_by_block(df, col="Q_abs_rolling_ac1", pre=int(cfg.pre), block_col="block_id")
    eligible_var = _eligible_indices_by_block(df, col="Q_abs_rolling_var", pre=int(cfg.pre), block_col="block_id")
    eligible_both = {i for xs in eligible_ac.values() for i in xs} & {i for xs in eligible_var.values() for i in xs}

    # 与收敛口径一致：事件从 both eligible 中选，避免因为缺口导致“看起来有事件但指标不可算”
    events, thr = pick_jump_events(
        df,
        freq=cfg.freq,
        q_col="abs_dQ_abs_per_hour",
        quantile=float(cfg.event_quantile),
        min_gap_windows=max(3, int(cfg.roll_win) // 2),
        block_col="block_id",
        allowed_idx=eligible_both,
    )

    ac = event_study(df, events, col="Q_abs_rolling_ac1", pre=int(cfg.pre), block_col="block_id")
    var = event_study(df, events, col="Q_abs_rolling_var", pre=int(cfg.pre), block_col="block_id")
    if ac is None or var is None:
        raise SystemExit("H4：事件不足或连续块不足，无法生成 event-aligned 曲线。")

    xs = np.arange(-int(cfg.pre), 1)
    ac_mean, ac_lo, ac_hi, _ = ac
    var_mean, var_lo, var_hi, _ = var

    _style_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=cfg.fig_size, sharex=True)
    color = "#1f77b4"

    axes[0].plot(xs, ac_mean, color=color, lw=2.4)
    axes[0].fill_between(xs, ac_lo, ac_hi, color=color, alpha=0.18, linewidth=0)
    axes[0].axvline(0, color="gray", linestyle=":", lw=1.2, alpha=0.6)
    axes[0].set_xlabel("windows to jump")
    axes[0].set_ylabel(r"$AC1(|Q|)$")
    axes[0].tick_params(direction="in", top=True, right=True)

    axes[1].plot(xs, var_mean, color=color, lw=2.4)
    axes[1].fill_between(xs, var_lo, var_hi, color=color, alpha=0.18, linewidth=0)
    axes[1].axvline(0, color="gray", linestyle=":", lw=1.2, alpha=0.6)
    axes[1].set_xlabel("windows to jump")
    axes[1].set_ylabel(r"$\mathrm{Var}(|Q|)$")
    axes[1].tick_params(direction="in", top=True, right=True)

    # 简洁说明（事件数/阈值不进图；留给 caption/table）
    fig.tight_layout()

    out_pdf = ROOT / "Essay" / "figures" / "fig5c_h4_event_batch3.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig5" / "fig5c_h4_event_batch3_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"events={len(events)} thr={thr:.4g}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()

