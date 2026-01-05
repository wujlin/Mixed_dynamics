"""
Fig 5c：Empirical micro example (H1) —— high-activity vs low-activity segment 的 |Q|(t) 对比。

目标：
- 用一个“代表性案例”补强 Fig 5a 的宏观散点关系（activity ↔ jump）
- 直观展示：高 activity 段更容易出现更大的 |Q| 变化（jump）

口径：
- 数据：outputs/annotations/derived/time_series_all_4h.csv（freq=4H）
- 段：segment=W（周）
- 选段：在满足连续窗口要求的 segment 中，按段内 a_mean 的 10/90 分位选取 low/high（只基于 activity，避免 cherry-pick）
- 画法：叠加两条 |Q|(t) 曲线；用竖虚线标出各自的最大单步 jump（|Δ|Q|| 最大）的发生位置

输出：
- Essay/figures/fig5c_h1_example_timeseries.pdf
- outputs/figs/fig5/fig5c_h1_example_timeseries_preview.png

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig5c_h1_example_timeseries.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import OKABE_ITO, FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig5cConfig:
    input_csv: Path = ROOT / "outputs" / "annotations" / "derived" / "time_series_all_4h.csv"
    freq: str = "4h"
    segment: str = "W"
    high_q: float = 0.90
    low_q: float = 0.10
    min_windows: int = 20
    min_run: int = 20
    fig_size: Tuple[float, float] = FIGSIZE_HALF


_SEGMENT_FLOOR_UNITS = {"H": "h", "h": "h", "D": "D", "d": "D"}


def _freq_to_step_hours(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 3600.0)


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


def add_window_metrics(ts: pd.DataFrame, *, freq: str) -> pd.DataFrame:
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
    return df


def segment_metrics(df: pd.DataFrame, *, segment: str) -> pd.DataFrame:
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
        if "n_public" in g_aq.columns:
            w = g_aq["n_public"].fillna(0).astype(float).values
            a_mean = float(np.average(g_aq["a"].values, weights=w)) if float(w.sum()) > 0 else float(g_aq["a"].mean())
        else:
            a_mean = float(g_aq["a"].mean())
        rows.append(
            {
                "seg": seg,
                "a_mean": a_mean,
                "n_windows_aq": int(len(g_aq)),
                "jump_q95": float(np.nanpercentile(g_jump["abs_dQ_abs_per_hour"].values, 95.0)),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("seg").reset_index(drop=True) if not out.empty else out


def segment_max_run(df: pd.DataFrame, *, segment: str) -> pd.DataFrame:
    x = df.dropna(subset=["time_window"]).copy()
    x["seg"] = _segment_start_time(x["time_window"], segment)
    y = x.dropna(subset=["Q"]).copy()
    if y.empty:
        return pd.DataFrame(columns=["seg", "max_run", "best_block"])
    block_len = y.groupby(["seg", "block_id"]).size().reset_index(name="run_len")
    best = block_len.sort_values(["seg", "run_len"], ascending=[True, False]).drop_duplicates("seg")
    best = best.rename(columns={"run_len": "max_run", "block_id": "best_block"}).reset_index(drop=True)
    return best


def _pick_segment(
    seg_df: pd.DataFrame,
    run_df: pd.DataFrame,
    *,
    q: float,
    side: str,
    min_windows: int,
    min_run: int,
) -> pd.Timestamp:
    if seg_df.empty:
        raise RuntimeError("无可用分段统计（seg_df 为空）。")
    merged = seg_df.merge(run_df, on="seg", how="left")
    merged["max_run"] = merged["max_run"].fillna(0).astype(int)
    merged = merged[(merged["n_windows_aq"] >= int(min_windows)) & (merged["max_run"] >= int(min_run))]
    if merged.empty:
        raise RuntimeError("未找到满足连续窗口要求的 segment（可尝试降低 min_windows/min_run）。")

    if side == "high":
        target = float(np.nanquantile(merged["a_mean"].values, float(q)))
        cand = merged[merged["a_mean"] >= target].copy()
    elif side == "low":
        target = float(np.nanquantile(merged["a_mean"].values, float(q)))
        cand = merged[merged["a_mean"] <= target].copy()
    else:
        raise ValueError("side 必须是 'high' 或 'low'")
    if cand.empty:
        raise RuntimeError("分位筛选后无候选 segment（可尝试调整 high_q/low_q）。")

    # 选取更“代表性”的案例：只基于 predictor（activity）选择接近分位数阈值的段，并优先选择更长的连续块。
    cand["a_dist"] = (cand["a_mean"] - target).abs()
    if side == "high":
        cand = cand.sort_values(["a_dist", "max_run", "a_mean"], ascending=[True, False, False])
    else:
        cand = cand.sort_values(["a_dist", "max_run", "a_mean"], ascending=[True, False, True])
    return pd.to_datetime(cand.iloc[0]["seg"])


def _extract_block(df: pd.DataFrame, *, seg: pd.Timestamp) -> pd.DataFrame:
    g = df[df["seg"] == seg].copy()
    g = g.dropna(subset=["Q"]).copy()
    if g.empty:
        raise RuntimeError(f"segment={seg} 无有效 Q 数据。")
    # 取最长连续块
    lens = g.groupby("block_id").size().sort_values(ascending=False)
    best_block = int(lens.index[0])
    out = g[g["block_id"] == best_block].copy().sort_values("time_window").reset_index(drop=True)
    return out


def _find_max_jump_idx(block: pd.DataFrame) -> int | None:
    v = block["abs_dQ_abs_per_hour"].to_numpy(dtype=float)
    if not np.isfinite(v).any():
        return None
    return int(np.nanargmax(v))

def _max_jump_midpoint(x: np.ndarray, *, idx: int) -> float:
    return float((x[idx - 1] + x[idx]) / 2.0)


def main() -> None:
    cfg = Fig5cConfig()
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{cfg.input_csv}")

    ts = pd.read_csv(cfg.input_csv)
    if "time_window" not in ts.columns:
        raise ValueError("CSV 缺少 time_window 列")
    ts["time_window"] = pd.to_datetime(ts["time_window"])

    df = add_window_metrics(ts, freq=cfg.freq)
    df["seg"] = _segment_start_time(df["time_window"], cfg.segment)
    seg_df = segment_metrics(df, segment=cfg.segment)
    run_df = segment_max_run(df, segment=cfg.segment)

    seg_high = _pick_segment(seg_df, run_df, q=cfg.high_q, side="high", min_windows=cfg.min_windows, min_run=cfg.min_run)
    seg_low = _pick_segment(seg_df, run_df, q=cfg.low_q, side="low", min_windows=cfg.min_windows, min_run=cfg.min_run)

    blk_high = _extract_block(df, seg=seg_high)
    blk_low = _extract_block(df, seg=seg_low)

    # x 轴：从各自片段起点计时（天）
    step_days = _freq_to_step_hours(cfg.freq) / 24.0
    x_high = np.arange(len(blk_high), dtype=float) * step_days
    x_low = np.arange(len(blk_low), dtype=float) * step_days

    y_high = blk_high["Q_abs"].to_numpy(dtype=float)
    y_low = blk_low["Q_abs"].to_numpy(dtype=float)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    c_high = OKABE_ITO["blue"]
    c_low = OKABE_ITO["gray"]
    ax.plot(x_low, y_low, color=c_low, lw=2.2, zorder=2, label="Low activity")
    ax.plot(x_high, y_high, color=c_high, lw=2.4, zorder=3, label="High activity")

    # 用竖虚线标出各自的“最大单步 jump”（|Δ|Q|| 最大）发生位置，避免在图内写文字
    idx_h = _find_max_jump_idx(blk_high)
    if idx_h is not None and idx_h > 0:
        ax.axvline(
            _max_jump_midpoint(x_high, idx=idx_h),
            color=c_high,
            lw=1.6,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.55,
            zorder=1.5,
            label="_nolegend_",
        )
    idx_l = _find_max_jump_idx(blk_low)
    if idx_l is not None and idx_l > 0:
        ax.axvline(
            _max_jump_midpoint(x_low, idx=idx_l),
            color=c_low,
            lw=1.6,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.55,
            zorder=1.4,
            label="_nolegend_",
        )

    ax.axhline(0.0, color="#666666", lw=1.2, zorder=1)
    ax.set_xlabel("Time within segment (days)")
    ax.set_ylabel(r"Polarization magnitude $|Q|$")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "c")

    # 图例放在 x 轴标题下方，避免在图内堆叠文字
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
    )

    # 留出左侧空间给 panel label（与其它图一致）
    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.30, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig5c_h1_example_timeseries.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig5" / "fig5c_h1_example_timeseries_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")
    print(f"High-a segment: {seg_high} (len={len(blk_high)})")
    print(f"Low-a segment:  {seg_low} (len={len(blk_low)})")


if __name__ == "__main__":
    main()
