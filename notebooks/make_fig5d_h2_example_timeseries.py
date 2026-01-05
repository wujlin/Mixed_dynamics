"""
Fig 5d：Empirical micro example (H2) —— high r_proxy vs low r_proxy segment 的 Q(t) 对比。

目标：
- 用一个“代表性案例”补强 Fig 5b 的宏观散点关系（r_proxy ↔ volatility）
- 直观展示：UGC dominance 更高的 segment 往往伴随更大的波动幅度

口径：
- 数据：outputs/annotations/derived/time_series_batch3_12h.csv（freq=12H，主文 H2 口径）
- 段：segment=W（周）
- 选段：在高密度（n_windows_aq >= median）的 segment 内，按 r_proxy_mean 的 10/90 分位取 low/high（只基于 r_proxy，避免 cherry-pick）；
        并要求有足够连续的有效窗口，优先选择更长的连续块
- 画法：在同一坐标轴叠加两条 Q(t) 曲线

输出：
- Essay/figures/fig5d_h2_example_timeseries.pdf
- outputs/figs/fig5/fig5d_h2_example_timeseries_preview.png

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig5d_h2_example_timeseries.py
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import OKABE_ITO, FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig5dConfig:
    input_csv: Path = ROOT / "outputs" / "annotations" / "derived" / "time_series_batch3_12h.csv"
    freq: str = "12h"
    segment: str = "W"
    high_q: float = 0.90
    low_q: float = 0.10
    min_windows: int = 10
    min_run: int = 10
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

    df["dt_hours"] = df["time_window"].diff().dt.total_seconds() / 3600.0
    df.loc[df["dt_hours"] <= 0, "dt_hours"] = np.nan
    df["dt_ok"] = np.isclose(df["dt_hours"], step_hours)
    if len(df) > 0:
        df.loc[df.index[0], "dt_ok"] = False
    df.loc[~df["dt_ok"], "dt_hours"] = np.nan

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
        g_aq = g.dropna(subset=["Q"])
        if len(g_aq) < 10:
            continue
        # 口径与主文 Fig5b 一致：用窗口内 counts 求段内 r_proxy_mean
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
    density_cut: float,
) -> pd.Timestamp:
    if seg_df.empty:
        raise RuntimeError("无可用分段统计（seg_df 为空）。")
    merged = seg_df.merge(run_df, on="seg", how="left")
    merged["max_run"] = merged["max_run"].fillna(0).astype(int)
    merged = merged[merged["n_windows_aq"] >= float(density_cut)]
    merged = merged[(merged["n_windows_aq"] >= int(min_windows)) & (merged["max_run"] >= int(min_run))]
    merged = merged[np.isfinite(merged["r_proxy_mean"].values)]
    if merged.empty:
        raise RuntimeError("未找到满足连续窗口要求的 segment（或高密度过滤过严）。")

    thr = float(np.nanquantile(merged["r_proxy_mean"].values, float(q)))
    if side == "high":
        cand = merged[merged["r_proxy_mean"] >= thr].copy()
    elif side == "low":
        cand = merged[merged["r_proxy_mean"] <= thr].copy()
    else:
        raise ValueError("side 必须是 'high' 或 'low'")
    if cand.empty:
        raise RuntimeError("分位筛选后无候选 segment（可尝试调整 high_q/low_q）。")

    # 选取更“代表性”的案例：只基于 predictor（r_proxy）选择接近分位数阈值的段，并优先选择更长的连续块。
    cand["r_dist"] = (cand["r_proxy_mean"] - thr).abs()
    if side == "high":
        cand = cand.sort_values(["r_dist", "max_run", "r_proxy_mean"], ascending=[True, False, False])
    else:
        cand = cand.sort_values(["r_dist", "max_run", "r_proxy_mean"], ascending=[True, False, True])
    return pd.to_datetime(cand.iloc[0]["seg"])


def _extract_block(df: pd.DataFrame, *, seg: pd.Timestamp) -> pd.DataFrame:
    g = df[df["seg"] == seg].copy()
    g = g.dropna(subset=["Q"]).copy()
    if g.empty:
        raise RuntimeError(f"segment={seg} 无有效 Q 数据。")
    lens = g.groupby("block_id").size().sort_values(ascending=False)
    best_block = int(lens.index[0])
    out = g[g["block_id"] == best_block].copy().sort_values("time_window").reset_index(drop=True)
    return out


def main() -> None:
    cfg = Fig5dConfig()
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

    density_cut = float(seg_df["n_windows_aq"].median()) if not seg_df.empty else 0.0

    seg_high = _pick_segment(
        seg_df,
        run_df,
        q=cfg.high_q,
        side="high",
        min_windows=cfg.min_windows,
        min_run=cfg.min_run,
        density_cut=density_cut,
    )
    seg_low = _pick_segment(
        seg_df,
        run_df,
        q=cfg.low_q,
        side="low",
        min_windows=cfg.min_windows,
        min_run=cfg.min_run,
        density_cut=density_cut,
    )

    blk_high = _extract_block(df, seg=seg_high)
    blk_low = _extract_block(df, seg=seg_low)

    step_days = _freq_to_step_hours(cfg.freq) / 24.0
    x_high = np.arange(len(blk_high), dtype=float) * step_days
    x_low = np.arange(len(blk_low), dtype=float) * step_days
    y_high = blk_high["Q"].to_numpy(dtype=float)
    y_low = blk_low["Q"].to_numpy(dtype=float)
    # 为了突出“波动性”而非均值偏移：按段内均值做居中，std 不变
    y_high = y_high - float(np.nanmean(y_high))
    y_low = y_low - float(np.nanmean(y_low))

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    c_high = OKABE_ITO["vermillion"]
    c_low = OKABE_ITO["gray"]
    ax.plot(x_low, y_low, color=c_low, lw=2.2, zorder=2, label=r"Low $r_{\mathrm{proxy}}$")
    ax.plot(x_high, y_high, color=c_high, lw=2.4, zorder=3, label=r"High $r_{\mathrm{proxy}}$")

    ax.axhline(0.0, color="#666666", lw=1.2, zorder=1)
    ax.set_xlabel("Time within segment (days)")
    ax.set_ylabel(r"$Q-\langle Q\rangle$")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "d")

    # 图例放在 x 轴标题下方，避免在图内堆叠文字
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.30, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig5d_h2_example_timeseries.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig5" / "fig5d_h2_example_timeseries_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")
    print(f"High-r_proxy segment: {seg_high} (len={len(blk_high)})")
    print(f"Low-r_proxy segment:  {seg_low} (len={len(blk_low)})")


if __name__ == "__main__":
    main()
