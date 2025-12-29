"""
Fig 5a：经验验证 H1（All）——段内活跃度 a 与 jump 强度的关系。

口径（与 Note07 收敛稿一致）：
- 数据：outputs/annotations/derived/time_series_all_4h.csv（freq=4H）
- 段内统计：segment=W（周）
- jump 强度：段内 |d|Q|/dt|| 的 95 分位（q95）

可视化规范（对齐 Fig2–4）：
- Times New Roman / STIX 数学字体
- 无网格、线宽字号统一
- 输出 PDF（矢量）+ 预览 PNG

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig5a_h1_all.py
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

from src.plot_style import FIGSIZE_HALF, add_panel_label, apply_paper_style  # noqa: E402

@dataclass(frozen=True)
class Fig5aConfig:
    input_csv: Path = ROOT / "outputs" / "annotations" / "derived" / "time_series_all_4h.csv"
    freq: str = "4h"
    segment: str = "W"
    jump_q: float = 0.95
    fig_size: Tuple[float, float] = FIGSIZE_HALF
    n_boot: int = 2000
    seed: int = 0


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


def segment_metrics(df: pd.DataFrame, *, segment: str, jump_q: float) -> pd.DataFrame:
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
                "n_windows_aq": int(len(g_aq)),
                "n_windows_jump": int(len(g_jump)),
                "a_mean": a_mean,
                "jump_q95": float(np.nanpercentile(g_jump["abs_dQ_abs_per_hour"].values, 100.0 * float(jump_q))),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("seg").reset_index(drop=True)


def _bootstrap_fit_band(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if x.size != y.size or x.size < 3:
        raise ValueError("x/y 长度不一致或样本过少")

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


def main() -> None:
    cfg = Fig5aConfig()
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{cfg.input_csv}")

    ts = pd.read_csv(cfg.input_csv)
    if "time_window" not in ts.columns:
        raise ValueError("CSV 缺少 time_window 列")
    ts["time_window"] = pd.to_datetime(ts["time_window"])

    df = add_window_metrics(ts, freq=cfg.freq)
    seg = segment_metrics(df, segment=cfg.segment, jump_q=cfg.jump_q)
    if seg.empty:
        raise SystemExit("无可用分段数据（可能数据过稀疏或阈值过严）。")

    x = seg["a_mean"].to_numpy(dtype=float)
    y = seg["jump_q95"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    mean, lo, hi = _bootstrap_fit_band(x, y, grid, n_boot=cfg.n_boot, seed=cfg.seed)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    color = "#0072B2"  # Okabe–Ito blue
    ax.scatter(x, y, s=48, alpha=0.85, c=color, edgecolors="white", linewidths=0.6, zorder=3)
    ax.plot(grid, mean, color=color, lw=2.4, zorder=4)
    ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)

    ax.set_xlabel(r"Activity $a$ (segment mean)")
    ax.set_ylabel(r"Jump intensity (q95)")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "a")

    handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=7,
            color=color,
            markeredgecolor="white",
            label="Segments",
        ),
        mpl.lines.Line2D([], [], linestyle="-", color=color, lw=2.4, label="Fit"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.34, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig5a_activity_jump.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig5" / "fig5a_activity_jump_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
