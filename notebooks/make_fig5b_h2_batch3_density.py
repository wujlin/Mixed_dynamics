"""
Fig 5b：经验验证 H2（Batch3）——r_proxy 与波动性（std(Q)）的关系，并展示密度分组。

口径（与 Note07 收敛稿一致）：
- 数据：outputs/annotations/derived/time_series_batch3_4h.csv（freq=4h）
- 段内统计：segment=W（周）
- 密度：段内可用窗口数 n_windows_aq，按中位数二分（high/low）

可视化决策（PI + 本轮确认）：
- 图内不放相关系数/partial r（放 Table/Caption）
- 图内强调趋势：两组各自的线性拟合 + 95% CI
- Times New Roman、无网格、线宽字号对齐 Fig2–4
- 输出 PDF（矢量）+ 预览 PNG

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig5b_h2_batch3_density.py
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
class Fig5bConfig:
    input_csv: Path = ROOT / "outputs" / "annotations" / "derived" / "time_series_batch3_4h.csv"
    freq: str = "4h"
    segment: str = "W"
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


def segment_metrics(df: pd.DataFrame, *, segment: str = "W") -> pd.DataFrame:
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
    cfg = Fig5bConfig()
    if not cfg.input_csv.exists():
        raise FileNotFoundError(f"未找到输入 CSV：{cfg.input_csv}")

    ts = pd.read_csv(cfg.input_csv)
    if "time_window" not in ts.columns:
        raise ValueError("CSV 缺少 time_window 列")
    ts["time_window"] = pd.to_datetime(ts["time_window"])

    df = add_window_metrics(ts, freq=cfg.freq)
    seg = segment_metrics(df, segment=cfg.segment)
    if seg.empty:
        raise SystemExit("无可用分段数据（可能数据过稀疏或阈值过严）。")

    cut = float(seg["n_windows_aq"].median())
    seg["density_group"] = np.where(seg["n_windows_aq"] >= cut, "High density", "Low density")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=cfg.fig_size)

    colors = {"High density": "#0072B2", "Low density": "#D55E00"}  # Okabe–Ito
    linestyles = {"High density": "-", "Low density": "--"}

    for grp, g in seg.groupby("density_group", sort=False):
        x = g["r_proxy_mean"].to_numpy(dtype=float)
        y = g["volatility"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]
        if x.size < 3:
            continue

        c = colors.get(str(grp), "#333333")
        ax.scatter(
            x,
            y,
            s=48,
            alpha=0.85,
            c=c,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )

        grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        mean, lo, hi = _bootstrap_fit_band(x, y, grid, n_boot=cfg.n_boot, seed=cfg.seed + (0 if grp == "High density" else 17))
        ax.plot(grid, mean, color=c, lw=2.4, linestyle=linestyles.get(str(grp), "-"), zorder=4)
        ax.fill_between(grid, lo, hi, color=c, alpha=0.16, linewidth=0, zorder=2)

    ax.set_xlabel(r"User-generated dominance $r_{\mathrm{proxy}}$")
    ax.set_ylabel(r"Volatility")
    ax.tick_params(direction="in", top=True, right=True)
    add_panel_label(ax, "b")

    # 简洁图例：只区分两组点（拟合线与置信带在 caption 说明）
    handles = []
    labels = []
    for grp in ["High density", "Low density"]:
        c = colors[grp]
        ls = linestyles[grp]
        handles.append(
            mpl.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle=ls,
                color=c,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                lw=2.4,
            )
        )
        labels.append(grp)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    fig.subplots_adjust(left=0.22, right=0.96, bottom=0.34, top=0.96)

    out_pdf = ROOT / "Essay" / "figures" / "fig5b_media_volatility_density.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig5" / "fig5b_media_volatility_density_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
