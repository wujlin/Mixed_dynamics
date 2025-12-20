#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.empirical import EventClusterer, TimeCluster, UserTypeMapper, aggregate_time_series, load_topic_dataset  # noqa: E402
from src.empirical.time_series import TimeSeriesConfig, calculate_r_proxy  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_csv: Path
    annotations_jsonl: Path


def _freq_to_step_hours(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 3600.0)


_FREQ_UNIT_NORMALIZE = {
    "H": "h",
    "D": "d",
}


def _normalize_pandas_freq(freq: str) -> str:
    """
    pandas>=2.2 对大写 'H' 给出 FutureWarning（建议使用 'h'）。
    我们只对小时/天做大小写归一，避免把 'M'（月）误伤成 minutes。
    """
    s = str(freq or "").strip()
    if not s:
        return s
    m = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", s)
    if not m:
        return s
    n, unit = m.group(1), m.group(2)
    if unit in _FREQ_UNIT_NORMALIZE:
        return f"{n}{_FREQ_UNIT_NORMALIZE[unit]}"
    return s


def load_annotations_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    if not records:
        raise ValueError(f"标注文件为空：{path}")
    df = pd.DataFrame(records)
    if "mid" not in df.columns:
        raise ValueError(f"标注文件缺少 mid：{path}")
    df["mid"] = df["mid"].astype(str)
    keep = [c for c in ["mid", "emotion_class", "risk_class", "emotion_confidence", "risk_confidence"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["mid"]).reset_index(drop=True)


def load_and_merge(spec: DatasetSpec, *, mapper: UserTypeMapper, user_meta_path: Optional[Path | str] = None) -> pd.DataFrame:
    df_raw = load_topic_dataset(spec.dataset_csv, mapper=mapper, user_meta_path=user_meta_path)
    if "mid" not in df_raw.columns:
        raise ValueError(f"dataset 缺少 mid 列：{spec.dataset_csv}")
    df_raw["mid"] = df_raw["mid"].astype(str)
    df_raw = df_raw.drop_duplicates(subset=["mid"]).reset_index(drop=True)
    df_ann = load_annotations_jsonl(spec.annotations_jsonl)
    df = df_raw.merge(df_ann, on=["mid"], how="inner", validate="one_to_one")
    return df


def build_time_series(
    df: pd.DataFrame,
    *,
    freq: str,
    min_posts_public: int,
    time_start: Optional[str],
    time_end: Optional[str],
) -> pd.DataFrame:
    cfg = TimeSeriesConfig(freq=freq, min_posts=int(min_posts_public))
    ts = aggregate_time_series(df, config=cfg)
    ts["r_proxy"] = calculate_r_proxy(ts)
    ts = ts.sort_values("time_window").reset_index(drop=True)
    if time_start:
        ts = ts[ts["time_window"] >= pd.Timestamp(time_start)]
    if time_end:
        ts = ts[ts["time_window"] <= pd.Timestamp(time_end)]
    return ts.reset_index(drop=True)


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


_SEGMENT_FLOOR_UNITS = {
    "H": "h",  # pandas>=2.2 建议用小写 h
    "h": "h",
    "D": "D",
    "d": "D",
}


def _segment_start_time(ts: pd.Series, segment: str) -> pd.Series:
    """
    生成用于分段统计的“段起点”时间戳。

    说明：
    - 对 `2D/3D/12h` 这类“固定长度”窗口，优先用 `.dt.floor()` 做 **全局对齐**，
      避免 `.dt.to_period('2D')` 在 pandas 中出现“按天滑动标号”的行为，导致每段样本过少。
    - 对 `W/M/Q` 等非固定长度窗口，使用 `.dt.to_period().dt.to_timestamp()`。
    """
    s = str(segment or "").strip()
    if not s:
        raise ValueError("segment 不能为空")

    m = re.fullmatch(r"(\d+)\s*([HhDd])", s)
    if m:
        n, unit = m.group(1), m.group(2)
        unit_norm = _SEGMENT_FLOOR_UNITS.get(unit, unit)
        return ts.dt.floor(f"{n}{unit_norm}")

    return ts.dt.to_period(s).dt.to_timestamp()


def segment_metrics(df: pd.DataFrame, *, segment: str = "M", jump_q: float = 0.95) -> pd.DataFrame:
    x = df.dropna(subset=["time_window"]).copy()
    x["seg"] = _segment_start_time(x["time_window"], segment)
    rows = []
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

        if "n_mainstream" in g_aq.columns and "n_wemedia" in g_aq.columns:
            nw = float(g_aq["n_wemedia"].fillna(0).sum())
            nm = float(g_aq["n_mainstream"].fillna(0).sum())
            ng = float(g_aq["n_government"].fillna(0).sum()) if "n_government" in g_aq.columns else 0.0
            denom = nw + nm + ng
            r_proxy_mean = float(nw / denom) if denom > 0 else np.nan
        else:
            r_proxy_mean = float(g_aq["r_proxy"].mean()) if "r_proxy" in g_aq.columns else np.nan

        rows.append(
            {
                "seg": seg,
                "n_windows_aq": int(len(g_aq)),
                "n_windows_jump": int(len(g_jump)),
                "a_mean": a_mean,
                "r_proxy_mean": r_proxy_mean,
                "volatility": float(g_aq["Q"].std()),
                "jump_q95": float(np.nanpercentile(g_jump["abs_dQ_abs_per_hour"].values, 100.0 * float(jump_q))),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("seg").reset_index(drop=True)


def safe_pearsonr(x: pd.Series, y: pd.Series):
    try:
        from scipy import stats

        m = x.notna() & y.notna()
        if int(m.sum()) < 5:
            return np.nan, np.nan
        r, p = stats.pearsonr(x[m].values, y[m].values)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def safe_spearmanr(x: pd.Series, y: pd.Series):
    try:
        from scipy import stats

        m = x.notna() & y.notna()
        if int(m.sum()) < 5:
            return np.nan, np.nan
        r, p = stats.spearmanr(x[m].values, y[m].values)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def partial_pearsonr(x: pd.Series, y: pd.Series, ctrl: pd.Series):
    m = x.notna() & y.notna() & ctrl.notna()
    if int(m.sum()) < 8:
        return np.nan, np.nan
    xv = x[m].values.astype(float)
    yv = y[m].values.astype(float)
    cv = ctrl[m].values.astype(float)
    X = np.column_stack([np.ones(len(cv)), cv])
    bx, *_ = np.linalg.lstsq(X, xv, rcond=None)
    by, *_ = np.linalg.lstsq(X, yv, rcond=None)
    xr = xv - X @ bx
    yr = yv - X @ by
    return safe_pearsonr(pd.Series(xr), pd.Series(yr))


def plot_basic(plt, ts: pd.DataFrame, title: str, out_path: Path) -> None:
    df = ts.sort_values("time_window").reset_index(drop=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["time_window"], df["Q"], lw=1)
    axes[0].set_ylabel("Q")
    axes[0].set_title(title)
    axes[1].plot(df["time_window"], df["a"], lw=1)
    axes[1].set_ylabel("a")
    axes[2].plot(df["time_window"], df["r_proxy"], lw=1)
    axes[2].set_ylabel("r_proxy")
    axes[2].set_xlabel("time")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_h1_h2_scatter(plt, seg: pd.DataFrame, *, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(seg["a_mean"], seg["jump_q95"], s=25, alpha=0.8)
    axes[0].set_xlabel("a_mean (segment)")
    axes[0].set_ylabel("jump_q95 (q95 |d|Q|/dt|)")
    axes[0].set_title(f"H1 ({title})")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(seg["r_proxy_mean"], seg["volatility"], s=25, alpha=0.8)
    axes[1].set_xlabel("r_proxy_mean (segment)")
    axes[1].set_ylabel("volatility (std(Q))")
    axes[1].set_title(f"H2 ({title})")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


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


def pick_jump_events(
    df: pd.DataFrame,
    *,
    freq: str,
    q_col: str = "abs_dQ_abs_per_hour",
    quantile: float = 0.95,
    min_gap_windows: int = 6,
    block_col: str = "block_id",
    allowed_idx: set[int] | None = None,
):
    x = df.dropna(subset=["time_window", q_col]).copy().sort_values("time_window")
    if allowed_idx is not None:
        if not allowed_idx:
            return [], np.nan, 0, 0
        x = x[x.index.isin(allowed_idx)]
    if len(x) < 30:
        return [], np.nan, int(len(x)), 0
    thr = float(x[q_col].quantile(float(quantile)))
    cand = x[x[q_col] >= thr]
    step_hours = _freq_to_step_hours(freq)
    events = []
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
    return events, thr, int(len(x)), int(len(cand))


def event_study(df: pd.DataFrame, event_idx: list[int], *, col: str, pre: int = 24, block_col: str = "block_id"):
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


def _event_tail_stat(curve, *, tail_k: int) -> float:
    if curve is None:
        return np.nan
    mean, _, _, _ = curve
    if len(mean) < int(tail_k) + 1:
        return np.nan
    # 取事件前最后 tail_k 个窗口（不含事件点 0）
    return float(np.mean(mean[-int(tail_k) - 1 : -1]))


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


def placebo_pvalue_for_event_curve(
    df: pd.DataFrame,
    real_events: list[int],
    *,
    col: str,
    pre: int,
    tail_k: int,
    iters: int,
    seed: int = 0,
    block_col: str = "block_id",
) -> dict:
    """
    H4 placebo（事件标签置换）：在同一团簇内、按 block_id 匹配事件数量，随机抽取非事件窗口作为 placebo。
    返回 one-sided p 值（real > placebo）。
    """
    if int(iters) <= 0 or not real_events:
        return {"p_one_sided": np.nan, "n_valid": 0, "real": np.nan, "placebo_mean": np.nan, "placebo_q95": np.nan}

    def is_valid_event(idx: int) -> bool:
        if idx - int(pre) < 0:
            return False
        w = df.loc[idx - int(pre) : idx, [col, block_col]]
        if w[block_col].nunique() != 1:
            return False
        if w[col].isna().any():
            return False
        return True

    valid_events = [int(i) for i in real_events if is_valid_event(int(i))]
    if not valid_events:
        return {"p_one_sided": np.nan, "n_valid": 0, "real": np.nan, "placebo_mean": np.nan, "placebo_q95": np.nan}

    real_curve = event_study(df, valid_events, col=col, pre=pre, block_col=block_col)
    real_stat = _event_tail_stat(real_curve, tail_k=tail_k)
    if np.isnan(real_stat):
        return {"p_one_sided": np.nan, "n_valid": 0, "real": np.nan, "placebo_mean": np.nan, "placebo_q95": np.nan}

    eligible = _eligible_indices_by_block(df, col=col, pre=pre, block_col=block_col)
    if not eligible:
        return {"p_one_sided": np.nan, "n_valid": 0, "real": float(real_stat), "placebo_mean": np.nan, "placebo_q95": np.nan}

    rng = np.random.default_rng(int(seed))
    ev_df = df.loc[valid_events, [block_col]].dropna()
    if ev_df.empty:
        return {"p_one_sided": np.nan, "n_valid": 0, "real": float(real_stat), "placebo_mean": np.nan, "placebo_q95": np.nan}

    counts = ev_df[block_col].astype(int).value_counts().to_dict()
    real_set = set(int(i) for i in valid_events)

    stats = []
    for _ in range(int(iters)):
        sample = []
        ok = True
        for b, k in counts.items():
            cand = [i for i in eligible.get(int(b), []) if i not in real_set]
            if len(cand) < int(k):
                ok = False
                break
            pick = rng.choice(cand, size=int(k), replace=False).tolist()
            sample.extend(int(x) for x in pick)
        if not ok:
            continue
        curve = event_study(df, sample, col=col, pre=pre, block_col=block_col)
        s = _event_tail_stat(curve, tail_k=tail_k)
        if not np.isnan(s):
            stats.append(float(s))

    if not stats:
        return {"p_one_sided": np.nan, "n_valid": 0, "real": float(real_stat), "placebo_mean": np.nan, "placebo_q95": np.nan}

    arr = np.asarray(stats, dtype=float)
    # one-sided: P(placebo >= real)
    p = (float(np.sum(arr >= float(real_stat))) + 1.0) / (float(len(arr)) + 1.0)
    return {
        "p_one_sided": float(p),
        "n_valid": int(len(arr)),
        "real": float(real_stat),
        "placebo_mean": float(np.mean(arr)),
        "placebo_q95": float(np.quantile(arr, 0.95)),
    }


def plot_h4_eventstudy(plt, pre: int, ac, var, *, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(-int(pre), 1)

    if ac is not None:
        mean, lo, hi, _ = ac
        axes[0].plot(xs, mean, lw=2)
        axes[0].fill_between(xs, lo, hi, alpha=0.2)
    axes[0].axvline(0, color="gray", linestyle="--", lw=1)
    axes[0].set_title(f"H4: AC1(|Q|) before jumps ({title})")
    axes[0].set_xlabel("windows to jump")
    axes[0].set_ylabel("AC1")
    axes[0].grid(True, alpha=0.2)

    if var is not None:
        mean, lo, hi, _ = var
        axes[1].plot(xs, mean, lw=2)
        axes[1].fill_between(xs, lo, hi, alpha=0.2)
    axes[1].axvline(0, color="gray", linestyle="--", lw=1)
    axes[1].set_title(f"H4: Var(|Q|) before jumps ({title})")
    axes[1].set_xlabel("windows to jump")
    axes[1].set_ylabel("Var")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_one(
    plt,
    ts: pd.DataFrame,
    *,
    name: str,
    title: Optional[str] = None,
    freq: str,
    segment: str,
    jump_quantile: float,
    event_quantile: float,
    roll_win: int,
    pre: int,
    event_on_eligible: str = "both",
    fig_dir: Path,
    plot: bool = True,
    placebo_iters: int = 0,
    placebo_tail_k: int = 6,
    placebo_seed: int = 0,
):
    title = title or name
    stats: dict = {
        "name": name,
        "title": title,
        "segment": segment,
    }
    df = add_window_metrics(ts, freq=freq)
    seg = segment_metrics(df, segment=segment, jump_q=jump_quantile)

    if not seg.empty:
        r1, p1 = safe_pearsonr(seg["a_mean"], seg["jump_q95"])
        rs1, ps1 = safe_spearmanr(seg["a_mean"], seg["jump_q95"])
        r2, p2 = safe_pearsonr(seg["r_proxy_mean"], seg["volatility"])
        rs2, ps2 = safe_spearmanr(seg["r_proxy_mean"], seg["volatility"])
        rp1, pp1 = partial_pearsonr(seg["a_mean"], seg["jump_q95"], seg["n_windows_jump"])

        print(f"\n[{name}] segments={len(seg)}")
        print(f"H1: corr(a_mean, jump_q95)={r1:.3f} (p={p1:.4g}); spearman={rs1:.3f} (p={ps1:.4g})")
        if not np.isnan(rp1):
            print(f"H1(partial, ctrl=n_windows_jump): r={rp1:.3f} (p={pp1:.4g})")
        print(f"H2: corr(r_proxy_mean, volatility)={r2:.3f} (p={p2:.4g}); spearman={rs2:.3f} (p={ps2:.4g})")

        stats.update(
            {
                "n_segments": int(len(seg)),
                "h1_pearson_r": r1,
                "h1_pearson_p": p1,
                "h1_spearman_r": rs1,
                "h1_spearman_p": ps1,
                "h1_partial_r": rp1,
                "h1_partial_p": pp1,
                "h2_pearson_r": r2,
                "h2_pearson_p": p2,
                "h2_spearman_r": rs2,
                "h2_spearman_p": ps2,
            }
        )

        if plot:
            plot_h1_h2_scatter(plt, seg, title=title, out_path=fig_dir / f"fig7b_h1_h2_scatter_{name}_{freq.lower()}.png")
    else:
        print(f"\n[{name}] 段内有效样本不足，跳过 H1-H3（可降低 MIN_POSTS_PUBLIC 或放宽 TIME_START）")
        stats.update({"n_segments": 0})

    # H4
    df = add_block_rolling(df, col="Q_abs", window=int(roll_win), block_col="block_id")
    block_sizes = df.groupby("block_id").size() if "block_id" in df.columns else pd.Series(dtype=int)
    n_blocks = int(block_sizes.shape[0]) if not block_sizes.empty else 0
    max_block_len = int(block_sizes.max()) if n_blocks > 0 else 0

    eligible_ac = _eligible_indices_by_block(df, col="Q_abs_rolling_ac1", pre=pre, block_col="block_id")
    eligible_var = _eligible_indices_by_block(df, col="Q_abs_rolling_var", pre=pre, block_col="block_id")
    eligible_ac_set = {i for xs in eligible_ac.values() for i in xs}
    eligible_var_set = {i for xs in eligible_var.values() for i in xs}
    eligible_both_set = eligible_ac_set & eligible_var_set

    mode = str(event_on_eligible or "both").strip().lower()
    allowed_idx: set[int] | None
    if mode == "none":
        allowed_idx = None
    elif mode == "ac":
        allowed_idx = eligible_ac_set
    elif mode == "var":
        allowed_idx = eligible_var_set
    else:
        allowed_idx = eligible_both_set

    events, thr, n_pool, n_cand = pick_jump_events(
        df,
        freq=freq,
        q_col="abs_dQ_abs_per_hour",
        quantile=float(event_quantile),
        min_gap_windows=max(3, roll_win // 2),
        allowed_idx=allowed_idx,
    )
    step_hours = _freq_to_step_hours(freq)
    print(
        f"[{name}] H4: roll_win={roll_win} (~{roll_win*step_hours:.0f}h), pre={pre} (~{pre*step_hours:.0f}h), "
        f"blocks={n_blocks} max_block={max_block_len}, eligible(ac/var/both)={len(eligible_ac_set)}/{len(eligible_var_set)}/{len(eligible_both_set)}, "
        f"events={len(events)} (q={event_quantile}, thr={thr:.4g}, pool={n_pool}, cand={n_cand}, mode={mode})"
    )

    ac = event_study(df, events, col="Q_abs_rolling_ac1", pre=pre)
    var = event_study(df, events, col="Q_abs_rolling_var", pre=pre)
    used_ac = int(ac[3].shape[0]) if ac is not None else 0
    used_var = int(var[3].shape[0]) if var is not None else 0
    stats.update(
        {
            "h4_event_on_eligible": str(mode),
            "h4_n_blocks": int(n_blocks),
            "h4_max_block_len": int(max_block_len),
            "h4_eligible_ac": int(len(eligible_ac_set)),
            "h4_eligible_var": int(len(eligible_var_set)),
            "h4_eligible_both": int(len(eligible_both_set)),
            "h4_event_quantile": float(event_quantile),
            "h4_event_thr": float(thr) if not np.isnan(thr) else np.nan,
            "h4_event_pool": int(n_pool),
            "h4_event_cand": int(n_cand),
            "h4_events": int(len(events)),
            "h4_events_used_ac": used_ac,
            "h4_events_used_var": used_var,
        }
    )
    print(f"[{name}] H4 used events: ac={used_ac}/{len(events)}, var={used_var}/{len(events)}")

    if int(placebo_iters) > 0:
        ac_p = placebo_pvalue_for_event_curve(
            df,
            events,
            col="Q_abs_rolling_ac1",
            pre=pre,
            tail_k=int(placebo_tail_k),
            iters=int(placebo_iters),
            seed=int(placebo_seed),
        )
        var_p = placebo_pvalue_for_event_curve(
            df,
            events,
            col="Q_abs_rolling_var",
            pre=pre,
            tail_k=int(placebo_tail_k),
            iters=int(placebo_iters),
            seed=int(placebo_seed) + 1,
        )
        stats.update(
            {
                "h4_placebo_iters": int(placebo_iters),
                "h4_placebo_tail_k": int(placebo_tail_k),
                "h4_placebo_ac1_p": ac_p["p_one_sided"],
                "h4_placebo_ac1_n": ac_p["n_valid"],
                "h4_placebo_ac1_real": ac_p["real"],
                "h4_placebo_ac1_mean": ac_p["placebo_mean"],
                "h4_placebo_ac1_q95": ac_p["placebo_q95"],
                "h4_placebo_var_p": var_p["p_one_sided"],
                "h4_placebo_var_n": var_p["n_valid"],
                "h4_placebo_var_real": var_p["real"],
                "h4_placebo_var_mean": var_p["placebo_mean"],
                "h4_placebo_var_q95": var_p["placebo_q95"],
            }
        )
        print(
            f"[{name}] H4 placebo: "
            f"AC1 p={ac_p['p_one_sided']:.3g} (real={ac_p['real']:.4g}, placebo_mean={ac_p['placebo_mean']:.4g}); "
            f"Var p={var_p['p_one_sided']:.3g} (real={var_p['real']:.4g}, placebo_mean={var_p['placebo_mean']:.4g})"
        )

    if plot:
        plot_h4_eventstudy(plt, pre, ac, var, title=title, out_path=fig_dir / f"fig7c_h4_eventstudy_{name}_{freq.lower()}.png")

    return stats


def main():
    ap = argparse.ArgumentParser(description="Note07 经验验证：H1-H4（支持多批次数据集与团簇/Placebo）")
    ap.add_argument(
        "--datasets",
        default="master,batch3",
        help="要分析的数据集：逗号分隔，可选 master,batch1,batch1_concept,batch1_base,batch3,batch4（默认 master,batch3）",
    )
    ap.add_argument("--freq", default="4H", help="时间聚合窗口，例如 1H/4H/1D")
    ap.add_argument("--min-posts-public", type=int, default=5, help="每个窗口 public 帖子阈值（低于此值 a/Q 置 NaN）")
    ap.add_argument("--time-start", default="", help="起始时间（含），例如 2023-01-01；空字符串表示不截断（建议显式设置以保证可复现）")
    ap.add_argument("--time-end", default="", help="结束时间（含），例如 2024-12-31；空字符串表示不截断")
    ap.add_argument("--segment", default="M", help="段内统计粒度：M(月)/W(周)等")
    ap.add_argument("--jump-quantile", type=float, default=0.95, help="H1 jump 指标用的分位数（默认 0.95）")
    ap.add_argument("--event-quantile", type=float, default=0.95, help="H4 事件点（jump windows）分位数阈值（默认 0.95）")
    ap.add_argument(
        "--event-on-eligible",
        default="both",
        choices=["none", "ac", "var", "both"],
        help="H4 事件点只在可评估窗口上选：none=全量候选；ac/var=仅要求对应指标可用；both=同时要求 AC1/Var 可用（推荐）",
    )
    ap.add_argument("--roll-win", type=int, default=12, help="H4 rolling 窗口（单位：窗口数）")
    ap.add_argument("--pre", type=int, default=24, help="H4 事件对齐回看长度（单位：窗口数）")
    ap.add_argument("--cluster", action="store_true", help="启用方案B：按时间团簇分别分析（基于 n_public 密度）")
    ap.add_argument("--cluster-only", action="store_true", help="只输出团簇结果（跳过全时段汇总图）")
    ap.add_argument("--cluster-roll-days", type=float, default=14.0, help="团簇平滑窗口（天）")
    ap.add_argument("--cluster-quantile", type=float, default=0.9, help="团簇阈值分位数（基于 rolling mean(n_public)）")
    ap.add_argument("--cluster-min-days", type=float, default=21.0, help="团簇最短长度（天）")
    ap.add_argument("--cluster-merge-gap-days", type=float, default=7.0, help="团簇合并允许的最大间隔（天）")
    ap.add_argument("--cluster-max", type=int, default=0, help="最多保留多少个团簇（0 表示不限制，按时间顺序输出）")
    ap.add_argument("--cluster-segment", default="W", help="团簇内段内统计粒度（默认 W；避免团簇太短导致段数不足）")
    ap.add_argument("--cluster-grid", action="store_true", help="运行稳健性栅格（roll_days×quantile），只输出 CSV（不画图）")
    ap.add_argument("--grid-roll-days", default="7,14,21", help="栅格：roll_days 列表（逗号分隔）")
    ap.add_argument("--grid-quantiles", default="0.85,0.9,0.95", help="栅格：quantile 列表（逗号分隔）")
    ap.add_argument(
        "--grid-event-quantiles",
        default="",
        help="栅格：H4 事件阈值 event_quantile 列表（逗号分隔）；空字符串表示只用 --event-quantile",
    )
    ap.add_argument("--placebo-iters", type=int, default=0, help="H4 placebo 重复次数（0=关闭；建议 1000~5000）")
    ap.add_argument("--placebo-tail-k", type=int, default=6, help="H4 placebo 统计量：事件前最后 k 个窗口的均值")
    ap.add_argument("--placebo-seed", type=int, default=0, help="H4 placebo 随机种子")
    ap.add_argument(
        "--user-meta",
        default="",
        help="可选：用户元信息 CSV/JSONL（uid->verify_typ/user_type），用于补齐用户类型（尤其是缺少 verify_typ 的数据源）",
    )
    ap.add_argument(
        "--batch4-csv",
        default=str(ROOT / "outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_loose.csv"),
        help="batch4 数据集 csv（默认上海候选池 loose）",
    )
    ap.add_argument(
        "--batch4-annotations",
        default=str(ROOT / "outputs/annotations/batches/batch_04_shanghai/new_batch4.jsonl"),
        help="batch4 标注 jsonl（默认 batch_04_shanghai/new_batch4.jsonl）",
    )
    args = ap.parse_args()

    args.freq = _normalize_pandas_freq(args.freq)

    time_start = args.time_start.strip() or None
    time_end = args.time_end.strip() or None
    user_meta = args.user_meta.strip() or None

    specs = {
        "master": DatasetSpec(
            name="master",
            dataset_csv=ROOT / "dataset/Topic_data/merged_topic_official.csv",
            annotations_jsonl=ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl",
        ),
        # batch1：单词条口径（严格）——#新冠后遗症# + 官媒补充（仅保留 content 命中“新冠后遗症”的官媒行）
        "batch1": DatasetSpec(
            name="batch1",
            dataset_csv=ROOT / "dataset/Topic_data/merged_topic_official_batch1_strict.csv",
            annotations_jsonl=ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl",
        ),
        # batch1_concept：单词条口径（概念扩展）——#新冠后遗症# + 官媒补充（命中“长新冠/后新冠/Long COVID/PASC”等）
        "batch1_concept": DatasetSpec(
            name="batch1_concept",
            dataset_csv=ROOT / "dataset/Topic_data/merged_topic_official_batch1_concept.csv",
            annotations_jsonl=ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl",
        ),
        # batch1_base：只用 #新冠后遗症# 原始数据（不含官媒补充），用于对照“官媒补充是否改变可检验性”
        "batch1_base": DatasetSpec(
            name="batch1_base",
            dataset_csv=ROOT / "dataset/Topic_data/#新冠后遗症#_filtered.csv",
            annotations_jsonl=ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl",
        ),
        "batch3": DatasetSpec(
            name="batch3",
            dataset_csv=ROOT / "outputs/annotations/intermediate/to_annotate_batch3_clean.csv",
            annotations_jsonl=ROOT / "outputs/annotations/batches/batch_03_expanded/new_batch3.jsonl",
        ),
        "batch4": DatasetSpec(
            name="batch4",
            dataset_csv=Path(args.batch4_csv),
            annotations_jsonl=Path(args.batch4_annotations),
        ),
    }

    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    run_tag = "_".join(selected) if selected else "none"
    allowed = set(specs.keys())
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise ValueError(f"--datasets 包含未知值：{unknown}，可选：{sorted(allowed)}")

    selected_specs = [specs[k] for k in selected]
    for s in selected_specs:
        if not s.dataset_csv.exists():
            raise FileNotFoundError(s.dataset_csv)
        if not s.annotations_jsonl.exists():
            raise FileNotFoundError(s.annotations_jsonl)

    mapper = UserTypeMapper()
    dfs: dict[str, pd.DataFrame] = {}
    for s in selected_specs:
        dfs[s.name] = load_and_merge(s, mapper=mapper, user_meta_path=user_meta)

    df_all = None
    if len(dfs) >= 2:
        df_all = pd.concat(list(dfs.values()), ignore_index=True).drop_duplicates(subset=["mid"]).reset_index(drop=True)

    print("config:", dict(freq=args.freq, min_posts_public=args.min_posts_public, time_start=time_start, time_end=time_end))
    print("datasets:", selected)
    print("user_meta:", user_meta or "(none)")
    print("merged rows:", {k: len(v) for k, v in dfs.items()} | ({"all": len(df_all)} if df_all is not None else {}))
    for k, v in dfs.items():
        print(f"time span ({k}):", v["publish_time"].min(), "~", v["publish_time"].max())

    ts_map: dict[str, pd.DataFrame] = {}
    for k, v in dfs.items():
        ts_map[k] = build_time_series(v, freq=args.freq, min_posts_public=args.min_posts_public, time_start=time_start, time_end=time_end)
    if df_all is not None:
        ts_map["all"] = build_time_series(df_all, freq=args.freq, min_posts_public=args.min_posts_public, time_start=time_start, time_end=time_end)

    print("valid windows (a notna):", {k: int(ts["a"].notna().sum()) for k, ts in ts_map.items()})

    out_dir = ROOT / "outputs/annotations/derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, ts in ts_map.items():
        ts.to_csv(out_dir / f"time_series_{k}_{args.freq.lower()}.csv", index=False)

    fig_dir = ROOT / "outputs/figs/empirical"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt = _ensure_matplotlib() if not args.cluster_grid else None

    if args.cluster_grid:
        roll_days_list = [float(x) for x in args.grid_roll_days.split(",") if x.strip()]
        quantiles_list = [float(x) for x in args.grid_quantiles.split(",") if x.strip()]
        if args.grid_event_quantiles.strip():
            event_q_list = [float(x) for x in args.grid_event_quantiles.split(",") if x.strip()]
        else:
            event_q_list = [float(args.event_quantile)]
        grid_rows = []

        for roll_days in roll_days_list:
            for q in quantiles_list:
                for tag, ts in ts_map.items():
                    clusterer = EventClusterer(
                        freq=args.freq,
                        density_col="n_public",
                        roll_days=float(roll_days),
                        quantile=float(q),
                        min_cluster_days=args.cluster_min_days,
                        merge_gap_days=args.cluster_merge_gap_days,
                        max_clusters=args.cluster_max,
                    )
                    clusters = clusterer.find_clusters(ts)
                    print(f"\n[grid] {tag} clusters={len(clusters)} (roll={roll_days}d, q={q})")
                    for c in clusters:
                        ts_slice = ts[(ts["time_window"] >= c.start) & (ts["time_window"] <= c.end)].reset_index(drop=True)
                        for event_q in event_q_list:
                            stats = run_one(
                                None,
                                ts_slice,
                                name=f"{tag}_grid",
                                title=f"{tag} cluster{c.cluster_id}: {c.start.date()} ~ {c.end.date()}",
                                freq=args.freq,
                                segment=args.cluster_segment,
                                jump_quantile=args.jump_quantile,
                                event_quantile=float(event_q),
                                roll_win=args.roll_win,
                                pre=args.pre,
                                event_on_eligible=args.event_on_eligible,
                                fig_dir=fig_dir,
                                plot=False,
                                placebo_iters=args.placebo_iters,
                                placebo_tail_k=args.placebo_tail_k,
                                placebo_seed=args.placebo_seed,
                            )
                            row = {
                                "dataset": tag,
                                "cluster_id": c.cluster_id,
                                "start": str(c.start),
                                "end": str(c.end),
                                "n_windows": c.n_windows,
                                "n_public_sum": c.n_public_sum,
                                "n_valid_a": c.n_valid_a,
                                "smooth_threshold": c.smooth_threshold,
                                "freq": args.freq,
                                "min_posts_public": args.min_posts_public,
                                "segment": args.cluster_segment,
                                "grid_event_quantile": float(event_q),
                                "cluster_roll_days": float(roll_days),
                                "cluster_quantile": float(q),
                                "cluster_min_days": float(args.cluster_min_days),
                                "cluster_merge_gap_days": float(args.cluster_merge_gap_days),
                            }
                            row.update(stats)
                            grid_rows.append(row)

        if grid_rows:
            pd.DataFrame(grid_rows).to_csv(out_dir / f"note07_cluster_grid_stats_{run_tag}_{args.freq.lower()}.csv", index=False)

        print("\nSaved:")
        print("-", out_dir)
        return

    if not args.cluster_only:
        for k in ts_map.keys():
            label = k
            if k == "all":
                label = f"all ({'+'.join(selected)})"
            plot_basic(plt, ts_map[k], f"{label}: Q/a/r_proxy ({args.freq})", fig_dir / f"fig7a_{k}_basic_{args.freq.lower()}.png")

        for k in ts_map.keys():
            title = f"{k} (all time range)"
            if k == "all":
                title = f"all ({'+'.join(selected)}) (all time range)"
            run_one(
                plt,
                ts_map[k],
                name=k,
                title=title,
                freq=args.freq,
                segment=args.segment,
                jump_quantile=args.jump_quantile,
                event_quantile=args.event_quantile,
                roll_win=args.roll_win,
                pre=args.pre,
                event_on_eligible=args.event_on_eligible,
                fig_dir=fig_dir,
            )

    if args.cluster:
        cluster_rows = []
        for tag, ts in ts_map.items():
            if tag == "all" and df_all is None:
                continue
            clusterer = EventClusterer(
                freq=args.freq,
                density_col="n_public",
                roll_days=args.cluster_roll_days,
                quantile=args.cluster_quantile,
                min_cluster_days=args.cluster_min_days,
                merge_gap_days=args.cluster_merge_gap_days,
                max_clusters=args.cluster_max,
            )
            clusters = clusterer.find_clusters(ts)
            print(f"\n[{tag}] clusters={len(clusters)} (roll={args.cluster_roll_days}d, q={args.cluster_quantile}, min={args.cluster_min_days}d)")
            for c in clusters:
                title = f"{tag} cluster{c.cluster_id}: {c.start.date()} ~ {c.end.date()}"
                name = f"{tag}_c{c.cluster_id}"
                ts_slice = ts[(ts["time_window"] >= c.start) & (ts["time_window"] <= c.end)].reset_index(drop=True)
                plot_basic(plt, ts_slice, f"{title} ({args.freq})", fig_dir / f"fig7a_{name}_basic_{args.freq.lower()}.png")
                stats = run_one(
                    plt,
                    ts_slice,
                    name=name,
                    title=title,
                    freq=args.freq,
                    segment=args.cluster_segment,
                    jump_quantile=args.jump_quantile,
                    event_quantile=args.event_quantile,
                    roll_win=args.roll_win,
                    pre=args.pre,
                    event_on_eligible=args.event_on_eligible,
                    fig_dir=fig_dir,
                    plot=True,
                    placebo_iters=args.placebo_iters,
                    placebo_tail_k=args.placebo_tail_k,
                    placebo_seed=args.placebo_seed,
                )
                row = {
                    "dataset": tag,
                    "cluster_id": c.cluster_id,
                    "start": str(c.start),
                    "end": str(c.end),
                    "n_windows": c.n_windows,
                    "n_public_sum": c.n_public_sum,
                    "n_valid_a": c.n_valid_a,
                    "smooth_threshold": c.smooth_threshold,
                    "freq": args.freq,
                    "min_posts_public": args.min_posts_public,
                    "segment": args.cluster_segment,
                    "cluster_roll_days": float(args.cluster_roll_days),
                    "cluster_quantile": float(args.cluster_quantile),
                    "cluster_min_days": float(args.cluster_min_days),
                    "cluster_merge_gap_days": float(args.cluster_merge_gap_days),
                }
                row.update(stats)
                cluster_rows.append(row)
        if cluster_rows:
            dfc = pd.DataFrame(cluster_rows)
            dfc.to_csv(out_dir / f"note07_cluster_stats_{run_tag}_{args.freq.lower()}.csv", index=False)
            dfc[
                [
                    "dataset",
                    "cluster_id",
                    "start",
                    "end",
                    "n_windows",
                    "n_public_sum",
                    "n_valid_a",
                    "smooth_threshold",
                    "freq",
                    "min_posts_public",
                    "segment",
                    "cluster_roll_days",
                    "cluster_quantile",
                    "cluster_min_days",
                    "cluster_merge_gap_days",
                ]
            ].to_csv(out_dir / f"note07_time_clusters_{run_tag}_{args.freq.lower()}.csv", index=False)

    print("\nSaved:")
    print("-", out_dir)
    print("-", fig_dir)


if __name__ == "__main__":
    main()
