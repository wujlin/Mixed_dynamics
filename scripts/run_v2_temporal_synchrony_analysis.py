#!/usr/bin/env python3
"""
V2 时间同步性分析（替代 Granger）

输出到 outputs/v2_content_analysis/time_response/：
1) ccf_values.csv / ccf_summary.json / fig_ccf_with_ci.png
2) ccf_by_segment.csv / fig_ccf_by_segment.png
3) event_coincidence.csv / event_coincidence_summary.json
   + event_coincidence_sensitivity.csv / event_coincidence_sensitivity_summary.json
4) fig_temporal_synchrony_combined.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 CCF + 事件共现时间同步性分析")
    p.add_argument(
        "--input-csv",
        default="outputs/v2_content_analysis/time_response/daily_risk_counts.csv",
        help="日级风险计数输入（需含 date/mainstream_risk_count/wemedia_risk_count）",
    )
    p.add_argument("--out-dir", default="outputs/v2_content_analysis/time_response", help="输出目录")
    p.add_argument("--max-lag", type=int, default=14, help="CCF lag 范围 [-max_lag, max_lag]")
    p.add_argument("--n-perm", type=int, default=1000, help="置换次数")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性阈值（用于 CI）")
    p.add_argument("--segment-split", default="2022-03", help="分段切割月份（YYYY-MM），该月作为剔除窗口")
    p.add_argument("--burst-windows", default="0,1,2,3", help="事件共现窗口，逗号分隔")
    p.add_argument(
        "--burst-sensitivity-multipliers",
        default="1,1.5,2",
        help="burst 阈值敏感性倍数（基于 mean_nonzero），逗号分隔",
    )
    p.add_argument("--block-size", type=int, default=7, help="block permutation 的块长度（天）")
    p.add_argument("--seed", type=int, default=20260302, help="随机种子")
    return p.parse_args()


def zscore(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - m) / s


def lagged_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag > 0:
        xa = x[:-lag]
        ya = y[lag:]
    elif lag < 0:
        k = -lag
        xa = x[k:]
        ya = y[:-k]
    else:
        xa = x
        ya = y

    if len(xa) < 3:
        return np.nan
    if np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return np.nan
    return float(np.corrcoef(xa, ya)[0, 1])


def compute_ccf(x: np.ndarray, y: np.ndarray, lags: np.ndarray) -> np.ndarray:
    return np.array([lagged_corr(x, y, int(lag)) for lag in lags], dtype=float)


def block_permute(arr: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr.copy()
    b = max(1, int(block_size))
    blocks = [arr[i : i + b] for i in range(0, n, b)]
    order = rng.permutation(len(blocks))
    out = np.concatenate([blocks[i] for i in order], axis=0)
    return out[:n]


def ccf_with_permutation_ci(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    lags: np.ndarray,
    n_perm: int,
    alpha: float,
    block_size: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    x = zscore(x_raw)
    y = zscore(y_raw)
    obs = compute_ccf(x, y, lags)

    perm_mat = np.zeros((int(n_perm), len(lags)), dtype=float)
    for i in range(int(n_perm)):
        yp = block_permute(y, block_size=block_size, rng=rng)
        perm_mat[i, :] = compute_ccf(x, yp, lags)

    lo = np.nanquantile(perm_mat, alpha / 2.0, axis=0)
    hi = np.nanquantile(perm_mat, 1.0 - alpha / 2.0, axis=0)
    sig = (obs < lo) | (obs > hi)

    out = pd.DataFrame(
        {
            "lag": lags.astype(int),
            "ccf": obs.astype(float),
            "ci_lower": lo.astype(float),
            "ci_upper": hi.astype(float),
            "significant": sig.astype(bool),
        }
    )

    idx_peak = int(np.nanargmax(np.abs(obs))) if np.isfinite(obs).any() else 0
    lag_peak = int(lags[idx_peak]) if len(lags) else 0
    ccf_peak = float(obs[idx_peak]) if len(obs) else np.nan

    pos = obs[lags > 0]
    neg = obs[lags < 0]
    sum_pos = float(np.nansum(pos)) if len(pos) else np.nan
    sum_neg = float(np.nansum(neg)) if len(neg) else np.nan
    sum_abs_pos = float(np.nansum(np.abs(pos))) if len(pos) else np.nan
    sum_abs_neg = float(np.nansum(np.abs(neg))) if len(neg) else np.nan

    summary = {
        "n_days": int(len(x_raw)),
        "n_perm": int(n_perm),
        "alpha": float(alpha),
        "max_lag": int(np.max(np.abs(lags))) if len(lags) else 0,
        "peak_lag": lag_peak,
        "peak_ccf": ccf_peak,
        "sum_ccf_positive_lags": sum_pos,
        "sum_ccf_negative_lags": sum_neg,
        "sum_abs_ccf_positive_lags": sum_abs_pos,
        "sum_abs_ccf_negative_lags": sum_abs_neg,
        "asymmetry_abs_ratio_pos_over_neg": float(sum_abs_pos / sum_abs_neg) if (sum_abs_neg and sum_abs_neg > 0) else np.nan,
        "n_significant_lags": int(np.nansum(sig)),
    }
    return out, summary


def parse_segment_split(split_ym: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # 例：split=2022-03 -> pre: <=2022-02-28, post: >=2022-04-01（剔除2022-03）
    split_start = pd.to_datetime(f"{split_ym}-01")
    pre_end = split_start - pd.Timedelta(days=1)
    post_start = split_start + pd.offsets.MonthBegin(1)
    return pre_end, post_start


def classify_burst(arr: np.ndarray, multiplier: float = 1.0) -> Tuple[np.ndarray, float, str, float]:
    x = np.asarray(arr, dtype=float)
    nz = x[x > 0]
    mean_nonzero = float(np.mean(nz)) if len(nz) else 0.0
    if mean_nonzero < 1.0:
        threshold = 1.0
        rule = ">=1_fallback"
        burst = x >= 1.0
    else:
        m = float(multiplier)
        threshold = m * mean_nonzero
        rule = f">{m:g}*mean_nonzero"
        burst = x > threshold
    return burst.astype(bool), float(threshold), rule, float(mean_nonzero)


def event_coincidence(
    trigger_burst: np.ndarray,
    response_burst: np.ndarray,
    windows: Iterable[int],
    direction: str,
) -> pd.DataFrame:
    t = np.asarray(trigger_burst, dtype=bool)
    r = np.asarray(response_burst, dtype=bool)
    n_days = len(t)
    trigger_idx = np.where(t)[0]
    n_trigger = int(len(trigger_idx))
    n_resp = int(np.sum(r))

    rows: List[Dict[str, object]] = []
    for w in windows:
        ww = int(w)
        hits = 0
        if n_trigger > 0:
            for ti in trigger_idx:
                lo = max(0, ti - ww)
                hi = min(n_days, ti + ww + 1)
                if np.any(r[lo:hi]):
                    hits += 1
        miss = int(n_trigger - hits)
        p_obs = float(hits / n_trigger) if n_trigger > 0 else np.nan
        p_base = float(min(1.0, (n_resp * (2 * ww + 1)) / max(1, n_days))) if n_days > 0 else np.nan

        exp_hits = int(round(n_trigger * p_base)) if n_trigger > 0 else 0
        exp_hits = min(max(exp_hits, 0), n_trigger)
        exp_miss = int(max(0, n_trigger - exp_hits))

        if n_trigger > 0:
            table = np.array([[hits, miss], [exp_hits, exp_miss]], dtype=int)
            try:
                odds_ratio, fisher_p = stats.fisher_exact(table)
                odds_ratio = float(odds_ratio)
                fisher_p = float(fisher_p)
            except Exception:
                odds_ratio, fisher_p = np.nan, np.nan
        else:
            odds_ratio, fisher_p = np.nan, np.nan

        rows.append(
            {
                "direction": direction,
                "window": ww,
                "n_trigger": n_trigger,
                "n_hits": int(hits),
                "p_observed": p_obs,
                "p_baseline": p_base,
                "fisher_p": fisher_p,
                "odds_ratio": odds_ratio,
                "n_response_burst_days": n_resp,
                "n_days": n_days,
            }
        )
    return pd.DataFrame(rows)


def plot_ccf_with_ci(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = df["lag"].values
    y = df["ccf"].values
    lo = df["ci_lower"].values
    hi = df["ci_upper"].values
    ax.plot(x, y, color="#1f77b4", lw=1.8, marker="o", ms=3, label="CCF")
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.18, label="95% permutation CI")
    ax.axhline(0.0, color="#666", lw=1.0, alpha=0.8)
    ax.axvline(0.0, color="#666", lw=1.0, alpha=0.8)
    sig = df[df["significant"]]
    if len(sig):
        ax.scatter(sig["lag"], sig["ccf"], color="#d62728", s=24, zorder=4, label="p<0.05")
    ax.set_xlabel("Lag (days)  [lag>0: mainstream leads]")
    ax.set_ylabel("CCF")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_csv = (ROOT / args.input_csv).resolve() if not Path(args.input_csv).is_absolute() else Path(args.input_csv)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(input_csv)
    required = {"date", "mainstream_risk_count", "wemedia_risk_count"}
    miss = required - set(d.columns)
    if miss:
        raise ValueError(f"输入缺少列: {sorted(miss)}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["mainstream_risk_count"] = pd.to_numeric(d["mainstream_risk_count"], errors="coerce").fillna(0.0)
    d["wemedia_risk_count"] = pd.to_numeric(d["wemedia_risk_count"], errors="coerce").fillna(0.0)
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    x_raw = d["mainstream_risk_count"].values.astype(float)
    y_raw = d["wemedia_risk_count"].values.astype(float)

    lags = np.arange(-int(args.max_lag), int(args.max_lag) + 1)
    rng_full = np.random.default_rng(int(args.seed))

    # Step 1: full CCF
    ccf_df, ccf_summary = ccf_with_permutation_ci(
        x_raw,
        y_raw,
        lags=lags,
        n_perm=int(args.n_perm),
        alpha=float(args.alpha),
        block_size=int(args.block_size),
        rng=rng_full,
    )
    ccf_df.to_csv(out_dir / "ccf_values.csv", index=False, encoding="utf-8-sig")
    (out_dir / "ccf_summary.json").write_text(json.dumps(ccf_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_ccf_with_ci(ccf_df, out_dir / "fig_ccf_with_ci.png", "CCF: mainstream vs wemedia risk counts")

    # 稀疏稳健性：仅保留至少一方非零日期
    nz_mask = (x_raw > 0) | (y_raw > 0)
    if int(np.sum(nz_mask)) >= (2 * int(args.max_lag) + 10):
        rng_nz = np.random.default_rng(int(args.seed) + 11)
        ccf_nz, ccf_nz_summary = ccf_with_permutation_ci(
            x_raw[nz_mask],
            y_raw[nz_mask],
            lags=lags,
            n_perm=int(args.n_perm),
            alpha=float(args.alpha),
            block_size=int(args.block_size),
            rng=rng_nz,
        )
        ccf_nz.to_csv(out_dir / "ccf_values_nonzero_days.csv", index=False, encoding="utf-8-sig")
    else:
        ccf_nz_summary = {"n_days": int(np.sum(nz_mask)), "note": "insufficient_days"}

    # Step 2: segmented CCF
    pre_end, post_start = parse_segment_split(str(args.segment_split))
    seg_defs = {
        "pre_spike": d[d["date"] <= pre_end].copy(),
        "post_spike": d[d["date"] >= post_start].copy(),
    }
    seg_frames: List[pd.DataFrame] = []
    seg_summaries: Dict[str, object] = {}
    for i, (seg_name, seg_df) in enumerate(seg_defs.items(), start=1):
        xr = seg_df["mainstream_risk_count"].values.astype(float)
        yr = seg_df["wemedia_risk_count"].values.astype(float)
        if len(seg_df) < (2 * int(args.max_lag) + 10):
            tmp = pd.DataFrame(
                {
                    "segment": seg_name,
                    "lag": lags.astype(int),
                    "ccf": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "significant": False,
                    "n_days": int(len(seg_df)),
                    "segment_start": seg_df["date"].min(),
                    "segment_end": seg_df["date"].max(),
                }
            )
            seg_frames.append(tmp)
            seg_summaries[seg_name] = {"n_days": int(len(seg_df)), "note": "insufficient_days"}
            continue
        rng_seg = np.random.default_rng(int(args.seed) + 100 * i)
        s_ccf, s_summary = ccf_with_permutation_ci(
            xr,
            yr,
            lags=lags,
            n_perm=int(args.n_perm),
            alpha=float(args.alpha),
            block_size=int(args.block_size),
            rng=rng_seg,
        )
        s_ccf["segment"] = seg_name
        s_ccf["n_days"] = int(len(seg_df))
        s_ccf["segment_start"] = seg_df["date"].min()
        s_ccf["segment_end"] = seg_df["date"].max()
        seg_frames.append(s_ccf[["segment", "lag", "ccf", "ci_lower", "ci_upper", "significant", "n_days", "segment_start", "segment_end"]])
        seg_summaries[seg_name] = s_summary

    ccf_seg = pd.concat(seg_frames, ignore_index=True)
    ccf_seg.to_csv(out_dir / "ccf_by_segment.csv", index=False, encoding="utf-8-sig")

    # 分段图（双 panel）
    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), sharey=True)
    for ax, seg_name, title in [
        (axes[0], "pre_spike", f"Pre-spike (<= {pre_end.date()})"),
        (axes[1], "post_spike", f"Post-spike (>= {post_start.date()})"),
    ]:
        sub = ccf_seg[ccf_seg["segment"] == seg_name].sort_values("lag")
        ax.plot(sub["lag"], sub["ccf"], color="#1f77b4", lw=1.6, marker="o", ms=3)
        ax.fill_between(sub["lag"], sub["ci_lower"], sub["ci_upper"], color="#1f77b4", alpha=0.18)
        ax.axhline(0, color="#666", lw=1.0, alpha=0.8)
        ax.axvline(0, color="#666", lw=1.0, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Lag (days)")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("CCF")
    fig.suptitle("Segmented CCF (mainstream vs wemedia)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ccf_by_segment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Step 3: event coincidence
    windows = [int(x.strip()) for x in str(args.burst_windows).split(",") if x.strip() != ""]
    sens_multipliers = []
    for t in str(args.burst_sensitivity_multipliers).split(","):
        s = t.strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception:
            continue
        if v > 0:
            sens_multipliers.append(v)
    if not sens_multipliers:
        sens_multipliers = [1.0, 1.5, 2.0]
    sens_multipliers = sorted(set([round(v, 6) for v in sens_multipliers]))

    # 主结果使用 multiplier=1.0
    m_burst, m_threshold, m_rule, m_mean_nonzero = classify_burst(x_raw, multiplier=1.0)
    w_burst, w_threshold, w_rule, w_mean_nonzero = classify_burst(y_raw, multiplier=1.0)

    ev_mw = event_coincidence(m_burst, w_burst, windows=windows, direction="mainstream_to_wemedia")
    ev_wm = event_coincidence(w_burst, m_burst, windows=windows, direction="wemedia_to_mainstream")
    ev = pd.concat([ev_mw, ev_wm], ignore_index=True)
    ev.to_csv(out_dir / "event_coincidence.csv", index=False, encoding="utf-8-sig")

    # 敏感性：1.0x / 1.5x / 2.0x mean_nonzero（或 fallback 规则）
    ev_sens_frames: List[pd.DataFrame] = []
    ev_sens_summary_rows: List[Dict[str, object]] = []
    for mult in sens_multipliers:
        mb, mth, mrule, mmnz = classify_burst(x_raw, multiplier=mult)
        wb, wth, wrule, wmnz = classify_burst(y_raw, multiplier=mult)
        tmp_mw = event_coincidence(mb, wb, windows=windows, direction="mainstream_to_wemedia")
        tmp_wm = event_coincidence(wb, mb, windows=windows, direction="wemedia_to_mainstream")
        tmp = pd.concat([tmp_mw, tmp_wm], ignore_index=True)
        tmp["burst_multiplier"] = float(mult)
        tmp["mainstream_threshold"] = float(mth)
        tmp["wemedia_threshold"] = float(wth)
        tmp["mainstream_rule"] = mrule
        tmp["wemedia_rule"] = wrule
        tmp["mainstream_n_burst_days"] = int(np.sum(mb))
        tmp["wemedia_n_burst_days"] = int(np.sum(wb))
        ev_sens_frames.append(tmp)

        ev_sens_summary_rows.append(
            {
                "burst_multiplier": float(mult),
                "mainstream_threshold": float(mth),
                "wemedia_threshold": float(wth),
                "mainstream_rule": mrule,
                "wemedia_rule": wrule,
                "mainstream_mean_nonzero": float(mmnz),
                "wemedia_mean_nonzero": float(wmnz),
                "mainstream_n_burst_days": int(np.sum(mb)),
                "wemedia_n_burst_days": int(np.sum(wb)),
                "mainstream_to_wemedia_window3_p_observed": float(
                    tmp[(tmp["direction"] == "mainstream_to_wemedia") & (tmp["window"] == 3)]["p_observed"].iloc[0]
                )
                if len(tmp[(tmp["direction"] == "mainstream_to_wemedia") & (tmp["window"] == 3)])
                else np.nan,
                "wemedia_to_mainstream_window3_p_observed": float(
                    tmp[(tmp["direction"] == "wemedia_to_mainstream") & (tmp["window"] == 3)]["p_observed"].iloc[0]
                )
                if len(tmp[(tmp["direction"] == "wemedia_to_mainstream") & (tmp["window"] == 3)])
                else np.nan,
            }
        )

    ev_sens = pd.concat(ev_sens_frames, ignore_index=True) if ev_sens_frames else pd.DataFrame()
    ev_sens.to_csv(out_dir / "event_coincidence_sensitivity.csv", index=False, encoding="utf-8-sig")
    (out_dir / "event_coincidence_sensitivity_summary.json").write_text(
        json.dumps({"multipliers": sens_multipliers, "rows": ev_sens_summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ev_summary = {
        "n_days": int(len(d)),
        "burst_definition": {
            "mainstream": {
                "mean_nonzero": m_mean_nonzero,
                "threshold": m_threshold,
                "rule": m_rule,
                "n_burst_days": int(np.sum(m_burst)),
            },
            "wemedia": {
                "mean_nonzero": w_mean_nonzero,
                "threshold": w_threshold,
                "rule": w_rule,
                "n_burst_days": int(np.sum(w_burst)),
            },
        },
        "windows": windows,
        "results": ev.to_dict(orient="records"),
        "sensitivity_multipliers": sens_multipliers,
        "sensitivity_summary_rows": ev_sens_summary_rows,
    }
    (out_dir / "event_coincidence_summary.json").write_text(json.dumps(ev_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 4: combined figure
    fig = plt.figure(figsize=(13.5, 10.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.0], hspace=0.45)

    # Panel A: full CCF
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ccf_df["lag"], ccf_df["ccf"], color="#1f77b4", lw=1.8, marker="o", ms=3, label="Full-sample CCF")
    ax1.fill_between(ccf_df["lag"], ccf_df["ci_lower"], ccf_df["ci_upper"], color="#1f77b4", alpha=0.16, label="95% CI")
    ax1.axhline(0, color="#666", lw=1.0, alpha=0.8)
    ax1.axvline(0, color="#666", lw=1.0, alpha=0.8)
    ax1.set_title("Panel A. Full-sample CCF")
    ax1.set_xlabel("Lag (days) [lag>0: mainstream leads]")
    ax1.set_ylabel("CCF")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, loc="best")

    # Panel B: segmented CCF overlay
    ax2 = fig.add_subplot(gs[1, 0])
    for seg_name, color, lab in [
        ("pre_spike", "#1f77b4", "Pre-spike"),
        ("post_spike", "#ff7f0e", "Post-spike"),
    ]:
        sub = ccf_seg[ccf_seg["segment"] == seg_name].sort_values("lag")
        ax2.plot(sub["lag"], sub["ccf"], color=color, lw=1.8, marker="o", ms=3, label=lab)
    ax2.axhline(0, color="#666", lw=1.0, alpha=0.8)
    ax2.axvline(0, color="#666", lw=1.0, alpha=0.8)
    ax2.set_title("Panel B. Segmented CCF (overlay)")
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("CCF")
    ax2.grid(axis="y", alpha=0.2)
    ax2.legend(frameon=False, loc="best")

    # Panel C: event coincidence bar
    ax3 = fig.add_subplot(gs[2, 0])
    bar = ev.pivot_table(index="window", columns="direction", values="p_observed", aggfunc="first").reset_index()
    bar = bar.sort_values("window")
    xpos = np.arange(len(bar))
    wbar = 0.36
    v1 = bar.get("mainstream_to_wemedia", pd.Series([np.nan] * len(bar))).values
    v2 = bar.get("wemedia_to_mainstream", pd.Series([np.nan] * len(bar))).values
    ax3.bar(xpos - wbar / 2, v1, width=wbar, color="#1f77b4", label="M burst -> W burst")
    ax3.bar(xpos + wbar / 2, v2, width=wbar, color="#ff7f0e", label="W burst -> M burst")
    ax3.set_xticks(xpos, [str(int(x)) for x in bar["window"]])
    ax3.set_ylim(0, 1.0)
    ax3.set_xlabel("Window (days)")
    ax3.set_ylabel("Co-occurrence rate")
    ax3.set_title("Panel C. Event coincidence symmetry")
    ax3.grid(axis="y", alpha=0.2)
    ax3.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_temporal_synchrony_combined.png", dpi=220)
    plt.close(fig)

    # 总结
    overall = {
        "inputs": {
            "input_csv": str(input_csv),
            "out_dir": str(out_dir),
            "max_lag": int(args.max_lag),
            "n_perm": int(args.n_perm),
            "alpha": float(args.alpha),
            "segment_split": str(args.segment_split),
            "burst_windows": windows,
            "block_size": int(args.block_size),
            "seed": int(args.seed),
        },
        "sample": {
            "n_days": int(len(d)),
            "date_min": d["date"].min().strftime("%Y-%m-%d"),
            "date_max": d["date"].max().strftime("%Y-%m-%d"),
            "mainstream_total": float(np.sum(x_raw)),
            "wemedia_total": float(np.sum(y_raw)),
            "nonzero_days_any": int(np.sum(nz_mask)),
            "nonzero_days_any_ratio": float(np.sum(nz_mask) / max(1, len(d))),
        },
        "ccf_summary": ccf_summary,
        "ccf_segment_summary": seg_summaries,
        "ccf_nonzero_day_robustness": ccf_nz_summary,
        "outputs": {
            "ccf_values_csv": str(out_dir / "ccf_values.csv"),
            "ccf_summary_json": str(out_dir / "ccf_summary.json"),
            "fig_ccf_with_ci": str(out_dir / "fig_ccf_with_ci.png"),
            "ccf_by_segment_csv": str(out_dir / "ccf_by_segment.csv"),
            "fig_ccf_by_segment": str(out_dir / "fig_ccf_by_segment.png"),
            "event_coincidence_csv": str(out_dir / "event_coincidence.csv"),
            "event_coincidence_summary_json": str(out_dir / "event_coincidence_summary.json"),
            "event_coincidence_sensitivity_csv": str(out_dir / "event_coincidence_sensitivity.csv"),
            "event_coincidence_sensitivity_summary_json": str(out_dir / "event_coincidence_sensitivity_summary.json"),
            "fig_temporal_synchrony_combined": str(out_dir / "fig_temporal_synchrony_combined.png"),
            "ccf_values_nonzero_days_csv": str(out_dir / "ccf_values_nonzero_days.csv"),
        },
    }
    (out_dir / "temporal_synchrony_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    print("时间同步性分析完成")
    print(f"- out dir: {out_dir}")
    print(f"- n_days: {len(d)}")
    print(f"- peak lag / ccf: {ccf_summary['peak_lag']} / {ccf_summary['peak_ccf']:.4f}")


if __name__ == "__main__":
    main()
