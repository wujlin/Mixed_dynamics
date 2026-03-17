#!/usr/bin/env python3
"""
V2 内容分析（任务1）：
1.1 情绪框架对比 + 卡方检验
1.2 风险帖子文本特征差异（mainstream vs wemedia）+ Cliff's delta
1.3 媒体风险帖子时间响应 + Granger（含去尖峰与滚动窗口稳健性）
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ["H", "M", "L"]
GROUP_ORDER = ["mainstream", "wemedia", "public"]


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 V2 内容分析")
    p.add_argument("--topic-csv", default="dataset/Topic_data/merged_topic_official.csv", help="Topic 主数据")
    p.add_argument("--annotations", default="outputs/annotations/master/long_covid_annotations_master.jsonl", help="标注 JSONL")
    p.add_argument("--out-dir", default="outputs/v2_content_analysis", help="输出目录")
    p.add_argument("--granger-max-lag", type=int, default=3, help="Granger 最大滞后阶数")
    p.add_argument(
        "--spike-z-threshold",
        type=float,
        default=8.0,
        help="尖峰检测阈值（基于 robust z-score）",
    )
    p.add_argument(
        "--spike-exclude-mode",
        choices=["none", "day", "month"],
        default="month",
        help="Granger 稳健性：剔除尖峰的方式",
    )
    p.add_argument("--rolling-window-days", type=int, default=180, help="滚动窗口长度（天）")
    p.add_argument("--rolling-step-days", type=int, default=30, help="滚动窗口步长（天）")
    return p.parse_args()


def _norm_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _norm_mid(x: object) -> str:
    s = _norm_text(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _map_group_from_verify(verify_typ: str) -> str:
    v = _norm_text(verify_typ)
    if "蓝V" in v:
        return "mainstream"
    if "黄V" in v:
        return "wemedia"
    return "public"


def load_merged(topic_csv: Path, ann_jsonl: Path) -> pd.DataFrame:
    topic = pd.read_csv(topic_csv, dtype=str, low_memory=False)
    topic = topic.rename(columns={c: c.lstrip("\ufeff") for c in topic.columns})
    for c in ["mid", "verify_typ", "publish_time", "content"]:
        if c not in topic.columns:
            topic[c] = ""
    topic["mid"] = topic["mid"].map(_norm_mid)
    topic["verify_typ"] = topic["verify_typ"].map(_norm_text)
    topic["publish_time"] = pd.to_datetime(topic["publish_time"], errors="coerce")
    topic["content"] = topic["content"].map(_norm_text)
    topic = topic[topic["mid"] != ""].drop_duplicates(subset=["mid"], keep="first").copy()

    ann_rows = []
    with ann_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except json.JSONDecodeError:
                continue
            ann_rows.append(
                {
                    "mid": _norm_mid(o.get("mid", "")),
                    "emotion_class": _norm_text(o.get("emotion_class", "")),
                    "risk_class": _norm_text(o.get("risk_class", "")),
                }
            )
    ann = pd.DataFrame(ann_rows)
    ann = ann[ann["mid"] != ""].drop_duplicates(subset=["mid"], keep="first")

    d = topic.merge(ann, on="mid", how="inner", validate="one_to_one")
    d = d[d["emotion_class"].isin(EMOTIONS) & d["risk_class"].isin(["risk", "norisk"])].copy()
    d["group"] = d["verify_typ"].map(_map_group_from_verify)
    d["date"] = d["publish_time"].dt.floor("D")
    return d.reset_index(drop=True)


def analyze_emotion_frame(d: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    out = out_dir / "emotion_frame"
    out.mkdir(parents=True, exist_ok=True)

    g = (
        d.groupby(["group", "risk_class", "emotion_class"], as_index=False)
        .size()
        .rename(columns={"size": "n_posts"})
    )
    totals = g.groupby(["group", "risk_class"], as_index=False)["n_posts"].sum().rename(columns={"n_posts": "n_total"})
    g = g.merge(totals, on=["group", "risk_class"], how="left")
    g["share"] = g["n_posts"] / g["n_total"]
    g.to_csv(out / "emotion_distribution_long.csv", index=False, encoding="utf-8-sig")

    wide = g.pivot_table(index=["group", "risk_class"], columns="emotion_class", values="share", fill_value=0.0).reset_index()
    for c in EMOTIONS:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide[["group", "risk_class"] + EMOTIONS]
    wide.to_csv(out / "emotion_distribution_wide.csv", index=False, encoding="utf-8-sig")

    # 卡方：mainstream risk vs wemedia risk 的情绪分布差异
    sub = g[(g["risk_class"] == "risk") & (g["group"].isin(["mainstream", "wemedia"]))].copy()
    mat = (
        sub.pivot_table(index="group", columns="emotion_class", values="n_posts", aggfunc="sum", fill_value=0)
        .reindex(index=["mainstream", "wemedia"], columns=EMOTIONS, fill_value=0)
        .values
    )
    chi2, p, dof, expected = stats.chi2_contingency(mat)
    chi = {
        "test": "chi2_emotion_distribution_mainstream_vs_wemedia_under_risk",
        "contingency_table": {
            "mainstream": {e: int(mat[0, i]) for i, e in enumerate(EMOTIONS)},
            "wemedia": {e: int(mat[1, i]) for i, e in enumerate(EMOTIONS)},
        },
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "expected": expected.tolist(),
    }
    (out / "chi2_mainstream_vs_wemedia_risk.json").write_text(json.dumps(chi, ensure_ascii=False, indent=2), encoding="utf-8")

    # grouped bar: x = 媒体类型×risk/norisk, y = H/M/L 占比
    plt = _ensure_matplotlib()
    x_labels = []
    rows = []
    for grp in GROUP_ORDER:
        for rk in ["risk", "norisk"]:
            row = wide[(wide["group"] == grp) & (wide["risk_class"] == rk)]
            if len(row):
                rr = row.iloc[0]
                rows.append([float(rr["H"]), float(rr["M"]), float(rr["L"])])
            else:
                rows.append([0.0, 0.0, 0.0])
            x_labels.append(f"{grp}\n{rk}")
    arr = np.array(rows)
    x = np.arange(len(x_labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.bar(x - w, arr[:, 0], width=w, label="H")
    ax.bar(x, arr[:, 1], width=w, label="M")
    ax.bar(x + w, arr[:, 2], width=w, label="L")
    ax.set_xticks(x, x_labels)
    ax.set_ylabel("Share")
    ax.set_ylim(0, 1.0)
    ax.set_title("Emotion Distribution by Group × Risk Condition")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_emotion_distribution_grouped_bar.png", dpi=220)
    plt.close(fig)

    # 2x3 faceted bar：行=risk/norisk，列=H/M/L
    risk_rows = ["risk", "norisk"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2), sharey=True)
    for i, rk in enumerate(risk_rows):
        for j, emo in enumerate(EMOTIONS):
            ax = axes[i, j]
            vals = []
            for grp in GROUP_ORDER:
                row = wide[(wide["group"] == grp) & (wide["risk_class"] == rk)]
                vals.append(float(row.iloc[0][emo]) if len(row) else 0.0)
            ax.bar(GROUP_ORDER, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            if i == 0:
                ax.set_title(emo)
            if j == 0:
                ax.set_ylabel(f"{rk}\nshare")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.2)
            ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Emotion Shares by Group under Risk/NoRisk", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "fig_emotion_distribution_facet_2x3.png", dpi=220)
    plt.close(fig)

    return chi


def _count_pronouns(text: str) -> int:
    # 长词优先，避免“我们”被“我”重复切分
    return len(re.findall(r"我们|自己|我", text))


def _word_count_rough(text: str) -> int:
    # 粗粒度 token：中文单字 + 英文数字串
    toks = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)
    return len(toks)


def _count_data_tokens(text: str) -> int:
    pats = re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?(?:例|人|万|亿)", text)
    return len(pats)


def extract_text_features(text: str) -> Dict[str, float]:
    t = _norm_text(text)
    n_chars = len(t)
    punct = len(re.findall(r"[!！?？]", t))
    words = _word_count_rough(t)
    pron = _count_pronouns(t)
    data_tok = _count_data_tokens(t)
    sent = len(re.findall(r"[。！？!?；;]", t))
    return {
        "text_len_chars": float(n_chars),
        "punct_density": float(punct / max(1, n_chars)),
        "first_person_ratio": float(pron / max(1, words)),
        "data_token_ratio": float(data_tok / max(1, words)),
        "avg_sentence_len": float(n_chars / max(1, sent)),
    }


def _cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float(2.0 * (u_stat / (n1 * n2)) - 1.0)


def _cliffs_magnitude(delta: float) -> str:
    if delta != delta:
        return "nan"
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def analyze_text_features(d: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out = out_dir / "text_features"
    out.mkdir(parents=True, exist_ok=True)

    sub = d[(d["risk_class"] == "risk") & (d["group"].isin(["mainstream", "wemedia"]))].copy()
    feats = sub["content"].map(extract_text_features).apply(pd.Series)
    f = pd.concat([sub[["mid", "group", "risk_class", "content"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    f.to_csv(out / "risk_posts_text_features_by_post.csv", index=False, encoding="utf-8-sig")

    feature_cols = ["punct_density", "first_person_ratio", "data_token_ratio", "avg_sentence_len"]
    rows = []
    for col in feature_cols:
        a = f.loc[f["group"] == "mainstream", col].astype(float).values
        b = f.loc[f["group"] == "wemedia", col].astype(float).values
        u_stat = np.nan
        try:
            u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = np.nan
        delta = _cliffs_delta_from_u(float(u_stat), len(a), len(b)) if u_stat == u_stat else np.nan
        rows.append(
            {
                "feature": col,
                "mainstream_mean": float(np.mean(a)) if len(a) else np.nan,
                "wemedia_mean": float(np.mean(b)) if len(b) else np.nan,
                "mainstream_median": float(np.median(a)) if len(a) else np.nan,
                "wemedia_median": float(np.median(b)) if len(b) else np.nan,
                "mainstream_nonzero_share": float(np.mean(a > 0)) if len(a) else np.nan,
                "wemedia_nonzero_share": float(np.mean(b > 0)) if len(b) else np.nan,
                "mainstream_n": int(len(a)),
                "wemedia_n": int(len(b)),
                "mannwhitney_u": float(u_stat) if u_stat == u_stat else np.nan,
                "mannwhitney_p": float(p) if p == p else np.nan,
                "cliffs_delta_mainstream_vs_wemedia": float(delta) if delta == delta else np.nan,
                "cliffs_delta_magnitude": _cliffs_magnitude(float(delta)) if delta == delta else "nan",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "risk_text_features_mannwhitney.csv", index=False, encoding="utf-8-sig")
    return summary


def _granger_test(y: np.ndarray, x: np.ndarray, lag: int) -> Dict[str, float]:
    n = len(y)
    if n <= lag + 2:
        return {"lag": lag, "n_obs": 0, "F": np.nan, "p_value": np.nan}
    y_lags = np.column_stack([np.roll(y, i) for i in range(1, lag + 1)])[lag:]
    x_lags = np.column_stack([np.roll(x, i) for i in range(1, lag + 1)])[lag:]
    Y = y[lag:]

    Xr = np.column_stack([np.ones(len(Y)), y_lags])
    Xu = np.column_stack([np.ones(len(Y)), y_lags, x_lags])
    br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
    rr = Y - Xr @ br
    ru = Y - Xu @ bu
    rss_r = float(np.sum(rr**2))
    rss_u = float(np.sum(ru**2))
    m = lag
    df2 = len(Y) - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0 or rss_r < rss_u:
        return {"lag": lag, "n_obs": int(len(Y)), "F": np.nan, "p_value": np.nan}
    F = ((rss_r - rss_u) / m) / (rss_u / df2)
    p = float(1.0 - stats.f.cdf(F, m, df2))
    return {"lag": lag, "n_obs": int(len(Y)), "F": float(F), "p_value": p}


def _run_granger_suite(daily: pd.DataFrame, max_lag: int) -> Dict[str, object]:
    y_m = daily["mainstream_risk_count"].values.astype(float)
    y_w = daily["wemedia_risk_count"].values.astype(float)
    res_mw = [_granger_test(y_w, y_m, lag) for lag in range(1, int(max_lag) + 1)]  # mainstream -> wemedia
    res_wm = [_granger_test(y_m, y_w, lag) for lag in range(1, int(max_lag) + 1)]  # wemedia -> mainstream
    return {
        "n_days": int(len(daily)),
        "mainstream_to_wemedia": res_mw,
        "wemedia_to_mainstream": res_wm,
    }


def _detect_spike_days(daily: pd.DataFrame, z_threshold: float) -> pd.DataFrame:
    out = daily[["date", "mainstream_risk_count", "wemedia_risk_count"]].copy()
    x = out["wemedia_risk_count"].astype(float).values
    if len(x) == 0:
        out["robust_z_wemedia"] = []
        return out.iloc[0:0]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad > 0:
        robust_z = 0.6745 * (x - med) / mad
    else:
        sd = float(np.std(x))
        robust_z = (x - med) / max(sd, 1e-9)
    out["robust_z_wemedia"] = robust_z
    spikes = out[np.abs(out["robust_z_wemedia"]) >= float(z_threshold)].copy()
    if len(spikes) == 0 and len(out):
        # 兜底：若阈值下无尖峰，保留极大值日作为单点敏感性检查
        idx = int(out["wemedia_risk_count"].astype(float).idxmax())
        spikes = out.loc[[idx]].copy()
    spikes = spikes.sort_values("date").reset_index(drop=True)
    spikes["year_month"] = spikes["date"].dt.strftime("%Y-%m")
    return spikes


def _rolling_granger(
    daily: pd.DataFrame,
    max_lag: int,
    window_days: int,
    step_days: int,
) -> pd.DataFrame:
    if len(daily) < max(window_days, max_lag + 5):
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "n_days",
                "lag",
                "direction",
                "F",
                "p_value",
            ]
        )
    rows: List[Dict[str, object]] = []
    end_limit = len(daily) - window_days
    for start in range(0, end_limit + 1, max(1, step_days)):
        chunk = daily.iloc[start : start + window_days].copy()
        g = _run_granger_suite(chunk, max_lag=max_lag)
        for r in g["mainstream_to_wemedia"]:
            rows.append(
                {
                    "window_start": chunk["date"].iloc[0],
                    "window_end": chunk["date"].iloc[-1],
                    "n_days": int(g["n_days"]),
                    "lag": int(r["lag"]),
                    "direction": "mainstream_to_wemedia",
                    "F": r["F"],
                    "p_value": r["p_value"],
                }
            )
        for r in g["wemedia_to_mainstream"]:
            rows.append(
                {
                    "window_start": chunk["date"].iloc[0],
                    "window_end": chunk["date"].iloc[-1],
                    "n_days": int(g["n_days"]),
                    "lag": int(r["lag"]),
                    "direction": "wemedia_to_mainstream",
                    "F": r["F"],
                    "p_value": r["p_value"],
                }
            )
    return pd.DataFrame(rows)


def _split_contiguous_segments(daily: pd.DataFrame) -> List[pd.DataFrame]:
    if daily.empty:
        return []
    x = daily.sort_values("date").copy()
    seg_id = (x["date"].diff().dt.days.fillna(1).ne(1)).cumsum()
    out: List[pd.DataFrame] = []
    for _, chunk in x.groupby(seg_id):
        out.append(chunk.reset_index(drop=True))
    return out


def analyze_time_response(
    d: pd.DataFrame,
    out_dir: Path,
    max_lag: int,
    spike_z_threshold: float,
    spike_exclude_mode: str,
    rolling_window_days: int,
    rolling_step_days: int,
) -> Dict[str, object]:
    out = out_dir / "time_response"
    out.mkdir(parents=True, exist_ok=True)

    x = d[d["risk_class"] == "risk"].copy()
    daily = (
        x.groupby(["date", "group"], as_index=False)
        .size()
        .rename(columns={"size": "risk_count"})
        .pivot_table(index="date", columns="group", values="risk_count", fill_value=0.0)
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for c in ["mainstream", "wemedia"]:
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values("date")
    if len(daily):
        idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
        daily = daily.set_index("date").reindex(idx).fillna(0.0).reset_index().rename(columns={"index": "date"})
    daily = daily.rename(columns={"mainstream": "mainstream_risk_count", "wemedia": "wemedia_risk_count"})
    daily["mainstream_risk_count_ma7"] = daily["mainstream_risk_count"].rolling(7, min_periods=1).mean()
    daily["wemedia_risk_count_ma7"] = daily["wemedia_risk_count"].rolling(7, min_periods=1).mean()
    daily.to_csv(out / "daily_risk_counts.csv", index=False, encoding="utf-8-sig")

    gr_full = _run_granger_suite(daily, max_lag=max_lag)

    spikes = _detect_spike_days(daily, z_threshold=spike_z_threshold)
    spikes.to_csv(out / "detected_spike_days.csv", index=False, encoding="utf-8-sig")

    daily_robust = daily.copy()
    excluded_months: List[str] = []
    excluded_days: List[str] = []
    if spike_exclude_mode != "none" and len(spikes):
        if spike_exclude_mode == "day":
            excluded_days = sorted(spikes["date"].dt.strftime("%Y-%m-%d").unique().tolist())
            daily_robust = daily_robust[~daily_robust["date"].dt.strftime("%Y-%m-%d").isin(excluded_days)].copy()
        elif spike_exclude_mode == "month":
            excluded_months = sorted(spikes["year_month"].dropna().unique().tolist())
            daily_robust = daily_robust[~daily_robust["date"].dt.strftime("%Y-%m").isin(excluded_months)].copy()
    daily_robust = daily_robust.sort_values("date").reset_index(drop=True)
    daily_robust.to_csv(out / "daily_risk_counts_spike_excluded.csv", index=False, encoding="utf-8-sig")

    segments = [s for s in _split_contiguous_segments(daily_robust) if len(s) >= (max_lag + 5)]
    seg_rows: List[Dict[str, object]] = []
    for i, seg in enumerate(segments, start=1):
        seg_rows.append(
            {
                "segment_id": i,
                "start_date": seg["date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": seg["date"].iloc[-1].strftime("%Y-%m-%d"),
                "n_days": int(len(seg)),
            }
        )
    pd.DataFrame(seg_rows).to_csv(out / "granger_spike_excluded_segments.csv", index=False, encoding="utf-8-sig")

    if segments:
        seg_results = []
        for i, seg in enumerate(segments, start=1):
            seg_results.append(
                {
                    "segment_id": i,
                    "start_date": seg["date"].iloc[0].strftime("%Y-%m-%d"),
                    "end_date": seg["date"].iloc[-1].strftime("%Y-%m-%d"),
                    "n_days": int(len(seg)),
                    "granger": _run_granger_suite(seg, max_lag=max_lag),
                }
            )
        largest = max(seg_results, key=lambda x: x["n_days"])
        gr_robust = {
            "n_segments": int(len(seg_results)),
            "largest_segment": largest,
            "segments": seg_results,
        }
    else:
        gr_robust = {
            "n_segments": 0,
            "largest_segment": None,
            "segments": [],
        }

    rolling_full = _rolling_granger(
        daily,
        max_lag=max_lag,
        window_days=int(rolling_window_days),
        step_days=int(rolling_step_days),
    )
    rolling_full.to_csv(out / "granger_rolling_windows.csv", index=False, encoding="utf-8-sig")

    rolling_robust_parts = []
    for i, seg in enumerate(segments, start=1):
        rr = _rolling_granger(
            seg,
            max_lag=max_lag,
            window_days=int(rolling_window_days),
            step_days=int(rolling_step_days),
        )
        if rr.empty:
            continue
        rr["segment_id"] = i
        rolling_robust_parts.append(rr)
    if rolling_robust_parts:
        rolling_robust = pd.concat(rolling_robust_parts, ignore_index=True)
    else:
        rolling_robust = pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "n_days",
                "lag",
                "direction",
                "F",
                "p_value",
                "segment_id",
            ]
        )
    rolling_robust.to_csv(out / "granger_rolling_windows_spike_excluded.csv", index=False, encoding="utf-8-sig")

    def _rolling_sig_summary(df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {"n_windows": 0, "share_sig_p_lt_0_05": {}}
        n_windows = int(df[["window_start", "window_end"]].drop_duplicates().shape[0])
        ss = (
            df.assign(sig=df["p_value"] < 0.05)
            .groupby(["direction", "lag"], as_index=False)["sig"]
            .mean()
            .rename(columns={"sig": "share_sig"})
        )
        return {
            "n_windows": n_windows,
            "share_sig_p_lt_0_05": [
                {
                    "direction": str(r["direction"]),
                    "lag": int(r["lag"]),
                    "share_sig": float(r["share_sig"]),
                }
                for _, r in ss.iterrows()
            ],
        }

    gr = {
        "params": {
            "max_lag": int(max_lag),
            "spike_z_threshold": float(spike_z_threshold),
            "spike_exclude_mode": spike_exclude_mode,
            "rolling_window_days": int(rolling_window_days),
            "rolling_step_days": int(rolling_step_days),
        },
        "full_series": gr_full,
        "spike_detection": {
            "n_spike_days": int(len(spikes)),
            "excluded_days": excluded_days,
            "excluded_months": excluded_months,
            "max_wemedia_day": {
                "date": daily.loc[daily["wemedia_risk_count"].idxmax(), "date"].strftime("%Y-%m-%d") if len(daily) else "",
                "wemedia_risk_count": float(daily["wemedia_risk_count"].max()) if len(daily) else 0.0,
                "mainstream_risk_count": float(daily.loc[daily["wemedia_risk_count"].idxmax(), "mainstream_risk_count"]) if len(daily) else 0.0,
            },
        },
        "spike_excluded_series": gr_robust,
        "rolling_full": _rolling_sig_summary(rolling_full),
        "rolling_spike_excluded": _rolling_sig_summary(rolling_robust),
    }
    (out / "granger_media_leadlag.json").write_text(json.dumps(gr, ensure_ascii=False, indent=2), encoding="utf-8")

    plt = _ensure_matplotlib()
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(daily["date"], daily["mainstream_risk_count"], color="#1f77b4", lw=1.6, label="mainstream_risk_count")
    ax2.plot(daily["date"], daily["wemedia_risk_count"], color="#ff7f0e", lw=1.6, label="wemedia_risk_count")
    ax1.set_ylabel("Mainstream Risk Count", color="#1f77b4")
    ax2.set_ylabel("Wemedia Risk Count", color="#ff7f0e")
    ax1.set_title("Daily Risk Post Counts: Mainstream vs Wemedia")
    ax1.grid(axis="y", alpha=0.2)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_daily_risk_counts_dual_axis.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily["date"], daily["mainstream_risk_count_ma7"], color="#1f77b4", lw=1.8, label="mainstream (7d MA)")
    ax.plot(daily["date"], daily["wemedia_risk_count_ma7"], color="#ff7f0e", lw=1.8, label="wemedia (7d MA)")
    if len(spikes):
        ax.scatter(
            spikes["date"],
            spikes["wemedia_risk_count"],
            color="#d62728",
            s=22,
            zorder=3,
            label="detected spikes",
        )
    ax.set_ylabel("Risk Post Count (7d MA)")
    ax.set_title("Daily Risk Post Counts (7-day Moving Average)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_daily_risk_counts_ma7.png", dpi=220)
    plt.close(fig)
    return gr


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_merged(ROOT / args.topic_csv, ROOT / args.annotations)

    chi = analyze_emotion_frame(d, out_dir)
    feat = analyze_text_features(d, out_dir)
    gr = analyze_time_response(
        d,
        out_dir,
        max_lag=args.granger_max_lag,
        spike_z_threshold=args.spike_z_threshold,
        spike_exclude_mode=args.spike_exclude_mode,
        rolling_window_days=args.rolling_window_days,
        rolling_step_days=args.rolling_step_days,
    )

    summary = {
        "inputs": {
            "topic_csv": str(ROOT / args.topic_csv),
            "annotations": str(ROOT / args.annotations),
            "out_dir": str(out_dir),
        },
        "sample": {
            "n_posts_merged": int(len(d)),
            "group_counts": {k: int(v) for k, v in d["group"].value_counts().to_dict().items()},
            "risk_counts": {k: int(v) for k, v in d["risk_class"].value_counts().to_dict().items()},
            "emotion_counts": {k: int(v) for k, v in d["emotion_class"].value_counts().to_dict().items()},
        },
        "chi2_mainstream_vs_wemedia_risk": chi,
        "text_feature_tests": feat.to_dict(orient="records"),
        "granger": gr,
        "outputs": {
            "emotion_distribution_long": str(out_dir / "emotion_frame/emotion_distribution_long.csv"),
            "emotion_distribution_wide": str(out_dir / "emotion_frame/emotion_distribution_wide.csv"),
            "chi2_json": str(out_dir / "emotion_frame/chi2_mainstream_vs_wemedia_risk.json"),
            "emotion_figure": str(out_dir / "emotion_frame/fig_emotion_distribution_grouped_bar.png"),
            "emotion_figure_facet": str(out_dir / "emotion_frame/fig_emotion_distribution_facet_2x3.png"),
            "text_feature_detail": str(out_dir / "text_features/risk_posts_text_features_by_post.csv"),
            "text_feature_summary": str(out_dir / "text_features/risk_text_features_mannwhitney.csv"),
            "daily_risk_counts": str(out_dir / "time_response/daily_risk_counts.csv"),
            "spike_days": str(out_dir / "time_response/detected_spike_days.csv"),
            "daily_risk_counts_spike_excluded": str(out_dir / "time_response/daily_risk_counts_spike_excluded.csv"),
            "granger_spike_excluded_segments": str(out_dir / "time_response/granger_spike_excluded_segments.csv"),
            "granger_rolling_windows": str(out_dir / "time_response/granger_rolling_windows.csv"),
            "granger_rolling_windows_spike_excluded": str(out_dir / "time_response/granger_rolling_windows_spike_excluded.csv"),
            "granger_json": str(out_dir / "time_response/granger_media_leadlag.json"),
            "time_figure": str(out_dir / "time_response/fig_daily_risk_counts_dual_axis.png"),
            "time_figure_ma7": str(out_dir / "time_response/fig_daily_risk_counts_ma7.png"),
        },
    }
    (out_dir / "content_analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("V2 内容分析完成")
    print(f"- out dir: {out_dir}")
    print(f"- merged posts: {len(d)}")
    print(f"- chi2 p (mainstream vs wemedia under risk): {chi['p_value']:.6g}")


if __name__ == "__main__":
    main()
