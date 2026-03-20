#!/usr/bin/env python3
"""
V2 路径 B：risk × exposure_group 交互分析（混合效应 logistic）。

主模型（按 one-vs-rest 分别拟合 H/M/L）：
  DV: P(next_state == s), s in {H, M, L}
  IV: exposure_group * env_type
  控制: n_total_interactions, active_days, n_posts（z 标准化）
  FE: 时间固定效应（默认 quarter）
  RE: 用户随机截距 (1 | user_name)

额外稳健性（dual-only）：
  - 分别用 mainstream_risk_ratio 与 wemedia_risk_ratio 建模
  - 不再把 dual 的两侧风险暴露平均为一个 env_type
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
STATES = ["H", "M", "L"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 V2 risk×group 交互混合效应 logistic 分析")
    p.add_argument("--analysis-dir", default="outputs/v2_analysis_fullwindow_v2", help="run_v2_pi_experiments 输出目录")
    p.add_argument("--out-dir", default="", help="输出目录，默认 analysis-dir/risk_interaction_glmm")
    p.add_argument("--groups", nargs="+", default=["mainstream_only", "wemedia_only", "dual"], help="纳入分组")
    p.add_argument("--states", nargs="+", default=STATES, help="要拟合的 next_state 集合")
    p.add_argument("--min-user-transitions", type=int, default=1, help="用户最少 risk/norisk 转移条数")
    p.add_argument("--max-gap-days", type=float, default=0.0, help="最大转移间隔天数（>0 时启用）")
    p.add_argument("--group-override-csv", default="", help="可选：外部分组覆盖 CSV（如 following_exposure_by_user.csv）")
    p.add_argument("--group-override-key", default="user_name", help="覆盖分组 CSV 的用户键列名")
    p.add_argument("--group-override-col", default="follow_group", help="覆盖分组 CSV 的分组列名")
    p.add_argument(
        "--time-fe",
        choices=["quarter", "month", "year_plus_month"],
        default="quarter",
        help="时间固定效应方式（默认 quarter，避免 month 过度参数化）",
    )
    p.add_argument("--skip-dual-side-check", action="store_true", help="跳过 dual-only 分侧检验")
    p.add_argument("--alpha", type=float, default=0.05, help="显著性阈值")
    return p.parse_args()


def _norm_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _standardize(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    m = float(v.mean()) if len(v) else 0.0
    sd = float(v.std(ddof=0)) if len(v) else 0.0
    if sd <= 1e-12:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return (v - m) / sd


def _add_control_columns(d: pd.DataFrame, specs: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, List[str]]:
    x = d.copy()
    control_cols: List[str] = []
    for raw_c, z_c in specs:
        if raw_c not in x.columns:
            continue
        x[raw_c] = pd.to_numeric(x[raw_c], errors="coerce").fillna(0.0)
        x[z_c] = _standardize(x[raw_c])
        control_cols.append(z_c)
    return x, control_cols


def _window_stats_for_targets(
    media_lookup: Dict[str, Dict[str, np.ndarray]],
    targets: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[int, int, int]:
    start64 = np.datetime64(start.to_datetime64())
    end64 = np.datetime64(end.to_datetime64())
    n_posts = 0
    n_risk_posts = 0
    n_accounts_observed = 0
    for t in targets:
        rec = media_lookup.get(t)
        if rec is None:
            continue
        ts = rec["ts"]
        lo = int(np.searchsorted(ts, start64, side="right"))
        hi = int(np.searchsorted(ts, end64, side="right"))
        if hi <= lo:
            continue
        n_accounts_observed += 1
        n_posts += int(hi - lo)
        n_risk_posts += int(rec["risk"][lo:hi].sum())
    return n_posts, n_risk_posts, n_accounts_observed


def _add_time_fe_columns(d: pd.DataFrame, ts_col: str, mode: str) -> Tuple[pd.DataFrame, str]:
    x = d.copy()
    if mode == "quarter":
        x["time_fe"] = pd.to_datetime(x[ts_col], errors="coerce").dt.to_period("Q").astype(str)
        return x, "C(time_fe)"
    if mode == "month":
        x["time_fe"] = pd.to_datetime(x[ts_col], errors="coerce").dt.strftime("%Y-%m")
        return x, "C(time_fe)"
    # year_plus_month
    t = pd.to_datetime(x[ts_col], errors="coerce")
    x["year_fe"] = t.dt.year.astype("Int64").astype(str)
    x["cal_month_fe"] = t.dt.month.astype("Int64").astype(str)
    return x, "C(year_fe) + C(cal_month_fe)"


def _term_type(name: str) -> str:
    if "C(exposure_group" in name and ":C(env_type" in name:
        return "interaction"
    if "C(exposure_group" in name:
        return "main_exposure"
    if "C(env_type" in name:
        return "main_risk_env"
    if name.startswith("C(time_fe)") or name.startswith("C(year_fe)") or name.startswith("C(cal_month_fe)"):
        return "time_fe"
    if name.startswith("z_"):
        return "control"
    if name == "Intercept":
        return "intercept"
    return "other"


def _parse_interaction_term(name: str) -> Tuple[str, str]:
    groups = re.findall(r"C\(exposure_group.*?\)\[T\.([^\]]+)\]", name)
    envs = re.findall(r"C\(env_type.*?\)\[T\.([^\]]+)\]", name)
    g = groups[0] if groups else ""
    e = envs[0] if envs else ""
    return g, e


def prepare_data(
    analysis_dir: Path,
    groups: List[str],
    min_user_transitions: int,
    time_fe_mode: str,
    max_gap_days: float,
    group_override_map: Optional[Dict[str, str]],
) -> Tuple[pd.DataFrame, str, List[str]]:
    trans_path = analysis_dir / "transition_with_environment.csv"
    user_path = analysis_dir / "transition_by_user.csv"

    trans = pd.read_csv(trans_path, dtype=str, low_memory=False)
    user = pd.read_csv(
        user_path,
        usecols=["user_name", "n_total_interactions", "active_days", "n_posts"],
        dtype=str,
        low_memory=False,
    )

    trans["user_name"] = trans["user_name"].map(_norm_text)
    trans["exposure_group"] = trans["exposure_group"].map(_norm_text)
    trans["env_type"] = trans["env_type"].map(_norm_text)
    trans["transition"] = trans["transition"].map(_norm_text)
    trans["prev_time"] = pd.to_datetime(trans["prev_time"], errors="coerce")
    trans["curr_time"] = pd.to_datetime(trans["curr_time"], errors="coerce")
    trans["next_state"] = trans["transition"].str.split("->").str[-1]

    d = trans[trans["env_type"].isin(["risk", "norisk"])].copy()
    d = d[d["exposure_group"].isin(groups)].copy()
    d = d[d["next_state"].isin(STATES)].copy()
    d = d[d["curr_time"].notna()].copy()
    d = d[d["prev_time"].notna()].copy()
    d["gap_days"] = (d["curr_time"] - d["prev_time"]).dt.total_seconds() / 86400.0
    d = d[d["gap_days"] >= 0].copy()
    if float(max_gap_days) > 0:
        d = d[d["gap_days"] <= float(max_gap_days)].copy()
    if group_override_map:
        d["exposure_group_original"] = d["exposure_group"]
        d["exposure_group"] = d["user_name"].map(lambda u: group_override_map.get(str(u), ""))
    d, fe_term = _add_time_fe_columns(d, "curr_time", time_fe_mode)

    user["user_name"] = user["user_name"].map(_norm_text)
    d = d.merge(user, on="user_name", how="left", validate="many_to_one")
    d, control_cols = _add_control_columns(
        d,
        [
            ("n_total_interactions", "z_n_total_interactions"),
            ("active_days", "z_active_days"),
            ("n_posts", "z_n_posts"),
            ("gap_days", "z_gap_days"),
            ("env_n_target_posts_observed", "z_env_n_target_posts_observed"),
        ],
    )

    if min_user_transitions > 1:
        cnt = d.groupby("user_name", as_index=False).size().rename(columns={"size": "n_t"})
        keep = set(cnt.loc[cnt["n_t"] >= int(min_user_transitions), "user_name"].tolist())
        d = d[d["user_name"].isin(keep)].copy()

    d = d[d["exposure_group"].isin(groups)].copy()
    return d.reset_index(drop=True), fe_term, control_cols


def fit_one_state(d: pd.DataFrame, state: str, fe_term: str, control_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    if state not in STATES:
        raise ValueError(f"不支持的状态: {state}")
    x = d.copy()
    x["y"] = (x["next_state"] == state).astype(int)

    formula = (
        "y ~ "
        "C(exposure_group, Treatment(reference='dual')) * "
        "C(env_type, Treatment(reference='norisk'))"
    )
    if control_cols:
        formula += " + " + " + ".join(control_cols)
    formula += " + " + fe_term
    random = {"user_re": "0 + C(user_name)"}
    model = sm.BinomialBayesMixedGLM.from_formula(formula, random, x)
    res = model.fit_vb()

    rows = []
    for name, mean, sd in zip(model.exog_names, res.fe_mean, res.fe_sd):
        z = float(mean / sd) if sd > 0 else np.nan
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "state": state,
                "term": name,
                "term_type": _term_type(name),
                "coef": float(mean),
                "std_err": float(sd),
                "z_value": z,
                "p_value": p,
                "odds_ratio": float(math.exp(mean)),
            }
        )

    coef = pd.DataFrame(rows)
    inter = coef[coef["term_type"] == "interaction"].copy()
    if len(inter):
        ge = inter["term"].map(_parse_interaction_term)
        inter["interaction_group"] = ge.map(lambda t: t[0])
        inter["interaction_env"] = ge.map(lambda t: t[1])
    else:
        inter["interaction_group"] = []
        inter["interaction_env"] = []

    model_info = {
        "state": state,
        "formula": formula,
        "n_obs": int(len(x)),
        "n_users": int(x["user_name"].nunique()),
        "prevalence": float(x["y"].mean()) if len(x) else np.nan,
        "n_fixed_effects": int(len(model.exog_names)),
        "n_random_effects": int(len(model.vc_names)),
        "controls": control_cols,
    }
    return coef, inter, model_info


def build_observed_rates(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, e), x in d.groupby(["exposure_group", "env_type"]):
        n = len(x)
        vc = x["next_state"].value_counts()
        rows.append(
            {
                "exposure_group": g,
                "env_type": e,
                "n_transitions": int(n),
                "p_H": float(vc.get("H", 0) / n),
                "p_M": float(vc.get("M", 0) / n),
                "p_L": float(vc.get("L", 0) / n),
            }
        )
    return pd.DataFrame(rows).sort_values(["exposure_group", "env_type"]).reset_index(drop=True)


def _split_targets(s: object) -> List[str]:
    txt = _norm_text(s)
    if not txt:
        return []
    return [x for x in txt.split("|") if x]


def prepare_dual_side_data(
    analysis_dir: Path,
    min_user_transitions: int,
    time_fe_mode: str,
    max_gap_days: float,
) -> Tuple[pd.DataFrame, str, List[str], Dict[str, int]]:
    trans = pd.read_csv(analysis_dir / "transition_with_environment.csv", dtype=str, low_memory=False)
    expo = pd.read_csv(
        analysis_dir / "user_exposure.csv",
        usecols=["user_name", "targets_mainstream", "targets_wemedia"],
        dtype=str,
        low_memory=False,
    )
    timeline = pd.read_csv(
        analysis_dir / "user_emotion_timeline.csv",
        usecols=["user_name", "publish_time", "risk_class", "node_type_3class"],
        dtype=str,
        low_memory=False,
    )
    user = pd.read_csv(
        analysis_dir / "transition_by_user.csv",
        usecols=["user_name", "n_total_interactions", "active_days", "n_posts"],
        dtype=str,
        low_memory=False,
    )

    trans["user_name"] = trans["user_name"].map(_norm_text)
    trans["exposure_group"] = trans["exposure_group"].map(_norm_text)
    trans["transition"] = trans["transition"].map(_norm_text)
    trans["prev_time"] = pd.to_datetime(trans["prev_time"], errors="coerce")
    trans["curr_time"] = pd.to_datetime(trans["curr_time"], errors="coerce")
    trans["next_state"] = trans["transition"].str.split("->").str[-1]
    trans = trans[(trans["exposure_group"] == "dual") & trans["next_state"].isin(STATES)].copy()
    trans = trans[trans["prev_time"].notna() & trans["curr_time"].notna()].copy()
    trans["gap_days"] = (trans["curr_time"] - trans["prev_time"]).dt.total_seconds() / 86400.0
    trans = trans[trans["gap_days"] >= 0].copy()
    if float(max_gap_days) > 0:
        trans = trans[trans["gap_days"] <= float(max_gap_days)].copy()
    expo["user_name"] = expo["user_name"].map(_norm_text)
    expo = expo.drop_duplicates(subset=["user_name"], keep="first")
    target_m = {u: _split_targets(ms) for u, ms in zip(expo["user_name"], expo["targets_mainstream"])}
    target_w = {u: _split_targets(ws) for u, ws in zip(expo["user_name"], expo["targets_wemedia"])}

    timeline["user_name"] = timeline["user_name"].map(_norm_text)
    timeline["publish_time"] = pd.to_datetime(timeline["publish_time"], errors="coerce")
    timeline = timeline[timeline["publish_time"].notna()].copy()
    media = timeline[timeline["node_type_3class"].isin(["mainstream", "wemedia"])].copy()
    media["is_risk"] = media["risk_class"].eq("risk").astype(np.int8)
    media = media.sort_values(["user_name", "publish_time"])
    media_lookup = {
        user_name: {
            "ts": g["publish_time"].to_numpy(dtype="datetime64[ns]"),
            "risk": g["is_risk"].to_numpy(dtype=np.int16),
        }
        for user_name, g in media.groupby("user_name", sort=False)
    }

    rows = []
    for r in trans.itertuples(index=False):
        u = r.user_name
        prev_t = pd.Timestamp(r.prev_time)
        curr_t = pd.Timestamp(r.curr_time)
        ms_targets = target_m.get(u, [])
        wm_targets = target_w.get(u, [])
        n_m_posts, n_m_risk, n_m_accounts = _window_stats_for_targets(media_lookup, ms_targets, prev_t, curr_t)
        n_w_posts, n_w_risk, n_w_accounts = _window_stats_for_targets(media_lookup, wm_targets, prev_t, curr_t)
        rows.append(
            {
                "user_name": u,
                "prev_time": prev_t,
                "curr_time": curr_t,
                "transition": r.transition,
                "next_state": r.next_state,
                "mainstream_risk_ratio": float(n_m_risk / n_m_posts) if n_m_posts else np.nan,
                "wemedia_risk_ratio": float(n_w_risk / n_w_posts) if n_w_posts else np.nan,
                "n_mainstream_target_posts_observed": int(n_m_posts),
                "n_wemedia_target_posts_observed": int(n_w_posts),
                "n_mainstream_targets_observed": int(n_m_accounts),
                "n_wemedia_targets_observed": int(n_w_accounts),
                "gap_days": float(r.gap_days),
            }
        )
    d = pd.DataFrame(rows)
    if len(d) == 0:
        return d, "C(time_fe)", [], {}

    d, fe_term = _add_time_fe_columns(d, "curr_time", time_fe_mode)

    user["user_name"] = user["user_name"].map(_norm_text)
    d = d.merge(user, on="user_name", how="left", validate="many_to_one")
    d, control_cols = _add_control_columns(
        d,
        [
            ("n_total_interactions", "z_n_total_interactions"),
            ("active_days", "z_active_days"),
            ("n_posts", "z_n_posts"),
            ("gap_days", "z_gap_days"),
        ],
    )

    if min_user_transitions > 1:
        cnt = d.groupby("user_name", as_index=False).size().rename(columns={"size": "n_t"})
        keep = set(cnt.loc[cnt["n_t"] >= int(min_user_transitions), "user_name"].tolist())
        d = d[d["user_name"].isin(keep)].copy()

    avail = {
        "rows_total": int(len(d)),
        "users_total": int(d["user_name"].nunique()) if len(d) else 0,
        "rows_mainstream_ratio_known": int(d["mainstream_risk_ratio"].notna().sum()) if len(d) else 0,
        "rows_wemedia_ratio_known": int(d["wemedia_risk_ratio"].notna().sum()) if len(d) else 0,
        "rows_both_ratio_known": int((d["mainstream_risk_ratio"].notna() & d["wemedia_risk_ratio"].notna()).sum()) if len(d) else 0,
    }
    return d.reset_index(drop=True), fe_term, control_cols, avail


def fit_dual_side_one_state(d_dual: pd.DataFrame, state: str, side_col: str, fe_term: str, control_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    x = d_dual[d_dual[side_col].notna()].copy()
    if len(x) == 0:
        return pd.DataFrame(), {"state": state, "side_col": side_col, "status": "no_data"}

    x["y"] = (x["next_state"] == state).astype(int)
    if x["y"].nunique() < 2:
        return pd.DataFrame(), {
            "state": state,
            "side_col": side_col,
            "status": "degenerate_y",
            "n_obs": int(len(x)),
            "n_users": int(x["user_name"].nunique()),
        }

    formula = "y ~ " + f"{side_col}"
    if control_cols:
        formula += " + " + " + ".join(control_cols)
    formula += " + " + fe_term
    random = {"user_re": "0 + C(user_name)"}
    try:
        model = sm.BinomialBayesMixedGLM.from_formula(formula, random, x)
        res = model.fit_vb()
    except Exception as e:
        return pd.DataFrame(), {
            "state": state,
            "side_col": side_col,
            "status": "fit_failed",
            "error": str(e),
            "n_obs": int(len(x)),
            "n_users": int(x["user_name"].nunique()),
        }

    rows = []
    for name, mean, sd in zip(model.exog_names, res.fe_mean, res.fe_sd):
        z = float(mean / sd) if sd > 0 else np.nan
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        term_type = "side_risk_ratio" if name == side_col else _term_type(name)
        rows.append(
            {
                "state": state,
                "side_col": side_col,
                "term": name,
                "term_type": term_type,
                "coef": float(mean),
                "std_err": float(sd),
                "z_value": z,
                "p_value": p,
                "odds_ratio": float(math.exp(mean)),
            }
        )
    coef = pd.DataFrame(rows)
    side_row = coef[coef["term"] == side_col]
    if len(side_row):
        r = side_row.iloc[0]
        info = {
            "state": state,
            "side_col": side_col,
            "status": "ok",
            "formula": formula,
            "n_obs": int(len(x)),
            "n_users": int(x["user_name"].nunique()),
            "coef": float(r["coef"]),
            "odds_ratio": float(r["odds_ratio"]),
            "z_value": float(r["z_value"]) if pd.notna(r["z_value"]) else np.nan,
            "p_value": float(r["p_value"]) if pd.notna(r["p_value"]) else np.nan,
            "controls": control_cols,
        }
    else:
        info = {
            "state": state,
            "side_col": side_col,
            "status": "term_missing",
            "formula": formula,
            "n_obs": int(len(x)),
            "n_users": int(x["user_name"].nunique()),
        }
    return coef, info


def run_dual_side_models(d_dual: pd.DataFrame, fe_term: str, states: List[str], control_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    coef_parts = []
    infos = []
    for side_col in ["mainstream_risk_ratio", "wemedia_risk_ratio"]:
        for s in states:
            c, info = fit_dual_side_one_state(d_dual, s, side_col, fe_term, control_cols)
            if len(c):
                coef_parts.append(c)
            infos.append(info)
    coef_all = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    info_df = pd.DataFrame(infos)
    return coef_all, info_df


def main() -> None:
    args = parse_args()
    analysis_dir = ROOT / args.analysis_dir
    out_dir = ROOT / args.out_dir if args.out_dir else analysis_dir / "risk_interaction_glmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    group_override_map: Optional[Dict[str, str]] = None
    group_override_info: Dict[str, object] = {"enabled": False}
    if args.group_override_csv:
        gp = (ROOT / args.group_override_csv).resolve() if not Path(args.group_override_csv).is_absolute() else Path(args.group_override_csv)
        gdf = pd.read_csv(gp, dtype=str, low_memory=False)
        if args.group_override_key not in gdf.columns or args.group_override_col not in gdf.columns:
            raise ValueError(
                f"group-override-csv 缺少列：{args.group_override_key} / {args.group_override_col}"
            )
        gdf[args.group_override_key] = gdf[args.group_override_key].map(_norm_text)
        gdf[args.group_override_col] = gdf[args.group_override_col].map(_norm_text)
        gdf = gdf[(gdf[args.group_override_key] != "") & (gdf[args.group_override_col] != "")]
        gdf = gdf.drop_duplicates(subset=[args.group_override_key], keep="first")
        group_override_map = dict(zip(gdf[args.group_override_key], gdf[args.group_override_col]))
        group_override_info = {
            "enabled": True,
            "path": str(gp),
            "key_col": args.group_override_key,
            "group_col": args.group_override_col,
            "n_users_in_override": int(len(gdf)),
        }

    # 主交互模型
    d, fe_term, control_cols = prepare_data(
        analysis_dir=analysis_dir,
        groups=args.groups,
        min_user_transitions=args.min_user_transitions,
        time_fe_mode=args.time_fe,
        max_gap_days=args.max_gap_days,
        group_override_map=group_override_map,
    )
    if len(d) == 0:
        if group_override_map:
            raise RuntimeError(
                "主模型过滤后无数据：覆盖分组后无用户落入当前 groups。"
                "请先完成 following 边抓取并构建分组，或调整 --groups。"
            )
        raise RuntimeError("主模型过滤后无数据：请检查 env_type/group/min-user-transitions 设置。")
    d.to_csv(out_dir / "risk_interaction_glmm_input.csv", index=False, encoding="utf-8-sig")
    obs = build_observed_rates(d)
    obs.to_csv(out_dir / "risk_interaction_glmm_observed_rates.csv", index=False, encoding="utf-8-sig")

    coef_parts = []
    inter_parts = []
    model_infos = []
    for s in args.states:
        c, i, info = fit_one_state(d, s, fe_term=fe_term, control_cols=control_cols)
        coef_parts.append(c)
        inter_parts.append(i)
        model_infos.append(info)
    coef_all = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    inter_all = pd.concat(inter_parts, ignore_index=True) if inter_parts else pd.DataFrame()
    coef_all.to_csv(out_dir / "risk_interaction_glmm_coefficients.csv", index=False, encoding="utf-8-sig")
    inter_all.to_csv(out_dir / "risk_interaction_glmm_interactions.csv", index=False, encoding="utf-8-sig")

    # dual-only 分侧模型
    dual_summary: Dict[str, object] = {"status": "skipped"}
    if group_override_map:
        dual_summary = {"status": "skipped_due_group_override"}
    elif not args.skip_dual_side_check:
        d_dual, fe_term_dual, dual_control_cols, avail = prepare_dual_side_data(
            analysis_dir=analysis_dir,
            min_user_transitions=args.min_user_transitions,
            time_fe_mode=args.time_fe,
            max_gap_days=args.max_gap_days,
        )
        d_dual.to_csv(out_dir / "risk_interaction_glmm_dual_side_input.csv", index=False, encoding="utf-8-sig")
        coef_dual, info_dual = run_dual_side_models(d_dual, fe_term_dual, args.states, dual_control_cols)
        coef_dual.to_csv(out_dir / "risk_interaction_glmm_dual_side_coefficients.csv", index=False, encoding="utf-8-sig")
        info_dual.to_csv(out_dir / "risk_interaction_glmm_dual_side_effects.csv", index=False, encoding="utf-8-sig")
        dual_summary = {
            "status": "ok",
            "availability": avail,
            "controls": dual_control_cols,
            "effects": info_dual.to_dict(orient="records"),
        }

    alpha = float(args.alpha)
    sig_inter = inter_all[inter_all["p_value"] < alpha].copy() if len(inter_all) else pd.DataFrame()
    summary = {
        "inputs": {
            "analysis_dir": str(analysis_dir),
            "groups": args.groups,
            "states": args.states,
            "min_user_transitions": int(args.min_user_transitions),
            "max_gap_days": float(args.max_gap_days),
            "time_fe": args.time_fe,
            "alpha": alpha,
            "group_override": group_override_info,
            "controls": control_cols,
        },
        "data_info": {
            "n_rows": int(len(d)),
            "n_users": int(d["user_name"].nunique()),
            "group_counts": {k: int(v) for k, v in d["exposure_group"].value_counts().to_dict().items()},
            "env_counts": {k: int(v) for k, v in d["env_type"].value_counts().to_dict().items()},
            "group_env_counts": {
                f"{g}|{e}": int(v)
                for (g, e), v in d.groupby(["exposure_group", "env_type"]).size().to_dict().items()
            },
            "next_state_counts": {k: int(v) for k, v in d["next_state"].value_counts().to_dict().items()},
            "gap_days": {
                "mean": float(d["gap_days"].mean()) if len(d) else np.nan,
                "median": float(d["gap_days"].median()) if len(d) else np.nan,
                "p90": float(d["gap_days"].quantile(0.9)) if len(d) else np.nan,
                "max": float(d["gap_days"].max()) if len(d) else np.nan,
            },
        },
        "model_info": model_infos,
        "interaction_terms_total": int(len(inter_all)),
        "interaction_terms_significant": int(len(sig_inter)),
        "significant_interactions": (
            sig_inter[
                [
                    "state",
                    "term",
                    "interaction_group",
                    "interaction_env",
                    "coef",
                    "odds_ratio",
                    "z_value",
                    "p_value",
                ]
            ].to_dict(orient="records")
            if len(sig_inter)
            else []
        ),
        "dual_side_check": dual_summary,
        "outputs": {
            "input_csv": str(out_dir / "risk_interaction_glmm_input.csv"),
            "observed_rates_csv": str(out_dir / "risk_interaction_glmm_observed_rates.csv"),
            "coefficients_csv": str(out_dir / "risk_interaction_glmm_coefficients.csv"),
            "interactions_csv": str(out_dir / "risk_interaction_glmm_interactions.csv"),
            "dual_side_input_csv": str(out_dir / "risk_interaction_glmm_dual_side_input.csv"),
            "dual_side_coefficients_csv": str(out_dir / "risk_interaction_glmm_dual_side_coefficients.csv"),
            "dual_side_effects_csv": str(out_dir / "risk_interaction_glmm_dual_side_effects.csv"),
            "summary_json": str(out_dir / "risk_interaction_glmm_summary.json"),
        },
    }
    (out_dir / "risk_interaction_glmm_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("risk×group 混合效应 logistic 分析完成")
    print(f"- out dir: {out_dir}")
    print(f"- time FE mode: {args.time_fe}")
    print(f"- main rows/users: {len(d)}/{d['user_name'].nunique()}")
    print(f"- significant interactions (p<{alpha}): {len(sig_inter)} / {len(inter_all)}")
    if not args.skip_dual_side_check and not group_override_map:
        print("- dual-side check: done")


if __name__ == "__main__":
    main()
