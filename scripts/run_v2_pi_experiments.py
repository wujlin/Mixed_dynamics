#!/usr/bin/env python3
"""
V2 实验方案（PI 指令）执行脚本。

目标：在不重新爬取的前提下，完成：
1) 数据准备（用户暴露分组 + 情绪时间线）
2) 聚合层面时间序列 + Granger
3) 个体转移 + PSM
4) 机制与稳健性

输出目录：outputs/v2_analysis/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _load_user_mapper_cls():
    module_path = ROOT / "src/empirical/user_mapper.py"
    spec = importlib.util.spec_from_file_location("user_mapper", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 user_mapper: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_mapper"] = module
    spec.loader.exec_module(module)
    return module.UserTypeMapper


UserTypeMapper = _load_user_mapper_cls()


EMOTIONS = ["H", "M", "L"]
TRANSITIONS = [f"{a}->{b}" for a in EMOTIONS for b in EMOTIONS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="运行 V2 PI 实验方案")
    p.add_argument("--network-dir", default="outputs/network/repost_user_network", help="网络结果目录")
    p.add_argument("--out-dir", default="outputs/v2_analysis", help="输出目录")
    p.add_argument("--study-start", default="2022-12-01", help="研究起始日（含），传 auto 时自动取最早时间")
    p.add_argument("--study-end", default="2022-12-31", help="研究结束日（含），传 auto 时自动取最晚时间")
    p.add_argument("--full-window", action="store_true", help="使用全部可用标注时段（忽略 study-start/end）")
    p.add_argument(
        "--annotations",
        nargs="+",
        default=["outputs/annotations/master/long_covid_annotations_master.jsonl"],
        help="一个或多个标注 JSONL 路径（按顺序合并，后者覆盖同 mid）",
    )
    p.add_argument("--min-posts", type=int, default=2, help="纳入分析最小帖子数")
    p.add_argument("--psm-ratio", type=int, default=3, help="PSM 匹配比例（1:k）")
    p.add_argument("--psm-caliper-mult", type=float, default=0.2, help="卡钳倍数（乘以 pscore 标准差）")
    p.add_argument("--bootstrap-iters", type=int, default=1000, help="Bootstrap 次数")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
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


def _map_verify_typ_from_numeric(v: object) -> str:
    try:
        x = int(float(str(v).strip()))
    except Exception:
        x = -1
    if x == 0:
        return "黄V认证"
    if x in (1, 2, 3, 4, 5, 6, 7):
        return "蓝V认证"
    return "无认证"


def _to_3class(raw: str) -> str:
    if raw in ("mainstream", "government"):
        return "mainstream"
    if raw == "wemedia":
        return "wemedia"
    return "ordinary"


def _mode_by_count(df: pd.DataFrame, key_col: str, value_col: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=[key_col, value_col])
    c = (
        df.groupby([key_col, value_col], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values([key_col, "n", value_col], ascending=[True, False, True])
    )
    return c.drop_duplicates(subset=[key_col], keep="first")[[key_col, value_col]].reset_index(drop=True)


def build_verify_lookup(topic_dir: Path, user_info_csv: Path) -> Dict[str, str]:
    # 1) topic verify_typ
    parts = []
    for p in sorted(topic_dir.glob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        if not {"user_name", "verify_typ"}.issubset(set(h.columns)):
            continue
        try:
            d = pd.read_csv(p, usecols=["user_name", "verify_typ"], dtype=str, low_memory=False)
        except Exception:
            continue
        d["user_name"] = d["user_name"].map(_norm_text)
        d["verify_typ"] = d["verify_typ"].map(_norm_text)
        d = d[d["user_name"] != ""]
        if len(d):
            parts.append(d)
    topic_mode = _mode_by_count(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["user_name", "verify_typ"]), "user_name", "verify_typ")

    # 2) USER_INFO_1 认证类型补充
    if user_info_csv.exists():
        try:
            u = pd.read_csv(user_info_csv, usecols=["user_name", "认证类型"], dtype=str, low_memory=False)
        except Exception:
            u = pd.DataFrame(columns=["user_name", "认证类型"])
    else:
        u = pd.DataFrame(columns=["user_name", "认证类型"])
    u["user_name"] = u["user_name"].map(_norm_text)
    u["verify_typ"] = u["认证类型"].map(_map_verify_typ_from_numeric) if "认证类型" in u.columns else "无认证"
    u = u[u["user_name"] != ""]
    u_mode = _mode_by_count(u[["user_name", "verify_typ"]], "user_name", "verify_typ")

    lut = dict(zip(topic_mode["user_name"], topic_mode["verify_typ"]))
    for n, v in zip(u_mode["user_name"], u_mode["verify_typ"]):
        if n not in lut:
            lut[n] = v
    return lut


def build_mid_user_map() -> Dict[str, str]:
    parts = []
    # topic
    for p in sorted((ROOT / "dataset/Topic_data").glob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        norm_to_raw = {}
        for c in h.columns:
            norm_to_raw[c.lstrip("\ufeff")] = c
        colset = set(norm_to_raw.keys())
        mid_norm = "mid" if "mid" in colset else ("id" if "id" in colset else ("微博id" if "微博id" in colset else ""))
        user_norm = "user_name" if "user_name" in colset else ("用户名称" if "用户名称" in colset else "")
        if not mid_norm or not user_norm:
            continue
        mid_col = norm_to_raw[mid_norm]
        user_col = norm_to_raw[user_norm]
        try:
            d = _read_topic_csv(p, [mid_col, user_col])
        except Exception:
            continue
        d = d.rename(columns={c: c.lstrip("\ufeff") for c in d.columns})
        d = d.rename(columns={mid_col: "mid", user_col: "user_name"})
        d["mid"] = d["mid"].map(_norm_mid)
        d["user_name"] = d["user_name"].map(_norm_text)
        d = d[(d["mid"] != "") & (d["user_name"] != "")]
        if len(d):
            parts.append(d)
    # repost
    rp = ROOT / "dataset/Repost/REPOST.csv"
    if rp.exists():
        d = pd.read_csv(rp, usecols=["mid", "user_name"], dtype=str, low_memory=False)
        d["mid"] = d["mid"].map(_norm_mid)
        d["user_name"] = d["user_name"].map(_norm_text)
        d = d[(d["mid"] != "") & (d["user_name"] != "")]
        if len(d):
            parts.append(d)
    # batch3 source
    b3 = ROOT / "outputs/annotations/intermediate/to_annotate_batch3_clean.csv"
    if b3.exists():
        d = pd.read_csv(b3, usecols=["mid", "user_name"], dtype=str, low_memory=False)
        d["mid"] = d["mid"].map(_norm_mid)
        d["user_name"] = d["user_name"].map(_norm_text)
        d = d[(d["mid"] != "") & (d["user_name"] != "")]
        if len(d):
            parts.append(d)

    full = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["mid", "user_name"])
    voted = (
        full.groupby(["mid", "user_name"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["mid", "n", "user_name"], ascending=[True, False, True])
        .drop_duplicates(subset=["mid"], keep="first")
    )
    return dict(zip(voted["mid"], voted["user_name"]))


def _read_topic_csv(path: Path, usecols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, lineterminator="\n", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip")


def load_topic_posts(topic_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(topic_dir.glob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        norm_to_raw = {}
        for c in h.columns:
            norm_to_raw[c.lstrip("\ufeff")] = c
        colset = set(norm_to_raw.keys())
        mid_norm = "mid" if "mid" in colset else ("id" if "id" in colset else ("微博id" if "微博id" in colset else ""))
        user_norm = "user_name" if "user_name" in colset else ("用户名称" if "用户名称" in colset else "")
        mid_col = norm_to_raw.get(mid_norm, "")
        user_col = norm_to_raw.get(user_norm, "")
        if not mid_col or not user_col:
            continue
        verify_norm = "verify_typ" if "verify_typ" in colset else ("用户认证" if "用户认证" in colset else "")
        time_norm = "publish_time" if "publish_time" in colset else ("发布时间" if "发布时间" in colset else "")
        content_norm = "content" if "content" in colset else ("text" if "text" in colset else ("微博正文" if "微博正文" in colset else ""))
        verify_col = norm_to_raw.get(verify_norm, "") if verify_norm else ""
        time_col = norm_to_raw.get(time_norm, "") if time_norm else ""
        content_col = norm_to_raw.get(content_norm, "") if content_norm else ""

        usecols = [mid_col, user_col]
        if verify_col:
            usecols.append(verify_col)
        if time_col:
            usecols.append(time_col)
        if content_col:
            usecols.append(content_col)

        try:
            d = _read_topic_csv(p, usecols)
        except Exception:
            continue
        d = d.rename(columns={c: c.lstrip("\ufeff") for c in d.columns})
        rename_map = {mid_col: "mid", user_col: "user_name"}
        if verify_col:
            rename_map[verify_col] = "verify_typ"
        if time_col:
            rename_map[time_col] = "publish_time"
        if content_col:
            rename_map[content_col] = "content"
        d = d.rename(columns=rename_map)
        for c in ["mid", "user_name", "verify_typ", "publish_time", "content"]:
            if c not in d.columns:
                d[c] = ""
        d["mid"] = d["mid"].map(_norm_mid)
        d["user_name"] = d["user_name"].map(_norm_text)
        d["verify_typ"] = d["verify_typ"].map(_norm_text)
        d["content"] = d["content"].map(_norm_text)
        d["publish_time"] = pd.to_datetime(d["publish_time"], errors="coerce")
        d = d[(d["mid"] != "") & (d["user_name"] != "")].copy()
        if len(d):
            parts.append(d[["mid", "user_name", "verify_typ", "publish_time", "content"]])

    if not parts:
        return pd.DataFrame(columns=["mid", "user_name", "verify_typ", "publish_time", "content"])

    full = pd.concat(parts, ignore_index=True)
    full["quality"] = (
        full["user_name"].ne("").astype(int)
        + full["verify_typ"].ne("").astype(int)
        + full["content"].ne("").astype(int)
        + full["publish_time"].notna().astype(int)
    )
    full = full.sort_values(["mid", "quality", "publish_time"], ascending=[True, False, True], na_position="last")
    full = full.drop_duplicates(subset=["mid"], keep="first").drop(columns=["quality"])
    return full.reset_index(drop=True)


def load_annotation_table(annotation_paths: List[Path]) -> pd.DataFrame:
    merged: Dict[str, Dict[str, object]] = {}
    for p in annotation_paths:
        if not p.exists():
            raise FileNotFoundError(f"标注文件不存在: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    o = json.loads(s)
                except json.JSONDecodeError:
                    continue
                mid = _norm_mid(o.get("mid", ""))
                if not mid:
                    continue
                merged[mid] = {
                    "mid": mid,
                    "emotion_class": _norm_text(o.get("emotion_class", "")),
                    "risk_class": _norm_text(o.get("risk_class", "")),
                    "emotion_confidence": o.get("emotion_confidence"),
                }
    ann = pd.DataFrame(list(merged.values()))
    if len(ann) == 0:
        return pd.DataFrame(columns=["mid", "emotion_class", "risk_class", "emotion_confidence"])
    return ann


def load_master_timeline(mapper: UserTypeMapper, topic_dir: Path, annotation_paths: List[Path]) -> pd.DataFrame:
    df = load_topic_posts(topic_dir)
    ann = load_annotation_table(annotation_paths)
    out = df.merge(ann, on="mid", how="inner", validate="one_to_one")
    out["user_type_raw"] = out.apply(lambda r: mapper.map_verify_type(r["verify_typ"], r["user_name"]).user_type, axis=1)
    out["node_type_3class"] = out["user_type_raw"].map(_to_3class)
    out["date"] = out["publish_time"].dt.floor("D")
    return out[
        [
            "user_name",
            "mid",
            "publish_time",
            "date",
            "emotion_class",
            "risk_class",
            "emotion_confidence",
            "verify_typ",
            "user_type_raw",
            "node_type_3class",
        ]
    ].reset_index(drop=True)


def build_user_exposure(network_dir: Path, verify_lut: Dict[str, str], mapper: UserTypeMapper) -> Tuple[pd.DataFrame, Dict[str, set], Dict[str, set]]:
    edge_path = network_dir / "edge_weighted.csv"
    node_path = network_dir / "node_attributes.csv"
    detail_path = network_dir / "edge_detail.csv"
    comment_path = ROOT / "dataset/User_data/USER_INFO_2.csv"

    edge = pd.read_csv(edge_path, dtype=str, low_memory=False)
    edge["source_user"] = edge["source_user"].map(_norm_text)
    edge["target_user"] = edge["target_user"].map(_norm_text)
    edge["weight"] = pd.to_numeric(edge["weight"], errors="coerce").fillna(0).astype(int)
    edge = edge[(edge["source_user"] != "") & (edge["target_user"] != "")]

    # 目标用户类型：优先 node_attributes
    if node_path.exists():
        node = pd.read_csv(node_path, dtype=str, low_memory=False)
        node["user_name"] = node["user_name"].map(_norm_text)
        node["node_type_3class"] = node["node_type_3class"].map(_norm_text)
        target_type = dict(zip(node["user_name"], node["node_type_3class"]))
    else:
        target_type = {}

    def infer_type(user_name: str) -> str:
        t = target_type.get(user_name, "")
        if t:
            return t
        vt = verify_lut.get(user_name, "")
        raw = mapper.map_verify_type(vt, user_name).user_type
        return _to_3class(raw)

    edge["target_type_3"] = edge["target_user"].map(infer_type)
    edge["is_media_target"] = edge["target_type_3"].isin(["mainstream", "wemedia"])

    # 来自边表的总交互计数
    inter_counts = (
        edge.groupby("source_user", as_index=False)["weight"]
        .sum()
        .rename(columns={"source_user": "user_name", "weight": "n_interactions_network"})
    )

    # 媒体目标去重
    dedup = edge[edge["is_media_target"]][["source_user", "target_user", "target_type_3"]].drop_duplicates()
    n_m = dedup[dedup["target_type_3"] == "mainstream"].groupby("source_user")["target_user"].nunique()
    n_w = dedup[dedup["target_type_3"] == "wemedia"].groupby("source_user")["target_user"].nunique()

    target_m: Dict[str, set] = {}
    target_w: Dict[str, set] = {}
    for src, g in dedup.groupby("source_user"):
        target_m[src] = set(g.loc[g["target_type_3"] == "mainstream", "target_user"].tolist())
        target_w[src] = set(g.loc[g["target_type_3"] == "wemedia", "target_user"].tolist())

    # 评论网络：comment_user -> weibo_mid 作者
    c = pd.read_csv(comment_path, usecols=["weibo_mid", "comment_time", "comment_user_name"], dtype=str, low_memory=False)
    c["source_user"] = c["comment_user_name"].map(_norm_text)
    c["weibo_mid"] = c["weibo_mid"].map(_norm_mid)
    c["comment_time"] = pd.to_datetime(c["comment_time"], errors="coerce")
    c = c[(c["source_user"] != "") & (c["weibo_mid"] != "")]
    mid_user = build_mid_user_map()
    c["target_user"] = c["weibo_mid"].map(lambda x: mid_user.get(x, ""))
    c = c[c["target_user"] != ""].copy()
    c["target_type_3"] = c["target_user"].map(infer_type)
    c["is_media_target"] = c["target_type_3"].isin(["mainstream", "wemedia"])

    # 评论计数
    c_counts = (
        c.groupby("source_user", as_index=False)
        .size()
        .rename(columns={"source_user": "user_name", "size": "n_interactions_comment"})
    )

    c_dedup = c[c["is_media_target"]][["source_user", "target_user", "target_type_3"]].drop_duplicates()
    c_m = c_dedup[c_dedup["target_type_3"] == "mainstream"].groupby("source_user")["target_user"].nunique()
    c_w = c_dedup[c_dedup["target_type_3"] == "wemedia"].groupby("source_user")["target_user"].nunique()

    for src, g in c_dedup.groupby("source_user"):
        target_m.setdefault(src, set()).update(g.loc[g["target_type_3"] == "mainstream", "target_user"].tolist())
        target_w.setdefault(src, set()).update(g.loc[g["target_type_3"] == "wemedia", "target_user"].tolist())

    users = pd.DataFrame({"user_name": pd.unique(pd.concat([edge["source_user"], c["source_user"]], ignore_index=True))})
    users = users.merge(inter_counts, on="user_name", how="left").merge(c_counts, on="user_name", how="left").fillna(0)
    users["n_interactions_network"] = users["n_interactions_network"].astype(int)
    users["n_interactions_comment"] = users["n_interactions_comment"].astype(int)
    users["n_total_interactions"] = users["n_interactions_network"] + users["n_interactions_comment"]
    users["n_m"] = users["user_name"].map(lambda u: len(target_m.get(u, set()))).astype(int)
    users["n_w"] = users["user_name"].map(lambda u: len(target_w.get(u, set()))).astype(int)

    def grp(r):
        if r["n_m"] + r["n_w"] == 0:
            return "no_media"
        if r["n_m"] > 0 and r["n_w"] == 0:
            return "mainstream_only"
        if r["n_m"] == 0 and r["n_w"] > 0:
            return "wemedia_only"
        return "dual"

    users["exposure_group"] = users.apply(grp, axis=1)
    users["targets_mainstream"] = users["user_name"].map(lambda u: "|".join(sorted(target_m.get(u, set()))))
    users["targets_wemedia"] = users["user_name"].map(lambda u: "|".join(sorted(target_w.get(u, set()))))

    # 交互活跃天数（repost/topic + comment）
    active_days = {}
    if detail_path.exists():
        de = pd.read_csv(detail_path, usecols=["source_user", "event_time"], dtype=str, low_memory=False)
        de["source_user"] = de["source_user"].map(_norm_text)
        de["event_time"] = pd.to_datetime(de["event_time"], errors="coerce")
        de["date"] = de["event_time"].dt.floor("D")
        t = de.dropna(subset=["date"]).groupby("source_user")["date"].nunique()
        active_days.update({k: int(v) for k, v in t.to_dict().items()})
    c["date"] = c["comment_time"].dt.floor("D")
    t2 = c.dropna(subset=["date"]).groupby("source_user")["date"].nunique()
    for u, n in t2.to_dict().items():
        active_days[u] = int(max(active_days.get(u, 0), int(n)))
    users["active_days_interaction"] = users["user_name"].map(lambda u: int(active_days.get(u, 0)))

    return users.sort_values("n_total_interactions", ascending=False).reset_index(drop=True), target_m, target_w


def build_daily_series(timeline: pd.DataFrame, study_mask: pd.Series) -> pd.DataFrame:
    d = timeline[study_mask].copy()
    d["is_risk"] = d["risk_class"].eq("risk")
    d["is_H"] = d["emotion_class"].eq("H")

    rows = []
    for day, g in d.groupby("date"):
        ms = g[g["user_type_raw"].isin(["mainstream", "government"])]
        wm = g[g["user_type_raw"] == "wemedia"]
        pub = g[g["node_type_3class"] == "ordinary"]
        rows.append(
            {
                "date": day,
                "n_mainstream_posts": int(len(ms)),
                "n_wemedia_posts": int(len(wm)),
                "n_public_posts": int(len(pub)),
                "mainstream_risk": float(ms["is_risk"].mean()) if len(ms) else np.nan,
                "wemedia_risk": float(wm["is_risk"].mean()) if len(wm) else np.nan,
                "public_H": float(pub["is_H"].mean()) if len(pub) else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def granger_test(y: np.ndarray, x: np.ndarray, lag: int) -> Dict[str, float]:
    # y_t ~ y_{t-1..lag}  vs  y_t ~ y_lags + x_lags
    n = len(y)
    if n <= 3 * lag + 5:
        return {"lag": lag, "n_obs": n, "F": np.nan, "p_value": np.nan}
    Y = y[lag:]
    y_lags = np.column_stack([y[lag - k : n - k] for k in range(1, lag + 1)])
    x_lags = np.column_stack([x[lag - k : n - k] for k in range(1, lag + 1)])
    Xr = np.column_stack([np.ones(len(Y)), y_lags])
    Xu = np.column_stack([np.ones(len(Y)), y_lags, x_lags])

    br, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, Y, rcond=None)
    rr = Y - Xr @ br
    ru = Y - Xu @ bu
    rss_r = float(np.sum(rr**2))
    rss_u = float(np.sum(ru**2))
    m = lag
    k_u = Xu.shape[1]
    df2 = len(Y) - k_u
    if df2 <= 0 or rss_u <= 0 or rss_r < rss_u:
        return {"lag": lag, "n_obs": len(Y), "F": np.nan, "p_value": np.nan}
    F = ((rss_r - rss_u) / m) / (rss_u / df2)
    p = float(1.0 - stats.f.cdf(F, m, df2))
    return {"lag": lag, "n_obs": int(len(Y)), "F": float(F), "p_value": p}


def build_user_transition_table(timeline: pd.DataFrame, exposure: pd.DataFrame, study_start: pd.Timestamp, study_end: pd.Timestamp, min_posts: int) -> pd.DataFrame:
    d = timeline[(timeline["publish_time"] >= study_start) & (timeline["publish_time"] <= study_end)].copy()
    d = d[d["emotion_class"].isin(EMOTIONS)].copy()
    d = d.sort_values(["user_name", "publish_time"]).reset_index(drop=True)

    exp = exposure[["user_name", "exposure_group", "n_total_interactions"]].copy()
    d = d.merge(exp, on="user_name", how="left")
    # 保留所有有发帖转移的用户：缺失交互信息的用户归入 no_media（观测上无媒体暴露）
    d["exposure_group"] = d["exposure_group"].fillna("no_media")
    d["n_total_interactions"] = pd.to_numeric(d["n_total_interactions"], errors="coerce").fillna(0).astype(int)

    rows = []
    for u, g in d.groupby("user_name"):
        if len(g) < min_posts:
            continue
        g = g.sort_values("publish_time")
        emo = g["emotion_class"].tolist()
        total_t = len(emo) - 1
        if total_t <= 0:
            continue
        c = {k: 0 for k in TRANSITIONS}
        trans_env = []
        for i in range(1, len(emo)):
            key = f"{emo[i-1]}->{emo[i]}"
            if key in c:
                c[key] += 1
            trans_env.append((g.iloc[i - 1]["publish_time"], emo[i - 1], emo[i]))
        upgrade = (c["M->H"] + c["L->H"]) / total_t
        cooling = (c["H->M"] + c["H->L"]) / total_t
        stable = c["M->M"] / total_t
        row = {
            "user_name": u,
            "n_posts": int(len(g)),
            "n_transitions": int(total_t),
            "exposure_group": g["exposure_group"].iloc[0],
            "n_total_interactions": int(g["n_total_interactions"].iloc[0]) if pd.notna(g["n_total_interactions"].iloc[0]) else 0,
            "active_days": int(g["publish_time"].dt.floor("D").nunique()),
            "first_post_month": g["publish_time"].iloc[0].strftime("%Y-%m") if pd.notna(g["publish_time"].iloc[0]) else "",
            "upgrade_rate": float(upgrade),
            "cooling_rate": float(cooling),
            "stable_rate": float(stable),
        }
        for k in TRANSITIONS:
            row[k] = int(c[k])
            row[f"p_{k}"] = float(c[k] / total_t)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.reset_index(drop=True)


def _standardize(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    std = float(x.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(np.zeros(len(x)), index=s.index)
    return (x - float(x.mean())) / std


def _fit_propensity(df: pd.DataFrame, covars: List[str], treat_col: str) -> np.ndarray:
    X = np.column_stack([np.ones(len(df))] + [_standardize(df[c]).values for c in covars])
    y = df[treat_col].astype(float).values

    def nll(beta):
        z = X @ beta
        # 稳定 sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        eps = 1e-8
        ll = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 1e-6 * np.sum(beta[1:] ** 2)
        return -ll + reg

    beta0 = np.zeros(X.shape[1])
    res = optimize.minimize(nll, beta0, method="L-BFGS-B")
    beta = res.x if res.success else beta0
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return p


@dataclass
class MatchResult:
    pair_df: pd.DataFrame
    caliper: float
    treated_n: int
    matched_treated_n: int
    matched_control_n: int


def nearest_neighbor_match(df: pd.DataFrame, treat_col: str, pscore_col: str, ratio: int, caliper_mult: float, seed: int) -> MatchResult:
    rng = np.random.default_rng(seed)
    t = df[df[treat_col] == 1].copy().reset_index(drop=True)
    c = df[df[treat_col] == 0].copy().reset_index(drop=True)
    if len(t) == 0 or len(c) == 0:
        return MatchResult(pd.DataFrame(columns=["treated_id", "control_id", "dist"]), np.nan, len(t), 0, 0)

    caliper = float(caliper_mult * df[pscore_col].std(ddof=0))
    available = set(c.index.tolist())
    t_idx = t.index.to_list()
    rng.shuffle(t_idx)
    pairs = []
    for i in t_idx:
        if not available:
            break
        ti = t.loc[i]
        cand = []
        for j in list(available):
            cj = c.loc[j]
            dist = abs(float(ti[pscore_col]) - float(cj[pscore_col]))
            if dist <= caliper:
                cand.append((dist, j))
        cand.sort(key=lambda x: x[0])
        pick = cand[: max(1, int(ratio))]
        if len(pick) < max(1, int(ratio)):
            continue
        for dist, j in pick:
            available.remove(j)
            pairs.append({"treated_id": int(ti["__id"]), "control_id": int(c.loc[j, "__id"]), "dist": float(dist)})
    pair_df = pd.DataFrame(pairs)
    return MatchResult(
        pair_df=pair_df,
        caliper=caliper,
        treated_n=int(len(t)),
        matched_treated_n=int(pair_df["treated_id"].nunique()) if len(pair_df) else 0,
        matched_control_n=int(pair_df["control_id"].nunique()) if len(pair_df) else 0,
    )


def smd(x_t: pd.Series, x_c: pd.Series) -> float:
    vt = float(x_t.var(ddof=1)) if len(x_t) > 1 else 0.0
    vc = float(x_c.var(ddof=1)) if len(x_c) > 1 else 0.0
    den = math.sqrt((vt + vc) / 2.0) if (vt + vc) > 0 else 0.0
    if den == 0:
        return 0.0
    return float((x_t.mean() - x_c.mean()) / den)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # O(n*m)；样本量中等可接受
    gt = 0
    lt = 0
    for a in x:
        gt += int(np.sum(a > y))
        lt += int(np.sum(a < y))
    n = len(x) * len(y)
    return float((gt - lt) / n) if n > 0 else np.nan


def compare_outcome(treated: np.ndarray, control: np.ndarray) -> Dict[str, float]:
    if len(treated) == 0 or len(control) == 0:
        return {"mannwhitney_p": np.nan, "cohen_d": np.nan, "cliffs_delta": np.nan}
    try:
        _u, p = stats.mannwhitneyu(treated, control, alternative="two-sided")
    except Exception:
        p = np.nan
    m1, m2 = float(np.mean(treated)), float(np.mean(control))
    s1 = float(np.std(treated, ddof=1)) if len(treated) > 1 else 0.0
    s2 = float(np.std(control, ddof=1)) if len(control) > 1 else 0.0
    pooled = math.sqrt(((len(treated) - 1) * s1**2 + (len(control) - 1) * s2**2) / max(1, len(treated) + len(control) - 2))
    d = (m1 - m2) / pooled if pooled > 0 else 0.0
    cd = cliffs_delta(treated, control)
    return {"mannwhitney_p": float(p) if p == p else np.nan, "cohen_d": float(d), "cliffs_delta": float(cd)}


def run_psm_pair(df_user: pd.DataFrame, treat_group: str, control_group: str, ratio: int, caliper_mult: float, seed: int) -> Dict[str, object]:
    d = df_user[df_user["exposure_group"].isin([treat_group, control_group])].copy().reset_index(drop=True)
    d["treat"] = (d["exposure_group"] == treat_group).astype(int)
    d["__id"] = np.arange(len(d))
    covars_base = ["n_total_interactions", "active_days", "n_posts"]
    month_col = "first_post_month"
    if month_col in d.columns:
        month_dummies = pd.get_dummies(d[month_col].fillna("unknown").astype(str), prefix="month", drop_first=True, dtype=float)
        for c in month_dummies.columns:
            d[c] = month_dummies[c].astype(float)
        month_covars = month_dummies.columns.tolist()
    else:
        month_covars = []
    covars = covars_base + month_covars
    d["pscore"] = _fit_propensity(d, covars=covars, treat_col="treat")

    mr = nearest_neighbor_match(d, "treat", "pscore", ratio=ratio, caliper_mult=caliper_mult, seed=seed)
    if len(mr.pair_df) == 0:
        return {
            "pair": f"{treat_group}_vs_{control_group}",
            "status": "no_match",
            "treated_n": int((d["treat"] == 1).sum()),
            "control_n": int((d["treat"] == 0).sum()),
        }

    t_map = d.set_index("__id")
    p = mr.pair_df.copy()
    p["treat_upgrade"] = p["treated_id"].map(t_map["upgrade_rate"])
    p["ctrl_upgrade"] = p["control_id"].map(t_map["upgrade_rate"])
    p["treat_cooling"] = p["treated_id"].map(t_map["cooling_rate"])
    p["ctrl_cooling"] = p["control_id"].map(t_map["cooling_rate"])
    p["treat_stable"] = p["treated_id"].map(t_map["stable_rate"])
    p["ctrl_stable"] = p["control_id"].map(t_map["stable_rate"])

    # ATT: 先按 treated 聚合 control 均值
    agg = p.groupby("treated_id", as_index=False).agg(
        treat_upgrade=("treat_upgrade", "first"),
        ctrl_upgrade=("ctrl_upgrade", "mean"),
        treat_cooling=("treat_cooling", "first"),
        ctrl_cooling=("ctrl_cooling", "mean"),
        treat_stable=("treat_stable", "first"),
        ctrl_stable=("ctrl_stable", "mean"),
    )
    agg["diff_upgrade"] = agg["treat_upgrade"] - agg["ctrl_upgrade"]
    agg["diff_cooling"] = agg["treat_cooling"] - agg["ctrl_cooling"]
    agg["diff_stable"] = agg["treat_stable"] - agg["ctrl_stable"]

    # balance
    t_before = d[d["treat"] == 1]
    c_before = d[d["treat"] == 0]
    t_after_ids = set(p["treated_id"].tolist())
    c_after_ids = set(p["control_id"].tolist())
    t_after = d[d["__id"].isin(t_after_ids)]
    c_after = d[d["__id"].isin(c_after_ids)]
    bal_rows = []
    for cv in covars:
        bal_rows.append(
            {
                "pair": f"{treat_group}_vs_{control_group}",
                "covariate": cv,
                "smd_before": smd(t_before[cv], c_before[cv]),
                "smd_after": smd(t_after[cv], c_after[cv]),
            }
        )

    # outcome tests（以 pair-level 展开样本）
    out = {}
    for nm, tc, cc, dc in [
        ("upgrade", "treat_upgrade", "ctrl_upgrade", "diff_upgrade"),
        ("cooling", "treat_cooling", "ctrl_cooling", "diff_cooling"),
        ("stable", "treat_stable", "ctrl_stable", "diff_stable"),
    ]:
        cmp = compare_outcome(p[tc].values.astype(float), p[cc].values.astype(float))
        out[nm] = {
            "treated_mean": float(np.mean(p[tc])),
            "control_mean": float(np.mean(p[cc])),
            "att_mean_diff": float(np.mean(agg[dc])),
            **cmp,
        }

    return {
        "pair": f"{treat_group}_vs_{control_group}",
        "status": "ok",
        "treated_n": int((d["treat"] == 1).sum()),
        "control_n": int((d["treat"] == 0).sum()),
        "caliper": mr.caliper,
        "matched_treated_n": mr.matched_treated_n,
        "matched_control_n": mr.matched_control_n,
        "covariates_used": covars,
        "outcomes": out,
        "balance_rows": bal_rows,
        "pair_level_path_data": p,
        "treated_level_path_data": agg,
    }


def compute_transition_group_table(df_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, x in df_user.groupby("exposure_group"):
        if len(x) == 0:
            continue
        row = {
            "exposure_group": g,
            "n_users": int(len(x)),
            "upgrade_rate_mean": float(x["upgrade_rate"].mean()),
            "cooling_rate_mean": float(x["cooling_rate"].mean()),
            "stable_rate_mean": float(x["stable_rate"].mean()),
        }
        for t in TRANSITIONS:
            row[f"p_{t}_mean"] = float(x[f"p_{t}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_users", ascending=False).reset_index(drop=True)


def classify_transition_environment(
    timeline: pd.DataFrame,
    df_user: pd.DataFrame,
    target_m: Dict[str, set],
    target_w: Dict[str, set],
) -> pd.DataFrame:
    # 每天每个媒体账号 risk 比例
    media = timeline[timeline["node_type_3class"].isin(["mainstream", "wemedia"])].copy()
    media["is_risk"] = media["risk_class"].eq("risk").astype(float)
    day_user_risk = media.groupby(["date", "user_name"], as_index=False).agg(risk_ratio=("is_risk", "mean"), n_posts=("is_risk", "size"))
    risk_lookup = {(r["date"], r["user_name"]): float(r["risk_ratio"]) for _, r in day_user_risk.iterrows()}

    # 逐用户构建转移并标注环境
    rows = []
    d = timeline[timeline["in_study_window"]].copy().sort_values(["user_name", "publish_time"])
    keep_users = set(df_user["user_name"].tolist())
    d = d[d["user_name"].isin(keep_users)]
    user_group = dict(zip(df_user["user_name"], df_user["exposure_group"]))
    for u, g in d.groupby("user_name"):
        emo = g["emotion_class"].tolist()
        ts = g["publish_time"].tolist()
        dates = g["date"].tolist()
        if len(emo) < 2:
            continue
        m_targets = target_m.get(u, set())
        w_targets = target_w.get(u, set())
        targets = sorted(m_targets | w_targets)
        for i in range(1, len(emo)):
            prev_day = dates[i - 1]
            vals = []
            for t in targets:
                key = (prev_day, t)
                if key in risk_lookup:
                    vals.append(risk_lookup[key])
            if len(vals) == 0:
                env = "unknown"
                env_ratio = np.nan
            else:
                env_ratio = float(np.mean(vals))
                env = "risk" if env_ratio > 0.5 else "norisk"
            rows.append(
                {
                    "user_name": u,
                    "exposure_group": user_group.get(u, ""),
                    "prev_time": ts[i - 1],
                    "curr_time": ts[i],
                    "transition": f"{emo[i-1]}->{emo[i]}",
                    "env_type": env,
                    "env_risk_ratio": env_ratio,
                }
            )
    out = pd.DataFrame(rows)
    return out


def summarize_mechanism(trans_env: pd.DataFrame) -> pd.DataFrame:
    d = trans_env[trans_env["env_type"].isin(["risk", "norisk"])].copy()
    if len(d) == 0:
        return pd.DataFrame(columns=["exposure_group", "env_type", "n_transitions", "p_MH", "p_HM", "upgrade_rate", "cooling_rate"])

    rows = []
    for (g, e), x in d.groupby(["exposure_group", "env_type"]):
        n = len(x)
        c = x["transition"].value_counts()
        p_mh = float(c.get("M->H", 0) / n)
        p_hm = float(c.get("H->M", 0) / n)
        upgrade = float((c.get("M->H", 0) + c.get("L->H", 0)) / n)
        cooling = float((c.get("H->M", 0) + c.get("H->L", 0)) / n)
        rows.append(
            {
                "exposure_group": g,
                "env_type": e,
                "n_transitions": int(n),
                "p_MH": p_mh,
                "p_HM": p_hm,
                "upgrade_rate": upgrade,
                "cooling_rate": cooling,
            }
        )
    return pd.DataFrame(rows).sort_values(["env_type", "exposure_group"]).reset_index(drop=True)


def bootstrap_ci(x: np.ndarray, iters: int, seed: int) -> Tuple[float, float]:
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(int(iters)):
        idx = rng.integers(0, n, size=n)
        vals.append(float(np.mean(x[idx])))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def rosenbaum_like_bounds(diff: np.ndarray, gammas: Iterable[float] = (1.0, 1.25, 1.5, 2.0)) -> Dict[str, float]:
    # 近似版：基于 matched treated-level 差异的符号检验上界
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {f"gamma_{g}": np.nan for g in gammas}
    s = int(np.sum(d > 0))
    out = {}
    for g in gammas:
        p = float(g / (1.0 + g))
        out[f"gamma_{g}"] = float(stats.binom.sf(s - 1, n, p))
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapper = UserTypeMapper()
    ann_paths = [ROOT / p for p in args.annotations]

    # 0) 数据准备
    verify_lut = build_verify_lookup(ROOT / "dataset/Topic_data", ROOT / "dataset/User_data/USER_INFO_1.csv")
    user_exposure, target_m, target_w = build_user_exposure(ROOT / args.network_dir, verify_lut, mapper)
    timeline = load_master_timeline(mapper, ROOT / "dataset/Topic_data", ann_paths)
    if len(timeline) == 0:
        raise RuntimeError("时间线为空：请检查 Topic_data 与标注文件是否匹配（mid）。")

    t_min = pd.to_datetime(timeline["publish_time"], errors="coerce").min()
    t_max = pd.to_datetime(timeline["publish_time"], errors="coerce").max()
    if pd.isna(t_min) or pd.isna(t_max):
        raise RuntimeError("时间线 publish_time 全为空，无法构建研究窗口。")
    if args.full_window or str(args.study_start).strip().lower() == "auto" or str(args.study_end).strip().lower() == "auto":
        study_start = pd.Timestamp(t_min).floor("D")
        study_end = pd.Timestamp(t_max).floor("D") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
        study_start = pd.Timestamp(args.study_start)
        study_end = pd.Timestamp(args.study_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    timeline["in_study_window"] = (timeline["publish_time"] >= study_start) & (timeline["publish_time"] <= study_end)

    # 把发布活跃信息补到 exposure
    in_win = timeline[timeline["in_study_window"]].copy()
    u_post = in_win.groupby("user_name", as_index=False).agg(
        n_posts=("mid", "size"),
        active_days_post=("date", "nunique"),
    )
    user_exposure = user_exposure.merge(u_post, on="user_name", how="left")
    user_exposure["n_posts"] = user_exposure["n_posts"].fillna(0).astype(int)
    user_exposure["active_days_post"] = user_exposure["active_days_post"].fillna(0).astype(int)
    user_exposure["active_days"] = np.maximum(user_exposure["active_days_interaction"], user_exposure["active_days_post"]).astype(int)
    user_exposure.to_csv(out_dir / "user_exposure.csv", index=False, encoding="utf-8-sig")

    timeline_out = timeline[
        [
            "user_name",
            "mid",
            "publish_time",
            "emotion_class",
            "risk_class",
            "emotion_confidence",
            "user_type_raw",
            "node_type_3class",
            "in_study_window",
        ]
    ].copy()
    timeline_out.to_csv(out_dir / "user_emotion_timeline.csv", index=False, encoding="utf-8-sig")

    # 1) 聚合层面 + Granger
    daily = build_daily_series(timeline, timeline["in_study_window"])
    daily.to_csv(out_dir / "daily_series.csv", index=False, encoding="utf-8-sig")

    # Granger 用 ffill + 0 回填序列，避免断点造成过多缺失
    g = daily.copy()
    for c in ["mainstream_risk", "wemedia_risk", "public_H"]:
        g[c] = g[c].astype(float).ffill().fillna(0.0)

    granger = {
        "study_window": {"start": str(study_start), "end": str(study_end)},
        "annotation_sources": [str(p) for p in ann_paths],
    }
    diag = {}
    for c in ["mainstream_risk", "wemedia_risk", "public_H"]:
        v = g[c].astype(float).values
        diag[c] = {
            "std": float(np.std(v, ddof=0)),
            "n_unique": int(pd.Series(v).nunique()),
            "is_constant": bool(pd.Series(v).nunique() <= 1),
            "mean": float(np.mean(v)),
        }
    granger["series_diagnostics"] = diag
    res_m = []
    res_w = []
    y = g["public_H"].values
    x_m = g["mainstream_risk"].values
    x_w = g["wemedia_risk"].values
    for lag in [1, 2, 3]:
        res_m.append(granger_test(y, x_m, lag))
        res_w.append(granger_test(y, x_w, lag))
    granger["mainstream_risk_to_public_H"] = res_m
    granger["wemedia_risk_to_public_H"] = res_w
    (out_dir / "granger_results.json").write_text(json.dumps(granger, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 个体状态转移 + PSM
    df_user = build_user_transition_table(timeline, user_exposure, study_start, study_end, min_posts=args.min_posts)
    df_user.to_csv(out_dir / "transition_by_user.csv", index=False, encoding="utf-8-sig")
    trans_group = compute_transition_group_table(df_user)
    trans_group.to_csv(out_dir / "transition_by_group.csv", index=False, encoding="utf-8-sig")

    comparisons = [("dual", "mainstream_only"), ("dual", "wemedia_only")]
    main_results = {"comparisons": [], "sample_info": {}}
    bal_rows = []
    pair_level_all = []
    treated_level_all = []
    for tg, cg in comparisons:
        r = run_psm_pair(df_user, tg, cg, ratio=args.psm_ratio, caliper_mult=args.psm_caliper_mult, seed=args.seed)
        main_results["comparisons"].append(
            {
                k: v
                for k, v in r.items()
                if k not in ["balance_rows", "pair_level_path_data", "treated_level_path_data"]
            }
        )
        if r.get("status") == "ok":
            bal_rows.extend(r["balance_rows"])
            p = r["pair_level_path_data"].copy()
            p["pair"] = r["pair"]
            pair_level_all.append(p)
            t = r["treated_level_path_data"].copy()
            t["pair"] = r["pair"]
            treated_level_all.append(t)

    if bal_rows:
        pd.DataFrame(bal_rows).to_csv(out_dir / "psm_balance.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["pair", "covariate", "smd_before", "smd_after"]).to_csv(out_dir / "psm_balance.csv", index=False, encoding="utf-8-sig")

    if pair_level_all:
        pd.concat(pair_level_all, ignore_index=True).to_csv(out_dir / "psm_pairs_detail.csv", index=False, encoding="utf-8-sig")
    if treated_level_all:
        pd.concat(treated_level_all, ignore_index=True).to_csv(out_dir / "psm_treated_level_diff.csv", index=False, encoding="utf-8-sig")

    psm_groups = ["mainstream_only", "wemedia_only", "dual"]
    psm_pool = df_user[df_user["exposure_group"].isin(psm_groups)].copy()
    main_results["sample_info"] = {
        "n_users_transition_sample_all_groups": int(len(df_user)),
        "group_counts_all_groups": {k: int(v) for k, v in df_user["exposure_group"].value_counts().to_dict().items()},
        "n_users_transition_sample": int(len(psm_pool)),
        "group_counts": {k: int(v) for k, v in psm_pool["exposure_group"].value_counts().to_dict().items()},
        "psm_groups": psm_groups,
        "month_fe": "first_post_month",
        "study_window": {"start": str(study_start), "end": str(study_end)},
        "annotation_sources": [str(p) for p in ann_paths],
        "min_posts": int(args.min_posts),
        "psm_ratio": int(args.psm_ratio),
        "psm_caliper_mult": float(args.psm_caliper_mult),
    }
    (out_dir / "main_results.json").write_text(json.dumps(main_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) 机制：risk/norisk 环境
    trans_env = classify_transition_environment(timeline, df_user, target_m, target_w)
    trans_env.to_csv(out_dir / "transition_with_environment.csv", index=False, encoding="utf-8-sig")
    mech = summarize_mechanism(trans_env)
    mech.to_csv(out_dir / "mechanism_results.csv", index=False, encoding="utf-8-sig")

    # 周度动态（按 prev_time 周）
    if len(trans_env):
        w = trans_env[trans_env["env_type"].isin(["risk", "norisk"])].copy()
        w["week"] = pd.to_datetime(w["prev_time"], errors="coerce").dt.to_period("W").astype(str)
        rows = []
        for (grp, wk), x in w.groupby(["exposure_group", "week"]):
            n = len(x)
            c = x["transition"].value_counts()
            rows.append(
                {
                    "exposure_group": grp,
                    "week": wk,
                    "n_transitions": int(n),
                    "upgrade_rate": float((c.get("M->H", 0) + c.get("L->H", 0)) / n),
                    "cooling_rate": float((c.get("H->M", 0) + c.get("H->L", 0)) / n),
                }
            )
        pd.DataFrame(rows).sort_values(["week", "exposure_group"]).to_csv(out_dir / "weekly_dynamic.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["exposure_group", "week", "n_transitions", "upgrade_rate", "cooling_rate"]).to_csv(out_dir / "weekly_dynamic.csv", index=False, encoding="utf-8-sig")

    # 4) 稳健性
    robustness = {
        "matching_ratio_sensitivity": [],
        "active_threshold_sensitivity": [],
        "bootstrap": {},
        "exclude_top1pct": {},
        "rosenbaum_like_bounds": {},
    }

    # 4.1 匹配比例
    for rr in [1, 3, 5]:
        r = run_psm_pair(df_user, "dual", "wemedia_only", ratio=rr, caliper_mult=args.psm_caliper_mult, seed=args.seed)
        record = {"ratio": rr, "status": r.get("status")}
        if r.get("status") == "ok":
            record["cooling_att"] = r["outcomes"]["cooling"]["att_mean_diff"]
            record["cooling_p"] = r["outcomes"]["cooling"]["mannwhitney_p"]
            record["matched_treated_n"] = r["matched_treated_n"]
        robustness["matching_ratio_sensitivity"].append(record)

    # 4.2 活跃阈值
    for th in [3, 5, 10]:
        d2 = df_user[df_user["n_posts"] >= th].copy()
        r = run_psm_pair(d2, "dual", "wemedia_only", ratio=args.psm_ratio, caliper_mult=args.psm_caliper_mult, seed=args.seed)
        record = {"min_posts_threshold": th, "status": r.get("status"), "n_users": int(len(d2))}
        if r.get("status") == "ok":
            record["cooling_att"] = r["outcomes"]["cooling"]["att_mean_diff"]
            record["cooling_p"] = r["outcomes"]["cooling"]["mannwhitney_p"]
            record["matched_treated_n"] = r["matched_treated_n"]
        robustness["active_threshold_sensitivity"].append(record)

    # 4.3 Bootstrap + Rosenbaum-like
    base = run_psm_pair(df_user, "dual", "wemedia_only", ratio=args.psm_ratio, caliper_mult=args.psm_caliper_mult, seed=args.seed)
    if base.get("status") == "ok":
        td = base["treated_level_path_data"]
        diff = td["diff_cooling"].values.astype(float)
        lo, hi = bootstrap_ci(diff, args.bootstrap_iters, args.seed)
        robustness["bootstrap"] = {
            "pair": "dual_vs_wemedia_only",
            "iters": int(args.bootstrap_iters),
            "att_mean": float(np.mean(diff)),
            "ci95_low": lo,
            "ci95_high": hi,
        }
        robustness["rosenbaum_like_bounds"] = {
            "pair": "dual_vs_wemedia_only",
            "metric": "diff_cooling_sign_test_upper_p",
            **rosenbaum_like_bounds(diff),
        }

    # 4.4 排除 top1%
    if len(df_user):
        q99 = float(df_user["n_total_interactions"].quantile(0.99))
        d3 = df_user[df_user["n_total_interactions"] <= q99].copy()
        r = run_psm_pair(d3, "dual", "wemedia_only", ratio=args.psm_ratio, caliper_mult=args.psm_caliper_mult, seed=args.seed)
        robustness["exclude_top1pct"] = {
            "threshold_q99": q99,
            "n_users_after_filter": int(len(d3)),
            "status": r.get("status"),
        }
        if r.get("status") == "ok":
            robustness["exclude_top1pct"]["cooling_att"] = r["outcomes"]["cooling"]["att_mean_diff"]
            robustness["exclude_top1pct"]["cooling_p"] = r["outcomes"]["cooling"]["mannwhitney_p"]

    (out_dir / "robustness_checks.json").write_text(json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可视化
    plt = _ensure_matplotlib()

    # fig_transition_heatmap: 三组平均转移概率
    grp_order = ["mainstream_only", "wemedia_only", "dual"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    for ax, grp in zip(axes, grp_order):
        g = df_user[df_user["exposure_group"] == grp]
        mat = np.zeros((3, 3), dtype=float)
        if len(g):
            for i, a in enumerate(EMOTIONS):
                for j, b in enumerate(EMOTIONS):
                    mat[i, j] = float(g[f"p_{a}->{b}"].mean())
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=max(0.4, float(mat.max()) if len(g) else 0.4))
        ax.set_title(grp)
        ax.set_xticks(range(3), EMOTIONS)
        ax.set_yticks(range(3), EMOTIONS)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    fig.suptitle("Transition Matrix by Exposure Group")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_transition_heatmap.png", dpi=220)
    plt.close(fig)

    # fig_buffering_effect: 组均值条形图（upgrade/cooling）
    stat = df_user.groupby("exposure_group", as_index=False).agg(
        n=("user_name", "size"),
        upgrade_mean=("upgrade_rate", "mean"),
        cooling_mean=("cooling_rate", "mean"),
    )
    stat = stat[stat["exposure_group"].isin(grp_order)].copy()
    stat["exposure_group"] = pd.Categorical(stat["exposure_group"], categories=grp_order, ordered=True)
    stat = stat.sort_values("exposure_group")
    x = np.arange(len(stat))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, stat["upgrade_mean"].values, width=w, label="upgrade_rate")
    ax.bar(x + w / 2, stat["cooling_mean"].values, width=w, label="cooling_rate")
    ax.set_xticks(x, stat["exposure_group"].astype(str).tolist())
    ax.set_ylabel("Mean Probability")
    ax.set_title("Buffering Effect by Exposure Group")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_buffering_effect.png", dpi=220)
    plt.close(fig)

    # 简要控制台输出
    print("V2 实验执行完成")
    print(f"- output dir: {out_dir}")
    print(f"- user_exposure: {len(user_exposure)} users")
    print(f"- timeline(annotated): {len(timeline)} posts")
    print(f"- study window: {study_start} ~ {study_end}")
    print(f"- annotation sources: {len(ann_paths)} files")
    print(f"- transition sample (all groups): {len(df_user)} users")
    print(f"- transition sample (PSM groups): {len(psm_pool)} users")
    print(f"- granger file: {out_dir / 'granger_results.json'}")
    print(f"- main results: {out_dir / 'main_results.json'}")
    print(f"- robustness: {out_dir / 'robustness_checks.json'}")


if __name__ == "__main__":
    main()
