#!/usr/bin/env python3
"""
基于微博转发行为构建用户有向网络（source_user -> target_user）。

两类边来源：
1) Topic 直连边：user_name -> origin_weibo_user_name
2) Repost 映射边：user_name -> (origin_mid 对应的原帖 user_name)

输出：
- edge_detail.csv: 事件级边（每条转发/直连记录一行）
- edge_weighted.csv: 聚合边（同一 source-target 合并，给出权重）
- node_degree.csv: 节点入/出度（加权与邻居数）
- unresolved_origin_mid.csv: 未映射 origin_mid（若存在）
- build_report.json: 过程统计与映射覆盖率
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建微博转发用户网络（有向）")
    p.add_argument("--topic-dir", default="dataset/Topic_data", help="Topic CSV 目录")
    p.add_argument("--repost-csv", default="dataset/Repost/REPOST.csv", help="Repost CSV 路径")
    p.add_argument(
        "--output-dir",
        default="outputs/network/repost_user_network",
        help="输出目录",
    )
    p.add_argument(
        "--map-source",
        choices=["topic", "topic_plus_repost"],
        default="topic_plus_repost",
        help="origin_mid 映射字典来源：仅 topic，或 topic+repost(mid->user_name)",
    )
    p.add_argument(
        "--drop-self-loops",
        action="store_true",
        help="是否删除 source_user == target_user 的自环边",
    )
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


def _read_csv_usecols(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(usecols), dtype=str, low_memory=False)


def _build_mid_user_map_from_topic(
    topic_files: List[Path],
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], List[Tuple[str, int]]]:
    rows = []
    stats = {
        "files_scanned": len(topic_files),
        "files_with_mid_user_cols": 0,
        "rows_mid_user_raw": 0,
        "rows_mid_user_valid": 0,
    }
    for f in topic_files:
        try:
            head = pd.read_csv(f, nrows=0)
        except Exception:
            continue
        cols = set(head.columns)
        if not {"mid", "user_name"}.issubset(cols):
            continue
        stats["files_with_mid_user_cols"] += 1
        try:
            d = _read_csv_usecols(f, ["mid", "user_name"])
        except Exception:
            continue
        stats["rows_mid_user_raw"] += len(d)
        d["mid"] = d["mid"].map(_norm_mid)
        d["user_name"] = d["user_name"].map(_norm_text)
        d = d[(d["mid"] != "") & (d["user_name"] != "")]
        stats["rows_mid_user_valid"] += len(d)
        rows.append(d)

    if not rows:
        return pd.DataFrame(columns=["mid", "user_name"]), stats, {}, []

    full = pd.concat(rows, ignore_index=True)

    conflict_counts = (
        full.groupby("mid")["user_name"].nunique().reset_index(name="n_users").query("n_users > 1")
    )
    conflict_dict = {r["mid"]: int(r["n_users"]) for _, r in conflict_counts.iterrows()}

    conflict_examples = []
    if len(conflict_counts) > 0:
        mids = set(conflict_counts["mid"].head(20).tolist())
        sample = (
            full[full["mid"].isin(mids)]
            .drop_duplicates(subset=["mid", "user_name"])
            .sort_values(["mid", "user_name"])
        )
        for mid, g in sample.groupby("mid"):
            names = g["user_name"].tolist()
            conflict_examples.append((mid, len(names)))

    # KISS + 稳定性：同一 mid 多人时采用“出现频次最高”的 user_name；若并列按字典序最小
    vote = (
        full.groupby(["mid", "user_name"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["mid", "n", "user_name"], ascending=[True, False, True])
    )
    uniq = vote.drop_duplicates(subset=["mid"], keep="first")[["mid", "user_name"]].reset_index(drop=True)
    return uniq, stats, conflict_dict, conflict_examples


def _build_topic_direct_edges(topic_files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows = []
    stats = {
        "files_scanned": len(topic_files),
        "files_with_direct_cols": 0,
        "rows_direct_raw": 0,
        "rows_direct_valid": 0,
        "rows_direct_dedup_by_mid": 0,
    }
    for f in topic_files:
        try:
            head = pd.read_csv(f, nrows=0)
        except Exception:
            continue
        cols = set(head.columns)
        if not {"user_name", "origin_weibo_user_name"}.issubset(cols):
            continue
        stats["files_with_direct_cols"] += 1

        use = ["user_name", "origin_weibo_user_name"]
        if "mid" in cols:
            use.append("mid")
        if "publish_time" in cols:
            use.append("publish_time")
        try:
            d = _read_csv_usecols(f, use)
        except Exception:
            continue
        stats["rows_direct_raw"] += len(d)

        d["source_user"] = d["user_name"].map(_norm_text)
        d["target_user"] = d["origin_weibo_user_name"].map(_norm_text)
        d["event_mid"] = d["mid"].map(_norm_mid) if "mid" in d.columns else ""
        d["event_time"] = d["publish_time"].map(_norm_text) if "publish_time" in d.columns else ""
        d = d[(d["source_user"] != "") & (d["target_user"] != "")]
        stats["rows_direct_valid"] += len(d)

        d = d.assign(
            edge_type="topic_direct",
            origin_mid="",
            mapped_flag=True,
            source_file=str(f),
        )[
            [
                "event_mid",
                "event_time",
                "source_user",
                "target_user",
                "edge_type",
                "origin_mid",
                "mapped_flag",
                "source_file",
            ]
        ]
        rows.append(d)

    if not rows:
        return pd.DataFrame(
            columns=[
                "event_mid",
                "event_time",
                "source_user",
                "target_user",
                "edge_type",
                "origin_mid",
                "mapped_flag",
                "source_file",
            ]
        ), stats

    full = pd.concat(rows, ignore_index=True)
    # Topic 文件间可能重叠，优先按 event_mid 去重（若缺失 mid，则保留）
    with_mid = full[full["event_mid"] != ""].drop_duplicates(subset=["event_mid"], keep="first")
    no_mid = full[full["event_mid"] == ""]
    dedup = pd.concat([with_mid, no_mid], ignore_index=True)
    stats["rows_direct_dedup_by_mid"] = len(dedup)
    return dedup, stats


def _build_repost_edges(
    repost_csv: Path,
    mid_to_user: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    d = _read_csv_usecols(repost_csv, ["origin_mid", "mid", "publish_time", "user_name"])
    rows_raw = len(d)

    d["origin_mid"] = d["origin_mid"].map(_norm_mid)
    d["event_mid"] = d["mid"].map(_norm_mid)
    d["event_time"] = d["publish_time"].map(_norm_text)
    d["source_user"] = d["user_name"].map(_norm_text)
    d = d[(d["origin_mid"] != "") & (d["source_user"] != "")]

    # Repost 记录可能重复，优先按转发 mid 去重
    with_mid = d[d["event_mid"] != ""].drop_duplicates(subset=["event_mid"], keep="first")
    no_mid = d[d["event_mid"] == ""]
    d = pd.concat([with_mid, no_mid], ignore_index=True)
    rows_dedup = len(d)

    d["target_user"] = d["origin_mid"].map(lambda m: mid_to_user.get(m, ""))
    d["mapped_flag"] = d["target_user"] != ""
    mapped = d[d["mapped_flag"]].copy()
    unresolved = d[~d["mapped_flag"]].copy()

    detail = mapped.assign(
        edge_type="repost_mid_mapped",
        source_file=str(repost_csv),
    )[
        [
            "event_mid",
            "event_time",
            "source_user",
            "target_user",
            "edge_type",
            "origin_mid",
            "mapped_flag",
            "source_file",
        ]
    ]

    stats = {
        "rows_raw": float(rows_raw),
        "rows_dedup_by_mid": float(rows_dedup),
        "mapped_rows": float(len(mapped)),
        "mapped_ratio": float(len(mapped) / rows_dedup) if rows_dedup else 0.0,
        "unresolved_rows": float(len(unresolved)),
        "unresolved_unique_origin_mid": float(unresolved["origin_mid"].nunique()),
    }
    return detail, unresolved, stats


def _aggregate_weighted_edges(detail: pd.DataFrame) -> pd.DataFrame:
    if len(detail) == 0:
        return pd.DataFrame(columns=["source_user", "target_user", "weight", "n_topic_direct", "n_repost_mapped"])

    out = detail.groupby(["source_user", "target_user"], as_index=False).agg(
        weight=("edge_type", "size"),
        n_topic_direct=("edge_type", lambda s: int((s == "topic_direct").sum())),
        n_repost_mapped=("edge_type", lambda s: int((s == "repost_mid_mapped").sum())),
    )
    out = out.sort_values("weight", ascending=False).reset_index(drop=True)
    return out


def _build_node_degree(weighted_edges: pd.DataFrame) -> pd.DataFrame:
    if len(weighted_edges) == 0:
        return pd.DataFrame(
            columns=[
                "user_name",
                "out_weight",
                "in_weight",
                "total_weight",
                "out_neighbors",
                "in_neighbors",
            ]
        )

    out_w = weighted_edges.groupby("source_user", as_index=False)["weight"].sum().rename(
        columns={"source_user": "user_name", "weight": "out_weight"}
    )
    in_w = weighted_edges.groupby("target_user", as_index=False)["weight"].sum().rename(
        columns={"target_user": "user_name", "weight": "in_weight"}
    )
    out_n = weighted_edges.groupby("source_user", as_index=False)["target_user"].nunique().rename(
        columns={"source_user": "user_name", "target_user": "out_neighbors"}
    )
    in_n = weighted_edges.groupby("target_user", as_index=False)["source_user"].nunique().rename(
        columns={"target_user": "user_name", "source_user": "in_neighbors"}
    )

    nodes = (
        pd.DataFrame({"user_name": pd.unique(pd.concat([weighted_edges["source_user"], weighted_edges["target_user"]]))})
        .merge(out_w, on="user_name", how="left")
        .merge(in_w, on="user_name", how="left")
        .merge(out_n, on="user_name", how="left")
        .merge(in_n, on="user_name", how="left")
        .fillna(0)
    )
    for c in ["out_weight", "in_weight", "out_neighbors", "in_neighbors"]:
        nodes[c] = nodes[c].astype(int)
    nodes["total_weight"] = nodes["out_weight"] + nodes["in_weight"]
    nodes = nodes.sort_values("total_weight", ascending=False).reset_index(drop=True)
    return nodes


def main() -> None:
    args = parse_args()
    topic_dir = Path(args.topic_dir)
    repost_csv = Path(args.repost_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_files = sorted(topic_dir.glob("*.csv"))

    # 1) Topic 直连边
    topic_direct, topic_direct_stats = _build_topic_direct_edges(topic_files)

    # 2) mid -> user 映射（默认 topic + repost）
    topic_mid_map_df, mid_map_stats, conflict_dict, conflict_examples = _build_mid_user_map_from_topic(topic_files)
    mid_map = dict(zip(topic_mid_map_df["mid"], topic_mid_map_df["user_name"]))
    if args.map_source == "topic_plus_repost":
        rp_mid = _read_csv_usecols(repost_csv, ["mid", "user_name"])
        rp_mid["mid"] = rp_mid["mid"].map(_norm_mid)
        rp_mid["user_name"] = rp_mid["user_name"].map(_norm_text)
        rp_mid = rp_mid[(rp_mid["mid"] != "") & (rp_mid["user_name"] != "")]
        rp_mid = rp_mid.drop_duplicates(subset=["mid"], keep="first")
        for m, u in zip(rp_mid["mid"], rp_mid["user_name"]):
            if m not in mid_map:
                mid_map[m] = u

    # 3) Repost 映射边
    repost_edges, unresolved, repost_stats = _build_repost_edges(repost_csv, mid_map)

    # 4) 汇总网络
    edge_detail = pd.concat([topic_direct, repost_edges], ignore_index=True)
    edge_detail = edge_detail[(edge_detail["source_user"] != "") & (edge_detail["target_user"] != "")]
    self_loop_removed = 0
    if args.drop_self_loops:
        before = len(edge_detail)
        edge_detail = edge_detail[edge_detail["source_user"] != edge_detail["target_user"]].reset_index(drop=True)
        self_loop_removed = before - len(edge_detail)
    weighted_edges = _aggregate_weighted_edges(edge_detail)
    node_degree = _build_node_degree(weighted_edges)

    # 5) 输出文件
    detail_path = output_dir / "edge_detail.csv"
    weighted_path = output_dir / "edge_weighted.csv"
    degree_path = output_dir / "node_degree.csv"
    unresolved_path = output_dir / "unresolved_origin_mid.csv"
    report_path = output_dir / "build_report.json"

    edge_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    weighted_edges.to_csv(weighted_path, index=False, encoding="utf-8-sig")
    node_degree.to_csv(degree_path, index=False, encoding="utf-8-sig")

    if len(unresolved) > 0:
        unresolved_out = (
            unresolved.groupby("origin_mid", as_index=False)
            .size()
            .rename(columns={"size": "count_rows"})
            .sort_values("count_rows", ascending=False)
            .reset_index(drop=True)
        )
        unresolved_out.to_csv(unresolved_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["origin_mid", "count_rows"]).to_csv(unresolved_path, index=False, encoding="utf-8-sig")

    report = {
        "inputs": {
            "topic_dir": str(topic_dir),
            "repost_csv": str(repost_csv),
            "map_source": args.map_source,
        },
        "topic_direct_stats": topic_direct_stats,
        "mid_user_map_stats": {
            **mid_map_stats,
            "map_size_final": len(mid_map),
            "mid_conflicts_count": len(conflict_dict),
            "mid_conflict_examples": conflict_examples[:20],
        },
        "repost_mapping_stats": repost_stats,
        "network_stats": {
            "edge_detail_rows": int(len(edge_detail)),
            "edge_weighted_rows": int(len(weighted_edges)),
            "node_count": int(len(node_degree)),
            "topic_direct_rows_in_network": int((edge_detail["edge_type"] == "topic_direct").sum()),
            "repost_mapped_rows_in_network": int((edge_detail["edge_type"] == "repost_mid_mapped").sum()),
            "drop_self_loops": bool(args.drop_self_loops),
            "self_loop_removed_rows": int(self_loop_removed),
        },
        "outputs": {
            "edge_detail_csv": str(detail_path),
            "edge_weighted_csv": str(weighted_path),
            "node_degree_csv": str(degree_path),
            "unresolved_origin_mid_csv": str(unresolved_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("网络构建完成")
    print(f"- edge_detail:   {detail_path} (rows={len(edge_detail)})")
    print(f"- edge_weighted: {weighted_path} (rows={len(weighted_edges)})")
    print(f"- node_degree:   {degree_path} (rows={len(node_degree)})")
    print(f"- unresolved:    {unresolved_path} (rows={len(unresolved)})")
    print(f"- report:        {report_path}")
    print(f"- repost 映射率: {repost_stats['mapped_ratio']:.4f} ({int(repost_stats['mapped_rows'])}/{int(repost_stats['rows_dedup_by_mid'])})")


if __name__ == "__main__":
    main()
