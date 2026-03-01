#!/usr/bin/env python3
"""
按论文口径（master + batch3，mid 对齐标注）重算网络节点类型占比与 B_i。

口径说明：
1) 数据口径与 run_note7_empirical 保持一致：
   - master: dataset/Topic_data/merged_topic_official.csv
     + outputs/annotations/master/long_covid_annotations_master.jsonl
   - batch3: outputs/annotations/intermediate/to_annotate_batch3_clean.csv
     + outputs/annotations/batches/batch_03_expanded/new_batch3.jsonl
2) 用户类型规则复用 src/empirical/user_mapper.py
3) 三类并法：
   - mainstream := mainstream + government
   - wemedia := wemedia
   - ordinary := public + other
4) 网络投影：
   - 基于既有转发网络 edge_weighted.csv
   - 仅保留 source_user 和 target_user 都在论文用户池中的边
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Set

import numpy as np
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按论文口径重算网络节点占比与 B_i")
    p.add_argument("--network-dir", default="outputs/network/repost_user_network", help="现有网络目录（包含 edge_weighted.csv）")
    p.add_argument("--edge-file", default="edge_weighted.csv", help="网络边文件名")
    p.add_argument(
        "--output-dir",
        default="outputs/network/repost_user_network_paper_scope",
        help="输出目录",
    )
    p.add_argument("--bins", type=int, default=20, help="B_i 分布图 bins")
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


def _load_annotation_mid_set(path: Path) -> Set[str]:
    mids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = _norm_mid(obj.get("mid", ""))
            if mid:
                mids.add(mid)
    return mids


def _load_dataset_with_annotation(dataset_csv: Path, ann_jsonl: Path) -> pd.DataFrame:
    d = pd.read_csv(dataset_csv, dtype=str, low_memory=False)
    d = d.rename(columns={c: c.lstrip("\ufeff") for c in d.columns})
    for c in ["mid", "user_name", "verify_typ", "content"]:
        if c not in d.columns:
            d[c] = ""
    d["mid"] = d["mid"].map(_norm_mid)
    d["user_name"] = d["user_name"].map(_norm_text)
    d["verify_typ"] = d["verify_typ"].map(_norm_text)
    d["content"] = d["content"].map(_norm_text)
    d = d[(d["mid"] != "") & (d["content"] != "")]
    d = d.drop_duplicates(subset=["mid"], keep="first").reset_index(drop=True)

    mids = _load_annotation_mid_set(ann_jsonl)
    d = d[d["mid"].isin(mids)].copy().reset_index(drop=True)
    return d[["mid", "user_name", "verify_typ"]]


def _vote_user_type(df_posts: pd.DataFrame) -> pd.DataFrame:
    # 先把帖子级 user_type 算出来，再对同名用户多数投票
    mapper = UserTypeMapper()
    d = df_posts.copy()
    d["user_type_raw"] = d.apply(
        lambda r: mapper.map_verify_type(r["verify_typ"], r["user_name"]).user_type,
        axis=1,
    )
    vote = (
        d.groupby(["user_name", "user_type_raw"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["user_name", "n", "user_type_raw"], ascending=[True, False, True])
    )
    out = vote.drop_duplicates(subset=["user_name"], keep="first")[["user_name", "user_type_raw"]].reset_index(drop=True)
    out["node_type_3class"] = out["user_type_raw"].map(_to_3class)
    return out


def _to_3class(user_type_raw: str) -> str:
    if user_type_raw in ("mainstream", "government"):
        return "mainstream"
    if user_type_raw == "wemedia":
        return "wemedia"
    return "ordinary"


def _save_distribution_5class(df: pd.DataFrame, path: Path, key_col: str) -> None:
    c = df[key_col].value_counts(dropna=False)
    tot = int(c.sum())
    rows = []
    for k, v in c.items():
        rows.append({"class": str(k), "count": int(v), "ratio": float(v / tot) if tot else 0.0})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _save_distribution_3class(df: pd.DataFrame, path: Path, key_col: str) -> None:
    c = df[key_col].value_counts(dropna=False)
    tot = int(c.sum())
    rows = []
    for k, v in c.items():
        rows.append({"class_3": str(k), "count": int(v), "ratio": float(v / tot) if tot else 0.0})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _compute_bi(edge_weighted: pd.DataFrame, node_attr: pd.DataFrame) -> pd.DataFrame:
    pair = edge_weighted[["source_user", "target_user"]].drop_duplicates().copy()
    target = node_attr[["user_name", "node_type_3class"]].rename(columns={"user_name": "target_user"})
    pair = pair.merge(target, on="target_user", how="left")

    m = pair[pair["node_type_3class"] == "mainstream"].groupby("source_user")["target_user"].nunique()
    w = pair[pair["node_type_3class"] == "wemedia"].groupby("source_user")["target_user"].nunique()
    all_t = pair.groupby("source_user")["target_user"].nunique()

    out = pd.DataFrame({"user_name": all_t.index})
    out["n_targets_total"] = out["user_name"].map(all_t).fillna(0).astype(int)
    out["n_m"] = out["user_name"].map(m).fillna(0).astype(int)
    out["n_w"] = out["user_name"].map(w).fillna(0).astype(int)
    denom = out["n_m"] + out["n_w"]
    out["B_i"] = np.where(denom > 0, 1.0 - np.abs((out["n_m"] - out["n_w"]) / denom), np.nan)
    return out.sort_values("B_i", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    network_dir = ROOT / args.network_dir
    edge_path = network_dir / args.edge_file
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not edge_path.exists():
        raise FileNotFoundError(f"未找到网络边文件: {edge_path}")

    # 论文口径源
    master_csv = ROOT / "dataset/Topic_data/merged_topic_official.csv"
    master_ann = ROOT / "outputs/annotations/master/long_covid_annotations_master.jsonl"
    batch3_csv = ROOT / "outputs/annotations/intermediate/to_annotate_batch3_clean.csv"
    batch3_ann = ROOT / "outputs/annotations/batches/batch_03_expanded/new_batch3.jsonl"

    df_master = _load_dataset_with_annotation(master_csv, master_ann)
    df_batch3 = _load_dataset_with_annotation(batch3_csv, batch3_ann)
    df_all = pd.concat([df_master, df_batch3], ignore_index=True).drop_duplicates(subset=["mid"], keep="first").reset_index(drop=True)

    # 用户标签（论文口径）
    user_attr = _vote_user_type(df_all[["user_name", "verify_typ"]])
    user_set = set(user_attr["user_name"].tolist())

    # 保存论文口径分布（帖子级 / 用户级）
    mapper = UserTypeMapper()
    post = df_all.copy()
    post["user_type_raw"] = post.apply(lambda r: mapper.map_verify_type(r["verify_typ"], r["user_name"]).user_type, axis=1)
    post["node_type_3class"] = post["user_type_raw"].map(_to_3class)
    _save_distribution_5class(post, out_dir / "paper_post_distribution_5class.csv", "user_type_raw")
    _save_distribution_3class(post, out_dir / "paper_post_distribution_3class.csv", "node_type_3class")
    _save_distribution_5class(user_attr, out_dir / "paper_user_distribution_5class.csv", "user_type_raw")
    _save_distribution_3class(user_attr, out_dir / "paper_user_distribution_3class.csv", "node_type_3class")

    # 网络投影到论文用户池（两端都在池中）
    edge = pd.read_csv(edge_path, dtype=str, low_memory=False)
    edge["source_user"] = edge["source_user"].map(_norm_text)
    edge["target_user"] = edge["target_user"].map(_norm_text)
    edge = edge[(edge["source_user"] != "") & (edge["target_user"] != "")]
    edge_proj = edge[edge["source_user"].isin(user_set) & edge["target_user"].isin(user_set)].copy().reset_index(drop=True)

    # 投影网络节点属性
    proj_nodes = sorted(set(edge_proj["source_user"]).union(set(edge_proj["target_user"])))
    node_proj = user_attr[user_attr["user_name"].isin(proj_nodes)].copy().reset_index(drop=True)

    # B_i
    bi = _compute_bi(edge_proj, node_proj)
    bi = bi.merge(node_proj, on="user_name", how="left")

    # 输出文件
    edge_proj_path = out_dir / "edge_weighted_paper_scope.csv"
    node_proj_path = out_dir / "node_attributes_paper_scope.csv"
    bi_path = out_dir / "bi_values_paper_scope.csv"
    fig_path = out_dir / "fig_bi_distribution_paper_scope.png"
    report_path = out_dir / "paper_scope_report.json"

    edge_proj.to_csv(edge_proj_path, index=False, encoding="utf-8-sig")
    node_proj.to_csv(node_proj_path, index=False, encoding="utf-8-sig")
    bi.to_csv(bi_path, index=False, encoding="utf-8-sig")

    # 绘图
    plt = _ensure_matplotlib()
    valid = bi["B_i"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(valid) > 0:
        bins = np.linspace(0, 1, int(max(5, args.bins)) + 1)
        ax.hist(valid.values, bins=bins, color="#2A9D8F", alpha=0.85, edgecolor="white")
        ax.axvline(float(valid.mean()), color="#E76F51", ls="--", lw=1.5, label=f"Mean={valid.mean():.3f}")
        ax.axvline(float(valid.median()), color="#264653", ls="-.", lw=1.5, label=f"Median={valid.median():.3f}")
        ax.legend(frameon=False)
    ax.set_title("Distribution of B_i (Paper Scope)")
    ax.set_xlabel("B_i")
    ax.set_ylabel("User Count")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    # 报告
    def _vc(df: pd.DataFrame, c: str) -> Dict[str, int]:
        return {str(k): int(v) for k, v in df[c].value_counts().to_dict().items()}

    bi_stats = {
        "n_users_with_media_links": int(valid.shape[0]),
        "mean": float(valid.mean()) if len(valid) else None,
        "median": float(valid.median()) if len(valid) else None,
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else (0.0 if len(valid) == 1 else None),
        "p10": float(valid.quantile(0.10)) if len(valid) else None,
        "p90": float(valid.quantile(0.90)) if len(valid) else None,
        "pct_eq_0": float((valid == 0).mean()) if len(valid) else None,
        "pct_eq_1": float((valid == 1).mean()) if len(valid) else None,
    }

    report = {
        "paper_scope_inputs": {
            "master_csv": str(master_csv),
            "master_ann": str(master_ann),
            "batch3_csv": str(batch3_csv),
            "batch3_ann": str(batch3_ann),
        },
        "paper_scope_sizes": {
            "master_rows_matched": int(len(df_master)),
            "batch3_rows_matched": int(len(df_batch3)),
            "all_rows_union_matched": int(len(df_all)),
            "paper_user_count": int(len(user_attr)),
        },
        "paper_scope_distributions": {
            "post_5class": _vc(post, "user_type_raw"),
            "post_3class": _vc(post, "node_type_3class"),
            "user_5class": _vc(user_attr, "user_type_raw"),
            "user_3class": _vc(user_attr, "node_type_3class"),
        },
        "network_projection": {
            "full_edge_weighted_rows": int(len(edge)),
            "paper_scope_edge_rows": int(len(edge_proj)),
            "paper_scope_node_rows": int(len(node_proj)),
            "paper_scope_node_3class": _vc(node_proj, "node_type_3class"),
        },
        "bi_stats_paper_scope": bi_stats,
        "outputs": {
            "edge_weighted_paper_scope_csv": str(edge_proj_path),
            "node_attributes_paper_scope_csv": str(node_proj_path),
            "bi_values_paper_scope_csv": str(bi_path),
            "bi_distribution_paper_scope_png": str(fig_path),
            "paper_scope_report_json": str(report_path),
            "paper_post_distribution_5class_csv": str(out_dir / "paper_post_distribution_5class.csv"),
            "paper_post_distribution_3class_csv": str(out_dir / "paper_post_distribution_3class.csv"),
            "paper_user_distribution_5class_csv": str(out_dir / "paper_user_distribution_5class.csv"),
            "paper_user_distribution_3class_csv": str(out_dir / "paper_user_distribution_3class.csv"),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("论文口径重算完成")
    print(f"- matched rows: master={len(df_master)}, batch3={len(df_batch3)}, union={len(df_all)}")
    print(f"- paper users: {len(user_attr)}")
    print(f"- projected edges: {len(edge_proj)}, projected nodes: {len(node_proj)}")
    print(f"- B_i valid users: {bi_stats['n_users_with_media_links']}, mean={bi_stats['mean']}, median={bi_stats['median']}")
    print(f"- report: {report_path}")


if __name__ == "__main__":
    main()
