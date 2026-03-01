#!/usr/bin/env python3
"""
给转发网络节点打用户类型标签，并计算媒体来源平衡度 B_i 分布。

定义（按用户 i 的出边邻居）：
  B_i = 1 - |(n_m,i - n_w,i) / (n_m,i + n_w,i)|

其中：
- n_m,i: i 连接到的主流媒体数量（mainstream + government，按项目既有官方叙事口径）
- n_w,i: i 连接到的自媒体数量（wemedia）

输出：
- node_attributes.csv: 节点属性（raw 与三分类）
- bi_values.csv: 每个 source_user 的 n_m, n_w, B_i 与 exposure_group
- bi_report.json: 统计报告
- fig_bi_distribution.png: B_i 分布直方图
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
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
    p = argparse.ArgumentParser(description="节点类型标注 + B_i 分布计算")
    p.add_argument("--network-dir", default="outputs/network/repost_user_network", help="网络目录（含 edge_weighted.csv）")
    p.add_argument("--edge-file", default="edge_weighted.csv", help="边文件名（默认加权边）")
    p.add_argument("--topic-dir", default="dataset/Topic_data", help="Topic 数据目录（用于 user_name->verify_typ）")
    p.add_argument("--user-info-csv", default="dataset/User_data/USER_INFO_1.csv", help="用户信息表（可选补充 verify_typ）")
    p.add_argument("--bins", type=int, default=20, help="直方图 bins 数")
    p.add_argument(
        "--connection-mode",
        choices=["out"],
        default="out",
        help="连接定义（当前实现 out：按出边目标账号统计 n_m/n_w）",
    )
    return p.parse_args()


def _norm_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def _mode_verify(df: pd.DataFrame, user_col: str = "user_name", verify_col: str = "verify_typ") -> Tuple[pd.DataFrame, int]:
    if len(df) == 0:
        return pd.DataFrame(columns=[user_col, verify_col]), 0
    c = (
        df.groupby([user_col, verify_col], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values([user_col, "n", verify_col], ascending=[True, False, True])
    )
    out = c.drop_duplicates(subset=[user_col], keep="first")[[user_col, verify_col]].reset_index(drop=True)
    n_conflict = int(df.groupby(user_col)[verify_col].nunique().gt(1).sum())
    return out, n_conflict


def _map_verify_typ_from_numeric(v: object) -> str:
    # 与 scripts/fix_user_meta_csv.py 的口径一致
    try:
        x = int(float(str(v).strip()))
    except Exception:
        x = -1
    if x == 0:
        return "黄V认证"
    if x in (1, 2, 3, 4, 5, 6, 7):
        return "蓝V认证"
    return "无认证"


def _build_verify_lookup(topic_dir: Path, user_info_csv: Path) -> Tuple[Dict[str, str], Dict[str, int]]:
    stats: Dict[str, int] = {}

    # 1) topic 中 user_name + verify_typ
    topic_parts: List[pd.DataFrame] = []
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
        if len(d) > 0:
            topic_parts.append(d)

    if topic_parts:
        topic_all = pd.concat(topic_parts, ignore_index=True)
    else:
        topic_all = pd.DataFrame(columns=["user_name", "verify_typ"])
    topic_mode, topic_conf = _mode_verify(topic_all)
    stats["topic_rows"] = int(len(topic_all))
    stats["topic_unique_users"] = int(topic_all["user_name"].nunique()) if len(topic_all) else 0
    stats["topic_verify_conflict_users"] = int(topic_conf)

    # 2) user_info_1 中 user_name + 认证类型（数值）
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
    u_mode, u_conf = _mode_verify(u[["user_name", "verify_typ"]])
    stats["user_info_rows"] = int(len(u))
    stats["user_info_unique_users"] = int(u["user_name"].nunique()) if len(u) else 0
    stats["user_info_verify_conflict_users"] = int(u_conf)

    # 3) 融合：优先 topic，再用 user_info 补空
    lut = dict(zip(topic_mode["user_name"], topic_mode["verify_typ"]))
    for n, vt in zip(u_mode["user_name"], u_mode["verify_typ"]):
        if n not in lut:
            lut[n] = vt
    stats["verify_lookup_users"] = int(len(lut))
    return lut, stats


def _to_three_class(raw_type: str) -> str:
    # 与既有口径一致：government 视为官方叙事并入 mainstream
    if raw_type in ("mainstream", "government"):
        return "mainstream"
    if raw_type == "wemedia":
        return "wemedia"
    # public + other 统一归入普通用户
    return "ordinary"


def _compute_bi_out(edge: pd.DataFrame, node_attr: pd.DataFrame) -> pd.DataFrame:
    pair = edge[["source_user", "target_user"]].drop_duplicates().copy()
    pair = pair[(pair["source_user"].map(_norm_text) != "") & (pair["target_user"].map(_norm_text) != "")]

    target_type = node_attr[["user_name", "node_type_3class"]].rename(columns={"user_name": "target_user"})
    pair = pair.merge(target_type, on="target_user", how="left")
    pair["node_type_3class"] = pair["node_type_3class"].fillna("ordinary")

    m = pair[pair["node_type_3class"] == "mainstream"].groupby("source_user")["target_user"].nunique()
    w = pair[pair["node_type_3class"] == "wemedia"].groupby("source_user")["target_user"].nunique()
    all_n = pair.groupby("source_user")["target_user"].nunique()

    out = pd.DataFrame({"user_name": all_n.index})
    out["n_targets_total"] = out["user_name"].map(all_n).fillna(0).astype(int)
    out["n_m"] = out["user_name"].map(m).fillna(0).astype(int)
    out["n_w"] = out["user_name"].map(w).fillna(0).astype(int)
    denom = out["n_m"] + out["n_w"]
    out["n_media_targets"] = denom.astype(int)
    out["B_i"] = np.where(denom > 0, 1.0 - np.abs((out["n_m"] - out["n_w"]) / denom), np.nan)
    out["exposure_group"] = np.where(denom > 0, "media_exposed", "no_media_exposure")
    out["has_media_exposure"] = denom > 0
    return out.sort_values("B_i", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    network_dir = Path(args.network_dir)
    edge_path = network_dir / args.edge_file
    topic_dir = Path(args.topic_dir)
    user_info_csv = Path(args.user_info_csv)

    if not edge_path.exists():
        raise FileNotFoundError(f"未找到边文件: {edge_path}")

    edge = pd.read_csv(edge_path, usecols=["source_user", "target_user", "weight"], dtype={"source_user": str, "target_user": str})
    edge["source_user"] = edge["source_user"].map(_norm_text)
    edge["target_user"] = edge["target_user"].map(_norm_text)
    edge = edge[(edge["source_user"] != "") & (edge["target_user"] != "")]

    users = pd.DataFrame(
        {
            "user_name": pd.unique(pd.concat([edge["source_user"], edge["target_user"]], ignore_index=True)),
        }
    )

    verify_lut, lut_stats = _build_verify_lookup(topic_dir, user_info_csv)
    users["verify_typ"] = users["user_name"].map(lambda x: verify_lut.get(x, ""))
    users["verify_typ_known"] = users["verify_typ"].map(lambda s: _norm_text(s) != "")

    mapper = UserTypeMapper()
    users["user_type_raw"] = users.apply(
        lambda r: mapper.map_verify_type(r["verify_typ"], r["user_name"]).user_type,
        axis=1,
    )
    users["node_type_3class"] = users["user_type_raw"].map(_to_three_class)

    bi = _compute_bi_out(edge, users)
    bi = bi.merge(users[["user_name", "node_type_3class", "user_type_raw", "verify_typ", "verify_typ_known"]], on="user_name", how="left")

    valid = bi["B_i"].dropna()
    n_total_sources = int(len(bi))
    n_exposed = int((bi["exposure_group"] == "media_exposed").sum())
    n_no_media = int((bi["exposure_group"] == "no_media_exposure").sum())
    exposure_stats = {
        "n_source_users_total": n_total_sources,
        "n_media_exposed": n_exposed,
        "n_no_media_exposure": n_no_media,
        "pct_media_exposed": float(n_exposed / n_total_sources) if n_total_sources else 0.0,
        "pct_no_media_exposure": float(n_no_media / n_total_sources) if n_total_sources else 0.0,
    }
    if len(valid) > 0:
        stats_bi = {
            "n_users_with_media_links": int(len(valid)),
            "mean": float(valid.mean()),
            "median": float(valid.median()),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "p10": float(valid.quantile(0.10)),
            "p90": float(valid.quantile(0.90)),
            "pct_eq_0": float((valid == 0).mean()),
            "pct_ge_0_5": float((valid >= 0.5).mean()),
            "pct_eq_1": float((valid == 1).mean()),
        }
    else:
        stats_bi = {
            "n_users_with_media_links": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p90": None,
            "pct_eq_0": None,
            "pct_ge_0_5": None,
            "pct_eq_1": None,
        }

    out_node = network_dir / "node_attributes.csv"
    out_bi = network_dir / "bi_values.csv"
    out_report = network_dir / "bi_report.json"
    out_fig = network_dir / "fig_bi_distribution.png"

    users.sort_values("user_name").to_csv(out_node, index=False, encoding="utf-8-sig")
    bi.to_csv(out_bi, index=False, encoding="utf-8-sig")

    plt = _ensure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.0, 2.4]})
    ax0, ax1 = axes
    grp = bi["exposure_group"].value_counts().reindex(["no_media_exposure", "media_exposed"]).fillna(0).astype(int)
    ax0.bar(grp.index.tolist(), grp.values.tolist(), color=["#9CA3AF", "#2A9D8F"])
    ax0.set_title("Exposure Groups")
    ax0.set_ylabel("User Count")
    ax0.tick_params(axis="x", rotation=20)
    ax0.grid(axis="y", alpha=0.2)

    if len(valid) > 0:
        bins = np.linspace(0.0, 1.0, int(max(5, args.bins)) + 1)
        ax1.hist(valid.values, bins=bins, color="#2A9D8F", alpha=0.85, edgecolor="white")
        ax1.axvline(float(valid.mean()), color="#E76F51", ls="--", lw=1.5, label=f"Mean={valid.mean():.3f}")
        ax1.axvline(float(valid.median()), color="#264653", ls="-.", lw=1.5, label=f"Median={valid.median():.3f}")
        ax1.legend(frameon=False)
    ax1.set_title("Distribution of B_i (media_exposed only)")
    ax1.set_xlabel("B_i")
    ax1.set_ylabel("User Count")
    ax1.set_xlim(0, 1)
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    report = {
        "inputs": {
            "network_dir": str(network_dir),
            "edge_file": str(edge_path),
            "topic_dir": str(topic_dir),
            "user_info_csv": str(user_info_csv),
            "connection_mode": args.connection_mode,
        },
        "verify_lookup_stats": lut_stats,
        "node_stats": {
            "n_nodes_in_edge_file": int(len(users)),
            "verify_typ_known_nodes": int(users["verify_typ_known"].sum()),
            "verify_typ_known_ratio": float(users["verify_typ_known"].mean()) if len(users) else 0.0,
            "user_type_raw_counts": {k: int(v) for k, v in users["user_type_raw"].value_counts().to_dict().items()},
            "node_type_3class_counts": {k: int(v) for k, v in users["node_type_3class"].value_counts().to_dict().items()},
        },
        "exposure_group_stats": exposure_stats,
        "bi_stats": stats_bi,
        "outputs": {
            "node_attributes_csv": str(out_node),
            "bi_values_csv": str(out_bi),
            "bi_distribution_fig": str(out_fig),
            "report_json": str(out_report),
        },
        "definition_note": {
            "n_m_definition": "出边连接到的 mainstream/government 目标账号去重数量",
            "n_w_definition": "出边连接到的 wemedia 目标账号去重数量",
            "three_class_mapping": "mainstream<=mainstream+government; wemedia<=wemedia; ordinary<=public+other",
        },
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("B_i 计算完成")
    print(f"- node attributes: {out_node} (rows={len(users)})")
    print(f"- bi values:       {out_bi} (rows={len(bi)})")
    print(f"- figure:          {out_fig}")
    print(f"- report:          {out_report}")
    print(
        f"- exposure groups: media_exposed={exposure_stats['n_media_exposed']}, "
        f"no_media_exposure={exposure_stats['n_no_media_exposure']}"
    )
    print(f"- valid B_i users: {stats_bi['n_users_with_media_links']}")
    if stats_bi["mean"] is not None:
        print(f"- B_i mean/median: {stats_bi['mean']:.4f} / {stats_bi['median']:.4f}")


if __name__ == "__main__":
    main()
