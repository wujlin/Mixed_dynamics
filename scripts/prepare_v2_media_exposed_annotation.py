#!/usr/bin/env python3
"""
为 V2 实验准备 media-exposed 用户的补充标注队列。

输入：
- outputs/v2_analysis/user_exposure.csv
- dataset/Topic_data/*.csv
- 一个或多个已标注 jsonl（默认 master）

输出：
- 待标注 CSV（可直接给 scripts/run_new_annotation.py）
- 统计报告 JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备 V2 media-exposed 补充标注样本")
    p.add_argument("--exposure-csv", default="outputs/v2_analysis/user_exposure.csv", help="用户暴露表路径")
    p.add_argument("--topic-dir", default="dataset/Topic_data", help="Topic 数据目录")
    p.add_argument(
        "--annotations",
        nargs="+",
        default=["outputs/annotations/master/long_covid_annotations_master.jsonl"],
        help="一个或多个已标注 jsonl（按顺序读取）",
    )
    p.add_argument("--output-csv", default="outputs/annotations/intermediate/to_annotate_v2_media_exposed.csv", help="待标注输出 CSV")
    p.add_argument(
        "--report-json",
        default="outputs/annotations/intermediate/to_annotate_v2_media_exposed_report.json",
        help="统计报告输出 JSON",
    )
    p.add_argument("--keep-duplicates", action="store_true", help="保留同 mid 的重复记录（默认去重）")
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


def _read_topic_csv(path: Path, usecols: List[str]) -> pd.DataFrame:
    try:
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, lineterminator="\n", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip")


def _load_annotated_mids(paths: List[Path]) -> Set[str]:
    mids: Set[str] = set()
    for p in paths:
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
                m = _norm_mid(o.get("mid", ""))
                if m:
                    mids.add(m)
    return mids


def _load_media_exposed_users(exposure_csv: Path) -> Set[str]:
    d = pd.read_csv(exposure_csv, usecols=["user_name", "exposure_group"], dtype=str, low_memory=False)
    d["user_name"] = d["user_name"].map(_norm_text)
    keep = d["exposure_group"].isin(["mainstream_only", "wemedia_only", "dual"])
    return set(d.loc[keep, "user_name"].tolist())


def _load_topic_subset(topic_dir: Path, media_users: Set[str]) -> pd.DataFrame:
    parts = []
    for p in sorted(topic_dir.glob("*.csv")):
        try:
            h = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        norm_to_raw: Dict[str, str] = {}
        for c in h.columns:
            norm_to_raw[c.lstrip("\ufeff")] = c
        colset = set(norm_to_raw.keys())

        mid_norm = "mid" if "mid" in colset else ("id" if "id" in colset else ("微博id" if "微博id" in colset else ""))
        user_norm = "user_name" if "user_name" in colset else ("用户名称" if "用户名称" in colset else "")
        if not mid_norm or not user_norm:
            continue
        verify_norm = "verify_typ" if "verify_typ" in colset else ("用户认证" if "用户认证" in colset else "")
        time_norm = "publish_time" if "publish_time" in colset else ("发布时间" if "发布时间" in colset else "")
        content_norm = "content" if "content" in colset else ("text" if "text" in colset else ("微博正文" if "微博正文" in colset else ""))

        mid_col = norm_to_raw[mid_norm]
        user_col = norm_to_raw[user_norm]
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
        d = d[(d["mid"] != "") & (d["user_name"].isin(media_users)) & (d["content"] != "")].copy()
        if len(d) == 0:
            continue
        d["source_file"] = p.name
        parts.append(d[["mid", "user_name", "verify_typ", "publish_time", "content", "source_file"]])

    if not parts:
        return pd.DataFrame(columns=["mid", "user_name", "verify_typ", "publish_time", "content", "source_file"])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    exposure_csv = ROOT / args.exposure_csv
    topic_dir = ROOT / args.topic_dir
    ann_paths = [ROOT / p for p in args.annotations]
    output_csv = ROOT / args.output_csv
    report_json = ROOT / args.report_json

    media_users = _load_media_exposed_users(exposure_csv)
    rows = _load_topic_subset(topic_dir, media_users)
    annotated_mids = _load_annotated_mids(ann_paths)

    total_rows = int(len(rows))
    unique_mids_before = int(rows["mid"].nunique()) if total_rows else 0
    users_in_rows = int(rows["user_name"].nunique()) if total_rows else 0

    if not args.keep_duplicates and total_rows:
        rows["quality"] = (
            rows["verify_typ"].ne("").astype(int)
            + rows["content"].ne("").astype(int)
            + rows["publish_time"].notna().astype(int)
        )
        rows = rows.sort_values(["mid", "quality", "publish_time"], ascending=[True, False, True], na_position="last")
        rows = rows.drop_duplicates(subset=["mid"], keep="first").drop(columns=["quality"]).reset_index(drop=True)

    rows["is_annotated"] = rows["mid"].isin(annotated_mids)
    to_ann = rows[~rows["is_annotated"]].copy()
    to_ann = to_ann.sort_values(["publish_time", "mid"], na_position="last")
    to_ann["publish_time"] = to_ann["publish_time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    to_ann[["mid", "user_name", "verify_typ", "publish_time", "content"]].to_csv(output_csv, index=False, encoding="utf-8-sig")

    report = {
        "inputs": {
            "exposure_csv": str(exposure_csv),
            "topic_dir": str(topic_dir),
            "annotations": [str(p) for p in ann_paths],
            "keep_duplicates": bool(args.keep_duplicates),
        },
        "counts": {
            "media_exposed_users": int(len(media_users)),
            "topic_rows_after_user_filter": total_rows,
            "topic_unique_mids_after_user_filter": unique_mids_before,
            "topic_unique_users_after_user_filter": users_in_rows,
            "rows_after_dedup": int(len(rows)),
            "unique_mids_after_dedup": int(rows["mid"].nunique()) if len(rows) else 0,
            "annotated_rows_in_pool": int(rows["is_annotated"].sum()) if len(rows) else 0,
            "to_annotate_rows": int(len(to_ann)),
            "to_annotate_unique_mids": int(to_ann["mid"].nunique()) if len(to_ann) else 0,
            "to_annotate_unique_users": int(to_ann["user_name"].nunique()) if len(to_ann) else 0,
        },
        "outputs": {
            "to_annotate_csv": str(output_csv),
            "report_json": str(report_json),
        },
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("准备完成")
    print(f"- media_exposed users: {len(media_users)}")
    print(f"- rows after user filter: {total_rows}")
    print(f"- rows to annotate: {len(to_ann)}")
    print(f"- output csv: {output_csv}")
    print(f"- report: {report_json}")


if __name__ == "__main__":
    main()
