"""
从合并后的数据集中筛出尚未标注的样本，生成待标注 CSV。

逻辑：
- 输入：合并后的 CSV（需含 mid/content），已有标注 jsonl。
- 键：优先使用 mid；mid 为空则用 content[:200]。
- 输出：待标注样本 CSV（保留 mid/user_name/verify_typ/publish_time/content）。

用法示例：
python scripts/extract_new_samples.py \
    --merged dataset/Topic_data/merged_topic_official.csv \
    --annotated outputs/annotations/master/long_covid_annotations_master.jsonl \
    --output outputs/annotations/intermediate/to_annotate.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="筛出未标注样本")
    p.add_argument("--merged", required=True, help="合并后的数据 CSV")
    p.add_argument("--annotated", required=True, help="已有标注 jsonl")
    p.add_argument("--output", required=True, help="待标注输出 CSV")
    return p.parse_args()


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def make_key(series: pd.Series) -> pd.Series:
    mid = series.get("mid", pd.Series([], dtype=str)).fillna("").astype(str)
    content = series.get("content", pd.Series([], dtype=str)).fillna("").astype(str).str[:200]
    return mid.where(mid != "", content)


def main() -> None:
    args = parse_args()
    merged = pd.read_csv(args.merged)
    annotated_records = load_jsonl(Path(args.annotated))
    annotated = pd.DataFrame(annotated_records)

    merged_key = make_key(merged)
    ann_key = make_key(annotated)
    ann_keys = set(ann_key)

    merged["key"] = merged_key
    todo = merged[~merged["key"].isin(ann_keys)].copy()

    out_cols = ["mid", "user_name", "verify_typ", "publish_time", "content"]
    for col in out_cols:
        if col not in todo.columns:
            todo[col] = ""

    todo[out_cols].to_csv(args.output, index=False)
    print(f"待标注样本: {len(todo)}，已保存到 {args.output}")


if __name__ == "__main__":
    main()
