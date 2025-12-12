"""
合并原有标注与新增标注，并输出合并后的 jsonl。

用法示例：
python scripts/merge_new_annotations.py \
    --base outputs/annotations/master/long_covid_annotations_master.jsonl \
    --new outputs/annotations/batches/batch_xx/new_batch.jsonl \
    --output outputs/annotations/master/long_covid_annotations_master_new.jsonl

去重逻辑：
- 优先使用 mid 去重；若缺少 mid，则使用 content（或 original_text/text）去重。
- 基本字段保留：mid（如有）、content（原文/清洗后）、emotion_class、risk_class 及其他原字段。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并原有标注与新增标注")
    p.add_argument("--base", required=True, help="原始标注 jsonl 路径")
    p.add_argument("--new", required=True, help="新增标注 jsonl 路径")
    p.add_argument("--output", required=True, help="合并输出路径")
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


def make_key(item: Dict[str, Any]) -> str:
    if item.get("mid"):
        return f"mid::{item['mid']}"
    # 回退用内容匹配
    content = item.get("original_text") or item.get("text") or item.get("content") or ""
    return f"content::{content[:200].strip()}"


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    new_path = Path(args.new)
    out_path = Path(args.output)

    base_records = load_jsonl(base_path)
    new_records = load_jsonl(new_path)

    merged = {}
    for rec in base_records + new_records:
        key = make_key(rec)
        merged[key] = rec  # 新记录覆盖同 key

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for rec in merged.values():
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"合并完成: base={len(base_records)}, new={len(new_records)}, merged={len(merged)}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
