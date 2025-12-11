"""
将 dataset/Topic_data/新增官媒数据/*.json 转换为标准 CSV，便于与主话题数据合并。

输入 JSON 结构：
- 顶层是 dict，键为账号名，值为包含 user_info / long_texts 的 dict。
- user_info: 列表，字段可能包含 created_at, mblogid, user_name, verified, verified_type, text_raw。
- long_texts: 列表，字段 id, long_text（通常无时间戳，默认不输出）。

输出字段：
- mid: mblogid
- user_name
- verify_typ: 根据 verified_type 粗略映射 (3/2 -> 蓝V, 1 -> 黄V, 0/False -> 无认证)
- publish_time: 由 created_at 解析
- content: text_raw
- source_file: 来源文件名

用法：
    python scripts/flatten_official_media.py \
        --input-dir dataset/Topic_data/新增官媒数据 \
        --output dataset/Topic_data/官媒补充_flat.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flatten official media JSON to CSV")
    p.add_argument("--input-dir", default="dataset/Topic_data/新增官媒数据", help="输入 JSON 目录")
    p.add_argument("--output", default="dataset/Topic_data/官媒补充_flat.csv", help="输出 CSV 路径")
    return p.parse_args()


def map_verify(verified_type: Any) -> str:
    """
    粗略映射 verified_type 为微博认证标签。
    3/2 -> 蓝V，1 -> 黄V，0/False/None -> 无认证
    """
    try:
        vt = int(verified_type)
    except Exception:
        vt = -1
    if vt in (2, 3):
        return "蓝V认证"
    if vt == 1:
        return "黄V认证"
    return "无认证"


def parse_created_at(value: str) -> str:
    """
    将微博风格时间 'Tue Apr 13 09:41:03 +0800 2021' 解析为 '%Y-%m-%d %H:%M:%S' 字符串。
    解析失败返回空字符串。
    """
    if not value:
        return ""
    try:
        dt = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    rows: List[Dict[str, Any]] = []

    for json_file in input_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for account, payload in data.items():
            user_infos = payload.get("user_info", [])
            if not isinstance(user_infos, list):
                continue
            for item in user_infos:
                content = item.get("text_raw", "") or item.get("text", "")
                publish_time = parse_created_at(item.get("created_at", ""))
                row = {
                    "mid": item.get("mblogid", ""),
                    "user_name": item.get("user_name", account),
                    "verify_typ": map_verify(item.get("verified_type", None)),
                    "publish_time": publish_time,
                    "content": content,
                    "source_file": json_file.name,
                }
                # 跳过完全空行
                if not row["content"] and not row["publish_time"]:
                    continue
                rows.append(row)

    if not rows:
        print("No records parsed.")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["mid", "content"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"保存 {len(df)} 行到 {output_path}")


if __name__ == "__main__":
    main()
