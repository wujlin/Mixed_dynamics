"""
合并原始话题数据与官媒补充数据，输出标准化 CSV 供后续标注或分析使用。

输入：
- --base dataset/Topic_data/#新冠后遗症#_filtered.csv （或其他主数据）
- --official dataset/Topic_data/官媒补充_flat.csv （由 flatten_official_media.py 生成）

输出：
- 合并后的 CSV（默认 dataset/Topic_data/merged_topic_official.csv）

对齐字段：
- mid
- user_name
- verify_typ
- publish_time
- content
- source_file （可选）

去重逻辑：
- 优先按 mid 去重；如 mid 为空，则按 content 截断(200) 去重。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并主数据与官媒补充数据")
    p.add_argument("--base", required=True, help="主数据 CSV 路径")
    p.add_argument("--official", required=True, help="官媒补充 CSV 路径")
    p.add_argument("--output", required=True, help="合并输出 CSV 路径")
    return p.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 标准化列名
    rename_map = {
        "text": "content",
        "original_text": "content",
    }
    df = df.rename(columns=rename_map)
    for col in ["mid", "user_name", "verify_typ", "publish_time", "content"]:
        if col not in df.columns:
            df[col] = ""
    return df[["mid", "user_name", "verify_typ", "publish_time", "content"] + [c for c in df.columns if c not in ["mid", "user_name", "verify_typ", "publish_time", "content"]]]


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    off_path = Path(args.official)
    out_path = Path(args.output)

    df_base = load_csv(base_path)
    df_off = load_csv(off_path)

    df_base["source"] = "base"
    df_off["source"] = "official"

    df_all = pd.concat([df_base, df_off], ignore_index=True)

    # 去重：优先 mid，其次 content 截断
    key_mid = df_all["mid"].fillna("").astype(str)
    key_content = df_all["content"].fillna("").astype(str).str[:200]
    df_all["dup_key"] = key_mid.where(key_mid != "", other=key_content)
    df_all = df_all.drop_duplicates(subset=["dup_key"])
    df_all = df_all.drop(columns=["dup_key"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"合并完成: base={len(df_base)}, official={len(df_off)}, merged={len(df_all)}")
    print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
