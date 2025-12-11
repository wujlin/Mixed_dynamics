"""
Phase 5.2 预处理脚本

功能：
- 读取话题数据 csv（含 publish_time、content、verify_typ 等）
- 读取已标注的情绪/风险 jsonl（需包含 mid, emotion_class, risk_class）
- 映射用户类型（mainstream/wemedia/public/other）
- 合并标注，按时间窗口聚合生成时间序列特征
- 输出聚合结果与覆盖率摘要

用法示例：
    python scripts/run_phase5_preprocessing.py \\
        --dataset dataset/Topic_data/#新冠后遗症#_filtered.csv \\
        --annotations outputs/annotations/v3/annotated_intent_rule_v3.jsonl \\
        --freq 1H \\
        --output outputs/annotations/v3/time_series_1h.csv

注意：
- 需要 pandas 依赖。
- 注释文件必须包含 mid 与 publish_time 对应的标注；若无 mid，将无法对齐。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# 确保能导入本地 src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.empirical import load_topic_dataset, aggregate_time_series, UserTypeMapper  # noqa: E402
from src.empirical.time_series import TimeSeriesConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5.2 数据预处理：用户类型映射 + 时间序列聚合")
    parser.add_argument("--dataset", required=True, help="话题数据 csv 路径")
    parser.add_argument(
        "--annotations",
        required=True,
        help="情绪/风险标注 jsonl 路径（需包含 mid, emotion_class, risk_class）",
    )
    parser.add_argument("--freq", default="1H", help="时间窗口频率，默认 1H，可选 4H/1D 等")
    parser.add_argument("--output", required=True, help="聚合结果输出 csv 路径")
    parser.add_argument("--limit", type=int, default=None, help="仅读取前 N 行（调试用）")
    return parser.parse_args()


def load_annotations(path: Path) -> pd.DataFrame:
    """
    读取标注 jsonl。
    优先使用 mid 对齐；若缺失 mid，则尝试使用 content 对齐（使用原文 original_text 或 text）。
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(data)
    if not records:
        raise ValueError(f"标注文件 {path} 为空，无法对齐。")
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    ann_path = Path(args.annotations)
    output_path = Path(args.output)

    mapper = UserTypeMapper()
    # 读取 csv 并映射用户类型
    df_raw = load_topic_dataset(dataset_path, mapper=mapper, limit=args.limit)

    # 读取标注
    df_ann = load_annotations(ann_path)

    # 合并标注（内连接，确保有标注的行）
    merge_keys = None
    if "mid" in df_ann.columns:
        merge_keys = ["mid"]
    elif "original_text" in df_ann.columns:
        df_ann = df_ann.rename(columns={"original_text": "content"})
        merge_keys = ["content"]
    elif "text" in df_ann.columns:
        df_ann = df_ann.rename(columns={"text": "content"})
        merge_keys = ["content"]
    else:
        raise ValueError("标注文件缺少 mid 和 content，无法对齐。")

    # 去重避免一对多
    df_ann = df_ann.drop_duplicates(subset=merge_keys)

    df = df_raw.merge(
        df_ann[merge_keys + ["emotion_class", "risk_class"]],
        on=merge_keys,
        how="inner",
    )

    if df.empty:
        raise ValueError(f"合并后无数据，请检查标注文件与 {merge_keys} 是否匹配。")

    # 覆盖率统计
    coverage = len(df) / max(len(df_raw), 1)
    user_type_cov = df["user_type"].notna().mean()
    print(f"合并后样本数: {len(df)} / 原始 {len(df_raw)} (覆盖率 {coverage:.2%})")
    print(f"用户类型覆盖率: {user_type_cov:.2%}")

    # 聚合
    ts_cfg = TimeSeriesConfig(freq=args.freq)
    df_ts = aggregate_time_series(df, config=ts_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ts.to_csv(output_path, index=False)
    print(f"时间序列已保存到: {output_path}")


if __name__ == "__main__":
    main()
