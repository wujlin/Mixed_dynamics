#!/usr/bin/env python3
"""
Batch4（上海疫情）数据清洗与候选池生成

目标：从 `dataset/Shanghai_data/上海疫情all.csv`（70万+）中筛出“更可能与疫情/封控相关、且适合标注”的样本，
用于后续 LLM 标注与经验验证（Note07）。

设计原则（KISS）：
- 不删除原始数据，只生成派生文件（CSV + JSON 报告）。
- 先做“硬过滤”（时间、文本有效性），再做“相关性过滤”（关键词口径可切换）。
- 以可复现的参数化 CLI 为准，避免 notebook 中硬编码。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # 正常情况下项目环境已安装 numpy/pandas，且 `src` 包可直接导入。
    from src.empirical.text_preprocessor import is_valid_for_annotation, preprocess_weibo_text  # noqa: E402
except Exception:
    # 兼容：在未激活项目 conda 环境时，`import src` 会触发 `src/__init__.py` 的 numpy 依赖。
    # 这里用“按文件路径加载模块”的方式，避免把清洗脚本与科学计算依赖强绑定。
    import importlib.util

    module_path = ROOT / "src/empirical/text_preprocessor.py"
    spec = importlib.util.spec_from_file_location("text_preprocessor", module_path)
    if spec is None or spec.loader is None:
        raise
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    is_valid_for_annotation = module.is_valid_for_annotation
    preprocess_weibo_text = module.preprocess_weibo_text


TIME_FMT = "%Y-%m-%d %H:%M"


STRONG_KEYWORDS = [
    "核酸",
    "阳性",
    "封控",
    "封城",
    "解封",
    "隔离",
    "方舱",
    "抗原",
    "确诊",
    "无症状",
    "病例",
    "静默",
    "团购",
    "物资",
    "抢菜",
    "保供",
    "通行证",
    "居委",
]

WEAK_KEYWORDS = [
    "疫情",
    "新冠",
    "小区",
    "居家",
    "外卖",
    "快递",
    "封闭",
]


def _parse_time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, TIME_FMT)
    except Exception:
        return None


def _norm_mid(s: str) -> str:
    s = (s or "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\u200b", "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_hit(text: str, *, mode: str) -> Tuple[bool, List[str]]:
    """
    mode:
      - loose: 任何 strong/weak 命中即可
      - strict: strong 命中，或满足 (疫情/新冠) + (小区/居家/外卖/快递/封闭) 组合
    """
    hits = [kw for kw in (STRONG_KEYWORDS + WEAK_KEYWORDS) if kw in text]
    if mode == "loose":
        return (len(hits) > 0), hits

    strong_hits = [kw for kw in STRONG_KEYWORDS if kw in text]
    if strong_hits:
        return True, strong_hits

    # 组合：疫情/新冠 +（生活/封闭类词）
    has_a = ("疫情" in text) or ("新冠" in text)
    has_b = any(k in text for k in ["小区", "居家", "外卖", "快递", "封闭"])
    if has_a and has_b:
        combo_hits = [kw for kw in WEAK_KEYWORDS if kw in text]
        return True, combo_hits

    return False, hits


@dataclass
class CleanReport:
    input_path: str
    output_path: str
    report_path: str
    time_start: str
    time_end: str
    mode: str
    min_length: int
    total_rows: int
    kept_rows: int
    excluded_outside_time: int
    excluded_invalid_text: int
    excluded_no_keyword: int
    kept_by_month: Dict[str, int]
    top_keywords: List[Tuple[str, int]]
    content_hash_dups: int


def _month_key(t: datetime) -> str:
    return f"{t.year:04d}-{t.month:02d}"


def iter_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main() -> None:
    ap = argparse.ArgumentParser(description="清洗上海疫情数据，生成 batch4 待标注候选池")
    ap.add_argument("--input", default="dataset/Shanghai_data/上海疫情all.csv", help="输入 CSV 路径")
    ap.add_argument(
        "--output",
        default="outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_strict.csv",
        help="输出候选 CSV 路径",
    )
    ap.add_argument(
        "--report",
        default="outputs/annotations/intermediate/batch4_shanghai_clean_report.json",
        help="输出报告 JSON 路径",
    )
    ap.add_argument("--time-start", default="2022-01-01", help="起始日期（含），YYYY-MM-DD")
    ap.add_argument("--time-end", default="2022-06-30", help="结束日期（含），YYYY-MM-DD")
    ap.add_argument("--mode", choices=["strict", "loose"], default="strict", help="相关性过滤口径")
    ap.add_argument("--min-length", type=int, default=8, help="文本有效性最短长度（core content）")
    ap.add_argument("--max-rows", type=int, default=0, help="仅用于调试：最多处理多少行（0=不限制）")
    args = ap.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output
    report_path = ROOT / args.report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # date bounds
    start_dt = datetime.strptime(args.time_start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.time_end, "%Y-%m-%d")
    end_dt = end_dt.replace(hour=23, minute=59, second=59)

    total = 0
    kept = 0
    excluded_outside = 0
    excluded_invalid = 0
    excluded_no_kw = 0

    kept_by_month: Dict[str, int] = defaultdict(int)
    kw_counter: Counter[str] = Counter()

    # 轻量重复检测（用于报告；正常情况下 mid 唯一，但 content hash 可能重复）
    seen_hash = set()
    hash_dups = 0

    fieldnames = [
        "mid",
        "publish_time",
        "user_name",
        "user_link",
        "weibo_link",
        "content",
        "content_hash",
        "keywords",
        "origin_file_name",
        "location",
        "lat",
        "lng",
        "forward_num",
        "comment_num",
        "like_num",
        "phone_type",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in iter_rows(input_path):
            total += 1
            if args.max_rows and total > int(args.max_rows):
                break

            t = _parse_time(row.get("publish_time", ""))
            if t is None or t < start_dt or t > end_dt:
                excluded_outside += 1
                continue

            raw = row.get("content", "") or ""
            cleaned = _strip_html(raw)
            cleaned = preprocess_weibo_text(cleaned, max_length=1000, keep_hashtags=True)
            if not is_valid_for_annotation(cleaned, min_length=int(args.min_length)):
                excluded_invalid += 1
                continue

            ok, hits = _keyword_hit(cleaned, mode=args.mode)
            if not ok:
                excluded_no_kw += 1
                continue

            content_hash = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
            if content_hash in seen_hash:
                hash_dups += 1
            else:
                seen_hash.add(content_hash)

            for kw in hits:
                kw_counter[kw] += 1

            kept += 1
            kept_by_month[_month_key(t)] += 1

            writer.writerow(
                {
                    "mid": _norm_mid(row.get("mid", "")),
                    "publish_time": t.strftime(TIME_FMT),
                    "user_name": (row.get("user_name", "") or "").strip(),
                    "user_link": (row.get("user_link", "") or "").strip(),
                    "weibo_link": (row.get("weibo_link", "") or "").strip(),
                    "content": cleaned,
                    "content_hash": content_hash,
                    "keywords": "|".join(hits),
                    "origin_file_name": (row.get("origin_file_name", "") or "").strip(),
                    "location": (row.get("location", "") or "").strip(),
                    "lat": (row.get("lat", "") or "").strip(),
                    "lng": (row.get("lng", "") or "").strip(),
                    "forward_num": (row.get("forward_num", "") or "").strip(),
                    "comment_num": (row.get("comment_num", "") or "").strip(),
                    "like_num": (row.get("like_num", "") or "").strip(),
                    "phone_type": (row.get("phone_type", "") or "").strip(),
                }
            )

    report = CleanReport(
        input_path=str(input_path),
        output_path=str(output_path),
        report_path=str(report_path),
        time_start=args.time_start,
        time_end=args.time_end,
        mode=args.mode,
        min_length=int(args.min_length),
        total_rows=total,
        kept_rows=kept,
        excluded_outside_time=excluded_outside,
        excluded_invalid_text=excluded_invalid,
        excluded_no_keyword=excluded_no_kw,
        kept_by_month=dict(sorted(kept_by_month.items())),
        top_keywords=kw_counter.most_common(30),
        content_hash_dups=hash_dups,
    )

    report_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] kept_rows:", kept, "/", total)
    print("[done] output:", output_path)
    print("[done] report:", report_path)


if __name__ == "__main__":
    main()
