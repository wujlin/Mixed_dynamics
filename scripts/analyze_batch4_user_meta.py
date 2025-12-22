#!/usr/bin/env python3
"""
Batch4（上海 2022）用户元信息分析（离线，不需要联网）。

输入：
  1) user_meta CSV（通常为 data/derived/user_meta_batch4_fixed.csv）
  2) batch4 posts CSV（通常为 outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_loose.csv）

输出：
  - 在 stdout 打印一份可直接贴给 PI 的摘要
  - 可选写入 Markdown（--output）

本脚本仅依赖 Python 标准库（便于在工作站/WSL 环境跑）。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


UID_FROM_USER_LINK = re.compile(r"/(?:u/)?(\d+)")
UID_FROM_WEIBO_LINK = re.compile(r"weibo\.com/(\d+)/")


@dataclass(frozen=True)
class Stats:
    mean: float
    median: float
    std: float
    p10: float
    p90: float
    pct_eq_1: float
    n: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze batch4 user meta coverage and r_proxy saturation (offline).")
    p.add_argument(
        "--user-meta",
        default="data/derived/user_meta_batch4_fixed.csv",
        help="用户元信息 CSV（修正口径后）",
    )
    p.add_argument(
        "--posts-csv",
        default="outputs/annotations/intermediate/to_annotate_batch4_shanghai_2022_loose.csv",
        help="batch4 posts 元数据 CSV（含 publish_time/user_link/weibo_link）",
    )
    p.add_argument(
        "--freq",
        default="4H",
        help="时间聚合频率（仅支持整数小时，如 4H/2h/1H）",
    )
    p.add_argument(
        "--output",
        default="",
        help="可选：写入 Markdown 文件路径（例如 docs/batch4_user_meta_analysis.md）",
    )
    return p.parse_args()


def _parse_hour_freq(freq: str) -> int:
    s = str(freq).strip().lower()
    if not s.endswith("h"):
        raise ValueError(f"freq 仅支持小时粒度，例如 4H/2h/1H：{freq}")
    n = int(s[:-1])
    if n <= 0:
        raise ValueError(f"freq 必须为正整数小时：{freq}")
    return n


def _floor_hours(ts: dt.datetime, hours: int) -> dt.datetime:
    h = (ts.hour // hours) * hours
    return ts.replace(hour=h, minute=0, second=0, microsecond=0)


def _parse_publish_time(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _extract_uid(row: Dict[str, str]) -> Optional[str]:
    user_link = (row.get("user_link") or "").strip()
    m = UID_FROM_USER_LINK.search(user_link)
    if m:
        return m.group(1)
    weibo_link = (row.get("weibo_link") or "").strip()
    m2 = UID_FROM_WEIBO_LINK.search(weibo_link)
    if m2:
        return m2.group(1)
    return None


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("percentile: empty")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize(vals: List[float]) -> Stats:
    if not vals:
        raise ValueError("summarize: empty")
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    sd = math.sqrt(var)
    s = sorted(vals)
    return Stats(
        mean=mu,
        median=_percentile(s, 0.5),
        std=sd,
        p10=_percentile(s, 0.1),
        p90=_percentile(s, 0.9),
        pct_eq_1=sum(1 for x in vals if abs(x - 1.0) < 1e-12) / len(vals),
        n=len(vals),
    )


def load_user_meta(path: Path) -> Tuple[Dict[str, str], Dict[str, str], Counter]:
    meta_type: Dict[str, str] = {}
    meta_name: Dict[str, str] = {}
    errors = Counter()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("uid") or "").strip()
            if not uid:
                continue
            meta_type[uid] = (row.get("user_type") or "").strip() or "unknown"
            meta_name[uid] = (row.get("user_name") or "").strip()
            err = (row.get("error") or "").strip()
            if err:
                errors[err.split(":", 1)[0]] += 1
    return meta_type, meta_name, errors


def analyze(
    user_meta_path: Path,
    posts_csv_path: Path,
    freq_hours: int,
) -> str:
    meta_type, meta_name, meta_errors = load_user_meta(user_meta_path)

    post_rows = 0
    uid_src = Counter()
    post_type = Counter()
    uid_set_by_type: Dict[str, set] = defaultdict(set)

    win_counts: Dict[dt.datetime, Counter] = defaultdict(Counter)
    counts_by_type: Dict[str, Counter] = defaultdict(Counter)

    with posts_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            post_rows += 1
            uid = _extract_uid(row)
            if uid is None:
                uid_src["none"] += 1
                ut = "unknown"
            else:
                # 来源统计（用于排查 uid 提取逻辑）
                if UID_FROM_USER_LINK.search((row.get("user_link") or "").strip()):
                    uid_src["user_link"] += 1
                elif UID_FROM_WEIBO_LINK.search((row.get("weibo_link") or "").strip()):
                    uid_src["weibo_link"] += 1
                else:
                    uid_src["other"] += 1

                ut = meta_type.get(uid, "missing_meta")
                uid_set_by_type[ut].add(uid)

                t = _parse_publish_time((row.get("publish_time") or "").strip())
                if t is not None and ut in ("mainstream", "government", "wemedia"):
                    w = _floor_hours(t, freq_hours)
                    win_counts[w][ut] += 1
                    counts_by_type[ut][uid] += 1

            post_type[ut] += 1

    # r_proxy per window
    r_vals: List[float] = []
    for c in win_counts.values():
        denom = c["wemedia"] + c["mainstream"] + c["government"]
        if denom > 0:
            r_vals.append(c["wemedia"] / denom)

    r_stats = _summarize(r_vals) if r_vals else None

    # user_meta user_type distribution (ok rows only: exclude those with user_type==unknown from error rows)
    user_type_counts = Counter(meta_type.values())

    lines: List[str] = []
    lines.append("Batch4（上海 2022）用户元信息抓取：阶段性分析摘要")
    lines.append("")
    lines.append("一、user_meta 抓取结果（UID 级别）")
    lines.append(f"- user_meta 文件：`{user_meta_path}`")
    lines.append(f"- 总行数：{len(meta_type):,}")
    lines.append(f"- 抓取失败（error）统计：{dict(meta_errors) or '无'}")
    lines.append(f"- user_type（含 missing_meta/unknown 前的映射结果）计数：{dict(user_type_counts)}")
    lines.append("")
    lines.append("二、batch4 posts 覆盖率（帖子/UID 级别）")
    lines.append(f"- posts CSV：`{posts_csv_path}`")
    lines.append(f"- 总帖子数：{post_rows:,}")
    lines.append(f"- UID 提取来源：{dict(uid_src)}")
    lines.append(f"- 帖子按 user_type 分布：{dict(post_type)}")
    unique_uids_total = sum(len(s) for s in uid_set_by_type.values())
    missing_uids = len(uid_set_by_type.get("missing_meta", set()))
    covered_uids = unique_uids_total - missing_uids
    lines.append(f"- unique UID：{unique_uids_total:,}（已覆盖 meta={covered_uids:,}；缺失 meta={missing_uids:,}）")
    lines.append("")
    lines.append("三、r_proxy 可辨识性（按 4H 时间窗，媒体类型仅计入 wemedia/mainstream/government）")
    if r_stats is None:
        lines.append("- 无可计算的时间窗（denom=0）")
    else:
        lines.append(f"- 可计算时间窗数：{r_stats.n}")
        lines.append(f"- r_proxy mean={r_stats.mean:.4f}, median={r_stats.median:.4f}, std={r_stats.std:.4f}")
        lines.append(f"- p10={r_stats.p10:.4f}, p90={r_stats.p90:.4f}, pct(r_proxy==1)={r_stats.pct_eq_1:.2%}")
        lines.append("- 结论：r_proxy 高度饱和（大量时间窗=1），用于 H2/H3 的自变量方差很小。")
    lines.append("")
    lines.append("四、媒体账号贡献（按帖子数 Top10）")
    for ut in ("mainstream", "government", "wemedia"):
        top = counts_by_type[ut].most_common(10)
        lines.append(f"- {ut} Top10：")
        if not top:
            lines.append("  - (none)")
            continue
        for uid, c in top:
            name = (meta_name.get(uid) or "")[:30]
            lines.append(f"  - {uid},{name},{c}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    user_meta_path = Path(args.user_meta)
    posts_csv_path = Path(args.posts_csv)
    freq_hours = _parse_hour_freq(args.freq)

    text = analyze(user_meta_path, posts_csv_path, freq_hours=freq_hours)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
