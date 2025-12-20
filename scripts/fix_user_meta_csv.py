#!/usr/bin/env python3
"""
修正 user_meta CSV 的认证/用户类型口径（离线，不需要联网）。

背景：m.weibo.cn 返回的 `verified_type` 采用数值编码。早期抓取脚本将部分编码
（尤其是 verified_type==0 的“个人认证”）错误映射为“无认证”，导致大量自媒体/大V
被归为 public，从而压低 n_wemedia 与 r_proxy。

本脚本在不重新爬取的前提下，基于 `verified_type + user_name + official_list` 重新生成：
- verify_typ（蓝V认证/黄V认证/无认证）
- user_type（mainstream/wemedia/government/public/other）

用法示例：
  python3 scripts/fix_user_meta_csv.py \
    --input data/derived/user_meta_batch4.csv \
    --output data/derived/user_meta_batch4_fixed.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _load_user_mapper():
    """
    直接从文件加载 `src/empirical/user_mapper.py`，避免触发 `src/__init__.py` 里对 numpy 的依赖导入。
    """
    import importlib.util

    module_path = ROOT / "src/empirical/user_mapper.py"
    spec = importlib.util.spec_from_file_location("user_mapper", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 user_mapper：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_mapper"] = module
    spec.loader.exec_module(module)
    return module.UserTypeMapper


UserTypeMapper = _load_user_mapper()


def map_verify_typ(verified_type: Any) -> str:
    """
    与 `scripts/fetch_user_meta_weibo.py` 保持一致的最小口径：
    - -1：无认证
    - 0 ：黄V认证（个人认证，自媒体/大V）
    - 1..7：蓝V认证（机构/政府/媒体/校园等）
    - 其它：无认证
    """
    try:
        vt = int(str(verified_type).strip())
    except Exception:
        vt = -1
    if vt == 0:
        return "黄V认证"
    if vt in (1, 2, 3, 4, 5, 6, 7):
        return "蓝V认证"
    return "无认证"


def load_official_uid_list(path: Path) -> Set[str]:
    """
    官媒/政府白名单（按 uid 精确匹配）。
    支持行格式：uid 或 uid,name；忽略空行与注释行。
    """
    uid_set: Set[str] = set()
    if not path.exists():
        return uid_set
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            left, _ = [x.strip() for x in s.split(",", 1)]
            if left.isdigit():
                uid_set.add(left)
            continue
        if s.isdigit():
            uid_set.add(s)
    return uid_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix user_meta CSV mapping (offline)")
    p.add_argument("--input", default="data/derived/user_meta_batch4.csv", help="输入 user_meta CSV")
    p.add_argument("--output", default="data/derived/user_meta_batch4_fixed.csv", help="输出修正后的 CSV")
    p.add_argument("--official-list", default="data/config/official_media_list.txt", help="官媒/政府白名单（uid）")
    p.add_argument("--drop-error-rows", action="store_true", help="丢弃 error 行（默认保留）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = (ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out_path = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    official_path = (ROOT / args.official_list).resolve() if not Path(args.official_list).is_absolute() else Path(args.official_list)

    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{in_path}")

    official_uid = load_official_uid_list(official_path)
    mapper = UserTypeMapper()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    user_type_before = Counter()
    user_type_after = Counter()
    vt_before = Counter()
    vt_after = Counter()
    changed = 0

    with in_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"输入 CSV 缺少表头：{in_path}")
        # 兼容旧表头：确保关键列存在
        for c in ["uid", "user_name", "verified_type", "verified_reason", "verify_typ", "user_type", "source", "error"]:
            if c not in fieldnames:
                fieldnames.append(c)

        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats["rows_in"] += 1
                uid = (row.get("uid") or "").strip()
                err = (row.get("error") or "").strip()
                vt_raw = (row.get("verified_type") or "").strip()
                vt_before[vt_raw] += 1
                before_ut = (row.get("user_type") or "").strip()
                user_type_before[before_ut] += 1

                # 保留错误行（默认），避免丢失抓取失败信息
                if err:
                    stats["rows_error"] += 1
                    if args.drop_error_rows:
                        continue
                    writer.writerow(row)
                    continue

                # 重新计算 verify_typ 与 user_type
                new_verify_typ = map_verify_typ(vt_raw)
                row["verify_typ"] = new_verify_typ

                if uid and uid in official_uid:
                    row["user_type"] = "mainstream"
                    row["source"] = (row.get("source") or "m.weibo.cn") + "|official_list_uid"
                else:
                    user_name = (row.get("user_name") or "").strip()
                    res = mapper.map_verify_type(new_verify_typ, user_name)
                    row["user_type"] = res.user_type
                after_ut = (row.get("user_type") or "").strip()

                stats["rows_ok"] += 1
                vt_after[(row.get("verified_type") or "").strip()] += 1
                user_type_after[after_ut] += 1
                if before_ut != after_ut:
                    changed += 1

                writer.writerow(row)

    print(f"[done] in={in_path} out={out_path}")
    print(f"[stats] rows_in={stats['rows_in']} ok={stats['rows_ok']} error={stats['rows_error']}")
    print(f"[stats] changed_user_type={changed}")
    print(f"[before] user_type: {dict(user_type_before)}")
    print(f"[after]  user_type: {dict(user_type_after)}")
    # 重点输出 vt==0 的影响
    print(f"[info] verified_type==0 count: {vt_before.get('0',0)} (这些将被视为黄V认证 -> wemedia)")


if __name__ == "__main__":
    main()
