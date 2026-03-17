#!/usr/bin/env python3
"""
V2 任务2：关注列表暴露构建管线

子命令：
1) extract-targets：从 Topic 数据提取目标 user_id（优先有标注用户）
2) crawl-following：按 user_id 抓取关注列表（m.weibo.cn / weibo.com ajax）
3) build-exposure：从关注边构建分组，并与转发分组做一致性检验（confusion + kappa）
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_user_mapper():
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
UID_RE = re.compile(r"weibo\.com/(?:u/)?(\d+)", re.IGNORECASE)


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


def map_verify_typ_from_numeric(v: Any) -> str:
    try:
        x = int(float(str(v).strip()))
    except Exception:
        x = -1
    if x == 0:
        return "黄V认证"
    if x in (1, 2, 3, 4, 5, 6, 7):
        return "蓝V认证"
    return "无认证"


def extract_uid(value: Any) -> Optional[str]:
    s = _norm_text(value)
    if not s:
        return None
    if s.isdigit():
        return s
    m = UID_RE.search(s)
    if not m:
        return None
    return m.group(1)


def load_cookie_header(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"未找到 cookie 文件：{path}")
    text = path.read_text(encoding="utf-8")
    looks_like_json = text.lstrip().startswith(("{", "["))
    if path.suffix.lower() in (".json", ".jsonl") or looks_like_json:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            pairs = []
            if isinstance(obj, dict):
                pairs = [(k, str(v)) for k, v in obj.items()]
            elif isinstance(obj, list):
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    name = _norm_text(it.get("name"))
                    value = _norm_text(it.get("value"))
                    if name and value:
                        pairs.append((name, value))
            header = "; ".join([f"{k}={v}" for k, v in pairs])
            if "=" in header:
                return header
    header = text.strip()
    lines = [ln.strip() for ln in header.splitlines() if ln.strip()]
    if len(lines) > 1:
        picked = None
        for ln in lines:
            if ln.lower().startswith("cookie:"):
                picked = ln
                break
        header = picked or lines[0]
    if header.lower().startswith("cookie:"):
        header = header.split(":", 1)[1].strip()
    if "=" not in header:
        raise ValueError(f"cookie 文本格式异常：{path}")
    return header


def extract_cookie_value(cookie_header: str, name: str) -> Optional[str]:
    if not cookie_header:
        return None
    for p in cookie_header.split(";"):
        p = p.strip()
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k.strip() == name:
            return v.strip()
    return None


def http_get_json(url: str, *, headers: Dict[str, str], timeout: int = 20) -> Dict[str, Any]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = getattr(resp.headers, "get_content_charset", lambda: None)() or "utf-8"
        text = raw.decode(charset, errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = " ".join(text.strip().split())[:220]
        raise ValueError(f"Non-JSON response: {snippet}") from exc


def load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_headers_for_endpoint(base_headers: Dict[str, str], endpoint: str, xsrf: Optional[str]) -> Dict[str, str]:
    h = dict(base_headers)
    ep = _norm_text(endpoint)
    if "weibo.com/ajax/" in ep:
        h["Referer"] = "https://weibo.com/"
        if xsrf:
            h.setdefault("x-xsrf-token", xsrf)
    elif "m.weibo.cn" in ep:
        h.setdefault("Referer", "https://m.weibo.cn/")
        if xsrf:
            h.setdefault("X-XSRF-TOKEN", xsrf)
            h.setdefault("XSRF-TOKEN", xsrf)
    return h


def parse_following_users(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    # weibo.com ajax: {"users":[...], ...}
    users_web = obj.get("users")
    if isinstance(users_web, list):
        users: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for u in users_web:
            if not isinstance(u, dict):
                continue
            uid = _norm_text(u.get("id") or u.get("idstr") or u.get("uid"))
            if not uid or uid in seen:
                continue
            seen.add(uid)
            users.append(
                {
                    "uid": uid,
                    "user_name": _norm_text(u.get("screen_name") or u.get("name")),
                    "verified_type": _norm_text(u.get("verified_type")),
                    "verified_reason": _norm_text(u.get("verified_reason")),
                }
            )
        return users

    # m.weibo.cn: {"data":{"cards":[...]}, ...}
    data = obj.get("data")
    if not isinstance(data, dict):
        return []
    cards = data.get("cards")
    if not isinstance(cards, list):
        return []
    users: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for c in cards:
        if not isinstance(c, dict):
            continue
        group = c.get("card_group")
        if isinstance(group, list):
            iter_items = group
        else:
            iter_items = [c]
        for it in iter_items:
            if not isinstance(it, dict):
                continue
            u = it.get("user") if isinstance(it.get("user"), dict) else it
            if not isinstance(u, dict):
                continue
            uid = _norm_text(u.get("id") or u.get("idstr") or u.get("uid"))
            if not uid:
                continue
            if uid in seen:
                continue
            seen.add(uid)
            users.append(
                {
                    "uid": uid,
                    "user_name": _norm_text(u.get("screen_name") or u.get("name")),
                    "verified_type": _norm_text(u.get("verified_type")),
                    "verified_reason": _norm_text(u.get("verified_reason")),
                }
            )
    return users


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 关注列表暴露构建管线")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("extract-targets", help="从 Topic 数据提取目标 user_id")
    p1.add_argument("--topic-csv", default="dataset/Topic_data/merged_topic_official.csv")
    p1.add_argument("--annotations", default="outputs/annotations/master/long_covid_annotations_master.jsonl")
    p1.add_argument("--out-dir", default="outputs/v2_following_exposure")

    p2 = sub.add_parser("crawl-following", help="抓取关注列表")
    p2.add_argument("--targets-csv", default="outputs/v2_following_exposure/following_targets_uid_priority.csv")
    p2.add_argument("--out-dir", default="outputs/v2_following_exposure")
    p2.add_argument("--rules", default="data/config/weibo_following_crawler_rules.json")
    p2.add_argument("--cookies", default="secrets/weibo_cookie_header.txt")
    p2.add_argument("--endpoint", default="", help="手工指定 endpoint 模板")
    p2.add_argument("--timeout", type=int, default=20)
    p2.add_argument("--sleep-min", type=float, default=0.0)
    p2.add_argument("--sleep-max", type=float, default=0.0)
    p2.add_argument("--max-pages", type=int, default=0)
    p2.add_argument("--max-following-per-user", type=int, default=0)
    p2.add_argument("--limit-users", type=int, default=0)
    p2.add_argument("--print-every", type=int, default=20)
    p2.add_argument("--blocked-threshold", type=int, default=5)
    p2.add_argument("--retry-failed", action="store_true")

    p3 = sub.add_parser("build-exposure", help="构建关注暴露分组并与转发分组对比")
    p3.add_argument("--targets-csv", default="outputs/v2_following_exposure/following_targets_uid_priority.csv")
    p3.add_argument("--edges-csv", default="outputs/v2_following_exposure/following_edges.csv")
    p3.add_argument("--interaction-exposure-csv", default="outputs/v2_analysis_fullwindow_v2/user_exposure.csv")
    p3.add_argument("--out-dir", default="outputs/v2_following_exposure")

    return p.parse_args()


def cmd_extract_targets(args: argparse.Namespace) -> None:
    topic_csv = (ROOT / args.topic_csv).resolve() if not Path(args.topic_csv).is_absolute() else Path(args.topic_csv)
    ann_path = (ROOT / args.annotations).resolve() if not Path(args.annotations).is_absolute() else Path(args.annotations)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topic = pd.read_csv(topic_csv, usecols=["mid", "user_name", "user_link", "weibo_link"], dtype=str, low_memory=False)
    topic = topic.rename(columns={c: c.lstrip("\ufeff") for c in topic.columns})
    topic["mid"] = topic["mid"].map(_norm_mid)
    topic["user_name"] = topic["user_name"].map(_norm_text)
    topic["uid"] = topic["user_link"].map(extract_uid)
    miss = topic["uid"].isna()
    topic.loc[miss, "uid"] = topic.loc[miss, "weibo_link"].map(extract_uid)
    topic = topic[topic["uid"].notna() & (topic["uid"].astype(str) != "") & (topic["user_name"] != "")].copy()
    topic["uid"] = topic["uid"].astype(str)

    ann_mids: Set[str] = set()
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except json.JSONDecodeError:
                continue
            mid = _norm_mid(o.get("mid"))
            if mid:
                ann_mids.add(mid)

    topic["is_annotated_post"] = topic["mid"].isin(ann_mids)

    pair = (
        topic.groupby(["uid", "user_name"], as_index=False)
        .agg(n_posts=("mid", "size"), n_annotated_posts=("is_annotated_post", "sum"))
        .sort_values(["uid", "n_posts", "n_annotated_posts", "user_name"], ascending=[True, False, False, True])
    )
    pair["has_annotation"] = pair["n_annotated_posts"] > 0
    pair.to_csv(out_dir / "following_targets_uid_user_map.csv", index=False, encoding="utf-8-sig")

    uid_summary = (
        pair.sort_values(["uid", "n_posts", "n_annotated_posts", "user_name"], ascending=[True, False, False, True])
        .drop_duplicates(subset=["uid"], keep="first")
        .rename(columns={"uid": "user_id"})
    )
    uid_summary = uid_summary[["user_id", "user_name", "n_posts", "n_annotated_posts", "has_annotation"]]
    uid_summary = uid_summary.sort_values(["has_annotation", "n_annotated_posts", "n_posts", "user_id"], ascending=[False, False, False, True])
    uid_summary.to_csv(out_dir / "following_targets_uid_priority.csv", index=False, encoding="utf-8-sig")

    report = {
        "inputs": {"topic_csv": str(topic_csv), "annotations": str(ann_path)},
        "counts": {
            "topic_rows_with_uid": int(len(topic)),
            "unique_uid": int(uid_summary["user_id"].nunique()),
            "uid_with_annotation": int(uid_summary["has_annotation"].sum()),
            "uid_without_annotation": int((~uid_summary["has_annotation"]).sum()),
        },
        "outputs": {
            "uid_user_map_csv": str(out_dir / "following_targets_uid_user_map.csv"),
            "uid_priority_csv": str(out_dir / "following_targets_uid_priority.csv"),
        },
    }
    (out_dir / "following_targets_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("extract-targets 完成")
    print(f"- unique uid: {uid_summary['user_id'].nunique()}")
    print(f"- with annotation: {int(uid_summary['has_annotation'].sum())}")
    print(f"- output: {out_dir / 'following_targets_uid_priority.csv'}")


def _detect_endpoint(
    candidates: Sequence[str],
    sample_uids: Sequence[str],
    base_headers: Dict[str, str],
    xsrf: Optional[str],
    timeout: int,
) -> str:
    if not candidates:
        raise ValueError("未提供 endpoint 候选")
    best = candidates[0]
    best_score = -1
    for ep in candidates:
        score = 0
        tested = 0
        headers = build_headers_for_endpoint(base_headers, ep, xsrf=xsrf)
        for uid in sample_uids:
            url = ep.format(uid=uid, page=1)
            try:
                obj = http_get_json(url, headers=headers, timeout=timeout)
            except Exception:
                continue
            tested += 1
            if int(obj.get("ok") or 0) == 1:
                score += len(parse_following_users(obj))
        if tested > 0 and score > best_score:
            best_score = score
            best = ep
    return best


def _load_done_from_log(path: Path, retry_failed: bool) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = _norm_text(row.get("source_uid"))
            status = _norm_text(row.get("status"))
            if not uid:
                continue
            if retry_failed:
                if status in {"ok", "empty"}:
                    done.add(uid)
            else:
                done.add(uid)
    return done


def cmd_crawl_following(args: argparse.Namespace) -> None:
    targets_csv = (ROOT / args.targets_csv).resolve() if not Path(args.targets_csv).is_absolute() else Path(args.targets_csv)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    rules_path = (ROOT / args.rules).resolve() if not Path(args.rules).is_absolute() else Path(args.rules)
    cookies_path = (ROOT / args.cookies).resolve() if not Path(args.cookies).is_absolute() else Path(args.cookies)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = load_rules(rules_path)
    base_headers = dict(rules.get("headers", {}) or {})
    base_headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    )
    cookie_header = load_cookie_header(cookies_path)
    base_headers["Cookie"] = cookie_header
    xsrf = extract_cookie_value(cookie_header, "XSRF-TOKEN")

    limits = rules.get("limits", {}) or {}
    sleep_range = rules.get("sleep_range", [3, 5]) or [3, 5]
    sleep_min = float(args.sleep_min) if args.sleep_min > 0 else float(sleep_range[0])
    sleep_max = float(args.sleep_max) if args.sleep_max > 0 else float(sleep_range[1])
    max_pages = int(args.max_pages) if args.max_pages > 0 else int(limits.get("max_pages", 20))
    max_following_per_user = (
        int(args.max_following_per_user) if args.max_following_per_user > 0 else int(limits.get("max_following_per_user", 200))
    )
    timeout = int(args.timeout)

    endpoint: str
    if args.endpoint:
        endpoint = args.endpoint
    else:
        eps = rules.get("endpoints", {}) or {}
        candidates = eps.get("following_candidates") or []
        if not candidates:
            if eps.get("following"):
                candidates = [eps.get("following")]
            else:
                candidates = ["https://m.weibo.cn/api/container/getIndex?containerid=231093_-_selffollowed_-_{uid}&page={page}"]
        t = pd.read_csv(targets_csv, usecols=["user_id"], dtype=str, low_memory=False)
        sample_uids = [u for u in t["user_id"].astype(str).tolist() if u][: min(8, len(t))]
        endpoint = _detect_endpoint(candidates, sample_uids, base_headers=base_headers, xsrf=xsrf, timeout=timeout)
    headers = build_headers_for_endpoint(base_headers, endpoint, xsrf=xsrf)

    targets = pd.read_csv(targets_csv, usecols=["user_id"], dtype=str, low_memory=False)
    uids = [u for u in targets["user_id"].astype(str).tolist() if _norm_text(u)]
    if args.limit_users and int(args.limit_users) > 0:
        uids = uids[: int(args.limit_users)]

    edge_path = out_dir / "following_edges.csv"
    log_path = out_dir / "following_crawl_log.csv"
    write_edge_header = not edge_path.exists()
    write_log_header = not log_path.exists()

    done = _load_done_from_log(log_path, retry_failed=bool(args.retry_failed))
    todo = [u for u in uids if u not in done]

    mapper = UserTypeMapper()
    edge_fields = [
        "source_uid",
        "page",
        "target_uid",
        "target_user_name",
        "verified_type",
        "verified_reason",
        "verify_typ",
        "target_user_type",
        "source",
    ]
    log_fields = [
        "source_uid",
        "status",
        "n_pages",
        "n_edges",
        "endpoint",
        "error",
    ]

    ok = 0
    fail = 0
    blocked_streak = 0
    with edge_path.open("a", encoding="utf-8-sig", newline="") as f_edge, log_path.open("a", encoding="utf-8-sig", newline="") as f_log:
        w_edge = csv.DictWriter(f_edge, fieldnames=edge_fields)
        w_log = csv.DictWriter(f_log, fieldnames=log_fields)
        if write_edge_header:
            w_edge.writeheader()
        if write_log_header:
            w_log.writeheader()

        for i, uid in enumerate(todo, start=1):
            seen_targets: Set[str] = set()
            n_pages = 0
            err = ""
            for page in range(1, max_pages + 1):
                n_pages = page
                url = endpoint.format(uid=uid, page=page)
                try:
                    obj = http_get_json(url, headers=headers, timeout=timeout)
                    if isinstance(obj, dict):
                        if "ok" in obj:
                            ok_code = int(obj.get("ok") or 0)
                        else:
                            ok_code = 1 if isinstance(obj.get("users"), list) else 0
                    else:
                        ok_code = 0
                    if ok_code != 1:
                        msg = _norm_text(obj.get("msg") if isinstance(obj, dict) else "")
                        redirect = _norm_text(obj.get("url") if isinstance(obj, dict) else "")
                        err = f"api_not_ok: ok={ok_code} msg={msg} url={redirect}"
                        break
                    users = parse_following_users(obj)
                    if not users:
                        break
                    for u in users:
                        tgt = _norm_text(u.get("uid"))
                        if not tgt or tgt in seen_targets:
                            continue
                        seen_targets.add(tgt)
                        vt_num = u.get("verified_type")
                        vt = map_verify_typ_from_numeric(vt_num)
                        ut = mapper.map_verify_type(vt, _norm_text(u.get("user_name"))).user_type
                        w_edge.writerow(
                            {
                                "source_uid": uid,
                                "page": page,
                                "target_uid": tgt,
                                "target_user_name": _norm_text(u.get("user_name")),
                                "verified_type": _norm_text(vt_num),
                                "verified_reason": _norm_text(u.get("verified_reason")),
                                "verify_typ": vt,
                                "target_user_type": ut,
                                "source": "weibo.com" if "weibo.com/ajax/" in endpoint else "m.weibo.cn",
                            }
                        )
                    f_edge.flush()
                    if len(seen_targets) >= max_following_per_user:
                        break
                except HTTPError as e:
                    err = f"HTTPError({getattr(e, 'code', 'na')}): {e}"
                    if getattr(e, "code", None) in (403, 418, 429, 432):
                        blocked_streak += 1
                    break
                except URLError as e:
                    err = f"URLError: {e}"
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    break

            status = "ok" if len(seen_targets) > 0 and not err else ("empty" if len(seen_targets) == 0 and not err else "fail")
            w_log.writerow(
                {
                    "source_uid": uid,
                    "status": status,
                    "n_pages": n_pages,
                    "n_edges": len(seen_targets),
                    "endpoint": endpoint,
                    "error": err,
                }
            )
            f_log.flush()
            if status == "ok":
                ok += 1
                blocked_streak = 0
            elif status == "fail":
                fail += 1

            if i % int(args.print_every) == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} ok={ok} fail={fail}", file=sys.stderr)

            if blocked_streak >= int(args.blocked_threshold):
                print(f"[abort] 连续触发封禁错误达到阈值：{blocked_streak}", file=sys.stderr)
                break

            if i < len(todo) and sleep_max >= sleep_min and sleep_max > 0:
                time.sleep(random.uniform(sleep_min, sleep_max))

    summary = {
        "inputs": {
            "targets_csv": str(targets_csv),
            "rules": str(rules_path),
            "cookies": str(cookies_path),
            "endpoint": endpoint,
        },
        "counts": {
            "uids_total": int(len(uids)),
            "uids_todo": int(len(todo)),
            "ok": int(ok),
            "fail": int(fail),
        },
        "outputs": {
            "edges_csv": str(edge_path),
            "crawl_log_csv": str(log_path),
        },
    }
    (out_dir / "following_crawl_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("crawl-following 完成")
    print(f"- todo: {len(todo)} ok: {ok} fail: {fail}")
    print(f"- edges: {edge_path}")


def _follow_group(n_m: int, n_w: int) -> str:
    if n_m > 0 and n_w > 0:
        return "dual"
    if n_m > 0 and n_w == 0:
        return "mainstream_only"
    if n_m == 0 and n_w > 0:
        return "wemedia_only"
    return "no_media"


def _cohen_kappa(y1: Sequence[str], y2: Sequence[str], labels: Sequence[str]) -> float:
    if len(y1) == 0 or len(y2) == 0 or len(y1) != len(y2):
        return float("nan")
    n = len(y1)
    idx = {k: i for i, k in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(y1, y2):
        if a not in idx or b not in idx:
            continue
        mat[idx[a], idx[b]] += 1
    po = float(np.trace(mat) / n)
    row = mat.sum(axis=1) / n
    col = mat.sum(axis=0) / n
    pe = float(np.sum(row * col))
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def cmd_build_exposure(args: argparse.Namespace) -> None:
    targets_csv = (ROOT / args.targets_csv).resolve() if not Path(args.targets_csv).is_absolute() else Path(args.targets_csv)
    edges_csv = (ROOT / args.edges_csv).resolve() if not Path(args.edges_csv).is_absolute() else Path(args.edges_csv)
    interaction_csv = (
        (ROOT / args.interaction_exposure_csv).resolve()
        if not Path(args.interaction_exposure_csv).is_absolute()
        else Path(args.interaction_exposure_csv)
    )
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(targets_csv, dtype=str, low_memory=False)
    if "user_id" not in targets.columns:
        raise ValueError(f"targets-csv 缺少 user_id 列：{targets_csv}")
    targets["user_id"] = targets["user_id"].map(_norm_text)
    targets["user_name"] = targets.get("user_name", "").map(_norm_text) if "user_name" in targets.columns else ""
    targets = targets[targets["user_id"] != ""].copy()

    if edges_csv.exists():
        edges = pd.read_csv(edges_csv, dtype=str, low_memory=False)
    else:
        edges = pd.DataFrame(columns=["source_uid", "target_uid", "target_user_type"])
    for c in ["source_uid", "target_uid", "target_user_type"]:
        if c not in edges.columns:
            edges[c] = ""
    edges["source_uid"] = edges["source_uid"].map(_norm_text)
    edges["target_uid"] = edges["target_uid"].map(_norm_text)
    edges["target_user_type"] = edges["target_user_type"].map(_norm_text)
    edges = edges[(edges["source_uid"] != "") & (edges["target_uid"] != "")].drop_duplicates(subset=["source_uid", "target_uid"], keep="first")

    rows = []
    all_source_uids = sorted(set(targets["user_id"].tolist()))
    for uid in all_source_uids:
        g = edges[edges["source_uid"] == uid]
        n_total = int(g["target_uid"].nunique())
        n_m = int(g[g["target_user_type"].isin(["mainstream", "government"])]["target_uid"].nunique())
        n_w = int(g[g["target_user_type"] == "wemedia"]["target_uid"].nunique())
        rows.append(
            {
                "user_id": uid,
                "n_follow_total": n_total,
                "n_follow_mainstream": n_m,
                "n_follow_wemedia": n_w,
                "follow_group": _follow_group(n_m, n_w),
            }
        )
    by_uid = pd.DataFrame(rows)
    by_uid.to_csv(out_dir / "following_exposure_by_uid.csv", index=False, encoding="utf-8-sig")

    # 合并用户名（一个 uid 取 priority 表中的 user_name）
    map_uid_name = targets[["user_id", "user_name"]].drop_duplicates(subset=["user_id"], keep="first")
    by_user = by_uid.merge(map_uid_name, on="user_id", how="left")
    by_user.to_csv(out_dir / "following_exposure_by_user.csv", index=False, encoding="utf-8-sig")

    summary = {
        "inputs": {
            "targets_csv": str(targets_csv),
            "edges_csv": str(edges_csv),
            "interaction_exposure_csv": str(interaction_csv),
        },
        "counts": {
            "n_target_uids": int(len(all_source_uids)),
            "n_edges_unique": int(len(edges)),
            "follow_group_counts": {k: int(v) for k, v in by_uid["follow_group"].value_counts().to_dict().items()},
        },
        "outputs": {
            "following_exposure_by_uid_csv": str(out_dir / "following_exposure_by_uid.csv"),
            "following_exposure_by_user_csv": str(out_dir / "following_exposure_by_user.csv"),
        },
    }

    if interaction_csv.exists():
        inter = pd.read_csv(interaction_csv, usecols=["user_name", "exposure_group"], dtype=str, low_memory=False)
        inter["user_name"] = inter["user_name"].map(_norm_text)
        inter["interaction_group"] = inter["exposure_group"].map(_norm_text)
        inter = inter[inter["user_name"] != ""]
        merged = by_user.merge(inter[["user_name", "interaction_group"]], on="user_name", how="inner")
        labels = ["no_media", "mainstream_only", "wemedia_only", "dual"]
        merged = merged[merged["follow_group"].isin(labels) & merged["interaction_group"].isin(labels)].copy()
        conf = (
            merged.groupby(["interaction_group", "follow_group"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .pivot_table(index="interaction_group", columns="follow_group", values="n", fill_value=0)
            .reindex(index=labels, columns=labels, fill_value=0)
            .reset_index()
        )
        conf.to_csv(out_dir / "group_confusion_matrix.csv", index=False, encoding="utf-8-sig")
        kappa = _cohen_kappa(
            merged["interaction_group"].tolist(),
            merged["follow_group"].tolist(),
            labels=labels,
        )
        kappa_obj = {
            "n_overlap_users": int(len(merged)),
            "labels": labels,
            "cohen_kappa": float(kappa) if kappa == kappa else np.nan,
        }
        (out_dir / "group_kappa.json").write_text(json.dumps(kappa_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["agreement"] = kappa_obj
        summary["outputs"]["confusion_matrix_csv"] = str(out_dir / "group_confusion_matrix.csv")
        summary["outputs"]["kappa_json"] = str(out_dir / "group_kappa.json")

    (out_dir / "following_exposure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("build-exposure 完成")
    print(f"- n_target_uids: {len(all_source_uids)}")
    if "agreement" in summary:
        print(f"- overlap users: {summary['agreement']['n_overlap_users']}")
        print(f"- kappa: {summary['agreement']['cohen_kappa']}")


def main() -> None:
    args = parse_args()
    if args.cmd == "extract-targets":
        cmd_extract_targets(args)
    elif args.cmd == "crawl-following":
        cmd_crawl_following(args)
    elif args.cmd == "build-exposure":
        cmd_build_exposure(args)
    else:
        raise ValueError(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()
