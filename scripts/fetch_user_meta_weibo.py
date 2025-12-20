#!/usr/bin/env python3
"""
批量抓取微博用户元信息（m.weibo.cn），用于补齐 batch4 等数据源缺失的 verify_typ/user_type。

设计目标（KISS）：
- 只产出 uid -> (verify_typ/user_type) 的可复现元信息表，便于 H2/H3 使用 r_proxy。
- 默认串行 + 随机 sleep（m.weibo.cn 频控严格），支持断点续跑（增量抓取）。
- Cookie 只从本地文件读取，不入库。

输出 CSV 列：
  uid,user_name,verified_type,verified_reason,verify_typ,user_type,source,error
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_user_mapper():
    try:
        # 正常情况下项目环境已安装 numpy/pandas，可直接 import src.*
        from src.empirical.user_mapper import UserTypeMapper  # type: ignore

        return UserTypeMapper
    except Exception:
        # 兼容：在未安装 numpy/pandas 的环境下，避免触发 src/__init__.py 的依赖导入
        import importlib.util

        module_path = ROOT / "src/empirical/user_mapper.py"
        spec = importlib.util.spec_from_file_location("user_mapper", module_path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_mapper"] = module
        spec.loader.exec_module(module)
        return module.UserTypeMapper


UserTypeMapper = _load_user_mapper()


_UID_RE = re.compile(r"weibo\.com/(?:u/)?(\d+)", re.IGNORECASE)


def extract_uid(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s == "nan":
        return None
    if s.isdigit():
        return s
    m = _UID_RE.search(s)
    if not m:
        return None
    return m.group(1)


def load_uids_from_csv(path: Path, *, uid_cols: Tuple[str, ...] = ("uid", "weibo_link", "user_link")) -> Set[str]:
    uids: Set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = None
            for c in uid_cols:
                if c in row and row.get(c):
                    uid = extract_uid(row.get(c))
                    if uid:
                        break
            if uid:
                uids.add(uid)
    return uids


def load_existing_uids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return set()
        if "uid" not in reader.fieldnames:
            return set()
        for row in reader:
            uid = (row.get("uid") or "").strip()
            if uid:
                done.add(uid)
    return done


def load_rules(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_cookie_header(path: Path) -> str:
    """
    支持：
    - JSON：[{name,value}, ...] 或 {name:value, ...}
    - TXT：整行 Cookie header（可直接粘贴浏览器复制内容）
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到 cookie 文件：{path}")

    text = path.read_text(encoding="utf-8")
    # 兼容：即使扩展名不是 .json，只要内容像 JSON 也尝试解析（用户常把 cookies 列表保存为 .txt）
    looks_like_json = text.lstrip().startswith(("{", "["))
    if path.suffix.lower() in (".json", ".jsonl") or looks_like_json:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if obj is not None:
            if isinstance(obj, dict):
                pairs = [(k, str(v)) for k, v in obj.items()]
            elif isinstance(obj, list):
                pairs = []
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    name = str(it.get("name") or "").strip()
                    value = str(it.get("value") or "").strip()
                    if name and value:
                        pairs.append((name, value))
            else:
                pairs = []
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
        raise ValueError(f"cookie 文本看起来不像 Cookie header：{path}（需要形如 'k=v; k2=v2'）")
    return header


def extract_cookie_value(cookie_header: str, name: str) -> Optional[str]:
    if not cookie_header or not name:
        return None
    parts = cookie_header.split(";")
    for p in parts:
        p = p.strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        if k.strip() == name:
            return v.strip()
    return None


def map_verify_typ(verified_type: Any) -> str:
    """
    将 m.weibo.cn 返回的 verified_type（数值）映射为项目使用的粗粒度认证类型。

    目标是服务于“媒体生态 r_proxy”的可复现口径（KISS），而非还原微博的全部认证体系。

    当前最小口径（可随研究需要扩展）：
    - verified_type == -1：无认证
    - verified_type == 0 ：个人认证（自媒体/大V）→ 视为黄V认证
    - verified_type in [1..7]：机构/政府/媒体/校园等 → 视为蓝V认证
    - 其它：暂按无认证处理
    """
    try:
        vt = int(verified_type)
    except Exception:
        vt = -1
    if vt == 0:
        return "黄V认证"
    if vt in (1, 2, 3, 4, 5, 6, 7):
        return "蓝V认证"
    return "无认证"


def http_get_json(url: str, *, headers: Dict[str, str], timeout: int = 20) -> Dict[str, Any]:
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = getattr(resp.headers, "get_content_charset", lambda: None)() or "utf-8"
        text = raw.decode(charset, errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = " ".join(text.strip().split())
        snippet = snippet[:200]
        raise ValueError(f"Non-JSON response: {snippet}") from exc


def extract_user_info(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = obj.get("data")
    if isinstance(data, dict):
        ui = data.get("userInfo")
        if isinstance(ui, dict):
            return ui
        ui = data.get("user")
        if isinstance(ui, dict):
            return ui

        cards = data.get("cards")
        if isinstance(cards, list):
            for c in cards:
                if not isinstance(c, dict):
                    continue
                mblog = c.get("mblog")
                if isinstance(mblog, dict):
                    u = mblog.get("user")
                    if isinstance(u, dict):
                        return u
                group = c.get("card_group")
                if isinstance(group, list):
                    for g in group:
                        if not isinstance(g, dict):
                            continue
                        u = g.get("user")
                        if isinstance(u, dict):
                            return u
    return None


def load_official_list(path: Path) -> Tuple[Set[str], Set[str]]:
    """
    返回 (uid_set, name_set)
    - 支持行格式：uid 或 uid,name 或 name
    """
    uid_set: Set[str] = set()
    name_set: Set[str] = set()
    if not path.exists():
        return uid_set, name_set
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "," in s:
            left, right = [x.strip() for x in s.split(",", 1)]
            if left.isdigit():
                uid_set.add(left)
            if right:
                name_set.add(right)
            continue
        if s.isdigit():
            uid_set.add(s)
        else:
            name_set.add(s)
    return uid_set, name_set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Weibo user meta via m.weibo.cn")
    p.add_argument("--input-csv", required=True, help="输入 CSV（包含 weibo_link/user_link 或 uid 列）")
    p.add_argument("--output", default="data/derived/user_meta_batch4.csv", help="输出 user_meta CSV")
    p.add_argument("--rules", default="data/config/weibo_crawler_rules.json", help="抓取规则配置 JSON")
    p.add_argument("--cookies", default="secrets/weibo_cookies.json", help="本地 cookie 文件（不入库）")
    p.add_argument("--official-list", default="data/config/official_media_list.txt", help="官媒/政府白名单（可选）")
    p.add_argument("--sleep-min", type=float, default=0.0, help="覆盖配置：最小 sleep 秒数（0=使用 rules）")
    p.add_argument("--sleep-max", type=float, default=0.0, help="覆盖配置：最大 sleep 秒数（0=使用 rules）")
    p.add_argument("--limit", type=int, default=0, help="调试：最多抓取多少个 uid（0=不限制）")
    p.add_argument("--timeout", type=int, default=20, help="HTTP 超时（秒）")
    p.add_argument("--print-every", type=int, default=50, help="进度打印间隔（条）")
    p.add_argument("--blocked-threshold", type=int, default=5, help="连续 403/418/429 达到阈值则中止")
    p.add_argument(
        "--abort-on-login",
        action="store_true",
        help="遇到 ok=-100（需要登录）立即中止，避免写入大量失败记录",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = (ROOT / args.input_csv).resolve() if not Path(args.input_csv).is_absolute() else Path(args.input_csv)
    output_path = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    rules_path = (ROOT / args.rules).resolve() if not Path(args.rules).is_absolute() else Path(args.rules)
    cookies_path = (ROOT / args.cookies).resolve() if not Path(args.cookies).is_absolute() else Path(args.cookies)
    official_list_path = (ROOT / args.official_list).resolve() if not Path(args.official_list).is_absolute() else Path(args.official_list)

    rules = load_rules(rules_path)
    endpoint = (
        rules.get("endpoints", {}).get("profile")
        or "https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}"
    )
    headers = dict(rules.get("headers", {}) or {})
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    )
    cookie_header = load_cookie_header(cookies_path)
    if cookie_header:
        headers["Cookie"] = cookie_header
        xsrf = extract_cookie_value(cookie_header, "XSRF-TOKEN")
        if xsrf:
            headers.setdefault("X-XSRF-TOKEN", xsrf)
            headers.setdefault("XSRF-TOKEN", xsrf)

    sleep_range = rules.get("sleep_range") or [3, 8]
    if args.sleep_min > 0 and args.sleep_max > 0 and args.sleep_max >= args.sleep_min:
        sleep_range = [args.sleep_min, args.sleep_max]

    uids = sorted(load_uids_from_csv(input_csv))
    done = load_existing_uids(output_path)
    todo = [u for u in uids if u not in done]
    if args.limit and int(args.limit) > 0:
        todo = todo[: int(args.limit)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    official_uid, _ = load_official_list(official_list_path)
    mapper = UserTypeMapper()

    ok = 0
    fail = 0
    blocked_streak = 0
    warned_login = False
    fieldnames = [
        "uid",
        "user_name",
        "verified_type",
        "verified_reason",
        "verify_typ",
        "user_type",
        "source",
        "error",
    ]

    with output_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for i, uid in enumerate(todo, start=1):
            url = endpoint.format(uid=uid)
            row = {
                "uid": uid,
                "user_name": "",
                "verified_type": "",
                "verified_reason": "",
                "verify_typ": "",
                "user_type": "",
                "source": "m.weibo.cn",
                "error": "",
            }
            try:
                obj = http_get_json(url, headers=headers, timeout=int(args.timeout))
                if not isinstance(obj, dict) or int(obj.get("ok") or 0) != 1:
                    msg = ""
                    if isinstance(obj, dict):
                        msg = str(obj.get("msg") or obj.get("errmsg") or obj.get("error") or "").strip()
                    ok_code = obj.get("ok") if isinstance(obj, dict) else "na"
                    login_url = obj.get("url") if isinstance(obj, dict) else None
                    if ok_code == -100 and login_url:
                        row["error"] = f"need_login: ok=-100 url={login_url}"
                        if not warned_login:
                            warned_login = True
                            print(
                                "[hint] m.weibo.cn 返回 ok=-100（需要登录态）。"
                                "你当前 cookie 很可能来自 weibo.com（PC 站），不足以直接访问 m.weibo.cn 的 XHR 接口。\n"
                                "请按以下方式获取 m.weibo.cn 的 Cookie header，并保存为纯文本文件（推荐：secrets/weibo_cookie_header.txt）：\n"
                                "  1) 浏览器隐身模式打开 https://m.weibo.cn 并完成登录\n"
                                "  2) F12 -> Network -> XHR -> 找到 api/container/getIndex 请求\n"
                                "  3) 复制 Request Headers 里的 Cookie 整行（k=v; k2=v2...）\n"
                                "  4) 运行本脚本时添加：--cookies secrets/weibo_cookie_header.txt\n",
                                file=sys.stderr,
                            )
                        if args.abort_on_login:
                            fail += 1
                            w.writerow(row)
                            f.flush()
                            print("[abort] ok=-100，已按 --abort-on-login 中止。", file=sys.stderr)
                            break
                    else:
                        row["error"] = f"api_not_ok: ok={ok_code} msg={msg}"
                    blocked_streak = 0
                else:
                    ui = extract_user_info(obj) or {}
                    row["user_name"] = str(ui.get("screen_name") or ui.get("name") or "").strip()
                    vt = ui.get("verified_type")
                    row["verified_type"] = "" if vt is None else str(vt)
                    row["verified_reason"] = str(ui.get("verified_reason") or "").strip()
                    row["verify_typ"] = map_verify_typ(vt)

                    if uid in official_uid:
                        row["user_type"] = "mainstream"
                        row["source"] = "official_list_uid"
                    else:
                        res = mapper.map_verify_type(row["verify_typ"], row["user_name"])
                        row["user_type"] = res.user_type
                    blocked_streak = 0
            except HTTPError as exc:
                code = getattr(exc, "code", None)
                row["error"] = f"HTTPError({code}): {exc}"
                if code in (403, 418, 429, 432):
                    blocked_streak += 1
                else:
                    blocked_streak = 0
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                blocked_streak = 0

            if row["error"]:
                fail += 1
            else:
                ok += 1
            w.writerow(row)
            f.flush()

            if i % int(args.print_every) == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} ok={ok} fail={fail}", file=sys.stderr)

            if blocked_streak >= int(args.blocked_threshold):
                print(
                    f"[abort] 连续触发 403/418/429/432 达到阈值：{blocked_streak}，请检查 cookie / 降低频率 / 更换网络",
                    file=sys.stderr,
                )
                break

            if sleep_range and len(todo) > 1:
                lo, hi = float(sleep_range[0]), float(sleep_range[1])
                if hi > 0 and hi >= lo:
                    time.sleep(random.uniform(lo, hi))

    print(f"[done] input_uids={len(uids)} todo={len(todo)} ok={ok} fail={fail}")
    print(f"[done] saved: {output_path}")


if __name__ == "__main__":
    main()
