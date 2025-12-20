"""
数据加载与预处理模块

功能：
- 读取 Weibo 话题数据 csv
- 规范列名、转换时间
- 调用 UserTypeMapper 映射用户类型

返回 pandas.DataFrame，列至少包含：
- mid, user_name, verify_typ, publish_time, content
- forward_num, comment_num, like_num
- user_type （mainstream/wemedia/public/government/other）
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .user_mapper import UserTypeMapper, map_user_types_batch

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "mid",
    "user_name",
    "verify_typ",
    "publish_time",
    "content",
    "forward_num",
    "comment_num",
    "like_num",
]


def _ensure_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - 运行时报错提示
        raise ImportError("需要 pandas 来加载经验数据，请先安装 pandas") from exc
    return pd


def load_topic_dataset(
    path: Path | str,
    mapper: Optional[UserTypeMapper] = None,
    user_meta_path: Optional[Path | str] = None,
    limit: Optional[int] = None,
    drop_empty: bool = True,
    parse_dates: bool = True,
):
    """
    加载并预处理话题数据 csv。

    Parameters
    ----------
    path : Path | str
        csv 文件路径
    mapper : UserTypeMapper, optional
        用户类型映射器，默认使用内置规则
    limit : int, optional
        仅读取前 N 行（调试用）
    drop_empty : bool
        是否丢弃空内容行
    parse_dates : bool
        是否解析 publish_time 为 datetime

    Returns
    -------
    pd.DataFrame
    """
    pd = _ensure_pandas()
    path = Path(path)
    mapper = mapper or UserTypeMapper()

    df = pd.read_csv(path, nrows=limit)
    # 清理 BOM 列名
    df = df.rename(columns={c: c.lstrip("\ufeff") for c in df.columns})

    # 仅保留常用列，缺失则填充
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # 可选：从 user_meta 回填 verify_typ / user_type（用于缺少认证字段的数据源）
    # user_meta 支持 CSV/JSONL，键为 uid 或 user_link
    uid_to_verify: dict[str, str] = {}
    uid_to_user_type: dict[str, str] = {}
    if user_meta_path:
        meta_path = Path(user_meta_path)
        if meta_path.exists():
            uid_to_verify, uid_to_user_type = _load_user_meta(meta_path, pd)
            if ("uid" not in df.columns) and ("user_link" in df.columns or "weibo_link" in df.columns):
                uid_user = df["user_link"].apply(_safe_extract_uid) if "user_link" in df.columns else None
                uid_weibo = df["weibo_link"].apply(_safe_extract_uid) if "weibo_link" in df.columns else None

                if uid_user is None and uid_weibo is None:
                    pass
                elif uid_user is None:
                    df["uid"] = uid_weibo
                elif uid_weibo is None:
                    df["uid"] = uid_user
                else:
                    df["uid"] = uid_user
                    fallback = df["uid"].isna() & uid_weibo.notna()
                    if fallback.any():
                        df.loc[fallback, "uid"] = uid_weibo[fallback]
                        logger.warning(
                            "uid 解析：%d 条 user_link 失败，已从 weibo_link 回退补齐",
                            int(fallback.sum()),
                        )
                    missing = df["uid"].isna()
                    if missing.any():
                        logger.warning("uid 解析：仍有 %d/%d 条缺失", int(missing.sum()), int(len(df)))

            if "uid" in df.columns and uid_to_verify:
                uid_series = df["uid"].astype(str)
                uid_ok = df["uid"].notna() & (uid_series != "nan") & (uid_series.str.strip() != "")
                verify_missing = df["verify_typ"].isna() | (df["verify_typ"].astype(str).str.strip().isin(["", "未知"]))
                mask = uid_ok & verify_missing & uid_series.isin(uid_to_verify)
                if mask.any():
                    df.loc[mask, "verify_typ"] = uid_series.map(uid_to_verify)

    if drop_empty:
        df = df[df["content"].notna() & (df["content"].str.strip() != "")]

    if parse_dates:
        # publish_time 在不同来源 csv 中可能存在混合格式；优先用 pandas>=2.0 的 mixed 解析
        try:
            df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce", format="mixed")
        except TypeError:
            df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    # 用户类型映射
    user_types = map_user_types_batch(
        df["verify_typ"].fillna("未知").tolist(),
        df["user_name"].fillna("").tolist(),
        mapper=mapper,
    )
    df["user_type"] = user_types

    # 若 user_meta 直接提供 user_type，则作为最终覆盖
    if "uid" in df.columns and uid_to_user_type:
        uid_series = df["uid"].astype(str)
        uid_ok = df["uid"].notna() & (uid_series != "nan") & (uid_series.str.strip() != "")
        mask = uid_ok & uid_series.isin(uid_to_user_type)
        if mask.any():
            df.loc[mask, "user_type"] = uid_series.map(uid_to_user_type)

    return df.reset_index(drop=True)


_UID_RE = re.compile(r"weibo\.com/(?:u/)?(\d+)", re.IGNORECASE)


def _safe_extract_uid(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if str(value) == "nan":
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return None
    m = _UID_RE.search(s)
    if not m:
        return None
    return m.group(1)


def _load_user_meta(meta_path: Path, pd):
    """
    读取用户元信息文件（CSV/JSONL），返回：
    - uid -> verify_typ
    - uid -> user_type
    """
    uid_to_verify: dict[str, str] = {}
    uid_to_user_type: dict[str, str] = {}

    def _non_empty_value(v) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if s.lower() == "nan":
            return None
        return s

    def put(rec: dict):
        # 抓取失败/错误行不参与回填（避免把 NaN 写成字符串 "nan" 覆盖掉真实 user_type）
        err = _non_empty_value(rec.get("error"))
        if err:
            return
        uid = rec.get("uid") or rec.get("user_id") or rec.get("id")
        if not uid:
            uid = _safe_extract_uid(rec.get("user_link") or rec.get("url") or "")
        uid = str(uid).strip() if uid is not None else ""
        if not uid or uid == "nan":
            return
        vt = _non_empty_value(rec.get("verify_typ") or rec.get("verify_type") or rec.get("verified_type"))
        if vt is not None:
            uid_to_verify[uid] = vt
        ut = _non_empty_value(rec.get("user_type"))
        if ut is not None:
            uid_to_user_type[uid] = ut

    suf = meta_path.suffix.lower()
    if suf == ".jsonl":
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    put(obj)
        return uid_to_verify, uid_to_user_type

    # 默认按 CSV 处理
    dfm = pd.read_csv(meta_path)
    dfm = dfm.rename(columns={c: c.lstrip("\ufeff") for c in dfm.columns})
    for _, row in dfm.iterrows():
        put(row.to_dict())
    return uid_to_verify, uid_to_user_type
