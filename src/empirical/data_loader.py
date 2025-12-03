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

from pathlib import Path
from typing import Optional

from .user_mapper import UserTypeMapper, map_user_types_batch


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

    if drop_empty:
        df = df[df["content"].notna() & (df["content"].str.strip() != "")]

    if parse_dates:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")

    # 用户类型映射
    user_types = map_user_types_batch(
        df["verify_typ"].fillna("未知").tolist(),
        df["user_name"].fillna("").tolist(),
        mapper=mapper,
    )
    df["user_type"] = user_types

    return df.reset_index(drop=True)
