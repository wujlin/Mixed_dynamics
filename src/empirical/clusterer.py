"""
时间团簇划分（Empirical clustering）

目标：把时间轴按“发帖密度”（默认 n_public）自动分成若干连续高密度团簇，
用于在不同阶段（regime）分别检验 H1–H4，避免把非平稳/非均匀采样的数据混成一团。

设计原则：
- 只使用“密度类”输入（n_public / n_posts 等），不使用 Q/a/jump 等结果变量，避免选段偏置。
- 参数化：freq/roll_days/quantile/min_cluster_days/merge_gap_days/max_clusters 可显式配置。
- 简洁：rolling mean + 分位数阈值 + 连续区间合并。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeCluster:
    cluster_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_windows: int
    n_public_sum: int
    n_valid_a: int
    smooth_threshold: float


def _freq_to_step_hours(freq: str) -> float:
    td = pd.to_timedelta(freq)
    return float(td.total_seconds() / 3600.0)


@dataclass(frozen=True)
class EventClusterer:
    freq: str = "4h"  # pandas>=2.2 建议小时用小写 h
    density_col: str = "n_public"
    roll_days: float = 14.0
    quantile: float = 0.9
    min_cluster_days: float = 21.0
    merge_gap_days: float = 7.0
    max_clusters: int = 0

    def find_clusters(self, ts: pd.DataFrame) -> list[TimeCluster]:
        if ts.empty:
            return []
        if "time_window" not in ts.columns:
            raise ValueError("ts 缺少 time_window")
        if self.density_col not in ts.columns:
            raise ValueError(f"ts 缺少密度列：{self.density_col}")

        df = ts.sort_values("time_window").reset_index(drop=True).copy()
        step_hours = _freq_to_step_hours(self.freq)
        win_per_day = 24.0 / float(step_hours)
        roll_win = max(1, int(round(float(self.roll_days) * win_per_day)))
        min_len = max(1, int(round(float(self.min_cluster_days) * win_per_day)))
        merge_gap = max(0, int(round(float(self.merge_gap_days) * win_per_day)))

        density = df[self.density_col].fillna(0).astype(float)
        smooth = density.rolling(roll_win, center=True, min_periods=max(1, roll_win // 3)).mean().fillna(0.0)
        positive = smooth[smooth > 0]
        if len(positive) < max(10, roll_win):
            return []
        thr = float(np.quantile(positive.values, float(self.quantile)))
        mask = smooth >= thr
        if not bool(mask.any()):
            return []

        runs: list[tuple[int, int]] = []
        start = None
        for i, v in enumerate(mask.values.tolist()):
            if v and start is None:
                start = i
            if (not v) and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(df) - 1))

        runs = [(s, e) for (s, e) in runs if (e - s + 1) >= min_len]
        if not runs:
            return []

        merged: list[tuple[int, int]] = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            if s <= cur_e + 1 + merge_gap:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))

        clusters: list[TimeCluster] = []
        for cid, (s, e) in enumerate(merged):
            g = df.loc[s:e]
            clusters.append(
                TimeCluster(
                    cluster_id=int(cid),
                    start=pd.Timestamp(g["time_window"].iloc[0]),
                    end=pd.Timestamp(g["time_window"].iloc[-1]),
                    n_windows=int(len(g)),
                    n_public_sum=int(g[self.density_col].fillna(0).sum()),
                    n_valid_a=int(g["a"].notna().sum()) if "a" in g.columns else 0,
                    smooth_threshold=thr,
                )
            )

        if int(self.max_clusters) > 0 and len(clusters) > int(self.max_clusters):
            top = sorted(clusters, key=lambda c: (c.n_windows, c.n_public_sum), reverse=True)[: int(self.max_clusters)]
            top = sorted(top, key=lambda c: c.start)
            clusters = [
                TimeCluster(i, c.start, c.end, c.n_windows, c.n_public_sum, c.n_valid_a, c.smooth_threshold) for i, c in enumerate(top)
            ]

        return clusters
