"""
临界慢化（CSD）验证：ABM（well-mixed / beta=0）多 seed、多 lag 扫描并缓存为 .npz。

设计目标：
- **口径唯一**：与 notebooks/03_Critical_Slowing_Down.ipynb 使用同一套统计定义。
- **可复现/可扩展**：重计算放到脚本里（适合工作站多进程），notebook 只负责加载与绘图。
- **避免伪影**：默认使用异步更新 update_rate=0.1，且 lag 用“sweep”时间单位统一解释。

说明：
- 当 beta=0 时，网络拓扑不影响个体感知（只剩全局信号），因此可用 well-mixed 向量化 ABM
  与 NetworkAgentModel 在统计意义上等价，但速度更快。

输出（npz）：
- r_vals: (nR,)
- seeds: (nSeeds,)
- lags_sweeps: (nLag,)   # 以“sweep”为单位的滞后（1 sweep = 1/update_rate 个 ABM steps）
- lags_index: (nLag,)    # 在记录序列上的索引滞后（考虑 record_interval 与 update_rate）
- ac: (nR, nSeeds, nLag) # 稳态窗口内的自相关
- var: (nR, nSeeds)      # 稳态窗口内方差
- mean: (nR, nSeeds)     # 稳态窗口内均值（用于检查漂移）
以及参数与理论 rc。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calculate_autocorrelation  # noqa: E402
from src import calculate_chi, calculate_rc  # noqa: E402

STATE_HIGH = 1
STATE_MEDIUM = 0
STATE_LOW = -1


def _macro_stats(state: np.ndarray) -> Tuple[float, float]:
    n = state.size
    n_high = int(np.sum(state == STATE_HIGH))
    n_low = int(np.sum(state == STATE_LOW))
    q = (n_high - n_low) / n
    a = (n_high + n_low) / n
    return float(q), float(a)


def _global_env(
    q: float,
    a: float,
    r: float,
    n_m: float,
    n_w: float,
    symmetric_mode: bool,
) -> float:
    p_main = (1.0 - q) / 2.0
    if symmetric_mode:
        p_we = 0.5 + q / 2.0
    else:
        p_we = (a + q) / 2.0
    num = (1.0 - r) * n_m * p_main + r * n_w * p_we
    denom = (1.0 - r) * n_m + r * n_w
    return float(np.clip(num / denom, 0.0, 1.0))


def _parse_int_list(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        a, b = text.split("-", 1)
        start = int(a.strip())
        end = int(b.strip())
        if end < start:
            raise ValueError(f"invalid range: {text}")
        return list(range(start, end + 1))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _r_scan_from_logdist(rc: float, dist_min: float, dist_max: float, r_points: int, include_zero: bool) -> np.ndarray:
    dist_min = float(dist_min)
    dist_max = float(dist_max)
    if dist_min <= 0 or dist_max <= 0:
        raise ValueError("dist_min/dist_max 必须为正数")
    if dist_max <= dist_min:
        raise ValueError("dist_max 必须大于 dist_min")
    d = np.logspace(np.log10(dist_min), np.log10(dist_max), int(r_points))
    r = rc - d
    r = r[(r > 0.0) & (r < rc)]
    if include_zero:
        r = np.unique(np.concatenate([np.array([0.0]), r]))
    return r.astype(float)


def _lags_index_from_sweeps(
    lags_sweeps: Sequence[float],
    update_rate: float,
    record_interval: int,
) -> np.ndarray:
    dt_eff = float(update_rate) * float(record_interval)  # 记录序列每个点对应的 sweep 时间
    if dt_eff <= 0:
        raise ValueError("update_rate 与 record_interval 必须为正")
    idx = []
    for lag in lags_sweeps:
        k = int(round(float(lag) / dt_eff))
        idx.append(max(1, k))
    return np.asarray(idx, dtype=int)


@dataclass(frozen=True)
class SeedTask:
    seed: int
    n: int
    r_vals: np.ndarray
    steps: int
    record_interval: int
    burn_in_frac: float
    metric_window: int
    update_rate: float
    init_state: str
    sample_n: int
    phi: float
    theta: float
    n_m: float
    n_w: float
    symmetric_mode: bool
    lag_index: np.ndarray


def _simulate_one_seed(task: SeedTask) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    - seed
    - ac:   (nR, nLag)
    - var:  (nR,)
    - mean: (nR,)
    """
    nR = int(task.r_vals.size)
    nLag = int(task.lag_index.size)
    ac = np.full((nR, nLag), np.nan, dtype=float)
    var = np.full(nR, np.nan, dtype=float)
    mean = np.full(nR, np.nan, dtype=float)

    burn_step = float(task.steps) * float(task.burn_in_frac)

    for ri, r in enumerate(task.r_vals):
        rng = np.random.default_rng(int(task.seed) * 1_000_003 + int(ri))
        if task.init_state == "random":
            state = rng.choice(
                [STATE_LOW, STATE_MEDIUM, STATE_HIGH],
                size=int(task.n),
                p=np.array([1 / 3, 1 / 3, 1 / 3]),
            ).astype(int)
        elif task.init_state == "medium":
            state = np.full(int(task.n), STATE_MEDIUM, dtype=int)
        else:
            raise ValueError("init_state 仅支持 'random' 或 'medium'")

        times: List[int] = []
        q_records: List[float] = []

        for step in range(int(task.steps)):
            if step % int(task.record_interval) == 0:
                q, _a = _macro_stats(state)
                times.append(step)
                q_records.append(q)

            q, a = _macro_stats(state)
            p_env = _global_env(
                q=q,
                a=a,
                r=float(r),
                n_m=float(task.n_m),
                n_w=float(task.n_w),
                symmetric_mode=bool(task.symmetric_mode),
            )
            signal_counts = rng.binomial(n=int(task.sample_n), p=p_env, size=int(task.n))
            perceived_risk = signal_counts / float(task.sample_n)
            new_state = np.where(
                perceived_risk >= float(task.phi),
                STATE_HIGH,
                np.where(perceived_risk <= float(task.theta), STATE_LOW, STATE_MEDIUM),
            ).astype(int)
            update_mask = rng.random(int(task.n)) < float(task.update_rate)
            state = np.where(update_mask, new_state, state)

        # 记录最后一步（对齐 NetworkAgentModel.run）
        q, _a = _macro_stats(state)
        times.append(int(task.steps))
        q_records.append(q)

        t_arr = np.asarray(times, dtype=float)
        q_arr = np.asarray(q_records, dtype=float)

        steady = q_arr[t_arr >= burn_step]
        if steady.size < 3:
            continue

        if int(task.metric_window) > 0:
            steady = steady[-int(task.metric_window) :]

        mean[ri] = float(np.mean(steady))
        var[ri] = float(np.var(steady))
        for li, lag in enumerate(task.lag_index):
            if int(lag) < steady.size:
                ac[ri, li] = calculate_autocorrelation(steady, lag=int(lag))

    return int(task.seed), ac, var, mean


def _default_out_path(
    *,
    output_dir: Path,
    phi: float,
    theta: float,
    k_avg: int,
    n_m: float,
    n_w: float,
    n: int,
    update_rate: float,
    steps: int,
    record_interval: int,
    burn_in_frac: float,
    metric_window: int,
    n_seeds: int,
    n_r: int,
    tag: str,
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    phi_tag = int(round(phi * 100))
    theta_tag = int(round(theta * 100))
    burn_tag = int(round(burn_in_frac * 100))
    name = (
        f"csd_abm_wm_sym_phi{phi_tag}_theta{theta_tag}_"
        f"nm{int(n_m)}_nw{int(n_w)}_k{k_avg}_"
        f"N{int(n)}_u{int(round(update_rate*100))}_"
        f"steps{int(steps)}_ri{int(record_interval)}_burn{burn_tag}_"
        f"win{int(metric_window)}_seeds{int(n_seeds)}_r{int(n_r)}_{tag}.npz"
    )
    return data_dir / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABM (well-mixed, beta=0) CSD scan and save npz.")
    parser.add_argument("--phi", type=float, default=0.54)
    parser.add_argument("--theta", type=float, default=0.46)
    parser.add_argument("--k-avg", type=int, default=50)
    parser.add_argument("--n-m", type=float, default=10.0)
    parser.add_argument("--n-w", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--update-rate", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--record-interval", type=int, default=1)
    parser.add_argument("--burn-in-frac", type=float, default=0.5)
    parser.add_argument(
        "--metric-window",
        type=int,
        default=5000,
        help="稳态窗口长度（按记录点计数；<=0 表示使用 burn-in 后全部样本）",
    )
    parser.add_argument("--init-state", choices=["random", "medium"], default="medium")
    parser.add_argument("--seeds", type=str, default="0-31")
    parser.add_argument("--lags-sweeps", type=str, default="0.5,1,2,4", help="以 sweep 为单位的 lag 列表")
    parser.add_argument("--dist-min", type=float, default=0.005)
    parser.add_argument("--dist-max", type=float, default=0.7)
    parser.add_argument("--r-points", type=int, default=25)
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("seeds 不能为空")
    lags_sweeps = _parse_float_list(args.lags_sweeps)
    if not lags_sweeps:
        raise SystemExit("lags_sweeps 不能为空")

    chi = calculate_chi(phi=float(args.phi), theta=float(args.theta), k_avg=int(args.k_avg))
    rc = float(calculate_rc(n_m=float(args.n_m), n_w=float(args.n_w), chi=chi))
    r_vals = _r_scan_from_logdist(
        rc=rc,
        dist_min=float(args.dist_min),
        dist_max=float(args.dist_max),
        r_points=int(args.r_points),
        include_zero=bool(args.include_zero),
    )
    lag_index = _lags_index_from_sweeps(
        lags_sweeps=lags_sweeps,
        update_rate=float(args.update_rate),
        record_interval=int(args.record_interval),
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = _default_out_path(
            output_dir=output_dir,
            phi=float(args.phi),
            theta=float(args.theta),
            k_avg=int(args.k_avg),
            n_m=float(args.n_m),
            n_w=float(args.n_w),
            n=int(args.n),
            update_rate=float(args.update_rate),
            steps=int(args.steps),
            record_interval=int(args.record_interval),
            burn_in_frac=float(args.burn_in_frac),
            metric_window=int(args.metric_window),
            n_seeds=len(seeds),
            n_r=int(r_vals.size),
            tag="v1",
        )

    if out_path.exists() and not args.force:
        print(f"[skip] 输出已存在：{out_path}")
        print("如需覆盖请加 --force，或用 --out 指定新文件名。")
        return

    print(f"theory: chi={chi:.6f}, rc={rc:.6f}")
    print(
        f"scan: n={args.n}, update_rate={args.update_rate}, steps={args.steps}, "
        f"ri={args.record_interval}, burn={args.burn_in_frac}, window={args.metric_window}, "
        f"seeds={len(seeds)}, r_points={r_vals.size}"
    )
    print(f"lags_sweeps={lags_sweeps}")
    print(f"lags_index={lag_index.tolist()} (recorded series)")

    tasks = [
        SeedTask(
            seed=int(s),
            n=int(args.n),
            r_vals=r_vals,
            steps=int(args.steps),
            record_interval=int(args.record_interval),
            burn_in_frac=float(args.burn_in_frac),
            metric_window=int(args.metric_window),
            update_rate=float(args.update_rate),
            init_state=str(args.init_state),
            sample_n=int(args.k_avg),
            phi=float(args.phi),
            theta=float(args.theta),
            n_m=float(args.n_m),
            n_w=float(args.n_w),
            symmetric_mode=True,
            lag_index=lag_index,
        )
        for s in seeds
    ]

    ac = np.full((int(r_vals.size), len(seeds), int(lag_index.size)), np.nan, dtype=float)
    var = np.full((int(r_vals.size), len(seeds)), np.nan, dtype=float)
    mean = np.full((int(r_vals.size), len(seeds)), np.nan, dtype=float)
    idx_seed = {int(s): i for i, s in enumerate(seeds)}

    def run_sequential() -> None:
        done = 0
        for t in tasks:
            seed, ac_s, var_s, mean_s = _simulate_one_seed(t)
            j = idx_seed[int(seed)]
            ac[:, j, :] = ac_s
            var[:, j] = var_s
            mean[:, j] = mean_s
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[progress] {done}/{len(tasks)} seeds finished")

    if int(args.jobs) <= 1:
        run_sequential()
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
                futs = {ex.submit(_simulate_one_seed, t): t.seed for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    seed, ac_s, var_s, mean_s = fut.result()
                    j = idx_seed[int(seed)]
                    ac[:, j, :] = ac_s
                    var[:, j] = var_s
                    mean[:, j] = mean_s
                    done += 1
                    if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                        print(f"[progress] {done}/{len(tasks)} seeds finished")
        except PermissionError as e:
            print(f"[warn] 多进程不可用（{e}），已降级串行；可设置 --jobs 1 或在工作站运行。")
            run_sequential()

    np.savez(
        out_path,
        # theory
        chi=float(chi),
        rc=float(rc),
        # grid
        r_vals=r_vals.astype(float),
        seeds=np.asarray(seeds, dtype=int),
        lags_sweeps=np.asarray(lags_sweeps, dtype=float),
        lags_index=lag_index.astype(int),
        # stats
        ac=ac.astype(float),
        var=var.astype(float),
        mean=mean.astype(float),
        # params
        phi=float(args.phi),
        theta=float(args.theta),
        k_avg=int(args.k_avg),
        n_m=float(args.n_m),
        n_w=float(args.n_w),
        n=int(args.n),
        update_rate=float(args.update_rate),
        steps=int(args.steps),
        record_interval=int(args.record_interval),
        burn_in_frac=float(args.burn_in_frac),
        metric_window=int(args.metric_window),
        init_state=str(args.init_state),
        symmetric_mode=True,
    )
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()

