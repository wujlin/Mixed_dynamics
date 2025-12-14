"""
有限尺寸效应（Finite-Size Scaling）：Binder cumulant 交点估计 r_c（对称模式 / beta=0）。

目的：
- 将 notebook 中较慢的 Binder 扫描抽成脚本，便于在工作站用多进程跑完并产出 .npz 数据。

默认口径（与 notebooks/02_Network_Topology.ipynb 保持一致）：
- phi=0.54, theta=0.46, k_avg=50, n_m=10, n_w=5
- N_list=[100,200,500,1000,2000], seeds=0..7
- steps=2000, record_interval=5, burn_in_frac=0.5, update_rate=0.1
- r 在 [rc_theory-r_span, rc_theory+r_span] 上均匀扫描（r_span=0.15, r_points=41）

输出：
- outputs/data/finite_size_binder_cross_*.npz（包含 binder 曲线、按 seed 的交点统计等）
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


# -----------------------------
# 路径与理论模块导入
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402


# -----------------------------
# 简化的 well-mixed ABM（beta=0 时与网络无关）
# -----------------------------
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
    p_env = num / denom
    return float(np.clip(p_env, 0.0, 1.0))


def binder_u4_from_q(q_window: np.ndarray) -> float:
    q2 = float(np.mean(q_window**2))
    if q2 <= 1e-12:
        return 0.0
    q4 = float(np.mean(q_window**4))
    return 1.0 - q4 / (3.0 * (q2**2))


def crossing_position(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, target: float) -> float:
    """
    线性插值求交点；若存在多个交点，返回最接近 target 的那个。
    若无交点，返回 NaN。
    """
    diff = y1 - y2
    crossings: List[float] = []
    for i in range(len(x) - 1):
        if diff[i] == 0:
            crossings.append(float(x[i]))
            continue
        if diff[i] * diff[i + 1] < 0:
            t = diff[i] / (diff[i] - diff[i + 1])
            crossings.append(float(x[i] + t * (x[i + 1] - x[i])))
    if not crossings:
        return float("nan")
    return float(min(crossings, key=lambda r: abs(r - target)))


@dataclass(frozen=True)
class SweepConfig:
    N: int
    seed: int
    r_scan: np.ndarray
    steps: int
    record_interval: int
    burn_step: float
    update_rate: float
    init_state: str
    sample_n: int
    phi: float
    theta: float
    n_m: float
    n_w: float
    symmetric_mode: bool


def simulate_u4_curve(cfg: SweepConfig) -> Tuple[int, int, np.ndarray]:
    """
    单个 (N, seed) 任务：扫描 r，返回该 seed 下的 U4(r) 曲线。
    返回： (N, seed, u4_values[r_index])
    """
    u4_vals = np.zeros(len(cfg.r_scan), dtype=float)

    for ri, r in enumerate(cfg.r_scan):
        rng = np.random.default_rng(cfg.seed)
        if cfg.init_state == "random":
            state = rng.choice(
                [STATE_LOW, STATE_MEDIUM, STATE_HIGH],
                size=cfg.N,
                p=np.array([1 / 3, 1 / 3, 1 / 3]),
            ).astype(int)
        elif cfg.init_state == "medium":
            state = np.full(cfg.N, STATE_MEDIUM, dtype=int)
        else:
            raise ValueError("init_state 仅支持 'random' 或 'medium'")

        times: List[int] = []
        q_records: List[float] = []

        for step in range(cfg.steps):
            if step % cfg.record_interval == 0:
                q, a = _macro_stats(state)
                times.append(step)
                q_records.append(q)

            q, a = _macro_stats(state)
            p_env = _global_env(
                q=q,
                a=a,
                r=float(r),
                n_m=cfg.n_m,
                n_w=cfg.n_w,
                symmetric_mode=cfg.symmetric_mode,
            )
            signal_counts = rng.binomial(n=cfg.sample_n, p=p_env, size=cfg.N)
            perceived_risk = signal_counts / float(cfg.sample_n)
            new_state = np.where(
                perceived_risk >= cfg.phi,
                STATE_HIGH,
                np.where(perceived_risk <= cfg.theta, STATE_LOW, STATE_MEDIUM),
            ).astype(int)
            update_mask = rng.random(cfg.N) < cfg.update_rate
            state = np.where(update_mask, new_state, state)

        # 记录最后一步（对齐 NetworkAgentModel.run 的行为）
        q, _a = _macro_stats(state)
        times.append(cfg.steps)
        q_records.append(q)

        t_arr = np.asarray(times, dtype=float)
        q_arr = np.asarray(q_records, dtype=float)
        q_window = q_arr[t_arr >= cfg.burn_step]
        u4_vals[ri] = binder_u4_from_q(q_window)

    return cfg.N, cfg.seed, u4_vals


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


def default_output_path(
    *,
    output_dir: Path,
    phi: float,
    theta: float,
    n_m: float,
    n_w: float,
    k_avg: int,
    N_list: Sequence[int],
    init_state: str,
    update_rate: float,
    record_interval: int,
    steps: int,
    burn_in_frac: float,
    seeds: Sequence[int],
    r_points: int,
    version: str = "v3",
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    burn_tag = int(round(burn_in_frac * 100))
    phi_tag = int(round(phi * 100))
    theta_tag = int(round(theta * 100))
    name = (
        f"finite_size_binder_cross_sym_phi{phi_tag}_theta{theta_tag}_"
        f"nm{int(n_m)}_nw{int(n_w)}_k{k_avg}_"
        f"N{min(N_list)}-{max(N_list)}_"
        f"init{init_state}_u{int(round(update_rate*100))}_"
        f"ri{record_interval}_steps{steps}_burn{burn_tag}_"
        f"seeds{len(seeds)}_r{r_points}_{version}.npz"
    )
    return data_dir / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-size scaling via Binder cumulant crossing (multiprocess).")
    parser.add_argument("--phi", type=float, default=0.54)
    parser.add_argument("--theta", type=float, default=0.46)
    parser.add_argument("--k-avg", type=int, default=50)
    parser.add_argument("--n-m", type=float, default=10.0)
    parser.add_argument("--n-w", type=float, default=5.0)
    parser.add_argument("--N-list", type=str, default="100,200,500,1000,2000")
    parser.add_argument("--seeds", type=str, default="0-7")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--record-interval", type=int, default=5)
    parser.add_argument("--burn-in-frac", type=float, default=0.5)
    parser.add_argument("--update-rate", type=float, default=0.1)
    parser.add_argument("--init-state", choices=["random", "medium"], default="random")
    parser.add_argument("--r-span", type=float, default=0.15)
    parser.add_argument("--r-points", type=int, default=41)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    N_list = _parse_int_list(args.N_list)
    seeds = _parse_int_list(args.seeds)
    if not N_list:
        raise SystemExit("N_list 不能为空")
    if not seeds:
        raise SystemExit("seeds 不能为空")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chi = theory.calculate_chi(args.phi, args.theta, k_avg=args.k_avg)
    rc_theory = float(theory.calculate_rc(args.n_m, args.n_w, chi))
    print(f"理论 rc = {rc_theory:.4f} (phi={args.phi}, theta={args.theta}, k_avg={args.k_avg})")

    r_min = max(0.0, rc_theory - args.r_span)
    r_max = min(1.0, rc_theory + args.r_span)
    r_scan = np.linspace(r_min, r_max, int(args.r_points))

    burn_step = float(args.steps) * float(args.burn_in_frac)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = default_output_path(
            output_dir=output_dir,
            phi=args.phi,
            theta=args.theta,
            n_m=args.n_m,
            n_w=args.n_w,
            k_avg=args.k_avg,
            N_list=N_list,
            init_state=args.init_state,
            update_rate=args.update_rate,
            record_interval=args.record_interval,
            steps=args.steps,
            burn_in_frac=args.burn_in_frac,
            seeds=seeds,
            r_points=len(r_scan),
            version="v3",
        )

    if out_path.exists() and not args.force:
        print(f"[skip] 输出已存在：{out_path}")
        print("如需覆盖请加 --force，或用 --out 指定新文件名。")
        return

    # 预分配：u4_seeds_by_N[i, si, ri]
    u4_seeds_by_N = np.full((len(N_list), len(seeds), len(r_scan)), np.nan, dtype=float)

    tasks: List[SweepConfig] = []
    for i, N in enumerate(N_list):
        sample_n = min(int(args.k_avg), int(N) - 1)
        if sample_n <= 0:
            raise SystemExit(f"N={N} 太小，导致 sample_n<=0")
        for seed in seeds:
            tasks.append(
                SweepConfig(
                    N=int(N),
                    seed=int(seed),
                    r_scan=r_scan,
                    steps=int(args.steps),
                    record_interval=int(args.record_interval),
                    burn_step=burn_step,
                    update_rate=float(args.update_rate),
                    init_state=str(args.init_state),
                    sample_n=int(sample_n),
                    phi=float(args.phi),
                    theta=float(args.theta),
                    n_m=float(args.n_m),
                    n_w=float(args.n_w),
                    symmetric_mode=True,
                )
            )

    print(
        "[run] Binder 扫描开始："
        f" N={N_list}, seeds={len(seeds)}, r_points={len(r_scan)},"
        f" steps={args.steps}, ri={args.record_interval}, burn={args.burn_in_frac},"
        f" update_rate={args.update_rate}, jobs={args.jobs}"
    )

    idx_N = {int(N): i for i, N in enumerate(N_list)}
    idx_seed = {int(s): i for i, s in enumerate(seeds)}

    with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
        futures = {ex.submit(simulate_u4_curve, cfg): cfg for cfg in tasks}
        done = 0
        for fut in as_completed(futures):
            cfg = futures[fut]
            N, seed, u4_vals = fut.result()
            u4_seeds_by_N[idx_N[int(N)], idx_seed[int(seed)], :] = u4_vals
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[progress] {done}/{len(tasks)} tasks finished")

    binder_mean_by_N = np.nanmean(u4_seeds_by_N, axis=1)
    binder_std_by_N = np.nanstd(u4_seeds_by_N, axis=1, ddof=1)
    binder_sem_by_N = binder_std_by_N / math.sqrt(len(seeds))

    # 相邻 N 的交点：按 seed 先算交点，再汇总 mean/std
    rc_cross_mean: List[float] = []
    rc_cross_std: List[float] = []
    pair_N_mid: List[float] = []
    pair_labels: List[str] = []
    rc_cross_seeds_by_pair: List[np.ndarray] = []

    for i in range(len(N_list) - 1):
        N1, N2 = int(N_list[i]), int(N_list[i + 1])
        rc_seeds = np.zeros(len(seeds), dtype=float)
        for si in range(len(seeds)):
            rc_seeds[si] = crossing_position(r_scan, u4_seeds_by_N[i, si], u4_seeds_by_N[i + 1, si], rc_theory)
        rc_cross_seeds_by_pair.append(rc_seeds)
        rc_mean = float(np.nanmean(rc_seeds))
        rc_std = float(np.nanstd(rc_seeds, ddof=1))
        rc_cross_mean.append(rc_mean)
        rc_cross_std.append(rc_std)
        pair_N_mid.append(float(math.sqrt(N1 * N2)))
        pair_labels.append(f"{N1}-{N2}")
        print(f"[cross] N={N1} vs {N2}: rc_cross ≈ {rc_mean:.4f} ± {rc_std:.4f} (seed-to-seed)")

    np.savez(
        out_path,
        # basic grid
        N_list=np.asarray(N_list, dtype=int),
        seeds=np.asarray(seeds, dtype=int),
        r_scan=r_scan.astype(float),
        # params
        phi=float(args.phi),
        theta=float(args.theta),
        k_avg=int(args.k_avg),
        n_m=float(args.n_m),
        n_w=float(args.n_w),
        steps=int(args.steps),
        record_interval=int(args.record_interval),
        burn_in_frac=float(args.burn_in_frac),
        update_rate=float(args.update_rate),
        init_state=str(args.init_state),
        r_span=float(args.r_span),
        rc_theory=float(rc_theory),
        # curves
        binder_seeds_by_N=u4_seeds_by_N,
        binder_mean_by_N=binder_mean_by_N,
        binder_sem_by_N=binder_sem_by_N,
        # crossings
        pair_N_mid=np.asarray(pair_N_mid, dtype=float),
        pair_labels=np.asarray(pair_labels, dtype=str),
        rc_cross_mean=np.asarray(rc_cross_mean, dtype=float),
        rc_cross_std=np.asarray(rc_cross_std, dtype=float),
        rc_cross_seeds_by_pair=np.asarray(rc_cross_seeds_by_pair, dtype=float),
    )
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()

