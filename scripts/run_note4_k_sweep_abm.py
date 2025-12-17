"""
Note04（04_Sensitivity_Chi_Landscape）补强：信息密度 k 的 ABM 对照扫参

目标：
- 对 Part 2 的理论曲线 rc(k) 给出 ABM 对照：在对称验证口径（symmetric_mode=True, beta=0）下，
  用“最大斜率法（含局部二次插值）”估计 rc(k) 并 bootstrap 给出 95% 区间。
- 输出 .npz 缓存供 notebook 加载绘图（不在此脚本画图，方便工作站多进程跑）。

说明：
- k 在理论里对应“个体每步采样到的信号数”，实现上应使用 sample_mode=fixed 且 sample_n=k。
- 为避免把网络拓扑效应与 k 效应混在一起，默认用 ER + 较大 avg_degree 近似 well-mixed。

用法示例（工作站）：
python3 scripts/run_note4_k_sweep_abm.py \\
  --jobs 48 \\
  --k-list 10,20,50,100,200,500 \\
  --n 400 --avg-degree 50 --model er \\
  --update-rate 0.1 --steps 8000 --record-interval 10 \\
  --burn-in-frac 0.5 --metric-window 200 \\
  --r-min 0 --r-max 1 --r-num 201 \\
  --seeds 0-127 --bootstrap 5000 --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.theory import calculate_chi, calculate_rc  # noqa: E402
from src.network_sim import NetworkAgentModel, NetworkConfig  # noqa: E402


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


def _rc_from_max_slope(r_vals: np.ndarray, y: np.ndarray, *, min_delta: float) -> float:
    r_vals = np.asarray(r_vals, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(r_vals) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    r = r_vals[mask]
    v = y[mask]
    if float(np.nanmax(v) - np.nanmin(v)) < float(min_delta):
        return float("nan")
    dv = np.gradient(v, r)
    if not np.isfinite(dv).any():
        return float("nan")
    idx = int(np.nanargmax(dv))

    # 局部二次拟合插值（减轻 rc 被 r 网格量化锁死的问题）
    if 0 < idx < (r.size - 1):
        x = r[idx - 1 : idx + 2]
        y3 = dv[idx - 1 : idx + 2]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y3)):
            try:
                a, b, _c = np.polyfit(x, y3, deg=2).astype(float).tolist()
                if a != 0.0:
                    x_star = -b / (2.0 * a)
                    if float(x[0]) <= float(x_star) <= float(x[2]) and np.isfinite(x_star):
                        if a < 0.0:
                            return float(x_star)
            except Exception:
                pass

    return float(r[idx])


def _bootstrap_rc(
    r_vals: np.ndarray,
    y_seed: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    min_delta: float,
) -> Tuple[float, float, float, int]:
    rng = np.random.default_rng(int(seed))
    y_seed = np.asarray(y_seed, dtype=float)
    if y_seed.ndim != 2:
        raise ValueError("y_seed 必须为二维数组 (nR, nSeeds)")
    n_seeds = int(y_seed.shape[1])
    if n_seeds <= 0:
        return float("nan"), float("nan"), float("nan"), 0

    rc_boot = np.full(int(n_boot), np.nan, dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        y = np.nanmedian(y_seed[:, idx], axis=1)
        rc_boot[b] = _rc_from_max_slope(r_vals, y, min_delta=min_delta)

    valid = rc_boot[np.isfinite(rc_boot)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    med = float(np.nanmedian(valid))
    lo, hi = np.percentile(valid, [2.5, 97.5]).astype(float).tolist()
    return med, float(lo), float(hi), int(valid.size)


@dataclass(frozen=True)
class SeedTask:
    k: int
    seed: int
    r_vals: np.ndarray
    # model cfg
    n: int
    avg_degree: float
    model: str
    update_rate: float
    steps: int
    record_interval: int
    burn_in_frac: float
    metric_window: int
    init_state: str
    phi: float
    theta: float
    n_m: float
    n_w: float


def _simulate_one_seed(task: SeedTask) -> Tuple[int, int, np.ndarray]:
    """返回：(k, seed, q_abs_mean_over_r)"""
    nR = int(task.r_vals.size)
    q_abs_mean = np.full(nR, np.nan, dtype=float)

    burn_step = float(task.steps) * float(task.burn_in_frac)
    for i, r in enumerate(task.r_vals):
        cfg = NetworkConfig(
            n=int(task.n),
            avg_degree=float(task.avg_degree),
            model=str(task.model),
            beta=0.0,
            r=float(r),
            n_m=float(task.n_m),
            n_w=float(task.n_w),
            phi=float(task.phi),
            theta=float(task.theta),
            seed=int(task.seed),
            init_state=str(task.init_state),
            sample_mode="fixed",
            sample_n=int(task.k),
            symmetric_mode=True,
            update_rate=float(task.update_rate),
            local_mode="high_only",
        )
        sim = NetworkAgentModel(cfg)
        t, q_traj, _a_traj = sim.run(steps=int(task.steps), record_interval=int(task.record_interval))
        mask = t >= burn_step
        q_ss = q_traj[mask]
        if q_ss.size < 3:
            continue
        if int(task.metric_window) > 0 and q_ss.size > int(task.metric_window):
            q_ss = q_ss[-int(task.metric_window) :]
        q_abs_mean[i] = float(np.mean(np.abs(q_ss)))

    return int(task.k), int(task.seed), q_abs_mean


def _int_list_sig(values: List[int]) -> str:
    vals = sorted({int(v) for v in values})
    if not vals:
        return "none"
    if len(vals) == 1:
        return str(vals[0])
    return f"{vals[0]}-{vals[-1]}"


def _unit_range_sig(r_min: float, r_max: float, *, scale: int = 1000) -> str:
    a = int(round(float(r_min) * int(scale)))
    b = int(round(float(r_max) * int(scale)))
    return f"{a}-{b}"


def _default_out_path(
    *,
    output_dir: Path,
    phi: float,
    theta: float,
    n_m: float,
    n_w: float,
    n: int,
    avg_degree: float,
    model: str,
    update_rate: float,
    steps: int,
    record_interval: int,
    burn_in_frac: float,
    metric_window: int,
    init_state: str,
    n_seeds: int,
    n_r: int,
    r_min: float,
    r_max: float,
    k_list: List[int],
    tag: str,
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    phi_tag = int(round(phi * 100))
    theta_tag = int(round(theta * 100))
    burn_tag = int(round(burn_in_frac * 100))
    deg_tag = int(round(avg_degree))
    k_sig = _int_list_sig(k_list)
    r_sig = _unit_range_sig(r_min, r_max)
    name = (
        f"note4_k_sweep_abm_phi{phi_tag}_theta{theta_tag}_"
        f"nm{int(n_m)}_nw{int(n_w)}_"
        f"N{int(n)}_deg{deg_tag}_{model}_u{int(round(update_rate*100))}_"
        f"steps{int(steps)}_ri{int(record_interval)}_burn{burn_tag}_"
        f"win{int(metric_window)}_seeds{int(n_seeds)}_r{int(n_r)}_rr{r_sig}_"
        f"k{len(k_list)}_k{k_sig}_init{str(init_state)}_{tag}.npz"
    )
    return data_dir / name


def main() -> None:
    p = argparse.ArgumentParser(description="Run Note04 Part2 ABM k sweep and save .npz")
    p.add_argument("--phi", type=float, default=0.54)
    p.add_argument("--theta", type=float, default=0.46)
    p.add_argument("--k-list", type=str, default="10,20,50,100,200,500")
    p.add_argument("--n-m", type=float, default=10.0)
    p.add_argument("--n-w", type=float, default=5.0)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--avg-degree", type=float, default=50.0)
    p.add_argument("--model", choices=["er", "ba"], default="er")
    p.add_argument("--update-rate", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--record-interval", type=int, default=10)
    p.add_argument("--burn-in-frac", type=float, default=0.5)
    p.add_argument("--metric-window", type=int, default=200)
    p.add_argument("--init-state", choices=["random", "medium"], default="random")
    p.add_argument("--r-min", type=float, default=0.0)
    p.add_argument("--r-max", type=float, default=1.0)
    p.add_argument("--r-num", type=int, default=201)
    p.add_argument("--seeds", type=str, default="0-127")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--min-delta", type=float, default=0.25)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    k_list = _parse_int_list(args.k_list)
    if not k_list:
        raise SystemExit("k-list 不能为空")
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("seeds 不能为空")
    r_vals = np.linspace(float(args.r_min), float(args.r_max), int(args.r_num), dtype=float)

    chi_theory = np.array([float(calculate_chi(phi=args.phi, theta=args.theta, k_avg=int(k))) for k in k_list], dtype=float)
    rc_theory = np.array([float(calculate_rc(n_m=float(args.n_m), n_w=float(args.n_w), chi=float(chi))) for chi in chi_theory], dtype=float)

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
            n_m=float(args.n_m),
            n_w=float(args.n_w),
            n=int(args.n),
            avg_degree=float(args.avg_degree),
            model=str(args.model),
            update_rate=float(args.update_rate),
            steps=int(args.steps),
            record_interval=int(args.record_interval),
            burn_in_frac=float(args.burn_in_frac),
            metric_window=int(args.metric_window),
            init_state=str(args.init_state),
            n_seeds=len(seeds),
            n_r=int(r_vals.size),
            r_min=float(args.r_min),
            r_max=float(args.r_max),
            k_list=k_list,
            tag="v1",
        )

    if out_path.exists() and not args.force:
        print(f"[skip] 输出已存在：{out_path}")
        print("如需覆盖请加 --force，或用 --out 指定新文件名。")
        return

    print(
        f"scan k={k_list}, r=[{args.r_min},{args.r_max}] x{r_vals.size}, "
        f"n={args.n}, deg={args.avg_degree}, model={args.model}, "
        f"steps={args.steps}, ri={args.record_interval}, burn={args.burn_in_frac}, win={args.metric_window}, "
        f"seeds={len(seeds)}, jobs={args.jobs}"
    )

    nK = len(k_list)
    nR = int(r_vals.size)
    nS = len(seeds)
    q_abs = np.full((nK, nR, nS), np.nan, dtype=float)

    idx_k = {int(k): i for i, k in enumerate(k_list)}
    idx_seed = {int(s): i for i, s in enumerate(seeds)}

    tasks = [
        SeedTask(
            k=int(k),
            seed=int(seed),
            r_vals=r_vals,
            n=int(args.n),
            avg_degree=float(args.avg_degree),
            model=str(args.model),
            update_rate=float(args.update_rate),
            steps=int(args.steps),
            record_interval=int(args.record_interval),
            burn_in_frac=float(args.burn_in_frac),
            metric_window=int(args.metric_window),
            init_state=str(args.init_state),
            phi=float(args.phi),
            theta=float(args.theta),
            n_m=float(args.n_m),
            n_w=float(args.n_w),
        )
        for k in k_list
        for seed in seeds
    ]

    def run_sequential() -> None:
        done = 0
        for t in tasks:
            k, seed, q_abs_s = _simulate_one_seed(t)
            ki = idx_k[int(k)]
            si = idx_seed[int(seed)]
            q_abs[ki, :, si] = q_abs_s
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[progress] {done}/{len(tasks)} tasks finished")

    if int(args.jobs) <= 1:
        run_sequential()
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
                futs = {ex.submit(_simulate_one_seed, t): (t.k, t.seed) for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    k, seed, q_abs_s = fut.result()
                    ki = idx_k[int(k)]
                    si = idx_seed[int(seed)]
                    q_abs[ki, :, si] = q_abs_s
                    done += 1
                    if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                        print(f"[progress] {done}/{len(tasks)} tasks finished")
        except PermissionError as e:
            print(f"[warn] 多进程不可用（{e}），已降级串行；可设置 --jobs 1 或在工作站运行。")
            run_sequential()

    # 估计 rc(k)：对 seed 聚合曲线做最大斜率，并给出 bootstrap CI
    rc_est = np.full(nK, np.nan, dtype=float)
    rc_ci_low = np.full(nK, np.nan, dtype=float)
    rc_ci_high = np.full(nK, np.nan, dtype=float)
    rc_valid_boot = np.zeros(nK, dtype=int)

    for ki, k in enumerate(k_list):
        med, lo, hi, valid = _bootstrap_rc(
            r_vals,
            q_abs[ki, :, :],
            n_boot=int(args.bootstrap),
            seed=20251215 + int(k),
            min_delta=float(args.min_delta),
        )
        rc_est[ki] = med
        rc_ci_low[ki] = lo
        rc_ci_high[ki] = hi
        rc_valid_boot[ki] = int(valid)
        print(f"[rc] k={int(k)}: rc_maxslope={med:.4f} CI95%=[{lo:.4f},{hi:.4f}] (valid_boot={valid}/{int(args.bootstrap)})")

    np.savez(
        out_path,
        # params
        phi=float(args.phi),
        theta=float(args.theta),
        n_m=float(args.n_m),
        n_w=float(args.n_w),
        n=int(args.n),
        avg_degree=float(args.avg_degree),
        model=str(args.model),
        update_rate=float(args.update_rate),
        steps=int(args.steps),
        record_interval=int(args.record_interval),
        burn_in_frac=float(args.burn_in_frac),
        metric_window=int(args.metric_window),
        init_state=str(args.init_state),
        # grids
        r_vals=r_vals.astype(float),
        seeds=np.asarray(seeds, dtype=int),
        k_list=np.asarray(k_list, dtype=int),
        # theory
        chi_theory=chi_theory.astype(float),
        rc_theory=rc_theory.astype(float),
        # raw (nK, nR, nS)
        q_abs_mean=q_abs.astype(float),
        # rc estimate per k
        rc_est=rc_est.astype(float),
        rc_ci_low=rc_ci_low.astype(float),
        rc_ci_high=rc_ci_high.astype(float),
        rc_valid_boot=rc_valid_boot.astype(int),
    )
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()
