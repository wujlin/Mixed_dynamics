"""
Note04（04_Sensitivity_Chi_Landscape）Part 4：
将“社会耦合 beta”的 ABM 扫描外置为脚本（适合工作站多进程），输出 .npz 供 notebook 加载绘图。

设计目标：
- **口径更严谨**：多 seed + 明确 burn-in + 稳态窗口统计 + 95% 区间。
- **rc(β) 可复现**：用“最大斜率法”（order parameter |Q| 随 r 的最大导数位置）估计有效临界点。
- **KISS**：脚本只负责重计算与缓存；绘图留给 notebook（本地更方便排版）。

重要说明：
- 当 beta>0 时，当前 NetworkAgentModel 的“本地耦合”项使用邻居高唤醒数量 neighbor_high，
  这本质上引入了非对称社会放大/抑制机制；因此此处的 rc(β) 应理解为“有效转变点”，
  不再等同于对称理论推导的 r_c。
- 若使用 init_state=medium（全中立启动），在 beta>0 时更容易出现“分支选择偏置”（Q 更倾向某一符号）。
  建议用 init_state=random 作为稳健性对照，并在 notebook 中保留 signed Q/分支偏置诊断图以避免误读。
"""

from __future__ import annotations

import argparse
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


def _parse_float_list(text: str) -> List[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def _rc_from_max_slope(r_vals: np.ndarray, y: np.ndarray, *, min_delta: float) -> float:
    """
    用最大斜率法估计“转变点”：
    rc = argmax_r d y / d r

    说明：
    - 这里 y 通常取稳态 order parameter（例如 |Q| 的 seed 聚合曲线）。
    - 若曲线整体变化幅度过小（max-min < min_delta），返回 NaN 表示“未检测到清晰转变”。
    """
    r_vals = np.asarray(r_vals, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(r_vals) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    r = r_vals[mask]
    v = y[mask]
    if float(np.nanmax(v) - np.nanmin(v)) < float(min_delta):
        return float("nan")
    # np.gradient 会在端点做一侧差分，内部做中心差分
    dv = np.gradient(v, r)
    if not np.isfinite(dv).any():
        return float("nan")
    idx = int(np.nanargmax(dv))

    # 用局部二次拟合对“最大斜率点”做亚网格插值，减轻 rc 被 r 网格量化锁死的问题
    if 0 < idx < (r.size - 1):
        x = r[idx - 1 : idx + 2]
        y3 = dv[idx - 1 : idx + 2]
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y3)):
            try:
                a, b, _c = np.polyfit(x, y3, deg=2).astype(float).tolist()
                if a != 0.0:
                    x_star = -b / (2.0 * a)
                    if float(x[0]) <= float(x_star) <= float(x[2]) and np.isfinite(x_star):
                        # 只有在局部呈现“凹向下”(a<0) 且顶点落在邻域内时才采用
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
    """
    对 seed 维度 bootstrap，输出 (median, q2.5, q97.5, valid_boot)。
    y_seed: (nR, nSeeds)
    """
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
    beta: float
    seed: int
    r_vals: np.ndarray
    n: int
    avg_degree: float
    model: str
    update_rate: float
    steps: int
    record_interval: int
    burn_in_frac: float
    metric_window: int
    init_state: str
    sample_mode: str
    sample_n: int
    phi: float
    theta: float
    n_m: float
    n_w: float
    symmetric_mode: bool
    local_mode: str


def _simulate_one_seed(task: SeedTask) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    - seed
    - beta
    - q_abs_mean: (nR,)
    - q_mean:     (nR,)
    - a_mean:     (nR,)
    """
    nR = int(task.r_vals.size)
    q_abs_mean = np.full(nR, np.nan, dtype=float)
    q_mean = np.full(nR, np.nan, dtype=float)
    a_mean = np.full(nR, np.nan, dtype=float)

    burn_step = float(task.steps) * float(task.burn_in_frac)

    for i, r in enumerate(task.r_vals):
        cfg = NetworkConfig(
            n=int(task.n),
            avg_degree=float(task.avg_degree),
            model=str(task.model),
            beta=float(task.beta),
            r=float(r),
            n_m=float(task.n_m),
            n_w=float(task.n_w),
            phi=float(task.phi),
            theta=float(task.theta),
            seed=int(task.seed),
            init_state=str(task.init_state),
            sample_mode=str(task.sample_mode),
            sample_n=int(task.sample_n),
            symmetric_mode=bool(task.symmetric_mode),
            update_rate=float(task.update_rate),
            local_mode=str(task.local_mode),
        )
        sim = NetworkAgentModel(cfg)
        t, q_traj, a_traj = sim.run(steps=int(task.steps), record_interval=int(task.record_interval))

        mask = t >= burn_step
        q_ss = q_traj[mask]
        a_ss = a_traj[mask]
        if q_ss.size < 3:
            continue
        if int(task.metric_window) > 0 and q_ss.size > int(task.metric_window):
            q_ss = q_ss[-int(task.metric_window) :]
            a_ss = a_ss[-int(task.metric_window) :]

        q_mean[i] = float(np.mean(q_ss))
        q_abs_mean[i] = float(np.mean(np.abs(q_ss)))
        a_mean[i] = float(np.mean(a_ss))

    return int(task.seed), float(task.beta), q_abs_mean, q_mean, a_mean


def _default_out_path(
    *,
    output_dir: Path,
    phi: float,
    theta: float,
    k_avg: int,
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
    n_seeds: int,
    n_r: int,
    n_beta: int,
    local_mode: str,
    tag: str,
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    phi_tag = int(round(phi * 100))
    theta_tag = int(round(theta * 100))
    burn_tag = int(round(burn_in_frac * 100))
    deg_tag = int(round(avg_degree))
    name = (
        f"note4_beta_sweep_abm_phi{phi_tag}_theta{theta_tag}_"
        f"nm{int(n_m)}_nw{int(n_w)}_k{k_avg}_"
        f"N{int(n)}_deg{deg_tag}_{model}_u{int(round(update_rate*100))}_"
        f"steps{int(steps)}_ri{int(record_interval)}_burn{burn_tag}_"
        f"win{int(metric_window)}_seeds{int(n_seeds)}_r{int(n_r)}_b{int(n_beta)}_lm{local_mode}_{tag}.npz"
    )
    return data_dir / name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Note04 Part4 beta sweep (network ABM) and save .npz.")
    parser.add_argument("--phi", type=float, default=0.54)
    parser.add_argument("--theta", type=float, default=0.46)
    parser.add_argument("--k-avg", type=int, default=50)
    parser.add_argument("--n-m", type=float, default=10.0)
    parser.add_argument("--n-w", type=float, default=5.0)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--avg-degree", type=float, default=50.0)
    parser.add_argument("--model", choices=["er", "ba"], default="er")
    parser.add_argument("--update-rate", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--burn-in-frac", type=float, default=0.5)
    parser.add_argument(
        "--metric-window",
        type=int,
        default=100,
        help="稳态窗口长度（按记录点计数；<=0 表示使用 burn-in 后全部样本）",
    )
    parser.add_argument("--init-state", choices=["random", "medium"], default="medium")
    parser.add_argument("--sample-mode", choices=["fixed", "degree"], default="fixed")
    parser.add_argument(
        "--local-mode",
        choices=["high_only", "symmetric"],
        default="high_only",
        help="beta>0 时的局部耦合口径：high_only=仅邻居高唤醒；symmetric=对称局部概率(含 low/medium 基线)",
    )
    parser.add_argument("--betas", type=str, default="0,0.02,0.05,0.1")
    # 为了避免高 beta 下曲线在局部窗口内“已饱和”导致 rc=NaN，默认扫描全区间 [0, 1]
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=1.0)
    parser.add_argument("--r-num", type=int, default=201)
    parser.add_argument("--seeds", type=str, default="0-31")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.25,
        help="最大斜率法判定所需的最小幅度：max(y)-min(y) < min_delta 则认为无清晰转变",
    )
    parser.add_argument("--q-low", type=float, default=0.1, help="|Q| 接近 0 的判定阈值（用于“无转变”分类）")
    parser.add_argument("--q-high", type=float, default=0.9, help="|Q| 接近 1 的判定阈值（用于“无转变”分类）")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    betas = _parse_float_list(args.betas)
    if not betas:
        raise SystemExit("betas 不能为空")
    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("seeds 不能为空")
    r_vals = np.linspace(float(args.r_min), float(args.r_max), int(args.r_num), dtype=float)

    chi_ref = float(calculate_chi(phi=float(args.phi), theta=float(args.theta), k_avg=int(args.k_avg)))
    rc_ref = float(calculate_rc(n_m=float(args.n_m), n_w=float(args.n_w), chi=chi_ref))

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
            avg_degree=float(args.avg_degree),
            model=str(args.model),
            update_rate=float(args.update_rate),
            steps=int(args.steps),
            record_interval=int(args.record_interval),
            burn_in_frac=float(args.burn_in_frac),
            metric_window=int(args.metric_window),
            n_seeds=len(seeds),
            n_r=int(r_vals.size),
            n_beta=len(betas),
            local_mode=str(args.local_mode),
            tag="v1",
        )

    if out_path.exists() and not args.force:
        print(f"[skip] 输出已存在：{out_path}")
        print("如需覆盖请加 --force，或用 --out 指定新文件名。")
        return

    print(f"theory reference: chi={chi_ref:.6f}, rc={rc_ref:.6f} (beta=0, symmetric)")
    print(
        f"scan: betas={betas}, r=[{args.r_min},{args.r_max}] x{r_vals.size}, "
        f"n={args.n}, deg={args.avg_degree}, model={args.model}, local_mode={args.local_mode}, "
        f"steps={args.steps}, ri={args.record_interval}, burn={args.burn_in_frac}, win={args.metric_window}, "
        f"seeds={len(seeds)}, jobs={args.jobs}"
    )

    nB = len(betas)
    nR = int(r_vals.size)
    nS = len(seeds)
    q_abs = np.full((nB, nR, nS), np.nan, dtype=float)
    q_mean = np.full((nB, nR, nS), np.nan, dtype=float)
    a_mean = np.full((nB, nR, nS), np.nan, dtype=float)

    idx_beta = {float(b): i for i, b in enumerate(betas)}
    idx_seed = {int(s): i for i, s in enumerate(seeds)}

    tasks = [
        SeedTask(
            beta=float(beta),
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
            sample_mode=str(args.sample_mode),
            sample_n=int(args.k_avg),
            phi=float(args.phi),
            theta=float(args.theta),
            n_m=float(args.n_m),
            n_w=float(args.n_w),
            symmetric_mode=True,
            local_mode=str(args.local_mode),
        )
        for beta in betas
        for seed in seeds
    ]

    def run_sequential() -> None:
        done = 0
        for t in tasks:
            seed, beta, q_abs_s, q_mean_s, a_mean_s = _simulate_one_seed(t)
            bi = idx_beta[float(beta)]
            si = idx_seed[int(seed)]
            q_abs[bi, :, si] = q_abs_s
            q_mean[bi, :, si] = q_mean_s
            a_mean[bi, :, si] = a_mean_s
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[progress] {done}/{len(tasks)} tasks finished")

    if int(args.jobs) <= 1:
        run_sequential()
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(args.jobs)) as ex:
                futs = {ex.submit(_simulate_one_seed, t): (t.beta, t.seed) for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    seed, beta, q_abs_s, q_mean_s, a_mean_s = fut.result()
                    bi = idx_beta[float(beta)]
                    si = idx_seed[int(seed)]
                    q_abs[bi, :, si] = q_abs_s
                    q_mean[bi, :, si] = q_mean_s
                    a_mean[bi, :, si] = a_mean_s
                    done += 1
                    if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                        print(f"[progress] {done}/{len(tasks)} tasks finished")
        except PermissionError as e:
            print(f"[warn] 多进程不可用（{e}），已降级串行；可设置 --jobs 1 或在工作站运行。")
            run_sequential()

    # 汇总诊断量
    q_abs_med = np.nanmedian(q_abs, axis=2)  # (nB, nR)
    q_abs_min = np.nanmin(q_abs_med, axis=1)  # (nB,)
    q_abs_max = np.nanmax(q_abs_med, axis=1)  # (nB,)

    # 分支偏置（以最大 r 点作为参考）：正分支占比
    sign_ref = np.sign(q_mean[:, -1, :])  # (nB, nS)
    sign_ref[sign_ref == 0] = 1
    pos_branch_frac = (sign_ref > 0).mean(axis=1).astype(float)

    # 估计 rc(beta)：对 seed 聚合曲线做最大斜率，并给出 bootstrap CI
    rc_est = np.full(nB, np.nan, dtype=float)
    rc_ci_low = np.full(nB, np.nan, dtype=float)
    rc_ci_high = np.full(nB, np.nan, dtype=float)
    rc_valid_boot = np.zeros(nB, dtype=int)
    rc_seed = np.full((nB, nS), np.nan, dtype=float)
    regimes: List[str] = []

    for bi, beta in enumerate(betas):
        curve_min = float(q_abs_min[bi])
        curve_max = float(q_abs_max[bi])
        curve_delta = float(curve_max - curve_min)

        scan_min = float(r_vals[0])
        scan_max = float(r_vals[-1])
        scan_covers_0 = scan_min <= 1e-12
        scan_covers_1 = scan_max >= 1.0 - 1e-12

        # 若窗口内几乎无变化：将其作为“无可辨识转变”而非强行给出 rc
        if curve_delta < float(args.min_delta):
            if curve_min >= float(args.q_high):
                regime = "polarized_all_r" if scan_covers_0 else "polarized_in_scan"
                rc_est[bi] = 0.0 if scan_covers_0 else scan_min
                rc_ci_low[bi] = rc_est[bi]
                rc_ci_high[bi] = rc_est[bi]
                rc_valid_boot[bi] = 0
                regimes.append(regime)
                print(
                    f"[rc] beta={float(beta):.4g}: {regime} (|Q|~[{curve_min:.3f},{curve_max:.3f}] in scan), "
                    f"rc_est={rc_est[bi]:.4f}"
                )
                continue

            if curve_max <= float(args.q_low):
                regime = "neutral_all_r" if scan_covers_1 else "neutral_in_scan"
                rc_est[bi] = 1.0 if scan_covers_1 else scan_max
                rc_ci_low[bi] = rc_est[bi]
                rc_ci_high[bi] = rc_est[bi]
                rc_valid_boot[bi] = 0
                regimes.append(regime)
                print(
                    f"[rc] beta={float(beta):.4g}: {regime} (|Q|~[{curve_min:.3f},{curve_max:.3f}] in scan), "
                    f"rc_est={rc_est[bi]:.4f}"
                )
                continue

            regime = "flat_no_transition"
            regimes.append(regime)
            print(
                f"[rc] beta={float(beta):.4g}: {regime} (|Q|~[{curve_min:.3f},{curve_max:.3f}] in scan), rc_est=nan"
            )
            continue

        regimes.append("transition")
        # seed-level diagnostics
        for si in range(nS):
            rc_seed[bi, si] = _rc_from_max_slope(r_vals, q_abs[bi, :, si], min_delta=float(args.min_delta))

        med, lo, hi, valid = _bootstrap_rc(
            r_vals,
            q_abs[bi, :, :],
            n_boot=int(args.bootstrap),
            seed=20251214 + int(round(float(beta) * 10_000)),
            min_delta=float(args.min_delta),
        )
        rc_est[bi] = med
        rc_ci_low[bi] = lo
        rc_ci_high[bi] = hi
        rc_valid_boot[bi] = int(valid)
        print(
            f"[rc] beta={float(beta):.4g}: rc_maxslope={med:.4f} "
            f"CI95%=[{lo:.4f},{hi:.4f}] (valid_boot={valid}/{int(args.bootstrap)})"
        )

    np.savez(
        out_path,
        # reference theory (beta=0 symmetric)
        chi_ref=float(chi_ref),
        rc_ref=float(rc_ref),
        # grids
        r_vals=r_vals.astype(float),
        betas=np.asarray(betas, dtype=float),
        seeds=np.asarray(seeds, dtype=int),
        # raw stats: (nB, nR, nS)
        q_abs_mean=q_abs.astype(float),
        q_mean=q_mean.astype(float),
        a_mean=a_mean.astype(float),
        q_abs_med=q_abs_med.astype(float),
        q_abs_min=q_abs_min.astype(float),
        q_abs_max=q_abs_max.astype(float),
        pos_branch_frac=pos_branch_frac.astype(float),
        # rc estimates
        rc_method="max_slope_parabola",
        min_delta=float(args.min_delta),
        q_low=float(args.q_low),
        q_high=float(args.q_high),
        regime=np.asarray(regimes, dtype=str),
        rc_est=rc_est.astype(float),
        rc_ci_low=rc_ci_low.astype(float),
        rc_ci_high=rc_ci_high.astype(float),
        rc_valid_boot=rc_valid_boot.astype(int),
        rc_seed=rc_seed.astype(float),
        # params
        phi=float(args.phi),
        theta=float(args.theta),
        k_avg=int(args.k_avg),
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
        sample_mode=str(args.sample_mode),
        symmetric_mode=True,
        local_mode=str(args.local_mode),
        bootstrap=int(args.bootstrap),
    )
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()
