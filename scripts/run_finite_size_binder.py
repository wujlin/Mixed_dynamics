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

交点选择（避免 “cross-window 依赖理论 rc” 的质疑）：
- window：在 rc_theory±cross_window 内选择交点（保留作对照/调试）。
- max-slope：在 “过渡区” 内选择局部斜率最大的交点（默认）。
  过渡区用 U4 的取值范围限定：u4_low <= U4_cross <= u4_high，
  以排除低 r 平台噪声交点与高 r 饱和平台“退化交点”。
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


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    if int(window) <= 1:
        return y.astype(float, copy=False)
    window = int(window)
    if window % 2 == 0:
        raise ValueError("smooth_window 必须为奇数（或 1 代表不平滑）")
    if window > y.size:
        raise ValueError("smooth_window 不能超过序列长度")
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    y_pad = np.pad(y.astype(float, copy=False), (pad, pad), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def crossing_position_in_window(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    center: float,
    window: float,
) -> float:
    """
    线性插值求交点，并限制在 [center-window, center+window] 内。
    若窗口内无交点，返回 NaN。
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
    lo, hi = center - window, center + window
    in_window = [v for v in crossings if lo <= v <= hi]
    if not in_window:
        return float("nan")
    return float(min(in_window, key=lambda r: abs(r - center)))


def crossing_position_max_slope(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    u4_low: float,
    u4_high: float,
    smooth_window: int = 1,
    slope_eps: float = 1e-12,
) -> float:
    """
    最大斜率交点：
    - 枚举 y1-y2 的所有交点（线性插值）。
    - 仅保留 U4_cross 位于 [u4_low, u4_high] 的候选（排除平台噪声/饱和退化交点）。
    - 选择 |dU1/dr|+|dU2/dr| 最大的交点（局部段斜率）。
    - 若无候选，返回 NaN。
    """
    x = x.astype(float, copy=False)
    y1s = _smooth_1d(y1, int(smooth_window))
    y2s = _smooth_1d(y2, int(smooth_window))

    diff = y1s - y2s
    best = None  # (score, tie_break, rc)
    x_mid = 0.5 * (float(x[0]) + float(x[-1]))
    for i in range(len(x) - 1):
        d0, d1 = float(diff[i]), float(diff[i + 1])
        if not (math.isfinite(d0) and math.isfinite(d1)):
            continue
        if d0 == 0.0:
            t = 0.0
        elif d0 * d1 < 0.0:
            t = d0 / (d0 - d1)
        else:
            continue

        dx = float(x[i + 1] - x[i])
        if dx <= 0:
            continue

        rc = float(x[i] + t * dx)
        u_cross = float(y1s[i] + t * float(y1s[i + 1] - y1s[i]))
        if (u_cross < float(u4_low)) or (u_cross > float(u4_high)):
            continue

        s1 = abs(float(y1s[i + 1] - y1s[i]) / dx)
        s2 = abs(float(y2s[i + 1] - y2s[i]) / dx)
        score = float(s1 + s2)
        if score <= float(slope_eps):
            continue

        # tie_break：优先靠近扫描区间中点，减少平台边界的退化选择
        cand = (score, -abs(rc - x_mid), rc)
        if best is None or cand > best:
            best = cand
    return float("nan") if best is None else float(best[2])


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
    cross_method: str,
    version: str = "v3",
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    burn_tag = int(round(burn_in_frac * 100))
    phi_tag = int(round(phi * 100))
    theta_tag = int(round(theta * 100))
    method_tag = str(cross_method).replace("-", "").replace("_", "")
    name = (
        f"finite_size_binder_cross_sym_phi{phi_tag}_theta{theta_tag}_"
        f"nm{int(n_m)}_nw{int(n_w)}_k{k_avg}_"
        f"N{min(N_list)}-{max(N_list)}_"
        f"init{init_state}_u{int(round(update_rate*100))}_"
        f"ri{record_interval}_steps{steps}_burn{burn_tag}_"
        f"seeds{len(seeds)}_r{r_points}_{version}_cm{method_tag}.npz"
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
    parser.add_argument(
        "--cross-method",
        choices=["max-slope", "window"],
        default="max-slope",
        help="交点选择策略：max-slope（默认，不依赖理论 rc）或 window（依赖 rc_theory±cross_window）。",
    )
    parser.add_argument(
        "--cross-window",
        type=float,
        default=0.05,
        help="（仅 window 法）交点搜索窗口：仅统计落在 rc_theory±window 内的交点。",
    )
    parser.add_argument("--u4-low", type=float, default=0.2, help="（max-slope）交点 U4 下界（过滤低 r 平台噪声交点）")
    parser.add_argument(
        "--u4-high",
        type=float,
        default=(2.0 / 3.0 - 1e-3),
        help="（max-slope）交点 U4 上界（过滤高 r 饱和平台退化交点）",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="（max-slope）对 U4 曲线做简单滑动平均平滑的窗口长度（奇数；1=不平滑）",
    )
    parser.add_argument(
        "--slope-eps",
        type=float,
        default=1e-12,
        help="（max-slope）斜率阈值（过滤退化交点；一般不用改）",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="对 seeds 进行 bootstrap 次数（>0 时额外输出 rc_cross_bootstrap 及其置信区间）。",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0)
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
    print(
        f"cross_method = {args.cross_method} "
        f"(u4_low={args.u4_low:.3f}, u4_high={args.u4_high:.3f}, smooth_window={args.smooth_window})"
        if args.cross_method == "max-slope"
        else f"cross_method = {args.cross_method} (cross_window=±{args.cross_window})"
    )

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
            cross_method=str(args.cross_method),
            version="v4",
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

    def run_sequential() -> None:
        done = 0
        for cfg in tasks:
            N, seed, u4_vals = simulate_u4_curve(cfg)
            u4_seeds_by_N[idx_N[int(N)], idx_seed[int(seed)], :] = u4_vals
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[progress] {done}/{len(tasks)} tasks finished")

    if int(args.jobs) <= 1:
        run_sequential()
    else:
        try:
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
        except PermissionError as e:
            # 部分环境（例如某些 WSL/容器配置）可能无法创建 SemLock，自动降级到串行。
            print(f"[warn] 多进程不可用（{e}），已自动降级为串行；如需并行请在工作站运行或设置 --jobs 1。")
            run_sequential()

    binder_mean_by_N = np.nanmean(u4_seeds_by_N, axis=1)
    binder_std_by_N = np.nanstd(u4_seeds_by_N, axis=1, ddof=1)
    binder_sem_by_N = binder_std_by_N / math.sqrt(len(seeds))

    # 相邻 N 的交点：按 seed 先算交点，再汇总 mean/std
    rc_cross_mean: List[float] = []
    rc_cross_std: List[float] = []
    rc_cross_median: List[float] = []
    rc_cross_valid: List[int] = []
    pair_N_mid: List[float] = []
    pair_labels: List[str] = []
    rc_cross_seeds_by_pair: List[np.ndarray] = []

    for i in range(len(N_list) - 1):
        N1, N2 = int(N_list[i]), int(N_list[i + 1])
        rc_seeds = np.zeros(len(seeds), dtype=float)
        for si in range(len(seeds)):
            if args.cross_method == "window":
                rc_seeds[si] = crossing_position_in_window(
                    r_scan,
                    u4_seeds_by_N[i, si],
                    u4_seeds_by_N[i + 1, si],
                    center=rc_theory,
                    window=float(args.cross_window),
                )
            else:
                rc_seeds[si] = crossing_position_max_slope(
                    r_scan,
                    u4_seeds_by_N[i, si],
                    u4_seeds_by_N[i + 1, si],
                    u4_low=float(args.u4_low),
                    u4_high=float(args.u4_high),
                    smooth_window=int(args.smooth_window),
                    slope_eps=float(args.slope_eps),
                )
        rc_cross_seeds_by_pair.append(rc_seeds)
        valid = np.isfinite(rc_seeds)
        valid_cnt = int(np.sum(valid))
        rc_mean = float(np.nanmean(rc_seeds)) if valid_cnt > 0 else float("nan")
        rc_std = float(np.nanstd(rc_seeds, ddof=1)) if valid_cnt > 1 else float("nan")
        rc_med = float(np.nanmedian(rc_seeds)) if valid_cnt > 0 else float("nan")
        rc_cross_mean.append(rc_mean)
        rc_cross_std.append(rc_std)
        rc_cross_median.append(rc_med)
        rc_cross_valid.append(valid_cnt)
        pair_N_mid.append(float(math.sqrt(N1 * N2)))
        pair_labels.append(f"{N1}-{N2}")
        print(
            f"[cross] N={N1} vs {N2}: rc_cross ≈ {rc_mean:.4f} ± {rc_std:.4f} "
            f"(seed-to-seed, median={rc_med:.4f}, valid={valid_cnt}/{len(seeds)})"
        )

    # Bootstrap（从 mean 曲线交点获取更稳的区间估计）
    rc_cross_bootstrap = None
    rc_cross_ci = None
    if int(args.bootstrap) > 0:
        rng = np.random.default_rng(int(args.bootstrap_seed))
        B = int(args.bootstrap)
        rc_seed = np.asarray(rc_cross_seeds_by_pair, dtype=float)  # (nPairs, nSeeds)
        rc_cross_bootstrap = np.full((len(N_list) - 1, B), np.nan, dtype=float)
        for b in range(B):
            idx = rng.integers(0, len(seeds), size=len(seeds))
            sample = rc_seed[:, idx]  # (nPairs, nSeeds)
            # 使用 median 抑制少量离群（更适合写进论文）
            #
            # 注意：当某个 pair 的 seed-level crossing 本身包含 NaN（例如被 u4_band 过滤掉），
            # bootstrap 重采样有极小概率抽到 “全是 NaN” 的样本；直接 np.nanmedian 会抛出 warning。
            # 这里显式避开 all-NaN 行，保持输出一致且日志更干净。
            med = np.full(sample.shape[0], np.nan, dtype=float)
            valid_row = np.any(np.isfinite(sample), axis=1)
            if np.any(valid_row):
                med[valid_row] = np.nanmedian(sample[valid_row], axis=1)
            rc_cross_bootstrap[:, b] = med
        # 2.5/50/97.5 分位数
        q = np.nanquantile(rc_cross_bootstrap, [0.025, 0.5, 0.975], axis=1)
        rc_cross_ci = q.astype(float)  # shape: (3, nPairs)
        for i in range(len(N_list) - 1):
            med = float(rc_cross_ci[1, i])
            lo = float(rc_cross_ci[0, i])
            hi = float(rc_cross_ci[2, i])
            valid_b = int(np.sum(np.isfinite(rc_cross_bootstrap[i])))
            print(
                f"[bootstrap] {N_list[i]}-{N_list[i+1]}: median={med:.4f}, "
                f"95%CI=[{lo:.4f}, {hi:.4f}] (valid_boot={valid_b}/{B})"
            )

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
        cross_method=str(args.cross_method),
        cross_window=(float(args.cross_window) if args.cross_method == "window" else float("nan")),
        u4_low=float(args.u4_low),
        u4_high=float(args.u4_high),
        smooth_window=int(args.smooth_window),
        slope_eps=float(args.slope_eps),
        # curves
        binder_seeds_by_N=u4_seeds_by_N,
        binder_mean_by_N=binder_mean_by_N,
        binder_sem_by_N=binder_sem_by_N,
        # crossings
        pair_N_mid=np.asarray(pair_N_mid, dtype=float),
        pair_labels=np.asarray(pair_labels, dtype=str),
        rc_cross_mean=np.asarray(rc_cross_mean, dtype=float),
        rc_cross_std=np.asarray(rc_cross_std, dtype=float),
        rc_cross_median=np.asarray(rc_cross_median, dtype=float),
        rc_cross_valid=np.asarray(rc_cross_valid, dtype=int),
        rc_cross_seeds_by_pair=np.asarray(rc_cross_seeds_by_pair, dtype=float),
        bootstrap=int(args.bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
        rc_cross_bootstrap=(rc_cross_bootstrap if rc_cross_bootstrap is not None else np.asarray([], dtype=float)),
        rc_cross_ci=(rc_cross_ci if rc_cross_ci is not None else np.asarray([], dtype=float)),
    )
    print(f"[done] Saved: {out_path}")


if __name__ == "__main__":
    main()
