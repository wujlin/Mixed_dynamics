"""
批量预计算网络模拟结果，支持对称/非对称两种模式，用于生成 r-q-a 相图数据。

用法示例（默认对称模式）：
    python scripts/precompute_sym_asym.py --phi-list 0.54 --theta-list 0.46 \\
        --sample-n-list 50 --N-list 500 --mode symmetric

关键特性：
- symmetric_mode 开关：symmetric 用于理论验证（p_env 在 q=0 时为 0.5），asymmetric 用于真实机制展示。
- sample_mode 固定为 fixed，sample_n 与理论 k_avg 对齐，剥离拓扑/度数对采样的影响。
- 缓存：输出保存至 outputs/data/sim_{mode}_phi{...}_theta{...}_k{...}_N{...}_beta{...}_seeds{...}_steps{...}_r{...}_model{...}.npz，存在且未指定 --overwrite 时跳过重新计算。
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from joblib import Parallel, delayed

import sys

# 将项目根目录加入 sys.path，便于直接 python scripts/xxx.py 运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calculate_chi, calculate_rc
from src.network_sim import NetworkAgentModel, NetworkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute network simulations (symmetric/asymmetric).")
    parser.add_argument("--phi-list", type=float, nargs="+", default=[0.54], help="列表形式的 phi 值")
    parser.add_argument("--theta-list", type=float, nargs="+", default=[0.46], help="列表形式的 theta 值")
    parser.add_argument("--sample-n-list", type=int, nargs="+", default=[50], help="sample_n 列表，对应固定采样次数/理论 k_avg")
    parser.add_argument("--N-list", type=int, nargs="+", default=[500], help="节点数列表")
    parser.add_argument("--model", type=str, default="er", choices=["er", "ba"], help="网络模型")
    parser.add_argument("--beta", type=float, default=0.0, help="邻居耦合强度 beta")
    parser.add_argument("--r-start", type=float, default=0.0, help="r 起始")
    parser.add_argument("--r-end", type=float, default=1.0, help="r 结束")
    parser.add_argument("--r-num", type=int, default=30, help="r 采样点数量")
    parser.add_argument("--steps", type=int, default=300, help="仿真步数")
    parser.add_argument("--record-interval", type=int, default=50, help="记录步长")
    parser.add_argument("--window", type=int, default=5, help="稳态平均窗口（末尾步数）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="随机种子列表")
    parser.add_argument("--mode", type=str, choices=["symmetric", "asymmetric"], default="symmetric", help="对称/非对称模式")
    parser.add_argument("--n-m", type=float, default=10.0, help="主流媒体基数 n_m")
    parser.add_argument("--n-w", type=float, default=5.0, help="自媒体基数 n_w")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data"), help="输出目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有缓存")
    parser.add_argument("--n-jobs", type=int, default=-1, help="并行 worker 数，joblib Parallel")
    return parser.parse_args()


def build_filename(
    mode: str,
    phi: float,
    theta: float,
    sample_n: int,
    n_nodes: int,
    beta: float,
    seeds: Iterable[int],
    steps: int,
    r_num: int,
    model: str,
) -> str:
    seed_str = "s" + "-".join(str(s) for s in seeds)
    return (
        f"sim_{mode}_phi{phi:.3f}_theta{theta:.3f}_k{sample_n}_N{n_nodes}"
        f"_beta{beta}_seeds{seed_str}_steps{steps}_r{r_num}_model{model}.npz"
    )


def run_single_config(
    phi: float,
    theta: float,
    sample_n: int,
    n_nodes: int,
    r_values: np.ndarray,
    seeds: List[int],
    steps: int,
    record_interval: int,
    window: int,
    beta: float,
    model: str,
    n_m: float,
    n_w: float,
    symmetric_mode: bool,
) -> Tuple[np.ndarray, dict]:
    results = {
        "mean_abs": [],
        "abs_mean": [],
        "signed_mean": [],
        "a_mean": [],
        "a_signed": [],
    }

    for r in r_values:
        q_ma_seeds = []
        q_am_seeds = []
        q_signed_seeds = []
        a_seeds = []
        a_signed_seeds = []
        for seed in seeds:
            cfg = NetworkConfig(
                n=n_nodes,
                avg_degree=sample_n,
                model=model,
                beta=beta,
                r=r,
                n_m=n_m,
                n_w=n_w,
                phi=phi,
                theta=theta,
                seed=seed,
                init_state="medium",
                sample_mode="fixed",
                sample_n=sample_n,
                symmetric_mode=symmetric_mode,
            )
            sim = NetworkAgentModel(cfg)
            _, q_traj, a_traj = sim.run(steps=steps, record_interval=record_interval)

            steady_q = q_traj[-window:]
            steady_a = a_traj[-window:]
            q_ma_seeds.append(float(np.mean(np.abs(steady_q))))
            q_am_seeds.append(float(np.abs(np.mean(steady_q))))
            q_signed_seeds.append(float(np.mean(steady_q)))
            a_seeds.append(float(np.mean(steady_a)))
            a_signed_seeds.append(float(np.mean(steady_a - 0.5)))

        results["mean_abs"].append(np.mean(q_ma_seeds))
        results["abs_mean"].append(np.mean(q_am_seeds))
        results["signed_mean"].append(np.mean(q_signed_seeds))
        results["a_mean"].append(np.mean(a_seeds))
        results["a_signed"].append(np.mean(a_signed_seeds))

    # 转为 ndarray
    for k in results:
        results[k] = np.array(results[k], dtype=float)
    return r_values, results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "sym" if args.mode == "symmetric" else "asym"
    symmetric_flag = args.mode == "symmetric"
    r_values = np.linspace(args.r_start, args.r_end, args.r_num)

    if len(args.phi_list) != len(args.theta_list):
        raise ValueError("phi_list 和 theta_list 长度必须一致")

    combos = list(itertools.product(
        zip(args.phi_list, args.theta_list),
        args.sample_n_list,
        args.N_list,
    ))

    def worker(combo):
        (phi, theta), sample_n, n_nodes = combo
        fname = build_filename(
            mode_tag,
            phi,
            theta,
            sample_n,
            n_nodes,
            args.beta,
            args.seeds,
            args.steps,
            args.r_num,
            args.model,
        )
        fpath = args.output_dir / fname
        if fpath.exists() and not args.overwrite:
            return f"[skip] exists: {fpath}"

        r_vals, res = run_single_config(
            phi=phi,
            theta=theta,
            sample_n=sample_n,
            n_nodes=n_nodes,
            r_values=r_values,
            seeds=args.seeds,
            steps=args.steps,
            record_interval=args.record_interval,
            window=args.window,
            beta=args.beta,
            model=args.model,
            n_m=args.n_m,
            n_w=args.n_w,
            symmetric_mode=symmetric_flag,
        )
        chi = calculate_chi(phi=phi, theta=theta, k_avg=sample_n)
        rc = float(calculate_rc(n_m=args.n_m, n_w=args.n_w, chi=chi))
        np.savez(
            fpath,
            mode=args.mode,
            phi=phi,
            theta=theta,
            sample_n=sample_n,
            n_nodes=n_nodes,
            beta=args.beta,
            seeds=np.array(args.seeds, dtype=int),
            steps=args.steps,
            r_values=r_vals,
            rc=rc,
            chi=chi,
            model=args.model,
            **res,
        )
        return f"[saved] {fpath} (rc={rc:.4f}, chi={chi:.4f})"

    messages = Parallel(n_jobs=args.n_jobs)(delayed(worker)(c) for c in combos)
    for msg in messages:
        print(msg)


if __name__ == "__main__":
    main()
