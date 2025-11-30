"""
Phase 3 验收脚本：
1) beta=0 对称阈值，扫描 r，与理论 rc 对比。
2) 不对称阈值演示：phi != 1 - theta，自发偏移。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calculate_chi, calculate_rc, network_sim  # noqa: E402


def run_theory_rc(phi: float, theta: float, k_avg: int, n_m: float, n_w: float) -> float:
    chi = calculate_chi(phi=phi, theta=theta, k_avg=k_avg)
    rc = float(calculate_rc(n_m=n_m, n_w=n_w, chi=chi))
    return rc


def simulate_mean_q(r: float, phi: float, theta: float, beta: float, seed: int) -> float:
    cfg = network_sim.NetworkConfig(
        n=150,
        avg_degree=6,
        model="ba",
        beta=beta,
        r=r,
        phi=phi,
        theta=theta,
        seed=seed,
    )
    model = network_sim.NetworkAgentModel(cfg)
    _, q, _ = model.run(steps=80, record_interval=10)
    return float(np.mean(q[-3:]))  # 末尾若干步平均


def validation_symmetry(output: Path) -> None:
    phi, theta, k_avg = 0.6, 0.4, 50
    n_m, n_w = 10, 5
    rc = run_theory_rc(phi, theta, k_avg, n_m, n_w)
    r_list = [rc - 0.15, rc, rc + 0.15]
    seeds = [0, 1, 2]
    rows = []
    for r in r_list:
        qs = [simulate_mean_q(r, phi, theta, beta=0.0, seed=s) for s in seeds]
        rows.append((r, np.mean(qs)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("r,mean_q\n")
        for r, q_mean in rows:
            f.write(f"{r:.4f},{q_mean:.6f}\n")
    print(f"[Phase3] beta=0 对称阈值，rc={rc:.3f}，结果写入 {output}")


def validation_asymmetry(output: Path) -> None:
    phi, theta = 0.7, 0.25  # 不对称
    r = 0.5
    seeds = [0, 1, 2]
    qs = [simulate_mean_q(r, phi, theta, beta=0.0, seed=s) for s in seeds]
    q_mean = np.mean(qs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("r,phi,theta,mean_q\n")
        f.write(f"{r:.3f},{phi:.3f},{theta:.3f},{q_mean:.6f}\n")
    print(f"[Phase3] 不对称阈值示例 mean_q={q_mean:.3f}，结果写入 {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase3 验收脚本：网络模拟对理论 rc 与不对称偏移的验证")
    parser.add_argument("--mode", choices=["sym", "asym", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("sym", "all"):
        validation_symmetry(data_dir / "network_symmetry.csv")
    if args.mode in ("asym", "all"):
        validation_asymmetry(data_dir / "network_asymmetry.csv")


if __name__ == "__main__":
    main()
