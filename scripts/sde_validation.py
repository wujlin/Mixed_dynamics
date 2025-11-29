"""
Phase 2 验收脚本：
1) 计算 chi/rc，设置 r>rc（双稳态）运行 SDE。
2) 输出稳态样本直方图与解析 PDF 的 L1 差异并可选保存图像。
3) 扫描 r 绘制/输出分岔数据（均值 q_vs_r），保存 CSV。
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    SDEConfig,
    calculate_chi,
    calculate_rc,
    get_gl_params,
    run_sde_simulation,
    theoretical_pdf,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_hist_pdf(output_dir: Path) -> None:
    phi, theta, k_avg = 0.6, 0.4, 50
    n_m, n_w = 10, 5
    chi = calculate_chi(phi=phi, theta=theta, k_avg=k_avg)
    rc = float(calculate_rc(n_m=n_m, n_w=n_w, chi=chi))
    r = rc + 0.1  # 大于 rc，形成双稳态
    alpha, u = get_gl_params(r=r, rc=rc)
    cfg = SDEConfig(alpha=float(alpha), u=u, sigma=0.2, dt=1e-2, steps=20000, n_trajectories=8, seed=0)

    # 运行模拟，丢弃前半段作为 burn-in
    _, q_traj = run_sde_simulation(cfg, q0=0.0, record_interval=50)
    burn = q_traj.shape[0] // 2
    samples = q_traj[burn:].ravel()

    bins = np.linspace(-2.5, 2.5, 101)
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = theoretical_pdf(centers, alpha=alpha, u=u, sigma=cfg.sigma)
    l1 = np.trapezoid(np.abs(hist - pdf), centers)

    print(f"[Phase2] chi={chi:.3f}, rc={rc:.3f}, r={r:.3f}, alpha={alpha:.3f}, L1(hist, pdf)={l1:.4f}")

    ensure_dir(output_dir)
    fig, ax = plt.subplots()
    ax.bar(centers, hist, width=centers[1]-centers[0], alpha=0.5, label="SDE histogram")
    ax.plot(centers, pdf, "r-", lw=2, label="theoretical pdf")
    ax.set_xlabel("q")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sde_hist_pdf.png", dpi=200)
    plt.close(fig)


def run_bifurcation_scan(output_dir: Path) -> None:
    phi, theta, k_avg = 0.6, 0.4, 50
    n_m, n_w = 10, 5
    chi = calculate_chi(phi=phi, theta=theta, k_avg=k_avg)
    rc = float(calculate_rc(n_m=n_m, n_w=n_w, chi=chi))
    r_list = np.linspace(0.0, 1.2, 13)
    results = []
    for r in r_list:
        alpha, u = get_gl_params(r=r, rc=rc)
        cfg = SDEConfig(alpha=float(alpha), u=u, sigma=0.15, dt=1e-2, steps=12000, n_trajectories=6, seed=42)
        _, traj = run_sde_simulation(cfg, q0=0.0, record_interval=100)
        steady = traj[-200:].mean(axis=(0, 1))  # 末尾平均
        results.append((r, steady))

    ensure_dir(output_dir)
    csv_path = output_dir / "sde_bifurcation.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("r,steady_q\n")
        for r, q_val in results:
            f.write(f"{r:.4f},{q_val:.6f}\n")
    print(f"[Phase2] 分岔扫描完成，rc={rc:.3f}，结果写入 {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase2 验收：SDE 直方图与分岔扫描")
    parser.add_argument("--mode", choices=["hist", "bifurcation", "all"], default="all")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()
    out_dir = Path(args.output)
    if args.mode in ("hist", "all"):
        run_hist_pdf(out_dir)
    if args.mode in ("bifurcation", "all"):
        run_bifurcation_scan(out_dir)


if __name__ == "__main__":
    main()
