import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保能导入 src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import theory
from src.network_sim import NetworkAgentModel, NetworkConfig

# 理论参数
phi, theta = 0.6, 0.4
n_m, n_w = 10, 5
beta_fixed = 0.0
r_scan = np.linspace(0, 1.0, 8)
seeds = [0]
steps = 200

# 注意：全连接 N 非常耗内存，已将超大 N 缩减到可运行范围
configs = [
    {"N": 500, "k": 499, "label": "Full Mesh (N=500)", "fmt": "o-", "color": "tab:red"},
    {"N": 2000, "k": 1999, "label": "Full Mesh (N=2000)", "fmt": "s-", "color": "darkred"},
    {"N": 5000, "k": 4999, "label": "Full Mesh (N=5000)", "fmt": "^-", "color": "maroon"},
]

def run_one(r: float, cfg: dict, steps: int = steps) -> tuple[float, float]:
    """单个 r 的模拟，返回 (r, Abs(Mean(Q_last_window)))."""
    qs = []
    for seed in seeds:
        net_cfg = NetworkConfig(
            n=cfg["N"],
            avg_degree=cfg["k"],
            model="er",  # ER 退化为全连接时等价使用大 k
            beta=beta_fixed,
            r=r,
            n_m=n_m,
            n_w=n_w,
            phi=phi,
            theta=theta,
            seed=seed,
        )
        model = NetworkAgentModel(net_cfg)
        _, q_traj, _ = model.run(steps=steps, record_interval=50)
        qs.append(np.abs(np.mean(q_traj[-5:])))
    return r, float(np.mean(qs))

def main() -> None:
    chi = theory.calculate_chi(phi, theta, k_avg=50)
    rc_ref = float(theory.calculate_rc(n_m, n_w, chi))
    print(f"Theory rc ≈ {rc_ref:.3f} (chi@k=50={chi:.3f})")

    plt.figure(figsize=(10, 6))
    plt.axvline(rc_ref, color="gray", linestyle=":", linewidth=2, label="Theory $r_c$")
    plt.hlines(0, 0, rc_ref, colors="gray", linestyles=":", linewidth=1)

    for cfg in configs:
        print(f"Simulating {cfg['label']} ...")
        q_final = np.zeros_like(r_scan, dtype=float)
        with ProcessPoolExecutor() as ex:
            futures = {ex.submit(run_one, float(r), cfg): idx for idx, r in enumerate(r_scan)}
            for fut in as_completed(futures):
                idx = futures[fut]
                r_val, q_val = fut.result()
                q_final[idx] = q_val
        plt.plot(r_scan, q_final, cfg["fmt"], color=cfg["color"], linewidth=2, label=cfg["label"])

    plt.xlabel("Control Parameter $r$")
    plt.ylabel("Abs(Mean) Polarization $|\\langle Q \\rangle|")
    plt.title("Approaching the Thermodynamic Limit: Finite-Size Scaling (Multiprocess)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "full_mesh_multiproc.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
