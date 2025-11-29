"""快速检查：当心理敏感度 chi 很大时，rc 是否小于 1。"""

import sys
from pathlib import Path

# 将项目根目录加入搜索路径，便于脚本直接执行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402


def main() -> None:
    phi, theta, k_avg = 0.6, 0.4, 50
    n_m, n_w = 10, 5
    chi = theory.calculate_chi(phi=phi, theta=theta, k_avg=k_avg)
    rc = theory.calculate_rc(n_m=n_m, n_w=n_w, chi=chi)
    print(f"phi={phi}, theta={theta}, k_avg={k_avg}")
    print(f"n_m={n_m}, n_w={n_w}")
    print(f"chi={chi:.4f}, rc={rc:.4f} (预期 rc<1)")


if __name__ == "__main__":
    main()
