"""验证确定性ODE的临界指数"""
import numpy as np
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, '.')
from src.theory import get_gl_params, calculate_chi, calculate_rc

# 参数
phi, theta, k_avg = 0.54, 0.46, 50
n_m, n_w = 10, 5
chi = calculate_chi(phi, theta, k_avg)
rc = float(calculate_rc(n_m, n_w, chi))
print(f"chi={chi:.4f}, rc={rc:.4f}")

def measure_relaxation_time(r, q0=0.05, dt=1e-4, max_steps=300000):
    """确定性ODE测量弛豫时间"""
    alpha, u = get_gl_params(r=r, rc=rc)
    alpha = float(alpha)
    if alpha <= 0:
        return np.nan
    q = q0
    target = q0 / np.e
    for step in range(max_steps):
        dq = (-alpha * q - u * q**3) * dt
        q += dq
        if abs(q) <= target:
            return step * dt
    return max_steps * dt

# 对数间隔采样
distances = np.logspace(-3, -0.3, 20)
r_vals = rc - distances
r_vals = r_vals[r_vals > 0]

print("\n测量弛豫时间...")
tau_measured = np.array([measure_relaxation_time(r) for r in r_vals])
tau_theory = 1.0 / (rc - r_vals)

# 过滤有效数据
valid = np.isfinite(tau_measured) & (tau_measured > 0) & (tau_measured < 30)
log_dist = np.log(rc - r_vals[valid])
log_tau = np.log(tau_measured[valid])
log_tau_th = np.log(tau_theory[valid])

# 拟合
def linear_func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear_func, log_dist, log_tau)
slope, intercept = popt
exponent = -slope
err = np.sqrt(pcov[0, 0])

print("\n" + "=" * 50)
print("         确定性ODE临界指数验证结果")
print("=" * 50)
print(f"  拟合临界指数: {exponent:.4f} ± {err:.4f}")
print(f"  理论预期指数: 1.0000")
print(f"  相对误差:     {abs(exponent - 1.0) * 100:.2f}%")
print("=" * 50)

# 打印部分数据对比
print("\nτ_measured vs τ_theory 对比:")
print(f"{'r':>8} {'τ_sim':>10} {'τ_theory':>10} {'ratio':>8}")
print("-" * 40)
for i in range(0, len(r_vals), 3):
    if valid[i]:
        ratio = tau_measured[i] / tau_theory[i]
        print(f"{r_vals[i]:8.4f} {tau_measured[i]:10.4f} {tau_theory[i]:10.4f} {ratio:8.4f}")

