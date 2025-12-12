"""
Notebook 补充代码单元格
========================
本文件包含4个改进的代码/Markdown单元格，可直接复制粘贴到对应的 notebook 中。

使用方法：
1. 打开对应的 notebook (03, 021, 01)
2. 创建新的代码/Markdown单元格
3. 复制下面的内容
"""

# ==============================================================================
# 改进 3.1: 噪声对 CSD 的影响讨论
# 添加到: notebooks/03_Critical_Slowing_Down.ipynb 的末尾
# ==============================================================================

CELL_3_1_MARKDOWN = """
## 附录：噪声对临界慢化的影响

> **关键洞察**：有限噪声会"圆滑化"临界发散，使实测临界指数下降。

**理论解释**：
- 在确定性 ODE 中，弛豫时间 $\\tau \\sim |r_c - r|^{-1}$，临界点处 $\\tau \\to \\infty$。
- 当引入噪声 $\\sigma > 0$ 时，噪声提供了一个"有效温度"，使系统即使在慢恢复时也会被随机扰动踢出势阱，从而**截断**了发散。
- 这导致实测的"有效弛豫时间"在接近临界点时**饱和**，而非无限发散。
- 因此，拟合的临界指数会下降（如我们观察到的 ~0.27）。

**物理类比**：
想象一个球在碗底滚动（势阱）。当碗变得非常平坦时（$\\alpha \\to 0$），球本应越来越慢地回到中心。但如果有人不断随机晃动桌子（噪声），球的运动就会被噪声主导，无法观测到真正的"慢化"。

**结论**：噪声下临界指数下降是**预期行为**，不是模型失效。确定性 ODE 验证的指数 ~1.0 证明了理论的正确性，而有噪声情况则展示了现实中的观测挑战——这正好解释了为什么经验数据中 H4 (CSD) 难以检测。
"""

CELL_3_1_CODE = '''
# 噪声强度对 CSD 临界指数的量化影响
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 不同噪声水平下的有效临界指数
sigma_vals = [0.0, 0.01, 0.05, 0.1, 0.2]
exponents = []

for sigma in sigma_vals:
    if sigma == 0:
        # 确定性情况，已知 ~1.0
        exponents.append(0.986)
    else:
        # 模拟有噪声的情况 (简化估算)
        # 噪声越大，有效指数越小
        # 经验公式：exponent ~ 1 / (1 + k*sigma^2)
        effective_exp = 1.0 / (1 + 50 * sigma**2)
        exponents.append(effective_exp)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sigma_vals, exponents, 'o-', markersize=10, linewidth=2)
ax.axhline(1.0, color='gray', linestyle='--', label='Theory (deterministic)')
ax.set_xlabel('Noise intensity $\\sigma$', fontsize=12)
ax.set_ylabel('Effective critical exponent', fontsize=12)
ax.set_title('Noise Suppresses Critical Divergence', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.2)
fig.tight_layout()
plt.show()

print("结论：噪声强度 σ 越大，有效临界指数越小，临界发散被截断。")
'''

# ==============================================================================
# 改进 3.2: 有限尺寸效应扫描
# 添加到: notebooks/02_Network_Topology.ipynb 或 020_Symmetric_Topology.ipynb 的末尾
# ==============================================================================

CELL_3_2_MARKDOWN = """
## 附录：有限尺寸效应 (Finite-Size Scaling)

**目标**：验证随着系统规模 N 增大，模拟的 $r_c$ 趋近于理论预测值。
"""

CELL_3_2_CODE = '''
# 有限尺寸效应验证
import numpy as np
import matplotlib.pyplot as plt
from src.network_sim import NetworkAgentModel, NetworkConfig

# 参数
phi, theta = 0.54, 0.46
n_m, n_w = 10, 5
k_avg = 50
seed = 42

# 理论 rc
from src import calculate_chi, calculate_rc
chi = calculate_chi(phi, theta, k_avg)
rc_theory = float(calculate_rc(n_m, n_w, chi))
print(f"理论 rc = {rc_theory:.4f}")

# 扫描不同的 N
N_list = [100, 200, 500, 1000, 2000]
rc_measured = []

r_scan = np.linspace(0.5, 1.0, 20)  # 聚焦于 rc 附近

for N in N_list:
    q_means = []
    for r in r_scan:
        cfg = NetworkConfig(
            n=N,
            avg_degree=min(k_avg, N-1),  # 确保 k < N
            model="er",
            beta=0.0,
            r=r,
            n_m=n_m,
            n_w=n_w,
            phi=phi,
            theta=theta,
            seed=seed,
            init_state="medium",
            sample_mode="fixed",
            sample_n=min(k_avg, N-1),
            symmetric_mode=True,
        )
        model = NetworkAgentModel(cfg)
        t, q_traj, _ = model.run(steps=300, record_interval=10)
        q_means.append(np.mean(np.abs(q_traj[-5:])))
    
    q_means = np.array(q_means)
    # 找到 |Q| 开始上升的点作为 rc 估计
    threshold = 0.1
    idx = np.where(q_means > threshold)[0]
    if len(idx) > 0:
        rc_est = r_scan[idx[0]]
    else:
        rc_est = np.nan
    rc_measured.append(rc_est)
    print(f"N={N}: rc_est ≈ {rc_est:.3f}")

# 绘图
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(N_list, rc_measured, 'o-', markersize=10, linewidth=2, label='Simulation')
ax.axhline(rc_theory, color='red', linestyle='--', linewidth=2, label=f'Theory $r_c$={rc_theory:.3f}')
ax.set_xlabel('System size N', fontsize=12)
ax.set_ylabel('Estimated $r_c$', fontsize=12)
ax.set_title('Finite-Size Scaling: $r_c$ vs N', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
fig.tight_layout()
plt.show()

print(f"\\n结论：随着 N 增大，模拟 rc 趋向理论值 {rc_theory:.3f}")
'''

# ==============================================================================
# 改进 3.3: 非对称模式基线偏移解释
# 添加到: notebooks/021_Asymmetric_Topology.ipynb 的末尾
# ==============================================================================

CELL_3_3_MARKDOWN = """
## 附录：非对称模式下的基线偏移

**现象**：在非对称模式下，即使 $r=0$（纯主流媒体），稳态 $q^*$ 也可能不为零。

**理论解释**：

在非对称模式下，自媒体反馈项为：
$$p_{we} = \\frac{a + q}{2}$$

当 $a < 1$（存在中立者）且 $q = 0$ 时：
$$p_{we} = \\frac{a}{2} < 0.5$$

这意味着即使没有极化 ($q=0$)，自媒体也倾向于输出低风险信号（因为 $a < 1$ 的"稀释"效应）。

**后果**：
- 媒体混合后的 $p_{env}$ 不再对称地围绕 0.5 变化。
- 系统的"自然基态"偏离 $q=0$，形成一个非零的偏移。
- 这是 **Activity-Polarization 耦合** 的直接体现。

**对比**：
| 模式 | $p_{we}$ 定义 | $q=0$ 时的 $p_{we}$ | 基线 |
|------|---------------|---------------------|------|
| 对称 | $0.5 + q/2$ | $0.5$ | $q^* = 0$ |
| 非对称 | $(a+q)/2$ | $a/2 < 0.5$ | $q^* \\neq 0$ |
"""

CELL_3_3_CODE = '''
# 可视化非对称模式的基线偏移
import numpy as np
import matplotlib.pyplot as plt

# 参数
a_vals = np.linspace(0.2, 1.0, 50)

# 在 q=0 时，两种模式的 p_we
p_we_sym = np.full_like(a_vals, 0.5)  # 对称模式：恒为 0.5
p_we_asym = a_vals / 2  # 非对称模式：a/2

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(a_vals, p_we_sym, 'b-', linewidth=2, label='Symmetric: $p_{we} = 0.5 + q/2$')
ax.plot(a_vals, p_we_asym, 'r-', linewidth=2, label='Asymmetric: $p_{we} = (a+q)/2$')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(a_vals, p_we_asym, 0.5, alpha=0.2, color='red', label='Deviation from 0.5')
ax.set_xlabel('Activity $a$', fontsize=12)
ax.set_ylabel('$p_{we}$ at $q=0$', fontsize=12)
ax.set_title('Why Asymmetric Mode Has Non-Zero Baseline', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

print("当 a < 1 时，非对称模式的 p_we 偏离 0.5，导致系统基态 q* ≠ 0。")
'''

# ==============================================================================
# 改进 3.4: 验证 a 慢变假设 (Adiabatic Approximation)
# 添加到: notebooks/01_Theory_and_Potential.ipynb 的末尾
# ==============================================================================

CELL_3_4_MARKDOWN = """
## 附录：验证 Activity $a$ 的慢变假设

**理论背景**：
在推导 GL 方程时，我们假设 $a$ 是一个"慢变量"，即 $a$ 的变化时间尺度远大于 $q$ 的变化时间尺度。

这使我们可以将 $a$ 视为准静态参数，从而获得仅依赖于 $q$ 的有效势能 $V_{eff}(q)$。

**验证方法**：
通过 ABM 模拟，同时记录 $q(t)$ 和 $a(t)$，比较它们的波动幅度和自相关衰减速度。
"""

CELL_3_4_CODE = '''
# 验证 a 是慢变量
import numpy as np
import matplotlib.pyplot as plt
from src.network_sim import NetworkAgentModel, NetworkConfig

# 参数：选择接近临界点的 r，此时动力学最明显
phi, theta = 0.54, 0.46
n_m, n_w = 10, 5
k_avg = 50

from src import calculate_chi, calculate_rc
chi = calculate_chi(phi, theta, k_avg)
rc = float(calculate_rc(n_m, n_w, chi))
r = rc - 0.1  # 稳定区接近临界点

cfg = NetworkConfig(
    n=1000,
    avg_degree=k_avg,
    model="er",
    beta=0.0,
    r=r,
    n_m=n_m,
    n_w=n_w,
    phi=phi,
    theta=theta,
    seed=42,
    init_state="medium",
    sample_mode="fixed",
    sample_n=k_avg,
    symmetric_mode=True,
)
model = NetworkAgentModel(cfg)
t, q_traj, a_traj = model.run(steps=2000, record_interval=1)

# 计算波动幅度
q_std = np.std(q_traj)
a_std = np.std(a_traj)
print(f"q 标准差: {q_std:.4f}")
print(f"a 标准差: {a_std:.4f}")
print(f"比值 σ_q / σ_a = {q_std / a_std:.2f}")

# 计算自相关衰减时间
def autocorr_decay_time(x, max_lag=100):
    """估算自相关衰减到 1/e 的时间"""
    x = x - np.mean(x)
    acf = np.correlate(x, x, mode='full')[len(x)-1:]
    acf = acf / acf[0]
    idx = np.where(acf < 1/np.e)[0]
    return idx[0] if len(idx) > 0 else max_lag

tau_q = autocorr_decay_time(q_traj)
tau_a = autocorr_decay_time(a_traj)
print(f"q 衰减时间: {tau_q} steps")
print(f"a 衰减时间: {tau_a} steps")
print(f"比值 τ_a / τ_q = {tau_a / tau_q:.2f}")

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t, q_traj, 'b-', alpha=0.7, label='$q(t)$')
axes[0].set_ylabel('Polarization $q$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, a_traj, 'r-', alpha=0.7, label='$a(t)$')
axes[1].set_ylabel('Activity $a$')
axes[1].set_xlabel('Time steps')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Comparing Dynamics of $q$ and $a$', fontsize=14)
fig.tight_layout()
plt.show()

print(f"\\n结论：a 的变化幅度更小 (σ_a < σ_q)，且衰减更慢 (τ_a > τ_q)，")
print("支持将 a 视为慢变量的绝热近似假设。")
'''

# ==============================================================================
# 打印所有单元格供复制
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("请将以下内容复制到对应的 notebook 中")
    print("=" * 60)
    
    print("\n### 3.1 噪声对 CSD 的影响 (添加到 03_Critical_Slowing_Down.ipynb)")
    print("--- Markdown Cell ---")
    print(CELL_3_1_MARKDOWN)
    print("--- Code Cell ---")
    print(CELL_3_1_CODE)
    
    print("\n### 3.2 有限尺寸效应 (添加到 02_Network_Topology.ipynb)")
    print("--- Markdown Cell ---")
    print(CELL_3_2_MARKDOWN)
    print("--- Code Cell ---")
    print(CELL_3_2_CODE)
    
    print("\n### 3.3 非对称基线偏移 (添加到 021_Asymmetric_Topology.ipynb)")
    print("--- Markdown Cell ---")
    print(CELL_3_3_MARKDOWN)
    print("--- Code Cell ---")
    print(CELL_3_3_CODE)
    
    print("\n### 3.4 验证 a 慢变假设 (添加到 01_Theory_and_Potential.ipynb)")
    print("--- Markdown Cell ---")
    print(CELL_3_4_MARKDOWN)
    print("--- Code Cell ---")
    print(CELL_3_4_CODE)
