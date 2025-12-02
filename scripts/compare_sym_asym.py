"""比较对称与非对称模型的相变特征"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path('outputs/data')
fig_dir = Path('outputs/figs')

# 加载数据
sym = np.load(data_dir / 'rq_a_scan_sym_k50_samp50_beta0.0_seeds3_steps300.npz')
asym = np.load(data_dir / 'rq_a_scan_asym_k50_samp50_beta0.0_seeds3_steps300.npz')

r = sym['r_scan']
q_sym = sym['abs_mean']
q_asym = asym['abs_mean']
a_sym = sym['a_mean']
a_asym = asym['a_mean']

# 理论 r_c
rc = 0.7533

print("=" * 60)
print("      对称 vs 非对称模型：相变特征比较")
print("=" * 60)

print("\n--- |Q| 在临界点附近的行为 ---")
print(f"{'r':>8} {'|Q|_sym':>12} {'|Q|_asym':>12} {'差值':>10}")
print("-" * 45)
for i in range(len(r)):
    if 0.55 <= r[i] <= 1.0:
        diff = q_asym[i] - q_sym[i]
        marker = " <-- r_c" if abs(r[i] - rc) < 0.03 else ""
        print(f"{r[i]:8.3f} {q_sym[i]:12.4f} {q_asym[i]:12.4f} {diff:10.4f}{marker}")

print("\n--- Activity A 在临界点附近的行为 ---")
print(f"{'r':>8} {'A_sym':>12} {'A_asym':>12} {'差值':>10}")
print("-" * 45)
for i in range(len(r)):
    if 0.55 <= r[i] <= 1.0:
        diff = a_asym[i] - a_sym[i]
        marker = " <-- r_c" if abs(r[i] - rc) < 0.03 else ""
        print(f"{r[i]:8.3f} {a_sym[i]:12.4f} {a_asym[i]:12.4f} {diff:10.4f}{marker}")

# 计算导数（斜率）来判断相变陡峭程度
def numerical_derivative(y, x):
    return np.gradient(y, x)

dq_sym = numerical_derivative(q_sym, r)
dq_asym = numerical_derivative(q_asym, r)

# 找到最大斜率位置
idx_max_sym = np.argmax(np.abs(dq_sym))
idx_max_asym = np.argmax(np.abs(dq_asym))

print("\n" + "=" * 60)
print("                相变特征分析")
print("=" * 60)
print(f"\n对称模型:")
print(f"  最大斜率位置: r = {r[idx_max_sym]:.3f}")
print(f"  最大斜率值:   d|Q|/dr = {dq_sym[idx_max_sym]:.3f}")

print(f"\n非对称模型:")
print(f"  最大斜率位置: r = {r[idx_max_asym]:.3f}")
print(f"  最大斜率值:   d|Q|/dr = {dq_asym[idx_max_asym]:.3f}")

print(f"\n理论临界点 r_c = {rc:.3f}")

# 绘制对比图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：|Q| 对比
ax1 = axes[0]
ax1.plot(r, q_sym, 'o-', color='tab:blue', linewidth=2, markersize=6, label='Symmetric')
ax1.plot(r, q_asym, 's--', color='tab:red', linewidth=2, markersize=6, label='Asymmetric')
ax1.axvline(rc, color='gray', linestyle=':', linewidth=2, label=f'$r_c$={rc:.3f}')
ax1.set_xlabel('Control Parameter $r$', fontsize=12)
ax1.set_ylabel('$|Q|$', fontsize=12)
ax1.set_title('Symmetric vs Asymmetric: $|Q|$ Bifurcation', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：导数对比
ax2 = axes[1]
ax2.plot(r, dq_sym, 'o-', color='tab:blue', linewidth=2, markersize=6, label='Symmetric')
ax2.plot(r, dq_asym, 's--', color='tab:red', linewidth=2, markersize=6, label='Asymmetric')
ax2.axvline(rc, color='gray', linestyle=':', linewidth=2, label=f'$r_c$={rc:.3f}')
ax2.set_xlabel('Control Parameter $r$', fontsize=12)
ax2.set_ylabel('$d|Q|/dr$', fontsize=12)
ax2.set_title('Rate of Change (Transition Sharpness)', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(fig_dir / 'fig_sym_vs_asym_comparison.png', dpi=200)
print(f"\n图已保存: {fig_dir / 'fig_sym_vs_asym_comparison.png'}")

# 判断相变类型
print("\n" + "=" * 60)
print("                结论")
print("=" * 60)
ratio = dq_asym[idx_max_asym] / dq_sym[idx_max_sym] if dq_sym[idx_max_sym] != 0 else 0
print(f"\n斜率比值 (asym/sym): {ratio:.3f}")

if abs(ratio - 1.0) < 0.2:
    print("\n⚠️  结论：两模型的相变陡峭程度相近！")
    print("    当前参数下（φ=0.54, θ=0.46），非对称模型也呈现陡峭的相变。")
    print("    这可能是因为参数接近对称条件（φ + θ = 1）。")
else:
    if ratio < 1:
        print("\n✓  结论：非对称模型的相变更平缓（渐变特征）")
    else:
        print("\n✓  结论：非对称模型的相变更陡峭")

