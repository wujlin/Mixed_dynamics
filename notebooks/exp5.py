# ... existing imports ...

def experiment_5_network_size_scaling():
    """
    图 5: 网络规模效应 (Finite-Size Scaling)
    探究不同网络规模 N 下，相变点是否有明显漂移。
    目标：解释为什么 Fig3 中网络模拟比理论相变得早很多。
    """
    print("Running Experiment 5: Network Size Scaling...")
    
    # 基础参数
    phi, theta, k_avg = 0.6, 0.4, 6
    n_m, n_w = 10, 5
    
    # 定义不同的网络规模
    network_sizes = [200, 500, 1000] # 可以试着跑更大，如 2000，但这取决于你的电脑性能
    colors = ['lightblue', 'dodgerblue', 'navy']
    
    # 扫描 r 的范围
    r_scan = np.linspace(0, 1.0, 21)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for N, color in zip(network_sizes, colors):
        q_mean_list = []
        
        # 对每个 N 跑一轮 r 扫描
        print(f"  Simulating N={N}...")
        for r in r_scan:
            # 每个 r 跑 3 次取平均，减少随机性
            q_trials = []
            for seed in range(3):
                cfg = NetworkConfig(
                    n=N, avg_degree=k_avg, model="ba", beta=0.0, # 暂时保持 beta=0 纯验证
                    r=r, n_m=n_m, n_w=n_w, phi=phi, theta=theta, seed=seed
                )
                model = NetworkAgentModel(cfg)
                # 稍微增加步数以保证大网络也能达到稳态
                _, q_traj, _ = model.run(steps=300, record_interval=10)
                q_trials.append(np.mean(np.abs(q_traj[-10:])))
            
            q_mean_list.append(np.mean(q_trials))
            
        ax.plot(r_scan, q_mean_list, 'o-', color=color, label=f'BA Network (N={N})', alpha=0.8)

    # 画上平均场理论参考线
    chi = theory.calculate_chi(phi, theta, k_avg)
    rc_theory = theory.calculate_rc(n_m, n_w, chi)
    ax.axvline(rc_theory, color='gray', linestyle='--', label=f'Mean-Field $r_c$ ({rc_theory:.2f})')
    
    ax.set_xlabel('Control Parameter $r$')
    ax.set_ylabel('Steady State Polarization $|q|$')
    ax.set_title('Finite-Size Effects on Phase Transition')
    ax.legend()
    ax.grid(True)
    
    plt.savefig(OUTPUT_DIR / "Fig5_Size_Scaling.png", dpi=300)
    plt.close()
    print("  -> Saved Fig5")

# ... existing code ...
# 在 if __name__ == "__main__": 中加入 experiment_5_network_size_