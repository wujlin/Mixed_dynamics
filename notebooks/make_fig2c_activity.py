"""
Fig 2c：Order parameters in ABM（symmetric vs activity-coupled）——论文版式统一导出 PDF。

目标（对应 reviewer 指出的问题）：
- 旧版 fig2c_activity_* 为 PNG，字体/网格/legend 风格与 Fig2a/b 不一致。
- 这里重新用统一的 paper style 输出矢量 PDF，并避免图例遮挡数据。
- 同时补齐与正文一致的叙事：在同一面板中同时展示稳态极化幅度 |Q| 与 activity a 随 r 的变化。

运行：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig2c_activity.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import NetworkAgentModel, NetworkConfig, theory  # noqa: E402
from src.plot_style import OKABE_ITO, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Fig2cConfig:
    # ABM 口径：对齐 notebooks/02_Network_Topology.ipynb 的基准验证设置
    n: int = 500
    network_model: str = "er"
    avg_degree: int = 50
    sample_mode: str = "fixed"
    sample_n: int = 50
    beta: float = 0.0
    update_rate: float = 0.1
    init_state: str = "medium"
    record_interval: int = 10
    steps: int = 300
    burn_in_frac: float = 0.5
    # 说明：Fig 2 的核心信息之一是“ABM 结果的统计不确定性”，因此这里默认用更多 seeds。
    seeds: Tuple[int, ...] = tuple(range(50))

    sym_path: Path = ROOT / "outputs" / "data" / "rq_a_scan_sym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    asym_path: Path = ROOT / "outputs" / "data" / "rq_a_scan_asym_N500_er_k50_fixed50_beta0.0_u10_ri10_burn50_seeds10_steps300_v3.npz"
    phi: float = 0.54
    theta: float = 0.46
    n_m: float = 10.0
    n_w: float = 5.0
    k_avg: int = 50
    # 稍微增加高度，为“共享 x-label + 轴外 legend”留出稳定空间
    fig_size: Tuple[float, float] = (6.5, 2.7)


def _bootstrap_ci_mean(
    rng: np.random.Generator,
    samples: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("samples 必须是一维数组")
    if samples.size == 0:
        return float("nan"), float("nan")
    if samples.size == 1:
        v = float(samples[0])
        return v, v
    idx = rng.integers(0, samples.size, size=(int(n_bootstrap), samples.size))
    boot_means = samples[idx].mean(axis=1)
    alpha = 1.0 - float(ci_level)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _seed_cache_path(cfg: Fig2cConfig, mean_path: Path) -> Path:
    """从 v3 mean 缓存路径推断 seed-level v4 缓存路径。

    注意：mean 缓存文件名里通常包含 seedsXX；这里会同步更新到当前 cfg.seeds 的长度，
    避免“文件名写 seeds10 但内容其实是 seeds50”的歧义。
    """

    name = mean_path.name
    name = re.sub(r"seeds\d+", f"seeds{len(cfg.seeds)}", name)
    name = name.replace("v3.npz", "v4.npz")
    if name == mean_path.name:
        raise ValueError(f"无法从文件名推断 seed 缓存版本：{mean_path.name}")
    return mean_path.with_name(name)


def _load_or_simulate_by_seed(cfg: Fig2cConfig, *, symmetric_mode: bool, mean_path: Path) -> dict[str, np.ndarray]:
    seed_path = _seed_cache_path(cfg, mean_path)
    if seed_path.exists():
        d = np.load(seed_path, allow_pickle=False)
        required = {"r_scan", "q_abs_by_seed", "a_by_seed"}
        missing = required.difference(set(d.files))
        if missing:
            raise KeyError(f"{seed_path} 缺少字段：{sorted(missing)}，现有：{d.files}")
        return {k: d[k].astype(float, copy=False) for k in required}

    # 用 v3 的 r_scan 保持完全一致
    mean_data = np.load(mean_path, allow_pickle=False)
    r_scan = mean_data["r_scan"].astype(float, copy=False)

    q_abs_by_seed = np.zeros((r_scan.size, len(cfg.seeds)), dtype=float)
    a_by_seed = np.zeros((r_scan.size, len(cfg.seeds)), dtype=float)
    burn_step = float(cfg.steps) * float(cfg.burn_in_frac)

    for i, r in enumerate(r_scan):
        for j, seed in enumerate(cfg.seeds):
            net_cfg = NetworkConfig(
                n=int(cfg.n),
                avg_degree=float(cfg.avg_degree),
                model=str(cfg.network_model),
                beta=float(cfg.beta),
                update_rate=float(cfg.update_rate),
                init_state=str(cfg.init_state),
                sample_mode=str(cfg.sample_mode),
                sample_n=int(cfg.sample_n),
                symmetric_mode=bool(symmetric_mode),
                r=float(r),
                n_m=float(cfg.n_m),
                n_w=float(cfg.n_w),
                phi=float(cfg.phi),
                theta=float(cfg.theta),
                seed=int(seed),
            )
            model = NetworkAgentModel(net_cfg)
            t, q_traj, a_traj = model.run(steps=int(cfg.steps), record_interval=int(cfg.record_interval))
            steady = t >= burn_step
            steady_q = q_traj[steady]
            steady_a = a_traj[steady]

            q_mean = float(np.mean(steady_q)) if steady_q.size else float(np.mean(q_traj))
            a_mean = float(np.mean(steady_a)) if steady_a.size else float(np.mean(a_traj))

            q_abs_by_seed[i, j] = abs(q_mean)
            a_by_seed[i, j] = a_mean

    # 兼容 v3 的字段（方便其他脚本未来切换到 v4）
    abs_mean = q_abs_by_seed.mean(axis=1)
    a_mean = a_by_seed.mean(axis=1)

    seed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        seed_path,
        r_scan=r_scan,
        q_abs_by_seed=q_abs_by_seed,
        a_by_seed=a_by_seed,
        abs_mean=abs_mean,
        a_mean=a_mean,
    )
    return {"r_scan": r_scan, "q_abs_by_seed": q_abs_by_seed, "a_by_seed": a_by_seed}


def _summarize_with_ci(
    d: dict[str, np.ndarray],
    *,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = d["r_scan"].astype(float, copy=False)
    q_by_seed = d["q_abs_by_seed"].astype(float, copy=False)
    a_by_seed = d["a_by_seed"].astype(float, copy=False)

    q_mean = q_by_seed.mean(axis=1)
    a_mean = a_by_seed.mean(axis=1)

    rng = np.random.default_rng(0)
    q_ci = np.zeros((r.size, 2), dtype=float)
    a_ci = np.zeros((r.size, 2), dtype=float)
    for i in range(r.size):
        q_ci[i, 0], q_ci[i, 1] = _bootstrap_ci_mean(
            rng,
            q_by_seed[i],
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
        )
        a_ci[i, 0], a_ci[i, 1] = _bootstrap_ci_mean(
            rng,
            a_by_seed[i],
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
        )

    return r, q_mean, q_ci, a_mean, a_ci


def main() -> None:
    # 避免多线程 BLAS 在受限环境下的噪声告警；不影响数值正确性
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = Fig2cConfig()
    if not cfg.sym_path.exists():
        raise FileNotFoundError(f"未找到 activity(sym) 缓存：{cfg.sym_path}")
    if not cfg.asym_path.exists():
        raise FileNotFoundError(f"未找到 activity(asym) 缓存：{cfg.asym_path}")

    sym_seed = _load_or_simulate_by_seed(cfg, symmetric_mode=True, mean_path=cfg.sym_path)
    asym_seed = _load_or_simulate_by_seed(cfg, symmetric_mode=False, mean_path=cfg.asym_path)

    r_sym, q_sym, q_ci_sym, a_sym, a_ci_sym = _summarize_with_ci(sym_seed)
    r_asym, q_asym, q_ci_asym, a_asym, a_ci_asym = _summarize_with_ci(asym_seed)

    chi = float(theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=int(cfg.k_avg)))
    rc = float(theory.calculate_rc(n_m=float(cfg.n_m), n_w=float(cfg.n_w), chi=chi))

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=cfg.fig_size, sharey=True)

    q_color = OKABE_ITO["blue"]
    activity_color = OKABE_ITO["bluish_green"]
    rc_style = dict(color="gray", linestyle=":", linewidth=1.2, alpha=0.6)

    for ax, r, a, q_abs, title in [
        (axes[0], r_sym, a_sym, q_sym, "Symmetric"),
        (axes[1], r_asym, a_asym, q_asym, "Activity-coupled"),
    ]:
        if title == "Symmetric":
            q_ci = q_ci_sym
            a_ci = a_ci_sym
        else:
            q_ci = q_ci_asym
            a_ci = a_ci_asym

        # 95% CI band across seeds (bootstrap over mean).
        # 说明：该图可能出现“CI 很窄不易察觉”的情况，因此 alpha 稍高于 Fig2a，确保印刷可见。
        # 该面板在多数 r 区间 CI 较窄，需提高 band 可见度以避免“画了但看不见”。
        # 为印刷尺度增强可见性：提高透明带 alpha，并给 band 一个极细边界线（同色、低 alpha）。
        ax.fill_between(
            r,
            q_ci[:, 0],
            q_ci[:, 1],
            color=q_color,
            alpha=0.38,
            linewidth=0.4,
            edgecolor=q_color,
            zorder=1,
        )
        ax.fill_between(
            r,
            a_ci[:, 0],
            a_ci[:, 1],
            color=activity_color,
            alpha=0.30,
            linewidth=0.4,
            edgecolor=activity_color,
            zorder=1,
        )

        ax.plot(r, q_abs, color=q_color, linewidth=2.2, zorder=3, label=r"Polarization $|Q|$")
        ax.plot(
            r,
            a,
            color=activity_color,
            marker="s",
            linewidth=2.0,
            markersize=4.0,
            zorder=4,
            label=r"Activity $a$",
        )
        ax.axvline(rc, **rc_style, zorder=1)
        ax.set_title(title)  # 使用全局 titlesize (11pt)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(direction="in", top=True, right=True)

    axes[0].set_ylabel("Order parameters")
    # 共享 x-label，避免每个子图都占用垂直空间，确保 legend 放在最下方仍不拥挤
    fig.supxlabel(r"Control parameter $r$", y=0.15)

    # Panel label for the whole subfigure block
    add_panel_label(axes[0], "c", dx=-55.0)

    # 图例放在 x 轴标题下方，避免在图内堆叠文字
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.30, top=0.86, wspace=0.18)

    out_pdf = ROOT / "Essay" / "figures" / "fig2c_activity.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    out_png = ROOT / "outputs" / "figs" / "fig2" / "fig2c_activity_preview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
