"""
Fig 2a：Network validation（ER vs BA）+ 95% CI + 理论分支（拟合缩放）+ PDF 导出。

设计目标：
- 科学严谨：同图展示 ER 与 BA，且给出统计不确定性（95% CI）。
- 叙事一致：对称模式下展示 ±|<Q>| 的 pitchfork 形态，并标注理论 r_c。
- 期刊排版：统一衬线字体（Times-like）、加粗线宽、去网格，导出矢量 PDF。

运行方式（使用你已有的 conda 环境 python 路径）：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_fig2a_network_validation.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory, NetworkAgentModel, NetworkConfig  # noqa: E402


@dataclass(frozen=True)
class Fig2aConfig:
    # 模型参数
    n: int = 500
    avg_degree: int = 50
    beta: float = 0.0
    update_rate: float = 0.1
    init_state: str = "medium"
    symmetric_mode: bool = True
    sample_mode: str = "degree"  # "degree" 使 ER/BA 拓扑差异进入动力学；"fixed" 则几乎重合
    sample_n: int = 50  # sample_mode="fixed" 时才生效
    local_mode: str = "high_only"  # beta=0 时不影响；若将来 beta>0，建议改为 "symmetric"

    # 理论参数
    phi: float = 0.54
    theta: float = 0.46
    n_m: float = 10.0
    n_w: float = 5.0

    # 扫描/统计参数
    seeds: Tuple[int, ...] = tuple(range(20))
    # 说明：update_rate=0.1 时，steps=2000 ≈ 200 sweeps；用于缓解临界附近的慢弛豫与有限时间偏置
    steps: int = 2000
    record_interval: int = 5
    burn_in_frac: float = 0.7
    n_bootstrap: int = 2000
    ci_level: float = 0.95

    # r 采样：全局 + rc 附近加密（强制包含 rc）
    r_coarse_n: int = 25
    r_dense_span: float = 0.12
    r_dense_n: int = 41

    # 理论分支拟合：只用临界附近一段，避免饱和段主导
    fit_span: float = 0.08


def _project_dirs() -> Dict[str, Path]:
    outputs = ROOT / "outputs"
    data_dir = outputs / "data"
    fig_dir = outputs / "figs" / "fig2"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {"outputs": outputs, "data": data_dir, "fig": fig_dir}


def _rc_values(cfg: Fig2aConfig, rc: float) -> np.ndarray:
    r_coarse = np.linspace(0.0, 1.0, int(cfg.r_coarse_n))
    r_dense = np.linspace(
        max(0.0, rc - float(cfg.r_dense_span)),
        min(1.0, rc + float(cfg.r_dense_span)),
        int(cfg.r_dense_n),
    )
    r_vals = np.unique(np.concatenate([r_coarse, r_dense, np.asarray([rc])]))
    r_vals.sort()
    return r_vals.astype(float, copy=False)


def _bootstrap_ci_mean(
    rng: np.random.Generator,
    samples: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
) -> Tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1:
        raise ValueError("samples 必须是一维数组")
    n = int(samples.size)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(samples[0])
        return v, v
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    boot_means = samples[idx].mean(axis=1)
    alpha = 1.0 - float(ci_level)
    lo, hi = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _simulate_abs_mean_by_seed(
    cfg: Fig2aConfig,
    r_vals: np.ndarray,
    *,
    topology: str,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        if "abs_by_seed" in loaded.files and np.allclose(loaded["r_vals"], r_vals):
            return loaded["abs_by_seed"].astype(float, copy=False)

    abs_by_seed = np.zeros((r_vals.size, len(cfg.seeds)), dtype=float)
    burn_step = float(cfg.steps) * float(cfg.burn_in_frac)

    for i, r in enumerate(r_vals):
        for j, seed in enumerate(cfg.seeds):
            net_cfg = NetworkConfig(
                n=int(cfg.n),
                avg_degree=float(cfg.avg_degree),
                model=topology,
                beta=float(cfg.beta),
                update_rate=float(cfg.update_rate),
                init_state=str(cfg.init_state),
                sample_mode=str(cfg.sample_mode),
                sample_n=int(cfg.sample_n),
                symmetric_mode=bool(cfg.symmetric_mode),
                r=float(r),
                n_m=float(cfg.n_m),
                n_w=float(cfg.n_w),
                phi=float(cfg.phi),
                theta=float(cfg.theta),
                seed=int(seed),
                local_mode=str(cfg.local_mode),
            )
            model = NetworkAgentModel(net_cfg)
            t, q_traj, _ = model.run(steps=int(cfg.steps), record_interval=int(cfg.record_interval))
            steady_q = q_traj[t >= burn_step]
            q_mean = float(np.mean(steady_q)) if steady_q.size else float(np.mean(q_traj))
            abs_by_seed[i, j] = abs(q_mean)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, r_vals=r_vals, abs_by_seed=abs_by_seed)
    return abs_by_seed


def _style_rcparams() -> None:
    # 优先使用 Windows 字体（WSL 常见路径），以实现 Times New Roman 统一字体。
    times_paths = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    if any(p.exists() for p in times_paths):
        for p in times_paths:
            if p.exists():
                fm.fontManager.addfont(str(p))
        font_family = "Times New Roman"
        serif_fallback = ["Times New Roman"]
    else:
        # 兜底：使用 Matplotlib 自带的 STIX 字体（Times-like）
        font_family = "STIXGeneral"
        serif_fallback = ["STIXGeneral", "DejaVu Serif"]

    mpl.rcParams.update(
        {
            # 字体：Times New Roman（优先）/ STIX（兜底）
            "font.family": font_family,
            "font.serif": serif_fallback,
            # 数学字体：STIX 与 Times 风格更一致
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            # 期刊观感：无网格、线条更厚、字体更大
            "axes.grid": False,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.4,
            "lines.markersize": 5.5,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.0,
            # 避免 Type3 字体（期刊/印刷更友好）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # 导出
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def main() -> None:
    # 避免多线程 BLAS 在受限 /dev/shm 下产生噪声告警；不影响数值正确性
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    cfg = Fig2aConfig()
    dirs = _project_dirs()

    # 理论 r_c：k_avg 的口径与采样模式对齐
    k_avg = int(cfg.avg_degree) if cfg.sample_mode == "degree" else int(cfg.sample_n)
    chi = float(theory.calculate_chi(phi=cfg.phi, theta=cfg.theta, k_avg=k_avg))
    rc = float(theory.calculate_rc(cfg.n_m, cfg.n_w, chi))

    r_vals = _rc_values(cfg, rc)

    tag = (
        f"N{cfg.n}_k{cfg.avg_degree}_sm{cfg.sample_mode}"
        f"_beta{cfg.beta}_u{int(cfg.update_rate*100)}"
        f"_steps{cfg.steps}_ri{cfg.record_interval}_burn{int(cfg.burn_in_frac*100)}"
        f"_seeds{len(cfg.seeds)}_phi{int(round(cfg.phi*100))}_theta{int(round(cfg.theta*100))}"
    )

    abs_er = _simulate_abs_mean_by_seed(
        cfg,
        r_vals,
        topology="er",
        cache_path=dirs["data"] / f"fig2a_abs_by_seed_er_{tag}.npz",
    )
    abs_ba = _simulate_abs_mean_by_seed(
        cfg,
        r_vals,
        topology="ba",
        cache_path=dirs["data"] / f"fig2a_abs_by_seed_ba_{tag}.npz",
    )

    rng = np.random.default_rng(0)
    mean_er = abs_er.mean(axis=1)
    mean_ba = abs_ba.mean(axis=1)

    ci_er = np.zeros((r_vals.size, 2), dtype=float)
    ci_ba = np.zeros((r_vals.size, 2), dtype=float)
    for i in range(r_vals.size):
        ci_er[i, 0], ci_er[i, 1] = _bootstrap_ci_mean(
            rng,
            abs_er[i],
            n_bootstrap=cfg.n_bootstrap,
            ci_level=cfg.ci_level,
        )
        ci_ba[i, 0], ci_ba[i, 1] = _bootstrap_ci_mean(
            rng,
            abs_ba[i],
            n_bootstrap=cfg.n_bootstrap,
            ci_level=cfg.ci_level,
        )

    # 理论分支：拟合缩放系数（相当于拟合 u）
    fit_mask = (r_vals > rc) & (r_vals <= rc + float(cfg.fit_span))
    if np.sum(fit_mask) >= 5:
        x = np.sqrt(r_vals[fit_mask] - rc)
        y = 0.5 * (mean_er[fit_mask] + mean_ba[fit_mask])
        denom = float(np.sum(x * x))
        scale = float(np.sum(x * y) / denom) if denom > 0 else 0.0
    else:
        scale = 0.0
    u_fit = float("inf") if scale <= 0 else float(1.0 / (scale * scale))

    _style_rcparams()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    # 参考线
    ax.axhline(0.0, color="#666666", linewidth=0.9, zorder=1)
    ax.axvline(rc, color="gray", linestyle=":", alpha=0.6, linewidth=1.2, zorder=1)

    # 理论分支（拟合）
    if scale > 0:
        r_dense = np.linspace(rc, 1.0, 400)
        q_dense = scale * np.sqrt(np.maximum(r_dense - rc, 0.0))
        q_dense = np.clip(q_dense, 0.0, 1.0)
        ax.plot(
            r_dense,
            q_dense,
            color="black",
            linestyle=":",
            linewidth=2.0,
            zorder=2,
            label=rf"Mean-field (theory $r_c$={rc:.3f}): $\propto\sqrt{{r-r_c}}$ (fit $u$={u_fit:.2g})",
        )
        ax.plot(r_dense, -q_dense, color="black", linestyle=":", linewidth=2.0, zorder=2)

    # ER：均值 + 95%CI（仅正支画 CI，负支用镜像线避免过密）
    er_color = "#0072B2"  # Okabe–Ito blue
    ax.fill_between(r_vals, ci_er[:, 0], ci_er[:, 1], color=er_color, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(r_vals, mean_er, color=er_color, linewidth=2.4, label="ER (mean ±95% CI)", zorder=3)
    ax.plot(r_vals, -mean_er, color=er_color, linewidth=2.0, alpha=0.35, zorder=3)

    # BA
    ba_color = "#D55E00"  # Okabe–Ito vermillion
    ax.fill_between(r_vals, ci_ba[:, 0], ci_ba[:, 1], color=ba_color, alpha=0.18, linewidth=0, zorder=0)
    ax.plot(
        r_vals,
        mean_ba,
        color=ba_color,
        linewidth=2.4,
        linestyle="--",
        label="BA (mean ±95% CI)",
        zorder=3,
    )
    ax.plot(r_vals, -mean_ba, color=ba_color, linewidth=2.0, linestyle="--", alpha=0.35, zorder=3)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"$\pm|\langle Q\rangle|$")

    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_pdf = ROOT / "Essay" / "figures" / "fig2a_network_validation.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)

    # 预览用 PNG（不用于论文编译）
    out_png = dirs["fig"] / "fig2a_network_validation_preview.png"
    fig.savefig(out_png, dpi=300)

    print(f"Saved PDF: {out_pdf}")
    print(f"Saved preview PNG: {out_png}")


if __name__ == "__main__":
    main()
