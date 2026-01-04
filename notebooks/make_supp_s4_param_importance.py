"""
Supplementary (Parameter landscape): parameter importance + k×(n_w/n_m) interaction.

目标（只读缓存，不重跑仿真）：
1) 用 Δr_c（扫描范围内 max-min）对比四类因素对“临界边界”的影响强度，并用 r_{c,baseline} 归一化。
2) 仅做理论层面的交互：在不同 k（不同 χ）下，比较 r_c(n_w/n_m) 曲线的杠杆差异。

输出：
  - Essay/figures_supp/fig_supp_param_importance.pdf
  - outputs/figs/supp/fig_supp_param_importance_preview.png
  - outputs/figs/supp/fig_supp_param_importance_table.md  (数值表，便于写作引用)

运行（需要 emotion conda env）：
  /home/wujlin/miniconda3/envs/emotion/bin/python notebooks/make_supp_s4_param_importance.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402
from src.plot_style import OKABE_ITO, add_panel_label, apply_paper_style  # noqa: E402


@dataclass(frozen=True)
class Config:
    note4_path: Path = ROOT / "outputs" / "data" / "note4_sensitivity_data.npz"
    n_m: float = 10.0
    n_w: float = 5.0
    baseline_phi: float = 0.54
    baseline_theta: float = 0.46
    baseline_k: int = 50
    baseline_ratio: float = 0.5
    interaction_k: tuple[int, ...] = (10, 50, 200, 500)

    out_pdf: Path = ROOT / "Essay" / "figures_supp" / "fig_supp_param_importance.pdf"
    out_png: Path = ROOT / "outputs" / "figs" / "supp" / "fig_supp_param_importance_preview.png"
    out_table_md: Path = ROOT / "outputs" / "figs" / "supp" / "fig_supp_param_importance_table.md"


def _baseline_rc(cfg: Config) -> float:
    chi = float(theory.calculate_chi(phi=cfg.baseline_phi, theta=cfg.baseline_theta, k_avg=int(cfg.baseline_k)))
    return float(theory.calculate_rc(n_m=float(cfg.n_m), n_w=float(cfg.n_w), chi=chi))


def _load_note4(cfg: Config) -> dict[str, np.ndarray]:
    if not cfg.note4_path.exists():
        raise FileNotFoundError(f"未找到 Note04 合成缓存：{cfg.note4_path}")
    d = np.load(cfg.note4_path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _format_md_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Factor | Scan range | $r_{c,\\min}$ | $r_{c,\\max}$ | $r_{c,\\mathrm{base}}$ | $\\Delta r_c$ | $\\Delta r_c/r_{c,\\mathrm{base}}$ |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            "| {name} | {scan} | {rc_min:.3f} | {rc_max:.3f} | {rc_base:.3f} | {delta:.3f} | {rel:.1%} |".format(**r)
        )
    return header + "\n".join(lines) + "\n"


def _importance_rows(cfg: Config, data: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rc_base_theory = _baseline_rc(cfg)

    # 1) thresholds (phi, theta): use full (phi,theta) grid where chi>2 and valid
    chi_map = np.asarray(data["chi_map"], dtype=float)
    rc_map = theory.calculate_rc(n_m=float(cfg.n_m), n_w=float(cfg.n_w), chi=chi_map).astype(float, copy=False)
    valid = np.isfinite(chi_map) & (chi_map > 2.0)
    rc_valid = rc_map[valid]
    thr_min = float(np.min(rc_valid))
    thr_max = float(np.max(rc_valid))

    # 2) k sweep (theory): baseline thresholds + baseline ratio
    k_list = np.asarray(data["k_list"], dtype=int)
    rc_k = np.asarray(data["rc_k"], dtype=float)
    k_min = float(np.min(rc_k))
    k_max = float(np.max(rc_k))

    # 3) media ratio sweep (theory)
    ratio_vals = np.asarray(data["ratio_vals"], dtype=float)
    rc_ratio = np.asarray(data["rc_ratio"], dtype=float)
    ratio_min = float(np.min(rc_ratio))
    ratio_max = float(np.max(rc_ratio))

    # 4) beta sweep (ABM): use symmetric local coupling (control) to avoid branch-bias artifacts
    beta_path = Path(str(data["beta_sweep_path_symmetric"]))
    if not beta_path.exists():
        beta_path = ROOT / "outputs" / "data" / str(data["beta_sweep_file_symmetric"])
    if not beta_path.exists():
        raise FileNotFoundError(f"未找到 beta-sweep(symmetric) 缓存：{beta_path}")
    db = np.load(beta_path, allow_pickle=False)
    betas = np.asarray(db["betas"], dtype=float)
    rc_beta = np.asarray(db["rc_est"], dtype=float)
    # baseline at beta=0: use the first matching entry (should exist)
    idx0 = np.where(np.isclose(betas, 0.0))[0]
    if idx0.size == 0:
        raise RuntimeError("beta-sweep 缓存缺少 beta=0 的基准点")
    rc_base_beta = float(rc_beta[int(idx0[0])])
    beta_min = float(np.min(rc_beta))
    beta_max = float(np.max(rc_beta))

    rows = [
        {
            "name": r"Thresholds $(\phi,\theta)$",
            "scan": f"grid ({len(data['phi_range'])}×{len(data['theta_range'])})",
            "rc_min": thr_min,
            "rc_max": thr_max,
            "rc_base": rc_base_theory,
        },
        {
            "name": "Information density $k$",
            "scan": f"{int(np.min(k_list))}–{int(np.max(k_list))}",
            "rc_min": k_min,
            "rc_max": k_max,
            "rc_base": rc_base_theory,
        },
        {
            "name": r"Media ratio $n_w/n_m$",
            "scan": f"{float(np.min(ratio_vals)):.1f}–{float(np.max(ratio_vals)):.1f}",
            "rc_min": ratio_min,
            "rc_max": ratio_max,
            "rc_base": rc_base_theory,
        },
        {
            "name": r"Local coupling $\beta$ (ABM)",
            "scan": f"{float(np.min(betas)):.2f}–{float(np.max(betas)):.2f}",
            "rc_min": beta_min,
            "rc_max": beta_max,
            "rc_base": rc_base_beta,
        },
    ]
    for r in rows:
        delta = float(r["rc_max"] - r["rc_min"])
        rc_base = float(r["rc_base"])
        r["delta"] = delta
        r["rel"] = float(delta / rc_base) if rc_base > 0 else float("nan")
    rows.sort(key=lambda x: float(x["rel"]), reverse=True)
    return rows


def _plot_importance(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    # Use a clean dumbbell plot: range as line, endpoints as dots, baseline as a tick.
    y = np.arange(len(rows), dtype=float)
    for yi, r in zip(y, rows, strict=True):
        rc_min = float(r["rc_min"])
        rc_max = float(r["rc_max"])
        rc_base = float(r["rc_base"])

        ax.plot([rc_min, rc_max], [yi, yi], color=OKABE_ITO["gray"], linewidth=2.0, zorder=1)
        ax.scatter([rc_min, rc_max], [yi, yi], color=OKABE_ITO["black"], s=18, zorder=2)
        ax.scatter([rc_base], [yi], marker="|", color=OKABE_ITO["black"], s=260, linewidths=2.2, zorder=3)

        # annotate normalized effect size to the right (outside axes)
        rel = float(r["rel"])
        ax.text(
            1.02,
            yi,
            f"{rel:.0%}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9,
            color=OKABE_ITO["gray"],
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([str(r["name"]) for r in rows])
    ax.set_ylim(len(rows) - 0.2, -0.8)
    ax.set_xlabel(r"Critical boundary $r_c$")
    ax.set_xlim(0.20, 1.00)
    ax.tick_params(direction="in", top=True, right=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.set_title(r"Importance (range; tick = baseline)", pad=6)


def _plot_interaction(ax: plt.Axes, cfg: Config, data: dict[str, np.ndarray]) -> None:
    ratio_vals = np.asarray(data["ratio_vals"], dtype=float)
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["bluish_green"], OKABE_ITO["orange"]]

    for k, c in zip(cfg.interaction_k, colors, strict=False):
        chi = float(theory.calculate_chi(phi=cfg.baseline_phi, theta=cfg.baseline_theta, k_avg=int(k)))
        # calculate_rc 目前只接受标量 n_w；这里直接用闭式表达做向量化
        rc = (chi + 2.0) / ((chi + 2.0) + ratio_vals * (chi - 2.0))
        ax.plot(ratio_vals, rc, color=c, linewidth=2.0, label=fr"$k={k}$")

    ax.axvline(cfg.baseline_ratio, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(r"Media ratio $n_w/n_m$")
    ax.set_ylabel(r"$r_c$")
    ax.set_xlim(float(np.min(ratio_vals)), float(np.max(ratio_vals)))
    ax.set_ylim(0.40, 1.00)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_title(r"Interaction: $\chi(k)$ modulates media leverage", pad=6)
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=2.0, columnspacing=1.0)


def main() -> None:
    cfg = Config()
    data = _load_note4(cfg)
    rows = _importance_rows(cfg, data)

    _ensure_parent(cfg.out_pdf)
    _ensure_parent(cfg.out_png)
    _ensure_parent(cfg.out_table_md)

    md = _format_md_table(rows)
    cfg.out_table_md.write_text(md, encoding="utf-8")

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.8))
    ax_a, ax_b = axes

    _plot_importance(ax_a, rows)
    _plot_interaction(ax_b, cfg, data)

    add_panel_label(ax_a, "a", dx=-36.0)
    add_panel_label(ax_b, "b", dx=-36.0)

    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.23, top=0.88, wspace=0.35)
    fig.savefig(cfg.out_pdf)
    fig.savefig(cfg.out_png, dpi=300)
    plt.close(fig)

    print(f"Saved PDF: {cfg.out_pdf}")
    print(f"Saved preview PNG: {cfg.out_png}")
    print(f"Saved table: {cfg.out_table_md}")


if __name__ == "__main__":
    main()
