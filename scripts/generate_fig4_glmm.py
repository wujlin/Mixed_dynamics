"""
Generate Figure 4: GLMM observed transition rates by exposure group × risk environment.

Three-panel grouped bar chart showing P(H), P(M), P(L) for each exposure group,
split by non-risk vs risk environment. Bootstrap 95% CIs from raw input data.

Usage:
    python scripts/generate_fig4_glmm.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager

# ── 项目路径 ──
ROOT = Path(__file__).resolve().parent.parent


# ── 内联样式（避免 src/__init__ 导入链问题） ──
@contextmanager
def paper_style():
    """轻量级论文样式上下文管理器。"""
    rc = {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    with mpl.rc_context(rc):
        yield


def add_panel_label(ax, label, x=0.0, y=1.0, dx=-42.0, dy=10.0, fontsize=11):
    """在子图左上角添加面板标签 (A, B, C ...)。"""
    ax.annotate(
        label,
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )

# ── 数据路径 ──
DATA_DIR = ROOT / "outputs" / "v2_content_analysis" / "glmm_maxgap7_fullwindow"
INPUT_CSV = DATA_DIR / "risk_interaction_glmm_input.csv"
RATES_CSV = DATA_DIR / "risk_interaction_glmm_observed_rates.csv"
OUT_PATH = ROOT / "Essay_Social" / "Essay" / "Figures" / "fig4_glmm_interaction.png"


def bootstrap_ci(data: pd.Series, n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    """Bootstrap confidence interval for a proportion."""
    rng = np.random.default_rng(42)
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    boot_means = np.array([
        data.sample(n, replace=True, random_state=rng.integers(1e9)).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (lo, hi)


def main():
    # ── 读取原始输入数据（逐行 transition 记录） ──
    df = pd.read_csv(INPUT_CSV)

    # 需要的列：exposure_group, env_type, next_state
    states = ["H", "M", "L"]
    groups = ["mainstream_only", "dual", "wemedia_only"]
    envs = ["norisk", "risk"]
    group_labels = ["Mainstream-\nonly", "Dual", "We-media-\nonly"]

    # ── 计算观测比率和 bootstrap CI ──
    results = {}
    for state in states:
        for grp in groups:
            for env in envs:
                mask = (df["exposure_group"] == grp) & (df["env_type"] == env)
                subset = df.loc[mask, "next_state"]
                is_state = (subset == state).astype(float)
                rate = is_state.mean() if len(is_state) > 0 else 0.0
                ci_lo, ci_hi = bootstrap_ci(is_state) if len(is_state) > 5 else (rate, rate)
                results[(state, grp, env)] = {
                    "rate": rate,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n": len(is_state),
                }

    # ── 绘图 ──
    with paper_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=False)

        # 颜色方案：与前几张图一致的浅/深配色
        c_norisk = "#a8d8b9"   # 浅绿（非风险）
        c_risk   = "#e8836b"   # 珊瑚红（风险）

        bar_width = 0.32
        x = np.arange(len(groups))

        state_titles = {
            "H": "P(next = High)",
            "M": "P(next = Moderate)",
            "L": "P(next = Low)",
        }

        for i, state in enumerate(states):
            ax = axes[i]

            rates_norisk = [results[(state, g, "norisk")]["rate"] for g in groups]
            rates_risk   = [results[(state, g, "risk")]["rate"] for g in groups]

            err_norisk_lo = [results[(state, g, "norisk")]["rate"] - results[(state, g, "norisk")]["ci_lo"] for g in groups]
            err_norisk_hi = [results[(state, g, "norisk")]["ci_hi"] - results[(state, g, "norisk")]["rate"] for g in groups]
            err_risk_lo   = [results[(state, g, "risk")]["rate"] - results[(state, g, "risk")]["ci_lo"] for g in groups]
            err_risk_hi   = [results[(state, g, "risk")]["ci_hi"] - results[(state, g, "risk")]["rate"] for g in groups]

            # 非风险条
            bars1 = ax.bar(
                x - bar_width / 2, rates_norisk, bar_width,
                yerr=[err_norisk_lo, err_norisk_hi],
                color=c_norisk, edgecolor="white", linewidth=0.5,
                capsize=2, error_kw={"linewidth": 0.8, "capthick": 0.8},
                label="Non-risk", zorder=3
            )
            # 风险条
            bars2 = ax.bar(
                x + bar_width / 2, rates_risk, bar_width,
                yerr=[err_risk_lo, err_risk_hi],
                color=c_risk, edgecolor="white", linewidth=0.5,
                capsize=2, error_kw={"linewidth": 0.8, "capthick": 0.8},
                label="Risk", zorder=3
            )

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, fontsize=7)
            ax.set_title(state_titles[state], fontsize=8, fontweight="bold", pad=6)
            ax.set_ylabel("Transition probability" if i == 0 else "", fontsize=7)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.set_ylim(0, None)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linewidth=0.4)
            ax.set_axisbelow(True)

            # 面板标签
            add_panel_label(ax, chr(65 + i))  # A, B, C

            # 标注显著交互效应
            if state == "H":
                # we-media-only × risk: OR=2.02, p<0.001
                _annotate_sig(ax, 2, rates_norisk[2], rates_risk[2],
                              err_norisk_hi[2], err_risk_hi[2],
                              "OR=2.02***", bar_width)
            elif state == "M":
                # mainstream-only × risk: OR=3.15, p=0.027
                _annotate_sig(ax, 0, rates_norisk[0], rates_risk[0],
                              err_norisk_hi[0], err_risk_hi[0],
                              "OR=3.15*", bar_width)

        # 图例
        axes[2].legend(fontsize=6.5, loc="upper right", framealpha=0.8)

        plt.tight_layout(w_pad=1.2)
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
        print(f"Saved: {OUT_PATH}")
        plt.close(fig)


def _annotate_sig(ax, idx, y_norisk, y_risk, err_hi_norisk, err_hi_risk,
                  text, bar_width):
    """在两根柱子之间画显著性标注横线。"""
    y_max = max(y_norisk + err_hi_norisk, y_risk + err_hi_risk)
    y_bar = y_max + 0.02
    x_left = idx - bar_width / 2
    x_right = idx + bar_width / 2

    ax.plot([x_left, x_left, x_right, x_right],
            [y_bar - 0.005, y_bar, y_bar, y_bar - 0.005],
            color="black", linewidth=0.7)
    ax.text((x_left + x_right) / 2, y_bar + 0.005, text,
            ha="center", va="bottom", fontsize=5.5, fontstyle="italic")


if __name__ == "__main__":
    main()
