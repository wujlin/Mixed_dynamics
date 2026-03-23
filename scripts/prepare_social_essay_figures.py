#!/usr/bin/env python3
"""
为 Essay_Social/Essay 生成统一的 Nature-inspired 图表。

输入：
- outputs/v2_content_analysis/*
- outputs/v2_analysis_fullwindow_v2/risk_interaction_glmm/*

输出：
- Essay_Social/Essay/Figures/fig1_overview_framework.{png,pdf}
- Essay_Social/Essay/Figures/fig2_emotion_distribution.{png,pdf}
- Essay_Social/Essay/Figures/fig3_temporal_synchrony.{png,pdf}
- Essay_Social/Essay/Figures/fig4_glmm_transition_rates.{png,pdf}
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]

FIGSIZE_NATURE_HALF = (3.5, 2.6)
FIGSIZE_NATURE_FULL = (7.2, 4.5)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#777777",
}


class NatureStyle:
    font_size = 7.0
    axes_labelsize = 8.0
    axes_titlesize = 8.0
    tick_labelsize = 7.0
    legend_fontsize = 7.0
    axes_linewidth = 0.8
    lines_linewidth = 1.2
    lines_markersize = 4.0
    figure_dpi = 150
    savefig_dpi = 300


def nature_rcparams(style: NatureStyle | None = None) -> Dict[str, object]:
    style = style or NatureStyle()
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.linewidth": style.axes_linewidth,
        "lines.linewidth": style.lines_linewidth,
        "lines.markersize": style.lines_markersize,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.size": style.font_size,
        "axes.titlesize": style.axes_titlesize,
        "axes.labelsize": style.axes_labelsize,
        "xtick.labelsize": style.tick_labelsize,
        "ytick.labelsize": style.tick_labelsize,
        "legend.fontsize": style.legend_fontsize,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": style.figure_dpi,
        "savefig.dpi": style.savefig_dpi,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }


@contextmanager
def nature_style(style: NatureStyle | None = None):
    with mpl.rc_context(nature_rcparams(style)):
        yield


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = 0.0,
    y: float = 1.0,
    dx: float = -42.0,
    dy: float = 4.0,
    fontsize: float = 8.0,
) -> None:
    text = ax.annotate(
        str(label),
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=fontsize,
        color="black",
        annotation_clip=False,
    )
    text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])


def save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
EMOTION_ORDER = ["H", "M", "L"]
EMOTION_COLORS = {
    "H": "#C98A72",
    "M": "#8E8E8E",
    "L": "#88A9C4",
}
MEDIA_COLORS = {
    "mainstream": "#6E8CA6",
    "wemedia": "#C98A72",
}
ENV_COLORS = {
    "norisk": "#B9B9B9",
    "risk": "#C98A72",
}
GROUP_LABELS = {
    "mainstream": "Mainstream",
    "wemedia": "We-media",
    "public": "Public",
    "mainstream_only": "Mainstream\nonly",
    "wemedia_only": "We-media\nonly",
    "dual": "Dual",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成社会篇论文图表")
    p.add_argument(
        "--essay-dir",
        default="Essay_Social/Essay",
        help="论文目录",
    )
    p.add_argument(
        "--content-dir",
        default="outputs/v2_content_analysis_unified",
        help="内容分析输出目录",
    )
    p.add_argument(
        "--glmm-dir",
        default="outputs/v2_analysis_unified_windowenv/risk_interaction_glmm_maxgap7",
        help="GLMM 输出目录",
    )
    p.add_argument("--seed", type=int, default=20260308, help="随机种子")
    p.add_argument("--bootstrap", type=int, default=1000, help="Figure 4 bootstrap 次数")
    return p.parse_args()


def save_png_pdf(fig: plt.Figure, base_path: Path) -> None:
    save_figure(fig, base_path.with_suffix(".png"))
    save_figure(fig, base_path.with_suffix(".pdf"))


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.04, title, ha="left", va="top", fontsize=8.2, fontweight="bold")
    ax.text(x + 0.02, y + h - 0.085, body, ha="left", va="top", fontsize=7.0, linespacing=1.35)


def _draw_arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.0,
        color="#4C4C4C",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def plot_figure1_overview(fig_dir: Path) -> None:
    with nature_style(NatureStyle()):
        fig, ax = plt.subplots(figsize=FIGSIZE_NATURE_FULL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        _draw_box(
            ax,
            0.04,
            0.60,
            0.18,
            0.25,
            "Dataset and labels",
            "92,536 Weibo posts\nMar 2020-Jul 2025\nLLM labels: arousal (H/M/L)\nand risk (risk/non-risk)",
            "#F4F6F8",
            "#AAB4BE",
        )
        _draw_box(
            ax,
            0.28,
            0.58,
            0.20,
            0.28,
            "1. Content framing",
            "Mainstream keeps risk posts\nin the moderate band.\nWe-media shifts risk posts\ntoward both high (19.7%)\nand low (27.2%) arousal.",
            "#EEF4FB",
            OKABE_ITO["blue"],
        )
        _draw_box(
            ax,
            0.52,
            0.58,
            0.20,
            0.28,
            "2. Temporal synchrony",
            "The two streams align most\nstrongly on the same day\n(r = 0.72 at lag 0), while\nwe-media covers a broader\nset of risk bursts.",
            "#FBEFE8",
            OKABE_ITO["vermillion"],
        )
        _draw_box(
            ax,
            0.76,
            0.58,
            0.20,
            0.28,
            "3. Individual sensitivity",
            "Under risk environments,\nwe-media-only users become\nmore likely to shift to high\narousal and less likely to\nremain moderate.",
            "#EEF7F3",
            OKABE_ITO["bluish_green"],
        )
        _draw_box(
            ax,
            0.20,
            0.18,
            0.60,
            0.20,
            "Compounding mechanism",
            "Media-type differentiation links three layers at once:\n"
            "polarizing frames + broader event coverage + stronger user sensitivity.\n"
            "This is why we-media produces a distinct emotional risk environment.",
            "#FFF8E8",
            OKABE_ITO["orange"],
        )

        _draw_arrow(ax, (0.22, 0.73), (0.28, 0.73))
        _draw_arrow(ax, (0.48, 0.73), (0.52, 0.73))
        _draw_arrow(ax, (0.72, 0.73), (0.76, 0.73))
        _draw_arrow(ax, (0.38, 0.58), (0.36, 0.38))
        _draw_arrow(ax, (0.62, 0.58), (0.50, 0.38))
        _draw_arrow(ax, (0.86, 0.58), (0.64, 0.38))

        ax.text(
            0.04,
            0.93,
            "Three-layer analytical framework",
            ha="left",
            va="top",
            fontsize=9.0,
            fontweight="bold",
        )

        fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.06)
        save_png_pdf(fig, fig_dir / "fig1_overview_framework")
        plt.close(fig)


def sync_manual_framework(fig_dir: Path) -> None:
    src_pdf = ROOT / "figures" / "framework_emotion.pdf"
    src_png = ROOT / "figures" / "framework_emotion.png"
    dst_pdf = fig_dir / "fig1_overview_framework.pdf"
    dst_png = fig_dir / "fig1_overview_framework.png"
    if src_pdf.exists() and src_png.exists():
        shutil.copy2(src_pdf, dst_pdf)
        shutil.copy2(src_png, dst_png)
    else:
        plot_figure1_overview(fig_dir)


def plot_figure2_emotion_distribution(content_dir: Path, fig_dir: Path) -> None:
    d = pd.read_csv(content_dir / "emotion_frame" / "emotion_distribution_long.csv")
    d["group"] = d["group"].astype(str)
    d["risk_class"] = d["risk_class"].astype(str)
    d["emotion_class"] = d["emotion_class"].astype(str)

    order = ["public", "wemedia", "mainstream"]
    risk_order = ["risk", "norisk"]

    with nature_style(NatureStyle()):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=FIGSIZE_NATURE_FULL,
            sharex=True,
            sharey=True,
            gridspec_kw={"width_ratios": [1.25, 0.95]},
        )

        for ax, risk_class, panel in zip(axes, risk_order, ["a", "b"]):
            sub = d[d["risk_class"] == risk_class].copy()
            y = np.arange(len(order))
            left = np.zeros(len(order))
            for emo in EMOTION_ORDER:
                vals = []
                for grp in order:
                    row = sub[(sub["group"] == grp) & (sub["emotion_class"] == emo)]
                    vals.append(float(row["share"].iloc[0]) if len(row) else 0.0)
                vals_arr = np.array(vals)
                ax.barh(
                    y,
                    vals_arr,
                    left=left,
                    color=EMOTION_COLORS[emo],
                    edgecolor="white",
                    height=0.66,
                    linewidth=0.7,
                    label=emo,
                )
                for yi, width, x0, grp in zip(y, vals_arr, left, order):
                    if width >= 0.12:
                        x_text = x0 + width / 2.0
                        ax.text(
                            x_text,
                            yi,
                            f"{width * 100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=7.0,
                            fontweight="bold",
                            color="#222222",
                            bbox=dict(boxstyle="round,pad=0.10", fc="#FAF9F7", ec="none", alpha=0.88),
                        )
                    elif risk_class == "risk" and grp in {"mainstream", "wemedia"} and width >= 0.01:
                        label = f"{width * 100:.1f}%"
                        if x0 < 1e-6:
                            ax.text(
                                x0 + width + 0.012,
                                yi,
                                label,
                                ha="left",
                                va="center",
                                fontsize=6.8,
                                fontweight="bold",
                                color="#222222",
                                bbox=dict(boxstyle="round,pad=0.12", fc="#FAF9F7", ec="none", alpha=0.96),
                            )
                        elif x0 + width > 0.999:
                            ax.text(
                                x0 - 0.012,
                                yi,
                                label,
                                ha="right",
                                va="center",
                                fontsize=6.8,
                                fontweight="bold",
                                color="#222222",
                                bbox=dict(boxstyle="round,pad=0.12", fc="#FAF9F7", ec="none", alpha=0.96),
                            )
                left += vals_arr

            ax.set_yticks(y, [GROUP_LABELS[g] for g in order])
            ax.set_xlim(0, 1.0)
            ax.set_xticks(np.linspace(0, 1, 6))
            ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
            ax.set_xlabel("Share of posts")
            ax.set_title(
                "Risk posts" if risk_class == "risk" else "Non-risk posts",
                pad=6,
                color="#222222",
                fontweight="bold",
                fontsize=8.6,
            )
            despine(ax)
            add_panel_label(ax, panel, dx=-18, dy=2, fontsize=8.0)

            for tick_label in ax.get_yticklabels():
                tick_label.set_fontweight("bold")
                tick_label.set_color("#1F1F1F")

        axes[0].set_ylabel("Media type", labelpad=12)
        legend_items = [
            Line2D([0], [0], color=EMOTION_COLORS["H"], lw=6, label="High arousal"),
            Line2D([0], [0], color=EMOTION_COLORS["M"], lw=6, label="Moderate arousal"),
            Line2D([0], [0], color=EMOTION_COLORS["L"], lw=6, label="Low arousal"),
        ]
        fig.legend(
            handles=legend_items,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=3,
            frameon=False,
            prop={"size": 7.6, "weight": "bold"},
        )
        fig.subplots_adjust(left=0.15, right=0.98, top=0.82, bottom=0.16, wspace=0.16)
        save_png_pdf(fig, fig_dir / "fig2_emotion_distribution")
        plt.close(fig)


def plot_figure3_temporal(content_dir: Path, fig_dir: Path) -> None:
    time_dir = content_dir / "time_response"
    ccf = pd.read_csv(time_dir / "ccf_values.csv")
    seg = pd.read_csv(time_dir / "ccf_by_segment.csv")
    eca = pd.read_csv(time_dir / "event_coincidence.csv")

    with nature_style(NatureStyle()):
        fig = plt.figure(figsize=FIGSIZE_NATURE_FULL)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ccf_color = "#5F7283"
        ccf_fill = "#DCE6ED"
        pre_color = "#7A7A7A"
        post_color = "#C98A72"

        # Panel a: full-sample CCF
        ax1.fill_between(
            ccf["lag"],
            ccf["ci_lower"],
            ccf["ci_upper"],
            color=ccf_fill,
            alpha=0.16,
            linewidth=0,
        )
        sig = ccf["significant"].astype(bool).values
        ax1.plot(
            ccf["lag"],
            ccf["ccf"],
            color=ccf_color,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
        ax1.scatter(
            ccf.loc[sig, "lag"],
            ccf.loc[sig, "ccf"],
            color=ccf_color,
            s=16,
            zorder=3,
        )
        ax1.axhline(0, color="#9A9A9A", linewidth=0.8)
        ax1.axvline(0, color="#9A9A9A", linewidth=0.8)
        ax1.set_xlim(ccf["lag"].min(), ccf["lag"].max())
        ax1.set_xticks([-14, -7, 0, 7, 14])
        ax1.set_ylabel("Cross-correlation")
        ax1.set_xlabel("Lag (days)", fontsize=10.4)
        despine(ax1)
        add_panel_label(ax1, "a", dx=-18, dy=8, fontsize=8.0)

        # Panel b: segmented CCF
        for segment, color, label, linestyle, markerface in [
            ("pre_spike", pre_color, "Pre-spike", "--", "white"),
            ("post_spike", post_color, "Post-spike", "-", post_color),
        ]:
            sub = seg[seg["segment"] == segment].copy()
            ax2.plot(
                sub["lag"],
                sub["ccf"],
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
                marker="o",
                markersize=2.5,
                markerfacecolor=markerface,
                markeredgecolor=color,
                label=label,
            )
        ax2.axhline(0, color="#9A9A9A", linewidth=0.8)
        ax2.axvline(0, color="#9A9A9A", linewidth=0.8)
        ax2.set_xlim(-14.5, 14.5)
        ax2.set_xticks([-14, -7, 0, 7, 14])
        ax2.set_xlabel("Lag (days)", fontsize=10.4)
        ax2.set_ylabel("Cross-correlation")
        ax2.legend(loc="upper right", frameon=False, ncol=1, handlelength=2.0, prop={"size": 7.6})
        despine(ax2)
        add_panel_label(ax2, "b", dx=-28, dy=8, fontsize=8.0)

        # Panel c: event coincidence
        windows = sorted(eca["window"].unique().tolist())
        width = 0.34
        x = np.arange(len(windows))
        m2w = eca[eca["direction"] == "mainstream_to_wemedia"].sort_values("window")
        w2m = eca[eca["direction"] == "wemedia_to_mainstream"].sort_values("window")
        ax3.bar(
            x - width / 2,
            m2w["p_observed"],
            width=width,
            color=MEDIA_COLORS["mainstream"],
            label="Mainstream → We-media",
        )
        ax3.bar(
            x + width / 2,
            w2m["p_observed"],
            width=width,
            color=MEDIA_COLORS["wemedia"],
            label="We-media → Mainstream",
        )
        ax3.set_xticks(x, [f"{w}" for w in windows])
        ax3.set_xlabel("Window (days)", fontsize=10.4)
        ax3.set_ylabel("Co-occurrence rate")
        ax3.set_ylim(0, 1.0)
        ax3.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            ncol=1,
            prop={"size": 7.6},
        )
        despine(ax3)
        add_panel_label(ax3, "c", dx=-28, dy=8, fontsize=8.0)

        fig.subplots_adjust(left=0.13, right=0.98, top=0.94, bottom=0.13)
        save_png_pdf(fig, fig_dir / "fig3_temporal_synchrony")
        plt.close(fig)


def bootstrap_state_rates(
    d: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (group, env), sub in d.groupby(["exposure_group", "env_type"], sort=False):
        values = sub["next_state"].astype(str).values
        n = len(values)
        for state in ["H", "M", "L"]:
            obs = float(np.mean(values == state)) if n else np.nan
            if n == 0:
                lo = hi = np.nan
            else:
                boot = np.empty(n_boot, dtype=float)
                for i in range(n_boot):
                    sample = rng.choice(values, size=n, replace=True)
                    boot[i] = np.mean(sample == state)
                lo = float(np.quantile(boot, 0.025))
                hi = float(np.quantile(boot, 0.975))
            rows.append(
                {
                    "exposure_group": group,
                    "env_type": env,
                    "state": state,
                    "n_transitions": int(n),
                    "rate": obs,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return pd.DataFrame(rows)


def plot_figure4_glmm(glmm_dir: Path, fig_dir: Path, n_boot: int, seed: int) -> None:
    d = pd.read_csv(glmm_dir / "risk_interaction_glmm_input.csv")
    rates = bootstrap_state_rates(d, n_boot=n_boot, seed=seed)

    group_order = ["mainstream_only", "dual", "wemedia_only"]
    env_order = ["norisk", "risk"]
    state_order = ["H", "M", "L"]
    ylims = {
        "H": (0.0, 0.28),
        "M": (0.65, 1.00),
        "L": (0.0, 0.14),
    }

    with nature_style(NatureStyle()):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.8), sharex=False)
        offsets = {"norisk": -0.12, "risk": 0.12}

        for ax, state, panel in zip(axes, state_order, ["a", "b", "c"]):
            sub = rates[rates["state"] == state].copy()
            x = np.arange(len(group_order))
            for i, grp in enumerate(group_order):
                row_nr = sub[(sub["exposure_group"] == grp) & (sub["env_type"] == "norisk")].iloc[0]
                row_r = sub[(sub["exposure_group"] == grp) & (sub["env_type"] == "risk")].iloc[0]
                ax.plot(
                    [i + offsets["norisk"], i + offsets["risk"]],
                    [row_nr["rate"], row_r["rate"]],
                    color="#B3B3B3",
                    linewidth=1.1,
                    zorder=1,
                )
                for env, row in [("norisk", row_nr), ("risk", row_r)]:
                    xpos = i + offsets[env]
                    ax.errorbar(
                        xpos,
                        row["rate"],
                        yerr=[[row["rate"] - row["ci_low"]], [row["ci_high"] - row["rate"]]],
                        fmt="o",
                        color=ENV_COLORS[env],
                        ecolor=ENV_COLORS[env],
                        elinewidth=1.2,
                        capsize=3.0,
                        capthick=1.2,
                        markersize=5.4,
                        zorder=3,
                    )

            ax.set_xticks(np.arange(len(group_order)), [GROUP_LABELS[g] for g in group_order])
            ax.set_ylim(*ylims[state])
            ax.set_ylabel("Observed transition rate" if state == "H" else "")
            ax.tick_params(axis="both", labelsize=8.2, width=1.0, length=4.2)
            if state == "H":
                ax.yaxis.label.set_size(9.2)
            despine(ax)
            ax.spines["left"].set_linewidth(1.1)
            ax.spines["bottom"].set_linewidth(1.1)
            add_panel_label(ax, panel, dx=-6, dy=12, fontsize=9.0)

        legend_items = [
            Line2D([0], [0], marker="o", color=ENV_COLORS["norisk"], lw=0, markersize=5.6, label="Non-risk environment"),
            Line2D([0], [0], marker="o", color=ENV_COLORS["risk"], lw=0, markersize=5.6, label="Risk environment"),
        ]
        fig.legend(
            handles=legend_items,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
            prop={"size": 8.4, "weight": "bold"},
        )
        fig.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.20, wspace=0.36)
        save_png_pdf(fig, fig_dir / "fig4_glmm_transition_rates")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    essay_dir = ROOT / args.essay_dir
    fig_dir = essay_dir / "Figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    content_dir = ROOT / args.content_dir
    glmm_dir = ROOT / args.glmm_dir

    sync_manual_framework(fig_dir)
    plot_figure2_emotion_distribution(content_dir, fig_dir)
    plot_figure3_temporal(content_dir, fig_dir)
    plot_figure4_glmm(glmm_dir, fig_dir, n_boot=int(args.bootstrap), seed=int(args.seed))


if __name__ == "__main__":
    main()
