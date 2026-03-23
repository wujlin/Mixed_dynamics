from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = Path("/mnt/e/newdesktop/emotion_dynamics/tmp_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def set_serif_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "DejaVu Serif",
            ],
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax, label):
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-32, 12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )


def plot_theme_share():
    set_serif_style()

    douyin_labels = [
        "Stigma & respect",
        "Tradition &\ndistinctiveness",
        "Shared belonging",
        "Identity & pride",
    ]
    douyin_values = [32.8, 29.8, 19.6, 17.8]
    douyin_colors = ["#667FAD", "#7FA7C7", "#81B89A", "#6FB2AD"]

    kuaishou_labels = [
        "Performance &\nrevival",
        "Han identity &\nrace",
        "Conflict &\nconfrontation",
    ]
    kuaishou_values = [68.4, 18.7, 12.9]
    kuaishou_colors = ["#D9A15B", "#D68F63", "#D4736E"]

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), gridspec_kw={"wspace": 0.28})

    for ax, labels, values, colors, title, panel in [
        (axes[0], douyin_labels, douyin_values, douyin_colors, "Douyin", "a"),
        (axes[1], kuaishou_labels, kuaishou_values, kuaishou_colors, "Kuaishou", "b"),
    ]:
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, height=0.52, edgecolor="none")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 75)
        ax.set_xlabel("Share of clustered comments (%)")
        ax.set_title(title, pad=10)
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        despine(ax)

        for tick in ax.get_yticklabels():
            tick.set_color("#222222")

        for bar, val in zip(bars, values):
            ax.text(
                val + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=12,
                color="#222222",
            )

        add_panel_label(ax, panel)

    fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(OUTDIR / "theme_share_refined.png", dpi=900, bbox_inches="tight")
    fig.savefig(OUTDIR / "theme_share_refined.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_network_metrics():
    set_serif_style()

    metrics = [
        ("Nodes", 90, 471),
        ("Edges", 192, 602),
        ("Mean path length", 3.41, 5.63),
        ("Clustering coefficient", 0.36, 0.28),
    ]
    platform_labels = ["Douyin", "Kuaishou"]
    colors = ["#6B84B0", "#D9A15B"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
    axes = axes.ravel()

    for ax, (title, douyin, kuaishou) in zip(axes, metrics):
        vals = [douyin, kuaishou]
        x = np.arange(2)
        bars = ax.bar(x, vals, width=0.56, color=colors, edgecolor="none")
        ax.set_xticks(x, platform_labels)
        ax.set_title(title, pad=8)
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        despine(ax)

        for bar, val in zip(bars, vals):
            fmt = f"{val:.2f}" if isinstance(val, float) and val < 10 and not float(val).is_integer() else f"{val:g}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.025,
                fmt,
                ha="center",
                va="bottom",
                fontsize=11.5,
                color="#222222",
            )

        if title == "Clustering coefficient":
            ax.set_ylim(0, 0.50)
        elif title == "Mean path length":
            ax.set_ylim(0, 6.8)
        elif title == "Nodes":
            ax.set_ylim(0, 560)
        elif title == "Edges":
            ax.set_ylim(0, 740)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.10, wspace=0.28, hspace=0.32)
    fig.savefig(OUTDIR / "network_metrics_refined.png", dpi=900, bbox_inches="tight")
    fig.savefig(OUTDIR / "network_metrics_refined.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_theme_share()
    plot_network_metrics()
