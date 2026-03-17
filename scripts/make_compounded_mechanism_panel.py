#!/usr/bin/env python3
"""
生成 framework 中的 "Compounded Mechanism" panel。

输出两套版本：
1. 透明背景（适合直接叠放到 PPT 色块上）
2. 同色底版本（避免部分 Office 渲染透明边缘发灰）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COLOR_BLUE = "#5B7B9A"
COLOR_GREY = "#8B8B8B"
COLOR_TERRA = "#C07C63"
COLOR_GUIDE = "#B8C7D6"
COLOR_BG_BLUE = "#DCEAF7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绘制 compounded mechanism panel")
    p.add_argument(
        "--out-dir",
        default="outputs/framework_panels",
        help="输出目录",
    )
    p.add_argument(
        "--width",
        type=int,
        default=1200,
        help="像素宽度",
    )
    p.add_argument(
        "--height",
        type=int,
        default=260,
        help="像素高度",
    )
    p.add_argument(
        "--bg-color",
        default=COLOR_BG_BLUE,
        help="同色底版本背景色",
    )
    return p.parse_args()


def sigmoid(z: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(z - center) / width))


def smooth_noise(n: int, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    kernel = np.hanning(25)
    kernel /= kernel.sum()
    smoothed = np.convolve(noise, kernel, mode="same")
    smoothed /= max(np.max(np.abs(smoothed)), 1e-6)
    return smoothed * scale


def build_traces(n: int = 1600) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x = np.linspace(0.0, 1.0, n)

    # 汇聚与放大过程：左侧分离，中段耦合，右侧增强但保持结构，而不是纯噪声。
    merge = sigmoid(x, center=0.43, width=0.030)
    amplify = sigmoid(x, center=0.67, width=0.045)
    base_freq = 2 * np.pi * (3.0 + 9.0 * merge + 10.0 * amplify) * x

    guide = np.full_like(x, 0.5)

    # 左段三条机制线的个性要明显不同。
    y_blue = (
        0.58
        + (1 - merge) * (0.030 * np.sin(2 * np.pi * 5.5 * x + 0.4) + 0.010 * np.sin(2 * np.pi * 16.0 * x))
        + merge * (0.085 + 0.070 * amplify) * np.sin(base_freq + 0.8)
        + 0.025 * amplify * smooth_noise(n, seed=3)
    )

    pulse = np.sin(2 * np.pi * 11.0 * x)
    pulse = np.clip(pulse, 0.0, None) ** 3
    y_grey = (
        0.50
        + (1 - merge) * (0.070 * pulse)
        + merge * (0.075 + 0.060 * amplify) * np.sin(base_freq * 1.02 + 2.0)
        + 0.018 * amplify * smooth_noise(n, seed=7)
    )

    fine = 0.018 * np.sin(2 * np.pi * 32.0 * x) + 0.016 * np.sin(2 * np.pi * 53.0 * x + 1.6)
    y_terra = (
        0.40
        + (1 - merge) * (fine + 0.010 * smooth_noise(n, seed=11))
        + merge * (0.060 + 0.095 * amplify) * np.sin(base_freq * 1.08 + 4.4)
        + amplify * (0.065 * np.sin(2 * np.pi * 62.0 * x + 1.1) + 0.040 * smooth_noise(n, seed=19))
    )

    # 右侧仍保留蓝/灰痕迹，不让它完全变成单一 terracotta。
    y_blue = np.clip(y_blue, 0.18, 0.84)
    y_grey = np.clip(y_grey, 0.18, 0.82)
    y_terra = np.clip(y_terra, 0.14, 0.90)

    return x, {
        "guide": guide,
        "blue": y_blue,
        "grey": y_grey,
        "terra": y_terra,
    }


def draw_panel(
    out_base: Path,
    *,
    width_px: int,
    height_px: int,
    facecolor: str | None,
    transparent: bool,
) -> None:
    dpi = 200
    figsize = (width_px / dpi, height_px / dpi)
    x, traces = build_traces()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(facecolor)
        ax.set_facecolor(facecolor)

    ax.plot(x, traces["guide"], color=COLOR_GUIDE, linewidth=0.85, linestyle=(0, (3, 3)), alpha=0.95, zorder=1)
    ax.plot(x, traces["blue"], color=COLOR_BLUE, linewidth=2.0, zorder=2)
    ax.plot(x, traces["grey"], color=COLOR_GREY, linewidth=2.0, zorder=3)
    ax.plot(x, traces["terra"], color=COLOR_TERRA, linewidth=2.1, zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.12, 0.92)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_base.with_suffix(".png"),
        transparent=transparent,
        facecolor="none" if transparent else facecolor,
        edgecolor="none",
    )
    fig.savefig(
        out_base.with_suffix(".svg"),
        transparent=transparent,
        facecolor="none" if transparent else facecolor,
        edgecolor="none",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    draw_panel(
        out_dir / "compounded_mechanism_panel_transparent",
        width_px=int(args.width),
        height_px=int(args.height),
        facecolor=None,
        transparent=True,
    )
    draw_panel(
        out_dir / "compounded_mechanism_panel_bluebg",
        width_px=int(args.width),
        height_px=int(args.height),
        facecolor=str(args.bg_color),
        transparent=False,
    )
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
