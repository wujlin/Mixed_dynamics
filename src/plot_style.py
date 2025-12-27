"""
论文级可视化统一样式（Nature/PNAS 风格取向，偏简洁、可打印）。

目标：
- 统一字体（sans-serif）、字号、线宽、配色、图例与网格
- 统一导出参数，避免子图错位与多余留白

说明：
- 不依赖 seaborn，保持最小依赖与可控性
- 通过 rc_context 提供“可局部启用”的样式上下文
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


# Okabe–Ito 色盲友好配色（推荐用于期刊图）
OKABE_ITO: Dict[str, str] = {
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


def paper_rcparams(font_scale: float = 1.0) -> Dict[str, object]:
    base = {
        # 字体：优先用通用 sans-serif；数学公式使用匹配的 sans mathtext
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        # 线条/边框：适配打印与缩放
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
        "patch.linewidth": 1.0,
        "axes.linewidth": 1.0,
        # 字号：以单栏图为基准（后续可按需要微调）
        "axes.titlesize": 10.0 * font_scale,
        "axes.labelsize": 9.0 * font_scale,
        "xtick.labelsize": 8.0 * font_scale,
        "ytick.labelsize": 8.0 * font_scale,
        "legend.fontsize": 8.0 * font_scale,
        # 网格：保持克制（可读但不过抢）
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        # 导出：固定 DPI，并使用 constrained layout 保持版面一致
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
    return base


@contextmanager
def paper_style(font_scale: float = 1.0) -> Iterator[None]:
    """在 with 作用域内启用论文统一样式。"""

    with mpl.rc_context(paper_rcparams(font_scale=font_scale)):
        yield


def save_figure(
    fig: mpl.figure.Figure,
    out_path: str | Path,
    *,
    dpi: Optional[int] = None,
) -> None:
    """
    统一保存入口：不使用 bbox_inches='tight'，避免不同文本元素导致的
    bounding box 抖动，从而引发 LaTeX 子图错位。
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)


def despine(ax: mpl.axes.Axes) -> None:
    """轻量去除上/右边框，保持期刊图常见风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

