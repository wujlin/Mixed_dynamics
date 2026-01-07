"""
生成 Essay/essay.tex 需要的 Supplementary 外部引用 aux 文件。

背景：
- 主文通过 xr-hyper 的 \\externaldocument 读取 supplementary.aux，以便在“主文/补充材料分开编译”
  的情况下仍能得到 “Supplementary Fig.~S#” 的正确编号。
- Overleaf 的构建产物（.aux）不会作为输入文件暴露给另一份主文件编译，因此需要把一个
  “可被读取的 supplementary.aux” 放在项目里（作为源文件）并保持更新。

用法：
  python scripts/generate_supplementary_aux.py

  # 指定输入/输出（例如 NC 打包目录）
  python scripts/generate_supplementary_aux.py \\
    --supp-tex Essay_nc/supplementary.tex \\
    --out-aux Essay_nc/supplementary.aux
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUPP_TEX = ROOT / "Essay" / "supplementary.tex"
DEFAULT_OUT_AUX = ROOT / "Essay" / "supplementary.aux"


def _iter_figure_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    pos = 0
    while True:
        start = text.find(r"\begin{figure", pos)
        if start < 0:
            break
        end = text.find(r"\end{figure}", start)
        if end < 0:
            break
        end += len(r"\end{figure}")
        blocks.append(text[start:end])
        pos = end
    return blocks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 supplementary.tex 生成可被主文读取的 supplementary.aux。")
    parser.add_argument(
        "--supp-tex",
        type=Path,
        default=DEFAULT_SUPP_TEX,
        help=f"supplementary.tex 路径（默认：{DEFAULT_SUPP_TEX}）",
    )
    parser.add_argument(
        "--out-aux",
        type=Path,
        default=DEFAULT_OUT_AUX,
        help=f"输出 supplementary.aux 路径（默认：{DEFAULT_OUT_AUX}）",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    supp_tex = args.supp_tex
    out_aux = args.out_aux

    if not supp_tex.exists():
        raise SystemExit(f"未找到：{supp_tex}")

    text = supp_tex.read_text(encoding="utf-8", errors="ignore")
    blocks = _iter_figure_blocks(text)

    labels: list[str] = []
    for blk in blocks:
        # TeX: \label{...}  (curly braces are literal, so escape with \{ \} in regex)
        m = re.search(r"\\label\{([^}]+)\}", blk)
        if m:
            labels.append(m.group(1).strip())

    # 仅为 figure label 生成 S1, S2, ...；page/anchor 字段留空或占位即可满足 \ref。
    lines: list[str] = [
        r"\relax",
        r"\providecommand\hyper@newdestlabel[2]{}",
        r"\providecommand\HyField@AuxAddToFields[1]{}",
        r"\providecommand\HyField@AuxAddToCoFields[2]{}",
    ]
    for idx, lab in enumerate(labels, start=1):
        fig_no = f"S{idx}"
        lines.append(rf"\newlabel{{{lab}}}{{{{{fig_no}}}{{0}}{{}}{{figure.{fig_no}}}{{}}}}")

    out_aux.parent.mkdir(parents=True, exist_ok=True)
    out_aux.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_aux} with {len(labels)} figure labels")


if __name__ == "__main__":
    main()
