"""
生成 Essay/essay.tex 需要的 Supplementary 外部引用 aux 文件。

背景：
- 主文通过 xr-hyper 的 \\externaldocument 读取 supplementary.aux，以便在“主文/补充材料分开编译”
  的情况下仍能得到 “Supplementary Fig.~S#” 的正确编号。
- Overleaf 的构建产物（.aux）不会作为输入文件暴露给另一份主文件编译，因此需要把一个
  “可被读取的 supplementary.aux” 放在项目里（作为源文件）并保持更新。

用法：
  /home/wujlin/miniconda3/envs/emotion/bin/python scripts/generate_supplementary_aux.py
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPP_TEX = ROOT / "Essay" / "supplementary.tex"
OUT_AUX = ROOT / "Essay" / "supplementary.aux"


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


def main() -> None:
    if not SUPP_TEX.exists():
        raise SystemExit(f"未找到：{SUPP_TEX}")

    text = SUPP_TEX.read_text(encoding="utf-8", errors="ignore")
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

    OUT_AUX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {OUT_AUX} with {len(labels)} figure labels")


if __name__ == "__main__":
    main()
