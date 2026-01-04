# Slides (Beamer)

This folder contains a Beamer slide deck (`slides.tex`) for presenting the paper to collaborators.

## Overleaf usage (recommended)

Option A (using this repo layout):
1. Upload the entire `Essay/` folder to Overleaf (or at least `Essay/slides/`, `Essay/figures/`, and `Essay/figures_supp/`).
2. Set **Main document** to `Essay/slides/slides.tex`.
3. Compile with pdfLaTeX.

## Notes

- The mechanism slide uses `Essay/figures/csdag.svg` via `\includesvg`; you must enable **shell escape** on Overleaf, otherwise compilation will fail (intended).
- Main talk is designed for ~20 minutes; backup slides are included under `\\appendix`.
