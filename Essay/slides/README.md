# Slides (Beamer)

This folder contains a Beamer slide deck (`slides.tex`) for presenting the paper to collaborators.

## Overleaf usage (recommended)

Option A (simplest):
1. Create a new Overleaf project (Beamer).
2. Upload the following into the project root:
   - `slides.tex` (this file)
   - `figures/` (copy from `Essay/figures/`)
   - `figures_supp/` (copy from `Essay/figures_supp/`)
3. Set **Main document** to `slides.tex`.
4. Compile with pdfLaTeX.

Option B (upload the whole `Essay/` folder):
1. Upload `Essay/slides/slides.tex` and the `Essay/figures/` + `Essay/figures_supp/` folders.
2. Set **Main document** to `Essay/slides/slides.tex`.

`slides.tex` is written to be robust to both layouts.

## Notes

- The mechanism slide uses `csdag.svg` (copied to `figures/csdag.svg`) via `\includesvg`; you must enable **shell escape** on Overleaf, otherwise compilation will fail (intended).
- Main talk is designed for ~20 minutes; backup slides are included under `\\appendix`.
