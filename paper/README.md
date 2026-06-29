# Paper Draft

This folder contains a clean LaTeX rewrite of the thesis report with the second
method integrated:

- Method 1: LLM/API pipeline already described in the original draft.
- Method 2: SPACE-trained SemAE inference on HASOS with M1-M4 variants,
  threshold/token sweeps, ROUGE evaluation, and DeepSeek judge metrics.

## Files

- `main.tex`: full report draft.
- `overleaf_latest_patched.tex`: latest Overleaf-ready draft with the M1-M4
  workflow, M1 token-budget sensitivity table, and interpretation guidance.
- `overleaf_chapter3_5_replacement.tex`: standalone replacement snippet for the
  method and evaluation chapters.
- `tables/metrics.tex`: paper-ready tables derived from repository reports.
- `figures/README.md`: notes for the inline TikZ figures used in `main.tex`.

## Build

For Overleaf, upload or paste `overleaf_latest_patched.tex` as the working
`main.tex` file and recompile there.

For local builds, install a LaTeX distribution with Vietnamese support, then run
one of:

```powershell
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

or, if you adapt the preamble to XeLaTeX fonts:

```powershell
cd paper
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

The current workspace did not have `pdflatex` or `xelatex` available at creation
time, so PDF compilation must be done after installing TeX locally.
