# Thesis Paper

This folder contains the LaTeX source for the graduation thesis report.

## Main Files

| Path | Purpose |
| --- | --- |
| `main.tex` | Thesis entry point, preamble, front matter, and chapter includes. |
| `Chap_1.tex` - `Chap_5.tex` | Main thesis chapters. |
| `references.bib` | Bibliography database. |
| `tables/metrics.tex` | Paper-ready metric tables generated from curated reports. |
| `figures/method2/` | Final Method 2 figures referenced by the report. |

Temporary backup folders, candidate image generations, Overleaf upload bundles,
and LaTeX build products are intentionally ignored by Git.

## Build

Install a LaTeX distribution with Vietnamese support, then run:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If your local environment uses XeLaTeX fonts, adapt the preamble first and run
the equivalent `xelatex` sequence.
