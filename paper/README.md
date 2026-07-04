# Thesis Source

LaTeX source for the graduation thesis.

## Files

| Path | Description |
| --- | --- |
| `main.tex` | Entry point, preamble, front matter, and chapter includes. |
| `Chap_1.tex` - `Chap_5.tex` | Thesis chapters. |
| `references.bib` | Bibliography. |
| `tables/metrics.tex` | Final metric tables. |
| `figures/method2/` | Final Method 2 figures. |

## Build

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Requires a local TeX distribution with Vietnamese language support.
