# Thesis Source

LaTeX source for the thesis report.

## Build

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Requires a local TeX distribution with Vietnamese language support.
