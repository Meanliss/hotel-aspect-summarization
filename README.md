# Hotel Aspect Summarization

Graduation thesis repository for hotel aspect-based opinion summarization.

The repository contains the implementation, curated evaluation summaries, thesis
LaTeX source, and a small dashboard prototype. Large local artifacts such as raw
datasets, checkpoints, generated outputs, logs, and judge caches are excluded
from Git.

## Contents

| Path | Description |
| --- | --- |
| `src/` | Core model and inference code. |
| `scripts/` | Data preparation, evaluation, export, and figure utilities. |
| `data/` | Small taxonomy, seed, and tokenizer assets. |
| `reports/` | Curated ROUGE, sweep, and judge-metric summaries. |
| `paper/` | Thesis LaTeX source, references, tables, and figures. |
| `web/` | Static Next.js dashboard prototype. |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_abstractive.txt
```

Build the dashboard:

```powershell
cd web
npm install
npm run build
```

Build the thesis:

```powershell
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Notes

- The current Git tree is a lightweight submission package.
- Reproduction runs should regenerate ignored checkpoints, outputs, caches, and
  logs locally.
- Upstream SemAE attribution is preserved through the bibliography and license.
