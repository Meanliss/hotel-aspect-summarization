# Hotel Aspect Summarization Thesis

This repository contains the source code, thesis draft, curated evaluation
tables, and web dashboard prototype for a graduation thesis on hotel
aspect-based opinion summarization.

The work studies two complementary approaches:

- **UASum**: a structured evidence pipeline for aspect and sentiment-aware
  hotel review summarization.
- **SemAE on HASOS**: a SemAE-based extractive evidence selector adapted to
  hotel aspect summarization, with M1-M4 variants for evidence selection,
  synthesis, thresholding, and evaluation.

The repository is prepared as a lightweight thesis submission package. Large
local artifacts such as checkpoints, raw generated outputs, request caches, and
full training logs are intentionally excluded from Git.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | Core SemAE model, dataset, and inference utilities. |
| `scripts/` | Training, inference, conversion, evaluation, and thesis figure helpers. |
| `data/` | Small taxonomy, seed, and tokenizer assets required by the included code. |
| `reports/` | Curated metric summaries and sweep tables used in the thesis. |
| `paper/` | LaTeX thesis source, references, tables, and final figures. |
| `web/` | Next.js dashboard prototype for inspecting hotel aspect summaries. |

## What Is Included

- Source code for training, inference, evaluation, and report generation.
- HASOS aspect taxonomy and seed lexicons.
- SentencePiece tokenizer files needed by the SemAE pipeline.
- Curated ROUGE, operational, and LLM-judge metric summaries.
- The thesis LaTeX source split into chapter files.
- Final method and evaluation figures referenced by the thesis.
- A dashboard prototype with small public data files.

## What Is Excluded

The following files are kept local and ignored to keep the GitHub repository
submission-friendly:

- virtual environments and dependency folders;
- raw datasets and CSV exports;
- model checkpoints and downloaded pretrained models;
- generated output directories with per-entity/per-aspect files;
- LLM judge request caches and raw JSONL judgment datasets;
- LaTeX build products and temporary Overleaf upload packages.

## Setup

Create a Python environment and install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_abstractive.txt
```

For the web dashboard:

```powershell
cd web
npm install
npm run build
```

## Thesis Source

The thesis source is in `paper/`.

```powershell
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

A local TeX distribution with Vietnamese support is required.

## Reproducibility Notes

This repository keeps the code and compact evidence needed to inspect the
methodology and thesis results. Full model checkpoints, raw output dumps, and
large intermediate artifacts should be regenerated from the scripts or stored
outside Git when a complete reproduction package is needed.

## License

This project preserves the upstream license in `LICENSE`.
