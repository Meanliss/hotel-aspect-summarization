# Hotel Aspect Summarization

Graduation thesis repository for hotel aspect-based opinion summarization.

## Contents

| Path | Description |
| --- | --- |
| `src/` | Core model and inference code. |
| `scripts/` | Data preparation, evaluation, export, and utility scripts. |
| `data/` | Small taxonomy, seed, and tokenizer assets. |
| `reports/` | Curated evaluation summaries. |
| `paper/` | Thesis LaTeX source and final figures. |
| `web/` | Static dashboard prototype. |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_abstractive.txt
```

Large local artifacts such as checkpoints, raw datasets, generated outputs,
logs, and caches are intentionally excluded from Git.
