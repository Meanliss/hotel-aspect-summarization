# Hotel Aspect Summarization

Minimal thesis repository for training and evaluating hotel aspect-based
summarization models.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements_abstractive.txt
```

## Prepare Data

Place the training data locally:

```text
data/space/json/space_train.json
data/sentencepiece/space_unigram_32k.model
data/sentencepiece/space_unigram_32k.vocab
```

For HASOS inputs:

```powershell
python scripts\prepare_hasos.py
python scripts\validate_hasos.py
```

## Train

Small/local run:

```powershell
cd scripts
bash train_space.sh
```

Full SPACE run:

```powershell
bash scripts/run_space_full_vast.sh
```

Useful overrides:

```powershell
$env:RUN_ID="space_full_11402x20"
$env:EPOCHS="20"
$env:WORKDIR="/workspace/SemAE"
```

Large datasets, checkpoints, logs, caches, and generated outputs are ignored by
Git and should be regenerated locally.
