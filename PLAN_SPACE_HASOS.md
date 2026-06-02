# Plan: SPACE training + HASOS-aspect inference + sentiment split

Standalone execution plan for resuming this work on another machine. All paths
are workspace-relative to `SemAE/`.

---

## Goal

1. **Train** SemAE on the SPACE TripAdvisor corpus (large, generic English hotel domain).
2. **Inference** with the **custom HASOS 29-aspect taxonomy** (your `data/seeds_hasos/`)
   on the 50 HASOS hotel entities (`data/hasos/hasos_summ.json`).
3. Produce **two parallel output trees**:
   - `outputs/<run_id>/<aspect>/<split>_<id>` — mixed pos/neg/neu sentences
   - `outputs/<run_id>_sentiment/<aspect>__<pos|neg|neu>/<split>_<id>` — split files
4. Score everything with the 8 reference-free metrics + BERTScore, then commit & push.

> SPACE = training only. HASOS = the actual target (29 aspects × 50 entities).
> No ROUGE (HASOS has no gold). Result directly comparable to existing
> `outputs/hasos_aspects_run1/` baseline.

---

## Current status (resume point)

| Phase | Status | Notes |
| --- | --- | --- |
| **A. Data** | ✅ DONE | `data/space/json/space_train.json` 1.18 GB, 11,402 entities, 1,135,806 reviews. Bonus: `space_summ.json` (5 MB) + `space_summ_splits.txt` |
| **B.1. SPM** | ✅ DONE | `data/sentencepiece/space_unigram_32k.{model,vocab}` — 8 min wall time |
| **B.2. Train SemAE** | ⏸ NEEDS DECISION | **Reality check**: full 11,402 ent × 20 epoch ≈ 5 DAYS on RTX 3500 Ada. Smoke test (50 ent × 1 epoch) = 2 min 25 s. See "Training scope" below. |
| C. Sentiment | NOT STARTED | cardiffnlp/twitter-roberta-base-sentiment-latest |
| D. Inference | BLOCKED ON B.2 |  |
| E. Score | BLOCKED ON D |  |
| F. Commit/push | BLOCKED ON E |  |

---

## Training scope — decision needed before B.2

The original qt/SemAE paper README example uses `--max_num_entities 500`. Full
corpus on RTX 3500 Ada will take ~5 days, not overnight. Pick one:

| Option | Entities | Epochs | Wall time (est.) | When to choose |
| --- | ---: | ---: | --- | --- |
| **A** | 500 | 20 | **~7 h overnight** | Matches qt README recipe, recommended. |
| B | 1000 | 20 | ~14 h | If you can wait 2 nights. |
| C | 500 | 10 | ~3.5 h | Mirror of HASOS recipe; fastest viable. |
| D | 11,402 | 20 | **~5 DAYS** | Only if you have dedicated training rig. |

Recommendation: **Option A** (500 × 20).

---

## Step-by-step commands

### Phase A — Data (already done, command preserved for reproducibility)

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
pip install gdown
New-Item -ItemType Directory -Force -Path data\space\json | Out-Null
# 405 MB tar.gz from Google Drive (qt repo link). Corporate SSL needs --no-check-certificate.
python -m gdown --no-check-certificate "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" `
    -O data\space\json\space_train.json
# Downloaded file is a TAR archive, not raw json — extract:
Move-Item data\space\json\space_train.json data\space\json\space_train.tar -Force
python -c "import tarfile; t=tarfile.open(r'data/space/json/space_train.tar','r'); t.extractall(r'data/space/json/'); t.close()"
Remove-Item data\space\json\space_train.tar -Force
# Result: space_train.json (1.18 GB), space_summ.json (5 MB), space_summ_splits.txt
```

### Phase B.1 — SentencePiece (already done)

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE\src\utils
$env:PYTHONIOENCODING='utf-8'
python train-spm.py ..\..\data\space\json\space_train.json ..\..\data\sentencepiece\space_unigram_32k
# ~8 min wall time on RTX 3500 Ada (CPU-bound)
```

### Phase B.2 — Train SemAE (Option A recommended)

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
New-Item -ItemType Directory -Force -Path logs | Out-Null
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'
# Option A: 500 entities × 20 epochs (~7 h)
python src\train.py `
    --data data\space\json\space_train.json `
    --sentencepiece data\sentencepiece\space_unigram_32k.model `
    --max_num_entities 500 `
    --run_id space_run1 `
    --gpu 0 `
    --epochs 20 2>&1 | Tee-Object -FilePath logs\train_space.log
# Output: models\space_run1_<1..20>_model.pt (~70 MB each = 1.4 GB total)
```

> Wrapper script alternative: `scripts/train_space.ps1 -Gpu 0 -Epochs 20 -RunId space_run1`
> (already created; defaults to full corpus — pass `--max_num_entities` manually
> by editing or running `train.py` directly as above.)

### Phase C — Sentiment classifier (parallel with B.2)

#### C.1 Create `src/utils/sentiment.py`

```python
"""Thin wrapper around cardiffnlp/twitter-roberta-base-sentiment-latest.
Returns 3-way {pos, neu, neg} labels."""
from functools import lru_cache
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# cardiffnlp label index mapping
_ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}


class SentimentScorer:
    def __init__(self, device: str = "cuda:0", fp16: bool = True, max_length: int = 256):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(_MODEL_ID)
        if fp16 and device.startswith("cuda"):
            self.model = self.model.half()
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict_batch(self, texts: List[str], batch_size: int = 64) -> List[Dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)
            best_idx = probs.argmax(dim=-1).tolist()
            best_score = probs.max(dim=-1).values.tolist()
            for idx, score in zip(best_idx, best_score):
                out.append({"label": _ID2LABEL[idx], "score": float(score)})
        return out


@lru_cache(maxsize=1)
def get_default_scorer() -> SentimentScorer:
    return SentimentScorer()
```

#### C.2 Smoke test `scripts/smoke_test_sentiment.py`

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.sentiment import SentimentScorer

s = SentimentScorer()
samples = [
    "The room was spacious and the staff super friendly.",          # pos
    "Terrible noise all night and the bed was broken.",              # neg
    "Check-in took about 15 minutes.",                               # neu
    "Loved every minute of our stay, will be back!",                 # pos
    "The buffet was bland and overpriced.",                          # neg
]
for txt, pred in zip(samples, s.predict_batch(samples)):
    print(f"[{pred['label']:<3s} {pred['score']:.2f}] {txt}")
```

Run:
```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
python scripts\smoke_test_sentiment.py
```
Expected output: pos/neg/neu/pos/neg labels.

### Phase D — Aspect inference with sentiment split

#### D.1 Modify `src/aspect_inference.py`

Find the file-write block (around line 426-475). Currently it writes one file per
(aspect, entity). Add:

1. **New CLI flag** (in argparse, around line 150 with the other out args):
   ```python
   out_arg_group.add_argument('--sentiment_split', action='store_true',
                              help='also write per-sentiment files in <output_path>_sentiment/')
   ```

2. **Before the per-aspect write loop**, load scorer once:
   ```python
   if args.sentiment_split:
       sys.path.insert(0, os.path.dirname(__file__))
       from utils.sentiment import SentimentScorer
       sentiment_scorer = SentimentScorer(device=f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu')
       sentiment_root = output_path + '_sentiment'
       os.makedirs(sentiment_root, exist_ok=True)
   else:
       sentiment_scorer = None
   ```

3. **Inside the write loop**, after `summary_sentences = truncate_summary(...)` and
   before `fout.write(delim.join(summary_sentences))`, add:
   ```python
   # existing mixed write — unchanged
   fout = open(file_path, 'w', encoding='utf-8')
   fout.write(delim.join(summary_sentences))
   fout.close()

   # NEW: per-sentiment split
   if sentiment_scorer is not None and summary_sentences:
       preds = sentiment_scorer.predict_batch(summary_sentences)
       buckets = {'pos': [], 'neg': [], 'neu': []}
       for sent, pred in zip(summary_sentences, preds):
           buckets[pred['label']].append(sent)
       split_prefix = 'dev_' if entity_id in summ_dataset.dev_entity_ids else 'test_'
       for label, sents in buckets.items():
           sent_dir = os.path.join(sentiment_root, f'{aspect}__{label}')
           os.makedirs(sent_dir, exist_ok=True)
           with open(os.path.join(sent_dir, split_prefix + entity_id), 'w', encoding='utf-8') as fsent:
               fsent.write(delim.join(sents))
   ```

#### D.2 Create `scripts/run_space_hasos_aspect_parallel.py`

Copy of `scripts/run_aspect_inference_parallel.py` with these changes in `build_cmd`:

```python
def build_cmd(args, shard_idx, shard_run_id):
    return [
        sys.executable, "-u", str(SRC_DIR / "aspect_inference.py"),
        "--summary_data", str(DATA_DIR / "hasos" / "hasos_summ.json"),
        "--sentencepiece", str(DATA_DIR / "sentencepiece" / "space_unigram_32k.model"),  # SPACE SPM
        "--seedsdir", str(DATA_DIR / "seeds_hasos"),                                     # HASOS seeds
        "--gold_aspects", args.gold_aspects,
        "--model", args.model,                                                           # SPACE model
        "--run_id", shard_run_id,
        "--gpu", str(args.gpu),
        "--max_tokens", str(args.max_tokens),
        "--shard_idx", str(shard_idx),
        "--num_shards", str(args.num_shards),
        "--sample_sentences",                                                            # paper recipe
        "--sentiment_split",                                                             # NEW
        "--no_eval",
    ]
```

Also update `aspect_codes_from_taxonomy` path default to `data/hasos/aspect_taxonomy.tsv`.

#### D.3 Run

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
$env:PYTHONIOENCODING='utf-8'
python scripts\run_space_hasos_aspect_parallel.py `
    --model models\space_run1_20_model.pt `
    --run_id space_hasos_aspects_run1 `
    --num_shards 4 `
    --gpu 0 `
    --max_tokens 40
# Wall time: ~30-45 min for 50 entities × 29 aspects × 4 shards
# (sentiment adds ~10 min for cardiffnlp classification)
```

#### Verification

```powershell
(Get-ChildItem outputs\space_hasos_aspects_run1\FAC_ROOM).Count             # → 50
(Get-ChildItem outputs\space_hasos_aspects_run1).Count                       # → 29
(Get-ChildItem outputs\space_hasos_aspects_run1_sentiment).Count             # → 87 (29 × 3)
(Get-ChildItem outputs\space_hasos_aspects_run1_sentiment\FAC_ROOM__pos).Count  # → 50 (some may be empty)
```

### Phase E — Scoring

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
$env:PYTHONIOENCODING='utf-8'
python scripts\summarize_aspect_outputs.py --run_id space_hasos_aspects_run1
python scripts\score_semae_run.py --run_id space_hasos_aspects_run1 --bert_score
```

Outputs:
- `outputs/space_hasos_aspects_run1_report.md` + `.json`
- `outputs/space_hasos_aspects_run1_metrics.md` + `.json`

**Compare to baseline** [outputs/hasos_aspects_run1_metrics.md](outputs/hasos_aspects_run1_metrics.md):

| Metric | hasos_run1 baseline | space_hasos_run1 (target) |
| --- | ---: | ---: |
| bert_f1_aspect | 0.8126 | ≥ 0.81 (expected ↑) |
| bert_f1_source | 0.8072 | ≥ 0.80 (expected ↑) |
| aspect_purity | 0.5348 | ≥ 0.55 (expected ↑) |
| self_bleu4 | 0.0085 | ≤ 0.01 (expected ↓) |

### Phase F — Commit & push

Update `.gitignore` whitelist:
```gitignore
!outputs/space_hasos_aspects_run1/
!outputs/space_hasos_aspects_run1/**
!outputs/space_hasos_aspects_run1_sentiment/
!outputs/space_hasos_aspects_run1_sentiment/**
!outputs/space_hasos_aspects_run1_report.md
!outputs/space_hasos_aspects_run1_report.json
!outputs/space_hasos_aspects_run1_metrics.md
!outputs/space_hasos_aspects_run1_metrics.json
```

Update [README.md](README.md): add "Pipeline 1B — SPACE-trained, HASOS-inferred"
section with comparison table + 1 sentiment-split sample.

```powershell
cd C:\Users\dso3hc\Downloads\Implement\SemAE
git add -A
git commit -m "SPACE-trained SemAE + HASOS-aspect inference with sentiment split (cardiffnlp)"
git push origin main
```

---

## File manifest

### Already exists / done
- `data/space/json/space_train.json` (1.18 GB)
- `data/space/json/space_summ.json` (5 MB, unused)
- `data/sentencepiece/space_unigram_32k.{model,vocab}`
- `data/hasos/hasos_summ.json` (50 entities — inference target)
- `data/hasos/aspect_taxonomy.tsv` (29 aspects)
- `data/seeds_hasos/*.txt` (29 seed files)
- `scripts/train_space.ps1` (wrapper, defaults to full corpus; override `--max_num_entities`)

### To create
- `src/utils/sentiment.py` (Phase C.1)
- `scripts/smoke_test_sentiment.py` (Phase C.2)
- `scripts/run_space_hasos_aspect_parallel.py` (Phase D.2)

### To modify
- `src/aspect_inference.py` — add `--sentiment_split` flag + dual-write (Phase D.1)
- `.gitignore` — whitelist new outputs (Phase F)
- `README.md` — add comparison section (Phase F)

---

## Hardware budget

- **Disk**: 1.2 GB (space_train.json) + 1.4 GB (20 checkpoints) + 5 MB outputs = ~3 GB free.
- **GPU 12 GB**: training uses ~3 GB; inference 4 shards × (SemAE ~1 GB + cardiffnlp 500 MB) = ~6 GB.
- **Training time** (RTX 3500 Ada): ~7 h overnight for Option A (500 ent × 20 ep).

---

## Decisions locked in

- Training data: SPACE only (1.1 M reviews available, will subsample).
- Inference: HASOS 29 aspects, 50 HASOS entities, custom seeds in `data/seeds_hasos/`.
- SPM: fresh from SPACE (NOT reuse hasos_unigram_32k.model).
- Sentiment: cardiffnlp/twitter-roberta-base-sentiment-latest, top-1 label, no threshold.
- Output: two parallel trees (mixed + sentiment-split).
- `--sample_sentences` ON (per paper).
- No ROUGE (HASOS has no gold).
- Excluded: SPACE 6-aspect inference, hyperparameter search, multi-GPU, Amazon.
