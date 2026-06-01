# SemAE — HASOS English-Only Hotel Review Scoring

Aspect-based opinion summarization on Vietnamese hotel review data, filtered to **English-only** content, using the HASOS aspect taxonomy.

This repository extends the [Semantic Autoencoder (SemAE)](README_ORIGINAL.md) (ACL 2022) pipeline with a HASOS-specific scoring path that reports **ASC** (Aspect Summary Coverage) and **CEC** (Cluster Evidence Coverage) instead of ROUGE — because the input CSVs contain reviews only, with no human-written gold summaries.

> Upstream SemAE README is preserved at [README_ORIGINAL.md](README_ORIGINAL.md).
> HASOS-specific instructions are in [README_HASOS.md](README_HASOS.md).

---

## Results — English-only run across 3 CSV files

ROUGE: **not available** (no gold summaries in source data).

| File | Total rows | English reviews | English ratio | English sentences | Matched aspects | ASC | Macro CEC | Weighted CEC | Top aspects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hotel_review1.csv` | 607,260 | 270,658 | 44.57% | 850,307 | 29/29 | **0.7337** | **0.5583** | **0.5624** | AM_POOL, EXP_OVERALL, FAC_BATH, SER_ATTITUDE, FAC_BUILDING |
| `hotel_review2.csv` | 1,150,415 | 602,919 | 52.41% | 1,933,276 | 29/29 | **0.7335** | **0.5695** | **0.5740** | FAC_VIEW_LOCATION, SER_ATTITUDE, AM_FOOD, FAC_ROOM, FAC_BUILDING |
| `hotel_review3.csv` | 591,023 | 538,515 | 91.12% | 3,146,670 | 29/29 | **0.7539** | **0.5713** | **0.5728** | EXP_OVERALL, FAC_BUILDING, AM_POOL, AM_FOOD, FAC_BATH |

**Interpretation**

- **ASC ≈ 0.73–0.75** — ~73–75% of the 29 HASOS aspects are covered by the extracted summary. Acceptable, but inflated by cross-aspect contamination (see issues per-file).
- **CEC ≈ 0.55–0.57** — each representative sentence covers only ~55% of the cluster's actual evidence. This is the main weakness — a TF-IDF-centroid pipeline (current implementation) cannot discriminate semantically similar aspects.
- Detailed per-hotel logs and **Pipeline Issues Found** sections are in each `outputs/hasos_english_only/hotel_reviewN/top10_hotels_pipeline_log.md`.

---

## Per-file outputs

Each folder contains: `file_scores.csv/json`, `file_summaries.txt`, `filtered_english_stats.json`, and a top-10 hotel pipeline trace.

- [outputs/hasos_english_only/hotel_review1/](outputs/hasos_english_only/hotel_review1/) — [pipeline log](outputs/hasos_english_only/hotel_review1/top10_hotels_pipeline_log.md)
- [outputs/hasos_english_only/hotel_review2/](outputs/hasos_english_only/hotel_review2/) — [pipeline log](outputs/hasos_english_only/hotel_review2/top10_hotels_pipeline_log.md)
- [outputs/hasos_english_only/hotel_review3/](outputs/hasos_english_only/hotel_review3/) — [pipeline log](outputs/hasos_english_only/hotel_review3/top10_hotels_pipeline_log.md)
- Cross-file summary: [outputs/hasos_english_only/summary_all_files.md](outputs/hasos_english_only/summary_all_files.md)

---

## Pipeline

```mermaid
flowchart LR
    A[Raw CSV] --> B[Group by hotel_id]
    B --> C[English language filter]
    C --> D[Sentence split + normalize]
    D --> E[HASOS aspect match<br/>keyword + taxonomy]
    E --> F[Dedup per aspect<br/>unique_opinions]
    F --> G[Cluster weight<br/>log(1+n) normalized]
    G --> H[TF-IDF centroid<br/>representative sentence]
    H --> I[CEC / ASC metrics]
```

**Components**

- Data prep & validation: [scripts/prepare_hasos.py](scripts/prepare_hasos.py), [scripts/validate_hasos.py](scripts/validate_hasos.py)
- English-only scoring: [scripts/run_english_only_all_files.py](scripts/run_english_only_all_files.py)
- Per-file scorer: [scripts/score_hotel_file.py](scripts/score_hotel_file.py)
- Top-10 hotel pipeline tracer: [scripts/log_10_hotels_pipeline.py](scripts/log_10_hotels_pipeline.py)
- Model (for full SemAE training, not used in this scoring run): [src/semae.py](src/semae.py), [src/train.py](src/train.py)

---

## How to reproduce

Raw CSVs (`hotel_review1.csv`, `hotel_review2.csv`, `hotel_review3.csv`) are **not** in this repo (too large). Place them one level above `SemAE/`, then run:

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\scripts
python .\run_english_only_all_files.py
# Per-file top-10 hotel detailed log:
python .\log_10_hotels_pipeline.py --input-csv ..\..\hotel_review1.csv --limit 10
```

Dependencies: `pip install -r requirements.txt` (Python 3.6+).

---

## Pipeline health summary

| Stage | Health | Notes |
| --- | :---: | --- |
| Load + group | ✅ | All rows ingested, `hotel_id` derived from `ref_id` |
| English filter | ✅ | Detector rule is conservative (multiple positive signals required) |
| Sentence split | ✅ | Standard splitter, language-aware |
| Aspect match | ⚠️ | Keyword-only — misses paraphrase; single sentence often matches 3–6 aspects (over-matching) |
| Dedup | ✅ | Per-aspect unique opinion count is stable |
| Cluster weight | ✅ | Log-normalized, well-bounded |
| Representative sentence | ❌ | **Same sentence picked for multiple aspects** (TF-IDF centroid is not aspect-discriminative). See *Pipeline Issues Found* in each per-file log. |
| CEC/ASC scoring | ⚠️ | Numerically correct, but inherits noise from step 8 |

---

## License

See [LICENSE](LICENSE).
