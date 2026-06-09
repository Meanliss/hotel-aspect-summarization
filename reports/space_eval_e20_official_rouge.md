# SPACE Official SemAE ROUGE Evaluation — `space_eval_e20`

This run evaluates the SPACE-trained epoch-20 SemAE checkpoint on the original
SPACE benchmark, using the official SemAE/pyrouge path rather than HASOS
reference-free metrics.

## Artifact Gate

| Artifact | Status | Path |
| --- | --- | --- |
| Checkpoint | OK | `checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_20_model.pt` |
| SPACE tokenizer | OK | `data/sentencepiece/space_unigram_32k.model` |
| SPACE summaries | OK | `data/space/json/space_summ.json` |
| SPACE gold summaries | OK | `data/space/gold` |
| pyrouge / ROUGE-1.5.5 | OK | `downloads/rouge_setup/ROUGE-1.5.5` |

`space_summ.json` was verified against the official SPACE Google Drive archive
from `stangelid/qt`; SHA256 matched exactly:

```text
958350e28c529c0143eb8818cabce451524e88a88a12908dca6c34a70dcf13a2
```

Gold summaries were generated from the official `space_summ.json` using the
repo utility `src/utils/json-to-dirs.py`, producing:

```text
6 aspect folders x 50 entities x 3 references = 900 aspect gold files
```

The `general` summaries also exist in the JSON, but this run is official aspect
evaluation over the six SPACE aspects: `building`, `cleanliness`, `food`,
`location`, `rooms`, `service`.

## Command

```bash
cd /home/llm/llm/tesing/src

/home/llm/miniconda3/envs/vllm_qwen312/bin/python -u aspect_inference.py \
  --summary_data ../data/space/json/space_summ.json \
  --gold_data ../data/space/gold \
  --sentencepiece ../data/sentencepiece/space_unigram_32k.model \
  --seedsdir ../data/seeds \
  --gold_aspects building,cleanliness,food,location,rooms,service \
  --model ../checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_20_model.pt \
  --run_id space_eval_e20 \
  --gpu 0 \
  --sample_sentences
```

Important: this was run without `--no_eval`, so `pyrouge` produced:

```text
outputs/eval_space_eval_e20.json
outputs/eval_space_eval_e20.txt
```

Copies for GitHub:

```text
reports/space_eval_e20_official_rouge.json
reports/space_eval_e20_official_rouge.txt
reports/space_eval_e20_official_rouge.log
```

## Output Health

| Check | Value |
| --- | ---: |
| Aspect output files | 300 |
| Empty output files | 0 |
| `building` files | 50 |
| `cleanliness` files | 50 |
| `food` files | 50 |
| `location` files | 50 |
| `rooms` files | 50 |
| `service` files | 50 |
| Fatal log matches | 0 |

Fatal log grep used:

```text
Traceback|Error|Exception|Cannot|Can't|No such|failed|FAIL|CUDA|RuntimeError
```

## ROUGE F1 by Split and Aspect

### Dev

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.28725 | 0.06653 | 0.20737 |
| cleanliness | 0.27100 | 0.08440 | 0.22622 |
| food | 0.28764 | 0.05630 | 0.19063 |
| location | 0.35313 | 0.10531 | 0.23191 |
| rooms | 0.34705 | 0.09167 | 0.23710 |
| service | 0.29477 | 0.09183 | 0.22371 |
| **Macro** | **0.30681** | **0.08267** | **0.21949** |

### Test

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.24752 | 0.05365 | 0.18305 |
| cleanliness | 0.25916 | 0.07509 | 0.20481 |
| food | 0.33286 | 0.12726 | 0.24688 |
| location | 0.32759 | 0.10673 | 0.23343 |
| rooms | 0.34250 | 0.08078 | 0.23113 |
| service | 0.29170 | 0.08374 | 0.20828 |
| **Macro** | **0.30022** | **0.08788** | **0.21793** |

### All

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.26558 | 0.05907 | 0.19387 |
| cleanliness | 0.26527 | 0.08034 | 0.21591 |
| food | 0.30896 | 0.09167 | 0.21832 |
| location | 0.34134 | 0.10718 | 0.23335 |
| rooms | 0.34411 | 0.08686 | 0.23451 |
| service | 0.29434 | 0.08879 | 0.21714 |
| **Macro** | **0.30327** | **0.08565** | **0.21885** |

## Notes

- This run uses SPACE, not HASOS.
- This is the official ROUGE evaluation path from SemAE: generated summaries are
  compared against human gold summaries with `pyrouge`.
- The `data/space/gold` folder is regenerated from `space_summ.json`; it is not
  committed because `data/space/` is ignored.
- The raw output summary tree remains local under `outputs/space_eval_e20/`.

