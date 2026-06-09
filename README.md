# SPACE-trained SemAE HASOS Report

This repository contains the SemAE training/inference code used to run the
SPACE-trained epoch-20 checkpoint on the HASOS hotel review benchmark.

Latest evaluated run:

```text
run_id: space_hasos_full_e20
checkpoint: checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_20_model.pt
taxonomy: HASOS 29 aspects
mode: aspect summary + sentiment split
metrics: reference-free metrics + BERTScore
```

## Status

The reference-free scoring run is complete. The original SemAE ROUGE evaluation
is not available for this HASOS run yet because the current HASOS source has
reviews but no human gold/reference summaries.

Validation shape:

| Item | Count |
| --- | ---: |
| Entities | 50 |
| Reviews | 5,000 |
| Sentences | 45,529 |
| HASOS aspects | 29 |

Generated outputs:

| Output | Count |
| --- | ---: |
| Aspect summary files | 1,450 |
| Sentiment-split files | 4,350 |
| Aspect line rows | 3,580 |
| Sentiment line rows | 3,580 |
| Sentence provenance rows | 3,580 |
| Empty aspect files retained | 80 |

The expected 1,450 aspect files are exactly `50 entities x 29 aspects`.
The sentiment tree contains `pos`, `neg`, and `neu` splits for each aspect/entity.

## Main Artifacts

The full generated outputs are intentionally ignored by git because they are
run artifacts. The repo tracks the code and a compact presentation report.

Tracked report deck:

- [reports/space_hasos_full_e20_report.pptx](reports/space_hasos_full_e20_report.pptx)
- [reports/space_hasos_stage_io.pptx](reports/space_hasos_stage_io.pptx)
- [reports/space_eval_e20_official_rouge.md](reports/space_eval_e20_official_rouge.md)
- [reports/space_old_aspects_e20_official_rouge.md](reports/space_old_aspects_e20_official_rouge.md)

GitHub release artifacts:

- [Release `space_hasos_full_e20`](https://github.com/Meanliss/tesing/releases/tag/space_hasos_full_e20)
- Checkpoint: `space_full_11402x20_stable_resume_e7_20_model.pt`
- Final result archive: `space_hasos_full_e20_results.tar.gz`
- Manifest/checksums: [artifacts/space_hasos_full_e20_release.md](artifacts/space_hasos_full_e20_release.md)

Local generated artifacts after running the pipeline:

```text
outputs/space_hasos_full_e20/
outputs/space_hasos_full_e20_sentiment/
outputs/space_hasos_full_e20_lines.jsonl
outputs/space_hasos_full_e20_lines.tsv
outputs/space_hasos_full_e20_aspect_sentiment_lines.jsonl
outputs/space_hasos_full_e20_aspect_sentiment_lines.tsv
outputs/space_hasos_full_e20_provenance.jsonl
outputs/space_hasos_full_e20_report.md
outputs/space_hasos_full_e20_report.json
outputs/space_hasos_full_e20_metrics.md
outputs/space_hasos_full_e20_metrics.json
outputs/space_hasos_full_e20_report.pptx
logs/space_hasos_full_e20_pipeline_trace.md
logs/space_hasos_full_e20_inference.log
logs/space_hasos_full_e20_score_outputs.log
```

## Original SemAE ROUGE Compatibility

The upstream SemAE repository evaluates generated summaries with `pyrouge`.
For aspect summarization, `src/aspect_inference.py` expects:

```text
system summaries: outputs/<run_id>/<aspect>/<dev|test>_<entity_id>
gold summaries:   data/<dataset>/gold/<aspect>/#ID#_[012].txt
output:           outputs/eval_<run_id>.txt
                  outputs/eval_<run_id>.json
```

That is the official SemAE comparison path. It cannot be computed on the
current HASOS file because `data/hasos/` contains taxonomy and review JSON only,
not gold summaries:

```text
data/hasos/hasos_summ.json
data/hasos/aspect_taxonomy.tsv
data/hasos/aspect_taxonomy.json
```

To compare exactly like the original repo, add HASOS gold summaries in pyrouge
format under `data/hasos/gold/<aspect>/`, then rerun aspect inference without
`--no_eval` and point `--gold_data` to `data/hasos/gold`.

The old/original SPACE baseline was rerun as `space_old_aspects_e20` with the
six original SPACE aspects and official ROUGE:

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| Dev macro | 0.30681 | 0.08267 | 0.21949 |
| Test macro | 0.30022 | 0.08787 | 0.21793 |
| All macro | 0.30327 | 0.08565 | 0.21885 |

## Reference-free Macro Metrics

HASOS does not provide human reference summaries, so ROUGE is not meaningful for
this run. The scoring uses reference-free extractive-summary metrics plus
BERTScore.

| Metric | Value | Interpretation |
| --- | ---: | --- |
| `source_fidelity` | 0.6094 | Fraction of summary sentences found verbatim in source reviews. Lower than 1.0 mainly because outputs are truncated by token budget. |
| `source_fidelity_excl_truncated` | 0.9692 | Same exact-match check, excluding sentences intentionally cut by `max_tokens`. |
| `aspect_keyword_coverage` | 0.7441 | Fraction of selected sentences containing at least one target-aspect or sentiment keyword. |
| `aspect_purity` | 0.5517 | Fraction of selected sentences whose strongest keyword match is the target aspect. Multi-aspect hotel sentences reduce this. |
| `distinct_1` | 0.2846 | Unique unigram ratio across summaries. |
| `distinct_2` | 0.7179 | Unique bigram ratio across summaries. |
| `self_bleu4` | 0.0109 | Pairwise BLEU-4 within the same aspect; lower means less template reuse. |
| `compression_ratio` | 0.0028 | Summary tokens divided by source-review tokens. |
| `avg_sentence_len` | 15.57 | Mean token length of selected sentences. |
| `cross_aspect_jaccard` | 0.1050 | Token overlap between aspect summaries of the same entity; lower means better separation. |
| `bert_f1_aspect` | 0.8098 | BERTScore-F1 between summary and target-aspect description. |
| `bert_f1_source` | 0.8074 | BERTScore-F1 between summary and source-review pool. |

## Best and Weakest Aspect Signals

Highest aspect purity:

| Aspect | Purity | Notes |
| --- | ---: | --- |
| `LOY_RECOMMEND` | 0.894 | Recommendation language is very explicit. |
| `SER_ATTITUDE` | 0.889 | Staff attitude vocabulary is strong and repeated. |
| `FAC_VIEW_LOCATION` | 0.815 | Location/view phrases are distinctive. |
| `FAC_ROOM` | 0.814 | Room-related phrases are common and clear. |
| `AM_FOOD` | 0.764 | Food/restaurant/breakfast signals are strong. |

Lowest aspect purity:

| Aspect | Purity | Likely reason |
| --- | ---: | --- |
| `BRA_REPUTE` | 0.258 | Reputation/brand language overlaps with overall experience. |
| `AM_UTILITY` | 0.338 | Utility words overlap with room amenities and service. |
| `EXP_EMOTION` | 0.345 | Emotion language is broad and often appears with overall satisfaction. |
| `EXP_OVERALL` | 0.372 | Overall summaries naturally borrow from many aspects. |
| `FAC_SECURITY` | 0.378 | Security appears sparsely and overlaps with safety/room issues. |

## Pipeline

The run uses the trained SemAE model as an extractive aspect summarizer:

1. Read HASOS entity-level hotel reviews.
2. Tokenize with the SPACE SentencePiece model used during training.
3. Encode review sentences with the trained SemAE encoder.
4. Build aspect prototypes from the fixed HASOS 29-aspect taxonomy/seeds.
5. Rank sentences for each `(entity, aspect)` by KL-divergence scoring.
6. Truncate selected sentences to the token budget.
7. Write aspect-only summaries.
8. Split selected aspect sentences into `pos`, `neg`, and `neu` buckets.
9. Export line-level JSONL/TSV.
10. Compute metrics, including BERTScore.
11. Build the Markdown/JSON/PPTX reports.

The aspect clusters are fixed by the HASOS taxonomy files:

```text
data/hasos/aspect_taxonomy.tsv
data/hasos/aspect_taxonomy.json
data/seeds_hasos/*.txt
```

They are not dynamic LLM labels.

## Reproduce

Run from the repository root:

```bash
conda activate vllm_qwen312
cd /home/llm/llm/tesing

python scripts/run_space_hasos_after_model.py \
  --run_id space_hasos_full_e20 \
  --source_json data/hasos/hasos_summ.source.json \
  --model checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_20_model.pt \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --gpu 0 \
  --num_shards 4 \
  --max_tokens 40
```

Do not pass `--skip_bert_score` if the final report must include BERTScore.

Expected checks:

```bash
find outputs/space_hasos_full_e20 -type f | wc -l
# 1450

find outputs/space_hasos_full_e20_sentiment -type f | wc -l
# 4350

grep -E "bert_f1_aspect|bert_f1_source" outputs/space_hasos_full_e20_metrics.md
```

## Notes

- Checkpoints, logs, raw HASOS source JSON, and generated outputs are ignored by
  git to avoid pushing large or regenerated artifacts.
- `src/train.py` includes a PyTorch 2.6 resume fix via `weights_only=False`.
- `src/train.py` also avoids CUDA boolean-indexing inside `grad_report`, which
  previously caused a CUDA launch timeout after epoch diagnostics.
- `scripts/run_space_hasos_aspect_parallel.py` resolves the checkpoint path to
  an absolute path before launching shards, so subprocesses do not fail from
  `cwd=src`.
- `scripts/export_space_hasos_lines.py` writes both detailed names and the
  expected aliases: `*_aspect_lines.*` and `*_lines.*`.
