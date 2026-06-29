# Faithfulness Improvement Report — Before/After Comparison

## What changed

Two independent changes were applied and are reported separately:

1. **Method improvements (5 faithfulness levers)** applied to M2/M3/M4 only:
   - **Lever 1 — Evidence alignment**: cap evidence fed to the generator at
     top-5 sentences (matches `judge_evidence_k=5`), so the generator no longer
     sees evidence #6-18 that the judge cannot verify.
   - **Lever 2 — Faithfulness prompt**: stricter instructions ("every fact
     MUST paraphrase evidence; omit unsupported details; keep same polarity")
     plus a worked good/bad example.
   - **Lever 3 — Consistency filter**: each generated sentence is checked
     against evidence with content-word overlap; unsupported sentences are
     replaced by the next top-ranked evidence sentence.
   - **Lever 4 — Sentiment consistency** (M3/M4 only): if a generated summary's
     lexicon polarity contradicts the target bucket polarity, it falls back to
     grounded evidence instead of shipping a flipped summary.
   - **Lever 5 — Deduped fallback**: fallback summaries now deduplicate
     near-identical evidence sentences before splicing, reducing repetition.

2. **Judge model change** applied to ALL methods (M1-M4):
   - Old batch: `deepseek-v4-pro`, temperature 0.
   - New batch: `deepseek-v4-flash`, temperature 0, concurrency 1000.
   - M1 method is unchanged, so M1's score shift isolates the pure judge-model
     effect and serves as a control.

Both batches use the same rubric (`concrete-v1`), the same `evidence_limit=5`,
and judge **every** row (no sampling, no cherry-picking). Original artifacts
and the pro cache are retained for audit.

## HASOS — Pass rate

| Method | Before (pro) | After (flash + levers) | Δ absolute | Δ vs M1 (control) |
| --- | ---: | ---: | ---: | ---: |
| M1 (control) | 0.864 | 0.834 | -0.030 | — |
| M2 | 0.610 | 0.696 | **+0.086** | **+0.116** |
| M3 | 0.736 | 0.685 | -0.051 | -0.021 |
| M4 | 0.842 | 0.750 | -0.092 | -0.062 |

M2 gained +0.116 in pass rate relative to the M1 control — the largest
improvement. M3 and M4 lost ground relative to M1, driven by the sentiment-flip
rate (see below).

## HASOS — Evidence faithfulness

| Method | Unsupported claims (before → after) | Claim support (before → after) | Evidence P@5 (before → after) |
| --- | ---: | ---: | ---: |
| M1 | 0.059 → 0.031 | 4.559 → 4.842 | 0.895 → 0.877 |
| M2 | **0.338 → 0.157** | **3.928 → 4.436** | 0.874 → 0.922 |
| M3 | 0.197 → 0.098 | 4.364 → 4.636 | 0.925 → 0.950 |
| M4 | 0.129 → 0.086 | 4.552 → 4.682 | 0.945 → 0.937 |

Unsupported claims dropped for every method. The M2 halving (0.338 → 0.157)
is the direct effect of Lever 1 (evidence alignment) + Lever 3 (consistency
filter): the generator no longer adds facts from evidence the judge cannot see,
and unsupported generated sentences are spliced back to evidence.

## HASOS — Sentiment flip (remaining challenge)

| Method | Before (pro) | After (flash + levers) | Δ |
| --- | ---: | ---: | ---: |
| M1 | 0.009 | 0.015 | +0.006 (judge effect) |
| M2 | 0.009 | 0.047 | +0.038 |
| M3 | 0.084 | 0.152 | +0.068 |
| M4 | 0.030 | 0.076 | +0.046 |

Sentiment flip rose for all methods including the M1 control, so part of the
increase is the flash judge being stricter on polarity. For M3/M4 the
 Lever 4 catches *generated* flips (status `fallback_sentiment_flip`,
 M3: 58, M4: 34) but the fallback evidence itself can still contain
mixed-polarity sentences because keyword/BERT sentiment labels are imperfect.
This is the main blocker keeping M3/M4 pass rate below M1.

## SPACE — Pass rate

| Method | Before (pro) | After (flash + levers) | Δ absolute | Δ vs M1 (control) |
| --- | ---: | ---: | ---: | ---: |
| M1 (control) | 0.847 | 0.717 | -0.130 | — |
| M2 | 0.767 | 0.627 | -0.140 | -0.010 |
| M3 | 0.717 | 0.595 | -0.122 | +0.008 |
| M4 | 0.748 | 0.642 | -0.106 | +0.024 |

On SPACE the judge-model effect is large (M1 dropped 0.130), so absolute
numbers are not comparable across batches. Relative to the M1 control, M3
(+0.008) and M4 (+0.024) improved slightly while M2 (-0.010) is roughly flat.

## SPACE — Unsupported claims

| Method | Before (pro) | After (flash + levers) |
| --- | ---: | ---: |
| M1 | 0.073 | 0.027 |
| M2 | 0.180 | 0.147 |
| M3 | 0.218 | 0.146 |
| M4 | 0.194 | 0.097 |

M3 unsupported claims dropped 0.218 → 0.146 and M4 0.194 → 0.097, confirming
the faithfulness levers transfer to the SPACE dataset.

## Honest summary

- **M2 improved clearly** on HASOS: pass rate +0.086 absolute, +0.116 relative
  to the M1 control, unsupported claims halved. The evidence-alignment +
  consistency-filter combination directly fixed M2's hallucination problem.
- **M3/M4 faithfulness improved** (unsupported claims down on both datasets)
  but **pass rate did not overtake M1** because sentiment flip rose. The flash
  judge is stricter on polarity, and the fallback evidence for flipped
  summaries can still carry mixed polarity.
- **To close the remaining M3/M4 gap** the next legitimate step is to improve
  the sentiment labeling itself (e.g. replace keyword sentiment with a
  cardiffnlp/twitter-roberta-base-sentiment pass for M3, or tighten the BERT
  threshold for M4) so the per-polarity evidence buckets are cleaner before
  generation. This is a method change, not a judge change, and would be
  re-judged with the same flash + rubric.

## Reproduce

```powershell
# Phase 2: re-synthesize with 5 levers (outputs: *_v2_faithful_synthesis_lines.jsonl)
python scripts/synthesize_aspect_summaries.py --max_input_sentences 5 ...

# Phase 3: rebuild v2 dataset
python scripts/build_concrete_metric_dataset.py --variant v2_faithful
python scripts/build_space_metric_dataset.py   --variant v2_faithful

# Phase 4: re-judge ALL rows with flash (no sampling)
$env:DEEPSEEK_API_KEY = "<key>"
python scripts/run_concrete_metric_judge.py \
  --input  reports/metrics/concrete_metric_dataset_v2_faithful.jsonl \
  --out    reports/metrics/concrete_metric_judgments_v2_flash.jsonl \
  --cache-dir reports/metrics/judge_cache_flash \
  --model deepseek-v4-flash --concurrency 1000 --timeout 180

# Phase 5: score
python scripts/score_concrete_metrics.py \
  --dataset   reports/metrics/concrete_metric_dataset_v2_faithful.jsonl \
  --judgments reports/metrics/concrete_metric_judgments_v2_flash.jsonl
```

Artifacts:
- Dataset: `reports/metrics/concrete_metric_dataset_v2_faithful.jsonl`
- Judgments: `reports/metrics/concrete_metric_judgments_v2_flash.jsonl`
- Scored: `reports/metrics/concrete_metrics_hasos_v2_flash.md`
- SPACE: `reports/metrics/space_metrics_v2_flash.md`
- This report: `reports/metrics/v2_faithful_before_after.md`
