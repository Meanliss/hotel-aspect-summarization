# Faithfulness Improvement Report — Same-Judge Before/After

## Method

To isolate the effect of the method improvements from judge-model variance,
both the baseline and the improved batch are judged by the **same** model
(`mimo-v2.5`, temperature 0, JSON mode) with the **same** rubric
(`concrete-v1`) and `evidence_limit=5`. Every row is judged — no sampling,
no cherry-picking. The original DeepSeek-v4-pro batch is retained for audit
but is not comparable across judge models.

### Method improvements applied to M2/M3/M4 only

- **Lever 1 — Evidence alignment**: cap evidence fed to the generator at
  top-5 sentences (matches `judge_evidence_k=5`).
- **Lever 2 — Faithfulness prompt**: stricter instructions plus a worked
  good/bad example.
- **Lever 3 — Consistency filter**: each generated sentence is checked
  against evidence with content-word overlap; unsupported sentences are
  replaced by the next top-ranked evidence sentence.
- **Lever 4 — Sentiment consistency** (M3/M4): if a generated summary's
  lexicon polarity contradicts the target bucket, fall back to grounded
  evidence. The splice pool and fallback are polarity-filtered so the
  replacement evidence never carries the opposite polarity.
- **Lever 5 — Deduped fallback**: fallback summaries deduplicate
  near-identical evidence before splicing.

M1 is the extractive baseline and is unchanged method-side; under the same
judge it moves by at most 0.005, confirming the control holds.

## HASOS — Pass rate (same judge `mimo-v2.5`)

| Method | Baseline (no levers) | v3 (5 levers + polarity) | Δ (levers) |
| --- | ---: | ---: | ---: |
| M1 (control) | 0.725 | 0.720 | -0.005 |
| M2 | 0.532 | 0.700 | **+0.168** |
| M3 | 0.665 | **0.734** | **+0.069** |
| M4 | 0.756 | **0.787** | **+0.031** |

**Final ordering (HASOS, v3): M4 (0.787) > M3 (0.734) > M1 (0.720) > M2 (0.700).**

M3 and M4 overtake M1; M2 narrows the gap from -0.193 to -0.020 but does not
overtake M1. The M1 control is stable (-0.005), so the gains are attributable
to the levers, not judge noise.

## HASOS — Faithfulness drivers

| Method | Unsupported claims (base → v3) | Claim support (base → v3) | Sentiment flip (base → v3) |
| --- | ---: | ---: | ---: |
| M1 | 0.103 → 0.093 | 4.440 → 4.401 | 0.012 → 0.016 |
| M2 | **0.348 → 0.147** | 3.853 → 4.353 | 0.030 → 0.020 |
| M3 | 0.215 → 0.110 | 4.281 → 4.530 | 0.114 → 0.099 |
| M4 | 0.144 → 0.091 | 4.448 → 4.627 | 0.056 → 0.061 |

Unsupported claims drop for every method; M2 is halved (0.348 → 0.147) from
Lever 1 + Lever 3. Sentiment flip drops for M2 and M3 (Lever 4 polarity
filter); M4 is essentially flat because its BERT-ABSA labels were already
cleaner, so there were fewer flips to catch.

## SPACE — Pass rate (same judge `mimo-v2.5`)

| Method | Baseline (no levers) | v3 (5 levers + polarity) | Δ (levers) |
| --- | ---: | ---: | ---: |
| M1 (control) | 0.737 | 0.740 | +0.003 |
| M2 | 0.647 | 0.693 | +0.046 |
| M3 | 0.642 | 0.698 | +0.056 |
| M4 | 0.652 | 0.715 | +0.063 |

**Final ordering (SPACE, v3): M1 (0.740) > M4 (0.715) > M3 (0.698) > M2 (0.693).**

On SPACE the levers still improve M2/M3/M4 (M1 control is flat at +0.003),
but M3/M4 do not overtake M1 — the gap closed from -0.085/-0.085 to
-0.025/-0.042. SPACE M3/M4 do overtake M2.

## Honest summary

- **HASOS (primary dataset): goal met.** M3 and M4 overtake M1 under the same
  judge, with M1 stable as the control. M4 is the best method at 0.787.
- **SPACE (secondary dataset): partial.** M3/M4 improve and overtake M2, but
  stay just below M1 (M4 0.715 vs M1 0.740). The remaining gap is driven by
  SPACE M3's keyword sentiment labels being noisier than HASOS's, which the
  polarity lexicon filter cannot fully compensate for.
- **M2 improves most in absolute terms** on both datasets (HASOS +0.168,
  SPACE +0.046) but does not overtake M1 because it has no sentiment split
  and its 18.4 avg evidence count still produces some unsupported claims even
  after Lever 1.
- **Next legitimate step for SPACE M3/M4 to overtake M1**: replace the
  keyword sentiment backend with `cardiffnlp/twitter-roberta-base-sentiment`
  (already used elsewhere in the repo) so the per-polarity evidence buckets
  are cleaner before generation. This is a method change, re-judged with the
  same `mimo-v2.5` + rubric.

## Reproduce

```powershell
# Phase 2: re-synthesize with 5 levers + polarity filter (run_id *_v3_polarity)
python scripts/synthesize_aspect_summaries.py --max_input_sentences 5 --split_sentiment ...

# Phase 3: rebuild v3 dataset (M2 reuses v2_faithful since no sentiment split)
python scripts/build_concrete_metric_dataset.py --variant v3_polarity
python scripts/build_space_metric_dataset.py   --variant v3_polarity

# Phase 4: judge ALL rows with the same model (no sampling)
$env:DEEPSEEK_API_KEY = "<key>"
python scripts/run_concrete_metric_judge.py \
  --input    reports/metrics/concrete_metric_dataset_v3_polarity.jsonl \
  --out      reports/metrics/concrete_metric_judgments_v3_flash.jsonl \
  --cache-dir reports/metrics/judge_cache_flash_v3 \
  --model mimo-v2.5 --base-url "https://token-plan-sgp.xiaomimimo.com/v1" \
  --concurrency 8 --max-retries 6 --timeout 180

# Baseline (no levers) judged by the same model for isolation
python scripts/run_concrete_metric_judge.py \
  --input    reports/metrics/concrete_metric_dataset.jsonl \
  --out      reports/metrics/concrete_metric_judgments_orig_mimo.jsonl \
  --cache-dir reports/metrics/judge_cache_mimo_orig \
  --model mimo-v2.5 --base-url "https://token-plan-sgp.xiaomimimo.com/v1" \
  --concurrency 8 --max-retries 6 --timeout 180

# Phase 5: score both
python scripts/score_concrete_metrics.py \
  --dataset   reports/metrics/concrete_metric_dataset_v3_polarity.jsonl \
  --judgments reports/metrics/concrete_metric_judgments_v3_flash.jsonl
```

## Artifacts

- v3 dataset: `concrete_metric_dataset_v3_polarity.jsonl` / `space_metric_dataset_v3_polarity.jsonl`
- v3 judgments: `concrete_metric_judgments_v3_flash.jsonl` / `space_metric_judgments_v3_flash.jsonl`
- v3 scored: `concrete_metrics_hasos_v3_flash.md` / `space_metrics_v3_flash.md`
- Baseline (same judge): `concrete_metrics_hasos_orig_mimo.md` / `space_metrics_orig_mimo.md`
- This report: `v3_faithful_before_after.md`
- Judge caches (`judge_cache_flash_v3/`, `judge_cache_mimo_orig/`,
  `space_judge_cache_flash_v3/`, `space_judge_cache_mimo_orig/`) are
  Git-ignored resume/audit caches.
