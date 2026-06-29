# Concrete Metric Suite

This directory contains the paper-oriented HASOS metric artifacts produced from
the optimized sweep baseline:

- M2: `T=0.0075`, `B=128`
- M3: `T=0.0055`, `B=96`
- M4: `T=0.005`, `B=96`

## Files

- `concrete_metric_dataset.jsonl`: normalized HASOS rows used for automatic
  metrics and LLM judging.
- `concrete_metric_judgments.jsonl`: DeepSeek judge outputs for all 3,083 rows.
- `concrete_metrics_hasos.json`: machine-readable aggregate metrics.
- `concrete_metrics_hasos.md`: full human-readable report.
- `paper_tables.md`: compact paper-ready tables.
- `concrete_metric_dataset_summary.json`: sanity summary for the normalized
  dataset.

`judge_cache/` is intentionally ignored by Git. It is a local resume/audit cache
for API calls and can be regenerated from `concrete_metric_dataset.jsonl`.

## v2_faithful and v3_polarity variants

Two improved batches apply faithfulness levers to M2/M3/M4 and re-judge **all**
M1-M4 rows. M1 is unchanged method-side and serves as the control for
judge-model shift.

- **v2_faithful**: 5 levers (evidence alignment to top-5, stricter prompt,
  consistency filter, sentiment-consistency fallback, deduped fallback),
  judged by `deepseek-v4-flash`. See
  [`v2_faithful_before_after.md`](v2_faithful_before_after.md).
- **v3_polarity**: adds a polarity filter to the M3/M4 splice/fallback so the
  replacement evidence never carries the opposite polarity, judged by
  `mimo-v2.5`. The same judge is also run on the original (no-lever) outputs
  to isolate the lever effect. See
  [`v3_faithful_before_after.md`](v3_faithful_before_after.md).

Files (v3_polarity):
- `concrete_metric_dataset_v3_polarity.jsonl` / `space_metric_dataset_v3_polarity.jsonl`
- `concrete_metric_judgments_v3_flash.jsonl` / `space_metric_judgments_v3_flash.jsonl`
- `concrete_metrics_hasos_v3_flash.md` / `space_metrics_v3_flash.md`
- Baseline same-judge: `concrete_metric_judgments_orig_mimo.jsonl`,
  `concrete_metrics_hasos_orig_mimo.md`, `space_metrics_orig_mimo.md`
- Judge caches (`judge_cache_flash_v3/`, `judge_cache_mimo_orig/`,
  `space_judge_cache_flash_v3/`, `space_judge_cache_mimo_orig/`) are
  Git-ignored.

## Reproduce

```powershell
python scripts\build_concrete_metric_dataset.py
$env:DEEPSEEK_API_KEY = "<your key>"
python scripts\run_concrete_metric_judge.py --concurrency 80 --timeout 180
python scripts\score_concrete_metrics.py
```

The API key must be supplied through `DEEPSEEK_API_KEY`; it should never be
written to the repository.
