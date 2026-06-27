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

## Reproduce

```powershell
python scripts\build_concrete_metric_dataset.py
$env:DEEPSEEK_API_KEY = "<your key>"
python scripts\run_concrete_metric_judge.py --concurrency 80 --timeout 180
python scripts\score_concrete_metrics.py
```

The API key must be supplied through `DEEPSEEK_API_KEY`; it should never be
written to the repository.
