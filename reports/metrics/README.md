# Metric Summaries

Curated metric outputs used by the thesis tables.

This folder keeps compact JSON and Markdown summaries only. Raw JSONL judge
datasets, API request caches, and failure logs are ignored to keep the
submission repository lightweight.

## Main Groups

- `concrete_metrics_hasos*.json` and `.md`: HASOS aggregate quality metrics.
- `space_metrics*.json` and `.md`: SPACE comparison metrics.
- `paper_tables*.md`: compact tables copied into the thesis.
- `*_summary.json`: sanity summaries for regenerated raw datasets.
- `v2_faithful_before_after.md` and `v3_faithful_before_after.md`: comparison
  notes for the faithfulness and polarity-filter variants.

API keys must be supplied through environment variables and must never be
written to repository files.
