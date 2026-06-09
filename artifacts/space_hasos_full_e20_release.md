# Release Artifacts — `space_hasos_full_e20`

This release stores the binary checkpoint and the final generated HASOS outputs.

GitHub release:

- https://github.com/Meanliss/tesing/releases/tag/space_hasos_full_e20

Assets:

| Asset | Size | SHA256 |
| --- | ---: | --- |
| `space_full_11402x20_stable_resume_e7_20_model.pt` | 106,191,291 bytes | `75f995e5e69ecabe57967c7cff6f2ff6b6cd3b50f2d450348fff1ab875156186` |
| `space_hasos_full_e20_results.tar.gz` | 2,513,494 bytes | `42439690b20fef98daa20450199cadbca80730bc6c5cb0e158b2c9342e6e366b` |

Validation snapshot:

- Aspect output files: 1,450 = 50 entities x 29 aspects.
- Sentiment output files: 4,350 = 1,450 x 3 buckets.
- Sentence provenance rows: 3,580.
- Aspect line rows: 3,580.
- Provenance coverage over non-empty summary sentences: 1.0000.
- Empty aspect output files retained in tree: 80.
- SPACE original baseline rerun: `space_old_aspects_e20`.
- SPACE original all-split macro ROUGE F1: R1 0.30327 / R2 0.08565 / RL 0.21885.

The result archive contains:

```text
outputs/space_hasos_full_e20/
outputs/space_hasos_full_e20_sentiment/
outputs/space_hasos_full_e20_lines.jsonl
outputs/space_hasos_full_e20_lines.tsv
outputs/space_hasos_full_e20_aspect_lines.jsonl
outputs/space_hasos_full_e20_aspect_lines.tsv
outputs/space_hasos_full_e20_aspect_sentiment_lines.jsonl
outputs/space_hasos_full_e20_aspect_sentiment_lines.tsv
outputs/space_hasos_full_e20_provenance.jsonl
outputs/space_hasos_full_e20_report.md
outputs/space_hasos_full_e20_report.json
outputs/space_hasos_full_e20_metrics.md
outputs/space_hasos_full_e20_metrics.json
outputs/space_hasos_full_e20_report.pptx
outputs/space_hasos_full_e20_pipeline_report.pptx
outputs/eval_space_old_aspects_e20.json
outputs/eval_space_old_aspects_e20.txt
outputs/space_old_aspects_e20/
reports/space_hasos_full_e20_report.pptx
reports/space_hasos_stage_io.pptx
reports/space_old_aspects_e20_official_rouge.json
reports/space_old_aspects_e20_official_rouge.txt
reports/space_old_aspects_e20_official_rouge.md
reports/space_old_aspects_e20_official_rouge.log
logs/space_hasos_full_e20_pipeline_trace.md
logs/space_hasos_full_e20_pipeline_trace.jsonl
logs/space_hasos_full_e20_inference.log
logs/space_hasos_full_e20_score_outputs.log
logs/space_hasos_full_e20__shard*.trace.jsonl
```

Quick validation after download:

```bash
sha256sum space_full_11402x20_stable_resume_e7_20_model.pt
sha256sum space_hasos_full_e20_results.tar.gz
tar -tzf space_hasos_full_e20_results.tar.gz | head
```
