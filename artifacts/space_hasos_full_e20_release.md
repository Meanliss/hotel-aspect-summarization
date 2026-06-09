# Release Artifacts — `space_hasos_full_e20`

This release stores the binary checkpoint and the final generated HASOS outputs.

GitHub release:

- https://github.com/Meanliss/tesing/releases/tag/space_hasos_full_e20

Assets:

| Asset | Size | SHA256 |
| --- | ---: | --- |
| `space_full_11402x20_stable_resume_e7_20_model.pt` | 106,191,291 bytes | `75f995e5e69ecabe57967c7cff6f2ff6b6cd3b50f2d450348fff1ab875156186` |
| `space_hasos_full_e20_results.tar.gz` | 1,376,466 bytes | `b5d20cde12593fa021d6c22ca61b08513f979712ba28018b0461bdbf35abd463` |

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
outputs/space_hasos_full_e20_report.md
outputs/space_hasos_full_e20_report.json
outputs/space_hasos_full_e20_metrics.md
outputs/space_hasos_full_e20_metrics.json
outputs/space_hasos_full_e20_report.pptx
outputs/space_hasos_full_e20_pipeline_report.pptx
reports/space_hasos_full_e20_report.pptx
reports/space_hasos_stage_io.pptx
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

