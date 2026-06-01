# hotel_review2.csv English-Only Results

Language detector:

```text
English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.
```

| Metric | Value |
| --- | ---: |
| Total rows | 1,150,415 |
| Non-empty reviews | 1,150,384 |
| English reviews kept | 602,919 |
| English ratio | 52.41% |
| English sentences kept | 1,933,276 |
| Matched aspects | 29/29 |
| ASC | 0.7335 |
| Macro CEC | 0.5695 |
| Weighted CEC | 0.5740 |

## Top Aspects

| Aspect | Unique opinions | Weight | CEC | ASC contribution |
| --- | ---: | ---: | ---: | ---: |
| `FAC_VIEW_LOCATION` | 321,132 | 0.0393 | 0.6185 | 0.0308 |
| `SER_ATTITUDE` | 328,886 | 0.0394 | 0.7289 | 0.0308 |
| `AM_FOOD` | 292,661 | 0.0390 | 0.6030 | 0.0306 |
| `FAC_ROOM` | 503,905 | 0.0407 | 0.6138 | 0.0295 |
| `FAC_BUILDING` | 109,009 | 0.0360 | 0.5488 | 0.0289 |
| `AM_POOL` | 226,720 | 0.0382 | 0.6877 | 0.0279 |
| `FAC_ENV` | 78,934 | 0.0350 | 0.5222 | 0.0276 |
| `FAC_BATH` | 256,847 | 0.0386 | 0.6548 | 0.0274 |
| `AM_ENT` | 34,153 | 0.0324 | 0.4941 | 0.0272 |
| `EXP_SAFETY` | 79,994 | 0.0350 | 0.7430 | 0.0271 |

## Artifacts

- `filtered_english_stats.json`
- `file_scores.json`
- `file_scores.csv`
- `file_summaries.txt`
- `top10_hotels_pipeline_log.md`
- `top10_hotels_pipeline_log.json`
