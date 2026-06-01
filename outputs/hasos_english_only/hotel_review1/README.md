# hotel_review1.csv English-Only Results

Language detector:

```text
English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.
```

| Metric | Value |
| --- | ---: |
| Total rows | 607,260 |
| Non-empty reviews | 607,260 |
| English reviews kept | 270,658 |
| English ratio | 44.57% |
| English sentences kept | 850,307 |
| Matched aspects | 29/29 |
| ASC | 0.7337 |
| Macro CEC | 0.5583 |
| Weighted CEC | 0.5624 |

## Top Aspects

| Aspect | Unique opinions | Weight | CEC | ASC contribution |
| --- | ---: | ---: | ---: | ---: |
| `AM_POOL` | 107,100 | 0.0386 | 0.6634 | 0.0312 |
| `EXP_OVERALL` | 93,538 | 0.0381 | 0.4806 | 0.0300 |
| `FAC_BATH` | 121,174 | 0.0390 | 0.6390 | 0.0297 |
| `SER_ATTITUDE` | 151,202 | 0.0397 | 0.6974 | 0.0292 |
| `FAC_BUILDING` | 48,602 | 0.0359 | 0.5016 | 0.0292 |
| `FAC_INTERIOR` | 42,721 | 0.0355 | 0.5991 | 0.0288 |
| `AM_FOOD` | 131,420 | 0.0393 | 0.5608 | 0.0285 |
| `FAC_ROOM` | 234,128 | 0.0412 | 0.6184 | 0.0282 |
| `AM_ENT` | 15,750 | 0.0322 | 0.4861 | 0.0276 |
| `AM_UTILITY` | 25,623 | 0.0338 | 0.4950 | 0.0273 |

## Artifacts

- `filtered_english_stats.json`
- `file_scores.json`
- `file_scores.csv`
- `file_summaries.txt`
- `top10_hotels_pipeline_log.md`
- `top10_hotels_pipeline_log.json`
