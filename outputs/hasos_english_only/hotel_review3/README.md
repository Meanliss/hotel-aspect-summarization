# hotel_review3.csv English-Only Results

Language detector:

```text
English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.
```

| Metric | Value |
| --- | ---: |
| Total rows | 591,023 |
| Non-empty reviews | 591,023 |
| English reviews kept | 538,515 |
| English ratio | 91.12% |
| English sentences kept | 3,146,670 |
| Matched aspects | 29/29 |
| ASC | 0.7539 |
| Macro CEC | 0.5713 |
| Weighted CEC | 0.5728 |

## Top Aspects

| Aspect | Unique opinions | Weight | CEC | ASC contribution |
| --- | ---: | ---: | ---: | ---: |
| `EXP_OVERALL` | 426,041 | 0.0383 | 0.4847 | 0.0317 |
| `FAC_BUILDING` | 176,500 | 0.0357 | 0.6431 | 0.0304 |
| `AM_POOL` | 345,571 | 0.0377 | 0.6468 | 0.0301 |
| `AM_FOOD` | 512,579 | 0.0388 | 0.5450 | 0.0299 |
| `FAC_BATH` | 372,007 | 0.0379 | 0.6404 | 0.0292 |
| `SER_ATTITUDE` | 499,367 | 0.0388 | 0.6234 | 0.0279 |
| `FAC_ROOM` | 755,457 | 0.0400 | 0.5806 | 0.0277 |
| `FAC_INTERIOR` | 161,667 | 0.0354 | 0.6129 | 0.0277 |
| `FAC_ENV` | 100,817 | 0.0340 | 0.5923 | 0.0275 |
| `AM_WELLNESS` | 144,104 | 0.0351 | 0.5303 | 0.0271 |

## Artifacts

- `filtered_english_stats.json`
- `file_scores.json`
- `file_scores.csv`
- `file_summaries.txt`
- `top10_hotels_pipeline_log.md`
- `top10_hotels_pipeline_log.json`
