# HASOS English-Only Summary

This folder contains English-only HASOS CEC/ASC scoring for all three hotel review CSV files.

Language detector:

```text
English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.
```

ROUGE status: not available because the current data has reviews only and no human gold/reference summaries.

| File | Total rows | English reviews | English ratio | English sentences | Matched aspects | ASC | Macro CEC | Weighted CEC | Top aspects |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `hotel_review1.csv` | 607,260 | 270,658 | 44.57% | 850,307 | 29/29 | 0.7337 | 0.5583 | 0.5624 | AM_POOL, EXP_OVERALL, FAC_BATH, SER_ATTITUDE, FAC_BUILDING |
| `hotel_review2.csv` | 1,150,415 | 602,919 | 52.41% | 1,933,276 | 29/29 | 0.7335 | 0.5695 | 0.5740 | FAC_VIEW_LOCATION, SER_ATTITUDE, AM_FOOD, FAC_ROOM, FAC_BUILDING |
| `hotel_review3.csv` | 591,023 | 538,515 | 91.12% | 3,146,670 | 29/29 | 0.7539 | 0.5713 | 0.5728 | EXP_OVERALL, FAC_BUILDING, AM_POOL, AM_FOOD, FAC_BATH |

## Per-file folders

- `hotel_review1/`
- `hotel_review2/`
- `hotel_review3/`
