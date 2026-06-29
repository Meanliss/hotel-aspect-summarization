# Concrete Metric Suite - SPACE

SPACE comparison uses the original 6 aspect groups (building, cleanliness, food, location, rooms, service). M1 is the extractive SemAE baseline; M2 rewrites evidence without sentiment split; M3 and M4 evaluate keyword and BERT-ABSA sentiment splits.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | 0.813 | 0.895 | 0.149 | 0.000 | 4.513 | 0.073 | 0.000 |
| M2 | 300 | 0.820 | 0.808 | 0.102 | 0.000 | 4.330 | 0.180 | 0.000 |
| M3 | 321 | 0.874 | 0.858 | 0.094 | 0.002 | 4.352 | 0.218 | 0.040 |
| M4 | 330 | 0.870 | 0.850 | 0.087 | 0.000 | 4.385 | 0.194 | 0.015 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.847 | 4.657 | 4.747 | 3.897 | 4.080 | 4.217 | 0.963 | 0.003 |
| M2 | 0.767 | 4.620 | 4.630 | 3.713 | 3.893 | 4.163 | 0.953 | 0.017 |
| M3 | 0.717 | 4.551 | 4.645 | 4.006 | 4.034 | 4.262 | 0.975 | 0.044 |
| M4 | 0.748 | 4.518 | 4.739 | 3.967 | 4.030 | 4.261 | 0.952 | 0.039 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | - | - | 1.000 | 2.623 | 1.000 | 0.000 | 0.304 | 0.087 | 0.222 |
| M2 | 300 | 0.997 | 0.003 | 0.370 | 2.477 | 0.690 | 0.000 | 0.307 | 0.086 | 0.235 |
| M3 | 321 | 0.632 | 0.368 | 0.467 | 1.885 | 0.779 | 0.000 | 0.309 | 0.089 | 0.240 |
| M4 | 330 | 0.642 | 0.358 | 0.464 | 1.773 | 0.857 | 0.000 | 0.308 | 0.089 | 0.239 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `778725`, completion tokens `341670`, estimated cost `$0.2047`.

## Notes

- SPACE judge uses aspect-level and sentiment-level outputs only; entity-level overall summaries are not included in this table.
- M2 cached outputs are counted as generated outputs because they are restored generated summaries, not extractive fallbacks.
- SPACE has ROUGE references for the six aspect groups, but not gold sentiment-level labels for ABSA F1 or sentiment-split F1.
