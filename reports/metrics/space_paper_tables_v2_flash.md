# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | 0.746 | 0.942 | 0.248 | 0.048 | 4.630 | 0.027 | 0.007 |
| M2 | 300 | 0.767 | 0.868 | 0.221 | 0.035 | 4.347 | 0.147 | 0.060 |
| M3 | 321 | 0.792 | 0.934 | 0.258 | 0.030 | 4.452 | 0.146 | 0.078 |
| M4 | 330 | 0.790 | 0.926 | 0.244 | 0.029 | 4.564 | 0.097 | 0.030 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.717 | 4.030 | 4.713 | 4.373 | 4.537 | 4.027 | 0.877 | 0.007 |
| M2 | 0.627 | 4.033 | 4.470 | 3.990 | 4.283 | 3.903 | 0.817 | 0.040 |
| M3 | 0.595 | 3.950 | 4.567 | 4.106 | 4.315 | 3.854 | 0.816 | 0.050 |
| M4 | 0.642 | 3.952 | 4.700 | 4.200 | 4.379 | 3.888 | 0.833 | 0.070 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | - | - | 1.000 | 2.623 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 300 | 0.580 | 0.380 | 0.447 | 2.477 | 0.720 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 321 | 0.495 | 0.421 | 0.514 | 1.885 | 0.789 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 330 | 0.476 | 0.458 | 0.542 | 1.773 | 0.840 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `779452`, completion tokens `1507998`, estimated cost `$0.5314`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
