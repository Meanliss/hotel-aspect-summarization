# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | 0.864 | 0.881 | 0.060 | 0.003 | 4.440 | 0.103 | 0.012 |
| M2 | 1372 | 0.859 | 0.675 | 0.045 | 0.006 | 3.853 | 0.348 | 0.030 |
| M3 | 907 | 0.904 | 0.805 | 0.058 | 0.024 | 4.281 | 0.215 | 0.114 |
| M4 | 804 | 0.911 | 0.856 | 0.057 | 0.011 | 4.448 | 0.144 | 0.056 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.725 | 4.464 | 4.589 | 4.020 | 4.075 | 4.059 | 0.907 | 0.104 |
| M2 | 0.532 | 4.186 | 4.305 | 3.471 | 3.456 | 3.634 | 0.822 | 0.244 |
| M3 | 0.665 | 4.315 | 4.393 | 3.963 | 3.988 | 3.991 | 0.875 | 0.168 |
| M4 | 0.756 | 4.439 | 4.641 | 4.159 | 4.138 | 4.162 | 0.881 | 0.157 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | - | - | 1.000 | 2.613 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 1372 | 0.630 | 0.370 | 0.125 | 18.389 | 0.560 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.589 | 0.411 | 0.456 | 2.834 | 0.730 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.519 | 0.481 | 0.563 | 1.919 | 0.823 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `2960857`, completion tokens `1317714`, estimated cost `$2.4344`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
