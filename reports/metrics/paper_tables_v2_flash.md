# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | 0.877 | 0.965 | 0.133 | 0.020 | 4.842 | 0.031 | 0.015 |
| M2 | 1372 | 0.922 | 0.740 | 0.100 | 0.024 | 4.436 | 0.157 | 0.047 |
| M3 | 907 | 0.950 | 0.898 | 0.149 | 0.057 | 4.636 | 0.098 | 0.152 |
| M4 | 804 | 0.937 | 0.922 | 0.129 | 0.032 | 4.682 | 0.086 | 0.076 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.834 | 4.518 | 4.835 | 4.623 | 4.674 | 4.290 | 0.958 | 0.014 |
| M2 | 0.696 | 4.447 | 4.603 | 3.754 | 4.348 | 4.069 | 0.820 | 0.061 |
| M3 | 0.685 | 4.506 | 4.300 | 4.311 | 4.515 | 4.141 | 0.897 | 0.037 |
| M4 | 0.750 | 4.485 | 4.607 | 4.437 | 4.593 | 4.312 | 0.898 | 0.044 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | - | - | 1.000 | 2.613 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 1372 | 0.562 | 0.329 | 0.185 | 18.389 | 0.491 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.421 | 0.447 | 0.541 | 2.834 | 0.751 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.337 | 0.527 | 0.632 | 1.919 | 0.830 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `3152286`, completion tokens `5616073`, estimated cost `$2.0138`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
