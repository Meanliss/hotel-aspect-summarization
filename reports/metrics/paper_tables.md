# Concrete Metric Suite - HASOS

Primary HASOS runs use the optimized base selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 1372 | 0.927 | 0.729 | 0.143 | 0.056 | 3.719 | 0.423 | 0.143 |
| M3 | 907 | 0.962 | 0.892 | 0.166 | 0.057 | 4.377 | 0.207 | 0.121 |
| M4 | 804 | 0.947 | 0.907 | 0.143 | 0.029 | 4.517 | 0.157 | 0.056 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 0.437 | 4.023 | 4.243 | 3.374 | 3.921 | 3.278 | 0.586 | 0.214 |
| M3 | 0.633 | 4.322 | 4.501 | 4.169 | 4.369 | 3.940 | 0.798 | 0.143 |
| M4 | 0.728 | 4.424 | 4.738 | 4.353 | 4.498 | 4.172 | 0.846 | 0.119 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | 1372 | 0.630 | 0.370 | 0.125 | 18.389 | 0.560 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.589 | 0.411 | 0.456 | 2.834 | 0.730 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.519 | 0.481 | 0.563 | 1.919 | 0.823 | 0.000 | 0.209 | 0.040 | 0.137 |

DeepSeek usage: prompt tokens `2249372`, completion tokens `6296435`, estimated cost `$6.4564`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
