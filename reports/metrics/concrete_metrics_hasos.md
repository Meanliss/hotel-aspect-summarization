# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | 0.895 | 0.893 | 0.056 | 0.000 | 4.559 | 0.059 | 0.009 |
| M2 | 1372 | 0.874 | 0.653 | 0.029 | 0.000 | 3.928 | 0.338 | 0.009 |
| M3 | 907 | 0.925 | 0.823 | 0.041 | 0.008 | 4.364 | 0.197 | 0.084 |
| M4 | 804 | 0.945 | 0.874 | 0.021 | 0.001 | 4.552 | 0.129 | 0.030 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.864 | 4.799 | 4.709 | 4.023 | 4.149 | 4.252 | 0.981 | 0.012 |
| M2 | 0.610 | 4.473 | 4.239 | 3.600 | 3.738 | 3.840 | 0.920 | 0.079 |
| M3 | 0.736 | 4.573 | 4.456 | 4.090 | 4.117 | 4.236 | 0.940 | 0.069 |
| M4 | 0.842 | 4.680 | 4.726 | 4.340 | 4.275 | 4.469 | 0.965 | 0.063 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | - | - | 1.000 | 2.613 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 1372 | 0.630 | 0.370 | 0.125 | 18.389 | 0.560 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.589 | 0.411 | 0.456 | 2.834 | 0.730 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.519 | 0.481 | 0.563 | 1.919 | 0.823 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `3159760`, completion tokens `1383558`, estimated cost `$0.8298`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |

## Notes

- ABSA gold F1 is not reported because the available HASOS artifact does not expose sentence-level gold aspect-sentiment labels.
- Judge-based metrics are reported separately from ROUGE to avoid treating semantic judgments as reference-summary overlap.
- Production p95/cache/retry/JSON-repair metrics require explicit runtime instrumentation and are not inferred from incomplete logs.
