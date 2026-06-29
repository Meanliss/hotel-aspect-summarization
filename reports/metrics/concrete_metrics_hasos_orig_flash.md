# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | 0.873 | 0.966 | 0.144 | 0.017 | 4.877 | 0.023 | 0.021 |
| M2 | 1372 | 0.929 | 0.707 | 0.086 | 0.028 | 3.706 | 0.428 | 0.099 |
| M3 | 907 | 0.957 | 0.874 | 0.128 | 0.067 | 4.289 | 0.216 | 0.157 |
| M4 | 804 | 0.933 | 0.904 | 0.102 | 0.029 | 4.481 | 0.157 | 0.078 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.834 | 4.504 | 4.839 | 4.631 | 4.674 | 4.287 | 0.953 | 0.014 |
| M2 | 0.472 | 4.039 | 4.146 | 3.440 | 3.818 | 3.560 | 0.751 | 0.120 |
| M3 | 0.601 | 4.257 | 4.227 | 4.085 | 4.234 | 3.902 | 0.845 | 0.105 |
| M4 | 0.715 | 4.342 | 4.586 | 4.302 | 4.399 | 4.162 | 0.869 | 0.098 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | - | - | 1.000 | 2.613 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 1372 | 0.630 | 0.370 | 0.125 | 18.389 | 0.560 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.589 | 0.411 | 0.456 | 2.834 | 0.730 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.519 | 0.481 | 0.563 | 1.919 | 0.823 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `3159760`, completion tokens `5585561`, estimated cost `$2.0063`.

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
