# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | 0.845 | 0.896 | 0.099 | 0.001 | 4.520 | 0.083 | 0.007 |
| M2 | 300 | 0.844 | 0.826 | 0.075 | 0.004 | 4.373 | 0.130 | 0.003 |
| M3 | 321 | 0.856 | 0.883 | 0.111 | 0.012 | 4.445 | 0.162 | 0.034 |
| M4 | 330 | 0.865 | 0.860 | 0.082 | 0.010 | 4.412 | 0.130 | 0.024 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.740 | 4.387 | 4.690 | 4.087 | 4.117 | 4.107 | 0.913 | 0.047 |
| M2 | 0.693 | 4.403 | 4.687 | 3.880 | 3.970 | 4.023 | 0.890 | 0.110 |
| M3 | 0.698 | 4.371 | 4.760 | 3.928 | 4.072 | 4.047 | 0.888 | 0.134 |
| M4 | 0.715 | 4.394 | 4.739 | 3.930 | 3.973 | 3.991 | 0.852 | 0.139 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | - | - | 1.000 | 2.623 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 300 | 0.580 | 0.380 | 0.447 | 2.477 | 0.720 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 321 | 0.495 | 0.421 | 0.514 | 1.885 | 0.784 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 330 | 0.476 | 0.458 | 0.542 | 1.773 | 0.832 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `729186`, completion tokens `321908`, estimated cost `$0.5973`.

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
