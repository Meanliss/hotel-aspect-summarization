# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | 0.860 | 0.876 | 0.067 | 0.006 | 4.401 | 0.093 | 0.016 |
| M2 | 1372 | 0.855 | 0.695 | 0.044 | 0.006 | 4.353 | 0.147 | 0.020 |
| M3 | 907 | 0.909 | 0.841 | 0.053 | 0.024 | 4.530 | 0.110 | 0.099 |
| M4 | 804 | 0.926 | 0.875 | 0.050 | 0.013 | 4.627 | 0.091 | 0.061 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.720 | 4.456 | 4.593 | 4.002 | 4.067 | 4.043 | 0.903 | 0.096 |
| M2 | 0.700 | 4.491 | 4.522 | 3.804 | 3.910 | 3.985 | 0.880 | 0.149 |
| M3 | 0.734 | 4.527 | 4.512 | 4.166 | 4.205 | 4.179 | 0.916 | 0.110 |
| M4 | 0.787 | 4.568 | 4.673 | 4.300 | 4.333 | 4.310 | 0.917 | 0.100 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 1370 | - | - | 1.000 | 2.613 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 1372 | 0.562 | 0.329 | 0.185 | 18.389 | 0.491 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 907 | 0.428 | 0.436 | 0.521 | 2.834 | 0.732 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 804 | 0.345 | 0.515 | 0.614 | 1.919 | 0.822 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `2950956`, completion tokens `1318441`, estimated cost `$2.4307`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
