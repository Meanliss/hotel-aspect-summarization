# Concrete Metric Suite - HASOS

Primary HASOS comparison uses M1 extractive baseline (SemAE epoch-20, B=40 words; no generative decoder) and the optimized synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, M4 T=0.005/B=96.

## Evidence And Faithfulness

| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | 0.853 | 0.900 | 0.095 | 0.000 | 4.487 | 0.077 | 0.010 |
| M2 | 300 | 0.830 | 0.787 | 0.071 | 0.005 | 4.223 | 0.197 | 0.010 |
| M3 | 321 | 0.858 | 0.841 | 0.096 | 0.012 | 4.227 | 0.215 | 0.050 |
| M4 | 330 | 0.844 | 0.810 | 0.108 | 0.003 | 4.188 | 0.242 | 0.045 |

## Summary Utility By LLM Judge

| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.737 | 4.407 | 4.703 | 4.067 | 4.163 | 4.143 | 0.917 | 0.060 |
| M2 | 0.647 | 4.343 | 4.563 | 3.733 | 3.773 | 3.860 | 0.897 | 0.147 |
| M3 | 0.642 | 4.287 | 4.654 | 3.788 | 3.841 | 3.907 | 0.847 | 0.168 |
| M4 | 0.652 | 4.312 | 4.658 | 3.706 | 3.733 | 3.833 | 0.824 | 0.200 |

## Production Readiness

| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 300 | - | - | 1.000 | 2.623 | 1.000 | 0.000 | 0.203 | 0.055 | 0.122 |
| M2 | 300 | 0.997 | 0.003 | 0.370 | 2.477 | 0.690 | 0.000 | 0.232 | 0.044 | 0.150 |
| M3 | 321 | 0.632 | 0.368 | 0.467 | 1.885 | 0.779 | 0.000 | 0.262 | 0.071 | 0.168 |
| M4 | 330 | 0.642 | 0.358 | 0.464 | 1.773 | 0.857 | 0.000 | 0.209 | 0.040 | 0.137 |

Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.

DeepSeek usage: prompt tokens `728627`, completion tokens `321902`, estimated cost `$0.5970`.

## SPACE ROUGE Appendix

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| m1_extractive | 0.304 | 0.087 | 0.222 |
| m2_abstractive | 0.307 | 0.086 | 0.235 |
| m3_kw | 0.309 | 0.089 | 0.240 |
| m4_bert | 0.308 | 0.089 | 0.239 |
