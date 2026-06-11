# Metrics report — `space_hasos_full_e20` (no ROUGE; gold summaries unavailable)

## Macro averages (mean over aspects)

| Metric | Value | What it measures |
| --- | ---: | --- |
| source_fidelity        | 0.6094 | fraction of summary sentences found verbatim in source reviews (extractive check, ideal=1.0) |
| source_fidelity_excl_truncated | 0.9692 | same exact-match check but excludes output sentences intentionally cut by max_tokens |
| aspect_keyword_coverage| 0.7441 | fraction of summary sentences containing ≥1 aspect/sentiment keyword for own aspect (higher=better) |
| aspect_purity          | 0.5517 | fraction of summary sentences whose top-matching aspect == target aspect (higher=better) |
| distinct_1             | 0.2846 | unique unigrams / total unigrams across aspect's summaries (lexical diversity) |
| distinct_2             | 0.7179 | unique bigrams  / total bigrams  across aspect's summaries |
| self_bleu4             | 0.0109 | avg pairwise BLEU-4 between summaries within same aspect (lower=more diverse) |
| compression_ratio      | 0.0028 | summary tokens / source tokens (extractive compression) |
| avg_sentence_len       | 15.57 | mean tokens per summary sentence |
| cross_aspect_jaccard   | 0.1050 | avg token-Jaccard between any two aspect summaries of same entity (lower=better separation) |
| bert_f1_aspect         | 0.8098 | BERTScore-F1 (raw) between summary and aspect description text (higher=better aspect alignment) |
| bert_f1_source         | 0.8074 | BERTScore-F1 (raw) between summary and entity source-review pool (higher=better semantic fidelity) |

## Per-aspect breakdown

| Aspect | n_files | n_sents | src_fid | src_fid_no_trunc | kw_cov | purity | distinct1 | distinct2 | self_bleu4 | compr | avg_len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AM_ENT | 50 | 126 | 0.587 | 0.961 | 0.778 | 0.413 | 0.311 | 0.784 | 0.001 | 0.0028 | 15.8 |
| AM_FOOD | 50 | 123 | 0.602 | 1.000 | 0.846 | 0.764 | 0.279 | 0.752 | 0.002 | 0.0028 | 16.1 |
| AM_POOL | 50 | 122 | 0.689 | 0.976 | 0.648 | 0.549 | 0.297 | 0.754 | 0.002 | 0.0028 | 13.1 |
| AM_ROOM_UTIL | 50 | 122 | 0.598 | 0.986 | 0.828 | 0.680 | 0.295 | 0.785 | 0.001 | 0.0028 | 16.4 |
| AM_TRANSPORT | 50 | 112 | 0.527 | 0.894 | 0.786 | 0.679 | 0.306 | 0.780 | 0.002 | 0.0028 | 17.7 |
| AM_UTILITY | 50 | 80 | 0.625 | 0.980 | 0.550 | 0.338 | 0.379 | 0.843 | 0.001 | 0.0028 | 15.9 |
| AM_WELLNESS | 50 | 127 | 0.701 | 0.978 | 0.606 | 0.449 | 0.330 | 0.788 | 0.000 | 0.0028 | 12.0 |
| AM_WIFI | 50 | 143 | 0.650 | 0.989 | 0.804 | 0.538 | 0.268 | 0.737 | 0.000 | 0.0028 | 13.9 |
| BRA_LUXURY | 50 | 115 | 0.626 | 1.000 | 0.704 | 0.470 | 0.328 | 0.799 | 0.001 | 0.0028 | 15.9 |
| BRA_REPUTE | 50 | 128 | 0.594 | 0.949 | 0.570 | 0.258 | 0.293 | 0.747 | 0.005 | 0.0028 | 15.5 |
| EXP_EMOTION | 50 | 116 | 0.629 | 1.000 | 0.681 | 0.345 | 0.326 | 0.799 | 0.000 | 0.0028 | 16.4 |
| EXP_OVERALL | 50 | 148 | 0.689 | 0.971 | 0.865 | 0.372 | 0.178 | 0.517 | 0.028 | 0.0028 | 13.4 |
| EXP_SAFETY | 50 | 132 | 0.614 | 0.964 | 0.712 | 0.477 | 0.342 | 0.801 | 0.002 | 0.0028 | 14.5 |
| EXP_VALUE | 50 | 118 | 0.492 | 0.866 | 0.737 | 0.636 | 0.253 | 0.730 | 0.004 | 0.0028 | 16.8 |
| FAC_BATH | 50 | 144 | 0.681 | 1.000 | 0.840 | 0.639 | 0.221 | 0.611 | 0.007 | 0.0028 | 14.0 |
| FAC_BUILDING | 50 | 94 | 0.468 | 0.978 | 0.734 | 0.468 | 0.333 | 0.795 | 0.000 | 0.0028 | 21.4 |
| FAC_CLIMATE | 50 | 100 | 0.550 | 0.982 | 0.740 | 0.510 | 0.311 | 0.785 | 0.000 | 0.0028 | 18.8 |
| FAC_ENV | 50 | 113 | 0.549 | 0.969 | 0.761 | 0.407 | 0.263 | 0.705 | 0.002 | 0.0028 | 17.6 |
| FAC_INTERIOR | 50 | 131 | 0.618 | 0.988 | 0.779 | 0.557 | 0.347 | 0.801 | 0.000 | 0.0028 | 15.3 |
| FAC_ROOM | 50 | 129 | 0.628 | 1.000 | 0.891 | 0.814 | 0.206 | 0.574 | 0.013 | 0.0028 | 15.5 |
| FAC_SECURITY | 50 | 90 | 0.578 | 0.981 | 0.544 | 0.378 | 0.400 | 0.873 | 0.000 | 0.0028 | 16.8 |
| FAC_VIEW_LOCATION | 50 | 124 | 0.548 | 0.944 | 0.863 | 0.815 | 0.213 | 0.594 | 0.017 | 0.0028 | 16.0 |
| LOY_PREFERENCE | 50 | 118 | 0.636 | 0.974 | 0.585 | 0.381 | 0.328 | 0.779 | 0.003 | 0.0028 | 14.2 |
| LOY_RECOMMEND | 50 | 198 | 0.758 | 0.974 | 0.955 | 0.894 | 0.147 | 0.375 | 0.116 | 0.0028 | 10.1 |
| LOY_RETURN | 50 | 155 | 0.671 | 0.937 | 0.897 | 0.748 | 0.172 | 0.476 | 0.034 | 0.0028 | 12.9 |
| SER_ATTITUDE | 50 | 189 | 0.730 | 0.972 | 0.910 | 0.889 | 0.170 | 0.445 | 0.070 | 0.0028 | 10.6 |
| SER_COMM | 50 | 106 | 0.623 | 0.971 | 0.623 | 0.396 | 0.354 | 0.813 | 0.002 | 0.0028 | 15.4 |
| SER_OPERATION | 50 | 100 | 0.480 | 0.941 | 0.730 | 0.680 | 0.283 | 0.760 | 0.002 | 0.0028 | 19.9 |
| SER_SUPPORT | 50 | 101 | 0.535 | 0.982 | 0.614 | 0.455 | 0.319 | 0.818 | 0.000 | 0.0028 | 19.8 |
