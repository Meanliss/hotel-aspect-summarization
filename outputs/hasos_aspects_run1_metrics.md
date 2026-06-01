# Metrics report — `hasos_aspects_run1` (no ROUGE; gold summaries unavailable)

## Macro averages (mean over aspects)

| Metric | Value | What it measures |
| --- | ---: | --- |
| source_fidelity        | 0.5972 | fraction of summary sentences found verbatim in source reviews (extractive check, ideal=1.0) |
| aspect_keyword_coverage| 0.7392 | fraction of summary sentences containing ≥1 aspect/sentiment keyword for own aspect (higher=better) |
| aspect_purity          | 0.5348 | fraction of summary sentences whose top-matching aspect == target aspect (higher=better) |
| distinct_1             | 0.2958 | unique unigrams / total unigrams across aspect's summaries (lexical diversity) |
| distinct_2             | 0.7365 | unique bigrams  / total bigrams  across aspect's summaries |
| self_bleu4             | 0.0085 | avg pairwise BLEU-4 between summaries within same aspect (lower=more diverse) |
| compression_ratio      | 0.0028 | summary tokens / source tokens (extractive compression) |
| avg_sentence_len       | 15.95 | mean tokens per summary sentence |
| cross_aspect_jaccard   | 0.1013 | avg token-Jaccard between any two aspect summaries of same entity (lower=better separation) |
| bert_f1_aspect         | 0.8126 | BERTScore-F1 (raw) between summary and aspect description text (higher=better aspect alignment) |
| bert_f1_source         | 0.8072 | BERTScore-F1 (raw) between summary and entity source-review pool (higher=better semantic fidelity) |

## Per-aspect breakdown

| Aspect | n_files | n_sents | src_fid | kw_cov | purity | distinct1 | distinct2 | self_bleu4 | compr | avg_len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AM_ENT | 50 | 128 | 0.594 | 0.750 | 0.359 | 0.313 | 0.767 | 0.000 | 0.0028 | 15.2 |
| AM_FOOD | 50 | 107 | 0.542 | 0.888 | 0.776 | 0.306 | 0.797 | 0.001 | 0.0028 | 18.6 |
| AM_POOL | 50 | 99 | 0.586 | 0.737 | 0.616 | 0.318 | 0.778 | 0.003 | 0.0028 | 16.1 |
| AM_ROOM_UTIL | 50 | 120 | 0.567 | 0.808 | 0.625 | 0.290 | 0.767 | 0.000 | 0.0028 | 16.6 |
| AM_TRANSPORT | 50 | 117 | 0.581 | 0.726 | 0.573 | 0.313 | 0.782 | 0.001 | 0.0028 | 16.8 |
| AM_UTILITY | 50 | 81 | 0.617 | 0.494 | 0.309 | 0.386 | 0.869 | 0.000 | 0.0028 | 15.8 |
| AM_WELLNESS | 50 | 123 | 0.699 | 0.577 | 0.423 | 0.332 | 0.784 | 0.000 | 0.0028 | 12.3 |
| AM_WIFI | 50 | 145 | 0.648 | 0.752 | 0.517 | 0.257 | 0.702 | 0.000 | 0.0028 | 13.7 |
| BRA_LUXURY | 50 | 117 | 0.590 | 0.709 | 0.487 | 0.289 | 0.755 | 0.002 | 0.0027 | 16.4 |
| BRA_REPUTE | 50 | 137 | 0.628 | 0.715 | 0.292 | 0.300 | 0.762 | 0.002 | 0.0028 | 14.5 |
| EXP_EMOTION | 50 | 125 | 0.648 | 0.592 | 0.288 | 0.334 | 0.817 | 0.000 | 0.0028 | 15.2 |
| EXP_OVERALL | 50 | 132 | 0.614 | 0.795 | 0.280 | 0.233 | 0.611 | 0.014 | 0.0028 | 15.2 |
| EXP_SAFETY | 50 | 130 | 0.600 | 0.723 | 0.485 | 0.320 | 0.793 | 0.001 | 0.0028 | 14.7 |
| EXP_VALUE | 50 | 116 | 0.569 | 0.810 | 0.741 | 0.249 | 0.708 | 0.004 | 0.0028 | 17.0 |
| FAC_BATH | 50 | 131 | 0.611 | 0.870 | 0.649 | 0.258 | 0.674 | 0.004 | 0.0028 | 15.2 |
| FAC_BUILDING | 50 | 106 | 0.538 | 0.708 | 0.443 | 0.328 | 0.770 | 0.002 | 0.0028 | 19.0 |
| FAC_CLIMATE | 50 | 113 | 0.593 | 0.735 | 0.566 | 0.307 | 0.771 | 0.001 | 0.0028 | 16.6 |
| FAC_ENV | 50 | 108 | 0.528 | 0.694 | 0.398 | 0.299 | 0.746 | 0.001 | 0.0028 | 18.4 |
| FAC_INTERIOR | 50 | 120 | 0.592 | 0.708 | 0.408 | 0.339 | 0.791 | 0.000 | 0.0028 | 16.7 |
| FAC_ROOM | 50 | 101 | 0.465 | 0.901 | 0.733 | 0.282 | 0.737 | 0.003 | 0.0028 | 19.7 |
| FAC_SECURITY | 50 | 97 | 0.588 | 0.588 | 0.454 | 0.368 | 0.838 | 0.000 | 0.0028 | 15.6 |
| FAC_VIEW_LOCATION | 50 | 108 | 0.537 | 0.852 | 0.769 | 0.263 | 0.675 | 0.011 | 0.0028 | 18.2 |
| LOY_PREFERENCE | 50 | 118 | 0.678 | 0.576 | 0.373 | 0.335 | 0.776 | 0.004 | 0.0028 | 14.3 |
| LOY_RECOMMEND | 50 | 190 | 0.737 | 0.942 | 0.905 | 0.188 | 0.442 | 0.112 | 0.0028 | 10.5 |
| LOY_RETURN | 50 | 137 | 0.664 | 0.898 | 0.759 | 0.213 | 0.575 | 0.018 | 0.0028 | 14.6 |
| SER_ATTITUDE | 50 | 159 | 0.692 | 0.925 | 0.893 | 0.205 | 0.510 | 0.055 | 0.0028 | 12.6 |
| SER_COMM | 50 | 116 | 0.638 | 0.586 | 0.336 | 0.341 | 0.798 | 0.004 | 0.0028 | 14.1 |
| SER_OPERATION | 50 | 103 | 0.485 | 0.757 | 0.660 | 0.291 | 0.771 | 0.001 | 0.0028 | 19.3 |
| SER_SUPPORT | 50 | 102 | 0.490 | 0.618 | 0.392 | 0.322 | 0.790 | 0.001 | 0.0028 | 19.7 |
