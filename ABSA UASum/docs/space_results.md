# SPACE Results

This repository includes a compact SPACE benchmark package for method 1.

## SPACE data note

Raw/converted SPACE input and gold-reference files are not included in this
curated Git package. This folder keeps only the canonical SPACE outputs and
ROUGE reports needed for submission.

## Canonical UASum/Qwen SPACE outputs

| File | Role |
|---|---|
| `answer_files/space/space_qwen_native_full_qwenroute_final_summary.csv` | Aspect-level SPACE output. |
| `answer_files/space/space_qwen_native_full_qwenroute_global_summary.csv` | Entity-level global output. |

## Official ROUGE summary

Metrics use pyrouge / ROUGE-1.5.5 F1 against SPACE human gold summaries.

| Run | Type | ROUGE-1 all | ROUGE-2 all | ROUGE-L all |
|---|---|---:|---:|---:|
| `space_native_extractive_t40_s2` | Extractive baseline | 0.3071 | 0.0837 | 0.2176 |
| `space_qwen_native_full_qwenroute` | UASum/Qwen SPACE-native aspect output | 0.2500 | 0.0572 | 0.1695 |
| `space_qwen_native_full_qwenroute_global` | UASum/Qwen generated global summary | 0.3124 | 0.0532 | 0.1668 |

## Interpretation

The extractive baseline remains strongest on official aspect-level ROUGE because
SPACE gold summaries reward lexical overlap. The UASum/Qwen output is more
abstractive and readable, but it should not be claimed as the best ROUGE
optimizer. The global-summary result is useful as an entity-level signal, while
the official aspect-level result is the stricter benchmark comparison.
