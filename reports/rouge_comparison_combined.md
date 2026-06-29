# ROUGE Comparison - 4 methods x 2 datasets (SPACE + HASOS)

Official pyrouge (ROUGE-1.5.5) macro F1 against human gold summaries, split = all. SPACE uses 6 flat generic aspects with 3 references each. HASOS uses 4 parent aspects aggregated from 29 sub-aspects.

- **M1** - raw SemAE extractive sentences
- **M2** - FLAN-T5 abstractive rewrite, no sentiment split
- **M3** - sentiment-split abstractive, keyword backend
- **M4** - sentiment-split abstractive, BERT-ABSA backend

## Macro ROUGE F1 (split = all)

| Method | SPACE R1 | SPACE R2 | SPACE RL | HASOS R1 | HASOS R2 | HASOS RL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 SemAE extractive | 0.3041 | 0.0868 | 0.2218 | 0.2035 | 0.0550 | 0.1223 |
| M2 abstractive before sentiment | 0.3068 | 0.0860 | 0.2346 | 0.2258 | 0.0377 | 0.1525 |
| M3 keyword sentiment-split | 0.3094 | 0.0886 | 0.2398 | 0.2253 | 0.0520 | 0.1637 |
| M4 BERT-ABSA sentiment-split | 0.3080 | 0.0886 | 0.2393 | 0.2128 | 0.0407 | 0.1394 |

## Best method per dataset (macro ROUGE-1, split = all)

| Dataset | Best method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | ---: | ---: | ---: |
| SPACE | M3 keyword sentiment-split | 0.3094 | 0.0886 | 0.2398 |
| HASOS | M2 abstractive before sentiment | 0.2258 | 0.0377 | 0.1525 |

## Verdict

- SPACE is very close across the four methods, with M3 slightly ahead on macro ROUGE-1/L in the baseline comparison.
- In the non-optimized HASOS baseline table, no single method wins all ROUGE metrics: M2 leads ROUGE-1, M1 leads ROUGE-2, and M3 leads ROUGE-L. Use `reports/metrics/paper_tables.md` for the optimized HASOS paper table.
- The optimized HASOS sweep is reported separately because M2/M3/M4 can be re-filtered and re-synthesized without changing the M1 extractive baseline.

## Notes

- ROUGE-1.5.5 via pyrouge + Strawberry Perl on Windows.
- SPACE M1 macro ROUGE-1 0.3041 reproduces the official SemAE baseline (0.3033) computed earlier on the GPU box.
- HASOS M1 is read from `reports/rouge_m1_hasos.json`; `reports/sweep/_sanity_m1_hasos.json` is a fixed-denominator sanity artifact and is not used for the paper table.
- Detailed per-aspect / per-split tables: `reports/rouge_comparison_space.md`, `reports/rouge_comparison_hasos.md`.
