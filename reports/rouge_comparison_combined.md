# ROUGE Comparison — 4 methods × 2 datasets (SPACE + HASOS)

Official pyrouge (ROUGE-1.5.5) macro F1 against human gold summaries, split = all. SPACE = 6 flat generic aspects (3 refs each). HASOS = 4 parent aspects aggregated from 29 sub-aspects. All four methods share identical SemAE sentence selection and differ only in how the selected evidence is rendered:

- **M1** — raw SemAE extractive sentences
- **M2** — FLAN-T5 abstractive rewrite, no sentiment split
- **M3** — sentiment-split abstractive, keyword backend
- **M4** — sentiment-split abstractive, BERT-ABSA backend

## Macro ROUGE F1 (split = all)

| Method | SPACE R1 | SPACE R2 | SPACE RL | HASOS R1 | HASOS R2 | HASOS RL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3041 | 0.0868 | 0.2218 | 0.1031 | 0.0403 | 0.0680 |
| M2 Trước sentiment (abstractive) | 0.3068 | 0.0860 | 0.2346 | 0.2002 | 0.0378 | 0.1380 |
| M3 Sau sentiment — Keyword | 0.3094 | 0.0886 | 0.2398 | 0.2253 | 0.0520 | 0.1637 |
| M4 Sau sentiment — BERT-ABSA | 0.3080 | 0.0886 | 0.2393 | 0.2128 | 0.0407 | 0.1394 |

## Best method per dataset (macro ROUGE-1, split = all)

| Dataset | Best method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | --- | ---: | ---: | ---: |
| SPACE | M3 Sau sentiment — Keyword | 0.3094 | 0.0886 | 0.2398 |
| HASOS | M3 Sau sentiment — Keyword | 0.2253 | 0.0520 | 0.1637 |

## Verdict

- **M3 (keyword-sentiment abstractive) is the overall best** — it wins macro ROUGE-1/2/L on HASOS and ties for the SPACE lead with M4.
- Both sentiment-split methods (M3, M4) beat the pre-sentiment abstractive (M2), which in turn beats the raw extractive baseline (M1). The ordering M3 ≳ M4 > M2 > M1 holds on both datasets.
- The gap is largest on HASOS (fine-grained 29-aspect taxonomy) and small on SPACE (6 coarse aspects), i.e. sentiment splitting helps more when aspects are fine-grained.

## Notes

- ROUGE-1.5.5 via pyrouge + Strawberry Perl on Windows.
- SPACE M1 macro ROUGE-1 0.3041 reproduces the official SemAE baseline (0.3033) computed earlier on the GPU box — cross-validates the local pipeline.
- Detailed per-aspect / per-split tables: `reports/rouge_comparison_space.md`, `reports/rouge_comparison_hasos.md`.
