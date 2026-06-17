# ROUGE Comparison — SPACE (4 methods)

Official pyrouge (ROUGE-1.5.5) F1 against the human SPACE gold summaries (6 flat aspects: building/cleanliness/food/location/rooms/service, 3 references each). All methods share identical SemAE sentence selection; they differ only in how selected evidence is rendered.

## Macro ROUGE F1 (mean over 6 aspects, split = all)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3041 | 0.0868 | 0.2218 |
| M2 Trước sentiment (abstractive) | 0.3068 | 0.0860 | 0.2346 |
| M3 Sau sentiment — Keyword | 0.3094 | 0.0886 | 0.2398 |
| M4 Sau sentiment — BERT-ABSA | 0.3080 | 0.0886 | 0.2393 |

## Overall / general ROUGE F1 (SPACE `general`, split = all)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | N |
| --- | ---: | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2842 | 0.0921 | 0.1774 | 50 |
| M2 Trước sentiment (abstractive) | 0.3604 | 0.0910 | 0.2076 | 50 |
| M3 Sau sentiment — Keyword | 0.3414 | 0.0871 | 0.2106 | 50 |
| M4 Sau sentiment — BERT-ABSA | 0.3430 | 0.0865 | 0.2088 | 50 |

## Winner per overall/general metric (SPACE, split = all)

| Metric | Best method | Score |
| --- | --- | ---: |
| ROUGE-1 | M2 Trước sentiment (abstractive) | 0.3604 |
| ROUGE-2 | M1 SemAE gốc (extractive) | 0.0921 |
| ROUGE-L | M3 Sau sentiment — Keyword | 0.2106 |

## Winner per metric (macro, split = all)

| Metric | Best method | Score |
| --- | --- | ---: |
| ROUGE-1 | M3 Sau sentiment — Keyword | 0.3094 |
| ROUGE-2 | M3 Sau sentiment — Keyword | 0.0886 |
| ROUGE-L | M3 Sau sentiment — Keyword | 0.2398 |

## Macro ROUGE F1 by split


### dev

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3073 | 0.0840 | 0.2233 |
| M2 Trước sentiment (abstractive) | 0.3078 | 0.0804 | 0.2331 |
| M3 Sau sentiment — Keyword | 0.3122 | 0.0850 | 0.2426 |
| M4 Sau sentiment — BERT-ABSA | 0.3053 | 0.0815 | 0.2378 |

### test

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3018 | 0.0894 | 0.2203 |
| M2 Trước sentiment (abstractive) | 0.3062 | 0.0916 | 0.2364 |
| M3 Sau sentiment — Keyword | 0.3069 | 0.0915 | 0.2367 |
| M4 Sau sentiment — BERT-ABSA | 0.3101 | 0.0941 | 0.2394 |

### all

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3041 | 0.0868 | 0.2218 |
| M2 Trước sentiment (abstractive) | 0.3068 | 0.0860 | 0.2346 |
| M3 Sau sentiment — Keyword | 0.3094 | 0.0886 | 0.2398 |
| M4 Sau sentiment — BERT-ABSA | 0.3080 | 0.0886 | 0.2393 |

## Per-aspect ROUGE-1 F1 (split = all)

| Method | Building | Cleanliness | Food | Location | Rooms | Service |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2671 | 0.2635 | 0.3068 | 0.3434 | 0.3429 | 0.3010 |
| M2 Trước sentiment (abstractive) | 0.2605 | 0.3136 | 0.2888 | 0.3510 | 0.3123 | 0.3145 |
| M3 Sau sentiment — Keyword | 0.2618 | 0.3379 | 0.2949 | 0.3219 | 0.3156 | 0.3242 |
| M4 Sau sentiment — BERT-ABSA | 0.2496 | 0.3397 | 0.2915 | 0.3401 | 0.3150 | 0.3121 |

## Notes

- ROUGE-1.5.5 via pyrouge + Strawberry Perl (Windows).
- Gold: `data/space/json/space_summ.json`, 3 references per aspect and 3 references for `general`.
- 6 flat generic aspects are averaged into MACRO; `GENERAL` scores the overall entity summary separately when present.
- SPACE has no sentiment-level gold references, so sentiment split is visualized but not scored independently.
- Evidence selection uses a score threshold of 0.0082; since the SPACE evidence scores top out at ~0.0081, this effectively keeps all threshold-eligible evidence (no additional filtering).
- M3/M4 concatenate positive + negative generated summaries per aspect.
- M1 = raw SemAE sentences; M2 = FLAN-T5 rewrite (no split); M3 = keyword-sentiment split; M4 = BERT-ABSA-sentiment split.
