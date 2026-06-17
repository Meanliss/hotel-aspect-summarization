# ROUGE Comparison — HASOS (4 methods)

Official pyrouge (ROUGE-1.5.5) F1 against human gold summaries, aggregated to the 4 parent aspects (facility/amenity/service/experience). All methods share identical SemAE sentence selection; they differ only in how selected evidence is rendered.

## Macro ROUGE F1 (mean over 4 aspects, split = all)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2035 | 0.0550 | 0.1223 |
| M2 Trước sentiment (abstractive) | 0.2258 | 0.0377 | 0.1525 |
| M3 Sau sentiment — Keyword | 0.2253 | 0.0520 | 0.1637 |
| M4 Sau sentiment — BERT-ABSA | 0.2128 | 0.0407 | 0.1394 |

## Winner per metric (macro, split = all)

| Metric | Best method | Score |
| --- | --- | ---: |
| ROUGE-1 | M2 Trước sentiment (abstractive) | 0.2258 |
| ROUGE-2 | M1 SemAE gốc (extractive) | 0.0550 |
| ROUGE-L | M3 Sau sentiment — Keyword | 0.1637 |

## Macro ROUGE F1 by split


### dev

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2086 | 0.0540 | 0.1225 |
| M2 Trước sentiment (abstractive) | 0.2228 | 0.0358 | 0.1469 |
| M3 Sau sentiment — Keyword | 0.2291 | 0.0577 | 0.1702 |
| M4 Sau sentiment — BERT-ABSA | 0.2183 | 0.0408 | 0.1387 |

### test

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2001 | 0.0566 | 0.1231 |
| M2 Trước sentiment (abstractive) | 0.2311 | 0.0401 | 0.1591 |
| M3 Sau sentiment — Keyword | 0.2262 | 0.0491 | 0.1624 |
| M4 Sau sentiment — BERT-ABSA | 0.2096 | 0.0411 | 0.1410 |

### all

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.2035 | 0.0550 | 0.1223 |
| M2 Trước sentiment (abstractive) | 0.2258 | 0.0377 | 0.1525 |
| M3 Sau sentiment — Keyword | 0.2253 | 0.0520 | 0.1637 |
| M4 Sau sentiment — BERT-ABSA | 0.2128 | 0.0407 | 0.1394 |

## Per-aspect ROUGE-1 F1 (split = all)

| Method | Facility | Amenity | Service | Experience |
| --- | ---: | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.3383 | 0.1917 | 0.2066 | 0.0774 |
| M2 Trước sentiment (abstractive) | 0.2446 | 0.2523 | 0.2564 | 0.1500 |
| M3 Sau sentiment — Keyword | 0.2400 | 0.2116 | 0.3262 | 0.1234 |
| M4 Sau sentiment — BERT-ABSA | 0.2913 | 0.2386 | 0.2070 | 0.1143 |

## Notes

- ROUGE-1.5.5 via pyrouge + Strawberry Perl (Windows).
- Gold: `data/hasos/hasos_summ.json`, multi-reference where available.
- 29 sub-aspects aggregated to 4 gold parents; Branding/Loyalty omitted (no gold).
- M3/M4 concatenate positive + negative generated summaries per aspect.
- M1 = raw SemAE sentences; M2 = FLAN-T5 rewrite (no split); M3 = keyword-sentiment split; M4 = BERT-ABSA-sentiment split.
