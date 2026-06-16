# ROUGE Comparison — HASOS (4 methods)

Official pyrouge (ROUGE-1.5.5) F1 against human gold summaries, aggregated to the 4 parent aspects (facility/amenity/service/experience). All methods share identical SemAE sentence selection; they differ only in how selected evidence is rendered.

## Macro ROUGE F1 (mean over 4 aspects, split = all)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.1031 | 0.0403 | 0.0680 |
| M2 Trước sentiment (abstractive) | 0.2002 | 0.0378 | 0.1380 |
| M3 Sau sentiment — Keyword | 0.2253 | 0.0520 | 0.1637 |
| M4 Sau sentiment — BERT-ABSA | 0.2128 | 0.0407 | 0.1394 |

## Winner per metric (macro, split = all)

| Metric | Best method | Score |
| --- | --- | ---: |
| ROUGE-1 | M3 Sau sentiment — Keyword | 0.2253 |
| ROUGE-2 | M3 Sau sentiment — Keyword | 0.0520 |
| ROUGE-L | M3 Sau sentiment — Keyword | 0.1637 |

## Macro ROUGE F1 by split


### dev

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.1068 | 0.0408 | 0.0697 |
| M2 Trước sentiment (abstractive) | 0.2140 | 0.0420 | 0.1442 |
| M3 Sau sentiment — Keyword | 0.2291 | 0.0577 | 0.1702 |
| M4 Sau sentiment — BERT-ABSA | 0.2183 | 0.0408 | 0.1387 |

### test

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.1001 | 0.0401 | 0.0666 |
| M2 Trước sentiment (abstractive) | 0.1888 | 0.0347 | 0.1336 |
| M3 Sau sentiment — Keyword | 0.2262 | 0.0491 | 0.1624 |
| M4 Sau sentiment — BERT-ABSA | 0.2096 | 0.0411 | 0.1410 |

### all

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.1031 | 0.0403 | 0.0680 |
| M2 Trước sentiment (abstractive) | 0.2002 | 0.0378 | 0.1380 |
| M3 Sau sentiment — Keyword | 0.2253 | 0.0520 | 0.1637 |
| M4 Sau sentiment — BERT-ABSA | 0.2128 | 0.0407 | 0.1394 |

## Per-aspect ROUGE-1 F1 (split = all)

| Method | Facility | Amenity | Service | Experience |
| --- | ---: | ---: | ---: | ---: |
| M1 SemAE gốc (extractive) | 0.1825 | 0.0942 | 0.1013 | 0.0344 |
| M2 Trước sentiment (abstractive) | 0.2260 | 0.2214 | 0.2296 | 0.1237 |
| M3 Sau sentiment — Keyword | 0.2400 | 0.2116 | 0.3262 | 0.1234 |
| M4 Sau sentiment — BERT-ABSA | 0.2913 | 0.2386 | 0.2070 | 0.1143 |

## Notes

- ROUGE-1.5.5 via pyrouge + Strawberry Perl (Windows).
- Gold: `data/hasos/hasos_summ.json`, multi-reference where available.
- 29 sub-aspects aggregated to 4 gold parents; Branding/Loyalty omitted (no gold).
- M3/M4 concatenate positive + negative generated summaries per aspect.
- M1 = raw SemAE sentences; M2 = FLAN-T5 rewrite (no split); M3 = keyword-sentiment split; M4 = BERT-ABSA-sentiment split.
