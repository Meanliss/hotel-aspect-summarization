# SPACE Official ROUGE Evaluation — hotel pipeline output

Metrics: pyrouge / ROUGE-1.5.5 F1 against human SPACE gold summaries.

Source CSV: `results/space_pipeline/space_qwen_native_full_qwenroute_20260627/space_qwen_native_full_qwenroute_20260627_final_summary.csv`

## Macro ROUGE F1

| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| dev | 0.2552 | 0.0527 | 0.1673 |
| test | 0.2457 | 0.0616 | 0.1721 |
| all | 0.2500 | 0.0572 | 0.1695 |

## dev by aspect

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L | N |
| --- | ---: | ---: | ---: | ---: |
| building | 0.2291 | 0.0359 | 0.1427 | 25 |
| cleanliness | 0.1982 | 0.0527 | 0.1437 | 25 |
| food | 0.2487 | 0.0457 | 0.1569 | 25 |
| location | 0.3278 | 0.0850 | 0.2072 | 25 |
| rooms | 0.2667 | 0.0461 | 0.1747 | 25 |
| service | 0.2610 | 0.0512 | 0.1786 | 25 |

## test by aspect

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L | N |
| --- | ---: | ---: | ---: | ---: |
| building | 0.1883 | 0.0244 | 0.1283 | 25 |
| cleanliness | 0.2172 | 0.0769 | 0.1722 | 25 |
| food | 0.2340 | 0.0597 | 0.1654 | 25 |
| location | 0.3048 | 0.0823 | 0.2029 | 25 |
| rooms | 0.2581 | 0.0542 | 0.1696 | 25 |
| service | 0.2719 | 0.0724 | 0.1942 | 25 |

## all by aspect

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L | N |
| --- | ---: | ---: | ---: | ---: |
| building | 0.2077 | 0.0301 | 0.1351 | 50 |
| cleanliness | 0.2072 | 0.0649 | 0.1578 | 50 |
| food | 0.2402 | 0.0525 | 0.1603 | 50 |
| location | 0.3176 | 0.0838 | 0.2060 | 50 |
| rooms | 0.2619 | 0.0502 | 0.1718 | 50 |
| service | 0.2653 | 0.0615 | 0.1857 | 50 |

## Notes

- Output rows use SPACE's six flat aspects directly; no hotel-taxonomy projection was applied.
