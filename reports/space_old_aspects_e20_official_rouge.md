# SPACE Original Aspect ROUGE Evaluation — `space_old_aspects_e20`

This rerun uses the epoch-20 SPACE-trained SemAE checkpoint on the original SPACE benchmark with the six original SPACE aspects. It is the old/as-original baseline, not the HASOS 29-aspect adaptation.

## Contract

- Dataset: `data/space/json/space_summ.json`
- Gold: `data/space/gold`
- Aspects: `building, cleanliness, food, location, rooms, service`
- Evaluation: `src/aspect_inference.py` without `--no_eval`, using pyrouge
- Outputs: `outputs/eval_space_old_aspects_e20.json` and `outputs/eval_space_old_aspects_e20.txt`

## Output Health

| Check | Value |
| --- | ---: |
| Aspect output files | 300 |
| Empty output files | 0 |
| Aspect folders | 6 |
| Files per aspect | 50 |
| Fatal log matches | 0 |

## ROUGE F1 by Split and Aspect

### Dev

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.28725 | 0.06653 | 0.20737 |
| cleanliness | 0.27100 | 0.08440 | 0.22622 |
| food | 0.28764 | 0.05630 | 0.19063 |
| location | 0.35313 | 0.10531 | 0.23191 |
| rooms | 0.34705 | 0.09167 | 0.23710 |
| service | 0.29477 | 0.09183 | 0.22371 |
| **Macro** | **0.30681** | **0.08267** | **0.21949** |

### Test

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.24752 | 0.05365 | 0.18305 |
| cleanliness | 0.25916 | 0.07509 | 0.20481 |
| food | 0.33286 | 0.12726 | 0.24688 |
| location | 0.32759 | 0.10673 | 0.23343 |
| rooms | 0.34250 | 0.08078 | 0.23113 |
| service | 0.29170 | 0.08374 | 0.20828 |
| **Macro** | **0.30022** | **0.08787** | **0.21793** |

### All

| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: |
| building | 0.26558 | 0.05907 | 0.19387 |
| cleanliness | 0.26527 | 0.08034 | 0.21591 |
| food | 0.30896 | 0.09167 | 0.21832 |
| location | 0.34134 | 0.10718 | 0.23335 |
| rooms | 0.34411 | 0.08686 | 0.23451 |
| service | 0.29434 | 0.08879 | 0.21714 |
| **Macro** | **0.30327** | **0.08565** | **0.21885** |

## Note

The JSON is byte-identical to the previous `space_eval_e20` official ROUGE run, confirming the provenance changes did not alter the old SPACE evaluation path.
