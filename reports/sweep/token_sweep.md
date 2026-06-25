# Token-budget sweep - macro ROUGE F1 (split=all), fixed denominator

Abstractive `--max_new_tokens` (M2/M3/M4). Each series holds the evidence threshold fixed at that method's selected threshold for the cell set. `*` = current code default (192). ROUGE-1 is the decision metric.


## SPACE

_(no cells yet)_

## HASOS


**M2 abstractive** (default max_new_tokens (abstractive) = 192)

| value | R1 | R2 | RL | coverage | n_asp | Delta R1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.23093 | 0.04262 | 0.15231 | 1.00 | 4 | +0.00323 |
| 128 **(best)** | 0.23242 | 0.04406 | 0.15033 | 1.00 | 4 | +0.00472 ok |
| 192* | 0.22770 | 0.04431 | 0.14472 | 1.00 | 4 | +0.00000 |
| 256 | 0.22427 | 0.04420 | 0.14183 | 1.00 | 4 | -0.00343 |

**M3 kw-sentiment** (default max_new_tokens (abstractive) = 192)

| value | R1 | R2 | RL | coverage | n_asp | Delta R1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 **(best)** | 0.26201 | 0.07124 | 0.16838 | 0.99 | 4 | +0.00065 ok |
| 128 | 0.26162 | 0.07116 | 0.16810 | 0.99 | 4 | +0.00027 |
| 192* | 0.26135 | 0.07108 | 0.16796 | 0.99 | 4 | +0.00000 |
| 256 | 0.26132 | 0.07107 | 0.16794 | 0.99 | 4 | -0.00004 |

**M4 bert-sentiment** (default max_new_tokens (abstractive) = 192)

| value | R1 | R2 | RL | coverage | n_asp | Delta R1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 96 **(best)** | 0.20890 | 0.04004 | 0.13677 | 0.98 | 4 | +0.00052 ok |
| 128 | 0.20867 | 0.03997 | 0.13655 | 0.98 | 4 | +0.00029 |
| 192* | 0.20838 | 0.03987 | 0.13634 | 0.98 | 4 | +0.00000 |
| 256 | 0.20837 | 0.03986 | 0.13634 | 0.98 | 4 | -0.00001 |
