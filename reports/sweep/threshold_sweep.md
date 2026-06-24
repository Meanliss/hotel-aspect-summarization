# Threshold sweep — macro ROUGE F1 (split=all), fixed denominator

Every value scores the SAME (split, entity) universe; instances with no generated summary count as ROUGE 0. So a stricter threshold that drops evidence is penalised through both ROUGE and the coverage column. `*` = current code default. ROUGE-1 is the decision metric.


## SPACE


**M2 abstractive** (default evidence_score_threshold = 0.0082)

| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.00164 | 0.00045 | 0.00102 | 0.01 | 6 | -0.30512 |
| 0.005 | 0.04902 | 0.01644 | 0.03897 | 0.15 | 6 | -0.25774 |
| 0.0067 | 0.24393 | 0.07272 | 0.19185 | 0.78 | 6 | -0.06283 |
| 0.0075 | 0.29910 | 0.08408 | 0.22951 | 0.98 | 6 | -0.00766 |
| 0.0082* **(best)** | 0.30676 | 0.08602 | 0.23464 | 1.00 | 6 | +0.00000 |

**M3 kw-sentiment** (default evidence_score_threshold = 0.0082)

| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.00164 | 0.00045 | 0.00102 | 0.01 | 6 | -0.30064 |
| 0.005 | 0.04684 | 0.01555 | 0.03740 | 0.14 | 6 | -0.25545 |
| 0.0067 | 0.24269 | 0.07457 | 0.19242 | 0.76 | 6 | -0.05960 |
| 0.0075 | 0.29444 | 0.08525 | 0.22926 | 0.95 | 6 | -0.00785 |
| 0.0082* **(best)** | 0.30229 | 0.08666 | 0.23448 | 0.98 | 6 | +0.00000 |

**M4 bert-sentiment** (default evidence_score_threshold = 0.0082)

| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.003 | 0.00164 | 0.00045 | 0.00102 | 0.01 | 6 | -0.29323 |
| 0.005 | 0.04233 | 0.01251 | 0.03298 | 0.13 | 6 | -0.25254 |
| 0.0067 | 0.23385 | 0.07325 | 0.18548 | 0.73 | 6 | -0.06102 |
| 0.0075 | 0.28459 | 0.08329 | 0.22197 | 0.92 | 6 | -0.01028 |
| 0.0082* **(best)** | 0.29487 | 0.08524 | 0.22917 | 0.95 | 6 | +0.00000 |

## HASOS


**M2 abstractive** (default evidence_score_threshold = 0.005)

| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.005* **(best)** | 0.19661 | 0.03720 | 0.13542 | 0.98 | 4 | +0.00000 |

**M3 kw-sentiment** (default evidence_score_threshold = 0.005)

| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 **(best)** | 0.08294 | 0.01519 | 0.06258 | 0.47 | 4 |  |
