# Optimality summary

For each swept parameter, the value with the highest macro ROUGE-1 (fixed-denominator, split=all) is compared against the current code default. A non-default winner is reported as an improvement when it does not lose more than 0.02 coverage versus the default; otherwise the gain is treated as a low-coverage artifact and the default is kept.


## SPACE

- **evidence_score_threshold / M2 abstractive**: **default 0.0082 is optimal** (R1=0.30676)
- **evidence_score_threshold / M3 kw-sentiment**: **default 0.0082 is optimal** (R1=0.30229)
- **evidence_score_threshold / M4 bert-sentiment**: **default 0.0082 is optimal** (R1=0.29487)

## HASOS

- **evidence_score_threshold / M2 abstractive**: **0.0075 beats default 0.005** by Delta R1=+0.03109 at equal coverage (cov 1.00 vs 0.98) -> recommend switching
- **evidence_score_threshold / M3 kw-sentiment**: **0.0055 beats default 0.005** by Delta R1=+0.10553 at equal coverage (cov 0.99 vs 0.73) -> recommend switching
- **evidence_score_threshold / M4 bert-sentiment**: **default 0.005 is optimal** (R1=0.20838)
- **max_new_tokens (abstractive) / M2 abstractive**: **128 beats default 192** by Delta R1=+0.00472 at equal coverage (cov 1.00 vs 1.00) -> recommend switching
- **max_new_tokens (abstractive) / M3 kw-sentiment**: **96 beats default 192** by Delta R1=+0.00065 at equal coverage (cov 0.99 vs 0.99) -> recommend switching
- **max_new_tokens (abstractive) / M4 bert-sentiment**: **96 beats default 192** by Delta R1=+0.00052 at equal coverage (cov 0.98 vs 0.98) -> recommend switching
