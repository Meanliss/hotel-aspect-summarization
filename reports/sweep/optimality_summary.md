# Optimality summary

For each swept parameter, the value with the highest macro ROUGE-1 (fixed-denominator, split=all) is compared against the current code default. A non-default winner is only reported as an improvement when its coverage is within 0.02 of the default's — otherwise the gain is a coverage artifact and the default is kept.


## SPACE

- **evidence_score_threshold / M2 abstractive**: **default 0.0082 is optimal** (R1=0.30676)
- **evidence_score_threshold / M3 kw-sentiment**: **default 0.0082 is optimal** (R1=0.30229)
- **evidence_score_threshold / M4 bert-sentiment**: **default 0.0082 is optimal** (R1=0.29487)

## HASOS

- **evidence_score_threshold / M2 abstractive**: **default 0.005 is optimal** (R1=0.19661)
- **evidence_score_threshold / M3 kw-sentiment**: default 0.005 not in grid; best=0.0 R1=0.08294
