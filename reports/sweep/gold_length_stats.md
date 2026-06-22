# Gold reference length statistics

Word counts (whitespace split) of human reference summaries. `tok≈` columns multiply words by 1.3 (FLAN-T5 sub-word estimate) to calibrate abstractive `--max_new_tokens`.


## SPACE

| Level | n | min | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aspect | 900 | 5 | 23 | 25.9 | 46 | 75 |
| general | 150 | 13 | 72 | 71.4 | 97 | 100 |

- aspect refs: median 23 w (~30 tok), p90 46 w (~60 tok)
- general refs: median 72 w (~94 tok), p90 97 w (~126 tok)

## HASOS

| Level | n | min | median | mean | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aspect | 499 | 3 | 51 | 64.0 | 132 | 187 |
| general | — | — | — | — | — | — |

- aspect refs: median 51 w (~66 tok), p90 132 w (~172 tok)
