# M1 Token Budget Sweep

M1 is extractive, so `B` is a word-level truncation budget, not a generative `max_new_tokens` value.
The official paper baseline is `space_hasos_full_e20` with `B=40`.
The table below is a post-hoc replay from `outputs/space_hasos_threshold_full_lines.jsonl`, useful as a sensitivity check for `B <= 120`.

| B words | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|---:|
| 40 **best replay** | 0.2078 | 0.0591 | 0.1245 |
| 64 | 0.1592 | 0.0524 | 0.0986 |
| 80 | 0.1374 | 0.0479 | 0.0869 |
| 96 | 0.1210 | 0.0447 | 0.0781 |
| 120 | 0.1031 | 0.0403 | 0.0680 |

Best replay by ROUGE-1: `B=40`.
Official M1 baseline (`space_hasos_full_e20`, `B=40`): ROUGE-1=0.2035, ROUGE-2=0.0550, ROUGE-L=0.1223.

Budgets above 120 require rerunning SemAE or exporting the full ranked sentence list before truncation.
