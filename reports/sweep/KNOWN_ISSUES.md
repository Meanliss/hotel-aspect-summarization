# Sweep Known Issues And Current State

Updated: 2026-06-25

Read together with:
- `reports/sweep/threshold_sweep.md`
- `reports/sweep/optimality_summary.md`
- `web/public/data/sweep.json`

## Current Progress

| Area | Status |
|---|---|
| SPACE threshold (M2/M3/M4 x 5 values) | Done and verified |
| HASOS threshold (M2/M3/M4 x 4 values up to 0.005) | Done and verified |
| Token phase | Not run yet |

The threshold phase now has 27 scored cells in `web/public/data/sweep.json`.
Both datasets currently support the same conclusion: the production threshold is
optimal within the tested grid.

## Resolved: HASOS M2 Child-Only Layout Scored As Zero

Old symptom: HASOS M2 threshold sweep cells could produce many child files but
score as zero when evaluated as `--parent_dir`.

Root cause:
- HASOS child output directories use sub-aspect codes such as `FAC_BATH` and
  `AM_ENT`.
- The HASOS gold/evaluation keys are parent aspects: `FACILITY`, `AMENITY`,
  `SERVICE`, and `EXPERIENCE`.
- M2 does not use the sentiment resolver that aggregates sub-aspect codes to
  parents, so child-only output is not a valid parent-dir layout.

Fix in `scripts/sweep_params.py`:
- `CONFIG["hasos"]["m2_hierarchical"] = True`.
- HASOS M2 synthesis now passes `--hierarchical`.
- HASOS M2 scoring now points `--parent_dir` to
  `outputs/{out_run_id}_parent`.
- All scored sweep cells use `--fixed_denominator --universe_dir` so sparse
  outputs are penalized instead of shrinking the denominator.

Verified evidence:
- `reports/sweep/rouge_m2_hasos_threshold_0p0.json`
- `reports/sweep/rouge_m2_hasos_threshold_0p0025.json`
- `reports/sweep/rouge_m2_hasos_threshold_0p004.json`
- `reports/sweep/rouge_m2_hasos_threshold_0p005.json`

M2 HASOS threshold result:

| T | ROUGE-1 | Coverage |
|---:|---:|---:|
| 0.0 | 0.15999 | 0.8653 |
| 0.0025 | 0.19045 | 0.9259 |
| 0.004 | 0.19624 | 0.9644 |
| 0.005 | 0.19661 | 0.9822 |

## Resolved: HASOS M3/M4 Default Was Missing From The Grid

Old problem: before running `T=0.005` for HASOS M3 and M4, the summary could
only say that `0.004` was the best among the incomplete non-default cells.
That was not enough to support an optimality claim.

Fix:
- Ran HASOS M3 at `T=0.005`.
- Ran HASOS M4 at `T=0.005`.
- Rebuilt `threshold_sweep.md`, `optimality_summary.md`, and
  `web/public/data/sweep.json`.

Final HASOS default results:

| Method | Default T | Best T | Best ROUGE-1 |
|---|---:|---:|---:|
| M2 | 0.005 | 0.005 | 0.19661 |
| M3 | 0.005 | 0.005 | 0.15582 |
| M4 | 0.005 | 0.005 | 0.20838 |

## Remaining Limitation: Thresholds Above 0.005 On HASOS

HASOS evidence was pooled at a maximum threshold of 0.005. The current sweep can
only test tighter thresholds (`T <= 0.005`) by re-filtering the existing evidence
pool. Testing looser thresholds (`T > 0.005`) requires rerunning SemAE evidence
generation, which has not been done.

## Remaining Work: Token Phase

The token-budget phase has not been run for both datasets. The code path should
inherit the HASOS M2 hierarchical scoring fix through `out_dirs()`, but the first
HASOS M2 token cell should still be verified before trusting a full token sweep.

## Commit Hygiene

Do not commit scratch logs such as `_log_*`, `_run_*`, `_diag_*`, `_chk_*`, or
`_sanity_*`. Commit the scored cell JSONs, summary tables, scripts, and web data.
