# HASOS Hotel Adaptation

This project keeps the original SemAE implementation and adds a HASOS hotel setup
for `space_summ_hasos.json` plus the Google Sheet SPACE MAPPING taxonomy.

## Current Evaluation Note

This file describes the early HASOS adaptation setup. The current repo has a
newer paper/web analysis that scores HASOS against `data/hasos/hasos_summ.json`
by aggregating the 29 child aspects to the parent aspects with available gold
references. The original SemAE `data/hasos/gold/<aspect>/#ID#_[012].txt`
pyrouge layout is still absent, so upstream-style SemAE evaluation remains a
separate compatibility task.

The completed threshold sweep shows the current production defaults are optimal
inside the tested grid: SPACE `T=0.0082` and HASOS `T=0.005`. Token-budget sweep
cells have not been run yet.

## Prepared Inputs

- `data/hasos/aspect_taxonomy.tsv`: aspect taxonomy copied from the provided Google Sheet tab.
- `data/hasos/aspect_taxonomy.json`: generated machine-readable taxonomy.
- `data/hasos/hasos_summ.json`: generated copy of the local hotel review JSON.
- `data/seeds_hasos/*.txt`: generated SemAE seed files, one per aspect code.

## Commands

From `scripts/`:

```powershell
python .\prepare_hasos.py
python .\validate_hasos.py
```

Train a HASOS model on CPU:

```powershell
.\train_hasos.ps1 -Gpu -1 -Epochs 10 -RunId hasos_run1
```

Run aspect extraction after training:

```powershell
.\evaluate_hasos_aspect.ps1 -Model ..\models\hasos_run1_10_model.pt -Gpu -1
```

The aspect extraction command uses `--no_eval` for the original SemAE pyrouge
path because the provided local data has reviews and ratings, but not the
upstream `gold/<aspect>/` directory layout.
