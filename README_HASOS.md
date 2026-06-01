# HASOS Hotel Adaptation

This project keeps the original SemAE implementation and adds a HASOS hotel setup
for `space_summ_hasos.json` plus the Google Sheet SPACE MAPPING taxonomy.

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

The aspect extraction command uses `--no_eval` because the provided local data has
reviews and ratings, but no human gold summaries for ROUGE evaluation.
