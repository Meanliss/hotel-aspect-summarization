#!/usr/bin/env python3
r"""Export the hyperparameter-sweep results into one JSON for the web app.

Reads the per-cell ROUGE JSONs that sweep_params.py / score_rouge_compare.py
write under reports/sweep/ (rouge_<method>_<dataset>_<phase>_<valuetag>.json) and
collapses each to its macro ROUGE F1 (split=all, fixed denominator) plus mean
coverage. The web Optimality view renders the threshold/token curves that are
available, marks the best value per (dataset, phase, method), and shows the
current code default alongside the recommended value.

Crucially, every cell carries COVERAGE next to ROUGE. A tighter threshold that
scores higher only because it answered fewer (aspect, entity) instances is NOT
better - the fixed-denominator scorer already penalises empties with 0, and the
coverage field makes any remaining sparsity visible to the reader.

Output: web/public/data/sweep.json (shape consumed by the Optimality view).

Run with the project venv python:
    .venv/Scripts/python.exe scripts/export_sweep_web_data.py
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(REPO, "reports", "sweep")

# Current production defaults, per the code (src/aspect_inference.py and
# synthesize_aspect_summaries.py). Marked as the default in the web view.
DEFAULTS = {
    "threshold": {"space": 0.0082, "hasos": 0.005},
    "tokabs": {"space": 192, "hasos": 192},
}

METHOD_META = {
    "m1": {"short": "M1", "label": "M1 - Extractive (SemAE)", "color": "slate"},
    "m2": {"short": "M2", "label": "M2 - Abstractive (no sentiment)",
           "color": "sky"},
    "m3": {"short": "M3", "label": "M3 - Sentiment split: Keyword",
           "color": "emerald"},
    "m4": {"short": "M4", "label": "M4 - Sentiment split: BERT-ABSA",
           "color": "violet"},
}

PHASE_META = {
    "threshold": {
        "label": "Evidence score threshold",
        "param": "--evidence_score_threshold",
        "note": "SemAE KL cutoff deciding how much evidence feeds the "
                "abstractive stage (M2/M3/M4; M1 is independent). Tightening it "
                "drops sentences, so aspects/entities can produce no summary - "
                "those count as ROUGE 0 here (fixed denominator) and the "
                "coverage column makes the collapse visible.",
    },
    "tokabs": {
        "label": "Abstractive token budget",
        "param": "--max_new_tokens",
        "note": "FLAN-T5 output length for M2/M3/M4. Each token-budget series "
                "holds evidence threshold fixed at the selected method-specific "
                "threshold used for that cell set.",
    },
}

DATASET_META = {
    "space": {"label": "SPACE", "maxBar": 0.4},
    "hasos": {"label": "HASOS", "maxBar": 0.25},
}

CELL_RE = re.compile(
    r"rouge_(m[1-4])_(space|hasos)_(threshold|tokabs)_(.+)\.json$")


def untag(vt: str):
    """Inverse of sweep_params.tag(): 0p0067 -> 0.0067, neg0p02 -> -0.02."""
    s = vt.replace("neg", "-").replace("p", ".")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def macro_all(data):
    """Return macro rouge1/2/L (split=all) + mean coverage over aspect keys."""
    allm = data.get("by_split", {}).get("all", {})
    macro = allm.get("MACRO", {})
    if not macro:
        return None
    covs = [v.get("coverage") for k, v in allm.items()
            if k not in ("MACRO", "GENERAL") and isinstance(v, dict)
            and v.get("coverage") is not None]
    mean_cov = sum(covs) / len(covs) if covs else None
    return {
        "rouge1": macro.get("rouge1"),
        "rouge2": macro.get("rouge2"),
        "rougeL": macro.get("rougeL"),
        "coverage": round(mean_cov, 4) if mean_cov is not None else None,
        "n_aspects": len(covs),
    }


def load_cells():
    """Return {(dataset, phase, method): {value: cell}}."""
    cells: dict = {}
    for path in glob.glob(os.path.join(SWEEP_DIR, "rouge_*.json")):
        m = CELL_RE.search(os.path.basename(path))
        if not m:
            continue
        method, dataset, phase, vt = m.groups()
        with io.open(path, encoding="utf-8") as f:
            data = json.load(f)
        ma = macro_all(data)
        if ma is None:
            continue
        cells.setdefault((dataset, phase, method), {})[untag(vt)] = ma
    return cells


def best_value(series, metric="rouge1"):
    best, best_v = None, -1.0
    for val, cell in series.items():
        r = cell.get(metric)
        if r is not None and r > best_v:
            best, best_v = val, r
    return best


def build_payload(cells):
    datasets = {}
    for (dataset, phase, method), series in cells.items():
        default_val = DEFAULTS.get(phase, {}).get(dataset)
        best = best_value(series, "rouge1")
        points = []
        for val in sorted(series.keys(),
                          key=lambda v: (isinstance(v, str), v)):
            cell = series[val]
            points.append({
                "value": val,
                "rouge1": cell["rouge1"],
                "rouge2": cell["rouge2"],
                "rougeL": cell["rougeL"],
                "coverage": cell["coverage"],
                "n_aspects": cell["n_aspects"],
                "is_default": default_val is not None and val == default_val,
                "is_best": val == best,
            })
        # optimality verdict, mirroring build_sweep_tables.conclude()
        def_cell = series.get(default_val)
        verdict = None
        if def_cell is not None and best is not None:
            best_cell = series[best]
            if best == default_val:
                verdict = {
                    "status": "default_optimal",
                    "default": default_val,
                    "best": best,
                    "best_r1": best_cell["rouge1"],
                    "default_r1": def_cell["rouge1"],
                    "delta": 0.0,
                }
            else:
                bc, dc = best_cell.get("coverage"), def_cell.get("coverage")
                cov_ok = (
                    bc is not None
                    and dc is not None
                    and bc + 0.02 >= dc
                )
                gain = best_cell["rouge1"] - def_cell["rouge1"]
                verdict = {
                    "status": "switch" if (cov_ok and gain > 0)
                    else "keep_default_coverage_artifact",
                    "default": default_val,
                    "best": best,
                    "best_r1": best_cell["rouge1"],
                    "default_r1": def_cell["rouge1"],
                    "best_cov": bc,
                    "default_cov": dc,
                    "delta": round(gain, 5),
                }
        block = datasets.setdefault(dataset, {})
        phase_block = block.setdefault(phase, {"methods": {}})
        phase_block["methods"][method] = {
            "default": default_val,
            "best": best,
            "points": points,
            "verdict": verdict,
        }
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/data/sweep.json")
    args = ap.parse_args()

    cells = load_cells()
    datasets = build_payload(cells)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision_metric": "macro ROUGE-1 F1, split=all, fixed denominator",
        "method_meta": METHOD_META,
        "phase_meta": PHASE_META,
        "dataset_meta": DATASET_META,
        "datasets": datasets,
    }

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_cells = sum(len(v) for v in cells.values())
    n_series = len(cells)
    print(f"aggregated {n_cells} cells across {n_series} "
          f"(dataset,phase,method) series")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()

