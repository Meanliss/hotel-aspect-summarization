#!/usr/bin/env python3
"""Run abstractive token-budget cells at fixed, method-specific thresholds.

This is deliberately separate from sweep_params.py because HASOS now has
method-specific exact-T winners above the original 0.005 evidence pool. The
token sweep should hold each method at its chosen threshold instead of falling
back to a stale shared cutoff.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SWEEP_DIR = REPO / "reports" / "sweep"
OUTPUTS = REPO / "outputs"

METHOD_SCORE = {
    "m2": "m2_abstractive",
    "m3": "m3_kw",
    "m4": "m4_bert",
}

DATASETS = {
    "space": {
        "gold": "data/space/json/space_summ.json",
        "universe_dir": "outputs/space_eval_4method",
        "run_id": "space_eval_4method",
        "include_general": True,
        "m2_hierarchical": False,
        "default_thresholds": {"m2": 0.0082, "m3": 0.0082, "m4": 0.0082},
        "evidence": {
            "m2": "outputs/space_eval_4method_threshold_evidence.jsonl",
            "m3": "outputs/space_eval_4method_kw_threshold_evidence.jsonl",
            "m4": "outputs/space_eval_4method_bert_threshold_evidence.jsonl",
        },
    },
    "hasos": {
        "gold": "data/hasos/hasos_summ.json",
        "universe_dir": "outputs/space_hasos_threshold_full",
        "include_general": False,
        "m2_hierarchical": True,
        "default_thresholds": {"m2": 0.0075, "m3": 0.0055, "m4": 0.005},
    },
}


def tag(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def parse_thresholds(raw: str | None, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        method, value = item.split("=", 1)
        values[method.strip()] = float(value)
    return values


def run(cmd: list[str], log_path: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    print("$", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(cmd, cwd=REPO, env=env)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")


def macro_all(result: dict) -> dict:
    return result.get("by_split", {}).get("all", {}).get("MACRO", {})


def hasos_evidence(method: str, threshold: float) -> tuple[str, str]:
    """Return (source_run_id, evidence_jsonl) for a HASOS method/T cell."""
    if abs(threshold - 0.005) < 1e-12:
        if method == "m4":
            return "space_hasos_threshold_full", (
                "outputs/space_hasos_threshold_full_bert_threshold_evidence.jsonl"
            )
        return "space_hasos_threshold_full", (
            "outputs/space_hasos_threshold_full_threshold_evidence.jsonl"
        )

    run_id = f"space_hasos_threshold_exact_{tag(threshold)}"
    if method == "m4":
        evidence = OUTPUTS / f"{run_id}_bert_threshold_evidence.jsonl"
    else:
        evidence = OUTPUTS / f"{run_id}_threshold_evidence.jsonl"
    if not evidence.exists() or evidence.stat().st_size == 0:
        raise FileNotFoundError(
            f"missing exact evidence for {method} T={threshold}: {evidence}"
        )
    return run_id, str(evidence.relative_to(REPO))


def dataset_method_source(dataset: str, method: str,
                          threshold: float) -> tuple[str, str]:
    if dataset == "hasos":
        return hasos_evidence(method, threshold)
    cfg = DATASETS[dataset]
    return cfg["run_id"], cfg["evidence"][method]


def synthesize(dataset: str, method: str, budget: int,
               threshold: float, force: bool) -> str:
    cfg = DATASETS[dataset]
    run_id, evidence_jsonl = dataset_method_source(dataset, method, threshold)
    bt = tag(budget)
    tt = tag(threshold)
    out_run_id = f"sweep_{dataset}_{method}_tokabs_{bt}_thr_{tt}"
    expected = OUTPUTS / (out_run_id + ("_parent" if method == "m2" and cfg.get("m2_hierarchical") else ""))
    if expected.exists() and any(expected.rglob("*")) and not force:
        print(f"[skip] existing synthesis: {expected}", flush=True)
        return out_run_id

    cmd = [
        PY, "-u", "scripts/synthesize_aspect_summaries.py",
        "--run_id", run_id,
        "--output_run_id", out_run_id,
        "--source_mode", "threshold_evidence",
        "--evidence_jsonl", evidence_jsonl,
        "--evidence_score_threshold", str(threshold),
        "--model_name", "google/flan-t5-base",
        "--device", "auto",
        "--max_new_tokens", str(budget),
        "--entity_max_new_tokens", str(budget),
        "--overwrite",
    ]
    if method in {"m3", "m4"}:
        cmd.append("--split_sentiment")
    if method == "m2" and cfg.get("m2_hierarchical"):
        cmd += ["--hierarchical", "--skip_entity_summary"]

    run(cmd, SWEEP_DIR / f"_log_{method}_{dataset}_tokabs_{bt}_thr_{tt}.log")
    return out_run_id


def score(dataset: str, method: str, budget: int, out_run_id: str) -> dict:
    cfg = DATASETS[dataset]
    out_json = SWEEP_DIR / f"rouge_{method}_{dataset}_tokabs_{tag(budget)}.json"
    cmd = [
        PY, "-u", "scripts/score_rouge_compare.py",
        "--method", METHOD_SCORE[method],
        "--dataset", dataset,
        "--gold", cfg["gold"],
        "--out", str(out_json.relative_to(REPO)),
        "--fixed_denominator",
        "--universe_dir", cfg["universe_dir"],
    ]
    if method == "m2":
        if cfg.get("m2_hierarchical"):
            cmd += ["--parent_dir", f"outputs/{out_run_id}_parent"]
        else:
            cmd += ["--parent_dir", f"outputs/{out_run_id}"]
    else:
        cmd += ["--senti_dir", f"outputs/{out_run_id}"]
    if cfg.get("include_general"):
        cmd.append("--include_general")
    run(cmd)
    with io.open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["space", "hasos"], required=True)
    ap.add_argument("--grid", default="96,128,192,256")
    ap.add_argument("--methods", default="m2,m3,m4")
    ap.add_argument(
        "--thresholds",
        default=None,
        help="method-specific thresholds, e.g. m2=0.0075,m3=0.0055,m4=0.005",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = DATASETS[args.dataset]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [int(x) for x in args.grid.split(",") if x.strip()]
    thresholds = parse_thresholds(args.thresholds, cfg["default_thresholds"])
    rows = []
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    for method in methods:
        threshold = thresholds[method]
        for budget in budgets:
            print(
                f"\n=== [tokabs] {args.dataset} {method} B={budget} "
                f"T={threshold} ===",
                flush=True,
            )
            out_run_id = synthesize(args.dataset, method, budget, threshold,
                                    args.force)
            result = score(args.dataset, method, budget, out_run_id)
            ma = macro_all(result)
            rows.append({
                "method": method,
                "value": budget,
                "threshold": threshold,
                **ma,
            })
            print(
                f"[score] {method} B={budget} T={threshold} "
                f"R1={ma.get('rouge1'):.5f} R2={ma.get('rouge2'):.5f} "
                f"RL={ma.get('rougeL'):.5f}",
                flush=True,
            )

    out = SWEEP_DIR / f"token_abstractive_{args.dataset}_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "phase": "token_abstractive",
            "grid": args.grid,
            "thresholds": thresholds,
            "rows": rows,
        }, f, indent=2)
    print(f"\nwritten summary -> {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
