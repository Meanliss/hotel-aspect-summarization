"""Sweep the extractive word budget B for the HASOS M1 baseline.

M1 is an extractive SemAE baseline: it writes selected evidence sentences, then
limits the output with ``truncate_summary(..., max_tokens=B, cut_sents=True)``.
The official HASOS M1 baseline in the paper uses ``space_hasos_full_e20`` with
``B=40``.  For sensitivity analysis, the repo also has a longer
``space_hasos_threshold_full`` export with ``B=120``.  Budgets at or below 120
can be replayed from that longer ranked sentence log without rerunning SemAE.

The script creates one output directory per budget and scores it with the
official HASOS ROUGE scorer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LINES_PATH = REPO / "outputs" / "space_hasos_threshold_full_lines.jsonl"
OUT_ROOT = REPO / "outputs"
REPORT_ROOT = REPO / "reports" / "sweep"
VENV_PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


def word_count(text: str) -> int:
    return len(text.split())


def truncate_summary(sentences: list[str], max_tokens: int) -> list[str]:
    count = 0
    summary: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        summary.append(sentence)
        count += len(words)
        if count > max_tokens:
            overflow = count - max_tokens
            keep = len(words) - overflow
            if keep > 0:
                summary[-1] = " ".join(words[:keep])
            else:
                summary.pop()
            break
    return summary


def load_ranked_lines(path: Path) -> dict[tuple[str, str, str], list[dict]]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["aspect"], row["split"], str(row["entity_id"]))
            groups[key].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (r.get("rank", 10**9), r.get("sentence_index", 10**9)))
    return groups


def write_budget_outputs(groups: dict[tuple[str, str, str], list[dict]], budget: int) -> Path:
    run_id = f"sweep_hasos_m1_tok_{budget}"
    run_dir = OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    total_words = 0
    for (aspect, split, entity_id), rows in groups.items():
        sentences = [str(r["sentence"]) for r in rows if str(r.get("sentence", "")).strip()]
        summary = truncate_summary(sentences, budget)
        if not summary:
            continue
        aspect_dir = run_dir / aspect
        aspect_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(summary).strip()
        (aspect_dir / f"{split}_{entity_id}").write_text(text, encoding="utf-8")
        written += 1
        total_words += word_count(text)

    meta = {
        "run_id": run_id,
        "budget_words": budget,
        "source_lines": str(LINES_PATH.relative_to(REPO)),
        "groups_written": written,
        "avg_words_per_group": round(total_words / written, 3) if written else 0.0,
        "note": "Replay of M1 extractive truncation from stored B=120 ranked lines.",
    }
    (REPORT_ROOT / f"{run_id}_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def score_budget(run_dir: Path, budget: int) -> dict:
    out = REPORT_ROOT / f"rouge_m1_hasos_tok_{budget}.json"
    python_exe = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    cmd = [
        python_exe,
        str(REPO / "scripts" / "score_rouge_compare.py"),
        "--method",
        "m1_extractive",
        "--run_dir",
        str(run_dir.relative_to(REPO)),
        "--out",
        str(out.relative_to(REPO)),
    ]
    subprocess.run(cmd, cwd=str(REPO), check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def macro_all(scores: dict) -> dict:
    macro = scores["by_split"]["all"]["MACRO"]
    return {
        "rouge1": macro["rouge1"],
        "rouge2": macro["rouge2"],
        "rougeL": macro["rougeL"],
    }


def write_sweep_report(rows: list[dict], budgets: list[int]) -> None:
    rows = sorted(rows, key=lambda r: r["B"])
    best = max(rows, key=lambda r: (r["rouge1"], r["rougeL"], r["rouge2"]))
    official_m1 = json.loads((REPO / "reports" / "rouge_m1_hasos.json").read_text(encoding="utf-8"))
    official_macro = macro_all(official_m1)
    payload = {
        "source": str(LINES_PATH.relative_to(REPO)),
        "budgets": budgets,
        "rows": rows,
        "best_by_rouge1": best,
        "official_m1_baseline": {
            "run_id": "space_hasos_full_e20",
            "B": 40,
            **official_macro,
        },
        "limitation": (
            "Replay rows use the space_hasos_threshold_full B=120 sentence log. "
            "They are a budget sensitivity check, not a replacement for the "
            "official M1 baseline scored from space_hasos_full_e20."
        ),
    }
    (REPORT_ROOT / "m1_token_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        "# M1 Token Budget Sweep",
        "",
        "M1 is extractive, so `B` is a word-level truncation budget, not a generative `max_new_tokens` value.",
        "The official paper baseline is `space_hasos_full_e20` with `B=40`.",
        "The table below is a post-hoc replay from `outputs/space_hasos_threshold_full_lines.jsonl`, useful as a sensitivity check for `B <= 120`.",
        "",
        "| B words | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        marker = " **best replay**" if row["B"] == best["B"] else ""
        md.append(
            f"| {row['B']}{marker} | {row['rouge1']:.4f} | "
            f"{row['rouge2']:.4f} | {row['rougeL']:.4f} |"
        )
    md.extend(
        [
            "",
            f"Best replay by ROUGE-1: `B={best['B']}`.",
            f"Official M1 baseline (`space_hasos_full_e20`, `B=40`): ROUGE-1={official_macro['rouge1']:.4f}, ROUGE-2={official_macro['rouge2']:.4f}, ROUGE-L={official_macro['rougeL']:.4f}.",
            "",
            "Budgets above 120 require rerunning SemAE or exporting the full ranked sentence list before truncation.",
        ]
    )
    (REPORT_ROOT / "m1_token_sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        default="40,64,80,96,120",
        help="Comma-separated M1 word budgets. Only B<=120 is replayable from the stored run.",
    )
    parser.add_argument(
        "--collect-existing",
        action="store_true",
        help="Build m1_token_sweep.* from existing rouge_m1_hasos_tok_<B>.json files without rescoring.",
    )
    args = parser.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if any(b <= 0 for b in budgets):
        raise SystemExit("budgets must be positive")
    if any(b > 120 for b in budgets):
        raise SystemExit("This replay sweep supports only budgets <= 120.")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.collect_existing:
        rows = []
        for budget in budgets:
            path = REPORT_ROOT / f"rouge_m1_hasos_tok_{budget}.json"
            if not path.exists():
                raise SystemExit(f"missing existing score file: {path}")
            rows.append({"B": budget, **macro_all(json.loads(path.read_text(encoding="utf-8")))})
        write_sweep_report(rows, budgets)
        return

    groups = load_ranked_lines(LINES_PATH)

    rows = []
    for budget in budgets:
        run_dir = write_budget_outputs(groups, budget)
        scores = score_budget(run_dir, budget)
        macro = macro_all(scores)
        rows.append({"B": budget, **macro})

    write_sweep_report(rows, budgets)


if __name__ == "__main__":
    main()
