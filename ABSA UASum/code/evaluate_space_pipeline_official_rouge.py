"""Evaluate hotel-pipeline SPACE outputs with the official SPACE ROUGE metrics.

This script is intentionally separate from the pipeline's internal
``*_final_summary_metrics.json`` files. Those internal metrics compare summaries
against source segments. The SPACE benchmark in ``results_for_codex.zip`` uses
human gold summaries and reports ROUGE-1/2/L F1 through pyrouge/ROUGE-1.5.5.

The hotel pipeline writes summaries in the hotel taxonomy
facility/amenity/service/experience/branding/loyalty, while SPACE gold uses six
flat aspects: building/cleanliness/food/location/rooms/service. The default
mapping below projects the hotel taxonomy back to SPACE so the outputs can be
scored on the same metric surface as the ZIP reports.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent
TESING_DIR = WORKSPACE / "tesing"
ROUGE_HOME = TESING_DIR / "downloads" / "rouge_setup" / "ROUGE-1.5.5"
SPACE_GOLD_DIR = TESING_DIR / "data" / "space" / "gold"
SPACE_SPLIT_SOURCE = TESING_DIR / "outputs" / "space_eval_e20"
SPACE_ASPECTS = ("building", "cleanliness", "food", "location", "rooms", "service")

DEFAULT_RESULTS_DIR = WORKSPACE / "results" / "space_pipeline" / "full_newprompt_specific_amenity_20260617"
DEFAULT_RUN_NAME = "space_full_newprompt_specific_amenity_20260617"

SPACE_TO_PIPELINE_ASPECTS = {
    "building": ("facility", "branding"),
    "cleanliness": ("facility",),
    "food": ("amenity",),
    "location": ("facility", "experience"),
    "rooms": ("facility", "amenity"),
    "service": ("service",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official ROUGE scoring for hotel-pipeline SPACE output.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--final-summary-csv", default="")
    parser.add_argument("--gold-dir", default=str(SPACE_GOLD_DIR))
    parser.add_argument("--split-source-dir", default=str(SPACE_SPLIT_SOURCE))
    parser.add_argument("--rouge-home", default=str(ROUGE_HOME))
    parser.add_argument("--out-prefix", default="")
    parser.add_argument(
        "--input-taxonomy",
        choices=("hotel", "space"),
        default="hotel",
        help="Use 'hotel' for facility/amenity/etc. rows that need projection, or 'space' for direct SPACE aspect rows.",
    )
    parser.add_argument(
        "--summary-fields",
        default="positive_summary,negative_summary,neutral_summary",
        help="Comma-separated final-summary columns to concatenate for each projected SPACE aspect.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace("\t", " ").replace("\r", " ")).strip()


def load_split_map(split_source_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for aspect in SPACE_ASPECTS:
        aspect_dir = split_source_dir / aspect
        if not aspect_dir.exists():
            continue
        for path in aspect_dir.iterdir():
            match = re.match(r"(dev|test|train)_(.+)$", path.name)
            if match:
                split_map[match.group(2)] = match.group(1)
    if not split_map:
        raise FileNotFoundError(f"No split files found under {split_source_dir}")
    return split_map


def load_gold_refs(gold_dir: Path) -> dict[str, dict[str, list[str]]]:
    refs: dict[str, dict[str, list[str]]] = {aspect: {} for aspect in SPACE_ASPECTS}
    for aspect in SPACE_ASPECTS:
        aspect_dir = gold_dir / aspect
        if not aspect_dir.exists():
            raise FileNotFoundError(aspect_dir)
        for path in sorted(aspect_dir.glob("*.txt")):
            match = re.match(r"(.+)_([0-9]+)\.txt$", path.name)
            if not match:
                continue
            entity_id = match.group(1)
            refs[aspect].setdefault(entity_id, []).append(path.read_text(encoding="utf-8", errors="replace"))
    return refs


def load_pipeline_systems(
    final_summary_csv: Path,
    summary_fields: list[str],
    input_taxonomy: str = "hotel",
) -> dict[str, dict[str, str]]:
    if not final_summary_csv.exists():
        raise FileNotFoundError(final_summary_csv)
    summary = pd.read_csv(final_summary_csv).fillna("")
    required = {"hotel_id", "aspect", *summary_fields}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Missing columns in {final_summary_csv}: {sorted(missing)}")

    by_entity_aspect: dict[tuple[str, str], str] = {}
    for _, row in summary.iterrows():
        hotel_id = str(row["hotel_id"])
        aspect = str(row["aspect"]).strip().lower()
        parts = [clean_text(row.get(field, "")) for field in summary_fields]
        text = clean_text(" ".join(part for part in parts if part))
        if text:
            by_entity_aspect[(hotel_id, aspect)] = text

    systems: dict[str, dict[str, str]] = {aspect: {} for aspect in SPACE_ASPECTS}
    if input_taxonomy == "space":
        for (entity_id, aspect), text in by_entity_aspect.items():
            if aspect in systems:
                systems[aspect][entity_id] = text
        return systems

    entity_ids = sorted({entity_id for entity_id, _aspect in by_entity_aspect})
    for entity_id in entity_ids:
        for space_aspect, pipeline_aspects in SPACE_TO_PIPELINE_ASPECTS.items():
            parts = [
                by_entity_aspect.get((entity_id, pipeline_aspect), "")
                for pipeline_aspect in pipeline_aspects
            ]
            systems[space_aspect][entity_id] = clean_text(" ".join(part for part in parts if part))
    return systems


def apply_rouge_patch(rouge_home: Path) -> None:
    sys.path.insert(0, str(TESING_DIR / "scripts"))
    import rouge_patch  # type: ignore  # noqa: PLC0415

    rouge_patch.apply(rouge_home=str(rouge_home))
    pyrouge_logger = logging.getLogger("global")
    pyrouge_logger.disabled = True


def parse_rouge(output: str) -> dict[str, float]:
    scores = {}
    for label, key in (("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL")):
        match = re.search(rf"{label} Average_F: ([0-9.]+)", output)
        if match:
            scores[key] = float(match.group(1))
    return scores


def run_rouge(pairs: list[tuple[str, list[str]]], rouge_home: Path) -> tuple[dict[str, float], int] | None:
    from pyrouge import Rouge155  # type: ignore  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="space_pipeline_rouge_") as tmp:
        system_dir = Path(tmp) / "system"
        model_dir = Path(tmp) / "model"
        system_dir.mkdir()
        model_dir.mkdir()

        written = 0
        for idx, (system_text, refs) in enumerate(pairs, start=1):
            if not system_text.strip() or not any(ref.strip() for ref in refs):
                continue
            (system_dir / f"text.{idx:03d}.txt").write_text(system_text, encoding="utf-8")
            for ref_idx, ref in enumerate(refs):
                letter = chr(ord("A") + ref_idx)
                (model_dir / f"text.{letter}.{idx:03d}.txt").write_text(clean_text(ref), encoding="utf-8")
            written += 1

        if written == 0:
            return None

        rouge = Rouge155(rouge_dir=str(rouge_home))
        rouge.system_dir = str(system_dir)
        rouge.model_dir = str(model_dir)
        rouge.system_filename_pattern = r"text.(\d+).txt"
        rouge.model_filename_pattern = "text.[A-Z].#ID#.txt"
        output = rouge.convert_and_evaluate()
        return parse_rouge(output), written


def evaluate(
    systems: dict[str, dict[str, str]],
    refs: dict[str, dict[str, list[str]]],
    split_map: dict[str, str],
    rouge_home: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    results: dict[str, dict[str, dict[str, float]]] = {}
    split_filters = {
        "dev": {"dev"},
        "test": {"test"},
        "all": {"dev", "test", "train"},
    }
    for split_name, allowed_splits in split_filters.items():
        split_result: dict[str, dict[str, float]] = {}
        for aspect in SPACE_ASPECTS:
            pairs = []
            for entity_id, ref_list in refs[aspect].items():
                if split_map.get(entity_id) not in allowed_splits:
                    continue
                system_text = systems.get(aspect, {}).get(entity_id, "")
                if system_text:
                    pairs.append((system_text, ref_list))
            rouge_out = run_rouge(pairs, rouge_home)
            if rouge_out is None:
                continue
            scores, n = rouge_out
            scores["n"] = n
            split_result[aspect] = scores

        macro = {}
        for key in ("rouge1", "rouge2", "rougeL"):
            values = [score[key] for score in split_result.values() if key in score]
            macro[key] = sum(values) / len(values) if values else 0.0
        split_result["MACRO"] = macro
        results[split_name] = split_result
    return results


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def write_report(
    path: Path,
    results: dict[str, dict[str, dict[str, float]]],
    source_csv: Path,
    input_taxonomy: str,
) -> None:
    lines = [
        "# SPACE Official ROUGE Evaluation — hotel pipeline output",
        "",
        "Metrics: pyrouge / ROUGE-1.5.5 F1 against human SPACE gold summaries.",
        "",
        f"Source CSV: `{source_csv}`",
        "",
        "## Macro ROUGE F1",
        "",
        "| Split | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("dev", "test", "all"):
        macro = results.get(split, {}).get("MACRO", {})
        lines.append(f"| {split} | {fmt(macro.get('rouge1'))} | {fmt(macro.get('rouge2'))} | {fmt(macro.get('rougeL'))} |")

    for split in ("dev", "test", "all"):
        lines.extend(["", f"## {split} by aspect", "", "| Aspect | ROUGE-1 | ROUGE-2 | ROUGE-L | N |", "| --- | ---: | ---: | ---: | ---: |"])
        for aspect in SPACE_ASPECTS:
            row = results.get(split, {}).get(aspect, {})
            lines.append(
                f"| {aspect} | {fmt(row.get('rouge1'))} | {fmt(row.get('rouge2'))} | "
                f"{fmt(row.get('rougeL'))} | {row.get('n', '-')} |"
            )
    lines.extend(["", "## Notes", ""])
    if input_taxonomy == "space":
        lines.append("- Output rows use SPACE's six flat aspects directly; no hotel-taxonomy projection was applied.")
    else:
        lines.extend(
            [
                "- Output is projected from hotel-pipeline taxonomy to SPACE's six flat aspects.",
                "- Mapping: building=facility+branding; cleanliness=facility; food=amenity; location=facility+experience; rooms=facility+amenity; service=service.",
                "- This is comparable by metric surface to the ZIP reports, but taxonomy projection can penalize scores.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_name = args.run_name
    final_summary_csv = Path(args.final_summary_csv) if args.final_summary_csv else results_dir / f"{run_name}_final_summary.csv"
    out_prefix = args.out_prefix or f"{run_name}_space_official_rouge"
    out_json = results_dir / f"{out_prefix}.json"
    out_md = results_dir / f"{out_prefix}.md"

    summary_fields = [field.strip() for field in args.summary_fields.split(",") if field.strip()]
    rouge_home = Path(args.rouge_home)
    apply_rouge_patch(rouge_home)

    split_map = load_split_map(Path(args.split_source_dir))
    refs = load_gold_refs(Path(args.gold_dir))
    systems = load_pipeline_systems(final_summary_csv, summary_fields, input_taxonomy=args.input_taxonomy)
    results = evaluate(systems, refs, split_map, rouge_home)

    payload = {
        "metric_type": "space_official_rouge_for_hotel_pipeline_output",
        "source_csv": str(final_summary_csv.resolve()),
        "gold_dir": str(Path(args.gold_dir).resolve()),
        "rouge_home": str(rouge_home.resolve()),
        "summary_fields": summary_fields,
        "input_taxonomy": args.input_taxonomy,
        "space_to_pipeline_aspects": SPACE_TO_PIPELINE_ASPECTS,
        "by_split": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_md, results, final_summary_csv, input_taxonomy=args.input_taxonomy)
    print(json.dumps({"json": str(out_json), "md": str(out_md), "macro_all": results["all"]["MACRO"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
