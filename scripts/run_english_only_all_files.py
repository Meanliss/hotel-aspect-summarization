#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from log_10_hotels_pipeline import (
    build_step_evidence,
    choose_top_hotels,
    read_reviews_for_hotels,
    render_markdown,
)
from score_hotel_file import (
    ENGLISH_DETECTOR_RULE,
    REPO_ROOT,
    WORKSPACE_ROOT,
    english_detection_details,
    load_taxonomy,
    read_reviews,
    repair_mojibake,
    score_reviews,
    split_sentences,
    write_outputs,
)


INPUT_FILES = ["hotel_review1.csv", "hotel_review2.csv", "hotel_review3.csv"]


def entity_from_ref(ref_id):
    parts = (ref_id or "").rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return ref_id or "unknown_entity"


def collect_language_stats(input_csv, max_rows=None):
    stats = {
        "input_csv": os.path.abspath(input_csv),
        "detector_rule": ENGLISH_DETECTOR_RULE,
        "total_rows": 0,
        "nonempty_reviews": 0,
        "english_reviews": 0,
        "english_sentences": 0,
        "rejected_reviews": 0,
        "reject_reasons": defaultdict(int),
        "top_english_hotels": [],
        "sample_kept": [],
        "sample_rejected": [],
    }
    hotel_counts = Counter()
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            stats["total_rows"] += 1
            review = repair_mojibake((row.get("review") or "").strip())
            if not review:
                continue
            stats["nonempty_reviews"] += 1
            details = english_detection_details(review)
            if details["is_english"]:
                stats["english_reviews"] += 1
                hotel_id = entity_from_ref(row.get("ref_id"))
                hotel_counts[hotel_id] += 1
                kept_sentences = [
                    sentence for sentence in split_sentences(review)
                    if english_detection_details(sentence)["is_english"]
                ]
                stats["english_sentences"] += len(kept_sentences)
                if len(stats["sample_kept"]) < 3:
                    stats["sample_kept"].append({
                        "ref_id": row.get("ref_id"),
                        "hotel_id": hotel_id,
                        "reason": details["reason"],
                        "ascii_letter_ratio": details["ascii_letter_ratio"],
                        "marker_hits": details["marker_hits"],
                        "hotel_hits": details["hotel_hits"],
                        "review": review,
                    })
            else:
                stats["rejected_reviews"] += 1
                stats["reject_reasons"][details["reason"]] += 1
                if len(stats["sample_rejected"]) < 3:
                    stats["sample_rejected"].append({
                        "ref_id": row.get("ref_id"),
                        "reason": details["reason"],
                        "ascii_letter_ratio": details["ascii_letter_ratio"],
                        "marker_hits": details["marker_hits"],
                        "hotel_hits": details["hotel_hits"],
                        "review": review,
                    })
    stats["english_ratio"] = (
        stats["english_reviews"] / float(stats["nonempty_reviews"])
        if stats["nonempty_reviews"] else 0.0
    )
    stats["reject_reasons"] = dict(stats["reject_reasons"])
    stats["top_english_hotels"] = [
        {"hotel_id": hotel_id, "english_reviews": count}
        for hotel_id, count in hotel_counts.most_common(20)
    ]
    return stats


def collect_english_reviews_and_stats(input_csv, max_rows=None):
    stats = {
        "input_csv": os.path.abspath(input_csv),
        "detector_rule": ENGLISH_DETECTOR_RULE,
        "total_rows": 0,
        "nonempty_reviews": 0,
        "english_reviews": 0,
        "english_sentences": 0,
        "rejected_reviews": 0,
        "reject_reasons": defaultdict(int),
        "top_english_hotels": [],
        "sample_kept": [],
        "sample_rejected": [],
    }
    hotel_counts = Counter()
    reviews = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            stats["total_rows"] += 1
            review_text = repair_mojibake((row.get("review") or "").strip())
            if not review_text:
                continue
            stats["nonempty_reviews"] += 1
            details = english_detection_details(review_text)
            if not details["is_english"]:
                stats["rejected_reviews"] += 1
                stats["reject_reasons"][details["reason"]] += 1
                if len(stats["sample_rejected"]) < 3:
                    stats["sample_rejected"].append({
                        "ref_id": row.get("ref_id"),
                        "reason": details["reason"],
                        "ascii_letter_ratio": details["ascii_letter_ratio"],
                        "marker_hits": details["marker_hits"],
                        "hotel_hits": details["hotel_hits"],
                        "review": review_text,
                    })
                continue

            sentences = [
                sentence for sentence in split_sentences(review_text)
                if english_detection_details(sentence)["is_english"]
            ]
            if not sentences:
                stats["rejected_reviews"] += 1
                stats["reject_reasons"]["no_english_sentences"] += 1
                continue

            ref_id = row.get("ref_id") or "review_{0}".format(row_idx)
            hotel_id = entity_from_ref(ref_id)
            stats["english_reviews"] += 1
            stats["english_sentences"] += len(sentences)
            hotel_counts[hotel_id] += 1
            review = {
                "review_id": ref_id,
                "entity_id": hotel_id,
                "sentences": sentences,
                "review": review_text,
            }
            reviews.append(review)
            if len(stats["sample_kept"]) < 3:
                stats["sample_kept"].append({
                    "ref_id": ref_id,
                    "hotel_id": hotel_id,
                    "reason": details["reason"],
                    "ascii_letter_ratio": details["ascii_letter_ratio"],
                    "marker_hits": details["marker_hits"],
                    "hotel_hits": details["hotel_hits"],
                    "review": review_text,
                })

    stats["english_ratio"] = (
        stats["english_reviews"] / float(stats["nonempty_reviews"])
        if stats["nonempty_reviews"] else 0.0
    )
    stats["reject_reasons"] = dict(stats["reject_reasons"])
    stats["top_english_hotels"] = [
        {"hotel_id": hotel_id, "english_reviews": count}
        for hotel_id, count in hotel_counts.most_common(20)
    ]
    return stats, reviews


def choose_top_hotels_from_reviews(reviews, limit):
    counts = Counter()
    first_review = {}
    for review in reviews:
        hotel_id = review["entity_id"]
        counts[hotel_id] += 1
        first_review.setdefault(hotel_id, review["review"])
    return [
        {
            "hotel_id": hotel_id,
            "review_count": review_count,
            "first_review": first_review.get(hotel_id, ""),
        }
        for hotel_id, review_count in counts.most_common(limit)
    ]


def group_reviews_by_hotels(reviews, hotel_ids):
    hotel_ids = set(hotel_ids)
    grouped = defaultdict(list)
    for review in reviews:
        if review["entity_id"] in hotel_ids:
            grouped[review["entity_id"]].append(review)
    return grouped


def write_stats(path, stats):
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(stats, fout, ensure_ascii=False, indent=2)
        fout.write("\n")


def write_file_readme(path, file_name, stats, overall, top_rows):
    lines = [
        "# {0} English-Only Results".format(file_name),
        "",
        "Language detector:",
        "",
        "```text",
        ENGLISH_DETECTOR_RULE,
        "```",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| Total rows | {0:,} |".format(stats["total_rows"]),
        "| Non-empty reviews | {0:,} |".format(stats["nonempty_reviews"]),
        "| English reviews kept | {0:,} |".format(stats["english_reviews"]),
        "| English ratio | {0:.2%} |".format(stats["english_ratio"]),
        "| English sentences kept | {0:,} |".format(stats["english_sentences"]),
        "| Matched aspects | {0}/{1} |".format(overall["matched_aspects"], overall["aspect_count"]),
        "| ASC | {0:.4f} |".format(overall["aspect_summary_cover"]),
        "| Macro CEC | {0:.4f} |".format(overall["macro_cec"]),
        "| Weighted CEC | {0:.4f} |".format(overall["weighted_cec"]),
        "",
        "## Top Aspects",
        "",
        "| Aspect | Unique opinions | Weight | CEC | ASC contribution |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows:
        lines.append(
            "| `{0}` | {1:,} | {2:.4f} | {3:.4f} | {4:.4f} |".format(
                row["aspect"],
                row["unique_opinions"],
                row["cluster_weight"],
                row["cec"],
                row["asc_contribution"],
            )
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `filtered_english_stats.json`",
        "- `file_scores.json`",
        "- `file_scores.csv`",
        "- `file_summaries.txt`",
        "- `top10_hotels_pipeline_log.md`",
        "- `top10_hotels_pipeline_log.json`",
        "",
    ])
    with open(path, "w", encoding="utf-8-sig") as fout:
        fout.write("\n".join(lines))


def write_summary(outdir, file_summaries):
    json_path = os.path.join(outdir, "summary_all_files.json")
    md_path = os.path.join(outdir, "summary_all_files.md")
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(file_summaries, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    lines = [
        "# HASOS English-Only Summary",
        "",
        "This folder contains English-only HASOS CEC/ASC scoring for all three hotel review CSV files.",
        "",
        "Language detector:",
        "",
        "```text",
        ENGLISH_DETECTOR_RULE,
        "```",
        "",
        "ROUGE status: not available because the current data has reviews only and no human gold/reference summaries.",
        "",
        "| File | Total rows | English reviews | English ratio | English sentences | Matched aspects | ASC | Macro CEC | Weighted CEC | Top aspects |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in file_summaries:
        overall = item["overall"]
        stats = item["stats"]
        top_aspects = ", ".join(row["aspect"] for row in item["top_aspects"][:5])
        lines.append(
            "| `{0}` | {1:,} | {2:,} | {3:.2%} | {4:,} | {5}/{6} | {7:.4f} | {8:.4f} | {9:.4f} | {10} |".format(
                item["file"],
                stats["total_rows"],
                stats["english_reviews"],
                stats["english_ratio"],
                stats["english_sentences"],
                overall["matched_aspects"],
                overall["aspect_count"],
                overall["aspect_summary_cover"],
                overall["macro_cec"],
                overall["weighted_cec"],
                top_aspects,
            )
        )
    lines.extend([
        "",
        "## Per-file folders",
        "",
    ])
    for item in file_summaries:
        lines.append("- `{0}/`".format(os.path.splitext(item["file"])[0]))
    with open(md_path, "w", encoding="utf-8-sig") as fout:
        fout.write("\n".join(lines) + "\n")
    return md_path, json_path


def run_file(input_csv, taxonomy, outdir, max_rows=None, max_evidence_per_aspect=5000):
    file_name = os.path.basename(input_csv)
    file_stem = os.path.splitext(file_name)[0]
    file_outdir = os.path.join(outdir, file_stem)
    os.makedirs(file_outdir, exist_ok=True)

    stats, reviews = collect_english_reviews_and_stats(input_csv, max_rows=max_rows)
    write_stats(os.path.join(file_outdir, "filtered_english_stats.json"), stats)

    overall, rows = score_reviews(
        reviews,
        taxonomy,
        max_summary_sentences=3,
        max_evidence_per_aspect=max_evidence_per_aspect,
        include_vietnamese_aliases=False,
        language="english",
    )
    write_outputs(file_outdir, input_csv, overall, rows, output_prefix="file")

    selected_hotels = choose_top_hotels_from_reviews(reviews, 10)
    reviews_by_hotel = group_reviews_by_hotels(
        reviews,
        [hotel["hotel_id"] for hotel in selected_hotels],
    )
    top10_results = {}
    for hotel in selected_hotels:
        hotel_id = hotel["hotel_id"]
        hotel_overall, hotel_rows = score_reviews(
            reviews_by_hotel[hotel_id],
            taxonomy,
            max_summary_sentences=3,
            max_evidence_per_aspect=1500,
            include_vietnamese_aliases=False,
            language="english",
        )
        top10_results[hotel_id] = {"overall": hotel_overall, "aspects": hotel_rows}

    evidence = build_step_evidence(
        input_csv,
        selected_hotels,
        reviews_by_hotel,
        taxonomy,
        top10_results,
        language="english",
    )
    top10_json = os.path.join(file_outdir, "top10_hotels_pipeline_log.json")
    top10_md = os.path.join(file_outdir, "top10_hotels_pipeline_log.md")
    with open(top10_json, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "input_csv": os.path.abspath(input_csv),
                "language_filter": "english",
                "language_detector_rule": ENGLISH_DETECTOR_RULE,
                "selected_hotels": selected_hotels,
                "results": top10_results,
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )
        fout.write("\n")
    with open(top10_md, "w", encoding="utf-8-sig") as fout:
        fout.write(render_markdown(
            input_csv,
            selected_hotels,
            top10_results,
            evidence,
            reviews_by_hotel,
            taxonomy,
            language="english",
        ))

    top_rows = sorted(rows, key=lambda row: row["asc_contribution"], reverse=True)[:10]
    write_file_readme(
        os.path.join(file_outdir, "README.md"),
        file_name,
        stats,
        overall,
        top_rows,
    )
    return {
        "file": file_name,
        "folder": os.path.abspath(file_outdir),
        "stats": stats,
        "overall": overall,
        "top_aspects": top_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run English-only HASOS scoring for all hotel review CSV files."
    )
    parser.add_argument(
        "--input-dir",
        default=WORKSPACE_ROOT,
        help="Directory containing hotel_review1.csv, hotel_review2.csv, hotel_review3.csv.",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(REPO_ROOT, "outputs", "hasos_english_only"),
        help="Output directory.",
    )
    parser.add_argument(
        "--taxonomy",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.tsv"),
        help="HASOS taxonomy TSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test row limit per file.",
    )
    parser.add_argument(
        "--max-evidence-per-aspect",
        type=int,
        default=5000,
        help="Evidence cap per file/aspect for representative summaries.",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    os.makedirs(args.outdir, exist_ok=True)
    summaries = []
    for file_name in INPUT_FILES:
        input_csv = os.path.join(args.input_dir, file_name)
        print("running {0}".format(input_csv))
        summaries.append(run_file(
            input_csv,
            taxonomy,
            args.outdir,
            max_rows=args.max_rows,
            max_evidence_per_aspect=args.max_evidence_per_aspect,
        ))
    md_path, json_path = write_summary(args.outdir, summaries)
    readme_path = os.path.join(args.outdir, "README.md")
    with open(readme_path, "w", encoding="utf-8-sig") as fout:
        fout.write(
            "# HASOS English-Only Output\n\n"
            "Use `summary_all_files.md` for the cross-file summary. "
            "Each `hotel_review*/` folder contains file-level scores, "
            "filtered language stats, summaries, and top-10 hotel pipeline evidence.\n"
        )
    print("summary_md={0}".format(md_path))
    print("summary_json={0}".format(json_path))
    print("readme={0}".format(readme_path))


if __name__ == "__main__":
    main()
