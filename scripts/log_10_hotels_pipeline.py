#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from score_hotel_file import (
    ENGLISH_DETECTOR_RULE,
    REPO_ROOT,
    WORKSPACE_ROOT,
    build_aspect_terms,
    build_term_matcher,
    english_detection_details,
    load_taxonomy,
    match_aspects,
    normalize,
    read_reviews,
    repair_mojibake,
    score_reviews,
    split_sentences,
)


def entity_from_ref(ref_id):
    parts = (ref_id or "").rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return ref_id or "unknown_entity"


def choose_top_hotels(input_csv, limit, language="all"):
    counts = Counter()
    first_review = {}
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            review = repair_mojibake((row.get("review") or "").strip())
            if not review:
                continue
            if language == "english" and not english_detection_details(review)["is_english"]:
                continue
            hotel_id = entity_from_ref(row.get("ref_id"))
            counts[hotel_id] += 1
            first_review.setdefault(hotel_id, review)
    return [
        {
            "hotel_id": hotel_id,
            "review_count": review_count,
            "first_review": first_review.get(hotel_id, ""),
        }
        for hotel_id, review_count in counts.most_common(limit)
    ]


def read_reviews_for_hotels(input_csv, hotel_ids, language="all"):
    hotel_ids = set(hotel_ids)
    reviews_by_hotel = defaultdict(list)
    for review in read_reviews(input_csv, language=language):
        if review["entity_id"] in hotel_ids:
            reviews_by_hotel[review["entity_id"]].append(review)
    return reviews_by_hotel


def count_csv_rows(input_csv):
    total = 0
    nonempty = 0
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            total += 1
            if (row.get("review") or "").strip():
                nonempty += 1
    return total, nonempty


def short_sentence(sentence, limit=240):
    sentence = " ".join(sentence.split())
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 3].rstrip() + "..."


def build_step_evidence(input_csv, selected_hotels, reviews_by_hotel, taxonomy, results,
                        language="all"):
    total_rows, nonempty_rows = count_csv_rows(input_csv)
    first_hotel = selected_hotels[0]
    first_review = reviews_by_hotel[first_hotel["hotel_id"]][0]
    first_sentences = split_sentences(first_review["review"])
    first_sentence = first_sentences[0] if first_sentences else first_review["review"]

    aspect_terms = build_aspect_terms(
        taxonomy,
        include_vietnamese_aliases=(language != "english"))
    matcher = build_term_matcher(aspect_terms)
    matched_sentence = None
    matched_aspects = []
    for review in reviews_by_hotel[first_hotel["hotel_id"]]:
        for sentence in review["sentences"]:
            matches = match_aspects(sentence, matcher)
            if matches:
                matched_sentence = sentence
                matched_aspects = sorted(matches)
                break
        if matched_sentence:
            break

    top_aspect = sorted(
        results[first_hotel["hotel_id"]]["aspects"],
        key=lambda row: row["asc_contribution"],
        reverse=True,
    )[0]
    denominator = sum(math.log1p(row["unique_opinions"])
                      for row in results[first_hotel["hotel_id"]]["aspects"])

    return {
        "total_rows": total_rows,
        "nonempty_rows": nonempty_rows,
        "first_hotel": first_hotel,
        "first_review": first_review,
        "first_sentences": first_sentences,
        "first_sentence": first_sentence,
        "language_detection": english_detection_details(first_review["review"]),
        "sentence_language_detection": english_detection_details(first_sentence),
        "normalized_sentence": normalize(first_sentence),
        "matched_sentence": matched_sentence or "",
        "matched_aspects": matched_aspects,
        "top_aspect": top_aspect,
        "weight_denominator": denominator,
        "overall": results[first_hotel["hotel_id"]]["overall"],
    }


def find_first_match_for_aspect(reviews, aspect, matcher):
    for review in reviews:
        for sentence in review["sentences"]:
            matches = sorted(match_aspects(sentence, matcher))
            if aspect in matches:
                return {
                    "review_id": review["review_id"],
                    "sentence": sentence,
                    "matches": matches,
                    "normalized": normalize(sentence),
                }
    return {
        "review_id": "",
        "sentence": "",
        "matches": [],
        "normalized": "",
    }


def render_final_output(top_aspects):
    lines = ["Final output for this hotel:", ""]
    for row in top_aspects[:3]:
        summary = row["summary"][0] if row["summary"] else ""
        lines.append(
            "- `{0}` summary: {1}".format(row["aspect"],
                                         short_sentence(summary, 220)))
    lines.append("")
    return lines


def render_markdown(input_csv, selected_hotels, results, evidence, reviews_by_hotel,
                    taxonomy, language="all"):
    input_name = os.path.splitext(os.path.basename(input_csv))[0]
    language_label = "English-only" if language == "english" else "Mixed-language"
    aspect_terms = build_aspect_terms(
        taxonomy,
        include_vietnamese_aliases=(language != "english"))
    matcher = build_term_matcher(aspect_terms)
    lines = [
        "# {0}.csv - 10 Hotel {1} Pipeline Log".format(input_name, language_label),
        "",
        "Input file:",
        "",
        "```text",
        os.path.abspath(input_csv),
        "```",
        "",
        "Hotel selection: top 10 `hotel_id` groups by number of {0} reviews. ".format(
            language_label.lower()) +
        "`hotel_id` is derived from `ref_id` by removing the final numeric suffix.",
        "",
        "## Pipeline",
        "",
        "1. Load `{0}.csv` with UTF-8 encoding.".format(input_name),
        "2. Group rows into hotels using `ref_id`: `booking_02157_42` becomes `booking_02157`.",
        "3. Filter to English-like reviews and English-like sentences." if language == "english"
        else "3. Split each review into sentences.",
        "4. Split each kept review into sentences." if language == "english"
        else "4. Normalize text: lowercase, remove Vietnamese accents, collapse punctuation/spaces.",
        "5. Normalize text: lowercase, remove accents, collapse punctuation/spaces." if language == "english"
        else "5. Match each sentence to HASOS aspect codes using the taxonomy TSV plus Vietnamese aliases.",
        "6. Match each sentence to HASOS aspect codes using English taxonomy keywords only." if language == "english"
        else "6. Deduplicate matched sentences per aspect to estimate `unique_opinions`.",
        "7. Deduplicate matched sentences per aspect to estimate `unique_opinions`." if language == "english"
        else "7. Compute cluster/aspect weight as `log(1 + unique_opinions)` normalized across aspects.",
        "8. Compute cluster/aspect weight as `log(1 + unique_opinions)` normalized across aspects." if language == "english"
        else "8. Select representative summary sentences per aspect with TF-IDF centroid similarity.",
        "9. Select representative summary sentences per aspect with TF-IDF centroid similarity." if language == "english"
        else "",
        "10. Compute image-note metrics:" if language == "english"
        else "9. Compute image-note metrics:",
        "   - `CEC`: average representative evidence coverage for the aspect cluster.",
        "   - `ASC`: weighted aspect summary cover contribution.",
        "11. Write this Markdown log and the detailed JSON artifact." if language == "english"
        else "10. Write this Markdown log and the detailed JSON artifact.",
        "",
        "Language detector: {0}".format(ENGLISH_DETECTOR_RULE)
        if language == "english" else "Language detector: not used.",
        "",
        "Paper note: SemAE's official paper metric path is ROUGE against human gold summaries. "
        "The current CSV has reviews only, so this log reports the CEC/ASC scoring from the provided image.",
        "",
        "## Step Evidence Log",
        "",
        "| Step | What happened | Evidence from this run |",
        "| ---: | --- | --- |",
        "| 1 | Loaded CSV | `{0:,}` total rows, `{1:,}` non-empty reviews from `{2}`. |".format(
            evidence["total_rows"],
            evidence["nonempty_rows"],
            os.path.abspath(input_csv),
        ),
        "| 2 | Grouped rows into hotels | Example: `{0}` -> `{1}`. Top hotel `{1}` has `{2:,}` reviews. |".format(
            evidence["first_review"]["review_id"],
            evidence["first_hotel"]["hotel_id"],
            evidence["first_hotel"]["review_count"],
        ),
        "| 3 | Split reviews into sentences | First sampled review produced `{0}` sentence(s). First sentence: `{1}` |".format(
            len(evidence["first_sentences"]),
            short_sentence(evidence["first_sentence"], 160),
        ),
        "| 3a | Language filter | Review detection `{0}` with ASCII ratio `{1:.3f}`, marker hits `{2}`, hotel hits `{3}`. Sentence detection `{4}`. |".format(
            evidence["language_detection"]["reason"],
            evidence["language_detection"]["ascii_letter_ratio"],
            evidence["language_detection"]["marker_hits"],
            evidence["language_detection"]["hotel_hits"],
            evidence["sentence_language_detection"]["reason"],
        ) if language == "english" else "| 3a | Language filter | Not used for mixed-language run. |",
        "| 4 | Normalized text | Normalized first sentence: `{0}` |".format(
            short_sentence(evidence["normalized_sentence"], 160),
        ),
        "| 5 | Matched HASOS aspects | Sample sentence `{0}` matched `{1}`. |".format(
            short_sentence(evidence["matched_sentence"], 140),
            ", ".join("`{0}`".format(x) for x in evidence["matched_aspects"]),
        ),
        "| 6 | Deduplicated unique opinions | Top aspect `{0}` has `{1:,}` unique opinions and `{2:,}` matched sentences. |".format(
            evidence["top_aspect"]["aspect"],
            evidence["top_aspect"]["unique_opinions"],
            evidence["top_aspect"]["matched_sentences"],
        ),
        "| 7 | Computed cluster weight | For `{0}`: `log(1 + {1}) / {2:.4f} = {3:.4f}`. |".format(
            evidence["top_aspect"]["aspect"],
            evidence["top_aspect"]["unique_opinions"],
            evidence["weight_denominator"],
            evidence["top_aspect"]["cluster_weight"],
        ),
        "| 8 | Selected representative summary | `{0}` summary example: `{1}` |".format(
            evidence["top_aspect"]["aspect"],
            short_sentence(evidence["top_aspect"]["summary"][0], 180)
            if evidence["top_aspect"]["summary"] else "",
        ),
        "| 9 | Computed CEC/ASC metrics | First hotel ASC `{0:.4f}`, macro CEC `{1:.4f}`, weighted CEC `{2:.4f}`. Top aspect CEC `{3:.4f}`, ASC contribution `{4:.4f}`. |".format(
            evidence["overall"]["aspect_summary_cover"],
            evidence["overall"]["macro_cec"],
            evidence["overall"]["weighted_cec"],
            evidence["top_aspect"]["cec"],
            evidence["top_aspect"]["asc_contribution"],
        ),
        "| 10 | Wrote artifacts | Markdown log and detailed JSON were written under `outputs/hasos_file_trials/`. |",
        "",
        "## Selected Hotels",
        "",
        "| # | Hotel ID | Reviews | Sentences | Matched Aspects | ASC | Macro CEC | Weighted CEC |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for idx, hotel in enumerate(selected_hotels, 1):
        overall = results[hotel["hotel_id"]]["overall"]
        lines.append(
            "| {idx} | `{hotel_id}` | {reviews:,} | {sentences:,} | {matched}/{total} | {asc:.4f} | {macro:.4f} | {weighted:.4f} |".format(
                idx=idx,
                hotel_id=hotel["hotel_id"],
                reviews=overall["review_count"],
                sentences=overall["sentence_count"],
                matched=overall["matched_aspects"],
                total=overall["aspect_count"],
                asc=overall["aspect_summary_cover"],
                macro=overall["macro_cec"],
                weighted=overall["weighted_cec"],
            )
        )

    lines.extend(["", "## Per-Hotel Highlights", ""])

    for idx, hotel in enumerate(selected_hotels, 1):
        result = results[hotel["hotel_id"]]
        overall = result["overall"]
        top_aspects = sorted(
            result["aspects"],
            key=lambda row: row["asc_contribution"],
            reverse=True,
        )[:5]
        reviews = reviews_by_hotel[hotel["hotel_id"]]
        first_review = reviews[0]
        first_sentences = first_review["sentences"]
        first_sentence = first_sentences[0] if first_sentences else first_review["review"]
        top_aspect = top_aspects[0]
        top_evidence = find_first_match_for_aspect(
            reviews, top_aspect["aspect"], matcher)
        denominator = sum(math.log1p(row["unique_opinions"])
                          for row in result["aspects"])
        lines.extend(
            [
                "### {0}. `{1}`".format(idx, hotel["hotel_id"]),
                "",
                "- Reviews: {0:,}".format(overall["review_count"]),
                "- Sentences: {0:,}".format(overall["sentence_count"]),
                "- ASC: {0:.4f}".format(overall["aspect_summary_cover"]),
                "- Macro CEC: {0:.4f}".format(overall["macro_cec"]),
                "- Weighted CEC: {0:.4f}".format(overall["weighted_cec"]),
                "- First review sample: {0}".format(short_sentence(hotel["first_review"])),
                "",
            ]
        )
        lines.extend(render_final_output(top_aspects))
        lines.extend(
            [
                "Pipeline evidence for this hotel:",
                "",
                "| Step | Evidence |",
                "| ---: | --- |",
                "| 1. Input | `{0:,}` review records for `{1}`; `{2:,}` sentence units after sentence split. |".format(
                    overall["review_count"],
                    hotel["hotel_id"],
                    overall["sentence_count"],
                ),
                "| 2. Grouping | Sample row `{0}` was grouped into hotel `{1}` by removing the final numeric suffix. |".format(
                    first_review["review_id"],
                    hotel["hotel_id"],
                ),
                "| 3. Sentence split | Sample review `{0}` split into `{1}` sentence(s). First sentence: `{2}` |".format(
                    first_review["review_id"],
                    len(first_sentences),
                    short_sentence(first_sentence, 160),
                ),
                "| 4. Normalization | Normalized first sentence: `{0}` |".format(
                    short_sentence(normalize(first_sentence), 160),
                ),
                "| 5. Aspect match | Evidence sentence from `{0}`: `{1}` matched `{2}`. |".format(
                    top_evidence["review_id"],
                    short_sentence(top_evidence["sentence"], 170),
                    ", ".join("`{0}`".format(item)
                              for item in top_evidence["matches"]),
                ),
                "| 6. Unique opinions | Top aspect `{0}` has `{1:,}` unique opinions from `{2:,}` matched sentences. |".format(
                    top_aspect["aspect"],
                    top_aspect["unique_opinions"],
                    top_aspect["matched_sentences"],
                ),
                "| 7. Weight | `{0}` weight = `log(1 + {1}) / {2:.4f} = {3:.4f}`. |".format(
                    top_aspect["aspect"],
                    top_aspect["unique_opinions"],
                    denominator,
                    top_aspect["cluster_weight"],
                ),
                "| 8. Representative summary | `{0}` summary chosen by TF-IDF centroid similarity: `{1}` |".format(
                    top_aspect["aspect"],
                    short_sentence(top_aspect["summary"][0], 180)
                    if top_aspect["summary"] else "",
                ),
                "| 9. Final score | Hotel ASC `{0:.4f}`, macro CEC `{1:.4f}`, weighted CEC `{2:.4f}`; top aspect ASC contribution `{3:.4f}`. |".format(
                    overall["aspect_summary_cover"],
                    overall["macro_cec"],
                    overall["weighted_cec"],
                    top_aspect["asc_contribution"],
                ),
                "",
                "Top aspect table:",
                "",
                "| Aspect | Unique opinions | Weight | CEC | ASC contribution |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in top_aspects:
            lines.append(
                "| `{0}` | {1:,} | {2:.4f} | {3:.4f} | {4:.4f} |".format(
                    row["aspect"],
                    row["unique_opinions"],
                    row["cluster_weight"],
                    row["cec"],
                    row["asc_contribution"],
                )
            )
        lines.extend(["", "Representative summaries used in final output:", ""])
        for row in top_aspects[:3]:
            if not row["summary"]:
                continue
            lines.append("- `{0}`: {1}".format(row["aspect"], short_sentence(row["summary"][0])))
        lines.append("")

    lines.extend(
        [
            "## Re-run Command",
            "",
            "```powershell",
            "$env:PYTHONIOENCODING='utf-8'",
            "python .\\log_10_hotels_pipeline.py --input-csv ..\\..\\hotel_review1.csv --limit 10",
            "```",
            "",
            "Run from:",
            "",
            "```text",
            os.path.join(REPO_ROOT, "scripts"),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Create a Markdown pipeline log for roughly 10 hotels from hotel_review1.csv."
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join(WORKSPACE_ROOT, "hotel_review1.csv"),
        help="CSV containing ref_id and review columns.",
    )
    parser.add_argument(
        "--taxonomy",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.tsv"),
        help="HASOS taxonomy TSV.",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(REPO_ROOT, "outputs", "hasos_file_trials"),
        help="Output directory.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of hotels.")
    parser.add_argument(
        "--max-evidence-per-aspect",
        type=int,
        default=1500,
        help="Evidence cap per hotel/aspect for representative summaries.",
    )
    parser.add_argument(
        "--language",
        choices=["all", "english"],
        default="all",
        help="Optional language filter applied before hotel selection and scoring.",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    selected_hotels = choose_top_hotels(args.input_csv,
                                        args.limit,
                                        language=args.language)
    reviews_by_hotel = read_reviews_for_hotels(
        args.input_csv,
        [hotel["hotel_id"] for hotel in selected_hotels],
        language=args.language,
    )

    results = {}
    for hotel in selected_hotels:
        hotel_id = hotel["hotel_id"]
        overall, rows = score_reviews(
            reviews_by_hotel[hotel_id],
            taxonomy,
            max_summary_sentences=3,
            max_evidence_per_aspect=args.max_evidence_per_aspect,
            include_vietnamese_aliases=(args.language != "english"),
            language=args.language,
        )
        results[hotel_id] = {"overall": overall, "aspects": rows}

    evidence = build_step_evidence(
        args.input_csv,
        selected_hotels,
        reviews_by_hotel,
        taxonomy,
        results,
        language=args.language)

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "top10_hotels_pipeline_log.json")
    md_path = os.path.join(args.outdir, "top10_hotels_pipeline_log.md")

    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "input_csv": os.path.abspath(args.input_csv),
                "language_filter": args.language,
                "language_detector_rule": ENGLISH_DETECTOR_RULE
                if args.language == "english" else None,
                "selected_hotels": selected_hotels,
                "results": results,
            },
            fout,
            ensure_ascii=False,
            indent=2,
        )
        fout.write("\n")

    with open(md_path, "w", encoding="utf-8-sig") as fout:
        fout.write(render_markdown(args.input_csv,
                                   selected_hotels,
                                   results,
                                   evidence,
                                   reviews_by_hotel,
                                   taxonomy,
                                   language=args.language))

    print("markdown={0}".format(md_path))
    print("json={0}".format(json_path))
    for hotel in selected_hotels:
        overall = results[hotel["hotel_id"]]["overall"]
        print(
            "{0}\treviews={1}\tASC={2:.4f}\tweighted_CEC={3:.4f}".format(
                hotel["hotel_id"],
                overall["review_count"],
                overall["aspect_summary_cover"],
                overall["weighted_cec"],
            )
        )


if __name__ == "__main__":
    main()
