#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)


def split_terms(value):
    return [term.strip() for term in value.split(",") if term.strip()]


def load_taxonomy(path):
    with open(path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        return [row for row in reader if row.get("CODE")]


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)
        fout.write("\n")


def write_seed_files(taxonomy, seeds_dir):
    os.makedirs(seeds_dir, exist_ok=True)
    for row in taxonomy:
        terms = split_terms(row["ASPECT_KEYWORDS"])
        scale_terms = split_terms(row["MEASUREMENT_SCALE"].replace("&", ","))
        merged_terms = []
        for term in terms + scale_terms:
            normalized = term.lower()
            if normalized not in merged_terms:
                merged_terms.append(normalized)

        seed_path = os.path.join(seeds_dir, row["CODE"] + ".txt")
        with open(seed_path, "w", encoding="utf-8") as fout:
            for idx, term in enumerate(merged_terms):
                weight = max(0.001, 1.0 - (idx * 0.001))
                fout.write("{0:.3f} {1}\n".format(weight, term))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the HASOS hotel data and aspect seeds for SemAE."
    )
    parser.add_argument(
        "--input-json",
        default=os.path.join(WORKSPACE_ROOT, "space_summ_hasos.json"),
        help="Hotel review JSON in SemAE entity/review format.",
    )
    parser.add_argument(
        "--taxonomy",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.tsv"),
        help="TSV exported from the Google Sheet SPACE MAPPING tab.",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(REPO_ROOT, "data", "hasos", "hasos_summ.json"),
        help="Destination JSON path used by train.py and inference.py.",
    )
    parser.add_argument(
        "--seeds-dir",
        default=os.path.join(REPO_ROOT, "data", "seeds_hasos"),
        help="Destination directory for SemAE aspect seed files.",
    )
    parser.add_argument(
        "--taxonomy-json",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.json"),
        help="Machine-readable taxonomy metadata.",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    if not taxonomy:
        raise SystemExit("No taxonomy rows found in {0}".format(args.taxonomy))

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    shutil.copyfile(args.input_json, args.output_json)
    write_seed_files(taxonomy, args.seeds_dir)

    taxonomy_payload = {
        "aspects": [
            {
                "group": row["ASPECT"],
                "code": row["CODE"],
                "measurement_scale": row["MEASUREMENT_SCALE"],
                "description": row["DESCRIPTION"],
                "aspect_keywords": split_terms(row["ASPECT_KEYWORDS"]),
                "positive_sentiment_keywords": split_terms(
                    row["POSITIVE_SENTIMENT_KEYWORDS"]
                ),
                "negative_sentiment_keywords": split_terms(
                    row["NEGATIVE_SENTIMENT_KEYWORDS"]
                ),
                "neutral_sentiment_keywords": split_terms(
                    row["NEUTRAL_SENTIMENT_KEYWORDS"]
                ),
            }
            for row in taxonomy
        ]
    }
    write_json(args.taxonomy_json, taxonomy_payload)

    aspect_codes = ",".join(row["CODE"] for row in taxonomy)
    print("Prepared HASOS data: {0}".format(args.output_json))
    print("Prepared {0} aspect seed files: {1}".format(len(taxonomy), args.seeds_dir))
    print("Use --gold_aspects {0}".format(aspect_codes))


if __name__ == "__main__":
    main()
