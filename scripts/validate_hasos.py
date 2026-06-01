#!/usr/bin/env python3
import argparse
import json
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Validate prepared HASOS SemAE inputs.")
    parser.add_argument(
        "--data",
        default=os.path.join(REPO_ROOT, "data", "hasos", "hasos_summ.json"),
        help="Prepared SemAE JSON.",
    )
    parser.add_argument(
        "--taxonomy",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.json"),
        help="Prepared taxonomy JSON.",
    )
    parser.add_argument(
        "--seeds-dir",
        default=os.path.join(REPO_ROOT, "data", "seeds_hasos"),
        help="Generated seed directory.",
    )
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    with open(args.taxonomy, "r", encoding="utf-8") as fin:
        taxonomy = json.load(fin)

    review_count = 0
    sentence_count = 0
    ratings = {}
    for entity in data:
        if "entity_id" not in entity or "reviews" not in entity:
            raise SystemExit("Invalid entity row: missing entity_id or reviews")
        for review in entity["reviews"]:
            if "review_id" not in review or "sentences" not in review:
                raise SystemExit("Invalid review row: missing review_id or sentences")
            review_count += 1
            sentence_count += len(review["sentences"])
            rating = review.get("rating", "missing")
            ratings[rating] = ratings.get(rating, 0) + 1

    missing_seeds = []
    for aspect in taxonomy["aspects"]:
        seed_path = os.path.join(args.seeds_dir, aspect["code"] + ".txt")
        if not os.path.exists(seed_path):
            missing_seeds.append(aspect["code"])

    if missing_seeds:
        raise SystemExit("Missing seed files: {0}".format(", ".join(missing_seeds)))

    print("entities={0}".format(len(data)))
    print("reviews={0}".format(review_count))
    print("sentences={0}".format(sentence_count))
    print("aspects={0}".format(len(taxonomy["aspects"])))
    print("ratings={0}".format(dict(sorted(ratings.items(), key=lambda item: str(item[0])))))


if __name__ == "__main__":
    main()
