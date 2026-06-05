#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[idx]


def summarize(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def load_spm(path):
    if not path:
        return None
    import sentencepiece as spm
    model = spm.SentencePieceProcessor()
    model.Load(path)
    return model


def analyze(data, spm_model=None, max_entities=None, seed=1, max_sen_len=40,
            max_rev_len=40):
    if max_entities is not None:
        rng = random.Random(seed)
        data = rng.sample(data, max_entities)

    review_counts = []
    sentence_counts = []
    raw_sentence_lengths = []
    token_lengths = []
    clipped_token_lengths = []
    empty_sentences = 0
    ratings = Counter()

    for entity in data:
        reviews = entity.get("reviews", [])
        review_counts.append(len(reviews))
        for review in reviews:
            rating = review.get("rating", "missing")
            ratings[str(rating)] += 1
            sentences = review.get("sentences", [])[:max_rev_len]
            sentence_counts.append(len(sentences))
            for sentence in sentences:
                if not sentence:
                    empty_sentences += 1
                raw_sentence_lengths.append(len(sentence))
                if spm_model is not None:
                    toks = spm_model.EncodeAsIds(sentence)
                    token_lengths.append(len(toks))
                    clipped_token_lengths.append(len(toks[:max_sen_len]))

    return {
        "entities": len(data),
        "reviews": sum(review_counts),
        "empty_sentences": empty_sentences,
        "ratings": dict(sorted(ratings.items())),
        "reviews_per_entity": summarize(review_counts),
        "sentences_per_review": summarize(sentence_counts),
        "raw_chars_per_sentence": summarize(raw_sentence_lengths),
        "spm_tokens_per_sentence": summarize(token_lengths),
        "clipped_spm_tokens_per_sentence": summarize(clipped_token_lengths),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SPACE data shape for SemAE training diagnostics.")
    parser.add_argument("--data", default="../data/space/json/space_train.json")
    parser.add_argument("--sentencepiece", default="")
    parser.add_argument("--max_entities", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_sen_len", type=int, default=40)
    parser.add_argument("--max_rev_len", type=int, default=40)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as fin:
        data = json.load(fin)
    spm_model = load_spm(args.sentencepiece) if args.sentencepiece else None
    summary = analyze(data,
                      spm_model=spm_model,
                      max_entities=args.max_entities,
                      seed=args.seed,
                      max_sen_len=args.max_sen_len,
                      max_rev_len=args.max_rev_len)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
