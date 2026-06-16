"""Re-label an evidence JSONL's sentiment_label with a GENERIC keyword lexicon.

The HASOS keyword backend reads per-aspect sentiment keyword lists from the HASOS
taxonomy. SPACE's 6 generic aspects (building/cleanliness/food/location/rooms/
service) have no such taxonomy, so for the M3 (keyword) method on SPACE we apply a
single domain-general positive/negative opinion lexicon to every sentence. This is
the closest faithful analogue of "keyword sentiment, no learned model" for SPACE.

Sentence SELECTION is identical across sentiment backends, so this only overwrites
`sentiment_label` on the shared evidence rows.

Usage:
    python scripts/relabel_evidence_keyword_generic.py \
        --in  outputs/space_eval_4method_threshold_evidence.jsonl \
        --out outputs/space_eval_4method_kw_threshold_evidence.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSITIVE = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic", "lovely",
    "perfect", "best", "beautiful", "clean", "comfortable", "comfy", "friendly",
    "helpful", "spacious", "quiet", "nice", "pleasant", "enjoyed", "enjoyable",
    "delicious", "tasty", "fresh", "convenient", "recommend", "recommended",
    "stunning", "gorgeous", "spotless", "modern", "cozy", "attentive", "polite",
    "welcoming", "value", "affordable", "fast", "efficient", "smooth", "superb",
    "outstanding", "exceptional", "warm", "generous", "impeccable", "charming",
    "happy", "loved", "love", "awesome", "brilliant", "favorite", "well",
}
NEGATIVE = {
    "bad", "poor", "terrible", "awful", "horrible", "worst", "dirty", "filthy",
    "rude", "unfriendly", "unhelpful", "slow", "noisy", "loud", "small", "cramped",
    "broken", "outdated", "old", "smelly", "smell", "stained", "uncomfortable",
    "expensive", "overpriced", "disappointing", "disappointed", "dated", "worn",
    "cold", "stale", "bland", "tasteless", "mediocre", "lacking", "lack", "problem",
    "issue", "issues", "complaint", "complaints", "wait", "waiting", "unclean",
    "mold", "moldy", "bugs", "roaches", "leaking", "leak", "tired", "shabby",
    "avoid", "never", "unacceptable", "disgusting", "nightmare", "ruined",
    "frustrating", "frustrated", "miserable", "subpar", "lousy", "grim",
}
NEGATORS = {"not", "no", "never", "n't", "without", "barely", "hardly"}

WORD_RE = re.compile(r"[a-z']+")


def classify(sentence: str) -> str:
    toks = WORD_RE.findall(sentence.lower())
    pos = neg = 0
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i > 0 else ""
        prev2 = toks[i - 2] if i > 1 else ""
        negated = prev in NEGATORS or prev2 in NEGATORS or prev.endswith("n't")
        if t in POSITIVE:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif t in NEGATIVE:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    if pos == 0 and neg == 0:
        return "neu"
    return "pos" if pos >= neg else "neg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    rows = []
    with io.open(os.path.join(REPO, args.inp), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dist = {}
    for r in rows:
        label = classify(r.get("sentence", ""))
        r["sentiment_label"] = label
        r["matched_sentiment_keywords"] = ["generic_kw"]
        dist[label] = dist.get(label, 0) + 1

    out_path = os.path.join(REPO, args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[relabel-kw] {len(rows)} rows; new distribution: {dist}")
    print(f"[relabel-kw] written -> {args.out}")


if __name__ == "__main__":
    main()
