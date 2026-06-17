#!/usr/bin/env python3
"""Strip leaked evidence-list labels from generated entity-overall summaries.

The entity/overall prompt feeds aspect summaries as a numbered list like::

    1. building: ...
    2. cleanliness (positive): ...

FLAN-T5 sometimes echoes those `N. aspect:` / `aspect (positive):` markers into
the generated overall summary. This pass removes the markers in place so the
overall text reads as clean prose and scores fairly against the SPACE `general`
gold references (which contain no such labels).

Usage:
    python scripts/clean_overall_summaries.py \
        --entity_dir outputs/space_eval_4method_m2_entity
    # or point at several dirs:
    python scripts/clean_overall_summaries.py \
        --entity_dir outputs/space_eval_4method_m2_entity \
        --entity_dir outputs/space_eval_4method_m3_kw_entity
"""
from __future__ import annotations

import argparse
import os
import re

# SPACE flat aspects + the HASOS parent group labels, lowercase.
ASPECTS = [
    "building", "cleanliness", "food", "location", "rooms", "service",
    "facility", "amenity", "experience",
]
_ASPECT_ALT = "|".join(ASPECTS)

# `1. building:` / `2. cleanliness (positive):` / bare `food:` enumerations.
LABEL_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))\d+\.\s*(?:%s)\s*(?:\((?:positive|negative)\))?\s*:\s*"
    % _ASPECT_ALT,
    re.IGNORECASE,
)
BARE_LABEL_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))(?:%s)\s*(?:\((?:positive|negative)\))?\s*:\s*"
    % _ASPECT_ALT,
    re.IGNORECASE,
)
# Leading orphan enumeration like "2. " left after a label removal.
ORPHAN_NUM_RE = re.compile(r"(?:(?<=^)|(?<=\s))\d+\.\s+")


def clean_text(text: str) -> str:
    out = LABEL_RE.sub(" ", text)
    out = BARE_LABEL_RE.sub(" ", out)
    out = ORPHAN_NUM_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity_dir", action="append", required=True,
                    help="directory of <split>_<eid> overall files; repeatable")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    total = 0
    changed = 0
    for d in args.entity_dir:
        if not os.path.isdir(d):
            print(f"skip (missing): {d}")
            continue
        for name in sorted(os.listdir(d)):
            fp = os.path.join(d, name)
            if not os.path.isfile(fp):
                continue
            total += 1
            original = open(fp, encoding="utf-8").read()
            cleaned = clean_text(original)
            if cleaned != original.strip():
                changed += 1
                if not args.dry_run:
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(cleaned)
    verb = "would change" if args.dry_run else "changed"
    print(f"files={total} {verb}={changed}")


if __name__ == "__main__":
    main()
