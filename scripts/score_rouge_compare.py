"""Compute official pyrouge ROUGE for the 4-method comparison against HASOS gold.

The four methods (all derive from the SAME SemAE sentence selection; they differ
only in how selected evidence is rendered):

  m1_extractive : raw SemAE-selected sentences (outputs/<run>/<SUBASPECT>/<split>_<id>)
  m2_abstractive: FLAN-T5 rewrite, no sentiment split (parent-level dir)
  m3_kw         : sentiment-split abstractive, keyword backend (pos+neg concatenated)
  m4_bert       : sentiment-split abstractive, BERT-ABSA backend (pos+neg concatenated)

HASOS gold (data/hasos/hasos_summ.json) only has reference summaries for the 4
PARENT aspects facility/amenity/service/experience, so every method's sub-aspect
outputs are aggregated up to their parent group before scoring. Branding/loyalty
have (almost) no gold and are skipped.

Usage:
    python scripts/score_rouge_compare.py --method m1_extractive \
        --run_id space_hasos_threshold_full \
        --out reports/rouge_m1_hasos.json

Mirrors the official SPACE report: per-aspect ROUGE-1/2/L F1, split by dev/test/all,
plus a macro average over aspects.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rouge_patch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUGE_HOME = os.path.join(REPO, "downloads", "rouge_setup", "ROUGE-1.5.5")

# Gold parent key (lowercase) -> taxonomy group (uppercase)
PARENT_KEYS = {
    "facility": "FACILITY",
    "amenity": "AMENITY",
    "service": "SERVICE",
    "experience": "EXPERIENCE",
}

# SPACE has 6 flat generic aspects; the gold key IS the output subdir name, so
# the "group" is an identity mapping. The optional "general" gold key is the
# entity-level / overall summary and is scored separately as GENERAL.
SPACE_ASPECTS = ["building", "cleanliness", "food", "location", "rooms", "service"]
SPACE_GENERAL_KEY = "general"
GENERAL_RESULT_KEY = "GENERAL"


def load_taxonomy_groups(path):
    """Return {SUBASPECT_CODE: GROUP}."""
    tax = json.load(io.open(path, encoding="utf-8"))
    return {a["code"]: a["group"] for a in tax["aspects"]}


def load_gold(path):
    """Return {entity_id: {parent_key: [ref_text, ...]}}."""
    data = json.load(io.open(path, encoding="utf-8"))
    gold = {}
    for ent in data:
        eid = str(ent["entity_id"])
        summaries = ent.get("summaries", {})
        gold[eid] = {k: list(v) for k, v in summaries.items() if v}
    return gold


def clean_text(s: str) -> str:
    s = s.replace("\t", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collect_entity_files(run_dir):
    """Map split_entity dir entries -> {(split, entity_id): filepath} per subdir.

    Files are named like dev_100597 / test_182002 inside each aspect subdir.
    """
    return run_dir  # resolver handles structure per method


def system_text_extractive(run_dir, group, split, eid, code2group):
    """M1: union all sub-aspect sentence files for this parent group + entity."""
    parts = []
    for sub in sorted(os.listdir(run_dir)):
        subdir = os.path.join(run_dir, sub)
        if not os.path.isdir(subdir):
            continue
        if code2group.get(sub) != group:
            continue
        fp = os.path.join(subdir, f"{split}_{eid}")
        if os.path.isfile(fp):
            with io.open(fp, encoding="utf-8") as f:
                parts.append(f.read())
    return clean_text(" ".join(parts))


def system_text_parent_dir(parent_dir, group, split, eid):
    """M2: one file per entity under <group>/<split>_<eid>."""
    fp = os.path.join(parent_dir, group, f"{split}_{eid}")
    if os.path.isfile(fp):
        with io.open(fp, encoding="utf-8") as f:
            return clean_text(f.read())
    return ""


def system_text_sentiment(senti_dir, group, split, eid, code2group,
                          polarities=("pos", "neg")):
    """M3/M4: concat pos+neg sentiment-split files for sub-aspects in this group.

    Synthesis writes a nested tree <ASPECT_CODE>/<sentiment>/<split>_<eid>
    (sentiment in {pos, neg, neu}). We union the requested polarities across
    every sub-aspect code belonging to this parent group.
    """
    parts = []
    for code in sorted(os.listdir(senti_dir)):
        code_dir = os.path.join(senti_dir, code)
        if not os.path.isdir(code_dir):
            continue
        if code2group.get(code) != group:
            continue
        for pol in polarities:
            fp = os.path.join(code_dir, pol, f"{split}_{eid}")
            if os.path.isfile(fp):
                with io.open(fp, encoding="utf-8") as f:
                    parts.append(f.read())
    return clean_text(" ".join(parts))


def system_text_space_general(root_dir, method, split, eid, entity_dir=None):
    """Return SPACE entity-level / overall system text for one method.

    Preference order:
      1. A generated entity-level overall file `<entity_dir>/<split>_<eid>`
         (produced by hierarchical synthesis with --entity_max_new_tokens).
      2. For M2, a flat `<root_dir>/<split>_<eid>` file.
      3. Fallback: concatenate the six SPACE aspect outputs (M1/M2) or
         pos+neg across the six aspects (M3/M4).

    This mirrors the available SPACE gold `general` references without
    inventing sentiment-level gold.
    """
    file_name = f"{split}_{eid}"
    if entity_dir:
        ent_fp = os.path.join(entity_dir, file_name)
        if os.path.isfile(ent_fp):
            with io.open(ent_fp, encoding="utf-8") as f:
                text = clean_text(f.read())
            if text:
                return text
    parts = []
    if method == "m2_abstractive":
        direct = os.path.join(root_dir, file_name)
        if os.path.isfile(direct):
            with io.open(direct, encoding="utf-8") as f:
                return clean_text(f.read())
    for aspect in SPACE_ASPECTS:
        aspect_dir = os.path.join(root_dir, aspect)
        if method in {"m1_extractive", "m2_abstractive"}:
            fp = os.path.join(aspect_dir, file_name)
            if os.path.isfile(fp):
                with io.open(fp, encoding="utf-8") as f:
                    parts.append(f.read())
        else:
            for pol in ("pos", "neg"):
                fp = os.path.join(aspect_dir, pol, file_name)
                if os.path.isfile(fp):
                    with io.open(fp, encoding="utf-8") as f:
                        parts.append(f.read())
    return clean_text(" ".join(parts))


def discover_entities(run_dir):
    """Return sorted [(split, eid)] from filenames at any depth under run_dir.

    Handles both flat m1 layout (<ASPECT>/<split>_<eid>) and nested m3/m4
    layout (<ASPECT>/<sentiment>/<split>_<eid>).
    """
    found = set()
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            m = re.match(r"(dev|test|train)_(.+)$", name)
            if m:
                found.add((m.group(1), m.group(2)))
    return sorted(found)


def run_rouge(pairs):
    """pairs: list of (system_text, [ref_texts]). Returns dict of ROUGE F1s."""
    from pyrouge import Rouge155
    tmp = tempfile.mkdtemp()
    sys_dir = os.path.join(tmp, "system")
    mod_dir = os.path.join(tmp, "model")
    os.makedirs(sys_dir)
    os.makedirs(mod_dir)

    n_written = 0
    for i, (sys_text, refs) in enumerate(pairs, 1):
        if not sys_text.strip() or not any(r.strip() for r in refs):
            continue
        with io.open(os.path.join(sys_dir, f"text.{i:03d}.txt"), "w",
                     encoding="utf-8") as f:
            f.write(sys_text)
        for j, ref in enumerate(refs):
            letter = chr(ord("A") + j)
            with io.open(os.path.join(mod_dir, f"text.{letter}.{i:03d}.txt"),
                         "w", encoding="utf-8") as f:
                f.write(clean_text(ref))
        n_written += 1

    if n_written == 0:
        return None

    r = Rouge155(rouge_dir=ROUGE_HOME)
    r.system_dir = sys_dir
    r.model_dir = mod_dir
    r.system_filename_pattern = r"text.(\d+).txt"
    r.model_filename_pattern = "text.[A-Z].#ID#.txt"
    output = r.convert_and_evaluate()
    return parse_rouge(output), n_written


def parse_rouge(output):
    """Extract Average_F for ROUGE-1, ROUGE-2, ROUGE-L."""
    res = {}
    for tag, key in (("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"),
                     ("ROUGE-L", "rougeL")):
        m = re.search(rf"{tag} Average_F: ([0-9.]+)", output)
        if m:
            res[key] = float(m.group(1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["m1_extractive", "m2_abstractive", "m3_kw", "m4_bert"])
    ap.add_argument("--run_dir", help="root dir holding aspect subdirs (m1)")
    ap.add_argument("--parent_dir", help="parent-level abstractive dir (m2)")
    ap.add_argument("--senti_dir", help="sentiment-split dir (m3/m4)")
    ap.add_argument("--taxonomy", default="data/hasos/aspect_taxonomy.json")
    ap.add_argument("--gold", default="data/hasos/hasos_summ.json")
    ap.add_argument("--dataset", choices=["hasos", "space"], default="hasos",
                    help="hasos: aggregate 29 sub-aspects -> 4 gold parents. "
                         "space: 6 flat generic aspects, identity mapping.")
    ap.add_argument("--include_general", action="store_true",
                    help="For SPACE, also score the non-aspectual 'general' gold "
                         "summary as GENERAL by concatenating the six aspect "
                         "system outputs.")
    ap.add_argument("--entity_dir",
                    help="For SPACE --include_general: directory of generated "
                         "entity-level overall files (<split>_<eid>), e.g. "
                         "outputs/space_eval_4method_m2_entity. Preferred over "
                         "aspect concatenation when present.")
    ap.add_argument("--fixed_denominator", action="store_true",
                    help="Score EVERY gold-bearing (aspect, entity) instance, "
                         "counting empty system outputs as ROUGE 0. This keeps "
                         "the macro denominator constant across a threshold/token "
                         "sweep so a method that simply answers fewer instances "
                         "is not flattered. Default off (reproduces the original "
                         "baseline that skips empty outputs).")
    ap.add_argument("--universe_dir",
                    help="Directory whose <split>_<eid> filenames define the FULL "
                         "set of (split, entity) instances to score. Use a "
                         "full-coverage baseline output dir so the universe is "
                         "stable even when the swept output dir is sparse. "
                         "Defaults to the scored output dir (legacy behavior).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rouge_patch.apply(rouge_home=ROUGE_HOME)

    if args.dataset == "space":
        # SPACE: aspect == output subdir == gold key; identity group mapping.
        parent_keys = {a: a for a in SPACE_ASPECTS}
        code2group = {a: a for a in SPACE_ASPECTS}
    else:
        parent_keys = PARENT_KEYS
        code2group = load_taxonomy_groups(os.path.join(REPO, args.taxonomy))
    gold = load_gold(os.path.join(REPO, args.gold))

    # entity discovery source. With --universe_dir, the full (split, entity)
    # universe comes from a stable full-coverage baseline dir so a sparse swept
    # output does not shrink the denominator.
    disc_dir = args.universe_dir or args.run_dir or args.senti_dir or args.parent_dir
    disc_dir = os.path.join(REPO, disc_dir)
    if args.method == "m2_abstractive" and not args.universe_dir:
        # parent dir has GROUP subdirs, files inside
        entities = set()
        for grp in os.listdir(disc_dir):
            gp = os.path.join(disc_dir, grp)
            if not os.path.isdir(gp):
                continue
            for name in os.listdir(gp):
                m = re.match(r"(dev|test|train)_(.+)$", name)
                if m:
                    entities.add((m.group(1), m.group(2)))
        entities = sorted(entities)
    else:
        entities = discover_entities(disc_dir)

    def get_system_text(group, split, eid):
        if args.method == "m1_extractive":
            return system_text_extractive(
                os.path.join(REPO, args.run_dir), group, split, eid, code2group)
        if args.method == "m2_abstractive":
            return system_text_parent_dir(
                os.path.join(REPO, args.parent_dir), group, split, eid)
        # m3/m4
        return system_text_sentiment(
            os.path.join(REPO, args.senti_dir), group, split, eid, code2group)

    results = {"method": args.method, "by_split": {}, "coverage": {},
               "fixed_denominator": bool(args.fixed_denominator)}

    for split_filter, label in ((("dev",), "dev"), (("test",), "test"),
                                (("dev", "test", "train"), "all")):
        per_aspect = {}
        for pkey, group in parent_keys.items():
            pairs = []
            eligible = 0  # (entity) with gold for this aspect+split
            for split, eid in entities:
                if split not in split_filter:
                    continue
                refs = gold.get(eid, {}).get(pkey, [])
                if not refs:
                    continue
                eligible += 1
                sys_text = get_system_text(group, split, eid)
                if not sys_text.strip():
                    continue
                pairs.append((sys_text, refs))
            out = run_rouge(pairs)
            n = out[1] if out is not None else 0
            scores = out[0] if out is not None else {}
            # coverage = fraction of gold-bearing instances we actually answered
            cov = (n / eligible) if eligible else 0.0
            if args.fixed_denominator:
                # exact: mean over ALL eligible instances, empties contribute 0.
                # pyrouge Average_F is over n written pairs, so rescale by n/eligible.
                scaled = {k: (scores.get(k, 0.0) * n / eligible) if eligible else 0.0
                          for k in ("rouge1", "rouge2", "rougeL")}
                scaled["n"] = n
                scaled["eligible"] = eligible
                scaled["coverage"] = round(cov, 4)
                # include the aspect whenever it has ANY gold (even if 0 answered)
                if eligible:
                    per_aspect[pkey] = scaled
            else:
                if out is None:
                    continue
                scores["n"] = n
                scores["eligible"] = eligible
                scores["coverage"] = round(cov, 4)
                per_aspect[pkey] = scores
        # macro average over aspect keys only. GENERAL, when present, is an
        # entity-level SPACE score and must not be folded into aspect MACRO.
        if per_aspect:
            macro = {}
            aspect_scores = [
                v for k, v in per_aspect.items()
                if k != GENERAL_RESULT_KEY
            ]
            for k in ("rouge1", "rouge2", "rougeL"):
                vals = [v[k] for v in aspect_scores if k in v]
                macro[k] = sum(vals) / len(vals) if vals else 0.0
            per_aspect["MACRO"] = macro

        if args.dataset == "space" and args.include_general:
            pairs = []
            # general system text comes from the dir being SCORED, not the
            # (possibly different) universe dir used only for entity discovery.
            general_root = os.path.join(
                REPO, args.run_dir or args.senti_dir or args.parent_dir)
            eligible = 0
            for split, eid in entities:
                if split not in split_filter:
                    continue
                refs = gold.get(eid, {}).get(SPACE_GENERAL_KEY, [])
                if not refs:
                    continue
                eligible += 1
                entity_dir = (os.path.join(REPO, args.entity_dir)
                              if args.entity_dir else None)
                sys_text = system_text_space_general(
                    general_root, args.method, split, eid, entity_dir)
                if not sys_text.strip():
                    continue
                pairs.append((sys_text, refs))
            out = run_rouge(pairs)
            n = out[1] if out is not None else 0
            gscores = out[0] if out is not None else {}
            cov = (n / eligible) if eligible else 0.0
            if args.fixed_denominator:
                if eligible:
                    scaled = {k: gscores.get(k, 0.0) * n / eligible
                              for k in ("rouge1", "rouge2", "rougeL")}
                    scaled["n"] = n
                    scaled["eligible"] = eligible
                    scaled["coverage"] = round(cov, 4)
                    per_aspect[GENERAL_RESULT_KEY] = scaled
            elif out is not None:
                gscores["n"] = n
                gscores["eligible"] = eligible
                gscores["coverage"] = round(cov, 4)
                per_aspect[GENERAL_RESULT_KEY] = gscores
        results["by_split"][label] = per_aspect

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # console summary
    print(f"\n=== {args.method} :: ROUGE F1 (macro over aspects) ===")
    for label in ("dev", "test", "all"):
        per = results["by_split"].get(label, {})
        m = per.get("MACRO")
        if m:
            covs = [v.get("coverage") for k, v in per.items()
                    if k not in ("MACRO", GENERAL_RESULT_KEY)
                    and isinstance(v, dict) and v.get("coverage") is not None]
            mean_cov = sum(covs) / len(covs) if covs else 0.0
            print(f"  {label:5s} R1={m['rouge1']:.5f} "
                  f"R2={m['rouge2']:.5f} RL={m['rougeL']:.5f} "
                  f"cov={mean_cov:.2f} ({len(covs)} aspects)")
    general = results["by_split"].get("all", {}).get(GENERAL_RESULT_KEY)
    if general:
        print(f"  {'general':5s} R1={general['rouge1']:.5f} "
              f"R2={general['rouge2']:.5f} RL={general['rougeL']:.5f}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
