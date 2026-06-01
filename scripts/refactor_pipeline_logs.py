"""Regenerate top10_hotels_pipeline_log.md from its JSON sibling.

Produces a refactored, easy-to-track Markdown report with:
  1. Pipeline Overview (mermaid)
  2. Pipeline Issues Found (auto-detected)
  3. Selected Hotels (with health Status)
  4. Step-by-Step Trace
  5. Per-Hotel Details (collapsible)
  6. Re-run Command

Usage:
    python refactor_pipeline_logs.py
        [--root ../outputs/hasos_english_only]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ---------- Heuristics ----------

ASC_OK = 0.73
CEC_OK = 0.54
ASC_BAD = 0.70
CEC_BAD = 0.50

# Topic vocabulary used to detect cross-aspect contamination in a summary
TOPIC_LEXICON: dict[str, set[str]] = {
    "room":     {"room", "rooms", "bed", "bedroom", "suite"},
    "bath":     {"bathroom", "shower", "toilet", "bath"},
    "food":     {"breakfast", "food", "buffet", "restaurant", "dinner", "lunch", "meal"},
    "pool":     {"pool", "rooftop"},
    "location": {"location", "located", "quarter", "street", "view", "views"},
    "staff":    {"staff", "staffs", "reception", "receptionist", "service", "host"},
    "wifi":     {"wifi", "wi-fi", "internet"},
}

GENERIC_PATTERNS = [
    re.compile(r"\bstaff (was|were|is|are) (very )?(friendly|helpful|nice|kind)", re.I),
    re.compile(r"\broom (was|is) (very )?(clean|comfortable)", re.I),
    re.compile(r"\bgreat location\b", re.I),
]


def status_badge(asc: float, cec: float) -> str:
    if asc >= ASC_OK and cec >= CEC_OK:
        return "[OK]"
    if asc < ASC_BAD and cec < CEC_BAD:
        return "[FAIL]"
    return "[WARN]"


def status_emoji(asc: float, cec: float) -> str:
    if asc >= ASC_OK and cec >= CEC_OK:
        return "[PASS]"
    if asc < ASC_BAD and cec < CEC_BAD:
        return "[FAIL]"
    return "[WARN]"


def topics_in(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z\-]+", text.lower()))
    return {topic for topic, vocab in TOPIC_LEXICON.items() if tokens & vocab}


def is_generic(text: str) -> bool:
    return any(p.search(text) for p in GENERIC_PATTERNS)


# ---------- Issue detection ----------

def detect_issues(results: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    # 1) Same representative sentence reused across aspects within a hotel
    for hotel_id, payload in results.items():
        first_summary_to_aspects: dict[str, list[str]] = defaultdict(list)
        for a in payload["aspects"]:
            if a.get("summary"):
                first_summary_to_aspects[a["summary"][0].strip()].append(a["aspect"])
        for sent, aspects in first_summary_to_aspects.items():
            if len(aspects) >= 2:
                issues.append({
                    "hotel": hotel_id,
                    "aspects": ", ".join(sorted(aspects)),
                    "issue": f"Identical representative sentence reused: \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"",
                    "step": "8 (TF-IDF centroid)",
                    "severity": "HIGH",
                })

    # 2) Cross-aspect contamination (summary mixes >=3 distinct topic vocabularies)
    for hotel_id, payload in results.items():
        for a in payload["aspects"]:
            if not a.get("summary"):
                continue
            sent = a["summary"][0]
            tops = topics_in(sent)
            if len(tops) >= 3:
                issues.append({
                    "hotel": hotel_id,
                    "aspects": a["aspect"],
                    "issue": f"Summary mixes {len(tops)} topics ({', '.join(sorted(tops))})",
                    "step": "5+8 (over-match + non-discriminative centroid)",
                    "severity": "MEDIUM",
                })

    # 3) Generic boilerplate repeated across hotels
    sentence_hotel_count: Counter[str] = Counter()
    sentence_hotels: dict[str, set[str]] = defaultdict(set)
    sentence_aspects: dict[str, set[str]] = defaultdict(set)
    for hotel_id, payload in results.items():
        for a in payload["aspects"]:
            if not a.get("summary"):
                continue
            sent = a["summary"][0].strip().lower()
            if is_generic(sent):
                key = sent
                sentence_hotel_count[key] += 1
                sentence_hotels[key].add(hotel_id)
                sentence_aspects[key].add(a["aspect"])
    for sent_key, count in sentence_hotel_count.most_common():
        if len(sentence_hotels[sent_key]) >= 4:
            issues.append({
                "hotel": f"{len(sentence_hotels[sent_key])} hotels",
                "aspects": ", ".join(sorted(sentence_aspects[sent_key])),
                "issue": f"Generic boilerplate repeated: \"{sent_key[:80]}{'...' if len(sent_key) > 80 else ''}\"",
                "step": "8 (no novelty filter)",
                "severity": "MEDIUM",
            })

    # 4) Aspects with CEC < 0.50
    low_cec_by_aspect: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for hotel_id, payload in results.items():
        for a in payload["aspects"]:
            if a["cec"] < CEC_BAD:
                low_cec_by_aspect[a["aspect"]].append((hotel_id, a["cec"]))
    for aspect, lst in sorted(low_cec_by_aspect.items(), key=lambda kv: -len(kv[1])):
        if len(lst) >= 3:
            avg = sum(c for _, c in lst) / len(lst)
            issues.append({
                "hotel": f"{len(lst)} hotels",
                "aspects": aspect,
                "issue": f"CEC below {CEC_BAD:.2f} (avg {avg:.3f}) — weak evidence coverage",
                "step": "7+8 (clustering/centroid)",
                "severity": "LOW",
            })

    return issues


# ---------- MD builders ----------

def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def build_overview(input_csv: str, lang_rule: str) -> str:
    return f"""## 1. Pipeline Overview

```mermaid
flowchart LR
    A[Raw CSV] --> B[Group rows by hotel_id]
    B --> C[English language filter]
    C --> D[Sentence split + normalize]
    D --> E[HASOS aspect match]
    E --> F[Dedup -> unique_opinions]
    F --> G[Cluster weight log(1+n)]
    G --> H[TF-IDF centroid -> representative sentence]
    H --> I[CEC / ASC metrics]
```

Input: `{input_csv}`

Language detector: {lang_rule}

ROUGE: not available (no human gold summaries in the source CSV).
"""


def build_issues(issues: list[dict[str, str]]) -> str:
    if not issues:
        return "## 2. Pipeline Issues Found\n\nNo automated issues detected.\n"
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    issues = sorted(issues, key=lambda x: (severity_rank.get(x["severity"], 9), x["hotel"]))
    counts = Counter(i["severity"] for i in issues)
    summary = " | ".join(f"{sev}: {counts.get(sev, 0)}" for sev in ("HIGH", "MEDIUM", "LOW"))
    lines = [
        "## 2. Pipeline Issues Found",
        "",
        f"Auto-detected from JSON log. **{summary}**",
        "",
        "| Severity | Hotel(s) | Aspect(s) | Issue | Failing Step |",
        "| :---: | --- | --- | --- | --- |",
    ]
    for it in issues:
        lines.append(
            f"| **{it['severity']}** | {it['hotel']} | `{it['aspects']}` | {it['issue']} | {it['step']} |"
        )
    return "\n".join(lines) + "\n"


def build_selected_table(results: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    lines = [
        "## 3. Selected Hotels",
        "",
        f"Thresholds — PASS: ASC >= {ASC_OK} AND CEC >= {CEC_OK}; FAIL: ASC < {ASC_BAD} AND CEC < {CEC_BAD}; WARN otherwise.",
        "",
        "| # | Hotel ID | Reviews | Sentences | Aspects | ASC | Macro CEC | Weighted CEC | Status |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for i, sel in enumerate(selected, 1):
        hid = sel["hotel_id"]
        o = results[hid]["overall"]
        lines.append(
            f"| {i} | `{hid}` | {fmt_int(o['review_count'])} | {fmt_int(o['sentence_count'])} | "
            f"{o['matched_aspects']}/{o['aspect_count']} | {o['aspect_summary_cover']:.4f} | "
            f"{o['macro_cec']:.4f} | {o['weighted_cec']:.4f} | "
            f"{status_emoji(o['aspect_summary_cover'], o['macro_cec'])} |"
        )
    return "\n".join(lines) + "\n"


def build_step_trace(results: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    if not selected:
        return ""
    first = selected[0]
    hid = first["hotel_id"]
    o = results[hid]["overall"]
    top_aspect = max(results[hid]["aspects"], key=lambda a: a["asc_contribution"])
    first_review = (first.get("first_review") or "").replace("|", "\\|")[:120]
    first_review_short = first_review + ("..." if len(first.get("first_review") or "") > 120 else "")
    sample_summary = (top_aspect["summary"][0] if top_aspect.get("summary") else "")[:100]

    rows = [
        ("1", "Load CSV", "Raw rows", f"{fmt_int(o['review_count'])} reviews kept", first_review_short, "OK"),
        ("2", "Group by hotel_id", "ref_id without numeric suffix", f"Hotel `{hid}`", f"{fmt_int(o['review_count'])} rows", "OK"),
        ("3", "Sentence split", "Per-review splitter", f"{fmt_int(o['sentence_count'])} sentences", first_review_short.split('.')[0], "OK"),
        ("4", "Normalize", "lowercase, strip accents", "Normalized text", first_review_short.lower(), "OK"),
        ("5", "Aspect match", "Keyword vs HASOS taxonomy", f"{o['matched_aspects']}/{o['aspect_count']} aspects hit", f"Top: `{top_aspect['aspect']}`", "WARN: over-matches"),
        ("6", "Dedup", "Unique opinions per aspect", f"{fmt_int(top_aspect['unique_opinions'])} unique / {fmt_int(top_aspect['matched_sentences'])} matched", "—", "OK"),
        ("7", "Cluster weight", "log(1+n) normalized", f"weight = {top_aspect['cluster_weight']:.4f}", "—", "OK"),
        ("8", "Representative", "TF-IDF centroid pick", "1 sentence per aspect", sample_summary, "FAIL: not discriminative"),
        ("9", "Metrics", "CEC / ASC", f"ASC {o['aspect_summary_cover']:.4f} / CEC {o['macro_cec']:.4f}", "—", "WARN: inherits step-8 noise"),
    ]
    lines = [
        "## 4. Step-by-Step Trace",
        "",
        f"Single example: hotel #1 `{hid}` (top aspect `{top_aspect['aspect']}`).",
        "",
        "| Step | Transform | Input | Output | Sample evidence | Health |",
        "| ---: | --- | --- | --- | --- | :---: |",
    ]
    for step, transform, _in, _out, sample, health in rows:
        sample = sample.replace("|", "\\|")
        lines.append(f"| {step} | {transform} | {_in} | {_out} | {sample} | {health} |")
    return "\n".join(lines) + "\n"


def build_hotel_details(results: dict[str, Any], selected: list[dict[str, Any]]) -> str:
    lines = ["## 5. Per-Hotel Details", "", "_Click a hotel to expand._", ""]
    for i, sel in enumerate(selected, 1):
        hid = sel["hotel_id"]
        o = results[hid]["overall"]
        st = status_emoji(o["aspect_summary_cover"], o["macro_cec"])
        # Top 5 aspects by ASC contribution
        top5 = sorted(results[hid]["aspects"], key=lambda a: -a["asc_contribution"])[:5]
        first_review = (sel.get("first_review") or "")[:200]
        first_review += "..." if len(sel.get("first_review") or "") > 200 else ""

        block = [
            "<details>",
            f"<summary><strong>{i}. <code>{hid}</code></strong> &mdash; "
            f"ASC {o['aspect_summary_cover']:.4f} &middot; CEC {o['macro_cec']:.4f} "
            f"&middot; {fmt_int(o['review_count'])} reviews &middot; {st}</summary>",
            "",
            f"- Sentences: {fmt_int(o['sentence_count'])}",
            f"- Matched aspects: {o['matched_aspects']}/{o['aspect_count']}",
            f"- Weighted CEC: {o['weighted_cec']:.4f}",
            f"- First review sample: {first_review}",
            "",
            "**Top 5 aspects by ASC contribution**",
            "",
            "| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for a in top5:
            summary = (a["summary"][0] if a.get("summary") else "—").replace("|", "\\|")
            if len(summary) > 120:
                summary = summary[:120] + "..."
            block.append(
                f"| `{a['aspect']}` | {fmt_int(a['unique_opinions'])} | "
                f"{a['cluster_weight']:.4f} | {a['cec']:.4f} | {a['asc_contribution']:.4f} | {summary} |"
            )
        block += ["", "</details>", ""]
        lines.append("\n".join(block))
    return "\n".join(lines)


def build_rerun(csv_path: str) -> str:
    name = Path(csv_path).name
    return f"""## 6. Re-run Command

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\\scripts
python .\\log_10_hotels_pipeline.py --input-csv ..\\..\\{name} --limit 10
```
"""


# ---------- Per-file orchestration ----------

def regenerate(json_path: Path) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = data["results"]
    selected: list[dict[str, Any]] = data["selected_hotels"]
    issues = detect_issues(results)

    csv_name = Path(data["input_csv"]).name
    title = f"# Top-10 Hotels Pipeline Log &mdash; `{csv_name}`"
    intro = "Refactored view of the SemAE HASOS English-only scoring run. " \
            "All numbers below are pulled from the sibling JSON log."

    out = "\n\n".join([
        title,
        intro,
        build_overview(data["input_csv"], data.get("language_detector_rule", "n/a")),
        build_issues(issues),
        build_selected_table(results, selected),
        build_step_trace(results, selected),
        build_hotel_details(results, selected),
        build_rerun(data["input_csv"]),
    ]).rstrip() + "\n"

    md_path = json_path.with_suffix(".md")
    md_path.write_text(out, encoding="utf-8")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "hasos_english_only",
    )
    args = parser.parse_args()
    found = sorted(args.root.glob("hotel_review*/top10_hotels_pipeline_log.json"))
    if not found:
        raise SystemExit(f"No pipeline logs under {args.root}")
    for jp in found:
        out = regenerate(jp)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
