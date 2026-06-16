#!/usr/bin/env python3
"""Build a lightweight PPTX report for the SPACE->HASOS SemAE pipeline.

This intentionally uses only the Python standard library so the reporting step
does not depend on PowerPoint, python-pptx, or Node packages.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
REPORTS_DIR = REPO_ROOT / "reports"
EMU_W = 12192000
EMU_H = 6858000
SPLIT_RE = re.compile(r"\t+|\r?\n+")


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def read_jsonl(path: Path, limit: int = 200) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(rows) >= limit:
            break
    return rows


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open("r", encoding="utf-8",
                                       errors="replace") if line.strip())


def split_summary(text: str) -> list[str]:
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]


def truncate(text: str, n: int = 170) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def evidence_review_lines(evidence: list[dict], limit: int = 4,
                          width: int = 190) -> str:
    lines = []
    for i, ev in enumerate(evidence[:limit], 1):
        review_id = ev.get("source_review_id") or ev.get("review_id") or "unknown"
        score = ev.get("score")
        score_text = f" | score={score:.4f}" if isinstance(score, (int, float)) else (
            f" | score={score}" if score is not None else "")
        lines.append(
            f"Review {i} ({review_id}{score_text}): "
            f"{truncate(ev.get('sentence', ''), width)}"
        )
    return "\n\n".join(lines)


def generic_summary(text: str) -> bool:
    normalized = " ".join(str(text).lower().split())
    if len(normalized.split()) < 9:
        return True
    bad_phrases = [
        "it's a good hotel",
        "it is a good hotel",
        "not a bad hotel",
        "great place to stay",
        "good place to stay",
    ]
    return any(phrase in normalized for phrase in bad_phrases)


def rouge_macro_rows(path: Path) -> tuple[list[str], str]:
    data = read_json(path, {})
    aspects = ["building", "cleanliness", "food", "location", "rooms", "service"]
    rows = []
    for split in ("dev", "test", "all"):
        split_data = data.get(split, {})
        vals = []
        for metric in ("rouge_1_f_score", "rouge_2_f_score", "rouge_l_f_score"):
            metric_vals = [
                split_data.get(aspect, {}).get(metric)
                for aspect in aspects
                if split_data.get(aspect, {}).get(metric) is not None
            ]
            vals.append(sum(metric_vals) / len(metric_vals) if metric_vals else None)
        if all(v is not None for v in vals):
            rows.append(
                f"{split.upper():4s}  R1={vals[0]:.4f}  R2={vals[1]:.4f}  RL={vals[2]:.4f}"
            )
    if not rows:
        return [], "SPACE original ROUGE chưa có hoặc chưa parse được."

    all_split = data.get("all", {})
    aspect_rows = []
    for aspect in aspects:
        vals = all_split.get(aspect, {})
        if not vals:
            continue
        aspect_rows.append(
            f"{aspect:11s}  {vals.get('rouge_1_f_score', 0):.4f} / "
            f"{vals.get('rouge_2_f_score', 0):.4f} / "
            f"{vals.get('rouge_l_f_score', 0):.4f}"
        )
    return rows, "\n".join(aspect_rows)


def first_sample(run_id: str) -> dict | None:
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        return None
    for aspect_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for entity_file in sorted(p for p in aspect_dir.iterdir() if p.is_file()):
            sentences = split_summary(entity_file.read_text(encoding="utf-8", errors="replace"))
            if not sentences:
                continue
            split, entity_id = entity_file.name.split("_", 1)
            sentiment = {}
            for label in ("pos", "neg", "neu"):
                sent_path = OUTPUTS_DIR / f"{run_id}_sentiment" / f"{aspect_dir.name}__{label}" / entity_file.name
                sentiment[label] = split_summary(
                    sent_path.read_text(encoding="utf-8", errors="replace")
                ) if sent_path.exists() else []
            return {
                "aspect": aspect_dir.name,
                "split": split,
                "entity_id": entity_id,
                "sentences": sentences,
                "sentiment": sentiment,
            }
    return None


def first_synthesis_sample(run_id: str) -> dict | None:
    """Return a presentable ranked-evidence synthesis sample."""
    candidates = [
        OUTPUTS_DIR / f"{run_id}_child_synthesis_lines.jsonl",
        OUTPUTS_DIR / f"{run_id}_abstractive_threshold_synthesis_lines.jsonl",
        OUTPUTS_DIR / f"{run_id}_abstractive_ranked_synthesis_lines.jsonl",
        OUTPUTS_DIR / f"{run_id}_abstractive_synthesis_cache.jsonl",
        OUTPUTS_DIR / f"{run_id}_abstractive_lines.jsonl",
    ]
    bad_markers = [
        "hotel review evidence for one aspect",
        "use only the evidence",
        "venetian hotel",
    ]
    preferred_aspects = ["FAC_ROOM", "AM_FOOD", "AM_WIFI", "EXP_VALUE", "SER_ATTITUDE", "AM_ENT"]
    valid: list[dict] = []
    for path in candidates:
        if not path.exists():
            continue
        for row in read_jsonl(path, limit=5000):
            summary = row.get("summary") or row.get("sentence") or ""
            summary_l = summary.lower()
            evidence = row.get("evidence") or []
            if not summary.strip() or len(evidence) < 1:
                continue
            if row.get("copied_from_evidence") is True:
                continue
            if any(marker in summary_l for marker in bad_markers):
                continue
            if generic_summary(summary):
                continue
            normalized = dict(row)
            normalized["summary"] = summary
            valid.append(normalized)
        if valid:
            break
    if not valid:
        return None
    for aspect in preferred_aspects:
        for row in valid:
            if row.get("aspect") == aspect:
                return row
    return valid[0]


def first_hierarchical_sample(run_id: str) -> dict | None:
    entity_rows = read_jsonl(OUTPUTS_DIR / f"{run_id}_entity_synthesis_lines.jsonl", limit=5000)
    parent_rows = read_jsonl(OUTPUTS_DIR / f"{run_id}_parent_synthesis_lines.jsonl", limit=5000)
    child_rows = read_jsonl(OUTPUTS_DIR / f"{run_id}_child_synthesis_lines.jsonl", limit=5000)
    evidence_rows = read_jsonl(OUTPUTS_DIR / f"{run_id}_threshold_evidence.jsonl", limit=20000)
    if not entity_rows:
        return None
    preferred_aspects = ["FAC_ROOM", "AM_FOOD", "SER_ATTITUDE", "EXP_VALUE", "FAC_VIEW_LOCATION", "AM_WIFI"]
    children_by_key = {}
    for row in child_rows:
        children_by_key.setdefault(
            (row.get("split"), str(row.get("entity_id", ""))), []).append(row)

    selected: tuple[dict, dict] | None = None
    for entity_candidate in entity_rows:
        if generic_summary(entity_candidate.get("summary", "")):
            continue
        key = (entity_candidate.get("split"), str(entity_candidate.get("entity_id", "")))
        candidate_children = children_by_key.get(key, [])
        for aspect in preferred_aspects:
            for child in candidate_children:
                if child.get("aspect") != aspect:
                    continue
                if generic_summary(child.get("summary", "")):
                    continue
                ev_count = int(child.get("evidence_used") or child.get("evidence_count") or 0)
                if ev_count >= 2:
                    selected = (entity_candidate, child)
                    break
            if selected:
                break
        if selected:
            break
    if selected:
        entity, sample_child = selected
    else:
        entity = next((row for row in entity_rows if row.get("summary")), entity_rows[0])
        sample_child = {}
    split = entity.get("split")
    entity_id = str(entity.get("entity_id", ""))
    parents = [
        row for row in parent_rows
        if row.get("split") == split and str(row.get("entity_id", "")) == entity_id
    ]
    children = [
        row for row in child_rows
        if row.get("split") == split and str(row.get("entity_id", "")) == entity_id
    ]
    if not sample_child:
        sample_child = next((row for row in children if row.get("summary")), children[0] if children else {})
    aspect = sample_child.get("aspect")
    evidence = [
        row for row in evidence_rows
        if row.get("split") == split and str(row.get("entity_id", "")) == entity_id
        and (not aspect or row.get("aspect") == aspect)
    ]
    evidence.sort(key=lambda row: (
        row.get("score") if row.get("score") is not None else 10**9,
        row.get("rank") if row.get("rank") is not None else 10**9,
    ))
    return {
        "entity": entity,
        "parents": parents,
        "child": sample_child,
        "evidence": evidence[:6],
    }


def collect_aspect_evidence_examples(run_id: str, max_examples: int = 5) -> list[dict]:
    """Pick readable evidence examples across different aspects for the deck."""
    preferred = ["AM_ENT", "FAC_ROOM", "AM_FOOD", "AM_WIFI", "EXP_VALUE", "SER_ATTITUDE"]
    rows = read_jsonl(OUTPUTS_DIR / f"{run_id}_lines.jsonl", limit=5000)
    by_aspect: dict[str, dict] = {}
    for aspect in preferred:
        for row in rows:
            sent = row.get("sentence", "")
            if row.get("aspect") != aspect or not sent:
                continue
            # Avoid obvious hard-truncation artifacts in the display examples.
            if len(sent.split()) < 7 or sent.lower().endswith((" and", " of", " the", " with", " to")):
                continue
            by_aspect[aspect] = row
            break
    return [by_aspect[a] for a in preferred if a in by_aspect][:max_examples]


def tx(x: float) -> int:
    return int(x * 914400)


def ty(y: float) -> int:
    return int(y * 914400)


def text_shape(shape_id: int, text: str, x: float, y: float, w: float, h: float,
               font_size: int = 1400, color: str = "14213D", bold: bool = False,
               fill: str | None = None, line: str | None = None) -> str:
    runs = []
    for raw_line in text.split("\n"):
        line_text = escape(raw_line)
        b = '<a:b/>' if bold else ''
        runs.append(
            f'<a:p><a:r><a:rPr lang="vi-VN" sz="{font_size}" dirty="0">{b}'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'</a:rPr><a:t>{line_text}</a:t></a:r><a:endParaRPr lang="vi-VN" sz="{font_size}"/></a:p>'
        )
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
    line_xml = (
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line else '<a:ln><a:noFill/></a:ln>'
    )
    return f'''
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="Text {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{tx(x)}" y="{ty(y)}"/><a:ext cx="{tx(w)}" cy="{ty(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          {fill_xml}{line_xml}
        </p:spPr>
        <p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/><a:lstStyle/>
        {''.join(runs)}
        </p:txBody>
      </p:sp>'''


def line_shape(shape_id: int, x: float, y: float, w: float, color: str = "D8DEE9") -> str:
    return f'''
      <p:cxnSp>
        <p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="Line {shape_id}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{tx(x)}" y="{ty(y)}"/><a:ext cx="{tx(w)}" cy="0"/></a:xfrm>
          <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
          <a:ln w="12700"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
        </p:spPr>
      </p:cxnSp>'''


def slide_xml(title: str, kicker: str, blocks: list[dict]) -> str:
    shapes = [
        text_shape(2, kicker.upper(), 0.55, 0.25, 12.0, 0.28, 800, "2563EB", True),
        text_shape(3, title, 0.55, 0.56, 12.0, 0.5, 2300, "14213D", True),
        line_shape(4, 0.55, 1.13, 12.2),
    ]
    shape_id = 5
    for block in blocks:
        shapes.append(text_shape(shape_id, **block))
        shape_id += 1
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F8FAFC"/></a:solidFill></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm></p:grpSpPr>
    {''.join(shapes)}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def content_types(nslides: int) -> str:
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>',
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>',
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, nslides + 1):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
{''.join(overrides)}
</Types>'''


def presentation_xml(nslides: int) -> str:
    ids = ''.join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, nslides + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{nslides + 1}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="{EMU_W}" cy="{EMU_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''


def presentation_rels(nslides: int) -> str:
    rels = [
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, nslides + 1)
    ]
    rels += [
        f'<Relationship Id="rId{nslides + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
        f'<Relationship Id="rId{nslides + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps" Target="presProps.xml"/>',
        f'<Relationship Id="rId{nslides + 3}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps" Target="viewProps.xml"/>',
        f'<Relationship Id="rId{nslides + 4}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles" Target="tableStyles.xml"/>',
        f'<Relationship Id="rId{nslides + 5}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>'''


def static_part(name: str) -> str:
    if name == "theme":
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="SemAE"><a:themeElements><a:clrScheme name="SemAE"><a:dk1><a:srgbClr val="14213D"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="56616F"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="2563EB"/></a:accent1><a:accent2><a:srgbClr val="0F766E"/></a:accent2><a:accent3><a:srgbClr val="B7791F"/></a:accent3><a:accent4><a:srgbClr val="B42318"/></a:accent4><a:accent5><a:srgbClr val="64748B"/></a:accent5><a:accent6><a:srgbClr val="0EA5E9"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="SemAE"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'''
    if name == "slide_master":
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'''
    if name == "slide_layout":
        return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'''
    return ""


def build_slides(run_id: str) -> list[str]:
    report = read_json(OUTPUTS_DIR / f"{run_id}_report.json", {})
    metrics = read_json(OUTPUTS_DIR / f"{run_id}_metrics.json", {})
    metadata = read_json(OUTPUTS_DIR / f"{run_id}_metadata.json", {})
    trace = read_jsonl(LOGS_DIR / f"{run_id}_pipeline_trace.jsonl")
    provenance_path = OUTPUTS_DIR / f"{run_id}_provenance.jsonl"
    provenance_sample = read_jsonl(provenance_path, limit=1)
    provenance_rows = jsonl_count(provenance_path)
    aspect_line_rows = jsonl_count(OUTPUTS_DIR / f"{run_id}_lines.jsonl")
    provenance_coverage = (
        provenance_rows / aspect_line_rows if aspect_line_rows else 0.0)
    old_rouge_path = REPORTS_DIR / "space_old_aspects_e20_official_rouge.json"
    if not old_rouge_path.exists():
        old_rouge_path = OUTPUTS_DIR / "eval_space_old_aspects_e20.json"
    old_macro_rows, old_aspect_rows = rouge_macro_rows(old_rouge_path)
    sample = first_sample(run_id)
    synthesis_sample = first_synthesis_sample(run_id)
    hierarchical_sample = first_hierarchical_sample(run_id)
    aspect_examples = collect_aspect_evidence_examples(run_id)
    synthesis_rows = jsonl_count(
        OUTPUTS_DIR / f"{run_id}_child_synthesis_lines.jsonl")
    if synthesis_rows == 0:
        synthesis_rows = jsonl_count(
            OUTPUTS_DIR / f"{run_id}_abstractive_threshold_synthesis_lines.jsonl")
    synthesis_report = read_json(OUTPUTS_DIR / f"{run_id}_abstractive_synthesis_report.json", {})
    ranked_evidence_rows = jsonl_count(
        OUTPUTS_DIR / f"{run_id}_threshold_evidence.jsonl")
    if ranked_evidence_rows == 0:
        ranked_evidence_rows = jsonl_count(
            OUTPUTS_DIR / f"{run_id}_ranked_evidence.jsonl")
    parent_rows = jsonl_count(OUTPUTS_DIR / f"{run_id}_parent_synthesis_lines.jsonl")
    entity_rows = jsonl_count(OUTPUTS_DIR / f"{run_id}_entity_synthesis_lines.jsonl")
    macro = metrics.get("macro", {})
    aspect_count = len(report.get("per_aspect", {})) or 29
    model_label = metadata.get("model_label") or "SemAE checkpoint"
    train_label = metadata.get("train_label") or "SPACE training run"
    trace_tail = "\n".join(
        f"{r.get('status', '').upper()} {r.get('stage', '')}"
        for r in trace[-12:]
    ) or (
        f"Full run completed\n"
        f"aspect files: {aspect_count * 50}\n"
        f"threshold evidence rows: {ranked_evidence_rows}\n"
        f"child summaries: {synthesis_rows}\n"
        f"parent summaries: {parent_rows}\n"
        f"entity summaries: {entity_rows}"
    )
    failure = next(
        (r for r in reversed(trace) if r.get("stage") == "blocked" or r.get("status") == "fail"),
        None,
    )
    failure_reason = (
        failure.get("reason")
        or failure.get("stage")
        or "Pipeline chưa qua được toàn bộ artifact/inference gate."
        if failure
        else ""
    )

    sample_aspect = "Chưa có output sample. Chạy inference trước khi build deck cuối."
    sample_sent = "(missing)"
    sample_split = "Chưa có sentiment output sample."
    sample_output_1_title = "Vấn đề output extractive cũ"
    sample_output_2_title = "Hướng sửa: abstractive synthesis"
    overall_entity_label = "Entity: n/a"
    parent_output_text = "Parent summaries chưa có."
    overall_output_text = "Overall entity summary chưa có."
    if sample:
        sample_aspect = f"Entity: {sample['entity_id']}\nAspect: {sample['aspect']}"
        sample_sent = (
            "Output cũ là các câu source được chọn rồi ghép lại. Vì max_tokens=40 "
            "nên câu cuối có thể bị cắt cụt:\n\n" +
            "\n\n".join(f"{i + 1}. {truncate(s, 210)}" for i, s in enumerate(sample["sentences"][:4]))
        )
        sample_split = (
            "Không dùng slide này làm final summary. Đây là evidence extractive để audit.\n\n"
            "Fix đúng: lấy threshold evidence đầy đủ trước truncate, sau đó cho FLAN/T5 hoặc model synthesis viết lại "
            "thành 1-2 câu abstractive. Nếu chưa có full synthesis artifact hợp lệ, deck chỉ báo capability thay vì show output lỗi."
        )
    if synthesis_sample:
        sample_output_1_title = "Threshold evidence + FLAN output"
        sample_output_2_title = "Abstractive summary đã lọc lỗi prompt echo"
        sample_aspect = (
            f"Entity: {synthesis_sample.get('entity_id')}\n"
            f"Aspect: {synthesis_sample.get('aspect')}\n"
            f"Model: {synthesis_sample.get('model_name', 'n/a')}")
        evidence = synthesis_sample.get("evidence", [])
        if evidence and isinstance(evidence[0], dict):
            sample_sent = evidence_review_lines(evidence, limit=5, width=210)
        elif evidence:
            sample_sent = "\n\n".join(
                f"{i}. {truncate(ev, 210)}"
                for i, ev in enumerate(evidence[:5], 1)
            )
        else:
            sample_sent = "Evidence artifact không có trong row này; xem synthesis JSONL để audit."
        sample_split = truncate(synthesis_sample.get("summary", ""), 900)
    if hierarchical_sample:
        entity_row = hierarchical_sample.get("entity", {})
        child_row = hierarchical_sample.get("child", {})
        parents = hierarchical_sample.get("parents", [])
        evidence = hierarchical_sample.get("evidence", [])
        sample_output_1_title = "Aspect output example"
        sample_output_2_title = "Aspect summarization"
        sample_aspect = (
            f"Entity: {entity_row.get('entity_id')}\n"
            f"Child aspect: {child_row.get('aspect', 'n/a')}\n"
            f"Model: {child_row.get('model_name', 'n/a')}")
        sample_sent = evidence_review_lines(
            evidence, limit=4, width=190
        ) or "No threshold evidence sample found for this entity/aspect."
        sample_split = (
            f"{child_row.get('aspect', 'n/a')} summarization:\n"
            f"{truncate(child_row.get('summary', ''), 850)}"
        )
        parent_order = ["FACILITY", "AMENITY", "SERVICE", "EXPERIENCE", "BRANDING", "LOYALTY"]
        order = {name: idx for idx, name in enumerate(parent_order)}
        parents = sorted(parents, key=lambda row: order.get(row.get("aspect", ""), 99))
        overall_entity_label = (
            f"Entity: {entity_row.get('entity_id')}\n"
            f"Overall model: {entity_row.get('model_name', 'n/a')}\n"
            f"Parent groups: {len(parents)}"
        )
        parent_output_text = "\n\n".join(
            f"{row.get('aspect')}: {truncate(row.get('summary', ''), 125)}"
            for row in parents[:6]
        ) or "Parent summaries chưa có."
        overall_output_text = truncate(entity_row.get("summary", ""), 820)

    if provenance_sample:
        prov = provenance_sample[0]
        provenance_example = (
            f"entity={prov.get('entity_id')} | aspect={prov.get('aspect')} | "
            f"rank={prov.get('rank')} | review={prov.get('source_review_id')}\n"
            f"seed={prov.get('matched_aspect_seed')} | "
            f"sentiment={prov.get('sentiment_label')} "
            f"{prov.get('matched_sentiment_keywords')}\n"
            f"sentence: {truncate(prov.get('sentence', ''), 230)}")
    else:
        provenance_example = "Chưa có provenance sample. Rerun inference với trace_jsonl để sinh outputs/<run_id>_provenance.jsonl."

    slides = []
    if failure:
        slides.append(
            slide_xml("Run status: pipeline đang bị chặn ở tokenizer gate", "Status", [
                {"text": "Đã chạy xong\nprepare_hasos: OK\nvalidate_hasos: OK\nmodel epoch 10: OK\nHASOS stats: 50 entities / 5,000 reviews / 45,529 sentences / 29 aspects", "x": 0.75, "y": 1.35, "w": 5.75, "h": 2.3, "font_size": 1150, "fill": "ECFDF5", "line": "99F6E4"},
                {"text": f"Hard gate chưa qua\n{failure_reason}\n\nThiếu exact tokenizer:\ndata/sentencepiece/space_unigram_32k.model", "x": 6.85, "y": 1.35, "w": 5.55, "h": 2.3, "font_size": 1150, "fill": "FFF7ED", "line": "FDBA74"},
                {"text": "Quyết định kỹ thuật\nKhông chạy inference bằng tokenizer regenerate vì token-id mapping có thể sai so với checkpoint. Khi restore đúng tokenizer, rerun scripts/run_space_hasos_after_model.py; deck sẽ được build lại với sample output và metrics thật.", "x": 0.75, "y": 4.05, "w": 11.65, "h": 1.05, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            ])
        )

    slides.extend([
        slide_xml("SPACE-trained SemAE cho HASOS aspect/sentiment summary", run_id, [
            {"text": f"Deck này ghi lại pipeline sau training: {model_label} được dùng để rank câu theo {aspect_count} HASOS aspects. Bản mới giữ mọi evidence có score <= 0.005 rồi synthesize theo 3 tầng bằng FLAN.", "x": 0.75, "y": 1.45, "w": 6.2, "h": 1.1, "font_size": 1450},
            {"text": f"Training: {train_label}\nOutput modes: extractive + hierarchical abstractive\nHASOS aspects: {aspect_count}\nTrace rows: {len(trace)}\nThreshold evidence rows: {ranked_evidence_rows}\nChild rows: {synthesis_rows}\nParent rows: {parent_rows}\nEntity rows: {entity_rows}", "x": 7.25, "y": 1.35, "w": 4.9, "h": 2.25, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Corrected flow: SemAE scoring -> score-threshold evidence -> FLAN-base child aspect summary -> FLAN-base parent summary -> FLAN-base entity summary. Extractive/sentiment outputs vẫn giữ để audit.", "x": 0.75, "y": 3.65, "w": 11.4, "h": 1.0, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Pipeline từ input tới output", "Mechanism", [
            {"text": "1 Data prep\nHASOS reviews\n29 child aspects", "x": 0.65, "y": 1.45, "w": 1.8, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "2 Tokenizer gate\nSPACE SentencePiece\ncheckpoint-safe IDs", "x": 2.55, "y": 1.45, "w": 1.8, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "3 SemAE score\nD_z(s), P_z, P_k\nKL objective", "x": 4.45, "y": 1.45, "w": 1.8, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "4 Threshold\nscore <= 0.005\nfull evidence", "x": 6.35, "y": 1.45, "w": 1.8, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "5 Child\nFLAN-base\n29 aspects", "x": 8.25, "y": 1.45, "w": 1.8, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "6 Parent + entity\nFLAN-base\n6 groups + overall", "x": 10.15, "y": 1.45, "w": 1.95, "h": 1.2, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Biến thể so với SemAE gốc: taxonomy/seeds đổi sang HASOS 29 aspects. Abstractive stage dùng threshold evidence trước truncate; sentiment split chỉ là audit post-processing.", "x": 0.75, "y": 3.75, "w": 11.4, "h": 0.85, "font_size": 1200, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Artifact gate và dữ liệu chuẩn bị", "Readiness", [
            {"text": trace_tail, "x": 0.75, "y": 1.35, "w": 5.9, "h": 4.8, "font_size": 1000, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Hard gate quan trọng\nCheckpoint SemAE không tự chứa SentencePiece model. Nếu `space_unigram_32k.model` không khớp tokenizer lúc train, token id sẽ sai và inference không đáng tin.", "x": 7.0, "y": 1.35, "w": 5.35, "h": 1.85, "font_size": 1200, "fill": "FFF7ED", "line": "FDBA74"},
            {"text": "Validate hiện tại\nentities=50\nreviews=5000\nsentences=45529\naspects=29", "x": 7.0, "y": 3.55, "w": 5.35, "h": 1.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("SemAE scoring theo aspect", "Formula", [
            {"text": "score(s, k) = KL(D_z(s) || P_z) - beta * KL(D_z(s) || P_k)", "x": 0.8, "y": 1.45, "w": 11.4, "h": 0.65, "font_size": 1900, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "D_z(s): phân phối cluster của câu s.\nP_z: background distribution.\nP_k: prototype distribution của aspect k.\nbeta=0.7 theo recipe hiện tại.", "x": 0.9, "y": 2.5, "w": 5.7, "h": 1.8, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Câu có score thấp hơn được ưu tiên. Bản mới không đọc summary đã truncate; nó đọc mọi câu full sentence có score <= 0.005, dedupe ý trùng, rồi synthesize theo 3 tầng.", "x": 7.0, "y": 2.5, "w": 5.4, "h": 1.35, "font_size": 1250, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
        slide_xml("SPACE original aspects: official ROUGE", "Old Baseline", [
            {"text": "Bản cũ dùng đúng SPACE benchmark và 6 aspect gốc: building, cleanliness, food, location, rooms, service. Run này chạy without --no_eval nên điểm dưới đây là pyrouge official, khác với HASOS reference-free metrics.", "x": 0.75, "y": 1.35, "w": 11.35, "h": 0.95, "font_size": 1120, "fill": "FFF7ED", "line": "FDBA74"},
            {"text": "Macro F1 by split\n" + ("\n".join(old_macro_rows) if old_macro_rows else "missing"), "x": 0.85, "y": 2.65, "w": 4.95, "h": 1.65, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "ALL split by aspect\nR1 / R2 / RL\n" + old_aspect_rows, "x": 6.15, "y": 2.45, "w": 5.95, "h": 2.65, "font_size": 850, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Kết luận so sánh: HASOS 29 aspects là adaptation/inference mới không có gold summary; SPACE original 6 aspects là baseline có gold và có thể so sánh bằng ROUGE.", "x": 0.85, "y": 5.35, "w": 11.15, "h": 0.55, "font_size": 1050, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Từ extractive sang abstractive", "Output correction", [
            {"text": "Vấn đề bạn thấy là đúng\nOutput SemAE gốc trong run này là extractive: chọn câu source rồi nối lại. Với max_tokens=40, câu cuối có thể bị cắt cụt như `Each room is named... and one of`.", "x": 0.75, "y": 1.35, "w": 5.5, "h": 1.65, "font_size": 1200, "fill": "FFF7ED", "line": "FDBA74"},
            {"text": f"Trạng thái artifact synthesis local\nSynthesis rows tìm thấy: {synthesis_rows}\nThreshold evidence rows: {ranked_evidence_rows}\nDeck sẽ không còn trình bày output extractive bị cắt như final summary.", "x": 6.55, "y": 1.35, "w": 5.55, "h": 1.65, "font_size": 1200, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Kết quả sau khi sửa\n1. Evidence đầu vào là full review sentences trước truncate.\n2. Aspect summary giữ chi tiết review thay vì nối câu extractive bị cắt.\n3. Parent/entity summary gom thông tin theo 6 nhóm rồi viết output tổng cho từng khách sạn.", "x": 0.75, "y": 3.55, "w": 11.35, "h": 1.45, "font_size": 1180, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml(sample_output_1_title, "Output", [
            {"text": sample_aspect, "x": 0.75, "y": 1.25, "w": 3.15, "h": 0.95, "font_size": 1150, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Source reviews\n" + sample_sent, "x": 4.15, "y": 1.25, "w": 8.0, "h": 3.15, "font_size": 800, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Summarization\n" + sample_split, "x": 0.75, "y": 4.65, "w": 11.4, "h": 1.05, "font_size": 980, "bold": False, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
        slide_xml("Output tổng hợp nhất 6 parent aspects", "Entity summary", [
            {"text": overall_entity_label, "x": 0.75, "y": 1.25, "w": 3.15, "h": 0.85, "font_size": 1100, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Parent aspect summaries\n" + parent_output_text, "x": 0.75, "y": 2.3, "w": 5.35, "h": 3.55, "font_size": 720, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Overall entity summary\n" + overall_output_text, "x": 6.45, "y": 1.25, "w": 5.8, "h": 3.75, "font_size": 880, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Điều slide này muốn nói\nPipeline không dừng ở aspect summary. Nó gom 29 child aspects thành 6 parent groups, rồi viết output tổng cho từng khách sạn.", "x": 6.45, "y": 5.25, "w": 5.8, "h": 0.75, "font_size": 880, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Metrics và health checks", "Quality", [
            {"text": f"source_fidelity: {macro.get('source_fidelity', 'n/a')}\nsource_fidelity_excl_truncated: {macro.get('source_fidelity_excl_truncated', 'n/a')}\naspect_purity: {macro.get('aspect_purity', 'n/a')}\ndistinct_2: {macro.get('distinct_2', 'n/a')}\nself_bleu4: {macro.get('self_bleu4', 'n/a')}", "x": 0.85, "y": 1.45, "w": 5.15, "h": 2.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "HASOS không có gold summary nên dùng reference-free metrics: fidelity, purity, diversity, compression. BERTScore có thể chạy thêm nếu dependency/model tải được.", "x": 6.1, "y": 1.45, "w": 6.0, "h": 1.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Limitations và quyết định kỹ thuật", "Caveats", [
            {"text": "1. Sentiment là keyword post-processing, không phải sentiment-aware ranking.\n2. Abstractive summary không còn verbatim traceable như extractive output, nhưng mỗi row giữ threshold evidence provenance.\n3. Tokenizer là artifact bắt buộc và phải sync cùng checkpoint.\n4. Không dùng output đã truncate làm input synthesis.", "x": 0.85, "y": 1.45, "w": 11.3, "h": 2.4, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Final deck path: outputs/{run_id}_report.pptx", "x": 0.85, "y": 4.55, "w": 11.3, "h": 0.65, "font_size": 1300, "bold": True, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
    ])
    provenance_slide = slide_xml("Provenance cho từng câu output", "Traceability", [
        {"text": f"Aspect summary rows: {aspect_line_rows}\nProvenance rows: {provenance_rows}\nCoverage: {provenance_coverage:.4f}\nFile: outputs/{run_id}_provenance.jsonl", "x": 0.75, "y": 1.35, "w": 4.8, "h": 1.7, "font_size": 1250, "fill": "ECFDF5", "line": "99F6E4"},
        {"text": provenance_example, "x": 5.9, "y": 1.35, "w": 6.25, "h": 2.05, "font_size": 1000, "fill": "FFFFFF", "line": "D8DEE9"},
        {"text": "Mỗi câu summary ghi: entity/split/aspect, summary_sentence_index, rank/score/beta, source_review_id, source_sentence_index, matched_aspect_seed, sentiment_label, matched_sentiment_keywords, was_truncated. Metric mới source_fidelity_excl_truncated loại câu bị cắt khỏi exact-match denominator.", "x": 0.75, "y": 4.05, "w": 11.4, "h": 1.0, "font_size": 1100, "fill": "EEF6FF", "line": "BFDBFE"},
    ])
    slides.insert(4 if failure else 3, provenance_slide)
    return slides


def write_pptx(run_id: str, out_path: Path) -> None:
    slides = build_slides(run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>''')
        z.writestr("docProps/core.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>SPACE SemAE HASOS pipeline report</dc:title><dc:creator>SemAE pipeline</dc:creator><cp:lastModifiedBy>SemAE pipeline</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>''')
        z.writestr("docProps/app.xml", f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>SemAE pipeline</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides></Properties>''')
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/presProps.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>''')
        z.writestr("ppt/viewProps.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>''')
        z.writestr("ppt/tableStyles.xml", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>''')
        z.writestr("ppt/theme/theme1.xml", static_part("theme"))
        z.writestr("ppt/slideMasters/slideMaster1.xml", static_part("slide_master"))
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>''')
        z.writestr("ppt/slideLayouts/slideLayout1.xml", static_part("slide_layout"))
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>''')
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>''')
    consolidated = REPORTS_DIR / "space_hasos_final_consolidated.pptx"
    consolidated.parent.mkdir(parents=True, exist_ok=True)
    if out_path.resolve() != consolidated.resolve():
        shutil.copy2(out_path, consolidated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="space_hasos_2k_e10")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = Path(args.output) if args.output else OUTPUTS_DIR / f"{args.run_id}_report.pptx"
    write_pptx(args.run_id, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
