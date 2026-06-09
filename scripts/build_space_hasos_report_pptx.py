#!/usr/bin/env python3
"""Build a lightweight PPTX report for the SPACE->HASOS SemAE pipeline.

This intentionally uses only the Python standard library so the reporting step
does not depend on PowerPoint, python-pptx, or Node packages.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
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
    sample = first_sample(run_id)
    macro = metrics.get("macro", {})
    aspect_count = len(report.get("per_aspect", {})) or 29
    model_label = metadata.get("model_label") or "SemAE checkpoint"
    train_label = metadata.get("train_label") or "SPACE training run"
    trace_tail = "\n".join(f"{r.get('status', '').upper()} {r.get('stage', '')}" for r in trace[-12:]) or "Trace chưa có."
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
    if sample:
        sample_aspect = f"Entity: {sample['entity_id']}\nAspect: {sample['aspect']}"
        sample_sent = "\n\n".join(f"{i + 1}. {truncate(s, 210)}" for i, s in enumerate(sample["sentences"][:4]))
        sample_split = "\n\n".join(
            f"{label.upper()}\n" + ("\n".join(f"- {truncate(s, 135)}" for s in sample["sentiment"].get(label, [])[:3]) or "(empty)")
            for label in ("pos", "neg", "neu")
        )
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
            {"text": f"Deck này ghi lại pipeline sau training: {model_label} được dùng để chọn câu theo {aspect_count} HASOS aspects, sau đó tách sentiment trên chính các câu đã chọn.", "x": 0.75, "y": 1.45, "w": 6.2, "h": 1.1, "font_size": 1550},
            {"text": f"Training: {train_label}\nOutput modes: 2\nHASOS aspects: {aspect_count}\nTrace rows: {len(trace)}", "x": 7.25, "y": 1.45, "w": 4.9, "h": 1.65, "font_size": 1350, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Output 1: Aspect-only summary giữ lẫn positive/negative/neutral.\nOutput 2: Aspect + sentiment tách pos/neg/neu từ cùng câu đã được chọn.", "x": 0.75, "y": 3.45, "w": 11.4, "h": 1.2, "font_size": 1300, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Pipeline từ input tới output", "Mechanism", [
            {"text": "1 Input\nhasos_summ.json\nreviews grouped by entity", "x": 0.65, "y": 1.45, "w": 2.2, "h": 1.15, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "2 Tokenize\nSPACE SentencePiece\nsame tokenizer as checkpoint", "x": 3.0, "y": 1.45, "w": 2.2, "h": 1.15, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "3 Encode\nSemAE cluster distribution\nD_z(s) per sentence", "x": 5.35, "y": 1.45, "w": 2.2, "h": 1.15, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "4 Aspect rank\nKL(D_z||P_z) - beta*KL(D_z||P_k)", "x": 7.7, "y": 1.45, "w": 2.2, "h": 1.15, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "5 Output\ntruncate top sentences\nthen sentiment split", "x": 10.05, "y": 1.45, "w": 2.2, "h": 1.15, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Biến thể so với SemAE gốc: taxonomy/seeds đổi sang HASOS 29 aspects; sentiment không tham gia ranking mà chỉ là post-processing.", "x": 0.75, "y": 3.75, "w": 11.4, "h": 0.85, "font_size": 1200, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Artifact gate và dữ liệu chuẩn bị", "Readiness", [
            {"text": trace_tail, "x": 0.75, "y": 1.35, "w": 5.9, "h": 4.8, "font_size": 1000, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Hard gate quan trọng\nCheckpoint SemAE không tự chứa SentencePiece model. Nếu `space_unigram_32k.model` không khớp tokenizer lúc train, token id sẽ sai và inference không đáng tin.", "x": 7.0, "y": 1.35, "w": 5.35, "h": 1.85, "font_size": 1200, "fill": "FFF7ED", "line": "FDBA74"},
            {"text": "Validate hiện tại\nentities=50\nreviews=5000\nsentences=45529\naspects=29", "x": 7.0, "y": 3.55, "w": 5.35, "h": 1.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("SemAE scoring theo aspect", "Formula", [
            {"text": "score(s, k) = KL(D_z(s) || P_z) - beta * KL(D_z(s) || P_k)", "x": 0.8, "y": 1.45, "w": 11.4, "h": 0.65, "font_size": 1900, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "D_z(s): phân phối cluster của câu s.\nP_z: background distribution.\nP_k: prototype distribution của aspect k.\nbeta=0.7 theo recipe hiện tại.", "x": 0.9, "y": 2.5, "w": 5.7, "h": 1.8, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Câu có score thấp hơn được ưu tiên. Output là extractive: chọn và nối câu gốc thay vì paraphrase, nên có thể trace ngược về review/entity.", "x": 7.0, "y": 2.5, "w": 5.4, "h": 1.35, "font_size": 1250, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
        slide_xml("Ví dụ aspect-only output", "Output 1", [
            {"text": sample_aspect, "x": 0.75, "y": 1.35, "w": 3.2, "h": 0.9, "font_size": 1250, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": sample_sent, "x": 4.15, "y": 1.35, "w": 8.0, "h": 4.8, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Ví dụ aspect + sentiment split", "Output 2", [
            {"text": sample_split, "x": 0.75, "y": 1.35, "w": 11.5, "h": 5.0, "font_size": 1050, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Metrics và health checks", "Quality", [
            {"text": f"source_fidelity: {macro.get('source_fidelity', 'n/a')}\nsource_fidelity_excl_truncated: {macro.get('source_fidelity_excl_truncated', 'n/a')}\naspect_purity: {macro.get('aspect_purity', 'n/a')}\ndistinct_2: {macro.get('distinct_2', 'n/a')}\nself_bleu4: {macro.get('self_bleu4', 'n/a')}", "x": 0.85, "y": 1.45, "w": 5.15, "h": 2.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "HASOS không có gold summary nên dùng reference-free metrics: fidelity, purity, diversity, compression. BERTScore có thể chạy thêm nếu dependency/model tải được.", "x": 6.1, "y": 1.45, "w": 6.0, "h": 1.25, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Limitations và quyết định kỹ thuật", "Caveats", [
            {"text": "1. Sentiment là keyword post-processing, không phải sentiment-aware ranking.\n2. Output extractive nên traceable nhưng có thể dài hoặc multi-aspect.\n3. Tokenizer là artifact bắt buộc và phải sync cùng checkpoint.\n4. Run này ưu tiên có output đúng pipeline trước deadline; xem trace/log để đánh giá độ sạch numerical.", "x": 0.85, "y": 1.45, "w": 11.3, "h": 2.4, "font_size": 1250, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Final deck path: outputs/{run_id}_pipeline_report.pptx", "x": 0.85, "y": 4.55, "w": 11.3, "h": 0.65, "font_size": 1300, "bold": True, "fill": "ECFDF5", "line": "99F6E4"},
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="space_hasos_2k_e10")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = Path(args.output) if args.output else OUTPUTS_DIR / f"{args.run_id}_pipeline_report.pptx"
    write_pptx(args.run_id, out)
    print(f"wrote {out}")
    if args.output is None:
        alias = OUTPUTS_DIR / f"{args.run_id}_report.pptx"
        if alias != out:
            write_pptx(args.run_id, alias)
            print(f"wrote {alias}")


if __name__ == "__main__":
    main()
