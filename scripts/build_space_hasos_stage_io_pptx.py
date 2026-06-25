#!/usr/bin/env python3
"""Build a stage-by-stage input/output deck for SPACE -> HASOS SemAE."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from build_space_hasos_report_pptx import (
    LOGS_DIR,
    OUTPUTS_DIR,
    content_types,
    presentation_rels,
    presentation_xml,
    slide_xml,
    static_part,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def read_jsonl(path: Path) -> list[dict]:
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
    return rows


def first(rows: list[dict], stage: str, **where) -> dict:
    for row in rows:
        if row.get("stage") != stage:
            continue
        if all(row.get(k) == v for k, v in where.items()):
            return row
    return {}


def truncate(text: str, n: int = 180) -> str:
    text = " ".join(str(text).replace("\t", " ").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def write_deck(slides: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""",
        )
        z.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>SPACE HASOS stage I/O report</dc:title><dc:creator>SemAE pipeline</dc:creator><cp:lastModifiedBy>SemAE pipeline</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>""",
        )
        z.writestr(
            "docProps/app.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>SemAE pipeline</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""",
        )
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/presProps.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentationPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>""")
        z.writestr("ppt/viewProps.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:viewPr xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>""")
        z.writestr("ppt/tableStyles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>""")
        z.writestr("ppt/theme/theme1.xml", static_part("theme"))
        z.writestr("ppt/slideMasters/slideMaster1.xml", static_part("slide_master"))
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""")
        z.writestr("ppt/slideLayouts/slideLayout1.xml", static_part("slide_layout"))
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""")
        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>""")


def build_slides(run_id: str) -> list[str]:
    data = read_json(DATA_DIR / "hasos" / "hasos_summ.json", [])
    metrics = read_json(OUTPUTS_DIR / f"{run_id}_metrics.json", {})
    trace = read_jsonl(LOGS_DIR / f"{run_id}__shard0.trace.jsonl")
    first_entity = data[0] if data else {}
    first_review = (first_entity.get("reviews") or [{}])[0]
    entity_encoded = first(trace, "entity_encoded", entity_id="100597")
    model_loaded = first(trace, "model_loaded")
    dataset_prepared = first(trace, "dataset_prepared")
    rank_rows = [
        row for row in trace
        if row.get("stage") == "ranked_sentence_sample"
        and row.get("entity_id") == "100597"
        and row.get("aspect") in {"FAC_ROOM", "AM_POOL", "AM_FOOD"}
    ][:6]
    summary = first(trace, "summary_written", entity_id="100597", aspect="FAC_ROOM")
    sent_pos = first(trace, "sentiment_written", entity_id="100597", aspect="FAC_ROOM", sentiment="pos")
    sent_neg = first(trace, "sentiment_written", entity_id="100597", aspect="FAC_ROOM", sentiment="neg")
    sent_neu = first(trace, "sentiment_written", entity_id="100597", aspect="FAC_ROOM", sentiment="neu")

    aspect_files = file_count(OUTPUTS_DIR / run_id)
    sentiment_files = file_count(OUTPUTS_DIR / f"{run_id}_sentiment")
    line_rows = line_count(OUTPUTS_DIR / f"{run_id}_lines.jsonl")
    sentiment_rows = line_count(OUTPUTS_DIR / f"{run_id}_aspect_sentiment_lines.jsonl")
    provenance = read_jsonl(OUTPUTS_DIR / f"{run_id}_provenance.jsonl")
    provenance_rows = len(provenance)
    macro = metrics.get("macro", {})

    sample_rank = "\n".join(
        f"{r.get('aspect')} #{r.get('rank')}: {truncate(r.get('sentence'), 115)}"
        for r in rank_rows[:5]
    )
    sample_review = "\n".join(
        f"- {truncate(s, 120)}"
        for s in (first_review.get("sentences") or [])[:4]
    )
    sample_summary = "\n".join(
        f"- {truncate(s, 120)}"
        for s in str(summary.get("summary", "")).split("\t")[:4]
        if s
    )
    if provenance:
        first_prov = provenance[0]
        provenance_sample = (
            f"entity={first_prov.get('entity_id')} aspect={first_prov.get('aspect')} "
            f"idx={first_prov.get('summary_sentence_index')}\n"
            f"review={first_prov.get('source_review_id')} "
            f"sent_idx={first_prov.get('source_sentence_index')} "
            f"rank={first_prov.get('rank')} score={first_prov.get('score')}\n"
            f"seed={first_prov.get('matched_aspect_seed')} "
            f"sentiment={first_prov.get('sentiment_label')} "
            f"{first_prov.get('matched_sentiment_keywords')}")
    else:
        provenance_sample = "No provenance file found yet."

    slides = [
        slide_xml("SPACE -> HASOS SemAE: input/output từng stage", run_id, [
            {"text": "Deck này mô tả dữ liệu đi vào và artifact đi ra ở từng stage. Nó không phải deck metric thuần túy.", "x": 0.75, "y": 1.45, "w": 6.3, "h": 1.1, "font_size": 1500},
            {"text": f"Run đã xử lý\nentities: {len(data)}\naspects: 29\naspect files: {aspect_files}\nsentiment files: {sentiment_files}", "x": 7.35, "y": 1.45, "w": 4.75, "h": 1.85, "font_size": 1300, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Cảnh báo so sánh\nRepo gốc SemAE chấm bằng ROUGE + gold summaries. HASOS không có layout gold theo SemAE gốc, nhưng báo cáo hiện tại có thể chấm ROUGE ở cấp parent từ data/hasos/hasos_summ.json.", "x": 0.75, "y": 4.0, "w": 11.35, "h": 1.05, "font_size": 1250, "fill": "FFF7ED", "line": "FDBA74"},
        ]),
        slide_xml("SemAE gốc chấm điểm như thế nào", "Original Metric Contract", [
            {"text": "Input cần có\n1. system summaries trong outputs/<run_id>/\n2. gold summaries trong data/<dataset>/gold/<aspect>/\n3. pyrouge + ROUGE data setup", "x": 0.75, "y": 1.35, "w": 5.2, "h": 1.95, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Xử lý\nsrc/aspect_inference.py tạo RougeEvaluator cho dev/test/all, rồi gọi pyrouge convert_and_evaluate().", "x": 6.3, "y": 1.35, "w": 5.85, "h": 1.2, "font_size": 1200, "fill": "EEF6FF", "line": "BFDBFE"},
            {"text": "Output đúng repo gốc\noutputs/eval_<run_id>.txt\noutputs/eval_<run_id>.json\nROUGE-1 / ROUGE-2 / ROUGE-L", "x": 6.3, "y": 2.9, "w": 5.85, "h": 1.45, "font_size": 1200, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Trạng thái HASOS\ndata/hasos hiện chỉ có review JSON + taxonomy. Không có data/hasos/gold, vì vậy không thể tạo điểm ROUGE comparable cho run này.", "x": 0.75, "y": 4.55, "w": 11.4, "h": 0.9, "font_size": 1200, "fill": "FFF7ED", "line": "FDBA74"},
        ]),
        slide_xml("Stage 0: Prepare + validate HASOS", "Input -> Prepared Dataset", [
            {"text": f"Input JSON\nentity_id: {first_entity.get('entity_id')}\nentity_name: {first_entity.get('entity_name')}\nreviews/entity: {len(first_entity.get('reviews', []))}\nfirst review rating: {first_review.get('rating')}", "x": 0.75, "y": 1.35, "w": 4.8, "h": 1.8, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"First review sentences\n{sample_review}", "x": 5.85, "y": 1.35, "w": 6.25, "h": 2.45, "font_size": 950, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Output\n- data/hasos/hasos_summ.json\n- data/hasos/aspect_taxonomy.json\n- data/seeds_hasos/*.txt\n- validate: 50 entities / 5000 reviews / 45529 sentences / 29 aspects", "x": 0.75, "y": 4.2, "w": 11.35, "h": 1.05, "font_size": 1150, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
        slide_xml("Stage 1: Artifact gate", "Tokenizer + Checkpoint", [
            {"text": "Input artifacts\n- SPACE SentencePiece model\n- epoch-20 SemAE checkpoint\n- HASOS seeds/taxonomy\n- run config: max_tokens=40, shards=4", "x": 0.75, "y": 1.4, "w": 5.35, "h": 1.65, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Dataset output\nvocab_size: {dataset_prepared.get('vocab_size')}\npad/bos/eos/unk: {dataset_prepared.get('pad_id')}/{dataset_prepared.get('bos_id')}/{dataset_prepared.get('eos_id')}/{dataset_prepared.get('unk_id')}", "x": 6.45, "y": 1.4, "w": 5.5, "h": 1.25, "font_size": 1200, "fill": "EEF6FF", "line": "BFDBFE"},
            {"text": f"Model output\nnheads: {model_loaded.get('nheads')}\ncodebook_size: {model_loaded.get('codebook_size')}\nd_model: {model_loaded.get('d_model')}\ndevice: {model_loaded.get('device')}", "x": 6.45, "y": 3.05, "w": 5.5, "h": 1.45, "font_size": 1200, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
        slide_xml("Stage 2: Entity -> sentence pool", "Review Flattening", [
            {"text": "Input\nMột entity gồm nhiều reviews. Mỗi review có rating + list sentences. Review boundary được bỏ khi tạo sentence pool cho entity.", "x": 0.75, "y": 1.35, "w": 5.25, "h": 1.45, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Trace output\nentity_id: {entity_encoded.get('entity_id')}\nsentences encoded: {entity_encoded.get('sentences')}\naspect candidates: {entity_encoded.get('aspect_candidates')}\nshard: {entity_encoded.get('shard_idx')}/{entity_encoded.get('num_shards')}", "x": 6.35, "y": 1.35, "w": 5.65, "h": 1.7, "font_size": 1300, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Output của stage này là tensorized sentence pool: token ids + sentence metadata để trace ngược review_id/sentence.", "x": 0.75, "y": 4.0, "w": 11.25, "h": 0.85, "font_size": 1200, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Stage 3: SemAE encode + aspect prototypes", "Latent Distributions", [
            {"text": "Input\n- token ids của từng sentence\n- seeds_hasos/<aspect>.txt\n- trained SemAE encoder + quantizer", "x": 0.75, "y": 1.35, "w": 4.9, "h": 1.45, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Output\nD_z(s): distribution của sentence\nP_z: background distribution\nP_k: prototype của aspect k\nMột entity được score lại cho từng aspect.", "x": 6.0, "y": 1.35, "w": 6.1, "h": 1.65, "font_size": 1200, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Ranking score\nscore(s,k) = KL(D_z(s)||P_z) - beta * KL(D_z(s)||P_k)\nbeta = 0.7; score thấp hơn được ưu tiên.", "x": 0.75, "y": 3.75, "w": 11.35, "h": 1.0, "font_size": 1350, "bold": True, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Stage 4: Ranking output", "Ranked Sentences", [
            {"text": "Input\nD_z(s), P_z, P_k cho từng aspect.\nMỗi aspect nhận cùng sentence pool nhưng ranking khác nhau.", "x": 0.75, "y": 1.35, "w": 4.7, "h": 1.35, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Trace sample\n{sample_rank}", "x": 5.85, "y": 1.35, "w": 6.2, "h": 3.35, "font_size": 900, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Output là ranked list có score, review_id, sentence. Đây là điểm kiểm tra quan trọng nhất để biết aspect selection đúng hay không.", "x": 0.75, "y": 4.55, "w": 11.3, "h": 0.75, "font_size": 1150, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Stage 5: Truncate + write aspect summary", "Aspect-only Output", [
            {"text": "Input\nRanked sentences của một entity/aspect.\nSau đó lọc câu ngắn, dedupe TF-IDF nếu bật, cắt theo max_tokens=40.", "x": 0.75, "y": 1.35, "w": 4.95, "h": 1.45, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Output file\n{summary.get('output_path')}\nsentences: {summary.get('sentences')}", "x": 6.05, "y": 1.35, "w": 6.05, "h": 1.05, "font_size": 1050, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": f"Sample summary\n{sample_summary}", "x": 0.75, "y": 3.0, "w": 11.35, "h": 2.1, "font_size": 1000, "fill": "FFFFFF", "line": "D8DEE9"},
        ]),
        slide_xml("Stage 5b: Sentence provenance", "Audit Artifact", [
            {"text": "Input\nSelected summary sentences + ranked sentence records + aspect seed hits + sentiment keyword hits.", "x": 0.75, "y": 1.35, "w": 5.0, "h": 1.25, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Output\noutputs/{run_id}_provenance.jsonl\nrows: {provenance_rows}\ncoverage target: 1 row per summary sentence", "x": 6.05, "y": 1.35, "w": 6.05, "h": 1.25, "font_size": 1200, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": f"Sample provenance\n{provenance_sample}", "x": 0.75, "y": 3.05, "w": 11.35, "h": 1.65, "font_size": 950, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Stage 6: Sentiment split", "Post-rank Buckets", [
            {"text": "Input\nChỉ các sentences đã được chọn cho aspect summary. Sentiment không thay đổi ranking, chỉ tách bucket sau khi đã chọn câu.", "x": 0.75, "y": 1.35, "w": 5.25, "h": 1.35, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Output FAC_ROOM / entity 100597\npos sentences: {sent_pos.get('sentences')}\nneg sentences: {sent_neg.get('sentences')}\nneu sentences: {sent_neu.get('sentences')}", "x": 6.35, "y": 1.35, "w": 5.65, "h": 1.35, "font_size": 1300, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": "Tree output\noutputs/<run_id>_sentiment/<aspect>__pos/<split>_<entity>\noutputs/<run_id>_sentiment/<aspect>__neg/<split>_<entity>\noutputs/<run_id>_sentiment/<aspect>__neu/<split>_<entity>", "x": 0.75, "y": 3.35, "w": 11.25, "h": 1.4, "font_size": 1050, "fill": "EEF6FF", "line": "BFDBFE"},
        ]),
        slide_xml("Stage 7: Export + scoring outputs", "Files for Audit", [
            {"text": f"Export input\nFile tree:\noutputs/{run_id}/\noutputs/{run_id}_sentiment/", "x": 0.75, "y": 1.35, "w": 4.85, "h": 1.35, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": f"Export output\naspect rows: {line_rows}\nsentiment rows: {sentiment_rows}\nprovenance rows: {provenance_rows}\nJSONL + TSV line format", "x": 5.95, "y": 1.35, "w": 2.9, "h": 1.55, "font_size": 1120, "fill": "ECFDF5", "line": "99F6E4"},
            {"text": f"Reference-free scores\nbert_f1_aspect: {macro.get('bert_f1_aspect'):.4f}\nbert_f1_source: {macro.get('bert_f1_source'):.4f}\naspect_purity: {macro.get('aspect_purity'):.4f}", "x": 9.15, "y": 1.35, "w": 2.95, "h": 1.55, "font_size": 1150, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Original ROUGE lane\nInput thiếu: data/hasos/gold/<aspect>/#ID#_[012].txt\nKhi có gold, rerun aspect_inference.py without --no_eval để sinh eval_<run_id>.json/txt.", "x": 0.75, "y": 3.85, "w": 11.35, "h": 1.05, "font_size": 1150, "fill": "FFF7ED", "line": "FDBA74"},
        ]),
        slide_xml("Muốn so sánh đúng repo gốc cần gì?", "Next Step", [
            {"text": "Cần bổ sung gold summaries\nMỗi aspect có thư mục riêng. Mỗi entity cần 1-3 reference files theo pattern pyrouge: #ID#_[012].txt.", "x": 0.75, "y": 1.35, "w": 5.45, "h": 1.35, "font_size": 1200, "fill": "FFFFFF", "line": "D8DEE9"},
            {"text": "Command sau khi có gold\npython src/aspect_inference.py --summary_data data/hasos/hasos_summ.json --gold_data data/hasos/gold --seedsdir data/seeds_hasos --gold_aspects <29 codes> --model <checkpoint> --run_id <run> --gpu 0", "x": 6.55, "y": 1.35, "w": 5.5, "h": 2.25, "font_size": 900, "fill": "EEF6FF", "line": "BFDBFE"},
            {"text": "Kết luận\nDeck này mô tả pipeline input/output. Điểm BERTScore/reference-free hiện có chỉ là audit bổ sung; chưa thay thế ROUGE official của SemAE gốc.", "x": 0.75, "y": 4.35, "w": 11.3, "h": 0.9, "font_size": 1250, "bold": True, "fill": "ECFDF5", "line": "99F6E4"},
        ]),
    ]
    return slides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="space_hasos_full_e20")
    parser.add_argument("--output", default=str(REPORTS_DIR / "space_hasos_stage_io.pptx"))
    args = parser.parse_args()
    out = Path(args.output)
    write_deck(build_slides(args.run_id), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

