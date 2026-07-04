from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
FONT_DIR = Path(r"C:\Windows\Fonts")


def font(size, bold=False, italic=False):
    names = []
    if bold:
        names += ["segoeuib.ttf", "arialbd.ttf"]
    elif italic:
        names += ["segoeuii.ttf", "ariali.ttf"]
    else:
        names += ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F = {
    "title": font(46, True),
    "subtitle": font(24),
    "head": font(27, True),
    "cardhead": font(23, True),
    "body": font(21),
    "small": font(18),
    "tiny": font(16),
    "badge": font(25, True),
}

NAVY = "#0B2A55"
BLUE = "#1357A6"
TEAL = "#087F7A"
GREEN = "#0B6B32"
GOLD = "#C98300"
RED = "#B22222"
PURPLE = "#5D3B9A"
GRAY = "#5B6573"
LINE = "#D7E1EE"
WHITE = "#FFFFFF"

BG = {
    BLUE: "#F1F7FF",
    TEAL: "#ECFAF7",
    GREEN: "#F0FAF2",
    GOLD: "#FFF7E8",
    RED: "#FFF2F2",
    PURPLE: "#F5F1FF",
}


def rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap(draw, text, fnt, width):
    lines = []
    for part in text.split("\n"):
        cur = ""
        for word in part.split():
            test = word if not cur else f"{cur} {word}"
            if size(draw, test, fnt)[0] <= width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
    return lines


def text(draw, xy, value, fnt, fill, width=None, line_gap=5, align="left"):
    x, y = xy
    lines = value.split("\n") if width is None else wrap(draw, value, fnt, width)
    offset = 0
    for line in lines:
        w, h = size(draw, line, fnt)
        tx = x
        if align == "center" and width is not None:
            tx = x + (width - w) / 2
        draw.text((tx, y + offset), line, font=fnt, fill=rgb(fill))
        offset += h + line_gap
    return offset


def rounded(draw, box, fill=WHITE, outline=BLUE, width=3, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=rgb(fill), outline=rgb(outline), width=width)


def arrow(draw, start, end, color=BLUE, width=4):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=rgb(color), width=width)
    if abs(y2 - y1) >= abs(x2 - x1):
        pts = [(x2, y2), (x2 - 9, y2 - 18), (x2 + 9, y2 - 18)] if y2 >= y1 else [(x2, y2), (x2 - 9, y2 + 18), (x2 + 9, y2 + 18)]
    else:
        pts = [(x2, y2), (x2 - 18, y2 - 9), (x2 - 18, y2 + 9)] if x2 >= x1 else [(x2, y2), (x2 + 18, y2 - 9), (x2 + 18, y2 + 9)]
    draw.polygon(pts, fill=rgb(color))


def icon(draw, kind, box, color):
    x1, y1, x2, y2 = box
    c = rgb(color)
    w = max(3, (x2 - x1) // 16)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if kind == "data":
        draw.ellipse((x1 + 8, y1 + 8, x2 - 8, y1 + 24), outline=c, width=w)
        draw.rectangle((x1 + 8, y1 + 16, x2 - 8, y2 - 16), outline=c, width=w)
        draw.ellipse((x1 + 8, y2 - 24, x2 - 8, y2 - 8), outline=c, width=w)
    elif kind == "doc":
        draw.rectangle((x1 + 14, y1 + 8, x2 - 12, y2 - 8), outline=c, width=w)
        for yy in (y1 + 25, y1 + 40, y1 + 55):
            draw.line((x1 + 25, yy, x2 - 25, yy), fill=c, width=w)
    elif kind == "chip":
        pad = max(8, (x2 - x1) // 4)
        inner = max(5, (x2 - x1) // 7)
        draw.rectangle((x1 + pad, y1 + pad, x2 - pad, y2 - pad), outline=c, width=w)
        draw.rectangle((x1 + pad + inner, y1 + pad + inner, x2 - pad - inner, y2 - pad - inner), fill=c)
        for i in range(5):
            yy = y1 + 15 + i * 13
            if y1 + 4 < yy < y2 - 4:
                draw.line((x1 + 5, yy, x1 + pad, yy), fill=c, width=w)
                draw.line((x2 - pad, yy, x2 - 5, yy), fill=c, width=w)
    elif kind == "chart":
        for i, h in enumerate((20, 35, 52)):
            draw.rectangle((x1 + 18 + i * 18, y2 - 14 - h, x1 + 30 + i * 18, y2 - 14), fill=c)
        draw.line((x1 + 12, y2 - 12, x2 - 10, y2 - 12), fill=c, width=w)
    elif kind == "split":
        draw.line((x1 + 12, cy, cx, cy), fill=c, width=w)
        arrow(draw, (cx, cy), (x2 - 12, y1 + 18), color, w)
        arrow(draw, (cx, cy), (x2 - 12, y2 - 18), color, w)
    elif kind == "spark":
        draw.polygon([(cx, y1 + 8), (cx + 12, cy - 12), (x2 - 8, cy), (cx + 12, cy + 12), (cx, y2 - 8), (cx - 12, cy + 12), (x1 + 8, cy), (cx - 12, cy - 12)], fill=c)
    elif kind == "shield":
        draw.polygon([(cx, y1 + 8), (x2 - 12, y1 + 20), (x2 - 18, cy + 18), (cx, y2 - 8), (x1 + 18, cy + 18), (x1 + 12, y1 + 20)], outline=c, width=w)
        draw.line((x1 + 24, cy, cx - 3, cy + 12, x2 - 20, cy - 14), fill=c, width=w)
    elif kind == "absa":
        draw.ellipse((x1 + 12, y1 + 10, x2 - 12, y2 - 10), outline=c, width=w)
        draw.line((cx, y1 + 14, cx, y2 - 14), fill=c, width=w)
        draw.arc((x1 + 20, y1 + 8, cx + 10, y2 - 8), 85, 275, fill=c, width=w)
        draw.arc((cx - 10, y1 + 8, x2 - 20, y2 - 8), -95, 95, fill=c, width=w)
    elif kind == "folder":
        draw.rectangle((x1 + 8, y1 + 24, x2 - 8, y2 - 8), outline=c, width=w)
        draw.rectangle((x1 + 8, y1 + 15, x1 + 38, y1 + 27), outline=c, width=w)
    elif kind == "judge":
        draw.line((cx, y1 + 10, cx, y2 - 10), fill=c, width=w)
        draw.line((x1 + 14, y1 + 25, x2 - 14, y1 + 25), fill=c, width=w)
        draw.polygon([(x1 + 20, y1 + 28), (x1 + 8, y1 + 55), (x1 + 34, y1 + 55)], outline=c, width=w)
        draw.polygon([(x2 - 20, y1 + 28), (x2 - 34, y1 + 55), (x2 - 8, y1 + 55)], outline=c, width=w)
    else:
        draw.ellipse((x1 + 12, y1 + 12, x2 - 12, y2 - 12), outline=c, width=w)


def title_bar(draw, w, title, subtitle, color=NAVY):
    rounded(draw, (36, 28, w - 36, 135), WHITE, color, 3, 18)
    tw, _ = size(draw, title, F["title"])
    draw.text(((w - tw) / 2, 38), title, font=F["title"], fill=rgb(color))
    sw, _ = size(draw, subtitle, F["subtitle"])
    draw.text(((w - sw) / 2, 94), subtitle, font=F["subtitle"], fill=rgb(GRAY))


def pipeline(filename, title, subtitle, color, steps, side_title, side_body):
    w, h = 1600, 1000
    img = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)
    title_bar(draw, w, title, subtitle, color)
    x, y = 60, 160
    step_w = 990
    step_h = 78 if len(steps) == 8 else 86
    gap = 18
    for idx, (label, kind) in enumerate(steps):
        yy = y + idx * (step_h + gap)
        fill = BG[color] if idx in (1, len(steps) - 1) else WHITE
        rounded(draw, (x, yy, x + step_w, yy + step_h), fill, color, 3, 16)
        draw.ellipse((x + 24, yy + 18, x + 76, yy + 70), fill=rgb(color))
        num = str(idx + 1)
        nw, nh = size(draw, num, F["badge"])
        draw.text((x + 50 - nw / 2, yy + 44 - nh / 2), num, font=F["badge"], fill=rgb(WHITE))
        icon(draw, kind, (x + 108, yy + 14, x + 178, yy + 74), color)
        draw.text((x + 210, yy + 22), label, font=F["head"], fill=rgb(NAVY if color == BLUE else color))
        if idx < len(steps) - 1:
            arrow(draw, (x + step_w / 2, yy + step_h), (x + step_w / 2, yy + step_h + gap - 2), color, 4)
    panel = (1130, 235, 1545, 890)
    rounded(draw, panel, BG[color], color, 3, 18)
    icon(draw, "doc", (1285, 290, 1390, 395), color)
    text(draw, (panel[0] + 35, 430), side_title, F["title"], color, panel[2] - panel[0] - 70, 7, "center")
    draw.line((panel[0] + 60, 505, panel[2] - 60, 505), fill=rgb(color), width=3)
    text(draw, (panel[0] + 48, 560), side_body, F["body"], "#1E293B", panel[2] - panel[0] - 96, 10, "center")
    last_y = y + (len(steps) - 1) * (step_h + gap) + step_h / 2
    arrow(draw, (x + step_w, last_y), (panel[0], last_y), color, 5)
    img.save(OUT / filename)


def overview():
    w, h = 1800, 1080
    img = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)
    title_bar(draw, w, "Phương pháp 2: SemAE", "Một nền bằng chứng, bốn biến thể tóm tắt", NAVY)
    cols = [
        (35, 165, 410, 925, BLUE, "1. Dữ liệu"),
        (455, 165, 875, 925, TEAL, "2. Bằng chứng SemAE"),
        (920, 165, 1360, 925, GOLD, "3. Biến thể"),
        (1405, 165, 1765, 925, PURPLE, "4. Đánh giá"),
    ]
    for x1, y1, x2, y2, color, head in cols:
        rounded(draw, (x1, y1, x2, y2), WHITE, color, 3, 18)
        text(draw, (x1 + 20, y1 + 18), head, F["head"], color, x2 - x1 - 40, 3, "center")
        draw.line((x1 + 18, y1 + 60, x2 - 18, y1 + 60), fill=rgb(LINE), width=3)

    def card(box, title, body, color, kind):
        x1, y1, x2, y2 = box
        rounded(draw, box, BG.get(color, WHITE), color, 2, 14)
        icon(draw, kind, (x1 + 15, y1 + 18, x1 + 85, y1 + 88), color)
        draw.text((x1 + 102, y1 + 20), title, font=F["cardhead"], fill=rgb(color))
        text(draw, (x1 + 102, y1 + 55), body, F["tiny"], "#1E293B", x2 - x1 - 122, 4)

    card((58, 250, 385, 390), "HASOS", "50 khách sạn\n5.000 đánh giá\n45.529 câu", BLUE, "data")
    card((58, 465, 385, 600), "Hệ phân loại", "29 khía cạnh\n6 nhóm cha", TEAL, "split")
    card((58, 675, 385, 805), "Từ khóa hạt giống", "Tạo câu ứng viên", GOLD, "spark")
    card((485, 240, 845, 345), "Tách câu", "giữ entity_id, review_id,\nsentence_index", BLUE, "doc")
    card((485, 380, 845, 485), "Mã hóa SemAE", "SentencePiece SPACE\n+ checkpoint", TEAL, "chip")
    card((485, 520, 845, 625), "Xếp hạng KL", "score(s,a)\ntheo khía cạnh", GOLD, "chart")
    card((485, 660, 845, 815), "Kho bằng chứng", "ranked_evidence.jsonl\nthreshold_evidence.jsonl\nprovenance.jsonl", GREEN, "data")
    variants = [
        ("M1", "Trích rút", "B = 40 từ", BLUE, "doc"),
        ("M2", "Sinh trừu tượng", "T = 0,0075; B = 128", TEAL, "spark"),
        ("M3", "Tách cảm xúc từ khóa", "T = 0,0055; B = 96", GREEN, "split"),
        ("M4", "Tách cảm xúc BERT-ABSA", "T = 0,0050; B = 96", RED, "absa"),
        ("", "Hậu kiểm sinh", "thiếu hỗ trợ -> dự phòng", PURPLE, "shield"),
    ]
    yy = 225
    for badge, title, body, color, kind in variants:
        rounded(draw, (945, yy, 1330, yy + 100), BG[color], color, 2, 14)
        if badge:
            draw.ellipse((962, yy + 24, 1012, yy + 74), fill=rgb(color))
            bw, bh = size(draw, badge, F["badge"])
            draw.text((987 - bw / 2, yy + 49 - bh / 2), badge, font=F["badge"], fill=rgb(WHITE))
            tx = 1030
        else:
            icon(draw, kind, (965, yy + 20, 1025, yy + 80), color)
            tx = 1040
        draw.text((tx, yy + 20), title, font=F["cardhead"], fill=rgb(color))
        draw.text((tx, yy + 58), body, font=F["tiny"], fill=rgb("#1E293B"))
        yy += 125
    card((1430, 240, 1740, 345), "ROUGE", "nhóm cha có tham chiếu", BLUE, "chart")
    card((1430, 390, 1740, 495), "Chỉ số vận hành", "sinh, dự phòng,\nsao chép, nén", TEAL, "chart")
    card((1430, 540, 1740, 645), "LLM judge", "đúng khía cạnh,\ncảm xúc, trung thực", GOLD, "judge")
    card((1430, 690, 1740, 795), "Truy vết nguồn", "tóm tắt -> bằng chứng\n-> câu gốc", PURPLE, "doc")
    arrow(draw, (410, 545), (455, 545), BLUE, 6)
    arrow(draw, (875, 545), (920, 545), TEAL, 6)
    arrow(draw, (1360, 545), (1405, 545), GOLD, 6)
    rounded(draw, (35, 955, 1765, 1040), WHITE, BLUE, 2, 18)
    text(draw, (90, 980), "M1-M4 dùng chung nền bằng chứng SemAE để so sánh công bằng.", F["head"], BLUE, 1620, 4, "center")
    img.save(OUT / "method2_overview.png")


def sharded():
    w, h = 1700, 880
    img = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(img)
    title_bar(draw, w, "Suy luận SemAE theo mảnh xử lý", "Chia theo khách sạn, chạy song song rồi hợp nhất kết quả", NAVY)

    def box(rect, title, body, color, kind):
        rounded(draw, rect, BG.get(color, WHITE), color, 3, 18)
        x1, y1, x2, y2 = rect
        icon(draw, kind, (x1 + 18, y1 + 30, x1 + 88, y1 + 100), color)
        title_height = text(draw, (x1 + 105, y1 + 28), title, F["cardhead"], color, x2 - x1 - 125, 2)
        text(draw, (x1 + 105, y1 + 34 + title_height), body, F["tiny"], "#1E293B", x2 - x1 - 125, 4)

    box((70, 250, 350, 395), "Đọc dataset", "HASOS\n50 khách sạn", BLUE, "data")
    box((455, 250, 760, 395), "Chia theo khách sạn", "Không tách rời câu của cùng khách sạn", TEAL, "split")
    rounded(draw, (865, 205, 1180, 470), WHITE, BLUE, 3, 18)
    text(draw, (895, 232), "Các mảnh SemAE", F["head"], BLUE, 255, 4, "center")
    for i, label in enumerate(["Mảnh 1 -> SemAE", "Mảnh 2 -> SemAE", "Mảnh n -> SemAE"]):
        yy = 295 + i * 55
        icon(draw, "chip", (895, yy - 23, 945, yy + 27), BLUE)
        draw.text((965, yy - 14), label, font=F["body"], fill=rgb("#1E293B"))
    box((1280, 250, 1580, 395), "Hợp nhất", "Gom theo run_id và entity_id", GOLD, "data")
    arrow(draw, (350, 322), (455, 322), BLUE, 5)
    arrow(draw, (760, 322), (865, 322), TEAL, 5)
    arrow(draw, (1180, 322), (1280, 322), BLUE, 5)
    outputs = [
        ((135, 620, 520, 780), "Thư mục tóm tắt", "mỗi tệp ứng với khách sạn-khía cạnh", BLUE, "folder"),
        ((660, 620, 1035, 780), "provenance.jsonl", "nguồn câu đi vào tóm tắt", PURPLE, "doc"),
        ((1150, 620, 1640, 780), "ranked_evidence.jsonl\nthreshold_evidence.jsonl", "bằng chứng xếp hạng và theo ngưỡng", TEAL, "data"),
    ]
    for rect, title, body, color, kind in outputs:
        box(rect, title, body, color, kind)
    bus_y = 555
    draw.line((1430, 395, 1430, bus_y, 330, bus_y), fill=rgb(GOLD), width=5)
    for x in (330, 850, 1395):
        arrow(draw, (x, bus_y), (x, 620), GOLD, 5)
    img.save(OUT / "method2_sharded_inference.png")


overview()
sharded()

pipeline(
    "method2_m1_pipeline.png",
    "M1 - Tóm tắt trích rút",
    "Giữ câu gốc, không dùng mô hình sinh",
    BLUE,
    [
        ("Đọc dataset", "data"),
        ("Tách câu", "doc"),
        ("Mã hóa SemAE", "chip"),
        ("Xếp hạng KL", "chart"),
        ("Chọn câu", "split"),
        ("B = 40 từ", "doc"),
        ("Truy vết nguồn", "folder"),
    ],
    "Bản chất M1",
    "Trích rút, bám nguồn, không ảo giác sinh.",
)

pipeline(
    "method2_m2_pipeline.png",
    "M2 - Sinh trừu tượng trực tiếp",
    "Sinh từ bằng chứng theo khía cạnh trước khi tách cảm xúc",
    TEAL,
    [
        ("Đọc dataset", "data"),
        ("SemAE score <= 0,0075", "chip"),
        ("Ghi bằng chứng", "data"),
        ("Chuẩn bị đầu vào", "split"),
        ("FLAN-T5", "spark"),
        ("Hậu kiểm", "shield"),
        ("Dự phòng", "folder"),
        ("Đầu ra", "doc"),
    ],
    "Rủi ro M2",
    "Chưa tách cảm xúc; dễ trộn khen và chê trong cùng khía cạnh.",
)

pipeline(
    "method2_m3_pipeline.png",
    "M3 - Tách cảm xúc bằng từ khóa",
    "Tách pos, neg, neu trước khi sinh",
    GREEN,
    [
        ("Đọc dataset", "data"),
        ("SemAE score <= 0,0055", "chip"),
        ("Tách pos / neg / neu", "split"),
        ("Nhóm bằng chứng", "data"),
        ("FLAN-T5", "spark"),
        ("Kiểm tra cảm xúc", "shield"),
        ("Dự phòng", "folder"),
        ("Đầu ra", "doc"),
    ],
    "Giới hạn M3",
    "Yếu với phủ định, mỉa mai và ý kiến gián tiếp.",
)

pipeline(
    "method2_m4_pipeline.png",
    "M4 - Tách cảm xúc bằng BERT-ABSA",
    "Giữ SemAE, thay bước gán cảm xúc bằng mô hình theo ngữ cảnh",
    RED,
    [
        ("Đọc dataset", "data"),
        ("SemAE score <= 0,0050", "chip"),
        ("BERT-ABSA", "absa"),
        ("Nhóm bằng chứng", "data"),
        ("FLAN-T5", "spark"),
        ("Kiểm tra đầu ra", "shield"),
        ("Dự phòng", "folder"),
        ("Đầu ra", "doc"),
    ],
    "Ý nghĩa M4",
    "Thay bộ tách cảm xúc, không thay bộ chọn bằng chứng SemAE.",
)

print("Rendered consistent method2 figures")
