"""Hotel aspect-sentiment pipeline for the three hotel review CSV files.

Default strategy is two-stage:
1. Stream all reviews locally with rule-based aspect/sentiment aggregation.
2. Send compact hotel profiles to Qwen for representative-sentence selection
   and tie/ambiguous sentiment validation.

The sentence-qwen strategy now runs a richer multi-stage workflow:
sentence splitting -> Qwen aspect segmentation/translation ->
per-aspect sentiment classification -> bilingual aspect summaries.

Important evaluation note:
The sentence-qwen path is a multi-aspect ABSA pipeline. A review sentence can
produce several opinion units, each with its own aspect and sentiment. Any
benchmark that feeds the same full sentence/evidence row to a single-label
classifier and expects one gold aspect per row will under-measure this behavior,
especially when gold evidence is a placeholder or a full sentence containing
several annotated quads. For extraction quality, prefer review/sentence-level
set matching between predicted opinion units and gold ABSA quads, with
precision/recall/F1 instead of only row-level accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd
from openai import OpenAI


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_ANNOTATION_TAXONOMY_CANDIDATES = [
    WORKSPACE / "data" / "taxonomy" / "ANOTATION_Hotel_ABSA_20260520.xlsx",
    WORKSPACE / "ANOTATION_Hotel_ABSA (1).xlsx",
]


def resolve_annotation_taxonomy_path() -> Path:
    env_path = str(os.environ.get("ANNOTATION_TAXONOMY_XLSX", "")).strip()
    if env_path:
        return Path(env_path).expanduser().resolve()
    for candidate in DEFAULT_ANNOTATION_TAXONOMY_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_ANNOTATION_TAXONOMY_CANDIDATES[-1]


ANNOTATION_TAXONOMY_XLSX = resolve_annotation_taxonomy_path()

DEFAULT_INPUTS = [
    WORKSPACE / "data" / "hotel_review1.csv",
    WORKSPACE / "data" / "hotel_review2.csv",
    WORKSPACE / "data" / "hotel_review3.csv",
]
DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_API_KEY = "local-dev-key"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"
CHECKPOINT_VERSION = 1
BERTSCORER_CACHE: dict[str, Any] = {}

ASPECTS: dict[str, str] = {
    "facility": (
        "Co so vat chat va moi truong vat ly: phong, phong tam, giuong, "
        "noi that, sanh/le tan, toa nha, dieu hoa, an ninh vat ly, vi tri, "
        "khu vuc xung quanh, canh quan va view."
    ),
    "amenity": (
        "Tien ich/chu nang khach san cung cap: wifi, bua sang, ho boi, gym, "
        "spa/massage, giai tri, bai dau xe, shuttle, giat ui, nha hang/bar/cafe, "
        "do an/thuc uong va tien nghi trong phong."
    ),
    "service": (
        "Dich vu do con nguoi/quy trinh thuc hien: thai do nhan vien, tinh "
        "chuyen nghiep, check-in/check-out, phan hoi, xu ly van de, room service, "
        "restaurant staff service, hospitality va giao tiep."
    ),
    "experience": (
        "Trai nghiem: trai nghiem tong the, bau khong khi, canh nhin tong the, "
        "muc do thu gian."
    ),
    "branding": (
        "Thuong hieu: muc do hai long cua khach hang voi khach san nhu mot "
        "thuong hieu/tong the, dua tren ky vong, danh tieng, tieu chuan, "
        "dang cap va hinh anh thuong hieu."
    ),
    "loyalty": (
        "Muc do trung thanh: kha nang khach quay lai va gioi thieu khach san "
        "cho nguoi khac."
    ),
}
ASPECT_NAMES = list(ASPECTS)
SENTIMENTS = ["positive", "negative", "neutral"]
ALL_ASPECTS_SUMMARY_KEY = "all_aspects"
ASPECT_GUARDRAIL_VERSION = "aspect-guardrail-v2-taxonomy-source"
ASPECT_LABELS_VI = {
    "facility": "cơ sở vật chất",
    "amenity": "tiện ích",
    "service": "dịch vụ",
    "experience": "trải nghiệm",
    "branding": "thương hiệu",
    "loyalty": "mức độ trung thành",
}
ASPECT_LABELS_EN = {
    "facility": "facility",
    "amenity": "amenity",
    "service": "service",
    "experience": "experience",
    "branding": "branding",
    "loyalty": "loyalty",
}
ASPECT_SUMMARY_CODES = {
    "facility": "FAC",
    "amenity": "AME",
    "service": "SER",
    "experience": "EXP",
    "branding": "BRA",
    "loyalty": "LOY",
    ALL_ASPECTS_SUMMARY_KEY: "ALL",
}

ASPECT_CLUSTER_FRAMEWORK: dict[str, dict[str, Any]] = {
    "facility": {
        "core_definition": "The physical and structural properties of the hotel environment.",
        "quick_decision_rule": "Is the review talking about the hotel's physical space, room condition, or built environment?",
        "covers": [
            "room condition",
            "furniture",
            "bed",
            "bathroom",
            "interior design",
            "lighting",
            "cleanliness",
            "noise insulation",
            "hallway",
            "elevator",
            "balcony",
            "scenery or view",
            "parking area",
            "air conditioning",
            "workspace",
            "physical security infrastructure",
        ],
        "does_not_cover": [
            "human service quality",
            "emotional satisfaction",
            "hotel-provided activities or services",
        ],
        "examples": [
            "room",
            "bed",
            "bathroom",
            "balcony",
            "interior",
            "hallway",
            "elevator",
            "workspace",
            "cleanliness",
            "view",
            "decoration",
            "lighting",
            "spacious",
            "noisy",
            "modern",
        ],
    },
    "amenity": {
        "core_definition": "The facilities, utilities, or extra functions provided by the hotel for guest usage and convenience.",
        "quick_decision_rule": "Is this something the hotel provides for guests to use or enjoy?",
        "covers": [
            "WiFi",
            "swimming pool",
            "gym",
            "spa",
            "buffet",
            "breakfast",
            "restaurant",
            "minibar",
            "laundry",
            "shuttle service",
            "entertainment area",
            "kids zone",
            "TV",
            "kitchen",
            "business center",
            "food and beverage offerings",
        ],
        "does_not_cover": [
            "physical room condition itself",
            "staff attitude",
        ],
        "examples": [
            "wifi",
            "breakfast",
            "buffet",
            "restaurant",
            "pool",
            "gym",
            "spa",
            "minibar",
            "shuttle",
            "laundry",
            "food",
            "drinks",
            "entertainment",
        ],
    },
    "service": {
        "core_definition": "The quality of human interaction and hospitality delivered by hotel staff.",
        "quick_decision_rule": "Is the review evaluating staff behavior, responsiveness, or support?",
        "covers": [
            "receptionist attitude",
            "waiter service",
            "check-in/check-out",
            "responsiveness",
            "friendliness",
            "professionalism",
            "hospitality",
            "communication",
            "problem handling",
            "service",
        ],
        "does_not_cover": [
            "hotel infrastructure",
            "emotional outcome of the stay",
        ],
        "examples": [
            "staff",
            "receptionist",
            "concierge",
            "waiter",
            "housekeeping",
            "check-in",
            "hospitality",
            "helpful",
            "polite",
            "responsive",
            "friendly",
        ],
    },
    "experience": {
        "core_definition": "The guest's overall emotional perception, comfort, and satisfaction during the stay.",
        "quick_decision_rule": "Is the user describing how the stay felt overall?",
        "covers": [
            "relaxation",
            "enjoyment",
            "atmosphere",
            "comfort",
            "convenience",
            "peacefulness",
            "safety perception",
            "value for money",
            "vacation feeling",
            "memorable experiences",
            "experience",
        ],
        "does_not_cover": [
            "explicit revisit intention",
            "brand reputation",
        ],
        "examples": [
            "relaxing",
            "enjoyable",
            "comfortable",
            "peaceful",
            "cozy",
            "memorable",
            "convenient",
            "safe",
            "worth the money",
            "luxurious feeling",
        ],
    },
    "branding": {
        "core_definition": "Perceptions related to the hotel's brand image, reputation, prestige, or expected standards.",
        "quick_decision_rule": "Does the review mention brand identity, reputation, or expectation of a hotel chain?",
        "covers": [
            "luxury image",
            "international standards",
            "brand consistency",
            "reputation",
            "prestige",
            "comparisons with expected brand quality",
        ],
        "does_not_cover": [
            "personal emotional satisfaction",
            "loyalty intention",
        ],
        "examples": [
            "Hilton standard",
            "Marriott quality",
            "luxury brand",
            "reputation",
            "premium image",
            "five-star standard",
        ],
    },
    "loyalty": {
        "core_definition": "The guest's future behavioral intention toward the hotel.",
        "quick_decision_rule": "Does the guest express intention to return or recommend the hotel?",
        "covers": [
            "revisit intention",
            "recommendation intention",
            "emotional attachment",
            "favorite hotel",
            "long-term preference",
        ],
        "does_not_cover": [
            "general satisfaction without future intention",
        ],
        "examples": [
            "stay again",
            "come back",
            "highly recommend",
            "favorite hotel",
            "will return",
            "revisit",
            "recommend to friends",
        ],
    },
}

ASPECT_ANNOTATION_PROMPT = """
Annotation framework for aspect assignment. Apply this table strictly:
- FACILITY
  Core definition: the physical and structural properties of the hotel environment.
  Quick decision rule: is the review talking about the hotel's physical space, room condition, or built environment?
  Covers: room condition, furniture, bed, bathroom, interior design, lighting, cleanliness, noise insulation, hallway, elevator, balcony, scenery/view, parking area, air conditioning, workspace, physical security infrastructure.
  Does not cover: human service quality, emotional satisfaction, or hotel-provided activities/services.
  Example keywords/terms: room, bed, bathroom, balcony, interior, hallway, elevator, workspace, cleanliness, view, decoration, lighting, spacious, noisy, modern.
- AMENITY
  Core definition: the facilities, utilities, or extra functions provided by the hotel for guest usage and convenience.
  Quick decision rule: is this something the hotel provides for guests to use or enjoy?
  Covers: WiFi, swimming pool, gym, spa, buffet, breakfast, restaurant, minibar, laundry, shuttle service, entertainment area, kids zone, TV, kitchen, business center, food and beverage offerings.
  Does not cover: physical room condition itself or staff attitude.
  Example keywords/terms: wifi, breakfast, buffet, restaurant, pool, gym, spa, minibar, shuttle, laundry, food, drinks, entertainment.
- SERVICE
  Core definition: the quality of human interaction and hospitality delivered by hotel staff.
  Quick decision rule: is the review evaluating staff behavior, responsiveness, or support?
  Covers: receptionist attitude, waiter service, check-in/check-out, responsiveness, friendliness, professionalism, hospitality, communication, problem handling, service.
  Does not cover: hotel infrastructure or emotional outcome of the stay.
  Example keywords/terms: staff, receptionist, concierge, waiter, housekeeping, check-in, hospitality, helpful, polite, responsive, friendly.
- EXPERIENCE
  Core definition: the guest's overall emotional perception, comfort, and satisfaction during the stay.
  Quick decision rule: is the user describing how the stay felt overall?
  Covers: relaxation, enjoyment, atmosphere, comfort, convenience, peacefulness, safety perception, value for money, vacation feeling, memorable experiences, experience.
  Does not cover: explicit revisit intention or brand reputation.
  Example keywords/terms: relaxing, enjoyable, comfortable, peaceful, cozy, memorable, convenient, safe, worth the money, luxurious feeling.
- BRANDING
  Core definition: perceptions related to the hotel's brand image, reputation, prestige, or expected standards.
  Quick decision rule: does the review mention brand identity, reputation, or expectation of a hotel chain?
  Covers: luxury image, international standards, brand consistency, reputation, prestige, comparisons with expected brand quality.
  Does not cover: personal emotional satisfaction or loyalty intention.
  Example keywords/terms: Hilton standard, Marriott quality, luxury brand, reputation, premium image, five-star standard.
- LOYALTY
  Core definition: the guest's future behavioral intention toward the hotel.
  Quick decision rule: does the guest express intention to return or recommend the hotel?
  Covers: revisit intention, recommendation intention, emotional attachment, favorite hotel, long-term preference.
  Does not cover: general satisfaction without future intention.
  Example keywords/terms: stay again, come back, highly recommend, favorite hotel, will return, revisit, recommend to friends.

Aspect splitting procedure:
1. Split the input into the smallest faithful opinion units where each unit has one main target and one main meaning.
2. For each unit, first ask the quick decision rules above, then check the Covers / Does not cover constraints.
3. If one sentence contains several covered targets, return several segments, one per target.
4. Assign the most specific concrete aspect before general EXPERIENCE unless the unit is only about overall feeling.
5. Never assign BRANDING from a hotel name alone, and never assign LOYALTY from satisfaction alone.
6. If one sentence contains conflicting opinions or contrastive polarity, split them into separate opinion units before sentiment classification.

Sub-aspect cues from the annotation sheet:
- Facility targets: room/bed/sleep quality; bathroom/sanitary area; interior design/ambience; building infrastructure; noise/environmental comfort; air conditioning/ventilation; physical security systems; view/surroundings/location.
- Amenity targets: WiFi/connectivity; food and beverage; pool/water facilities; gym/spa/wellness; entertainment/recreation; transportation/parking; in-room utilities; laundry/utility services.
- Service targets: staff attitude/hospitality; operational efficiency such as check-in/check-out/booking; support and problem resolution; communication/language ability.
- Experience targets: overall stay experience; emotional comfort/relaxation/atmosphere; value for money; subjective safety feeling.
- Branding targets: brand reputation/reliability; luxury or premium perception.
- Loyalty targets: revisit intention; recommendation intention; customer preference or attachment.

Boundary rules:
- Prefer the most concrete aspect mentioned. A concrete room, bathroom, view, location, food, WiFi, pool, spa, shuttle, or staff comment should not be labeled experience just because the sentiment is emotional.
- Food, breakfast, buffet, restaurant, cafe, bar, drinks, and menu quality are amenity unless the text specifically evaluates staff behavior or service process.
- Room condition, bathroom, bed, cleanliness of room, furniture, interior design, lighting, hallway, elevator, building, air conditioning, noise insulation, views, surroundings, and location are facility.
- CCTV, locks, keycards, guards, gates, alarms, surveillance, and other physical safety systems are facility. The guest's feeling of being safe or unsafe is experience.
- Parking area physical condition can be facility; parking availability, valet, shuttle, airport transfer, taxi service, and transport support are amenity.
- Brand/hotel name alone is not branding. Use branding only for reputation, chain standard, premium/luxury positioning, expected quality, or brand consistency.
- General satisfaction is experience. Use loyalty only when there is explicit future behavior or attachment: return, revisit, stay again, recommend, favorite/preferred hotel, membership, or long-term preference.
""".strip()

SENTIMENT_ANNOTATION_PROMPT = """
Use aspect-specific polarity cues from the annotation sheet, but do not classify by keyword matching alone.

Sentiment decision rules:
- POSITIVE: the opinion unit is overall favorable toward the specified aspect target.
- NEGATIVE: the opinion unit is overall unfavorable toward the specified aspect target.
- NEUTRAL: the unit is factual, descriptive, acceptable/average, or mixed without a dominant polarity.

Mixed and contrast handling:
- If the input still contains conflicting opinions that should have been split, classify the dominant polarity only when it is clear from the text.
- If praise and complaint are balanced or the text says something is acceptable despite a limitation, use neutral.
- Example: "The room was small but acceptable" is neutral unless the wording clearly emphasizes dissatisfaction.
- Example: "Breakfast was delicious but very expensive" should normally have been split before sentiment classification into positive food quality and negative price; if it appears as one unsplit unit, choose the dominant polarity only if explicit, otherwise neutral.

Polarity cues:
- Positive examples: clean, spotless, spacious, comfortable, modern, fast, stable, delicious, varied, refreshing, convenient, punctual, friendly, helpful, responsive, smooth, clear, enjoyable, memorable, worth it, safe, reputable, premium, favorite, definitely return, highly recommend.
- Negative examples: dirty, cramped, noisy, broken, smelly, outdated, slow, weak, unstable, bland, cold, repetitive, crowded, limited, delayed, rude, ignored, confusing, disappointing, overpriced, unsafe, unreliable, low-standard, never return, would not recommend.
- Neutral examples: standard, average, acceptable, decent, basic, normal, okay, usable, available, fair, moderate, may return, can recommend.
""".strip()

# These definitions intentionally follow the current framework. Some manual
# labels can still disagree at natural boundaries such as experience vs
# facility/amenity, branding vs loyalty/experience, and service vs amenity.
# Treat those mismatches as evaluation/annotation caveats before tuning rules.

REF_ID_RE = re.compile(r"^(?P<data_source>.+)_(?P<hotel_id>[^_]+)_(?P<review_index>\d+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+|\n+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
POSITIVE_RE = re.compile(
    r"\b(good|great|excellent|amazing|nice|clean|friendly|helpful|perfect|"
    r"comfortable|love|loved|recommend|delicious|beautiful|satisfied|"
    r"tuyet|tuyệt|tot|tốt|dep|đẹp|sach|sạch|thich|thích|hai long|hài lòng|"
    r"nhiet tinh|nhiệt tình|than thien|thân thiện|ngon|thoai mai|thoải mái)\b",
    re.IGNORECASE,
)

SEMANTIC_SEGMENTATION_PROMPT_VERSION = "semantic-preseg-v3-extractive-opinion-units"
ASPECT_EXTRACTION_PROMPT_VERSION = "aspect-extract-v4-taxonomy-proposal-rule-audit"
LEGACY_ASPECT_EXTRACTION_PROMPT_VERSIONS = ("aspect-extract-v2-mixed-units",)
SENTIMENT_CLASSIFICATION_PROMPT_VERSION = "sentiment-v3-after-aspect-rule-normalization"
CLUSTER_ASSIGNMENT_PROMPT_VERSION = "cluster-assign-v7-evidence-locked-taxonomy"
FINAL_SENTIMENT_SUMMARY_PROMPT_VERSION = "final-sentiment-summary-v2-specific-claim-guardrails"
FINAL_ASPECT_SUMMARY_PROMPT_VERSION = "final-aspect-summary-v2-specific-claim-guardrails"
ALL_ASPECTS_SUMMARY_PROMPT_VERSION = "all-aspects-summary-v1"
NEGATIVE_RE = re.compile(
    r"\b(bad|poor|dirty|rude|noisy|broken|terrible|awful|disappoint|"
    r"uncomfortable|complain|expensive|weak|hard|kem|kém|ban|bẩn|on|ồn|"
    r"that vong|thất vọng|khong hai long|không hài lòng|te|tệ|hoi|hôi|"
    r"dat|đắt|cu|cũ|hong|hỏng)\b",
    re.IGNORECASE,
)

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "facility": (
        "room",
        "bed",
        "bathroom",
        "pool",
        "balcony",
        "interior",
        "furniture",
        "window",
        "soundproof",
        "phong",
        "phòng",
        "giuong",
        "giường",
        "nha tam",
        "nhà tắm",
        "ho boi",
        "hồ bơi",
        "ban cong",
        "bể bơi",
        "cua so",
        "cửa sổ",
        "cach am",
        "cách âm",
        "view",
        "sea view",
        "ocean view",
        "city view",
        "mountain view",
        "river view",
        "landscape",
        "scenery",
        "high floor",
        "location",
        "area",
        "neighborhood",
        "surroundings",
        "old quarter",
        "city center",
        "downtown",
        "beach",
        "seaside",
        "nearby",
        "near",
        "close",
        "walking distance",
        "accessible",
        "street",
        "district",
        "attraction",
        "transport",
        "station",
        "airport",
        "vi tri",
        "vị trí",
        "khu vuc",
        "khu vực",
        "xung quanh",
        "trung tam",
        "trung tâm",
        "bai bien",
        "bãi biển",
        "gan",
        "gần",
        "canh quan",
        "cảnh quan",
        "phong canh",
        "phong cảnh",
        "keycard",
        "door lock",
        "cctv",
        "safe box",
    ),
    "amenity": (
        "parking",
        "spa",
        "massage",
        "restaurant",
        "cafe",
        "bar",
        "gym",
        "wifi",
        "internet",
        "breakfast",
        "payment",
        "pool",
        "swimming pool",
        "buffet",
        "food",
        "beverage",
        "drink",
        "menu",
        "cuisine",
        "coffee",
        "tea",
        "dessert",
        "minibar",
        "kettle",
        "netflix",
        "tv",
        "shuttle",
        "airport transfer",
        "taxi service",
        "valet",
        "transport service",
        "laundry",
        "bai dau xe",
        "bãi đậu xe",
        "nha hang",
        "nhà hàng",
        "quan ca phe",
        "quán cà phê",
        "an sang",
        "ăn sáng",
        "bua sang",
        "bữa sáng",
        "buffet sang",
        "buffet sáng",
        "do an",
        "đồ ăn",
        "thuc an",
        "thức ăn",
        "mon an",
        "món ăn",
        "do uong",
        "đồ uống",
        "ho boi",
        "hồ bơi",
        "be boi",
        "bể bơi",
        "thanh toan",
        "thanh toán",
        "giat ui",
        "giặt ủi",
    ),
    "service": (
        "staff",
        "reception",
        "service",
        "employee",
        "host",
        "manager",
        "concierge",
        "waiter",
        "waitress",
        "housekeeping",
        "room service",
        "check-in",
        "check out",
        "support",
        "assistance",
        "response",
        "solve",
        "issue",
        "complaint",
        "communication",
        "english speaking",
        "nhan vien",
        "nhân viên",
        "le tan",
        "lễ tân",
        "phuc vu",
        "phục vụ",
        "quan ly",
        "quản lý",
        "ho tro",
        "hỗ trợ",
        "phan hoi",
        "phản hồi",
        "giai quyet",
        "giải quyết",
        "van de",
        "vấn đề",
        "chu",
        "chủ",
    ),
    "experience": (
        "experience",
        "atmosphere",
        "relax",
        "quiet",
        "stay",
        "noise",
        "overall",
        "view",
        "scenery",
        "landscape",
        "value",
        "worth",
        "worth it",
        "price",
        "cost",
        "affordable",
        "expensive",
        "safe",
        "safety",
        "secure",
        "dangerous",
        "risky",
        "trai nghiem",
        "trải nghiệm",
        "khong khi",
        "không khí",
        "thu gian",
        "thư giãn",
        "yen tinh",
        "yên tĩnh",
        "ky nghi",
        "kỳ nghỉ",
        "khung canh",
        "khung cảnh",
        "canh quan",
        "cảnh quan",
        "gia",
        "giá",
        "dang tien",
        "đáng tiền",
        "hop ly",
        "hợp lý",
        "an toan",
        "an toàn",
    ),
    "branding": (
        "brand",
        "reputation",
        "standard",
        "expectation",
        "expectations",
        "expected",
        "luxury",
        "premium",
        "five-star",
        "5-star",
        "star",
        "best hotel",
        "worst hotel",
        "property",
        "thuong hieu",
        "thương hiệu",
        "danh tiếng",
        "danh tieng",
        "tiêu chuẩn",
        "tieu chuan",
        "kỳ vọng",
        "ky vong",
        "mong đợi",
        "mong doi",
        "đẳng cấp",
        "dang cap",
        "sang trọng",
        "sang trong",
        "cao cấp",
        "cao cap",
        "tốt nhất",
        "tot nhat",
    ),
    "loyalty": (
        "return",
        "come back",
        "recommend",
        "next time",
        "will be back",
        "favorite",
        "favourite",
        "preferred",
        "revisit",
        "stay again",
        "quay lai",
        "quay lại",
        "gioi thieu",
        "giới thiệu",
        "lan sau",
        "lần sau",
        "yeu thich",
        "yêu thích",
    ),
}

BRANDING_ENTITY_RE = re.compile(
    r"\b("
    r"hotel|resort|property|brand|chain|lotte|marriott|vinpearl|amiana|bespoke|trendy|la siesta|"
    r"khach san|khách sạn|khu nghi duong|khu nghỉ dưỡng|thuong hieu|thương hiệu"
    r")\b",
    re.IGNORECASE,
)
BRANDING_JUDGMENT_RE = re.compile(
    r"\b("
    r"expect(?:ed|ation|ations)?|exceed(?:ed)? expectations?|meet(?:s|ing)? expectations?|"
    r"not as expected|standard|high standard|five[- ]?star|5[- ]?star|"
    r"reputation|deserve(?:s|d)?|worth(?:y)?|premium|luxury|luxurious|"
    r"best hotel|best hotels|worst hotel|worst hotels|one of the best|one of the worst|"
    r"ky vong|kỳ vọng|mong doi|mong đợi|vuot mong doi|vượt mong đợi|"
    r"tieu chuan|tiêu chuẩn|danh tieng|danh tiếng|xung dang|xứng đáng|"
    r"dang cap|đẳng cấp|sang trong|sang trọng|cao cap|cao cấp|"
    r"tot nhat|tốt nhất|te nhat|tệ nhất|hang dau|hàng đầu"
    r")\b",
    re.IGNORECASE,
)
LOCATION_CONVENIENCE_RE = re.compile(
    r"\b("
    r"location|near|nearby|center|central|walk|walking|convenient|easy to get|"
    r"vi tri|vị trí|gan|gần|trung tam|trung tâm|di lai|đi lại|di chuyen|di chuyển|"
    r"thuan tien|thuận tiện|cho lon|chợ|bien|biển|diem tham quan|điểm tham quan|checkin"
    r")\b",
    re.IGNORECASE,
)
FOOD_CONTEXT_RE = re.compile(
    r"\b(food|meal|breakfast|restaurant|dish|drink|do an|đồ ăn|mon an|món ăn|"
    r"bua sang|bữa sáng|an sang|ăn sáng|nha hang|nhà hàng|buffet|cafe|bar|"
    r"do uong|đồ uống|thuc an|thức ăn|coffee|tea|dessert|menu|cuisine)\b",
    re.IGNORECASE,
)
FOOD_QUALITY_RE = re.compile(
    r"\b(delicious|tasty|good|bad|poor|cold|slow|quality|ngon|do|dở|te|tệ|"
    r"chat luong|chất lượng|fresh|varied|flavorful|limited|bland|repetitive|"
    r"high quality|poor quality)\b",
    re.IGNORECASE,
)
LOYALTY_INTENT_RE = re.compile(
    r"\b("
    r"will (?:definitely )?(?:return|come back|stay again|revisit)|"
    r"would (?:definitely )?(?:return|come back|stay again|revisit)|"
    r"definitely (?:return|come back|stay again|revisit|recommend)|"
    r"highly recommend|strongly recommend|would not recommend|cannot recommend|can't recommend|"
    r"recommend(?:ed|ing)? (?:this|the)? ?(?:hotel|resort|place|property|stay)|"
    r"recommend(?:ed|ing)? (?:to|for) (?:friends|family|others|everyone|anyone)|"
    r"come back|stay again|visit again|revisit|next time|will be back|"
    r"favorite hotel|favourite hotel|preferred hotel|loyalty|membership|"
    r"se quay lai|sẽ quay lại|quay lai nua|quay lại nữa|quay lai|quay lại|"
    r"lan sau|lần sau|gioi thieu|giới thiệu|de xuat|đề xuất|"
    r"yeu thich|yêu thích|khach quen|khách quen"
    r")\b",
    re.IGNORECASE,
)
SERVICE_HUMAN_RE = re.compile(
    r"\b(staff|staffs|receptionist|reception|employee|host|manager|concierge|waiter|waitress|"
    r"housekeeping|room service|service staff|restaurant service|support|assistance|"
    r"response|solve|issue|complaint|check[- ]?in|check[- ]?out|communication|"
    r"nhan vien|nhân viên|le tan|lễ tân|phuc vu|phục vụ|quan ly|quản lý|"
    r"ho tro|hỗ trợ|phan hoi|phản hồi|giai quyet|giải quyết|van de|vấn đề|"
    r"boi ban|bồi bàn|chu|chủ|they|he|she|ban|bạn|cac ban|các bạn)\b",
    re.IGNORECASE,
)
SERVICE_ACTION_RE = re.compile(
    r"\b("
    r"organised|organized|arranged|booked|reserved|help|helped|helpful|assist|assisted|"
    r"support|supported|solved|welcomed|greeted|responded|took care|went the extra mile|"
    r"to chuc|tổ chức|sap xep|sắp xếp|dat giup|đặt giúp|ho tro|hỗ trợ|giup|giúp|"
    r"giai quyet|giải quyết|chao don|chào đón|don tiep|đón tiếp"
    r")\b",
    re.IGNORECASE,
)
AMENITY_EXISTENCE_RE = re.compile(
    r"\b(has|have|had|with|available|provided|offer|include|included|co|có|"
    r"cung cap|cung cấp|bao gom|bao gồm|mien phi|miễn phí)\b",
    re.IGNORECASE,
)
PHYSICAL_SECURITY_RE = re.compile(
    r"\b("
    r"cctv|camera|security camera|surveillance|guard|gate|alarm|lock|door lock|"
    r"keycard|safe box|security system|"
    r"bao ve|bảo vệ|camera an ninh|cong|cổng|khoa cua|khóa cửa|the tu|thẻ từ|"
    r"he thong an ninh|hệ thống an ninh"
    r")\b",
    re.IGNORECASE,
)
SUBJECTIVE_SAFETY_RE = re.compile(
    r"\b("
    r"safe|safety|secure|unsafe|unsecured|dangerous|risky|reassuring|"
    r"felt safe|feel safe|feels safe|"
    r"an toan|an toàn|khong an toan|không an toàn|nguy hiem|nguy hiểm|rui ro|rủi ro|"
    r"yen tam|yên tâm|bat an|bất an"
    r")\b",
    re.IGNORECASE,
)
FACILITY_PHYSICAL_CONTEXT_RE = re.compile(
    r"\b(room|bedroom|balcony|window|wall|door|soundproof|bathroom|design|"
    r"phong|phòng|ban cong|cửa sổ|cua so|tuong|tường|cua|cửa|cach am|cách âm|"
    r"nha tam|nhà tắm|thiet ke|thiết kế|floor|elevator|lobby|hallway|corridor|"
    r"building|furniture|sofa|chair|table|aircon|air conditioning|ac|keycard|"
    r"door lock|security camera|cctv|safe box|sanh|sảnh|hanh lang|hành lang|"
    r"toa nha|tòa nhà|thang may|thang máy|noi that|nội thất|dieu hoa|điều hòa)\b",
    re.IGNORECASE,
)
VIEW_NOISE_CONTEXT_RE = re.compile(
    r"\b(view|quiet|noise|noisy|sound|scenery|landscape|nhin|nhìn|yen tinh|yên tĩnh|"
    r"on|ồn|tieng on|tiếng ồn|cach am|cách âm|khung canh|khung cảnh|canh quan|cảnh quan)\b",
    re.IGNORECASE,
)
FACILITY_STRONG_ANCHOR_RE = re.compile(
    r"\b("
    r"room|rooms|bed|beds|mattress|pillow|bathroom|toilet|shower|tub|sink|"
    r"aircon|air conditioning|ac|heater|lighting|window|wall|door|floor|"
    r"furniture|sofa|chair|table|desk|closet|balcony|lobby|hallway|corridor|"
    r"elevator|building|interior|decor|noise|noisy|quiet|soundproof|view|"
    r"phong|phòng|giuong|giường|nha tam|nhà tắm|voi sen|vòi sen|bon tam|bồn tắm|"
    r"dieu hoa|điều hòa|anh sang|ánh sáng|cua so|cửa sổ|tuong|tường|cua|cửa|"
    r"noi that|nội thất|ban cong|ban công|sanh|sảnh|hanh lang|hành lang|"
    r"thang may|thang máy|toa nha|tòa nhà|tieng on|tiếng ồn|on|ồn|"
    r"cach am|cách âm|tam nhin|tầm nhìn"
    r")\b",
    re.IGNORECASE,
)
AMENITY_STRONG_ANCHOR_RE = re.compile(
    r"\b("
    r"wifi|internet|breakfast|buffet|food|meal|restaurant|dish|drink|beverage|"
    r"coffee|tea|menu|cuisine|cafe|bar|pool|swimming pool|spa|sauna|massage|"
    r"gym|fitness center|minibar|mini-bar|kettle|tv|netflix|parking|garage|"
    r"valet|shuttle|airport transfer|taxi service|laundry|payment|"
    r"bua sang|bữa sáng|an sang|ăn sáng|do an|đồ ăn|mon an|món ăn|thuc an|thức ăn|"
    r"nha hang|nhà hàng|do uong|đồ uống|ho boi|hồ bơi|be boi|bể bơi|"
    r"bai dau xe|bãi đậu xe|giu xe|giữ xe|dua don|đưa đón|giat ui|giặt ủi|"
    r"thanh toan|thanh toán"
    r")\b",
    re.IGNORECASE,
)
SERVICE_STRONG_ANCHOR_RE = re.compile(
    r"\b("
    r"staff|staffs|employee|front desk|reception|receptionist|manager|concierge|"
    r"housekeeping|waiter|waitress|service staff|check[- ]?in|check[- ]?out|"
    r"responsive|responded|response|helped|helpful|assisted|support|solved|"
    r"welcomed|greeted|professional|rude|attentive|"
    r"nhan vien|nhân viên|le tan|lễ tân|quan ly|quản lý|boi ban|bồi bàn|"
    r"check in|check out|ho tro|hỗ trợ|phan hoi|phản hồi|giai quyet|giải quyết|"
    r"chao don|chào đón|don tiep|đón tiếp|chuyen nghiep|chuyên nghiệp|"
    r"tho lo|thô lỗ|chu dao|chu đáo"
    r")\b",
    re.IGNORECASE,
)
EXPERIENCE_STRONG_ANCHOR_RE = re.compile(
    r"\b("
    r"pet[- ]?friendly|dog[- ]?friendly|cat[- ]?friendly|pets?|dogs?|atmosphere|"
    r"ambience|vibe|overall stay|overall experience|experience|stay feeling|"
    r"memorable|relaxing|relax|peaceful|enjoyable|felt|feel|"
    r"khong khi|không khí|trai nghiem|trải nghiệm|cam giac|cảm giác|"
    r"thu gian|thư giãn|yen binh|yên bình|de chiu|dễ chịu|dang nho|đáng nhớ|"
    r"than thien voi thu cung|thân thiện với thú cưng"
    r")\b",
    re.IGNORECASE,
)
FACILITY_LOCATION_VIEW_RE = re.compile(
    r"\b("
    r"location|near|nearby|center|central|city center|downtown|old quarter|area|"
    r"neighborhood|surroundings|beach|seaside|street|district|attraction|transport|"
    r"station|airport|view|sea view|ocean view|city view|mountain view|river view|"
    r"landscape|scenery|high floor|window view|"
    r"vi tri|vị trí|gan|gần|trung tam|trung tâm|khu vuc|khu vực|xung quanh|"
    r"bai bien|bãi biển|duong|đường|quan|quận|canh quan|cảnh quan|phong canh|"
    r"phong cảnh|tam nhin|tầm nhìn|view|tang cao|tầng cao"
    r")\b",
    re.IGNORECASE,
)
AMENITY_CONTEXT_RE = re.compile(
    r"\b("
    r"wifi|internet|breakfast|buffet|pool|swimming pool|spa|sauna|massage|gym|"
    r"fitness center|restaurant|cafe|bar|food|beverage|drink|menu|cuisine|meal|"
    r"coffee|tea|dessert|minibar|kettle|tv|netflix|parking|garage|shuttle|"
    r"airport transfer|taxi service|valet|laundry|payment|"
    r"an sang|ăn sáng|bua sang|bữa sáng|ho boi|hồ bơi|be boi|bể bơi|"
    r"nha hang|nhà hàng|do an|đồ ăn|thuc an|thức ăn|mon an|món ăn|do uong|"
    r"đồ uống|bai dau xe|bãi đậu xe|giat ui|giặt ủi|thanh toan|thanh toán"
    r")\b",
    re.IGNORECASE,
)
TRANSPORT_AMENITY_RE = re.compile(
    r"\b("
    r"parking|garage|shuttle|airport transfer|taxi service|valet|transport service|"
    r"bai dau xe|bãi đậu xe|dua don|đưa đón|xe dua don|xe đưa đón|"
    r"taxi|giu xe|giữ xe"
    r")\b",
    re.IGNORECASE,
)
EXPERIENCE_FEELING_RE = re.compile(
    r"\b(atmosphere|relax|relaxing|peaceful|comfortable|vibe|overall|experience|stay|"
    r"khong khi|không khí|thu gian|thư giãn|yen binh|yên bình|de chiu|dễ chịu|"
    r"thoai mai|thoải mái|cam giac|cảm giác|trai nghiem|trải nghiệm|ky nghi|kỳ nghỉ)\b",
    re.IGNORECASE,
)
VALUE_CONTEXT_RE = re.compile(
    r"\b("
    r"value|worth(?: it)?|price|cost|affordable|reasonable|overpriced|costly|expensive|"
    r"waste(?: of money)?|money|"
    r"gia|giá|dang tien|đáng tiền|dang dong tien|đáng đồng tiền|"
    r"hop ly|hợp lý|dat|đắt|mac|mắc|phi tien|phí tiền"
    r")\b",
    re.IGNORECASE,
)
INTERIOR_DESIGN_RE = re.compile(
    r"\b("
    r"interior|decor|decoration|furniture|design|lighting|aesthetic|theme|"
    r"noi that|nội thất|trang tri|trang trí|thiet ke|thiết kế|anh sang|ánh sáng"
    r")\b",
    re.IGNORECASE,
)
STYLE_CONTEXT_RE = re.compile(r"\b(style|vibe|phong cach|phong cách)\b", re.IGNORECASE)


@dataclass
class SentenceChoice:
    text: str = ""
    score: float = -1.0


@dataclass
class AspectSummaryMetrics:
    rouge1_recall: float = 0.0
    rouge2_recall: float = 0.0
    rouge_l_recall: float = 0.0
    coverage_score: float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall: float = 0.0
    bertscore_f1: float = 0.0
    bertscore_available: bool = False
    reference_sentence_count: int = 0
    reference_token_count: int = 0
    summary_token_count: int = 0


@dataclass
class AspectSegment:
    aspect: str
    segment_text: str
    detected_language: str
    normalized_text_vi: str
    normalized_text_en: str
    confidence: float = 0.0


@dataclass
class AspectAggregate:
    counts: Counter[str] = field(default_factory=Counter)
    confidence_sums: Counter[str] = field(default_factory=Counter)
    cluster_assignment_source_counts: Counter[str] = field(default_factory=Counter)
    representative: dict[str, SentenceChoice] = field(
        default_factory=lambda: {sent: SentenceChoice() for sent in SENTIMENTS}
    )
    representative_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    reference_unigrams: Counter[str] = field(default_factory=Counter)
    reference_bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)
    reference_token_count: int = 0
    reference_sentence_count: int = 0
    reference_token_sequence: list[str] = field(default_factory=list)
    reference_texts: list[str] = field(default_factory=list)
    summary_vi: str = ""
    summary_en: str = ""

    def add(
        self,
        sentiment: str,
        confidence: float,
        text: str,
        aspect_score: float,
        sentence_id: str,
        keep_debug_samples: int,
        sample_payload: dict[str, Any] | None = None,
        keep_reference_text: bool = True,
        keep_reference_stats: bool = True,
        cluster_assignment_source: str = "",
    ) -> None:
        sentiment = normalize_sentiment(sentiment)
        confidence = clamp_confidence(confidence)
        self.counts[sentiment] += 1
        self.confidence_sums[sentiment] += confidence
        if not hasattr(self, "cluster_assignment_source_counts") or self.cluster_assignment_source_counts is None:
            self.cluster_assignment_source_counts = Counter()
        self.cluster_assignment_source_counts[normalize_cluster_assignment_source(cluster_assignment_source)] += 1
        self.add_reference_text(text, keep_reference_text, keep_reference_stats)

        clarity = min(len(WORD_RE.findall(text)) / 14.0, 1.0)
        rep_score = (0.62 * confidence) + (0.28 * aspect_score) + (0.10 * clarity)
        if rep_score > self.representative[sentiment].score:
            self.representative[sentiment] = SentenceChoice(text=text, score=rep_score)
            meta = dict(sample_payload or {})
            meta["text"] = text
            meta["confidence"] = confidence
            meta["aspect_score"] = aspect_score
            self.representative_meta[sentiment] = meta

        if keep_debug_samples > 0 and len(self.samples) < keep_debug_samples:
            sample = {
                "sentence_id": sentence_id,
                "sentence_text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "aspect_score": aspect_score,
            }
            if sample_payload:
                sample.update(sample_payload)
            self.samples.append(sample)

    def add_reference_text(
        self,
        text: str,
        keep_reference_text: bool = True,
        keep_reference_stats: bool = True,
    ) -> None:
        if not keep_reference_stats:
            return
        tokens = tokenize_text(text)
        if not tokens:
            return
        if keep_reference_text:
            self.reference_texts.append(text)
        self.reference_token_sequence.extend(tokens)
        self.reference_unigrams.update(tokens)

        def build_cluster_sentence_summary_rows(final_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for row in final_summary_rows:
                aspect = str(row.get("aspect", ""))
                if aspect not in ASPECT_NAMES:
                    continue
                for sentiment in SENTIMENTS:
                    count = int(row.get(f"{sentiment}_count", 0) or 0)
                    summary = clean_text(row.get(f"{sentiment}_summary", ""))
                    if count <= 0 and not summary:
                        continue
                    rows.append(
                        {
                            "hotel_id": row.get("hotel_id", ""),
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "count": count,
                            "summary_sentence": summary,
                            "clusters": row.get(f"{sentiment}_clusters", "[]"),
                        }
                    )
            return rows

        def build_aspect_summary_only_rows(final_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "hotel_id": row.get("hotel_id", ""),
                    "aspect": row.get("aspect", ""),
                    "aspect_summary": row.get("overall_aspect_summary", ""),
                }
                for row in final_summary_rows
                if str(row.get("aspect", "")) in ASPECT_NAMES
            ]

        def build_hotel_overall_summary_rows(final_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "hotel_id": row.get("hotel_id", ""),
                    "overall_hotel_summary": normalize_one_sentence_summary(row.get("overall_aspect_summary", "")),
                    "positive_summary": row.get("positive_summary", ""),
                    "negative_summary": row.get("negative_summary", ""),
                    "neutral_summary": row.get("neutral_summary", ""),
                }
                for row in final_summary_rows
                if str(row.get("aspect", "")) == ALL_ASPECTS_SUMMARY_KEY
            ]
        self.reference_bigrams.update(make_ngrams(tokens, 2))
        self.reference_token_count += len(tokens)
        self.reference_sentence_count += 1

    def final_sentiment(self) -> str:
        max_count = max((self.counts.get(s, 0) for s in SENTIMENTS), default=0)
        if max_count <= 0:
            return "neutral"
        candidates = [s for s in SENTIMENTS if self.counts.get(s, 0) == max_count]
        if len(candidates) == 1:
            return candidates[0]
        max_conf = max(self.confidence_sums.get(s, 0.0) for s in candidates)
        conf_candidates = [s for s in candidates if self.confidence_sums.get(s, 0.0) == max_conf]
        if "neutral" in conf_candidates:
            return "neutral"
        return conf_candidates[0]

    def final_representative(self) -> str:
        sentiment = self.final_sentiment()
        return self.representative.get(sentiment, SentenceChoice()).text

    def compute_summary_metrics(
        self,
        summary_text: str,
        bertscore_language: str = "en",
        enable_bertscore: bool = True,
    ) -> AspectSummaryMetrics:
        summary_tokens = tokenize_text(summary_text)
        if not summary_tokens or self.reference_token_count <= 0 or self.reference_sentence_count <= 0:
            return AspectSummaryMetrics()

        summary_unigrams = Counter(summary_tokens)
        summary_bigrams = Counter(make_ngrams(summary_tokens, 2))
        bertscore = (
            bertscore_summary(summary_text, self.reference_texts, bertscore_language)
            if enable_bertscore
            else empty_bertscore()
        )
        return AspectSummaryMetrics(
            rouge1_recall=ngram_recall(summary_unigrams, self.reference_unigrams),
            rouge2_recall=ngram_recall(summary_bigrams, self.reference_bigrams),
            rouge_l_recall=rouge_l_recall(summary_tokens, self.reference_token_sequence),
            coverage_score=coverage_score(summary_tokens, self.reference_unigrams),
            bertscore_precision=bertscore.get("precision", 0.0),
            bertscore_recall=bertscore.get("recall", 0.0),
            bertscore_f1=bertscore.get("f1", 0.0),
            bertscore_available=bool(bertscore.get("available", False)),
            reference_sentence_count=self.reference_sentence_count,
            reference_token_count=self.reference_token_count,
            summary_token_count=len(summary_tokens),
        )


def make_final_summary_sentiment_buckets() -> dict[str, FinalSentimentBucket]:
    return {sentiment: FinalSentimentBucket() for sentiment in SENTIMENTS}


def make_final_summary_aspect_buckets() -> dict[str, dict[str, FinalSentimentBucket]]:
    return {aspect: make_final_summary_sentiment_buckets() for aspect in ASPECT_NAMES}


@dataclass
class HotelAggregate:
    data_source: str
    hotel_id: str
    review_indexes: set[str] = field(default_factory=set)
    sentence_count: int = 0
    aspects: dict[str, AspectAggregate] = field(
        default_factory=lambda: {aspect: AspectAggregate() for aspect in ASPECT_NAMES}
    )

    def add_sentence(
        self,
        review_index: str,
        sentence_id: str,
        text: str,
        aspect: str,
        sentiment: str,
        confidence: float,
        aspect_score: float,
        keep_debug_samples: int,
        sample_payload: dict[str, Any] | None = None,
        keep_reference_text: bool = True,
        keep_reference_stats: bool = True,
        cluster_assignment_source: str = "",
    ) -> None:
        self.review_indexes.add(str(review_index))
        self.sentence_count += 1
        aspect = normalize_aspect(aspect, text)
        self.aspects[aspect].add(
            sentiment=sentiment,
            confidence=confidence,
            text=text,
            aspect_score=aspect_score,
            sentence_id=sentence_id,
            keep_debug_samples=keep_debug_samples,
            sample_payload=sample_payload,
            keep_reference_text=keep_reference_text,
            keep_reference_stats=keep_reference_stats,
            cluster_assignment_source=cluster_assignment_source,
        )


@dataclass
class ProfileAspectDecision:
    sentiment: str = ""
    representative_sentence: str = ""
    note: str = ""


ProfileDecisions = dict[str, dict[str, ProfileAspectDecision]]


@dataclass
class OpinionCluster:
    label: str
    code: str = ""
    count: int = 0
    confidence_sum: float = 0.0
    samples: list[str] = field(default_factory=list)
    token_counts: Counter[str] = field(default_factory=Counter)
    descriptor_counts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        text: str,
        confidence: float,
        sample_limit: int,
        sample_char_limit: int,
        aspect: str = "",
    ) -> None:
        cleaned = clean_text(text)
        self.count += 1
        self.confidence_sum += clamp_confidence(confidence)
        self.token_counts.update(cluster_tokens(cleaned))
        if not hasattr(self, "descriptor_counts") or self.descriptor_counts is None:
            self.descriptor_counts = Counter()
        self.descriptor_counts.update(extract_cluster_descriptors(cleaned, aspect))
        if cleaned and len(self.samples) < sample_limit:
            self.samples.append(clip_text(cleaned, sample_char_limit))

    def average_confidence(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.confidence_sum / self.count


@dataclass
class FinalSentimentBucket:
    count: int = 0
    confidence_sum: float = 0.0
    samples: list[str] = field(default_factory=list)
    clusters: dict[str, OpinionCluster] = field(default_factory=dict)
    reference_unigrams: Counter[str] = field(default_factory=Counter)
    reference_bigrams: Counter[tuple[str, str]] = field(default_factory=Counter)
    reference_token_count: int = 0
    reference_sentence_count: int = 0
    reference_token_sequence: list[str] = field(default_factory=list)
    reference_texts: list[str] = field(default_factory=list)

    def add(
        self,
        text: str,
        confidence: float,
        sample_limit: int,
        sample_char_limit: int,
        keep_reference_text: bool = True,
        keep_reference_stats: bool = True,
        aspect: str = "",
        sentiment: str = "",
        cluster_similarity_threshold: float = 0.45,
        cluster_sample_limit: int = 5,
        cluster_code: str = "",
        cluster_label: str = "",
        cluster_descriptors: list[str] | None = None,
    ) -> None:
        self.count += 1
        self.confidence_sum += clamp_confidence(confidence)
        cleaned = clean_text(text)
        if keep_reference_stats:
            tokens = tokenize_text(cleaned)
            if tokens:
                if keep_reference_text:
                    self.reference_texts.append(cleaned)
                self.reference_token_sequence.extend(tokens)
                self.reference_unigrams.update(tokens)
                self.reference_bigrams.update(make_ngrams(tokens, 2))
                self.reference_token_count += len(tokens)
                self.reference_sentence_count += 1
        if cleaned and len(self.samples) < sample_limit:
            self.samples.append(clip_text(cleaned, sample_char_limit))
        if cleaned and aspect in ASPECTS:
            self.add_cluster(
                aspect=aspect,
                sentiment=sentiment,
                text=cleaned,
                confidence=confidence,
                sample_limit=cluster_sample_limit,
                sample_char_limit=sample_char_limit,
                similarity_threshold=cluster_similarity_threshold,
                assigned_cluster_code=cluster_code,
                assigned_cluster_label=cluster_label,
                assigned_descriptors=cluster_descriptors,
            )

    def add_cluster(
        self,
        aspect: str,
        sentiment: str,
        text: str,
        confidence: float,
        sample_limit: int,
        sample_char_limit: int,
        similarity_threshold: float,
        assigned_cluster_code: str = "",
        assigned_cluster_label: str = "",
        assigned_descriptors: list[str] | None = None,
    ) -> None:
        if not hasattr(self, "clusters") or self.clusters is None:
            self.clusters = {}
        candidate_tokens = cluster_tokens(text)
        if not candidate_tokens and not (assigned_cluster_code or assigned_cluster_label):
            return
        cluster_code = clean_text(assigned_cluster_code)
        label = clean_text(assigned_cluster_label)
        cluster_code, label = canonical_cluster_fields(
            aspect,
            sentiment,
            text,
            cluster_code,
            label,
            assigned_descriptors,
        )
        key = "canon:" + (cluster_code or normalize_for_cache(label))
        if key not in self.clusters:
            self.clusters[key] = OpinionCluster(label=label, code=cluster_code)
        elif cluster_code and not getattr(self.clusters[key], "code", ""):
            self.clusters[key].code = cluster_code
        self.clusters[key].add(text, confidence, sample_limit, sample_char_limit, aspect=aspect)
        if assigned_descriptors:
            if not hasattr(self.clusters[key], "descriptor_counts") or self.clusters[key].descriptor_counts is None:
                self.clusters[key].descriptor_counts = Counter()
            self.clusters[key].descriptor_counts.update(
                unique_preserve_order([str(value) for value in assigned_descriptors if clean_text(value)], 18)
            )

    def average_confidence(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.confidence_sum / self.count

    def cluster_items(
        self,
        max_clusters: int,
        sample_char_limit: int,
        max_descriptors: int = 30,
    ) -> list[dict[str, Any]]:
        clusters = getattr(self, "clusters", {}) or {}
        ranked = sorted(clusters.values(), key=lambda cluster: (-cluster.count, cluster.label))
        descriptor_limit = int(max_descriptors)
        cluster_limit = int(max_clusters)

        def cluster_descriptors(cluster: OpinionCluster) -> list[str]:
            descriptor_counts = getattr(cluster, "descriptor_counts", Counter())
            max_items = descriptor_limit if descriptor_limit > 0 else len(descriptor_counts)
            return [descriptor for descriptor, _ in descriptor_counts.most_common(max_items)]

        selected_clusters = ranked[:cluster_limit] if cluster_limit > 0 else ranked
        return [
            {
                "label": cluster.label,
                "code": getattr(cluster, "code", ""),
                "measurement_scale": cluster.label,
                "count": cluster.count,
                "avg_confidence": round(cluster.average_confidence(), 6),
                "descriptors": cluster_descriptors(cluster),
                "samples": [clip_text(sample, sample_char_limit) for sample in cluster.samples if clean_text(sample)],
            }
            for cluster in selected_clusters
        ]

    def compute_summary_metrics(
        self,
        summary_text: str,
        bertscore_language: str = "en",
        enable_bertscore: bool = True,
    ) -> AspectSummaryMetrics:
        summary_tokens = tokenize_text(summary_text)
        if not summary_tokens or self.reference_token_count <= 0 or self.reference_sentence_count <= 0:
            return AspectSummaryMetrics()

        summary_unigrams = Counter(summary_tokens)
        summary_bigrams = Counter(make_ngrams(summary_tokens, 2))
        bertscore = (
            bertscore_summary(summary_text, self.reference_texts, bertscore_language)
            if enable_bertscore
            else empty_bertscore()
        )
        return AspectSummaryMetrics(
            rouge1_recall=ngram_recall(summary_unigrams, self.reference_unigrams),
            rouge2_recall=ngram_recall(summary_bigrams, self.reference_bigrams),
            rouge_l_recall=rouge_l_recall(summary_tokens, self.reference_token_sequence),
            coverage_score=coverage_score(summary_tokens, self.reference_unigrams),
            bertscore_precision=bertscore.get("precision", 0.0),
            bertscore_recall=bertscore.get("recall", 0.0),
            bertscore_f1=bertscore.get("f1", 0.0),
            bertscore_available=bool(bertscore.get("available", False)),
            reference_sentence_count=self.reference_sentence_count,
            reference_token_count=self.reference_token_count,
            summary_token_count=len(summary_tokens),
        )


class FinalSummaryAggregate:
    def __init__(
        self,
        sample_limit: int,
        sample_char_limit: int,
        cluster_similarity_threshold: float = 0.45,
        cluster_sample_limit: int = 5,
        enable_clusters: bool = True,
    ):
        self.sample_limit = int(sample_limit) if int(sample_limit) > 0 else 10**9
        self.sample_char_limit = int(sample_char_limit)
        self.cluster_similarity_threshold = max(0.0, min(1.0, float(cluster_similarity_threshold)))
        self.cluster_sample_limit = int(cluster_sample_limit) if int(cluster_sample_limit) > 0 else 10**9
        self.enable_clusters = bool(enable_clusters)
        self.group_metadata: dict[str, dict[str, str]] = {}
        self.buckets: dict[str, dict[str, dict[str, FinalSentimentBucket]]] = defaultdict(
            make_final_summary_aspect_buckets
        )

    def configure_clustering(self, cluster_similarity_threshold: float, cluster_sample_limit: int) -> None:
        self.cluster_similarity_threshold = max(0.0, min(1.0, float(cluster_similarity_threshold)))
        self.cluster_sample_limit = int(cluster_sample_limit) if int(cluster_sample_limit) > 0 else 10**9

    def add(
        self,
        entity_id: str,
        data_source: str,
        hotel_id: str,
        source_file: str,
        aspect: str,
        sentiment: str,
        text: str,
        confidence: float,
        keep_reference_text: bool = True,
        keep_reference_stats: bool = True,
        reroute_aspect: bool = True,
        cluster_code: str = "",
        cluster_label: str = "",
        cluster_descriptors: list[str] | None = None,
    ) -> None:
        cluster_descriptors = filter_source_faithful_descriptors(cluster_descriptors or [], text)
        if reroute_aspect:
            guardrail = guardrail_aspect_assignment(aspect, text)
            original_aspect = guardrail["original_aspect"]
            aspect = guardrail["final_aspect"]
            if aspect != original_aspect:
                # The cluster code/label was selected under the pre-guardrail aspect.
                # Reset it so final summaries cannot carry an amenity cluster into
                # facility, service, or experience after rerouting.
                cluster_code = ""
                cluster_label = ""
        else:
            aspect = canonicalize_aspect_label(aspect)
        if aspect not in ASPECTS:
            return
        sentiment = normalize_sentiment(sentiment)
        group_key = clean_text(entity_id) or clean_text(hotel_id) or clean_text(source_file)
        if not group_key:
            return
        self.group_metadata.setdefault(
            group_key,
            {
                "entity_id": group_key,
                "data_source": clean_text(data_source),
                "hotel_id": clean_text(hotel_id),
                "source_file": clean_text(source_file),
            },
        )
        self.buckets[group_key][aspect][sentiment].add(
            text,
            confidence,
            self.sample_limit,
            self.sample_char_limit,
            keep_reference_text=keep_reference_text,
            keep_reference_stats=keep_reference_stats,
            aspect=aspect if getattr(self, "enable_clusters", True) else "",
            sentiment=sentiment,
            cluster_similarity_threshold=getattr(self, "cluster_similarity_threshold", 0.45),
            cluster_sample_limit=getattr(self, "cluster_sample_limit", 5),
            cluster_code=cluster_code if getattr(self, "enable_clusters", True) else "",
            cluster_label=cluster_label if getattr(self, "enable_clusters", True) else "",
            cluster_descriptors=cluster_descriptors if getattr(self, "enable_clusters", True) else [],
        )

    def add_overall_group(self) -> None:
        overall = "__all_hotels__"
        self.group_metadata[overall] = {
            "entity_id": overall,
            "data_source": "__all_sources__",
            "hotel_id": overall,
            "source_file": "__all_files__",
        }
        for group_key, aspects in list(self.buckets.items()):
            if group_key == overall:
                continue
            for aspect in ASPECT_NAMES:
                for sentiment in SENTIMENTS:
                    source_bucket = aspects[aspect][sentiment]
                    target_bucket = self.buckets[overall][aspect][sentiment]
                    target_bucket.count += source_bucket.count
                    target_bucket.confidence_sum += source_bucket.confidence_sum
                    target_bucket.reference_unigrams.update(source_bucket.reference_unigrams)
                    target_bucket.reference_bigrams.update(source_bucket.reference_bigrams)
                    target_bucket.reference_token_count += source_bucket.reference_token_count
                    target_bucket.reference_sentence_count += source_bucket.reference_sentence_count
                    target_bucket.reference_token_sequence.extend(source_bucket.reference_token_sequence)
                    target_bucket.reference_texts.extend(source_bucket.reference_texts)
                    remaining = self.sample_limit - len(target_bucket.samples)
                    if remaining > 0:
                        target_bucket.samples.extend(source_bucket.samples[:remaining])
                    for cluster_key, source_cluster in (getattr(source_bucket, "clusters", {}) or {}).items():
                        if not hasattr(target_bucket, "clusters") or target_bucket.clusters is None:
                            target_bucket.clusters = {}
                        target_cluster = target_bucket.clusters.setdefault(
                            cluster_key,
                            OpinionCluster(
                                label=source_cluster.label,
                                code=getattr(source_cluster, "code", ""),
                            ),
                        )
                        source_code = getattr(source_cluster, "code", "")
                        if source_code and not getattr(target_cluster, "code", ""):
                            target_cluster.code = source_code
                        target_cluster.count += source_cluster.count
                        target_cluster.confidence_sum += source_cluster.confidence_sum
                        target_cluster.token_counts.update(source_cluster.token_counts)
                        if not hasattr(target_cluster, "descriptor_counts") or target_cluster.descriptor_counts is None:
                            target_cluster.descriptor_counts = Counter()
                        target_cluster.descriptor_counts.update(
                            getattr(source_cluster, "descriptor_counts", Counter())
                        )
                        remaining_cluster_samples = self.cluster_sample_limit - len(target_cluster.samples)
                        if remaining_cluster_samples > 0:
                            target_cluster.samples.extend(source_cluster.samples[:remaining_cluster_samples])

    def sorted_group_keys(self) -> list[str]:
        return sorted(
            self.buckets,
            key=lambda group_key: (
                self.group_metadata.get(group_key, {}).get("data_source", ""),
                self.group_metadata.get(group_key, {}).get("hotel_id", group_key),
                self.group_metadata.get(group_key, {}).get("source_file", ""),
            ),
        )


class ClassificationCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentence_cache (
                sentence_hash TEXT PRIMARY KEY,
                normalized_text TEXT NOT NULL,
                aspect TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason_short TEXT,
                raw_json TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_cache (
                entity_id TEXT PRIMARY KEY,
                profile_hash TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_extraction_cache (
                sentence_hash TEXT PRIMARY KEY,
                normalized_text TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_segmentation_cache (
                review_hash TEXT PRIMARY KEY,
                normalized_text TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS aspect_sentiment_cache (
                segment_hash TEXT PRIMARY KEY,
                aspect TEXT NOT NULL,
                classification_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_json TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cluster_assignment_cache (
                cluster_hash TEXT PRIMARY KEY,
                aspect TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                classification_text TEXT NOT NULL,
                cluster_code TEXT,
                cluster_label TEXT NOT NULL,
                descriptors_json TEXT,
                confidence REAL NOT NULL,
                raw_json TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_cache (
                summary_hash TEXT PRIMARY KEY,
                summary_type TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model TEXT NOT NULL,
                language TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def get_many(self, hashes: list[str]) -> dict[str, dict[str, Any]]:
        if not hashes:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT sentence_hash, aspect, sentiment, confidence, reason_short, raw_json
                FROM sentence_cache
                WHERE sentence_hash IN ({placeholders})
                """,
                batch,
            )
            for row in cur.fetchall():
                out[row[0]] = {
                    "aspect": row[1],
                    "sentiment": row[2],
                    "confidence": float(row[3]),
                    "reason_short": row[4] or "",
                    "raw_json": row[5] or "",
                    "from_cache": True,
                }
        return out

    def set_many(self, rows: dict[str, tuple[str, dict[str, Any]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = []
        for sentence_hash, (normalized_text, item) in rows.items():
            payload.append(
                (
                    sentence_hash,
                    normalized_text,
                    normalize_aspect(item.get("aspect", ""), normalized_text),
                    normalize_sentiment(item.get("sentiment", "")),
                    clamp_confidence(item.get("confidence", 0.45)),
                    str(item.get("reason_short", ""))[:200],
                    json.dumps(item, ensure_ascii=False),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO sentence_cache (
                sentence_hash, normalized_text, aspect, sentiment, confidence,
                reason_short, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sentence_hash) DO UPDATE SET
                normalized_text=excluded.normalized_text,
                aspect=excluded.aspect,
                sentiment=excluded.sentiment,
                confidence=excluded.confidence,
                reason_short=excluded.reason_short,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_profiles(self, entity_ids: list[str]) -> dict[str, tuple[str, dict[str, Any]]]:
        if not entity_ids:
            return {}
        out: dict[str, tuple[str, dict[str, Any]]] = {}
        for start in range(0, len(entity_ids), 900):
            batch = entity_ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT entity_id, profile_hash, decision_json
                FROM profile_cache
                WHERE entity_id IN ({placeholders})
                """,
                batch,
            )
            for entity_id, profile_hash_value, decision_json in cur.fetchall():
                try:
                    out[entity_id] = (profile_hash_value, json.loads(decision_json))
                except json.JSONDecodeError:
                    continue
        return out

    def set_profiles(self, rows: dict[str, tuple[str, dict[str, Any]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = [
            (entity_id, profile_hash_value, json.dumps(decision, ensure_ascii=False), now)
            for entity_id, (profile_hash_value, decision) in rows.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO profile_cache (entity_id, profile_hash, decision_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                profile_hash=excluded.profile_hash,
                decision_json=excluded.decision_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_aspect_extractions(self, hashes: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not hashes:
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT sentence_hash, raw_json
                FROM aspect_extraction_cache
                WHERE sentence_hash IN ({placeholders})
                """,
                batch,
            )
            for sentence_hash_key, raw_json in cur.fetchall():
                try:
                    value = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, list):
                    out[sentence_hash_key] = value
        return out

    def set_aspect_extractions(self, rows: dict[str, tuple[str, list[dict[str, Any]]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = [
            (
                sentence_hash_key,
                normalized_text,
                json.dumps(items, ensure_ascii=False),
                now,
            )
            for sentence_hash_key, (normalized_text, items) in rows.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO aspect_extraction_cache (
                sentence_hash, normalized_text, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sentence_hash) DO UPDATE SET
                normalized_text=excluded.normalized_text,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_semantic_segments(self, hashes: list[str]) -> dict[str, list[dict[str, Any]]]:
        if not hashes:
            return {}
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT review_hash, raw_json
                FROM semantic_segmentation_cache
                WHERE review_hash IN ({placeholders})
                """,
                batch,
            )
            for review_hash_key, raw_json in cur.fetchall():
                try:
                    value = json.loads(raw_json)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, list):
                    out[review_hash_key] = value
        return out

    def set_semantic_segments(self, rows: dict[str, tuple[str, list[dict[str, Any]]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = [
            (
                review_hash_key,
                normalized_text,
                json.dumps(items, ensure_ascii=False),
                now,
            )
            for review_hash_key, (normalized_text, items) in rows.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO semantic_segmentation_cache (
                review_hash, normalized_text, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(review_hash) DO UPDATE SET
                normalized_text=excluded.normalized_text,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_segment_sentiments(self, hashes: list[str]) -> dict[str, dict[str, Any]]:
        if not hashes:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT segment_hash, aspect, sentiment, confidence, raw_json
                FROM aspect_sentiment_cache
                WHERE segment_hash IN ({placeholders})
                """,
                batch,
            )
            for row in cur.fetchall():
                out[row[0]] = {
                    "aspect": row[1],
                    "sentiment": row[2],
                    "confidence": float(row[3]),
                    "raw_json": row[4] or "",
                    "from_cache": True,
                }
        return out

    def set_segment_sentiments(self, rows: dict[str, tuple[str, str, dict[str, Any]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = []
        for segment_hash_key, (aspect, classification_text, item) in rows.items():
            payload.append(
                (
                    segment_hash_key,
                    normalize_aspect(aspect, classification_text),
                    classification_text,
                    normalize_sentiment(item.get("sentiment", "")),
                    clamp_confidence(item.get("confidence", 0.45)),
                    json.dumps(item, ensure_ascii=False),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO aspect_sentiment_cache (
                segment_hash, aspect, classification_text, sentiment, confidence, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(segment_hash) DO UPDATE SET
                aspect=excluded.aspect,
                classification_text=excluded.classification_text,
                sentiment=excluded.sentiment,
                confidence=excluded.confidence,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_cluster_assignments(self, hashes: list[str]) -> dict[str, dict[str, Any]]:
        if not hashes:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT cluster_hash, aspect, sentiment, cluster_code, cluster_label,
                       descriptors_json, confidence, raw_json
                FROM cluster_assignment_cache
                WHERE cluster_hash IN ({placeholders})
                """,
                batch,
            )
            for row in cur.fetchall():
                try:
                    descriptors = json.loads(row[5] or "[]")
                except json.JSONDecodeError:
                    descriptors = []
                if not isinstance(descriptors, list):
                    descriptors = []
                source = ""
                if row[7]:
                    try:
                        raw_item = json.loads(row[7])
                        if isinstance(raw_item, dict):
                            source = clean_text(raw_item.get("source", ""))
                    except json.JSONDecodeError:
                        source = ""
                out[row[0]] = {
                    "aspect": row[1],
                    "sentiment": row[2],
                    "cluster_code": row[3] or "",
                    "cluster_label": row[4] or "",
                    "descriptors": [clean_text(value) for value in descriptors if clean_text(value)],
                    "confidence": float(row[6]),
                    "source": source,
                    "raw_json": row[7] or "",
                    "from_cache": True,
                }
        return out

    def set_cluster_assignments(self, rows: dict[str, tuple[str, str, str, dict[str, Any]]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = []
        for cluster_hash_key, (aspect, sentiment, classification_text, item) in rows.items():
            descriptors = unique_preserve_order(
                [str(value) for value in item.get("descriptors", []) if clean_text(value)],
                18,
            )
            payload.append(
                (
                    cluster_hash_key,
                    canonicalize_aspect_label(aspect) or normalize_aspect(aspect, classification_text),
                    normalize_sentiment(sentiment),
                    classification_text,
                    clean_text(item.get("cluster_code", "")),
                    clean_text(item.get("cluster_label", "")) or make_fallback_cluster_label(classification_text),
                    json.dumps(descriptors, ensure_ascii=False),
                    clamp_confidence(item.get("confidence", 0.45)),
                    json.dumps(item, ensure_ascii=False),
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT INTO cluster_assignment_cache (
                cluster_hash, aspect, sentiment, classification_text, cluster_code,
                cluster_label, descriptors_json, confidence, raw_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cluster_hash) DO UPDATE SET
                aspect=excluded.aspect,
                sentiment=excluded.sentiment,
                classification_text=excluded.classification_text,
                cluster_code=excluded.cluster_code,
                cluster_label=excluded.cluster_label,
                descriptors_json=excluded.descriptors_json,
                confidence=excluded.confidence,
                raw_json=excluded.raw_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def get_summaries(self, hashes: list[str]) -> dict[str, Any]:
        if not hashes:
            return {}
        out: dict[str, Any] = {}
        for start in range(0, len(hashes), 900):
            batch = hashes[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cur = self.conn.execute(
                f"""
                SELECT summary_hash, result_json
                FROM summary_cache
                WHERE summary_hash IN ({placeholders})
                """,
                batch,
            )
            for summary_hash_key, result_json in cur.fetchall():
                try:
                    out[summary_hash_key] = json.loads(result_json)
                except json.JSONDecodeError:
                    continue
        return out

    def set_summaries(self, rows: dict[str, tuple[str, str, str, str, str, Any]]) -> None:
        if not rows:
            return
        now = time.time()
        payload = [
            (
                summary_hash_key,
                summary_type,
                prompt_version,
                model,
                language,
                payload_json,
                json.dumps(result, ensure_ascii=False),
                now,
            )
            for summary_hash_key, (summary_type, prompt_version, model, language, payload_json, result) in rows.items()
        ]
        self.conn.executemany(
            """
            INSERT INTO summary_cache (
                summary_hash, summary_type, prompt_version, model, language,
                payload_json, result_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(summary_hash) DO UPDATE SET
                summary_type=excluded.summary_type,
                prompt_version=excluded.prompt_version,
                model=excluded.model,
                language=excluded.language,
                payload_json=excluded.payload_json,
                result_json=excluded.result_json,
                updated_at=excluded.updated_at
            """,
            payload,
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class PipelineCheckpointManager:
    def __init__(self, path: str, every_chunks: int, resume_payload: dict[str, Any] | None = None):
        self.path = Path(path) if path else None
        self.every_chunks = max(1, int(every_chunks))
        self._chunks_since_save = 0
        self.file_states: dict[str, dict[str, Any]] = dict((resume_payload or {}).get("file_states", {}))

    @property
    def enabled(self) -> bool:
        return self.path is not None

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        if payload.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {payload.get('version')} != {CHECKPOINT_VERSION}"
            )
        return payload

    def get_file_state(self, path: Path) -> dict[str, Any]:
        return dict(self.file_states.get(str(path.resolve()), {}))

    def update_file_state(self, path: Path, state: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.file_states[str(path.resolve())] = dict(state)

    def maybe_save(
        self,
        *,
        inputs: list[Path],
        strategy: str,
        hotels: dict[str, HotelAggregate],
        stats: dict[str, Any],
        final_summary: FinalSummaryAggregate | None,
        profile_decisions: ProfileDecisions,
        force: bool = False,
        run_completed: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self._chunks_since_save += 1
        if not force and self._chunks_since_save < self.every_chunks:
            return
        self._chunks_since_save = 0
        payload = {
            "version": CHECKPOINT_VERSION,
            "saved_at": time.time(),
            "strategy": strategy,
            "inputs": [str(path.resolve()) for path in inputs],
            "stats": stats,
            "hotels": hotels,
            "final_summary": final_summary,
            "profile_decisions": profile_decisions,
            "file_states": self.file_states,
            "run_completed": bool(run_completed),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self.path)


def build_file_progress_state(
    *,
    rows_seen: int,
    bad_ref_ids: int,
    empty_reviews: int,
    seen_hotels_for_source: dict[str, set[str]],
    skipped_min_reviews: int = 0,
    aspect_segments: int | None = None,
    completed: bool = False,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "rows_seen": int(rows_seen),
        "bad_ref_ids": int(bad_ref_ids),
        "empty_reviews": int(empty_reviews),
        "skipped_min_reviews": int(skipped_min_reviews),
        "seen_hotels_for_source": {
            source: set(values) for source, values in seen_hotels_for_source.items()
        },
        "completed": bool(completed),
    }
    if aspect_segments is not None:
        state["aspect_segments"] = int(aspect_segments)
    return state


class QwenClassifier:
    """Legacy/profile classifier: one aspect and sentiment per input sentence.

    This is useful for profile-level or simple sentence classification, but it
    is not a faithful evaluator for gold ABSA rows when the same evidence text
    contains multiple quads. Use QwenAspectSegmenter plus set matching for that
    benchmark.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def classify_batch(self, sentences: list[str]) -> list[dict[str, Any]]:
        if self.args.skip_qwen:
            return [fallback_classify(sentence) for sentence in sentences]

        numbered = [{"id": i, "text": sentence[: self.args.max_sentence_chars]} for i, sentence in enumerate(sentences)]
        user_payload = {
            "aspects": ASPECTS,
            "sentiments": {
                "positive": "hai long, vui ve, hanh phuc, danh gia tot",
                "negative": "khong hai long, that vong, phan nan, kho chiu",
                "neutral": "binh thuong, khong ro tich cuc/tieu cuc, it cam xuc",
            },
            "items": numbered,
        }
        system = (
            "You classify hotel review sentences. Return strict JSON only. "
            "Each item must use exactly one aspect key and one sentiment key. "
            "Aspect keys: facility, amenity, service, experience, branding, loyalty. "
            f"{ASPECT_ANNOTATION_PROMPT}\n"
            f"{SENTIMENT_ANNOTATION_PROMPT}\n"
            "Sentiment keys: positive, negative, neutral. "
            "Use Vietnamese hotel-domain meanings when text is Vietnamese. "
            "Keep the JSON compact and do not include explanations."
        )
        user = (
            "Classify every item in this JSON payload. Return exactly this compact schema:\n"
            '{"items":[{"id":0,"aspect":"service","sentiment":"positive","confidence":0.90}]}\n'
            "Do not add reason_short unless it is needed to disambiguate a difficult case.\n\n"
            f"Payload:\n{json.dumps(user_payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    reasoning = getattr(rsp.choices[0].message, "reasoning", "")
                    raise ValueError(
                        "empty Qwen content"
                        + (f" finish_reason={finish}" if finish else "")
                        + (f" reasoning_prefix={str(reasoning)[:120]}" if reasoning else "")
                    )
                return parse_qwen_items(content, sentences)
            except Exception as exc:  # noqa: BLE001 - Qwen endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(f"[WARN] Qwen batch failed; using fallback rules. Error: {last_error}", file=sys.stderr)
        return [fallback_classify(sentence) for sentence in sentences]


class QwenSemanticPreSegmenter:
    """Normalize reviews into source-faithful opinion units before ABSA."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def segment_batch(self, reviews: list[str]) -> list[list[dict[str, Any]]]:
        if self.args.skip_qwen:
            return [fallback_semantic_units(review, self.args.min_words) for review in reviews]

        numbered = [
            {"id": i, "text": review[: self.args.semantic_max_review_chars]}
            for i, review in enumerate(reviews)
        ]
        system = (
            "You prepare noisy human hotel reviews for aspect-based sentiment analysis. "
            "Return strict JSON only. "
            "Your job is extractive opinion-unit normalization, not summarization. "
            "Split each review into minimal self-contained opinion or observation units that preserve the source meaning. "
            "Each unit must keep the hotel target plus the opinion, fact, limitation, or request needed for ABSA. "
            "Use source wording whenever possible; do not translate, paraphrase into new concepts, or infer aspect/sentiment labels. "
            "Remove only filler, repeated words, greetings, emojis, and standalone noise when meaning is unchanged. "
            "Repair missing punctuation or broken clauses when the writer cut ideas across sentence boundaries. "
            "Merge adjacent fragments when they share the same target and only list descriptors, such as staff friendly and helpful. "
            "Keep negation and contrast in the same unit when splitting would change polarity or meaning. "
            "Drop fragments that are only 'no', 'none', 'không có', 'ko', thanks, or other non-opinion noise. "
            "Do not assign aspect, sentiment, cluster, business interpretation, or recommendations beyond the source text."
        )
        user = (
            "Process every review in this payload. Return exactly this compact schema:\n"
            '{"items":[{"id":0,"units":[{"text":"The staff were friendly and helpful","source_text":"The staff were very friendly and helpful","confidence":0.92,"flags":["merged_descriptors"]}]}]}\n'
            "Rules:\n"
            "- text is the cleaned source-faithful unit shown later as shortened_sentence.\n"
            "- source_text is the shortest source phrase/clause that supports the unit; use the full clause if unsure.\n"
            "- A unit should usually contain a target and an opinion/context, not a single descriptor by itself.\n"
            "- Keep mixed contrast together when splitting would create a false positive or false negative.\n"
            "- Merge same-target descriptor fragments instead of returning many tiny units.\n"
            "- Do not summarize a long review into one small point; preserve every hotel-related issue or praise.\n"
            "- Prefer exact deletion of filler over rewriting with synonyms.\n\n"
            f"Payload:\n{json.dumps({'items': numbered}, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.semantic_max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen semantic pre-segmentation content finish_reason={finish}")
                return parse_qwen_semantic_units_lenient(content, reviews, self.args)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(
            f"[WARN] Qwen semantic pre-segmentation failed; using sentence split fallback. Error: {last_error}",
            file=sys.stderr,
        )
        return [fallback_semantic_units(review, self.args.min_words) for review in reviews]


class QwenAspectSegmenter:
    """Split a sentence into focused opinion units before sentiment scoring."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def segment_batch(self, sentences: list[str]) -> list[list[dict[str, Any]]]:
        if self.args.skip_qwen:
            return [fallback_aspect_segments(sentence) for sentence in sentences]

        numbered = [{"id": i, "text": sentence[: self.args.max_sentence_chars]} for i, sentence in enumerate(sentences)]
        user_payload = {
            "aspects": ASPECTS,
            "items": numbered,
        }
        system = (
            "You split hotel review text into opinion units for downstream aspect analysis. Return strict JSON only. "
            "Each input item is a short hotel-review sentence or clause. "
            "A sentence may contain zero, one, or multiple opinion units about: facility, amenity, service, experience, branding, loyalty. "
            "Return one segment per opinion unit. Keep each segment faithful to the source and focused on one main idea only. "
            "If the input contains multiple aspects, split it into multiple concise opinion units and assign the most specific aspect to each one. "
            "Treat the aspect label as a taxonomy proposal that will be audited by deterministic rule-based guardrails; make the proposal easy to audit by keeping the target explicit in segment_text. "
            "Do not rely on broad emotional wording when a concrete taxonomy target is present; the rule base will prefer concrete targets such as room, bathroom, location, breakfast, WiFi, pool, parking, staff, or check-in. "
            "If one segment contains conflicting opinions, split it into separate opinion units even when they share the same aspect. "
            "For example, split 'Breakfast was delicious but very expensive' into one amenity unit about delicious breakfast and one amenity unit about expensive breakfast. "
            "Never return an empty segments array for text that praises, criticizes, or observes a hotel target such as room, bathroom, bed, cleanliness, view, location, pool, rooftop bar, breakfast, restaurant, food, WiFi, staff, host, service, check-in, price, overall stay, or hotel/property/place. "
            "When the text is hotel-related but the concrete target is broad, return an experience segment rather than an empty array. "
            "Examples that must not be empty: 'The hotel is absolutely beautiful' -> experience or facility; 'The pool and rooftop bar are amazing and the breakfast was delicious' -> separate amenity segments for pool/rooftop bar and breakfast; 'Everything was great' -> experience. "
            f"{ASPECT_ANNOTATION_PROMPT}\n"
            "For each segment, detect the source language and provide a clean Vietnamese version and a clean English version. "
            "Include optional start_char and end_char offsets when you can infer them reliably. "
            "Do not invent facts. Do not merge unrelated opinions. Do not duplicate the same detail across aspects unless the text truly states both."
        )
        user = (
            "Process every input item in this payload. Return the same id from each input item and exactly this compact JSON schema:\n"
            '{"items":[{"id":0,"segments":[{"segment_text":"Staff were helpful","aspect":"service","detected_language":"en","normalized_text_vi":"Nhân viên hỗ trợ nhiệt tình","normalized_text_en":"The staff were helpful","split_reason":"single service opinion","confidence":0.92}]}]}\n'
            "Return an empty segments array only for non-hotel filler such as 'none', 'nothing', unrelated travel notes, or text with no hotel target/opinion/observation.\n\n"
            f"Payload:\n{json.dumps(user_payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen extraction content finish_reason={finish}")
                return parse_qwen_aspect_segments_lenient(content, sentences)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(f"[WARN] Qwen aspect extraction failed; using fallback rules. Error: {last_error}", file=sys.stderr)
        return [fallback_aspect_segments(sentence) for sentence in sentences]


class QwenAspectSentimentClassifier:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def classify_batch(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        if self.args.skip_qwen:
            return [fallback_sentiment_item(item["aspect"], item["text"]) for item in items]

        payload = {
            "items": [
                {
                    "id": idx,
                    "aspect": item["aspect"],
                    "text": item["text"][: self.args.max_sentence_chars],
                }
                for idx, item in enumerate(items)
            ]
        }
        system = (
            "You classify sentiment for aspect-specific hotel review snippets. Return strict JSON only. "
            "Each input already belongs to exactly one aspect after taxonomy/rule-based aspect normalization. "
            "Choose one sentiment: positive, negative, or neutral. "
            f"{SENTIMENT_ANNOTATION_PROMPT}\n"
            "Use the snippet text only; do not infer unstated issues. "
            "Judge sentiment toward the provided aspect target only; ignore unrelated sentiment if the snippet still contains more than one target. "
            "Do not use sentiment counts, hotel reputation, or surrounding review context. "
            "Return one classification for each input id."
        )
        user = (
            "Classify every item in this payload. Return the same id from each input item and exactly this compact schema:\n"
            '{"items":[{"id":0,"aspect":"service","segment":"Staff were helpful","sentiment":"positive","confidence":0.90}]}\n\n'
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen sentiment content finish_reason={finish}")
                return parse_qwen_sentiment_items_lenient(content, items)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(f"[WARN] Qwen aspect sentiment failed; using fallback rules. Error: {last_error}", file=sys.stderr)
        return [fallback_sentiment_item(item["aspect"], item["text"]) for item in items]


class QwenClusterAssigner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def assign_batch(self, items: list[dict[str, str]], split_depth: int = 0) -> list[dict[str, Any]]:
        if self.args.skip_qwen or self.args.cluster_assignment != "llm":
            return [rule_cluster_assignment(item["aspect"], item["sentiment"], item["text"]) for item in items]

        payload = {
            "aspect_framework": aspect_cluster_framework_payload(),
            "cluster_taxonomy": cluster_taxonomy_payload(),
            "items": [
                {
                    "id": idx,
                    "aspect": item["aspect"],
                    "sentiment": item["sentiment"],
                    "text": item["text"][: self.args.max_sentence_chars],
                }
                for idx, item in enumerate(items)
            ],
        }
        system = (
            "You assign already-extracted hotel ABSA opinion units to a fixed canonical cluster taxonomy. "
            "Return strict JSON only. Each input already has an aspect and sentiment; do not change them. "
            "Use aspect_framework to stay inside the correct aspect boundary, then choose exactly one cluster_code from "
            "cluster_taxonomy for that same aspect. Do not invent new cluster labels or dynamic target labels. "
            "Do not introduce a target that is not explicitly present in the input text. For example, do not output wifi, "
            "parking, breakfast, pool, restaurant, room, bathroom, AC, or staff descriptors unless that target appears in the text. "
            "Each cluster_label is the MEASUREMENT SCALE from the annotation workbook. "
            "The cluster_label must exactly match the measurement_scale/label paired with the chosen row in cluster_taxonomy. "
            "Cluster codes are unique within each aspect; treat duplicate codes in one aspect as invalid taxonomy. "
            "Always choose the closest measurement scale from cluster_taxonomy for the item's aspect. "
            "Use surrounding context only when the current item is ambiguous, underspecified, or low-confidence; "
            "never override a clear current anchor such as breakfast, staff friendliness, room cleanliness, location, or view. "
            "Never merge positive, negative, and neutral snippets into the same output item, even if they share the same taxonomy cluster. "
            "Extract descriptors as short adjective/property/action phrases explicitly stated in the text, for example sạch, đẹp, rộng, "
            "cách âm tốt, chật, ồn, hỏng, thân thiện, dễ gần, phục vụ tốt, thạo tiếng Anh, ngon, đa dạng, đáng tiền. "
            "Descriptors must not be broad nouns like hotel, room, staff, location, service, or amenity unless they are part of a meaningful phrase. "
            "Descriptors must be supported by exact wording or an obvious translation of exact wording in the text. "
            "Do not invent facts; keep labels and descriptors source-faithful."
        )
        user = (
            "Return the same id from each input item. Return exactly this compact JSON schema:\n"
            '{"items":[{"id":0,"aspect":"service","sentiment":"positive","unit_ids":["0"],"cluster_code":"SER1","cluster_label":"Staff Friendliness","descriptors":["thân thiện","dễ gần","phục vụ tốt","thạo tiếng Anh"],"confidence":0.90}]}\n\n'
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.cluster_assignment_max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen cluster assignment content finish_reason={finish}")
                return parse_qwen_cluster_assignments_lenient(content, items)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        if len(items) > 1:
            midpoint = max(1, len(items) // 2)
            print(
                "[WARN] Qwen cluster assignment batch failed; retrying smaller LLM batches. "
                f"items={len(items)} split_depth={split_depth} Error: {last_error}",
                file=sys.stderr,
            )
            return self.assign_batch(items[:midpoint], split_depth + 1) + self.assign_batch(
                items[midpoint:],
                split_depth + 1,
            )

        print(f"[WARN] Qwen cluster assignment failed; using rule clusters. Error: {last_error}", file=sys.stderr)
        return [rule_cluster_assignment(item["aspect"], item["sentiment"], item["text"]) for item in items]


class QwenAspectSummarizer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def summarize_batch(self, items: list[dict[str, Any]]) -> list[dict[str, str]]:
        if self.args.skip_qwen:
            return [local_aspect_summary(item) for item in items]

        payload = {"items": [{"id": idx, **item} for idx, item in enumerate(items)]}
        system = (
            "You write structured hotel aspect summaries from sentiment summaries and clustered evidence. Return strict JSON only. "
            "Each item contains one hotel and one aspect with sentiment counts, sentiment-specific summaries, representative evidence, "
            "and optional clusters grouped by sentiment. Produce both Vietnamese and English summaries. "
            "Use exactly this structured multi-line format for each language:\n"
            "<aspect_code>: <positive_count> positive sentences, <negative_count> negative sentences, <neutral_count> neutral sentences\n"
            "Pos: <cluster-aware positive summary>\n"
            "Neg: <cluster-aware negative summary>\n"
            "Neu: <cluster-aware neutral/context summary>\n"
            "For Vietnamese, translate only the words 'positive sentences', 'negative sentences', and 'neutral sentences' into natural Vietnamese. "
            "The first line must use the input field aspect_code exactly, such as FAC, AME, SER, EXP, BRA, or LOY. "
            "Never output the literal placeholder '<aspect_code>', '<REAL_ASPECT_CODE>', or '<ASPECT_CODE>'. "
            "Do not output XML tags, placeholders, raw cluster IDs, bullets, or extra fields. "
            "Use clusters as the evidence map: identify the dominant recurring targets, preserve specific descriptors, and include a smaller cluster when it adds a distinct customer-impact or operational detail. "
            "Prefer the most specific recurring claims over broad generic aspect coverage. Large clusters are important, but they should not crowd out narrower claims that are repeated, concrete, and useful to a guest or operator. "
            "The Pos, Neg, and Neu lines are the important content: they should say what customers perceive, what works, what hurts, and the practical priority when the evidence supports it. "
            "Preserve sentiment contrast: if positive and negative evidence are both meaningful, describe the aspect as mixed instead of forcing a one-sided conclusion. "
            "If evidence is mostly positive, mention the main strength and any material risk; if mostly negative, state the clearest weakness and what should be fixed first. "
            "If neutral evidence is important, use it as context rather than filler. "
            "Treat cluster labels as candidate targets, descriptors as the customer perception, and samples as concrete support. "
            "Evidence lock: every target or problem you mention must appear in a cluster label, descriptor, sample, or sentiment summary. "
            "If a cluster label conflicts with its descriptors or samples, trust the descriptors/samples and either write the narrower supported target or omit that cluster. "
            "For amenity, be extra conservative: never mention wifi, parking, pool, spa, restaurant, breakfast, minibar, shuttle, TV, or gym unless that exact target appears in the evidence. "
            "Do not add common amenity targets merely because they are large clusters; mention them only when their evidence is salient for the current sentiment. "
            "When the strongest amenity signal is narrower, such as vending machines, minibar, laundry, a breakfast detail, coffee maker, ice machine, cookies, or a specific food item, preserve that narrower target even if broader amenity clusters are larger. "
            "For service, keep staff behavior, responsiveness, check-in, housekeeping, and restaurant service separate unless the same evidence links them. "
            "For loyalty, summarize only explicit return, revisit, recommendation, or referral intent; do not infer reasons from service, location, value, luxury, or branding unless the loyalty evidence states them. "
            "For experience, summarize only directly evidenced stay-level feelings, atmosphere, value perception, comfort, or overall satisfaction; avoid broad satisfaction labels unless the evidence explicitly supports them. "
            "For branding, focus on explicit expectation fit, reputation, perceived standard, trust, chain/brand comparison, luxury/boutique status, or value perception only when supported. "
            "Do not infer branding from clean rooms, good service, convenient location, or nice amenities unless the evidence explicitly frames them as brand, reputation, standard, class, or expectation signals. "
            "Do not reduce evidence to broad aspect labels like staff, room, food, amenity, or service without the descriptors that explain them. "
            "Do not write statistical reporting phrases such as 'Trong các phản hồi', 'khách nhắc nhiều nhất', 'chiếm tỷ trọng', "
            "'positive feedback mentions', 'the most mentioned', or 'there are N reviews'. "
            "Do not use fixed openings such as 'Về tiện ích', 'Về dịch vụ', 'For facility', or similar aspect-intro phrases. "
            "Do not mention hotel_id, source name, raw review counts, or metadata. Do not invent facts. "
            "If evidence is weak or noisy, say the signal is limited or mixed instead of over-claiming. "
            "If a sentiment has no meaningful evidence, keep its line and write that there is no clear signal for a separate conclusion."
        )
        user = (
            "Return the same id from each input item. Return exactly this compact schema:\n"
            '{"items":[{"id":0,"summary_vi":"...","summary_en":"..."}]}\n'
            "The summary_vi and summary_en fields must preserve line breaks using newline characters. "
            "Do not add any fields outside the schema. Do not wrap the JSON in markdown.\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=min(int(self.args.summary_max_output_tokens), 2048),
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen summary content finish_reason={finish}")
                return parse_qwen_summary_items(content, items)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        if is_context_length_error(last_error):
            if len(items) > 1:
                mid = max(1, len(items) // 2)
                return self.summarize_batch(items[:mid]) + self.summarize_batch(items[mid:])
            compact_items = [compact_aspect_summary_item_for_retry(item) for item in items]
            if compact_items != items:
                print("[aspect-summary] compacting oversized Qwen aspect-summary payload", flush=True)
                return self.summarize_batch(compact_items)

        print(f"[WARN] Qwen aspect summary failed; using local summaries. Error: {last_error}", file=sys.stderr)
        return [local_aspect_summary(item) for item in items]


def is_context_length_error(message: str) -> bool:
    lowered = str(message or "").lower()
    return "maximum context length" in lowered or "input_tokens" in lowered or "reduce the length" in lowered


def json_payload_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False))


def compact_aspect_summary_item_for_retry(item: dict[str, Any]) -> dict[str, Any]:
    compact = dict(item)
    compact["_context_compacted"] = True
    compact["clusters_by_sentiment"] = {}
    return compact


def chunk_text_values(values: list[Any], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        value_chars = len(text)
        if current and current_chars + value_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += value_chars
    if current:
        chunks.append(current)
    return chunks


def split_cluster_for_prompt(cluster: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    if json_payload_chars(cluster) <= max_chars:
        return [cluster]
    samples = [clean_text(sample) for sample in cluster.get("samples", []) if clean_text(sample)]
    descriptors = [clean_text(value) for value in cluster.get("descriptors", []) if clean_text(value)]
    base = dict(cluster)
    base["samples"] = []
    if json_payload_chars(base) <= max_chars and samples:
        return [{**base, "samples": sample_chunk} for sample_chunk in chunk_text_values(samples, max_chars)]
    base["descriptors"] = []
    pieces: list[dict[str, Any]] = []
    descriptor_chunks = chunk_text_values(descriptors, max(1000, max_chars // 4)) or [[]]
    sample_chunks = chunk_text_values(samples, max_chars) or [[]]
    for descriptor_chunk in descriptor_chunks:
        for sample_chunk in sample_chunks:
            pieces.append({**base, "descriptors": descriptor_chunk, "samples": sample_chunk})
    return pieces or [cluster]


def split_final_summary_item_for_prompt(item: dict[str, Any], max_chars: int = 18000) -> list[dict[str, Any]]:
    if json_payload_chars(item) <= max_chars:
        return [item]
    base = dict(item)
    clusters = [cluster for cluster in item.get("clusters", []) if isinstance(cluster, dict)]
    samples = [clean_text(sample) for sample in item.get("samples", []) if clean_text(sample)]
    base["clusters"] = []
    base["samples"] = []
    chunks: list[dict[str, Any]] = []
    current_clusters: list[dict[str, Any]] = []
    for cluster in clusters:
        for piece in split_cluster_for_prompt(cluster, max_chars):
            candidate = {**base, "clusters": current_clusters + [piece]}
            if current_clusters and json_payload_chars(candidate) > max_chars:
                chunks.append({**base, "clusters": current_clusters, "_chunked_final_item": True})
                current_clusters = [piece]
            else:
                current_clusters.append(piece)
    if current_clusters:
        chunks.append({**base, "clusters": current_clusters, "_chunked_final_item": True})
    if not chunks and samples:
        for sample_chunk in chunk_text_values(samples, max_chars):
            chunks.append({**base, "samples": sample_chunk, "_chunked_final_item": True})
    return chunks or [{**item, "_chunked_final_item": True}]


class QwenFinalSummaryWriter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def summarize_batch(self, items: list[dict[str, Any]]) -> list[str]:
        if self.args.skip_qwen:
            return [business_final_sentiment_summary(item) for item in items]

        payload = {"items": [{"id": idx, **item} for idx, item in enumerate(items)]}
        output_language = "English" if getattr(self.args, "summary_language", "vi") == "en" else "Vietnamese"
        empty_text = (
            "There is no clear signal strong enough for a separate conclusion."
            if output_language == "English"
            else "Không có tín hiệu nổi bật đủ rõ để rút ra nhận định riêng."
        )
        example_summary = (
            "Staff are friendly and responsive; check-in is smooth, while restaurant service is sometimes slow."
            if output_language == "English"
            else "Quản lý thân thiện; nhân viên thân thiện, dễ gần, phục vụ tốt và thạo tiếng Anh."
        )
        merged_target_example = (
            "'the manager is friendly; staff are approachable, helpful, and speak English well'"
            if output_language == "English"
            else "'quản lý thân thiện; nhân viên thân thiện, dễ gần, phục vụ tốt và thạo tiếng Anh'"
        )
        system = (
            f"You write business-friendly hotel insight summaries in {output_language}. Return strict JSON only. "
            "Each item contains one hotel, one aspect, one sentiment bucket, counts, clustered themes, and representative evidence. "
            "Write a sentiment-specific aspect insight for that bucket only. The summary may be one to three short natural sentences when needed to cover distinct targets without becoming a list. "
            "Start directly with what customers praised, criticized, or described. Do not start with aggregation or reporting phrases. "
            "Use clusters to make the summary fuller: cover the dominant recurring target-level themes, then add any smaller cluster that contributes a distinct customer-impact or operational detail. "
            "Prefer the most specific recurring claims over broad generic aspect coverage. Large clusters are important, but they should not crowd out narrower claims that are repeated, concrete, and useful to a guest or operator. "
            "Cluster labels define candidate targets; descriptors explain the customer perception; samples provide concrete wording and missing specifics. "
            "Use representative evidence only to clarify or fill missing descriptors. Do not introduce new targets from raw evidence unless they are not represented in clusters. "
            "Each cluster label is a concrete target, and descriptors/samples are the explicit "
            "properties, actions, capabilities, or problems attached to that target. "
            "Evidence lock: every target you mention must appear in the cluster label, descriptors, or samples. "
            "Do not add amenities, facilities, or service targets that are not present in the item's evidence. "
            "For amenity, never mention wifi, parking, pool, spa, restaurant, breakfast, minibar, shuttle, TV, or gym unless that exact target appears in the cluster evidence. "
            "Do not add common amenity targets merely because they are large clusters; mention them only when their evidence is salient for this sentiment bucket. "
            "When the strongest amenity signal is narrower, such as vending machines, minibar, laundry, a breakfast detail, coffee maker, ice machine, cookies, or a specific food item, preserve that narrower target even if broader amenity clusters are larger. "
            "For facility, never mention room, bathroom, AC, noise, bed, furniture, lobby, elevator, view, or location unless that exact target appears in the cluster evidence. "
            "For service, never mention staff, check-in, responsiveness, concierge, housekeeping, or restaurant service unless that exact target appears in the cluster evidence. "
            "If a cluster's label conflicts with its descriptors or samples, trust the descriptors/samples and either write the narrower supported target or omit that cluster. "
            "If a cluster label is broad but its samples/descriptors point to a narrower target, write the narrower target. "
            "For each important cluster, preserve the target plus its descriptor inventory first; use samples only to retain concrete details "
            "that descriptors miss, such as named roles, menu items, locations, or process details. "
            "Merge repeated mentions of the same target into one clause, and keep different targets separate. "
            "For example, if one cluster says manager/staff are friendly and another says staff are approachable, serve well, and speak English, "
            f"summarize as {merged_target_example}. "
            "Do not over-compress to a generic theme. Positive summaries should preserve distinct praise items; negative summaries should "
            "preserve distinct complaints and convert repeated raw descriptors into clear problem statements; neutral summaries should preserve distinct factual observations. "
            "Prioritize material and repeated target-level signals. If there are many clusters, include the most informative targets and omit minor one-off details. "
            "Include a lower-count cluster only when it adds a unique target or actionable detail. "
            "It is acceptable to use compact clauses so the important target-level information is kept, but avoid raw descriptor lists. "
            "Do not include counts in the summary text; counts are stored in separate columns. "
            "Avoid openings such as 'Khach danh gia', 'Khách đánh giá', 'Ghi nhan', 'Ghi nhận', 'Co N phan hoi', 'Có N phản hồi', "
            "'Nguoi dung cho biet', 'Trong các phản hồi', 'Ở chiều tiêu cực', 'Các phản hồi trung lập', 'Các lời khen được ghi nhận', "
            "'Các phàn nàn được ghi nhận', 'Các ghi nhận trung lập', or any 'Ve <aspect>' phrase. "
            "Do not include the aspect name in the sentence; the output row already contains the aspect and sentiment. "
            "If the count is 0 or there is no useful evidence, return an empty string for that item. "
            "Do not invent facts. Do not mention another aspect. "
            "For experience, summarize only directly evidenced stay-level feelings, atmosphere, value perception, comfort, or overall satisfaction; avoid broad satisfaction labels unless the evidence explicitly supports them. "
            "For branding, summarize only explicit expectation fit, reputation, perceived standard, trust, chain/brand comparison, luxury/boutique status, or value perception. "
            "Do not infer branding from clean rooms, good service, convenient location, or nice amenities unless the evidence explicitly frames them as brand, reputation, standard, class, or expectation signals. "
            "For loyalty, summarize only explicit return, revisit, recommendation, or referral intent found in the samples. "
            "Do not infer drivers such as service quality, reputation, location, luxury, or value unless those reasons are explicitly stated in the same loyalty samples. "
            "If loyalty samples only express a general intent, say the intent is general and do not add a reason. "
            "Keep the wording natural, direct, and readable for business users."
        )
        user = (
            "Return the same id from each input item. Return exactly this compact schema:\n"
            f'{{"items":[{{"id":0,"summary":"{example_summary}"}}]}}\n'
            f"If there is no useful evidence for an item, use an empty string, not '{empty_text}'.\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=min(int(self.args.final_summary_max_output_tokens), 2048),
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen final-summary content finish_reason={finish}")
                return parse_final_summary_items(content, items)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        if is_context_length_error(last_error):
            if len(items) > 1:
                mid = max(1, len(items) // 2)
                print(
                    f"[sentiment-summary] splitting oversized batch {len(items)} into {mid}+{len(items) - mid}",
                    flush=True,
                )
                return self.summarize_batch(items[:mid]) + self.summarize_batch(items[mid:])
            if items and not items[0].get("_chunked_final_item") and not items[0].get("_chunk_summary_collapse"):
                return [self.summarize_oversized_item(items[0])]

        print(f"[WARN] Qwen final summary failed; using local summaries. Error: {last_error}", file=sys.stderr)
        return [business_final_sentiment_summary(item) for item in items]

    def summarize_oversized_item(self, item: dict[str, Any]) -> str:
        chunks = split_final_summary_item_for_prompt(item)
        if len(chunks) <= 1 and chunks[0] == item:
            return business_final_sentiment_summary(item)
        print(f"[sentiment-summary] chunking oversized item into {len(chunks)} evidence chunks", flush=True)
        partials = [summary for chunk in chunks for summary in self.summarize_batch([chunk]) if clean_text(summary)]
        if not partials:
            return business_final_sentiment_summary(item)
        collapse_item = {
            "hotel_id": item.get("hotel_id", ""),
            "aspect": item.get("aspect", ""),
            "sentiment": item.get("sentiment", ""),
            "count": item.get("count", 0),
            "avg_confidence": item.get("avg_confidence", 0.0),
            "clusters": [],
            "samples": partials,
            "_chunk_summary_collapse": True,
        }
        return self.summarize_batch([collapse_item])[0]


class QwenProfileValidator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def validate_batch(self, profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if self.args.skip_qwen:
            return {profile["entity_id"]: local_profile_decision(profile) for profile in profiles}

        compact_profiles = []
        for i, profile in enumerate(profiles):
            qwen_profile = compact_profile_for_qwen(profile)
            compact_profiles.append(
                {
                    "id": i,
                    "entity_id": profile["entity_id"],
                    "aspects": qwen_profile["aspects"],
                }
            )
        system = (
            "You validate compact hotel review profiles. Return strict JSON only. "
            "Only provided aspects are tied/ambiguous; decide their final sentiment from counts and candidates. "
            "For each aspect, choose the best representative sentence from the provided candidates; "
            "lightly clean obvious spacing/encoding only, do not invent facts. "
            "Aspect keys are facility, amenity, service, experience, branding, loyalty. "
            "Sentiment keys are positive, negative, neutral."
        )
        user = (
            "Return compact JSON with this schema:\n"
            '{"items":[{"id":0,"entity_id":"booking_1","aspects":'
            '{"facility":{"sentiment":"positive","representative_sentence":"source sentence","note":"optional"}}}]}\n'
            "Include only the aspect keys provided in each profile. Empty representative_sentence is allowed when count is 0.\n\n"
            f"Profiles:\n{json.dumps({'items': compact_profiles}, ensure_ascii=False)}"
        )
        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.profile_max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen profile content finish_reason={finish}")
                return parse_profile_items(content, profiles)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(f"[WARN] Qwen profile batch failed; using local profile decisions. Error: {last_error}", file=sys.stderr)
        return {profile["entity_id"]: local_profile_decision(profile) for profile in profiles}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hotel aspect-sentiment pipeline with Qwen.")
    parser.add_argument("--inputs", nargs="+", default=[str(p) for p in DEFAULT_INPUTS])
    parser.add_argument("--output-csv", default=str(WORKSPACE / "hotel_aspect_sentiment_output.csv"))
    parser.add_argument("--debug-json", default="")
    parser.add_argument("--summary-metrics-csv", default=str(WORKSPACE / "hotel_aspect_summary_metrics.csv"))
    parser.add_argument("--summary-metrics-json", default=str(WORKSPACE / "hotel_aspect_summary_metrics.json"))
    parser.add_argument(
        "--aspect-summary-csv",
        default="",
        help="CSV export with one reader-facing overall aspect summary per final summary row.",
    )
    parser.add_argument(
        "--aspect-summary-json",
        default="",
        help="JSON export with one reader-facing overall aspect summary per final summary row.",
    )
    parser.add_argument(
        "--preseg-metrics-csv",
        default=str(WORKSPACE / "semantic_presegmentation_metrics.csv"),
        help=(
            "CSV for semantic pre-segmentation Information Coverage rows. "
            "Set to empty to disable."
        ),
    )
    parser.add_argument(
        "--preseg-metrics-json",
        default=str(WORKSPACE / "semantic_presegmentation_metrics.json"),
        help=(
            "JSON aggregate for semantic pre-segmentation Information Coverage. "
            "Set to empty to disable."
        ),
    )
    parser.add_argument("--final-summary-csv", default=str(WORKSPACE / "final_aspect_sentiment_summary.csv"))
    parser.add_argument("--final-summary-json", default=str(WORKSPACE / "final_aspect_sentiment_summary.json"))
    parser.add_argument(
        "--final-summary-metrics-csv",
        default=str(WORKSPACE / "final_aspect_sentiment_summary_metrics.csv"),
    )
    parser.add_argument(
        "--final-summary-metrics-json",
        default=str(WORKSPACE / "final_aspect_sentiment_summary_metrics.json"),
    )
    parser.add_argument(
        "--cluster-evidence-csv",
        default="",
        help="CSV export of per-cluster evidence rows derived from final-summary cluster data.",
    )
    parser.add_argument(
        "--cluster-evidence-json",
        default="",
        help="JSON export of per-cluster evidence rows derived from final-summary cluster data.",
    )
    parser.add_argument("--cache-db", default=str(WORKSPACE / "hotel_aspect_sentiment_cache.sqlite"))
    parser.add_argument(
        "--checkpoint-path",
        default=str(WORKSPACE / "hotel_aspect_sentiment_checkpoint.pkl"),
        help="Pickle checkpoint used to resume long-running aggregation jobs.",
    )
    parser.add_argument(
        "--checkpoint-every-chunks",
        type=int,
        default=5,
        help="Persist pipeline state after this many processed CSV chunks.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from the checkpoint file instead of rebuilding aggregates from scratch.",
    )
    parser.add_argument("--aspect-output-dir", default=str(WORKSPACE / "aspect_outputs"))
    parser.add_argument(
        "--disable-aspect-output",
        action="store_true",
        help="Skip per-aspect segment CSV files; useful for full-data summary evaluation runs.",
    )
    parser.add_argument(
        "--processed-sentences-csv",
        default="",
        help=(
            "Optional flat CSV of every processed review unit/aspect segment, including "
            "the source review, source phrase, shortened sentence, normalized text, aspect, and sentiment."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["profile-qwen", "local-only", "sentence-qwen"],
        default="profile-qwen",
        help="profile-qwen is the default fast full-data strategy.",
    )
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-api-key", default="")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--qwen-extract-workers",
        type=int,
        default=1,
        help="Parallel Qwen requests for aspect extraction.",
    )
    parser.add_argument(
        "--qwen-sentiment-workers",
        type=int,
        default=1,
        help="Parallel Qwen requests for segment sentiment classification.",
    )
    parser.add_argument(
        "--cluster-assignment",
        choices=["rule", "llm"],
        default="rule",
        help="Use rule-based fallback only, or ask Qwen to create source-faithful dynamic clusters from extracted ABSA units.",
    )
    parser.add_argument(
        "--qwen-cluster-workers",
        type=int,
        default=1,
        help="Parallel Qwen requests for LLM cluster assignment.",
    )
    parser.add_argument(
        "--cluster-assignment-max-output-tokens",
        type=int,
        default=5000,
        help="Maximum output tokens for Qwen cluster-assignment batches.",
    )
    parser.add_argument(
        "--cluster-assignment-min-confidence",
        type=float,
        default=0.55,
        help="Minimum LLM confidence required before using the LLM cluster instead of the rule fallback.",
    )
    parser.add_argument("--profile-batch-size", type=int, default=20)
    parser.add_argument(
        "--profile-qwen-filter",
        choices=["all", "tied"],
        default="all",
        help="Use tied to send only hotels with at least one tied aspect sentiment to Qwen.",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--semantic-max-output-tokens", type=int, default=2200)
    parser.add_argument("--profile-max-output-tokens", type=int, default=9000)
    parser.add_argument("--summary-max-output-tokens", type=int, default=5000)
    parser.add_argument("--final-summary-max-output-tokens", type=int, default=7000)
    parser.add_argument("--max-sentence-chars", type=int, default=700)
    parser.add_argument("--semantic-max-review-chars", type=int, default=2500)
    parser.add_argument(
        "--information-coverage-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help=(
            "SentenceTransformer model used for Information Coverage = "
            "cos(E_review, E_decomposition)."
        ),
    )
    parser.add_argument(
        "--information-coverage-batch-size",
        type=int,
        default=64,
        help="Batch size for pre-segmentation Information Coverage embedding.",
    )
    parser.add_argument("--profile-candidate-chars", type=int, default=260)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument(
        "--pre-segmentation",
        choices=["sentence", "semantic-qwen"],
        default="sentence",
        help=(
            "sentence keeps the legacy punctuation splitter. semantic-qwen uses Qwen "
            "to lightly compress noisy reviews into source-faithful opinion units before ABSA."
        ),
    )
    parser.add_argument(
        "--semantic-min-source-precision",
        type=float,
        default=0.72,
        help=(
            "Minimum share of compressed-unit tokens that must be present in the source review. "
            "Higher values keep wording closer to the original and reject aggressive paraphrases."
        ),
    )
    parser.add_argument(
        "--semantic-max-units-per-review",
        type=int,
        default=24,
        help="Maximum compressed semantic units accepted from one review.",
    )
    parser.add_argument(
        "--skip-rows-per-source",
        type=int,
        default=0,
        help=(
            "Skip this many data rows at the start of each CSV before processing. "
            "Use 100000 to begin from data row 100001."
        ),
    )
    parser.add_argument("--max-rows-per-source", type=int, default=0)
    parser.add_argument("--max-hotels-per-source", type=int, default=0)
    parser.add_argument(
        "--min-reviews-per-hotel",
        type=int,
        default=0,
        help="Only process hotels whose input file contains at least this many review rows.",
    )
    parser.add_argument("--debug-samples-per-aspect", type=int, default=3)
    parser.add_argument("--summary-batch-size", type=int, default=12)
    parser.add_argument(
        "--summary-workers",
        type=int,
        default=4,
        help="Parallel Qwen calls used when writing per-hotel aspect summaries.",
    )
    parser.add_argument(
        "--summary-progress-every-batches",
        type=int,
        default=100,
        help="Print progress after this many completed aspect-summary batches.",
    )
    parser.add_argument("--final-summary-batch-size", type=int, default=12)
    parser.add_argument("--final-summary-samples-per-sentiment", type=int, default=40)
    parser.add_argument(
        "--final-summary-sample-chars",
        type=int,
        default=0,
        help="Maximum characters kept for each evidence sample sent to final Qwen summaries (0 means no clipping).",
    )
    parser.add_argument(
        "--final-summary-cluster-threshold",
        type=float,
        default=0.45,
        help="Token Jaccard threshold for grouping similar uncategorized ABSA units into the same final-summary cluster.",
    )
    parser.add_argument(
        "--final-summary-max-clusters",
        type=int,
        default=0,
        help="Maximum clusters sent to Qwen for each hotel/aspect/sentiment final summary bucket (0 means no cap).",
    )
    parser.add_argument(
        "--final-summary-cluster-samples",
        type=int,
        default=0,
        help="Maximum representative samples stored per final-summary cluster (0 means no cap).",
    )
    parser.add_argument(
        "--final-summary-cluster-descriptors",
        type=int,
        default=0,
        help="Maximum descriptor words/phrases kept per final-summary cluster (0 means no cap).",
    )
    parser.add_argument(
        "--disable-final-summary-clusters",
        action="store_true",
        help=(
            "Do not assign or pass clusters into final summaries. "
            "This keeps the run as a standard ABSA pipeline: pre-segmentation, aspect, sentiment, then evidence-sample summary."
        ),
    )
    parser.add_argument(
        "--final-summary-workers",
        type=int,
        default=4,
        help="Parallel Qwen calls used when writing final aspect/sentiment summaries.",
    )
    parser.add_argument(
        "--final-summary-progress-every-batches",
        type=int,
        default=100,
        help="Print progress after this many completed final-summary batches.",
    )
    parser.add_argument(
        "--final-summary-only",
        action="store_true",
        help=(
            "Only write the final aspect x sentiment report. "
            "This skips the wide hotel-level output, but still keeps the hotel_id-based final summary and its metrics."
        ),
    )
    parser.add_argument(
        "--sentiment-language",
        choices=["vi", "en"],
        default="en",
        help="Normalized language used for aspect-level sentiment classification.",
    )
    parser.add_argument(
        "--summary-language",
        choices=["vi", "en"],
        default="vi",
        help="Normalized language used for representative text and summary metrics.",
    )
    parser.add_argument(
        "--bertscore-language",
        choices=["vi", "en"],
        default="en",
        help="Language code passed to BERTScore when summary metrics are enabled.",
    )
    parser.add_argument(
        "--disable-bertscore",
        action="store_true",
        help="Skip BERTScore while still writing ROUGE-1/2/L and coverage metrics.",
    )
    parser.add_argument(
        "--disable-summary-reference-stats",
        action="store_true",
        help=(
            "Do not keep summary reference token/text corpora in memory. "
            "Use this for very large resume runs when summary metrics are disabled."
        ),
    )
    parser.add_argument("--skip-qwen", action="store_true", help="Use rule-based fallback only.")
    parser.add_argument(
        "--qwen-enable-thinking",
        action="store_true",
        help="Allow Qwen thinking mode. Default keeps it off so content contains strict JSON.",
    )
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    if args.final_summary_only and args.strategy != "sentence-qwen":
        parser.error("--final-summary-only requires --strategy sentence-qwen")
    if args.pre_segmentation == "semantic-qwen" and args.strategy != "sentence-qwen":
        parser.error("--pre-segmentation semantic-qwen requires --strategy sentence-qwen")
    args.semantic_min_source_precision = max(0.0, min(1.0, float(args.semantic_min_source_precision)))
    args.semantic_max_units_per_review = max(1, int(args.semantic_max_units_per_review))
    args.semantic_max_review_chars = max(200, int(args.semantic_max_review_chars))
    args.information_coverage_batch_size = max(1, int(args.information_coverage_batch_size))
    args.final_summary_cluster_threshold = max(0.0, min(1.0, float(args.final_summary_cluster_threshold)))
    args.final_summary_max_clusters = max(0, int(args.final_summary_max_clusters))
    args.final_summary_cluster_samples = max(0, int(args.final_summary_cluster_samples))
    args.final_summary_cluster_descriptors = max(0, int(args.final_summary_cluster_descriptors))
    args.min_reviews_per_hotel = max(0, int(args.min_reviews_per_hotel))
    args.qwen_cluster_workers = max(1, int(args.qwen_cluster_workers))
    args.cluster_assignment_max_output_tokens = max(1000, int(args.cluster_assignment_max_output_tokens))
    args.cluster_assignment_min_confidence = max(0.0, min(1.0, float(args.cluster_assignment_min_confidence)))
    if not args.qwen_api_key:
        import os

        args.qwen_api_key = os.getenv("QWEN_API_KEY", DEFAULT_QWEN_API_KEY)
    return args


def parse_ref_id(ref_id: str) -> tuple[str, str, str] | None:
    match = REF_ID_RE.match(str(ref_id or "").strip())
    if not match:
        return None
    return match.group("data_source"), match.group("hotel_id"), match.group("review_index")


def make_entity_id(data_source: str, hotel_id: str) -> str:
    return f"{data_source}_{hotel_id}"


def build_min_review_filter(
    inputs: list[Path],
    min_reviews_per_hotel: int,
) -> tuple[set[str] | None, dict[str, Any]]:
    threshold = max(0, int(min_reviews_per_hotel))
    if threshold <= 0:
        return None, {}
    counts: Counter[str] = Counter()
    rows_seen = 0
    bad_ref_ids = 0
    for path in inputs:
        for chunk in pd.read_csv(path, dtype=str, chunksize=100000, usecols=["ref_id"]):
            for ref_id in chunk["ref_id"]:
                rows_seen += 1
                parsed = parse_ref_id(str(ref_id or ""))
                if parsed is None:
                    bad_ref_ids += 1
                    continue
                data_source, hotel_id, _ = parsed
                counts[make_entity_id(data_source, hotel_id)] += 1
    eligible = {entity_id for entity_id, count in counts.items() if count >= threshold}
    eligible_rows = sum(count for entity_id, count in counts.items() if entity_id in eligible)
    stats = {
        "min_reviews_per_hotel": threshold,
        "rows_scanned_for_filter": rows_seen,
        "bad_ref_ids_for_filter": bad_ref_ids,
        "total_hotels_before_filter": len(counts),
        "eligible_hotels": len(eligible),
        "eligible_review_rows": eligible_rows,
        "filtered_out_hotels": max(0, len(counts) - len(eligible)),
        "filtered_out_review_rows": max(0, rows_seen - bad_ref_ids - eligible_rows),
    }
    return eligible, stats


def clean_text(text: Any) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_cluster_assignment_source(source: Any) -> str:
    cleaned = clean_text(source).lower()
    return cleaned or "unknown"


def cluster_assignment_source_group(source: Any) -> str:
    cleaned = normalize_cluster_assignment_source(source)
    if cleaned.startswith("llm"):
        return "llm"
    if cleaned.startswith("rule"):
        return "rule_fallback"
    return "unknown"


def cluster_assignment_source_group_counts(
    source_counts: Counter[str],
    expected_total: int = 0,
) -> tuple[int, int, int]:
    llm_count = 0
    rule_fallback_count = 0
    unknown_count = 0
    for source, count in source_counts.items():
        group = cluster_assignment_source_group(source)
        if group == "llm":
            llm_count += int(count)
        elif group == "rule_fallback":
            rule_fallback_count += int(count)
        else:
            unknown_count += int(count)
    counted_total = llm_count + rule_fallback_count + unknown_count
    if expected_total > counted_total:
        unknown_count += expected_total - counted_total
    return llm_count, rule_fallback_count, unknown_count


def format_cluster_assignment_source_counts(
    source_counts: Counter[str],
    expected_total: int = 0,
) -> str:
    normalized_counts = Counter()
    for source, count in source_counts.items():
        normalized_counts[normalize_cluster_assignment_source(source)] += int(count)
    counted_total = sum(normalized_counts.values())
    if expected_total > counted_total:
        normalized_counts["unknown"] += expected_total - counted_total
    preferred_order = [
        "llm",
        "llm_repaired",
        "rule",
        "rule_low_llm_confidence",
        "rule_cross_aspect_llm_label",
        "unknown",
    ]
    ordered_sources = [source for source in preferred_order if normalized_counts.get(source, 0)]
    ordered_sources.extend(sorted(source for source in normalized_counts if source not in preferred_order))
    return "; ".join(f"{source}={normalized_counts[source]}" for source in ordered_sources)


def clip_text(text: str, max_chars: int) -> str:
    cleaned = clean_text(text)
    if max_chars is None or int(max_chars) <= 0:
        return cleaned
    return cleaned[: int(max_chars)]


def parse_json_list_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def strip_formulaic_aspect_intro(text: Any) -> str:
    cleaned = clean_text(text).rstrip()
    if not cleaned:
        return ""
    patterns = [
        r"^(?:Về|Ve)\s+[^,.:;]{1,80}[,.:;]\s*",
        r"^For\s+[^,.:;]{1,80}[,.:;]\s*",
        r"^Điểm được khen nhiều nhất về\s+[^,.:;]{1,80}\s+là\s+",
        r"^Phần phàn nàn chính về\s+[^,.:;]{1,80}\s+tập trung vào\s+",
        r"^Ý kiến trung lập về\s+[^,.:;]{1,80}\s+chủ yếu xoay quanh\s+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def clean_summary_text(text: Any, max_chars: int = 0) -> str:
    """Clean summary text while preserving Pos/Neg/Neu line breaks."""
    raw = "" if text is None else str(text)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in raw.split("\n"):
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(strip_formulaic_aspect_intro(cleaned))
    summary = "\n".join(line for line in lines if line)
    if max_chars and max_chars > 0 and len(summary) > max_chars:
        summary = summary[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,.;")
    return summary


def clean_overall_summary_text(text: Any, max_chars: int = 0) -> str:
    """Clean reader-facing overall summaries and drop old code/count scaffolding."""
    raw = clean_summary_text(text)
    if not raw:
        return ""
    fragments: list[str] = []
    for line in raw.split("\n"):
        line = clean_text(line)
        if not line:
            continue
        if "<ASPECT_CODE>" in line:
            continue
        if re.match(r"^(?:[A-Z]{2,5}|ALL)\s*:\s*[\d,]+\s+", line):
            continue
        label_match = re.match(r"^(?:Pos|Neg|Neu|Positive|Negative|Neutral)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if label_match:
            line = label_match.group(1).strip()
        if line:
            fragments.append(line)
    summary = normalize_one_sentence_summary(" ".join(fragments))
    if max_chars and max_chars > 0 and len(summary) > max_chars:
        summary = summary[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,.;")
    return summary


def has_summary_placeholder(text: Any) -> bool:
    cleaned = clean_text(text).lower()
    return "<aspect_code>" in cleaned or "<real_aspect_code>" in cleaned


def has_vietnamese_summary_leak(text: Any) -> bool:
    cleaned = clean_text(text).lower()
    if not cleaned:
        return False
    return bool(
        re.search(
            r"\b(khách|khong|không|tiện ích|dịch vụ|phòng|được|với|vị trí|"
            r"bữa|đánh giá|nhân viên|sạch|ồn|rộng|đẹp|thuận tiện|trải nghiệm)\b",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def has_raw_cluster_dump(text: Any) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False
    return bool(
        re.search(
            r"\b(?:Parking|Breakfast Quality|Food & Beverage Quality|Restaurant Availability|"
            r"Shuttle Service|Pool|Wifi Quality|Entertainment Facilities|In-room Amenities|"
            r"Value for Money|Comfort & Relaxation|Responsiveness|Hospitality|Check-in/Check-out)"
            r"\s*:\s*[^.\n]{20,}",
            cleaned,
        )
    )


def has_incomplete_english_summary_line(text: Any) -> bool:
    cleaned = clean_text(text).rstrip(".!?").strip()
    if not cleaned:
        return False
    return bool(
        re.search(
            r"\b(?:is|are|was|were|be|been|being|with|and|or|to|of|the|a|an|"
            r"include|includes|including|but|while|as|for|from)\s*$",
            cleaned,
            flags=re.IGNORECASE,
        )
    )


def enforce_structured_overall_summary(row: dict[str, Any], text: Any, summary_language: str = "vi") -> str:
    """Ensure overall_aspect_summary keeps CODE/count + Pos/Neg/Neu format."""
    aspect = str(row.get("aspect", ""))
    aspect_code = ASPECT_SUMMARY_CODES.get(aspect, aspect.upper()[:3] or "ASP")
    positive_count = int(row.get("positive_count", 0) or 0)
    negative_count = int(row.get("negative_count", 0) or 0)
    neutral_count = int(row.get("neutral_count", 0) or 0)
    if summary_language == "en":
        header = (
            f"{aspect_code}: {positive_count:,} positive sentences, "
            f"{negative_count:,} negative sentences, {neutral_count:,} neutral sentences"
        )
        fallback = "There is no clear signal strong enough for a separate conclusion."
    else:
        header = (
            f"{aspect_code}: {positive_count:,} câu tích cực, "
            f"{negative_count:,} câu tiêu cực, {neutral_count:,} câu trung lập"
        )
        fallback = "Không có tín hiệu nổi bật đủ rõ để rút ra nhận định riêng."

    cleaned = clean_summary_text(text, 1800)
    if has_summary_placeholder(cleaned):
        cleaned = ""
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    values: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^(Pos|Neg|Neu|Positive|Negative|Neutral)\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        label = match.group(1).lower()
        key = {"positive": "Pos", "negative": "Neg", "neutral": "Neu"}.get(label, label[:3].capitalize())
        values[key] = match.group(2).strip()
    if not values:
        return build_overall_aspect_summary(row, summary_language)
    if summary_language == "en":
        joined_values = " ".join(values.values())
        if (
            has_vietnamese_summary_leak(joined_values)
            or has_raw_cluster_dump(joined_values)
            or any(has_incomplete_english_summary_line(value) for value in values.values())
        ):
            return build_overall_aspect_summary(row, summary_language)
    return "\n".join(
        [
            header,
            f"Pos: {finish_sentence(values.get('Pos') or fallback)}",
            f"Neg: {finish_sentence(values.get('Neg') or fallback)}",
            f"Neu: {finish_sentence(values.get('Neu') or fallback)}",
        ]
    )


def has_non_formulaic_summary(text: Any) -> bool:
    cleaned = clean_text(text)
    return bool(cleaned and strip_formulaic_aspect_intro(cleaned) == cleaned)


def split_sentences(text: str, min_words: int) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw or raw.lower() == "nan":
        return []
    parts = SENTENCE_SPLIT_RE.split(raw)
    out = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip(" \t\r\n\"'")
        if len(WORD_RE.findall(part)) >= min_words:
            out.append(part)
    fallback = clean_text(raw)
    return out or ([fallback] if len(WORD_RE.findall(fallback)) >= min_words else [])


def semantic_review_hash(normalized_text: str) -> str:
    payload = f"{SEMANTIC_SEGMENTATION_PROMPT_VERSION}||{normalized_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_for_cache(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def sentence_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def aspect_extraction_hash(normalized_text: str) -> str:
    payload = f"{ASPECT_EXTRACTION_PROMPT_VERSION}||{normalized_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aspect_extraction_hash_for_version(version: str, normalized_text: str) -> str:
    payload = f"{version}||{normalized_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_aspect_extraction_hashes(normalized_text: str) -> list[str]:
    return [
        aspect_extraction_hash_for_version(version, normalized_text)
        for version in LEGACY_ASPECT_EXTRACTION_PROMPT_VERSIONS
    ]


def tokenize_text(text: str) -> list[str]:
    return WORD_RE.findall(normalize_for_cache(text))


CLUSTER_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "was",
    "were",
    "are",
    "is",
    "very",
    "really",
    "quite",
    "just",
    "also",
    "but",
    "have",
    "has",
    "had",
    "been",
    "they",
    "their",
    "there",
    "from",
    "our",
    "out",
    "about",
    "hotel",
    "room",
    "rooms",
    "khach",
    "khách",
    "san",
    "sạn",
    "va",
    "và",
    "la",
    "là",
    "co",
    "có",
    "rat",
    "rất",
    "qua",
    "quá",
    "thi",
    "thì",
    "nhung",
    "nhưng",
    "cung",
    "cũng",
    "duoc",
    "được",
    "cho",
    "voi",
    "với",
    "trong",
    "khi",
    "mot",
    "một",
    "cac",
    "các",
    "những",
}

DESCRIPTOR_STOPWORDS = CLUSTER_STOPWORDS | {
    "hotel",
    "khách sạn",
    "review",
    "guest",
    "khach",
    "khách",
    "place",
    "cho",
    "nơi",
    "noi",
    "stay",
    "stayed",
    "visit",
    "booking",
    "booked",
    "này",
    "nay",
    "đó",
    "do",
    "đây",
    "day",
}

ASPECT_TARGET_TOKENS = {
    "facility": {
        "room",
        "rooms",
        "phong",
        "phòng",
        "bed",
        "giuong",
        "giường",
        "bathroom",
        "toilet",
        "view",
        "location",
        "building",
        "lobby",
        "sảnh",
        "sanh",
    },
    "amenity": {
        "wifi",
        "breakfast",
        "bua",
        "bữa",
        "sang",
        "restaurant",
        "nha",
        "nhà",
        "hang",
        "hàng",
        "pool",
        "gym",
        "spa",
        "parking",
        "food",
        "drink",
    },
    "service": {
        "staff",
        "nhan",
        "nhân",
        "vien",
        "viên",
        "service",
        "phuc",
        "phục",
        "vu",
        "vụ",
        "reception",
        "receptionist",
        "le",
        "lễ",
        "tan",
        "tân",
    },
    "experience": {
        "experience",
        "trai",
        "trải",
        "nghiem",
        "nghiệm",
        "stay",
        "trip",
        "vacation",
        "ky",
        "kỳ",
        "nghi",
        "nghỉ",
    },
    "branding": {
        "brand",
        "thuong",
        "thương",
        "hieu",
        "hiệu",
        "standard",
        "tieu",
        "tiêu",
        "chuan",
        "chuẩn",
    },
    "loyalty": {
        "return",
        "recommend",
        "revisit",
        "quay",
        "lai",
        "lại",
        "gioi",
        "giới",
        "thieu",
        "thiệu",
    },
}

DESCRIPTOR_PHRASE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cách âm", ("soundproof", "soundproofed", "well soundproofed", "cach am", "cách âm")),
    (
        "rộng",
        (
            "spacious",
            "large",
            "big",
            "huge",
            "roomy",
            "rộng",
            "rong",
            "lớn",
            "lon",
            "khổng lồ",
            "khong lo",
            "rất to",
            "rat to",
            "phòng to",
            "phong to",
            "to rộng",
            "to rong",
        ),
    ),
    ("sạch", ("clean", "spotless", "sạch", "sach")),
    ("đẹp", ("beautiful", "nice", "pretty", "lovely", "đẹp", "dep")),
    ("hiện đại", ("modern", "new", "mới", "moi", "hiện đại", "hien dai")),
    ("thoải mái", ("comfortable", "comfy", "cozy", "thoải mái", "thoai mai", "ấm cúng", "am cung")),
    ("êm", ("soft", "êm", "em")),
    ("yên tĩnh", ("quiet", "peaceful", "calm", "yên tĩnh", "yen tinh", "yên bình", "yen binh")),
    ("view đẹp", ("great view", "beautiful view", "nice view", "view đẹp", "view dep", "tầm nhìn đẹp", "tam nhin dep")),
    ("gần trung tâm", ("central", "city center", "near center", "downtown", "gần trung tâm", "gan trung tam", "trung tâm", "trung tam")),
    ("thuận tiện", ("convenient", "accessible", "easy", "thuận tiện", "thuan tien", "dễ đi", "de di")),
    ("ngon", ("delicious", "tasty", "good food", "ngon")),
    ("đa dạng", ("varied", "diverse", "many choices", "đa dạng", "da dang", "nhiều món", "nhieu mon")),
    ("nhanh", ("fast", "quick", "prompt", "nhanh")),
    ("ổn định", ("stable", "reliable", "ổn định", "on dinh")),
    ("thân thiện", ("friendly", "warm", "welcoming", "thân thiện", "than thien")),
    ("nhiệt tình", ("helpful", "enthusiastic", "supportive", "nhiệt tình", "nhiet tinh", "hỗ trợ", "ho tro")),
    ("chuyên nghiệp", ("professional", "chuyên nghiệp", "chuyen nghiep")),
    ("chu đáo", ("attentive", "thoughtful", "chu đáo", "chu dao")),
    ("đáng tiền", ("worth it", "good value", "đáng tiền", "dang tien", "đáng đồng tiền", "dang dong tien")),
    ("giá hợp lý", ("reasonable price", "affordable", "giá hợp lý", "gia hop ly", "hợp lý", "hop ly")),
    ("sang trọng", ("luxury", "luxurious", "premium", "sang trọng", "sang trong", "cao cấp", "cao cap")),
    ("đúng tiêu chuẩn", ("standard", "consistent", "đúng chuẩn", "dung chuan", "đúng tiêu chuẩn", "dung tieu chuan")),
    ("muốn quay lại", ("return", "come back", "stay again", "quay lại", "quay lai", "trở lại", "tro lai")),
    ("muốn giới thiệu", ("recommend", "highly recommend", "suggest", "giới thiệu", "gioi thieu", "đề xuất", "de xuat")),
    ("bẩn", ("dirty", "unclean", "bẩn", "ban")),
    ("ồn", ("noisy", "noise", "ồn", "on", "tiếng ồn", "tieng on")),
    ("chật", ("small", "cramped", "narrow", "chật", "chat", "nhỏ", "nho")),
    ("cũ", ("old", "outdated", "dated", "cũ", "cu", "xuống cấp", "xuong cap")),
    ("hỏng", ("broken", "not working", "hỏng", "hong", "lỗi", "loi")),
    ("mùi hôi", ("smelly", "bad smell", "mùi hôi", "mui hoi", "hôi", "hoi")),
    ("yếu", ("weak", "slow", "unstable", "yếu", "yeu", "chậm", "cham")),
    ("đắt", ("expensive", "overpriced", "costly", "đắt", "dat", "mắc", "mac")),
    ("thất vọng", ("disappointing", "disappointed", "thất vọng", "that vong")),
    ("thô lỗ", ("rude", "impolite", "thô lỗ", "tho lo", "khó chịu", "kho chiu")),
)

DESCRIPTOR_TOKEN_ALLOWLIST = {
    "amazing",
    "attentive",
    "awful",
    "bad",
    "beautiful",
    "big",
    "broken",
    "calm",
    "central",
    "cheap",
    "clean",
    "cozy",
    "cramped",
    "delicious",
    "dirty",
    "disappointing",
    "diverse",
    "easy",
    "excellent",
    "expensive",
    "fast",
    "friendly",
    "good",
    "great",
    "hard",
    "helpful",
    "huge",
    "impolite",
    "large",
    "lovely",
    "luxurious",
    "modern",
    "new",
    "nice",
    "noisy",
    "old",
    "outdated",
    "overpriced",
    "peaceful",
    "polite",
    "poor",
    "premium",
    "professional",
    "quiet",
    "reasonable",
    "responsive",
    "roomy",
    "rude",
    "safe",
    "slow",
    "small",
    "smelly",
    "soft",
    "spacious",
    "spotless",
    "stable",
    "tasty",
    "terrible",
    "thoughtful",
    "unstable",
    "unsafe",
    "weak",
    "welcoming",
    "ấm",
    "am",
    "bẩn",
    "ban",
    "chậm",
    "cham",
    "chật",
    "chat",
    "chu",
    "cũ",
    "cu",
    "đắt",
    "dat",
    "đẹp",
    "dep",
    "êm",
    "em",
    "hỏng",
    "hong",
    "lớn",
    "lon",
    "mắc",
    "mac",
    "mềm",
    "mem",
    "mới",
    "moi",
    "ngon",
    "nhanh",
    "nhỏ",
    "nho",
    "nhiệt",
    "nhiệt tình",
    "nhiet",
    "ổn",
    "ồn",
    "on",
    "rộng",
    "rong",
    "sạch",
    "sach",
    "sang",
    "tệ",
    "te",
    "thân",
    "thân thiện",
    "than",
    "thiện",
    "thoải",
    "thoải mái",
    "tốt",
    "tot",
    "tuyệt",
    "tuyet",
    "yên",
    "yên tĩnh",
    "yen",
    "yếu",
    "yeu",
}


MEASUREMENT_SCALE_ALIASES: dict[str, tuple[str, ...]] = {
    "Room Cleanliness": ("sạch phòng", "độ sạch phòng", "vệ sinh phòng", "phòng sạch", "bụi", "bẩn", "mùi phòng"),
    "Room Comfort": ("thoải mái", "dễ chịu", "ấm cúng", "bí", "ngột ngạt", "phòng thoải mái"),
    "Bathroom Condition": ("phòng tắm", "nhà tắm", "nhà vệ sinh", "vòi sen", "bồn tắm", "toilet", "khăn"),
    "Bed Quality": ("giường", "nệm", "đệm", "gối", "chăn", "ngủ", "chất lượng giấc ngủ"),
    "Interior Design": ("nội thất", "thiết kế", "trang trí", "ánh sáng", "phong cách", "decor"),
    "Furniture Quality": ("đồ nội thất", "bàn ghế", "sofa", "bàn", "ghế", "tủ", "furniture"),
    "Reception Area": ("sảnh", "khu lễ tân", "lobby", "quầy lễ tân", "lối vào"),
    "Building Condition": ("tòa nhà", "toà nhà", "thang máy", "hạ tầng", "cơ sở vật chất", "xuống cấp", "bảo trì"),
    "Public Area Cleanliness": ("khu vực chung", "hành lang", "sảnh sạch", "vệ sinh khu vực chung", "public area"),
    "Spaciousness": ("rộng", "rộng rãi", "chật", "nhỏ", "diện tích", "kích thước phòng", "không gian phòng"),
    "Noise Condition": ("ồn", "tiếng ồn", "cách âm", "yên tĩnh", "nghe rõ", "soundproof"),
    "Air Conditioning": ("điều hòa", "điều hoà", "máy lạnh", "quạt", "thông gió", "nhiệt độ", "nóng", "lạnh"),
    "Physical Security Infrastructure": ("camera", "cctv", "khóa cửa", "khoá cửa", "thẻ từ", "an ninh", "bảo vệ", "két sắt"),
    "View & Surrounding Scenery": ("view", "cảnh quan", "tầm nhìn", "cảnh biển", "hướng biển", "phong cảnh", "scenery"),
    "Location & Surroundings": ("vị trí", "trung tâm", "gần", "xa", "phố cổ", "bãi biển", "xung quanh", "địa điểm"),
    "Wifi Quality": ("wifi", "wi-fi", "internet", "mạng", "kết nối", "tín hiệu"),
    "Breakfast Quality": ("bữa sáng", "ăn sáng", "buffet sáng", "đồ ăn sáng", "chất lượng bữa sáng"),
    "Pool": ("hồ bơi", "bể bơi", "pool", "jacuzzi", "bể sục"),
    "Gym": ("gym", "phòng gym", "phòng tập", "phòng tập thể dục", "fitness"),
    "Spa & Sauna": ("spa", "sauna", "xông hơi", "massage", "mát xa"),
    "Entertainment Facilities": ("giải trí", "khu vui chơi", "karaoke", "netflix", "tivi", "tv", "smart tv"),
    "Parking": ("bãi đậu xe", "bãi đỗ xe", "chỗ đậu xe", "giữ xe", "đỗ xe", "parking"),
    "Shuttle Service": ("đưa đón", "xe đưa đón", "shuttle", "airport transfer", "taxi", "di chuyển"),
    "Laundry Service": ("giặt ủi", "giặt là", "laundry", "dry cleaning"),
    "Restaurant Availability": ("nhà hàng", "quán ăn", "cafe", "bar", "quầy bar", "restaurant"),
    "In-room Amenities": ("tiện nghi trong phòng", "minibar", "tủ lạnh", "ấm đun", "máy sấy", "đồ vệ sinh cá nhân"),
    "Food & Beverage Quality": ("đồ ăn", "món ăn", "thức ăn", "đồ uống", "nước uống", "chất lượng món ăn", "ngon", "dở"),
    "Staff Friendliness": ("thân thiện", "dễ thương", "vui vẻ", "niềm nở", "thái độ nhân viên", "nhân viên thân thiện"),
    "Staff Professionalism": ("chuyên nghiệp", "nghiệp vụ", "được đào tạo", "chất lượng nhân viên", "professional"),
    "Check-in/Check-out": ("check-in", "check in", "check-out", "checkout", "nhận phòng", "trả phòng", "thủ tục", "đặt phòng"),
    "Responsiveness": ("phản hồi", "đáp ứng", "hỗ trợ nhanh", "nhanh chóng", "yêu cầu", "responsive"),
    "Problem Solving": ("xử lý vấn đề", "giải quyết", "khiếu nại", "phàn nàn", "sự cố", "complaint"),
    "Restaurant Service": ("phục vụ nhà hàng", "nhân viên nhà hàng", "bồi bàn", "waiter", "waitress"),
    "Room Service": ("room service", "dọn phòng", "buồng phòng", "housekeeping", "dịch vụ phòng"),
    "Hospitality": ("hiếu khách", "chu đáo", "chủ nhà", "chủ khách sạn", "đón tiếp", "hospitality"),
    "Communication Ability": ("giao tiếp", "tiếng anh", "tiếng Anh", "ngôn ngữ", "giải thích", "communication"),
    "Overall Satisfaction": ("hài lòng", "trải nghiệm chung", "trải nghiệm tổng thể", "đánh giá chung", "tổng thể"),
    "Comfort & Relaxation": ("thư giãn", "nghỉ dưỡng", "yên bình", "bầu không khí", "thoải mái", "relax"),
    "Value for Money": ("đáng tiền", "giá trị", "giá cả", "hợp lý", "rẻ", "đắt", "value for money"),
    "Convenience": ("tiện lợi", "thuận tiện", "dễ dàng", "convenient", "tiện ích di chuyển"),
    "Enjoyment": ("thích", "yêu thích", "vui", "tuyệt vời", "enjoy", "enjoyment"),
    "Safety Perception": ("an toàn", "yên tâm", "bất an", "nguy hiểm", "safe", "unsafe"),
    "Brand Reputation": ("danh tiếng", "thương hiệu", "nổi tiếng", "reputation"),
    "Brand Trust": ("tin cậy", "uy tín", "đáng tin", "trust", "reliable"),
    "Luxury Perception": ("sang trọng", "cao cấp", "đẳng cấp", "5 sao", "năm sao", "luxury"),
    "Brand Consistency": ("đồng nhất", "chuỗi", "tiêu chuẩn thương hiệu", "brand consistency"),
    "Expectation Fulfillment": ("kỳ vọng", "mong đợi", "đạt chuẩn", "không đạt", "expectation"),
    "Revisit Intention": ("quay lại", "trở lại", "ghé lại", "lần sau", "stay again", "come back"),
    "Recommendation Intention": ("giới thiệu", "khuyên", "đề xuất", "recommend"),
    "Customer Preference": ("ưa thích", "yêu thích", "khách quen", "lựa chọn", "favorite", "preferred"),
    "Loyalty Behavior": ("trung thành", "thành viên", "ủng hộ", "đặt lại", "loyalty", "membership"),
}


def split_annotation_keywords(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in re.split(r"[,;]\s*", text) if part.strip())


def normalize_taxonomy_header(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def taxonomy_row_get(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(normalize_taxonomy_header(name), "")
        if str(value).strip():
            return str(value).strip()
    return ""


def parse_xlsx_sheet_rows(path: Path, sheet_index: int = 1) -> list[dict[str, str]]:
    if not path.exists():
        return []
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        sheet_member = f"xl/worksheets/sheet{sheet_index}.xml"
        if sheet_member not in archive.namelist():
            return []
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", ns):
                texts = [
                    text_node.text or ""
                    for text_node in item.iter(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                    )
                ]
                shared_strings.append("".join(texts))

        sheet_root = ET.fromstring(archive.read(sheet_member))
        raw_rows: list[list[str]] = []
        for row in sheet_root.findall("a:sheetData/a:row", ns):
            cells: list[tuple[int, str]] = []
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                column = "".join(ch for ch in ref if ch.isalpha())
                column_index = 0
                for ch in column:
                    column_index = (column_index * 26) + (ord(ch.upper()) - 64)
                value_node = cell.find("a:v", ns)
                if cell.attrib.get("t") == "inlineStr":
                    texts = [
                        text_node.text or ""
                        for text_node in cell.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    ]
                    value = "".join(texts)
                elif value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text or 0)]
                else:
                    value = value_node.text or ""
                cells.append((column_index, value))
            max_column = max((idx for idx, _ in cells), default=0)
            values = [""] * max_column
            for idx, value in cells:
                values[idx - 1] = value
            raw_rows.append(values)

    if not raw_rows:
        return []
    headers = [normalize_taxonomy_header(value) for value in raw_rows[0]]
    rows: list[dict[str, str]] = []
    for values in raw_rows[1:]:
        if not any(str(value).strip() for value in values):
            continue
        rows.append(
            {
                header: str(values[idx]).strip() if idx < len(values) else ""
                for idx, header in enumerate(headers)
                if header
            }
        )
    return rows


def parse_xlsx_workbook_rows(path: Path) -> list[list[dict[str, str]]]:
    if not path.exists():
        return []
    with ZipFile(path) as archive:
        sheet_indices = sorted(
            int(match.group(1))
            for name in archive.namelist()
            if (match := re.fullmatch(r"xl/worksheets/sheet(\d+)\.xml", name))
        )
    return [parse_xlsx_sheet_rows(path, sheet_index=index) for index in sheet_indices]


def load_annotation_aspect_framework(path: Path) -> dict[str, dict[str, Any]]:
    aspect_map = {
        "FACILITY": "facility",
        "AMENITY": "amenity",
        "SERVICE": "service",
        "EXPERIENCE": "experience",
        "BRANDING": "branding",
        "LOYALTY": "loyalty",
    }
    framework: dict[str, dict[str, Any]] = {}
    for rows in parse_xlsx_workbook_rows(path):
        if not rows:
            continue
        sample_keys = set(rows[0])
        if "ASPECT" not in sample_keys or not (
            {"CORE_DEFINITION", "QUICK_DECISION_RULE"} & sample_keys
        ):
            continue
        for row in rows:
            aspect = aspect_map.get(taxonomy_row_get(row, "ASPECT").upper(), "")
            if aspect not in ASPECT_NAMES:
                continue
            covers = split_annotation_keywords(taxonomy_row_get(row, "WHAT_IT_COVERS", "COVERS"))
            does_not_cover = split_annotation_keywords(
                taxonomy_row_get(row, "WHAT_IT_DOES_NOT_COVER", "DOES_NOT_COVER")
            )
            examples = split_annotation_keywords(
                taxonomy_row_get(row, "EXAMPLE_KEYWORDS_TERMS", "EXAMPLE_KEYWORDS", "EXAMPLES")
            )
            framework[aspect] = {
                "core_definition": taxonomy_row_get(row, "CORE_DEFINITION", "CORE"),
                "quick_decision_rule": taxonomy_row_get(row, "QUICK_DECISION_RULE"),
                "covers": list(covers),
                "does_not_cover": list(does_not_cover),
                "examples": list(examples),
            }
        if framework:
            return framework
    return {}


def load_annotation_cluster_rules(path: Path) -> dict[str, list[dict[str, Any]]]:
    aspect_map = {
        "FACILITY": "facility",
        "AMENITY": "amenity",
        "SERVICE": "service",
        "EXPERIENCE": "experience",
        "BRANDING": "branding",
        "LOYALTY": "loyalty",
    }
    rules: dict[str, list[dict[str, Any]]] = {aspect: [] for aspect in ASPECT_NAMES}
    seen_codes: dict[str, dict[str, str]] = defaultdict(dict)
    for rows in parse_xlsx_workbook_rows(path):
        if not rows:
            continue
        sample_keys = set(rows[0])
        if not {"ASPECT", "CODE"} <= sample_keys or not (
            "MEASUREMENT_SCALE" in sample_keys or "MEASUREMENT" in sample_keys
        ):
            continue
        for row in rows:
            aspect = aspect_map.get(taxonomy_row_get(row, "ASPECT").upper(), "")
            measurement_scale = taxonomy_row_get(row, "MEASUREMENT_SCALE", "MEASUREMENT SCALE")
            if aspect not in ASPECT_NAMES or not measurement_scale:
                continue
            code = taxonomy_row_get(row, "CODE")
            if aspect == "facility" and measurement_scale == "Location & Surroundings":
                code = "FAC15"
            normalized_code = clean_text(code).upper()
            if normalized_code:
                previous_label = seen_codes[aspect].get(normalized_code)
                if previous_label and previous_label != measurement_scale:
                    raise ValueError(
                        "Duplicate cluster code in taxonomy for "
                        f"aspect={aspect}: code={normalized_code} labels={previous_label!r}, {measurement_scale!r}"
                    )
                seen_codes[aspect][normalized_code] = measurement_scale
            aspect_keywords = split_annotation_keywords(
                taxonomy_row_get(row, "ASPECT_KEYWORDS", "ASPECT KEYWORDS")
            )
            anchor_keywords = (
                aspect_keywords
                + split_annotation_keywords(measurement_scale.replace("&", ","))
                + MEASUREMENT_SCALE_ALIASES.get(measurement_scale, ())
            )
            keywords = (
                anchor_keywords
                + split_annotation_keywords(taxonomy_row_get(row, "DESCRIPTION"))
            )
            rules[aspect].append(
                {
                    "code": code,
                    "label": measurement_scale,
                    "measurement_scale": measurement_scale,
                    "description": taxonomy_row_get(row, "DESCRIPTION"),
                    "keywords": tuple(dict.fromkeys(keyword for keyword in keywords if keyword)),
                    "anchor_keywords": tuple(dict.fromkeys(keyword for keyword in anchor_keywords if keyword)),
                    "positive_keywords": split_annotation_keywords(
                        taxonomy_row_get(row, "POSITIVE_SENTIMENT_KEYWORDS", "POSITIVE SENTIMENT KEYWORDS")
                    ),
                    "negative_keywords": split_annotation_keywords(
                        taxonomy_row_get(row, "NEGATIVE_SENTIMENT_KEYWORDS", "NEGATIVE SENTIMENT KEYWORDS")
                    ),
                    "neutral_keywords": split_annotation_keywords(
                        taxonomy_row_get(row, "NEUTRAL_SENTIMENT_KEYWORDS", "NEUTRAL SENTIMENT KEYWORDS")
                    ),
                }
            )
    return {aspect: aspect_rules for aspect, aspect_rules in rules.items() if aspect_rules}


ASPECT_CLUSTER_RULES: dict[str, list[dict[str, Any]]] = {
    "facility": [
        {
            "code": "FAC_ROOM",
            "label": "phòng, giường và chất lượng giấc ngủ",
            "keywords": (
                "room",
                "suite",
                "bedroom",
                "villa",
                "accommodation",
                "bed",
                "mattress",
                "pillow",
                "blanket",
                "duvet",
                "bedding",
                "balcony",
                "workspace",
                "desk",
                "chair",
                "sleep",
                "sleeping",
                "window",
                "phong",
                "phòng",
                "giuong",
                "giường",
                "nem",
                "nệm",
                "goi",
                "gối",
                "chan",
                "chăn",
                "ban cong",
                "bàn làm việc",
            ),
        },
        {
            "code": "FAC_BATH",
            "label": "phòng tắm và khu vệ sinh",
            "keywords": (
                "bathroom",
                "toilet",
                "shower",
                "bathtub",
                "sink",
                "restroom",
                "washroom",
                "faucet",
                "towel",
                "nha tam",
                "nhà tắm",
                "phong tam",
                "phòng tắm",
                "nha ve sinh",
                "nhà vệ sinh",
                "voi sen",
                "vòi sen",
                "bon tam",
                "bồn tắm",
                "khan",
                "khăn",
            ),
        },
        {
            "code": "FAC_INTERIOR",
            "label": "nội thất, thiết kế và ánh sáng",
            "keywords": (
                "interior",
                "decor",
                "decoration",
                "furniture",
                "design",
                "style",
                "ambiance",
                "lighting",
                "atmosphere",
                "aesthetic",
                "theme",
                "noi that",
                "nội thất",
                "trang tri",
                "trang trí",
                "thiet ke",
                "thiết kế",
                "anh sang",
                "ánh sáng",
                "phong cach",
                "phong cách",
            ),
        },
        {
            "code": "FAC_BUILDING",
            "label": "hạ tầng tòa nhà và khu vực chung",
            "keywords": (
                "building",
                "elevator",
                "hallway",
                "corridor",
                "infrastructure",
                "property",
                "lobby",
                "entrance",
                "toa nha",
                "tòa nhà",
                "thang may",
                "thang máy",
                "hanh lang",
                "hành lang",
                "sanh",
                "sảnh",
                "loi vao",
                "lối vào",
            ),
        },
        {
            "code": "FAC_ENV",
            "label": "tiếng ồn và môi trường xung quanh",
            "keywords": (
                "quiet",
                "peaceful",
                "noisy",
                "noise",
                "soundproof",
                "calm",
                "environment",
                "neighborhood",
                "surroundings",
                "yen tinh",
                "yên tĩnh",
                "on",
                "ồn",
                "tieng on",
                "tiếng ồn",
                "cach am",
                "cách âm",
                "moi truong",
                "môi trường",
                "xung quanh",
            ),
        },
        {
            "code": "FAC_CLIMATE",
            "label": "điều hòa và thông gió",
            "keywords": (
                "aircon",
                "air conditioning",
                "ac",
                "ventilation",
                "airflow",
                "temperature",
                "fan",
                "cooling",
                "dieu hoa",
                "điều hòa",
                "may lanh",
                "máy lạnh",
                "thong gio",
                "thông gió",
                "nhiet do",
                "nhiệt độ",
                "quat",
                "quạt",
            ),
        },
        {
            "code": "FAC_SECURITY",
            "label": "hạ tầng an ninh vật lý",
            "keywords": (
                "cctv",
                "lock",
                "keycard",
                "guard",
                "surveillance",
                "gate",
                "security",
                "alarm",
                "safe box",
                "camera",
                "bao ve",
                "bảo vệ",
                "camera an ninh",
                "khoa cua",
                "khóa cửa",
                "the tu",
                "thẻ từ",
                "cong",
                "cổng",
            ),
        },
        {
            "code": "FAC_VIEW_LOCATION",
            "label": "view, cảnh quan và vị trí",
            "keywords": (
                "view",
                "scenery",
                "sea view",
                "mountain view",
                "city view",
                "landscape",
                "downtown",
                "central",
                "nearby",
                "location",
                "accessibility",
                "neighborhood",
                "beachfront",
                "riverside",
                "vi tri",
                "vị trí",
                "trung tam",
                "trung tâm",
                "gan",
                "gần",
                "canh quan",
                "cảnh quan",
                "tam nhin",
                "tầm nhìn",
                "bai bien",
                "bãi biển",
            ),
        },
    ],
    "amenity": [
        {
            "code": "AM_WIFI",
            "label": "wifi và kết nối internet",
            "keywords": ("wifi", "internet", "connection", "network", "signal", "mang", "mạng", "ket noi", "kết nối", "tin hieu", "tín hiệu"),
        },
        {
            "code": "AM_FOOD",
            "label": "đồ ăn, đồ uống và nhà hàng",
            "keywords": (
                "breakfast",
                "buffet",
                "restaurant",
                "cafe",
                "menu",
                "dining",
                "meal",
                "drink",
                "beverage",
                "coffee",
                "bar",
                "cuisine",
                "food",
                "dish",
                "an sang",
                "ăn sáng",
                "bua sang",
                "bữa sáng",
                "nha hang",
                "nhà hàng",
                "do an",
                "đồ ăn",
                "mon an",
                "món ăn",
                "do uong",
                "đồ uống",
                "ca phe",
                "cà phê",
            ),
        },
        {
            "code": "AM_POOL",
            "label": "hồ bơi và tiện ích nước",
            "keywords": ("pool", "swimming pool", "jacuzzi", "water park", "hot tub", "ho boi", "hồ bơi", "be boi", "bể bơi"),
        },
        {
            "code": "AM_WELLNESS",
            "label": "gym, spa và wellness",
            "keywords": ("gym", "spa", "sauna", "fitness center", "yoga", "massage", "phong gym", "phòng gym", "xong hoi", "xông hơi"),
        },
        {
            "code": "AM_ENT",
            "label": "giải trí và khu vui chơi",
            "keywords": ("entertainment", "kids zone", "playground", "game room", "karaoke", "netflix", "tv", "cinema", "giai tri", "giải trí", "khu vui chơi"),
        },
        {
            "code": "AM_TRANSPORT",
            "label": "di chuyển, đưa đón và bãi đỗ xe",
            "keywords": ("parking", "shuttle", "transport", "airport transfer", "taxi service", "valet", "bai dau xe", "bãi đậu xe", "dua don", "đưa đón", "giu xe", "giữ xe"),
        },
        {
            "code": "AM_ROOM_UTIL",
            "label": "tiện ích trong phòng",
            "keywords": ("minibar", "kettle", "microwave", "coffee maker", "refrigerator", "tv", "kitchen", "hairdryer", "tu lanh", "tủ lạnh", "am dun", "ấm đun", "may say", "máy sấy"),
        },
        {
            "code": "AM_UTILITY",
            "label": "giặt ủi và dịch vụ tiện ích phụ",
            "keywords": ("laundry", "washing", "ironing", "dry cleaning", "payment", "giat ui", "giặt ủi", "thanh toan", "thanh toán"),
        },
    ],
    "service": [
        {
            "code": "SER_ATTITUDE",
            "label": "thái độ và sự hiếu khách của nhân viên",
            "keywords": ("staff", "receptionist", "concierge", "waiter", "waitress", "employee", "housekeeping", "manager", "friendly", "polite", "helpful", "attentive", "warm", "professional", "nhan vien", "nhân viên", "le tan", "lễ tân", "than thien", "thân thiện", "lich su", "lịch sự", "nhiet tinh", "nhiệt tình"),
        },
        {
            "code": "SER_OPERATION",
            "label": "quy trình vận hành và check-in/check-out",
            "keywords": ("check-in", "check in", "check-out", "checkout", "booking", "reservation", "process", "operation", "queue", "smooth", "efficient", "delayed", "le tan", "lễ tân", "dat phong", "đặt phòng", "thu tuc", "thủ tục", "xep hang", "xếp hàng"),
        },
        {
            "code": "SER_SUPPORT",
            "label": "hỗ trợ và xử lý vấn đề",
            "keywords": ("support", "assistance", "complaint", "issue", "solve", "response", "request", "help", "responsive", "ignored", "unresolved", "ho tro", "hỗ trợ", "giup", "giúp", "phan hoi", "phản hồi", "giai quyet", "giải quyết", "van de", "vấn đề", "yeu cau", "yêu cầu"),
        },
        {
            "code": "SER_COMM",
            "label": "giao tiếp và khả năng ngôn ngữ",
            "keywords": ("communication", "english", "speaking", "language", "explanation", "clear", "fluent", "unclear", "giao tiep", "giao tiếp", "tieng anh", "tiếng anh", "ngon ngu", "ngôn ngữ", "giai thich", "giải thích"),
        },
    ],
    "experience": [
        {
            "code": "EXP_OVERALL",
            "label": "trải nghiệm lưu trú tổng thể",
            "keywords": ("stay", "experience", "trip", "vacation", "overall", "amazing", "satisfying", "disappointing", "trai nghiem", "trải nghiệm", "ky nghi", "kỳ nghỉ", "chuyen di", "chuyến đi", "tong the", "tổng thể"),
        },
        {
            "code": "EXP_EMOTION",
            "label": "cảm xúc, thư giãn và bầu không khí",
            "keywords": ("relaxing", "memorable", "peaceful", "enjoyable", "atmosphere", "comfort", "cozy", "stressful", "boring", "vibe", "thu gian", "thư giãn", "yen binh", "yên bình", "khong khi", "không khí", "thoai mai", "thoải mái", "de chiu", "dễ chịu"),
        },
        {
            "code": "EXP_VALUE",
            "label": "giá trị so với chi phí",
            "keywords": ("value", "worth", "price", "affordable", "expensive", "cost", "overpriced", "reasonable", "gia", "giá", "dang tien", "đáng tiền", "hop ly", "hợp lý", "dat", "đắt", "mac", "mắc"),
        },
        {
            "code": "EXP_SAFETY",
            "label": "cảm giác an toàn chủ quan",
            "keywords": ("safe", "secure", "dangerous", "risky", "safety", "reassuring", "unsafe", "comfortable", "an toan", "an toàn", "nguy hiem", "nguy hiểm", "yen tam", "yên tâm", "bat an", "bất an"),
        },
    ],
    "branding": [
        {
            "code": "BRA_REPUTE",
            "label": "danh tiếng, độ tin cậy và tiêu chuẩn thương hiệu",
            "keywords": ("brand", "reputation", "standard", "trust", "chain hotel", "reputable", "reliable", "trusted", "consistent", "overrated", "expectation", "expected", "thuong hieu", "thương hiệu", "danh tieng", "danh tiếng", "tieu chuan", "tiêu chuẩn", "ky vong", "kỳ vọng", "mong doi", "mong đợi"),
        },
        {
            "code": "BRA_LUXURY",
            "label": "cảm nhận sang trọng và cao cấp",
            "keywords": ("luxury", "premium", "five-star", "5-star", "upscale", "prestige", "exclusive", "high-end", "cheap", "low-standard", "dang cap", "đẳng cấp", "sang trong", "sang trọng", "cao cap", "cao cấp", "nam sao", "năm sao"),
        },
    ],
    "loyalty": [
        {
            "code": "LOY_RETURN",
            "label": "ý định quay lại",
            "keywords": ("return", "revisit", "stay again", "come back", "visit again", "definitely return", "not come back", "quay lai", "quay lại", "lan sau", "lần sau", "tro lai", "trở lại"),
        },
        {
            "code": "LOY_RECOMMEND",
            "label": "ý định giới thiệu",
            "keywords": ("recommend", "recommendation", "suggest", "referral", "highly recommend", "strongly recommend", "would not recommend", "gioi thieu", "giới thiệu", "de xuat", "đề xuất"),
        },
        {
            "code": "LOY_PREFERENCE",
            "label": "sở thích và gắn bó với khách sạn",
            "keywords": ("favorite", "favourite", "preferred", "loyalty", "membership", "attachment", "regular", "switch", "yeu thich", "yêu thích", "ua thich", "ưa thích", "khach quen", "khách quen", "thanh vien", "thành viên"),
        },
    ],
}


def build_aspect_annotation_prompt(
    framework: dict[str, dict[str, Any]],
    cluster_rules: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = [
        "Annotation framework for aspect assignment. Apply this table strictly:",
    ]
    for aspect in ASPECT_NAMES:
        info = framework.get(aspect, {})
        aspect_title = aspect.upper()
        lines.append(f"- {aspect_title}")
        lines.append(f"  Core definition: {info.get('core_definition', '')}")
        lines.append(f"  Quick decision rule: {info.get('quick_decision_rule', '')}")
        covers = ", ".join(str(value) for value in info.get("covers", []) if str(value).strip())
        does_not_cover = ", ".join(
            str(value) for value in info.get("does_not_cover", []) if str(value).strip()
        )
        examples = ", ".join(str(value) for value in info.get("examples", []) if str(value).strip())
        if covers:
            lines.append(f"  Covers: {covers}.")
        if does_not_cover:
            lines.append(f"  Does not cover: {does_not_cover}.")
        if examples:
            lines.append(f"  Example keywords/terms: {examples}.")

    lines.extend(
        [
            "",
            "Aspect splitting procedure:",
            "1. Split the input into the smallest faithful opinion units where each unit has one main target and one main meaning.",
            "2. For each unit, first ask the quick decision rules above, then check the Covers / Does not cover constraints.",
            "3. If one sentence contains several covered targets, return several segments, one per target.",
            "4. Assign the most specific concrete aspect before general EXPERIENCE unless the unit is only about overall feeling.",
            "5. Never assign BRANDING from a hotel name alone, and never assign LOYALTY from satisfaction alone.",
            "6. If one sentence contains conflicting opinions or contrastive polarity, split them into separate opinion units before sentiment classification.",
            "",
            "Sub-aspect cues from the annotation sheet:",
        ]
    )
    for aspect in ASPECT_NAMES:
        labels = [
            str(rule.get("measurement_scale") or rule.get("label", "")).strip()
            for rule in cluster_rules.get(aspect, [])
            if str(rule.get("measurement_scale") or rule.get("label", "")).strip()
        ]
        if labels:
            lines.append(f"- {aspect.capitalize()} targets: {', '.join(labels)}.")

    lines.extend(
        [
            "",
            "Boundary rules:",
            "- Prefer the most concrete aspect mentioned. A concrete room, bathroom, view, location, food, WiFi, pool, spa, shuttle, or staff comment should not be labeled experience just because the sentiment is emotional.",
            "- Food, breakfast, buffet, restaurant, cafe, bar, drinks, and menu quality are amenity unless the text specifically evaluates staff behavior or service process.",
            "- Room condition, bathroom, bed, cleanliness of room, furniture, interior design, lighting, hallway, elevator, building, air conditioning, noise insulation, views, surroundings, and location are facility.",
            "- CCTV, locks, keycards, guards, gates, alarms, surveillance, and other physical safety systems are facility. The guest's feeling of being safe or unsafe is experience.",
            "- Parking area physical condition can be facility; parking availability, valet, shuttle, airport transfer, taxi service, and transport support are amenity.",
            "- Brand/hotel name alone is not branding. Use branding only for reputation, chain standard, premium/luxury positioning, expected quality, or brand consistency.",
            "- General satisfaction is experience. Use loyalty only when there is explicit future behavior or attachment: return, revisit, stay again, recommend, favorite/preferred hotel, membership, or long-term preference.",
        ]
    )
    return "\n".join(line for line in lines if line is not None).strip()


ANNOTATION_CLUSTER_RULES = load_annotation_cluster_rules(ANNOTATION_TAXONOMY_XLSX)
if ANNOTATION_CLUSTER_RULES:
    ASPECT_CLUSTER_RULES = ANNOTATION_CLUSTER_RULES
ANNOTATION_ASPECT_FRAMEWORK = load_annotation_aspect_framework(ANNOTATION_TAXONOMY_XLSX)
if ANNOTATION_ASPECT_FRAMEWORK:
    ASPECT_CLUSTER_FRAMEWORK = ANNOTATION_ASPECT_FRAMEWORK
if ANNOTATION_CLUSTER_RULES or ANNOTATION_ASPECT_FRAMEWORK:
    ASPECT_ANNOTATION_PROMPT = build_aspect_annotation_prompt(
        ASPECT_CLUSTER_FRAMEWORK,
        ASPECT_CLUSTER_RULES,
    )


def cluster_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize_text(text)
        if len(token) > 2 and token not in CLUSTER_STOPWORDS and not token.isdigit()
    ]


def unique_preserve_order(values: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value).strip(" ,;:.")
        if not cleaned:
            continue
        key = normalize_for_cache(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if limit is not None and len(out) >= limit:
            break
    return out


def descriptor_supported_by_evidence(descriptor: str, text: str) -> bool:
    descriptor_norm = normalize_for_cache(descriptor)
    text_norm = normalize_for_cache(text)
    if not descriptor_norm or not text_norm:
        return False
    if contains_keyword(text_norm, descriptor_norm):
        return True
    descriptor_tokens = [token for token in tokenize_text(descriptor_norm) if len(token) > 2]
    text_tokens = set(tokenize_text(text_norm))
    if descriptor_tokens and all(token in text_tokens for token in descriptor_tokens):
        return True
    for canonical, variants in DESCRIPTOR_PHRASE_RULES:
        if normalize_for_cache(canonical) != descriptor_norm:
            continue
        return any(contains_keyword(text_norm, normalize_for_cache(variant)) for variant in variants)
    return False


def filter_source_faithful_descriptors(descriptors: list[str], text: str, limit: int = 18) -> list[str]:
    return unique_preserve_order(
        [descriptor for descriptor in descriptors if descriptor_supported_by_evidence(descriptor, text)],
        limit,
    )


def extract_cluster_descriptors(text: str, aspect: str = "") -> list[str]:
    """Return source-faithful descriptor words/phrases for a cluster sample."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    lower = normalize_for_cache(cleaned)
    descriptors: list[str] = []
    for canonical, variants in DESCRIPTOR_PHRASE_RULES:
        if any(contains_keyword(lower, normalize_for_cache(variant)) for variant in variants):
            descriptors.append(canonical)
    multi_descriptor_tokens: set[str] = set()
    for descriptor in descriptors:
        descriptor_tokens = tokenize_text(descriptor)
        if len(descriptor_tokens) > 1:
            multi_descriptor_tokens.update(descriptor_tokens)
    target_tokens = ASPECT_TARGET_TOKENS.get(aspect, set()) if aspect else set()
    for token in cluster_tokens(cleaned):
        if token in DESCRIPTOR_STOPWORDS or token in target_tokens:
            continue
        if token in multi_descriptor_tokens:
            continue
        if token not in DESCRIPTOR_TOKEN_ALLOWLIST:
            continue
        if len(token) <= 2 or token.isdigit():
            continue
        descriptors.append(token)
    return unique_preserve_order(descriptors, 18)


def token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def infer_opinion_cluster(aspect: str, sentiment: str, text: str) -> tuple[str, str]:
    lower = normalize_for_cache(text)
    best_rule: dict[str, Any] | None = None
    best_hits = 0
    for rule in ASPECT_CLUSTER_RULES.get(aspect, []):
        keywords = tuple(rule.get("keywords", ()))
        hits = sum(1 for keyword in keywords if contains_keyword(lower, keyword))
        if hits > best_hits:
            best_rule = rule
            best_hits = hits
    if best_rule:
        return str(best_rule.get("code", "")), str(best_rule.get("label", ""))
    return "", ""


def cluster_assignment_hash(aspect: str, sentiment: str, text: str) -> str:
    payload = (
        f"{CLUSTER_ASSIGNMENT_PROMPT_VERSION}||"
        f"{canonicalize_aspect_label(aspect) or normalize_aspect(aspect, text)}||"
        f"{normalize_sentiment(sentiment)}||{normalize_for_cache(text)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_json_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def summary_cache_hash(
    summary_type: str,
    prompt_version: str,
    args: argparse.Namespace,
    item: dict[str, Any],
) -> tuple[str, str]:
    language = str(getattr(args, "summary_language", "vi"))
    payload_json = stable_json_payload(
        {
            "summary_type": summary_type,
            "prompt_version": prompt_version,
            "model": str(getattr(args, "qwen_model", "")),
            "language": language,
            "item": item,
        }
    )
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest(), payload_json


def get_runtime_summary_cache(args: argparse.Namespace) -> ClassificationCache | None:
    cache = getattr(args, "_summary_cache", None)
    return cache if isinstance(cache, ClassificationCache) else None


def summarize_batch_with_cache(
    args: argparse.Namespace,
    summary_type: str,
    prompt_version: str,
    batch: list[dict[str, Any]],
    summarize_pending: Any,
) -> list[Any]:
    cache = get_runtime_summary_cache(args)
    if cache is None or not batch:
        return summarize_pending(batch)

    key_payloads = [summary_cache_hash(summary_type, prompt_version, args, item) for item in batch]
    keys = [key for key, _ in key_payloads]
    cached = cache.get_summaries(keys)
    results: list[Any | None] = [cached.get(key) for key in keys]
    pending_positions = [idx for idx, key in enumerate(keys) if key not in cached]
    if pending_positions:
        pending_items = [batch[idx] for idx in pending_positions]
        pending_results = summarize_pending(pending_items)
        rows_to_cache: dict[str, tuple[str, str, str, str, str, Any]] = {}
        for position, result in zip(pending_positions, pending_results):
            key, payload_json = key_payloads[position]
            results[position] = result
            rows_to_cache[key] = (
                summary_type,
                prompt_version,
                str(getattr(args, "qwen_model", "")),
                str(getattr(args, "summary_language", "vi")),
                payload_json,
                result,
            )
        cache.set_summaries(rows_to_cache)
    return [result for result in results if result is not None]


def cluster_rule_by_code(aspect: str, code: str) -> dict[str, Any] | None:
    normalized_code = clean_text(code).upper()
    if not normalized_code:
        return None
    matches = [
        rule
        for rule in ASPECT_CLUSTER_RULES.get(aspect, [])
        if clean_text(rule.get("code", "")).upper() == normalized_code
    ]
    return matches[0] if len(matches) == 1 else None


def cluster_rule_by_label(aspect: str, label: str) -> dict[str, Any] | None:
    normalized_label = normalize_for_cache(label)
    if not normalized_label:
        return None
    for rule in ASPECT_CLUSTER_RULES.get(aspect, []):
        if normalize_for_cache(str(rule.get("label", ""))) == normalized_label:
            return rule
    return None


def cluster_rule_by_any_label(label: str) -> tuple[str, dict[str, Any]] | None:
    normalized_label = normalize_for_cache(label)
    if not normalized_label:
        return None
    for aspect, rules in ASPECT_CLUSTER_RULES.items():
        for rule in rules:
            if normalize_for_cache(str(rule.get("label", ""))) == normalized_label:
                return aspect, rule
    return None


def other_cluster_for_aspect(aspect: str) -> tuple[str, str]:
    aspect = canonicalize_aspect_label(aspect) or clean_text(aspect)
    rules = ASPECT_CLUSTER_RULES.get(aspect, [])
    if rules:
        return str(rules[0].get("code", "")), str(rules[0].get("label", ""))
    return "", ""


def canonical_cluster_rule_from_text(aspect: str, text: str) -> dict[str, Any] | None:
    lower = normalize_for_cache(text)
    if not lower:
        return None
    best_rule: dict[str, Any] | None = None
    best_hits = 0
    for rule in ASPECT_CLUSTER_RULES.get(aspect, []):
        hits = sum(1 for keyword in tuple(rule.get("keywords", ())) if contains_keyword(lower, keyword))
        if hits > best_hits:
            best_rule = rule
            best_hits = hits
    return best_rule


def canonical_cluster_fields(
    aspect: str,
    sentiment: str,
    text: str,
    cluster_code: str = "",
    cluster_label: str = "",
    descriptors: list[str] | None = None,
) -> tuple[str, str]:
    aspect = canonicalize_aspect_label(aspect) or normalize_aspect(aspect, text)
    code = clean_text(cluster_code).upper()
    label = clean_text(cluster_label)
    descriptors_text = " ".join(clean_text(value) for value in (descriptors or []) if clean_text(value))

    rule = None
    if label:
        rule = cluster_rule_by_label(aspect, label)
    if rule is None:
        rule = cluster_rule_by_code(aspect, code)
    if rule is None:
        # Use the LLM label/descriptors as hints, but only to choose a fixed taxonomy bucket.
        rule = canonical_cluster_rule_from_text(aspect, " ".join(part for part in (label, descriptors_text, text) if part))
    if rule is not None:
        return str(rule.get("code", "")), str(rule.get("label", ""))

    other_code, other_label = other_cluster_for_aspect(aspect)
    return other_code, other_label


def canonicalize_cluster_assignment_item(
    item: dict[str, Any],
    aspect: str,
    sentiment: str,
    text: str,
) -> dict[str, Any]:
    if not isinstance(item, dict):
        item = {}
    aspect = canonicalize_aspect_label(aspect) or normalize_aspect(aspect, text)
    sentiment = normalize_sentiment(sentiment)
    descriptors_raw = item.get("descriptors", [])
    if isinstance(descriptors_raw, str):
        descriptors = [value.strip() for value in re.split(r"[,;]", descriptors_raw)]
    elif isinstance(descriptors_raw, list):
        descriptors = [str(value) for value in descriptors_raw]
    else:
        descriptors = []
    descriptors = filter_source_faithful_descriptors(
        unique_preserve_order([value for value in descriptors if clean_text(value)], 18),
        text,
    )
    if not descriptors:
        descriptors = extract_cluster_descriptors(text, aspect)
    code, label = canonical_cluster_fields(
        aspect,
        sentiment,
        text,
        clean_text(item.get("cluster_code", "") or item.get("code", "")),
        clean_text(item.get("cluster_label", "") or item.get("label", "")),
        descriptors,
    )
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "cluster_code": code,
        "cluster_label": label,
        "descriptors": descriptors,
        "confidence": clamp_confidence(item.get("confidence", 0.45)),
        "source": clean_text(item.get("source", "")) or "rule",
    }


def cluster_taxonomy_payload() -> dict[str, list[dict[str, Any]]]:
    return {
        aspect: [
            {
                "code": str(rule.get("code", "")),
                "label": str(rule.get("label", "")),
                "measurement_scale": str(rule.get("measurement_scale", rule.get("label", ""))),
                "description": str(rule.get("description", "")),
                "keywords": list(rule.get("keywords", ()))[:24],
            }
            for rule in rules
        ]
        for aspect, rules in ASPECT_CLUSTER_RULES.items()
    }


def aspect_cluster_framework_payload() -> dict[str, dict[str, Any]]:
    return {
        aspect: {
            "core_definition": str(framework.get("core_definition", "")),
            "quick_decision_rule": str(framework.get("quick_decision_rule", "")),
            "covers": list(framework.get("covers", [])),
            "does_not_cover": list(framework.get("does_not_cover", [])),
            "examples": list(framework.get("examples", [])),
        }
        for aspect, framework in ASPECT_CLUSTER_FRAMEWORK.items()
    }


def rule_cluster_assignment(aspect: str, sentiment: str, text: str) -> dict[str, Any]:
    aspect = canonicalize_aspect_label(aspect) or normalize_aspect(aspect, text)
    sentiment = normalize_sentiment(sentiment)
    raw_cluster_code, raw_label = infer_opinion_cluster(aspect, sentiment, text)
    cluster_code, label = canonical_cluster_fields(aspect, sentiment, text, raw_cluster_code, raw_label, [])
    matched_taxonomy = cluster_rule_by_code(aspect, cluster_code) is not None
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "cluster_code": cluster_code,
        "cluster_label": label,
        "descriptors": filter_source_faithful_descriptors(extract_cluster_descriptors(text, aspect), text),
        "confidence": 0.62 if matched_taxonomy else 0.45,
        "source": "rule",
    }


def normalize_cluster_assignment(raw: dict[str, Any], aspect: str, sentiment: str, text: str) -> dict[str, Any]:
    aspect = canonicalize_aspect_label(aspect) or normalize_aspect(aspect, text)
    sentiment = normalize_sentiment(sentiment)
    fallback = rule_cluster_assignment(aspect, sentiment, text)
    descriptors_raw = raw.get("descriptors", [])
    if isinstance(descriptors_raw, str):
        descriptors = [value.strip() for value in re.split(r"[,;]", descriptors_raw)]
    elif isinstance(descriptors_raw, list):
        descriptors = [str(value) for value in descriptors_raw]
    else:
        descriptors = []
    descriptors = filter_source_faithful_descriptors(
        unique_preserve_order(
            [value for value in descriptors if clean_text(value)],
            18,
        ),
        text,
    )
    if not descriptors:
        descriptors = fallback["descriptors"]
    code, label = canonical_cluster_fields(
        aspect,
        sentiment,
        text,
        clean_text(raw.get("cluster_code", "") or raw.get("code", "")),
        clean_text(raw.get("cluster_label", "") or raw.get("label", "")),
        descriptors,
    )
    confidence = clamp_confidence(raw.get("confidence", fallback.get("confidence", 0.45)))
    return {
        "aspect": aspect,
        "sentiment": sentiment,
        "cluster_code": code,
        "cluster_label": label,
        "descriptors": descriptors,
        "confidence": confidence,
        "source": clean_text(raw.get("source", "")) or "llm",
    }


def cluster_needs_llm_refinement(item: dict[str, Any], min_confidence: float) -> bool:
    if not isinstance(item, dict):
        return True
    source = clean_text(item.get("source", "")).lower()
    confidence = clamp_confidence(item.get("confidence", 0.0))
    label = clean_text(item.get("cluster_label", ""))
    descriptors = item.get("descriptors", [])
    has_descriptors = isinstance(descriptors, list) and any(clean_text(value) for value in descriptors)
    if source.startswith("llm") and confidence >= float(min_confidence) and label and has_descriptors:
        return False
    return True


def infer_opinion_cluster_label(aspect: str, sentiment: str, text: str) -> str:
    return infer_opinion_cluster(aspect, sentiment, text)[1]


def make_fallback_cluster_label(text: str) -> str:
    tokens = cluster_tokens(text)
    if tokens:
        return " ".join(tokens[:5])[:90]
    return clean_text(text).rstrip(".")[:90] or "ý kiến khác"


def source_token_precision(candidate: str, source: str) -> float:
    candidate_tokens = tokenize_text(candidate)
    if not candidate_tokens:
        return 0.0
    source_counts = Counter(tokenize_text(source))
    if not source_counts:
        return 0.0
    overlap = 0
    for token, count in Counter(candidate_tokens).items():
        overlap += min(count, source_counts.get(token, 0))
    return overlap / len(candidate_tokens)


SEMANTIC_NOISE_ONLY_RE = re.compile(
    r"^(?:no|none|nothing|n/?a|nil|thanks?|thank you|see you soon|good job|"
    r"khong|không|ko|k|khong co|không có|khong co gi|không có gì|"
    r"khong co gi phan nan|không có gì phàn nàn|khong co gi de che|không có gì để chê|"
    r"khong thich gi|không thích gì|khong co diem gi|không có điểm gì)$",
    re.IGNORECASE,
)
SEMANTIC_BOUNDARY_RE = re.compile(
    r"\s+(?=(?:nhưng|tuy nhiên|tuy vậy|dù vậy|điểm trừ|điểm cộng|ngoài ra|bù lại|"
    r"còn|nên|mong|nếu|không có|ko có|no\b))",
    re.IGNORECASE,
)
SEMANTIC_GLUED_BOUNDARY_RE = re.compile(r"(?<=[a-zà-ỹ])(?=[A-ZĐ])")


def normalize_semantic_compare(text: str) -> str:
    return re.sub(r"[^\w]+", " ", normalize_for_cache(text)).strip()


def is_semantic_noise_unit(text: str) -> bool:
    cleaned = clean_text(text).strip(" .,!?:;。！？…\"'")
    if not cleaned:
        return True
    normalized = normalize_semantic_compare(cleaned)
    return bool(SEMANTIC_NOISE_ONLY_RE.fullmatch(normalized))


def split_semantic_candidate_units(text: str, min_words: int) -> list[str]:
    raw = clean_text(text)
    if not raw:
        return []
    normalized = SEMANTIC_GLUED_BOUNDARY_RE.sub(". ", raw)
    primary_parts = split_sentences(normalized, min_words)
    candidates: list[str] = []
    for part in primary_parts:
        chunks = SEMANTIC_BOUNDARY_RE.split(part) if len(WORD_RE.findall(part)) >= 18 else [part]
        for chunk in chunks:
            cleaned = clean_text(chunk).strip(" \t\r\n\"'")
            if len(WORD_RE.findall(cleaned)) >= min_words and not is_semantic_noise_unit(cleaned):
                candidates.append(cleaned)
    return candidates or [raw]


def fallback_semantic_units(text: str, min_words: int) -> list[dict[str, Any]]:
    return [
        {
            "text": sentence,
            "source_text": sentence,
            "confidence": 0.4,
            "source_precision": 1.0,
            "fallback": True,
            "fallback_reason": "semantic_validation_or_qwen_failure",
        }
        for sentence in split_semantic_candidate_units(text, min_words)
    ]


def should_fallback_semantic_units(review_text: str, units: list[dict[str, Any]]) -> bool:
    if not units:
        return True
    review_words = len(WORD_RE.findall(review_text))
    if review_words < 25:
        return False
    unit_texts = [clean_text(unit.get("text", "")) for unit in units if clean_text(unit.get("text", ""))]
    if not unit_texts:
        return True
    decomposition_text = clean_text(" ".join(unit_texts))
    decomposition_words = len(WORD_RE.findall(decomposition_text))
    if not decomposition_words:
        return True
    ratio = decomposition_words / max(1, review_words)
    review_compact = normalize_semantic_compare(review_text)
    decomposition_compact = normalize_semantic_compare(decomposition_text)
    if review_compact == decomposition_compact:
        return True
    if len(unit_texts) == 1 and ratio >= 0.70:
        return True
    if ratio < 0.55:
        return True
    return False


def clean_semantic_units(
    raw_units: list[dict[str, Any]],
    review_text: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    cleaned_units: list[dict[str, Any]] = []
    for raw_unit in raw_units[: args.semantic_max_units_per_review]:
        if not isinstance(raw_unit, dict):
            continue
        unit = validate_semantic_unit(raw_unit, review_text, args)
        if unit is not None:
            cleaned_units.append(unit)
    if should_fallback_semantic_units(review_text, cleaned_units):
        return fallback_semantic_units(review_text, args.min_words)
    return cleaned_units


class InformationCoverageScorer:
    """Scores cos(E_review, E_decomposition) for semantic pre-segmentation."""

    def __init__(self, model_name: str, batch_size: int):
        self.model_name = str(model_name or "").strip()
        self.batch_size = max(1, int(batch_size))
        self.backend = "sentence_transformers"
        self.available = True
        self._model: Any | None = None
        self._fallback_vectorizer: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence_transformers"
            self.available = True
            return self._model
        except Exception as exc:  # noqa: BLE001 - metric should not stop pipeline.
            print(
                "[WARN] sentence-transformers unavailable for Information Coverage; "
                f"using lexical HashingVectorizer fallback. Error: {exc}",
                file=sys.stderr,
            )
            self.backend = "hashing_vectorizer_fallback"
            self.available = False
            return None

    @staticmethod
    def _cosine_rows(left: Any, right: Any) -> list[float]:
        import numpy as np

        left_arr = np.asarray(left, dtype=float)
        right_arr = np.asarray(right, dtype=float)
        left_norm = np.linalg.norm(left_arr, axis=1)
        right_norm = np.linalg.norm(right_arr, axis=1)
        denom = left_norm * right_norm
        scores = np.divide(
            np.sum(left_arr * right_arr, axis=1),
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom > 0,
        )
        return [max(-1.0, min(1.0, float(score))) for score in scores]

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load_model()
        reviews = [left for left, _ in pairs]
        decompositions = [right for _, right in pairs]
        if model is not None:
            review_embeddings = model.encode(
                reviews,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            decomposition_embeddings = model.encode(
                decompositions,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return self._cosine_rows(review_embeddings, decomposition_embeddings)

        from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

        if self._fallback_vectorizer is None:
            self._fallback_vectorizer = HashingVectorizer(
                n_features=2**18,
                alternate_sign=False,
                norm="l2",
                analyzer="word",
                ngram_range=(1, 2),
            )
        left_matrix = self._fallback_vectorizer.transform(reviews)
        right_matrix = self._fallback_vectorizer.transform(decompositions)
        scores = left_matrix.multiply(right_matrix).sum(axis=1)
        return [max(-1.0, min(1.0, float(score))) for score in scores.A1]


class PreSegmentationMetricWriter:
    FIELDNAMES = [
        "source_file",
        "data_source",
        "hotel_id",
        "review_index",
        "review_hash",
        "unit_count",
        "review_word_count",
        "decomposition_word_count",
        "avg_unit_source_precision",
        "min_unit_source_precision",
        "avg_pre_segment_confidence",
        "information_coverage",
        "information_coverage_backend",
        "information_coverage_model",
        "embedding_available",
        "review_text",
        "decomposition_text",
    ]

    def __init__(
        self,
        csv_path: Path | None,
        json_path: Path | None,
        scorer: InformationCoverageScorer,
        reset_existing: bool = True,
    ):
        self.csv_path = csv_path
        self.json_path = json_path
        self.scorer = scorer
        self.rows = 0
        self.sum_score = 0.0
        self.min_score = 1.0
        self.max_score = -1.0
        self.by_file: dict[str, dict[str, float]] = defaultdict(
            lambda: {"rows": 0.0, "sum": 0.0, "min": 1.0, "max": -1.0}
        )
        self.has_rows = False
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if reset_existing and self.csv_path.exists():
                self.csv_path.unlink()
            elif not reset_existing and self.csv_path.exists():
                with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            score = float(row.get("information_coverage", ""))
                        except Exception:
                            continue
                        self._update_aggregate(str(row.get("source_file", "")), score)
                self.has_rows = self.rows > 0
        if self.json_path is not None:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            if reset_existing and self.json_path.exists():
                self.json_path.unlink()

    @property
    def enabled(self) -> bool:
        return self.csv_path is not None or self.json_path is not None

    def _update_aggregate(self, source_file: str, score: float) -> None:
        self.rows += 1
        self.sum_score += score
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        file_stats = self.by_file[source_file]
        file_stats["rows"] += 1.0
        file_stats["sum"] += score
        file_stats["min"] = min(file_stats["min"], score)
        file_stats["max"] = max(file_stats["max"], score)

    def write_review_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows or not self.enabled:
            return
        pairs = [(str(row["review_text"]), str(row["decomposition_text"])) for row in rows]
        scores = self.scorer.score_pairs(pairs)
        output_rows: list[dict[str, Any]] = []
        for row, score in zip(rows, scores):
            payload = dict(row)
            payload["information_coverage"] = round(float(score), 6)
            payload["information_coverage_backend"] = self.scorer.backend
            payload["information_coverage_model"] = self.scorer.model_name
            payload["embedding_available"] = self.scorer.available
            output_rows.append(payload)
            self._update_aggregate(str(payload.get("source_file", "")), float(score))
        if self.csv_path is not None and output_rows:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                if write_header:
                    writer.writeheader()
                writer.writerows(output_rows)
        self.has_rows = self.has_rows or bool(output_rows)

    def write_json(self, stats: dict[str, Any]) -> None:
        if self.json_path is None:
            return
        by_file = {}
        for source_file, item in sorted(self.by_file.items()):
            rows = int(item["rows"])
            by_file[source_file] = {
                "rows": rows,
                "mean_information_coverage": round(item["sum"] / rows, 6) if rows else 0.0,
                "min_information_coverage": round(item["min"], 6) if rows else 0.0,
                "max_information_coverage": round(item["max"], 6) if rows else 0.0,
            }
        payload = {
            "metric_type": "semantic_presegmentation_information_coverage",
            "definition": "Information Coverage = cos(E_review, E_decomposition), where E_review embeds the full original review and E_decomposition embeds the concatenation of the split/compressed review units.",
            "embedding_backend": self.scorer.backend,
            "embedding_model": self.scorer.model_name,
            "embedding_available": self.scorer.available,
            "stats": stats,
            "overall": {
                "rows": self.rows,
                "mean_information_coverage": round(self.sum_score / self.rows, 6) if self.rows else 0.0,
                "min_information_coverage": round(self.min_score, 6) if self.rows else 0.0,
                "max_information_coverage": round(self.max_score, 6) if self.rows else 0.0,
            },
            "by_file": by_file,
        }
        self.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_semantic_unit(unit: dict[str, Any], review_text: str, args: argparse.Namespace) -> dict[str, Any] | None:
    text = clean_text(unit.get("text", "") or unit.get("unit_text", "") or unit.get("segment_text", ""))
    if is_semantic_noise_unit(text):
        return None
    if len(WORD_RE.findall(text)) < max(1, int(args.min_words)):
        return None
    source_text = clean_text(unit.get("source_text", "")) or review_text
    precision = source_token_precision(text, review_text)
    if precision < float(args.semantic_min_source_precision):
        return None
    out: dict[str, Any] = {
        "text": text,
        "source_text": source_text,
        "confidence": clamp_confidence(unit.get("confidence", 0.65)),
        "source_precision": round(precision, 6),
    }
    flags = unit.get("flags")
    if isinstance(flags, list):
        out["flags"] = [clean_text(flag) for flag in flags if clean_text(flag)][:8]
    return out


def parse_qwen_semantic_units(
    content: str,
    reviews: list[str],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, list[dict[str, Any]]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if not (0 <= idx < len(reviews)):
                continue
            raw_units = item.get("units", [])
            if not isinstance(raw_units, list):
                continue
            raw_unit_dicts = [raw_unit for raw_unit in raw_units if isinstance(raw_unit, dict)]
            by_id[idx] = clean_semantic_units(raw_unit_dicts, reviews[idx], args)
    out: list[list[dict[str, Any]]] = []
    for idx, review in enumerate(reviews):
        units = by_id.get(idx)
        out.append(units if units is not None else fallback_semantic_units(review, args.min_words))
    return out


def parse_qwen_semantic_units_lenient(
    content: str,
    reviews: list[str],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    try:
        return parse_qwen_semantic_units(content, reviews, args)
    except json.JSONDecodeError:
        pass

    repaired_items: list[dict[str, Any]] = []
    item_pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"units"\s*:\s*\[(.*?)\]\s*\}', re.DOTALL)
    unit_pattern = re.compile(
        r'\{\s*"(?:text|unit_text)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        r'(?:\s*,\s*"source_text"\s*:\s*"((?:[^"\\]|\\.)*)")?'
        r'(?:\s*,\s*"confidence"\s*:\s*([0-9.]+))?',
        re.DOTALL,
    )
    for item_match in item_pattern.finditer(content):
        idx = int(item_match.group(1))
        raw_units = item_match.group(2)
        units: list[dict[str, Any]] = []
        for unit_match in unit_pattern.finditer(raw_units):
            try:
                text_value = json.loads(f'"{unit_match.group(1)}"')
            except json.JSONDecodeError:
                text_value = unit_match.group(1)
            try:
                source_value = json.loads(f'"{unit_match.group(2)}"') if unit_match.group(2) else ""
            except json.JSONDecodeError:
                source_value = unit_match.group(2) or ""
            units.append(
                {
                    "text": text_value,
                    "source_text": source_value,
                    "confidence": float(unit_match.group(3) or 0.65),
                }
            )
        repaired_items.append({"id": idx, "units": units})

    if repaired_items:
        return parse_qwen_semantic_units(json.dumps({"items": repaired_items}, ensure_ascii=False), reviews, args)
    return [fallback_semantic_units(review, args.min_words) for review in reviews]


def make_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def contains_keyword(text: str, keyword: str) -> bool:
    if not keyword:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, flags=re.IGNORECASE))


TAXONOMY_GUARDRAIL_GENERIC_KEYWORDS = {
    "hotel",
    "property",
    "area",
    "quality",
    "condition",
    "experience",
    "standard",
    "average",
    "normal",
    "basic",
    "okay",
    "ok",
    "good",
    "bad",
    "great",
    "poor",
    "nice",
    "clean",
    "dirty",
    "modern",
    "old",
    "comfortable",
    "comfort",
    "convenient",
    "service",
    "available",
    "acceptable",
    "decent",
    "environment",
    "atmosphere",
    "safe",
    "secure",
    "khach san",
    "khách sạn",
    "tot",
    "tốt",
    "te",
    "tệ",
    "sach",
    "sạch",
    "ban",
    "bẩn",
    "thoai mai",
    "thoải mái",
    "tien loi",
    "tiện lợi",
}


def taxonomy_anchor_keywords(aspect: str) -> tuple[str, ...]:
    values: list[str] = []
    for rule in ASPECT_CLUSTER_RULES.get(aspect, []):
        raw_keywords = rule.get("anchor_keywords", rule.get("keywords", ()))
        if isinstance(raw_keywords, str):
            keywords = split_annotation_keywords(raw_keywords)
        else:
            keywords = tuple(str(value) for value in raw_keywords)
        for keyword in keywords:
            normalized = normalize_for_cache(keyword)
            if not normalized or normalized in TAXONOMY_GUARDRAIL_GENERIC_KEYWORDS:
                continue
            if len(normalized) <= 2 and aspect != "amenity":
                continue
            values.append(normalized)
    return tuple(dict.fromkeys(values))


def taxonomy_guardrail_aspect(text: str, candidate: str = "") -> tuple[str, str]:
    lower = normalize_for_cache(text)
    if not lower:
        return "", ""
    scores: dict[str, int] = {}
    examples: dict[str, list[str]] = defaultdict(list)
    for aspect in ASPECT_NAMES:
        for keyword in taxonomy_anchor_keywords(aspect):
            if contains_keyword(lower, keyword):
                scores[aspect] = scores.get(aspect, 0) + 1
                if len(examples[aspect]) < 3:
                    examples[aspect].append(keyword)
    if not scores:
        return "", ""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], ASPECT_NAMES.index(item[0])))
    best_aspect, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score <= second_score:
        return "", ""
    if candidate and candidate in scores and candidate != best_aspect and best_score - scores[candidate] < 2:
        return "", ""
    return best_aspect, f"taxonomy_anchor:{','.join(examples[best_aspect])}"


def ngram_recall(summary_counts: Counter[Any], reference_counts: Counter[Any]) -> float:
    reference_total = int(sum(reference_counts.values()))
    if reference_total <= 0 or not summary_counts:
        return 0.0
    overlap = sum(min(count, reference_counts.get(ngram, 0)) for ngram, count in summary_counts.items())
    return overlap / reference_total


def rouge_l_recall(summary_tokens: list[str], reference_tokens: list[str]) -> float:
    if not summary_tokens or not reference_tokens:
        return 0.0
    previous = [0] * (len(reference_tokens) + 1)
    for summary_token in summary_tokens:
        current = [0]
        for idx, reference_token in enumerate(reference_tokens, start=1):
            if summary_token == reference_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1] / len(reference_tokens)


def coverage_score(summary_tokens: list[str], reference_unigrams: Counter[str]) -> float:
    if not summary_tokens or not reference_unigrams:
        return 0.0
    summary_vocab = set(summary_tokens)
    reference_vocab = set(reference_unigrams)
    return len(summary_vocab & reference_vocab) / len(reference_vocab)


def bertscore_summary(summary_text: str, reference_texts: list[str], language: str) -> dict[str, Any]:
    references = [clean_text(text) for text in reference_texts if clean_text(text)]
    summary = clean_text(summary_text)
    if not summary or not references:
        return {"available": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from bert_score import BERTScorer  # type: ignore
    except Exception:
        return {"available": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        scorer_language = language if language in {"en", "vi"} else "en"
        scorer = BERTSCORER_CACHE.get(scorer_language)
        if scorer is None:
            scorer = BERTScorer(
                lang=scorer_language,
                rescale_with_baseline=False,
            )
            BERTSCORER_CACHE[scorer_language] = scorer
        candidates = [summary] * len(references)
        precision, recall, f1 = scorer.score(candidates, references)
    except Exception:
        return {"available": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(f1) == 0:
        return {"available": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_idx = int(f1.argmax().item())
    return {
        "available": True,
        "precision": float(precision[best_idx].item()),
        "recall": float(recall[best_idx].item()),
        "f1": float(f1[best_idx].item()),
    }


def empty_bertscore() -> dict[str, Any]:
    return {"available": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def modified_precision(summary_counts: Counter[Any], reference_counts: Counter[Any]) -> float:
    summary_total = int(sum(summary_counts.values()))
    if summary_total <= 0:
        return 0.0
    overlap = sum(min(count, reference_counts.get(ngram, 0)) for ngram, count in summary_counts.items())
    return overlap / summary_total


def bleu2_score(
    summary_tokens: list[str],
    reference_unigrams: Counter[str],
    reference_bigrams: Counter[tuple[str, str]],
    avg_reference_length: float,
) -> float:
    if not summary_tokens:
        return 0.0
    unigram_precision = modified_precision(Counter(summary_tokens), reference_unigrams)
    bigram_precision = modified_precision(Counter(make_ngrams(summary_tokens, 2)), reference_bigrams)
    if unigram_precision <= 0.0:
        return 0.0
    smoothed_bigram = bigram_precision if bigram_precision > 0.0 else 1e-9
    summary_length = len(summary_tokens)
    reference_length = max(avg_reference_length, 1.0)
    brevity_penalty = 1.0 if summary_length >= reference_length else math.exp(1.0 - (reference_length / summary_length))
    return brevity_penalty * math.exp((math.log(unigram_precision) + math.log(smoothed_bigram)) / 2.0)


def unigram_perplexity(
    summary_tokens: list[str],
    reference_unigrams: Counter[str],
    reference_token_count: int,
) -> float:
    if not summary_tokens or reference_token_count <= 0:
        return 0.0
    vocab_size = max(len(reference_unigrams), 1)
    denominator = reference_token_count + vocab_size
    neg_log_prob_sum = 0.0
    for token in summary_tokens:
        prob = (reference_unigrams.get(token, 0) + 1.0) / denominator
        neg_log_prob_sum -= math.log(prob)
    return math.exp(neg_log_prob_sum / len(summary_tokens))


def clamp_confidence(value: Any) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.45
    return max(0.0, min(1.0, val))


def has_branding_evidence(text: str) -> bool:
    cleaned = normalize_for_cache(text)
    if not BRANDING_JUDGMENT_RE.search(cleaned):
        return False
    if BRANDING_ENTITY_RE.search(cleaned):
        return True
    component_context = (
        FOOD_CONTEXT_RE.search(cleaned)
        or FACILITY_PHYSICAL_CONTEXT_RE.search(cleaned)
        or LOCATION_CONVENIENCE_RE.search(cleaned)
        or SERVICE_HUMAN_RE.search(cleaned)
        or SERVICE_ACTION_RE.search(cleaned)
    )
    return not bool(component_context)


def route_aspect_by_definition(candidate: str, text: str) -> str:
    cleaned = normalize_for_cache(text)
    candidate = candidate if candidate in ASPECTS else fallback_aspect(cleaned)
    if candidate == "loyalty" and not LOYALTY_INTENT_RE.search(cleaned):
        return fallback_aspect(cleaned, allow_branding=False)
    if LOYALTY_INTENT_RE.search(cleaned):
        return "loyalty"
    if candidate == "branding" and not has_branding_evidence(cleaned):
        return fallback_aspect(cleaned, allow_branding=False)
    if PHYSICAL_SECURITY_RE.search(cleaned):
        return "facility"
    if SUBJECTIVE_SAFETY_RE.search(cleaned) and not (
        FOOD_CONTEXT_RE.search(cleaned)
        or AMENITY_CONTEXT_RE.search(cleaned)
        or TRANSPORT_AMENITY_RE.search(cleaned)
    ):
        return "experience"
    if VALUE_CONTEXT_RE.search(cleaned) and (
        FOOD_CONTEXT_RE.search(cleaned)
        or AMENITY_CONTEXT_RE.search(cleaned)
        or TRANSPORT_AMENITY_RE.search(cleaned)
    ):
        return "amenity"
    if VALUE_CONTEXT_RE.search(cleaned):
        return "experience"
    if FOOD_CONTEXT_RE.search(cleaned):
        if SERVICE_HUMAN_RE.search(cleaned) and re.search(
            r"\b(waiter|waitress|restaurant service|served|serve|service|slow|rude|attentive|"
            r"phuc vu|phục vụ|boi ban|bồi bàn|nhan vien|nhân viên|le tan|lễ tân)\b",
            cleaned,
            re.IGNORECASE,
        ):
            return "service"
        return "amenity"
    if TRANSPORT_AMENITY_RE.search(cleaned):
        return "amenity"
    if AMENITY_CONTEXT_RE.search(cleaned):
        return "amenity"
    if INTERIOR_DESIGN_RE.search(cleaned):
        return "facility"
    if FACILITY_LOCATION_VIEW_RE.search(cleaned):
        return "facility"
    if SERVICE_HUMAN_RE.search(cleaned) or SERVICE_ACTION_RE.search(cleaned):
        return "service"
    if VIEW_NOISE_CONTEXT_RE.search(cleaned):
        if FACILITY_PHYSICAL_CONTEXT_RE.search(cleaned):
            return "facility"
        if EXPERIENCE_FEELING_RE.search(cleaned) or candidate in {"facility", "experience"}:
            return "experience"
    return candidate


def canonicalize_aspect_label(value: Any) -> str:
    aspect = str(value or "").strip().lower()
    return aspect if aspect in ASPECTS else ""


def dominant_guardrail_aspect(text: str, candidate: str = "") -> tuple[str, str]:
    """Return a high-confidence aspect route based on concrete target anchors.

    This is intentionally conservative and is used after ABSA extraction, before
    cluster/final-summary aggregation. It prevents broad cluster labels from
    pulling room/bathroom/noise evidence into amenity or food/wifi/pool evidence
    into facility.
    """
    lower = normalize_for_cache(text)
    if not lower:
        return canonicalize_aspect_label(candidate), ""
    if LOYALTY_INTENT_RE.search(lower):
        return "loyalty", "explicit_loyalty_intent"
    if SERVICE_STRONG_ANCHOR_RE.search(lower):
        if FOOD_CONTEXT_RE.search(lower) and not re.search(
            r"\b(staff|employee|waiter|waitress|reception|manager|concierge|"
            r"nhan vien|nhân viên|boi ban|bồi bàn|le tan|lễ tân|quan ly|quản lý)\b",
            lower,
            re.IGNORECASE,
        ):
            return "amenity", "food_or_restaurant_target"
        return "service", "service_human_or_process_anchor"
    taxonomy_aspect, taxonomy_reason = taxonomy_guardrail_aspect(lower, candidate)
    if taxonomy_aspect:
        return taxonomy_aspect, taxonomy_reason
    if AMENITY_STRONG_ANCHOR_RE.search(lower):
        return "amenity", "amenity_target_anchor"
    if FACILITY_STRONG_ANCHOR_RE.search(lower):
        return "facility", "facility_physical_anchor"
    if EXPERIENCE_STRONG_ANCHOR_RE.search(lower):
        return "experience", "experience_feeling_anchor"
    return canonicalize_aspect_label(candidate), ""


def guardrail_aspect_assignment(aspect: Any, text: str) -> dict[str, str]:
    original = canonicalize_aspect_label(aspect) or fallback_aspect(text)
    routed = normalize_aspect(original, text)
    anchored, reason = dominant_guardrail_aspect(text, routed)
    final_aspect = anchored or routed
    action = "kept" if final_aspect == original else "rerouted"
    if final_aspect == routed and routed != original:
        reason = reason or "definition_reroute"
    return {
        "original_aspect": original,
        "routed_aspect": routed,
        "final_aspect": final_aspect,
        "action": action,
        "reason": reason,
        "version": ASPECT_GUARDRAIL_VERSION,
    }


def normalize_aspect(value: Any, text: str = "") -> str:
    aspect = canonicalize_aspect_label(value)
    if aspect in ASPECTS:
        return route_aspect_by_definition(aspect, text)
    return fallback_aspect(text)


def normalize_sentiment(value: Any) -> str:
    sentiment = str(value or "").strip().lower()
    if sentiment in SENTIMENTS:
        return sentiment
    if "pos" in sentiment or "tích cực" in sentiment or "tich cuc" in sentiment:
        return "positive"
    if "neg" in sentiment or "tiêu cực" in sentiment or "tieu cuc" in sentiment:
        return "negative"
    return "neutral"


def normalize_language_code(value: Any) -> str:
    language = str(value or "").strip().lower()
    if not language:
        return "unknown"
    language = re.split(r"[-_ ]+", language)[0]
    aliases = {
        "eng": "en",
        "english": "en",
        "vie": "vi",
        "vietnamese": "vi",
        "cn": "zh",
        "jpn": "ja",
        "japanese": "ja",
        "kor": "ko",
        "korean": "ko",
        "thai": "th",
    }
    return aliases.get(language, language or "unknown")


def segment_cache_hash(aspect: str, text: str) -> str:
    payload = (
        f"{SENTIMENT_CLASSIFICATION_PROMPT_VERSION}||"
        f"{normalize_aspect(aspect, text)}||{normalize_for_cache(text)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preferred_text(segment: dict[str, Any], language: str) -> str:
    target_language = "vi" if language == "vi" else "en"
    value = clean_text(segment.get(f"normalized_text_{target_language}", ""))
    if value:
        return value
    return clean_text(segment.get("segment_text", ""))


NON_ACTIONABLE_UNIT_RE = re.compile(
    r"^(?:none|nothing|nil|n/?a|no comment|no complaints?|khong co gi|không có gì|"
    r"khong|không|khong co|không có|binh thuong|bình thường)[.!?。！？…]*$",
    re.IGNORECASE,
)
LOW_VALUE_CONTEXT_RE = re.compile(
    r"(?:không biết|khong biet).{0,80}(?:không tắm biển|khong tam bien)|"
    r"(?:not sure|do not know|don't know).{0,80}(?:did not use|didn't use|not use)",
    re.IGNORECASE,
)
HOTEL_CONTEXT_RE = re.compile(
    r"\b(hotel|hostel|homestay|property|place|stay|accommodation|everything|"
    r"khach san|khách sạn|nha nghi|nhà nghỉ|homestay|cho o|chỗ ở|noi nay|nơi này|"
    r"tat ca|tất cả|moi thu|mọi thứ)\b",
    re.IGNORECASE,
)
FALLBACK_SPLIT_TARGET_RE = (
    r"(?:staff|host|owner|manager|reception(?:ist)?|housekeeping|service|"
    r"room|rooms|bathroom|toilet|bed|window|balcony|elevator|lift|view|location|"
    r"breakfast|buffet|restaurant|cafe|coffee|bar|pool|gym|spa|wifi|food|drink|"
    r"nhan vien|nhân viên|le tan|lễ tân|chu nha|chủ nhà|quan ly|quản lý|"
    r"phong|phòng|nha tam|nhà tắm|giuong|giường|ban cong|ban công|thang may|thang máy|"
    r"view|vi tri|vị trí|bua sang|bữa sáng|an sang|ăn sáng|nha hang|nhà hàng|"
    r"ho boi|hồ bơi|be boi|bể bơi|wifi|do an|đồ ăn|do uong|đồ uống)"
)
FALLBACK_CLAUSE_SPLIT_RE = re.compile(
    rf"(?:[;,]\s+|\s+(?:but|while|whereas|nhưng|tuy nhiên)\s+|"
    rf"\s+and\s+(?=(?:the|a|an)\s+)|\s+và\s+(?=(?:các|những|phòng|nhân viên|lễ tân|bữa sáng|vị trí|view)\b))"
    rf"(?=(?:the\s+|a\s+|an\s+|các\s+|những\s+)?{FALLBACK_SPLIT_TARGET_RE}\b)",
    re.IGNORECASE,
)


def is_non_actionable_unit(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True
    lower = normalize_for_cache(cleaned)
    return bool(NON_ACTIONABLE_UNIT_RE.fullmatch(lower) or LOW_VALUE_CONTEXT_RE.search(lower))


def has_explicit_hotel_signal(text: str) -> bool:
    lower = normalize_for_cache(text)
    if is_non_actionable_unit(lower):
        return False
    for keywords in ASPECT_KEYWORDS.values():
        if any(contains_keyword(lower, keyword) for keyword in keywords):
            return True
    if HOTEL_CONTEXT_RE.search(lower) and (POSITIVE_RE.search(lower) or NEGATIVE_RE.search(lower)):
        return True
    return False


def concrete_target_flags(text: str) -> dict[str, bool]:
    lower = normalize_for_cache(text)
    return {
        "service": bool(SERVICE_HUMAN_RE.search(lower) or SERVICE_ACTION_RE.search(lower)),
        "room_or_building": bool(
            FACILITY_PHYSICAL_CONTEXT_RE.search(lower)
            or INTERIOR_DESIGN_RE.search(lower)
            or PHYSICAL_SECURITY_RE.search(lower)
        ),
        "location_or_view": bool(FACILITY_LOCATION_VIEW_RE.search(lower) or VIEW_NOISE_CONTEXT_RE.search(lower)),
        "food_or_beverage": bool(FOOD_CONTEXT_RE.search(lower)),
        "amenity": bool(AMENITY_CONTEXT_RE.search(lower) or TRANSPORT_AMENITY_RE.search(lower)),
        "loyalty": bool(LOYALTY_INTENT_RE.search(lower)),
    }


def has_multiple_concrete_targets(text: str) -> bool:
    flags = concrete_target_flags(text)
    return sum(1 for value in flags.values() if value) >= 2


def should_reuse_legacy_aspect_segments(text: str, segments: list[dict[str, Any]]) -> bool:
    cleaned = clean_text(text)
    if not segments:
        return not has_explicit_hotel_signal(cleaned)
    if is_non_actionable_unit(cleaned):
        return True
    if has_multiple_concrete_targets(cleaned) and len(segments) <= 1:
        return False
    if re.search(r"\s+(?:but|nhưng|tuy nhiên)\s+", cleaned, flags=re.IGNORECASE) and len(segments) <= 1:
        return False
    if len(segments) == 1:
        segment = segments[0]
        segment_text = clean_text(segment.get("segment_text", "")) or clean_text(segment.get("text", ""))
        aspect = normalize_aspect(segment.get("aspect"), segment_text or cleaned)
        if aspect == "experience" and any(
            concrete_target_flags(cleaned)[key]
            for key in ("service", "room_or_building", "location_or_view", "food_or_beverage", "amenity")
        ):
            return False
    return True


def infer_contrast_subject(text: str) -> tuple[str, str]:
    cleaned = clean_text(text)
    match = re.match(
        r"(?P<subject>.{2,80}?)\s+(?P<copula>was|were|is|are|had|has|có|co|là|la)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_text(match.group("subject")), clean_text(match.group("copula"))
    words = cleaned.split()
    if 1 <= len(words) <= 8:
        return cleaned, "is"
    return "", ""


def split_contrast_fallback_units(text: str) -> list[str]:
    cleaned = clean_text(text)
    parts = re.split(r"\s+(?:but|nhưng|tuy nhiên)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []
    left, right = [clean_text(part.strip(" ,;")) for part in parts]
    if not left or not right or not has_explicit_hotel_signal(left):
        return []
    left_polar = bool(POSITIVE_RE.search(left) or NEGATIVE_RE.search(left))
    right_polar = bool(POSITIVE_RE.search(right) or NEGATIVE_RE.search(right))
    if not (left_polar and right_polar):
        return []
    if not re.search(rf"\b{FALLBACK_SPLIT_TARGET_RE}\b", normalize_for_cache(right), flags=re.IGNORECASE):
        subject, copula = infer_contrast_subject(left)
        if not subject:
            return []
        right = f"{subject} {copula} {right}"
    return [left, right] if has_explicit_hotel_signal(right) else []


def split_fallback_opinion_units(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    contrast_parts = split_contrast_fallback_units(cleaned)
    if contrast_parts:
        return contrast_parts
    parts = [part.strip(" ,;") for part in FALLBACK_CLAUSE_SPLIT_RE.split(cleaned)]
    parts = [part for part in parts if part]
    if len(parts) <= 1:
        return [cleaned]
    return [part for part in parts if has_explicit_hotel_signal(part)] or [cleaned]


def make_fallback_aspect_segment(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    return {
        "aspect": fallback_aspect(cleaned),
        "segment_text": cleaned,
        "detected_language": "unknown",
        "normalized_text_vi": cleaned,
        "normalized_text_en": cleaned,
        "confidence": 0.35,
        "split_reason": "rule fallback after empty or failed aspect extraction",
    }


def fallback_aspect_segments(text: str) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    if not cleaned or is_non_actionable_unit(cleaned):
        return []
    return [make_fallback_aspect_segment(part) for part in split_fallback_opinion_units(cleaned)]


def fallback_sentiment_item(aspect: str, text: str) -> dict[str, Any]:
    sentiment, confidence = fallback_sentiment(text)
    return {
        "aspect": normalize_aspect(aspect, text),
        "sentiment": sentiment,
        "confidence": confidence,
    }


def parse_qwen_aspect_segments(content: str, sentences: list[str]) -> list[list[dict[str, Any]]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    return normalize_qwen_aspect_items(raw_items, sentences)


def normalize_qwen_aspect_items(raw_items: Any, sentences: list[str]) -> list[list[dict[str, Any]]]:
    by_id: dict[int, list[dict[str, Any]]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if not (0 <= idx < len(sentences)):
                continue
            raw_segments = item.get("segments", [])
            segments: list[dict[str, Any]] = []
            if isinstance(raw_segments, list):
                for raw_segment in raw_segments:
                    if not isinstance(raw_segment, dict):
                        continue
                    segment_text = (
                        clean_text(raw_segment.get("segment_text", ""))
                        or clean_text(raw_segment.get("text", ""))
                        or clean_text(raw_segment.get("opinion_unit", ""))
                        or clean_text(sentences[idx])
                    )
                    if not segment_text:
                        continue
                    aspect = normalize_aspect(raw_segment.get("aspect"), segment_text)
                    normalized_vi = clean_text(raw_segment.get("normalized_text_vi", "")) or segment_text
                    normalized_en = clean_text(raw_segment.get("normalized_text_en", "")) or segment_text
                    segments.append(
                        {
                            "aspect": aspect,
                            "segment_text": segment_text,
                            "detected_language": normalize_language_code(
                                raw_segment.get("detected_language", "") or raw_segment.get("lang", "")
                            ),
                            "normalized_text_vi": normalized_vi,
                            "normalized_text_en": normalized_en,
                            "confidence": clamp_confidence(raw_segment.get("confidence", 0.65)),
                            "start_char": int(raw_segment.get("start_char", -1) or -1),
                            "end_char": int(raw_segment.get("end_char", -1) or -1),
                            "split_reason": clean_text(raw_segment.get("split_reason", ""))[:120],
                        }
                    )
            by_id[idx] = segments
    out: list[list[dict[str, Any]]] = []
    for idx, sentence in enumerate(sentences):
        if idx not in by_id:
            out.append(fallback_aspect_segments(sentence))
            continue
        segments = by_id[idx]
        if not segments and has_explicit_hotel_signal(sentence):
            segments = fallback_aspect_segments(sentence)
        out.append(segments)
    return out


def parse_qwen_aspect_segments_lenient(content: str, sentences: list[str]) -> list[list[dict[str, Any]]]:
    try:
        return parse_qwen_aspect_segments(content, sentences)
    except json.JSONDecodeError:
        pass

    repaired_items: list[dict[str, Any]] = []
    item_pattern = re.compile(r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"segments"\s*:\s*\[(.*?)\]\s*\}', re.DOTALL)
    segment_pattern = re.compile(
        r'\{\s*"aspect"\s*:\s*"([^"]+)"\s*,\s*"(?:segment_text|text|opinion_unit)"\s*:\s*"((?:[^"\\]|\\.)*)"'
        r'(?:\s*,\s*"(?:detected_language|lang)"\s*:\s*"([^"]*)")?'
        r'(?:\s*,\s*"normalized_text_vi"\s*:\s*"((?:[^"\\]|\\.)*)")?'
        r'(?:\s*,\s*"normalized_text_en"\s*:\s*"((?:[^"\\]|\\.)*)")?'
        r'(?:\s*,\s*"confidence"\s*:\s*([0-9.]+))?',
        re.DOTALL,
    )
    for item_match in item_pattern.finditer(content):
        idx = int(item_match.group(1))
        raw_segments = item_match.group(2)
        segments: list[dict[str, Any]] = []
        for segment_match in segment_pattern.finditer(raw_segments):
            try:
                text_value = json.loads(f'"{segment_match.group(2)}"')
            except json.JSONDecodeError:
                text_value = segment_match.group(2)
            try:
                normalized_vi = json.loads(f'"{segment_match.group(4)}"') if segment_match.group(4) else text_value
            except json.JSONDecodeError:
                normalized_vi = segment_match.group(4) or text_value
            try:
                normalized_en = json.loads(f'"{segment_match.group(5)}"') if segment_match.group(5) else text_value
            except json.JSONDecodeError:
                normalized_en = segment_match.group(5) or text_value
            segments.append(
                {
                    "aspect": segment_match.group(1),
                    "text": text_value,
                    "lang": segment_match.group(3) or "unknown",
                    "normalized_text_vi": normalized_vi,
                    "normalized_text_en": normalized_en,
                    "confidence": float(segment_match.group(6) or 0.65),
                }
            )
        repaired_items.append({"id": idx, "segments": segments})

    if repaired_items:
        return normalize_qwen_aspect_items(repaired_items, sentences)
    return [fallback_aspect_segments(sentence) for sentence in sentences]


def parse_qwen_sentiment_items(content: str, items: list[dict[str, str]]) -> list[dict[str, Any]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(items):
                by_id[idx] = {
                    "aspect": normalize_aspect(items[idx]["aspect"], items[idx]["text"]),
                    "sentiment": normalize_sentiment(item.get("sentiment", "")),
                    "confidence": clamp_confidence(item.get("confidence", 0.65)),
                }
    out = []
    for idx, item in enumerate(items):
        out.append(by_id.get(idx, fallback_sentiment_item(item["aspect"], item["text"])))
    return out


def jsonish_object_slices(content: str, array_key: str = "items") -> list[str]:
    key_pos = content.find(f'"{array_key}"')
    scan_from = content.find("[", key_pos if key_pos >= 0 else 0)
    if scan_from < 0:
        scan_from = 0
    slices: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for idx in range(scan_from, len(content)):
        char = content[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                slices.append(content[start : idx + 1])
                start = -1
    return slices


def decode_json_string_fragment(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def regex_string_field(text: str, key: str) -> str:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    if not match:
        return ""
    return decode_json_string_fragment(match.group(1))


def regex_float_field(text: str, key: str, default: float) -> float:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9.]+)', text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def regex_int_field(text: str, key: str) -> int | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def regex_string_list_field(text: str, key: str) -> list[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if not match:
        return []
    values = []
    for value_match in re.finditer(r'"((?:[^"\\]|\\.)*)"', match.group(1), flags=re.DOTALL):
        value = clean_text(decode_json_string_fragment(value_match.group(1)))
        if value:
            values.append(value)
    return values


def parse_qwen_sentiment_items_lenient(content: str, items: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        return parse_qwen_sentiment_items(content, items)
    except json.JSONDecodeError:
        pass

    by_id: dict[int, dict[str, Any]] = {}
    for item_text in jsonish_object_slices(content):
        try:
            raw_item = json_loads_lenient(item_text)
        except json.JSONDecodeError:
            raw_item = {
                "id": regex_int_field(item_text, "id"),
                "sentiment": regex_string_field(item_text, "sentiment"),
                "confidence": regex_float_field(item_text, "confidence", 0.65),
            }
        if not isinstance(raw_item, dict):
            continue
        try:
            idx = int(raw_item.get("id"))
        except Exception:
            continue
        if 0 <= idx < len(items):
            by_id[idx] = {
                "aspect": normalize_aspect(items[idx]["aspect"], items[idx]["text"]),
                "sentiment": normalize_sentiment(raw_item.get("sentiment", "")),
                "confidence": clamp_confidence(raw_item.get("confidence", 0.65)),
            }
    if not by_id:
        raise json.JSONDecodeError("could not repair Qwen sentiment JSON", content, 0)
    return [by_id.get(idx, fallback_sentiment_item(item["aspect"], item["text"])) for idx, item in enumerate(items)]


def parse_qwen_cluster_assignments(content: str, items: list[dict[str, str]]) -> list[dict[str, Any]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(items):
                by_id[idx] = normalize_cluster_assignment(
                    item,
                    items[idx]["aspect"],
                    items[idx]["sentiment"],
                    items[idx]["text"],
                )
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        out.append(
            by_id.get(
                idx,
                rule_cluster_assignment(item["aspect"], item["sentiment"], item["text"]),
            )
        )
    return out


def parse_qwen_cluster_assignments_lenient(content: str, items: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        return parse_qwen_cluster_assignments(content, items)
    except json.JSONDecodeError:
        pass

    by_id: dict[int, dict[str, Any]] = {}
    for item_text in jsonish_object_slices(content):
        try:
            raw_item = json_loads_lenient(item_text)
        except json.JSONDecodeError:
            raw_item = {
                "id": regex_int_field(item_text, "id"),
                "aspect": regex_string_field(item_text, "aspect"),
                "sentiment": regex_string_field(item_text, "sentiment"),
                "cluster_code": regex_string_field(item_text, "cluster_code"),
                "cluster_label": regex_string_field(item_text, "cluster_label"),
                "descriptors": regex_string_list_field(item_text, "descriptors"),
                "confidence": regex_float_field(item_text, "confidence", 0.65),
                "source": "llm_repaired",
            }
        if not isinstance(raw_item, dict):
            continue
        try:
            idx = int(raw_item.get("id"))
        except Exception:
            continue
        if 0 <= idx < len(items):
            raw_item["source"] = clean_text(raw_item.get("source", "")) or "llm_repaired"
            by_id[idx] = normalize_cluster_assignment(
                raw_item,
                items[idx]["aspect"],
                items[idx]["sentiment"],
                items[idx]["text"],
            )
    if not by_id:
        raise json.JSONDecodeError("could not repair Qwen cluster JSON", content, 0)
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        out.append(
            by_id.get(
                idx,
                rule_cluster_assignment(item["aspect"], item["sentiment"], item["text"]),
            )
        )
    return out


def local_aspect_summary(item: dict[str, Any]) -> dict[str, str]:
    aspect = str(item.get("aspect", ""))
    sentiment = normalize_sentiment(item.get("final_sentiment", ""))
    counts = item.get("counts", {}) if isinstance(item.get("counts", {}), dict) else {}
    representative_vi = clean_text(str(item.get("representative_vi", "")))
    representative_en = clean_text(str(item.get("representative_en", "")))
    if representative_vi and not representative_en:
        representative_en = representative_vi
    if representative_en and not representative_vi:
        representative_vi = representative_en
    if aspect == "branding" and not has_branding_evidence(f"{representative_vi} {representative_en}"):
        return {"summary_vi": "", "summary_en": ""}
    clusters_by_sentiment = item.get("clusters_by_sentiment", {})
    cluster_items: list[dict[str, Any]] = []
    if isinstance(clusters_by_sentiment, dict):
        for value in clusters_by_sentiment.values():
            if isinstance(value, list):
                cluster_items.extend(cluster for cluster in value if isinstance(cluster, dict))
    positive_count = int(counts.get("positive", 0))
    negative_count = int(counts.get("negative", 0))
    neutral_count = int(counts.get("neutral", 0))
    if isinstance(clusters_by_sentiment, dict):
        sentiment_summaries: dict[str, str] = {}
        for current_sentiment in SENTIMENTS:
            current_clusters = [
                {**cluster, "sentiment": current_sentiment}
                for cluster in clusters_by_sentiment.get(current_sentiment, [])
                if isinstance(cluster, dict)
            ]
            theme = (
                summarize_cluster_evidence(
                    current_clusters,
                    aspect,
                    max_clusters=4,
                    sentiment=current_sentiment,
                )
                or summarize_cluster_themes(current_clusters, aspect)
            )
            if not theme:
                continue
            sentiment_summaries[current_sentiment] = finish_sentence(theme)
        if sentiment_summaries:
            summary_row = {
                "aspect": aspect,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_summary": sentiment_summaries.get("positive", ""),
                "negative_summary": sentiment_summaries.get("negative", ""),
                "neutral_summary": sentiment_summaries.get("neutral", ""),
            }
            vi = build_overall_aspect_summary(summary_row)
            en = build_overall_aspect_summary(summary_row, "en")
            if vi:
                return {
                    "summary_vi": clean_summary_text(vi, 1600),
                    "summary_en": clean_summary_text(en, 1600),
                }
    mixed_feedback = sum(1 for value in (positive_count, negative_count, neutral_count) if value > 0) > 1
    detail_vi = representative_vi.rstrip(". ")
    detail_en = representative_en.rstrip(". ")
    cluster_detail = summarize_cluster_evidence(cluster_items, aspect, max_clusters=4) or summarize_cluster_themes(
        cluster_items,
        aspect,
    )
    if sentiment == "positive":
        vi_lead = "ý kiến khá trái chiều nhưng nhìn chung vẫn tích cực" if mixed_feedback else "phản hồi nhìn chung tích cực"
        en_lead = "feedback is mixed but still leans positive" if mixed_feedback else "feedback is broadly positive"
        vi_detail = cluster_detail or detail_vi or "nhiều điểm được đánh giá tốt"
        en_detail = cluster_detail or detail_en or "several positive points"
        vi = f"{vi_lead.capitalize()}, nổi bật ở {vi_detail}."
        en = f"{en_lead.capitalize()}, with highlights around {en_detail}."
    elif sentiment == "negative":
        vi_lead = "ý kiến khá trái chiều nhưng nhìn chung vẫn tiêu cực" if mixed_feedback else "phản hồi nhìn chung tiêu cực"
        en_lead = "feedback is mixed but still leans negative" if mixed_feedback else "feedback is broadly negative"
        vi_detail = cluster_detail or detail_vi or "một số vấn đề cần được ưu tiên cải thiện"
        en_detail = cluster_detail or detail_en or "several issues that need attention"
        vi = f"{vi_lead.capitalize()}, chủ yếu xoay quanh {vi_detail}."
        en = f"{en_lead.capitalize()}, mainly around {en_detail}."
    else:
        vi_lead = "ý kiến hiện khá trung tính và chưa có xu hướng thật sự rõ" if mixed_feedback else "ý kiến hiện khá trung tính"
        en_lead = "comments are fairly neutral without a strong direction yet" if mixed_feedback else "comments are fairly neutral"
        vi_detail = cluster_detail or detail_vi or "những ghi nhận mang tính mô tả trong quá trình lưu trú"
        en_detail = cluster_detail or detail_en or "descriptive observations from the stay"
        vi = f"{vi_lead.capitalize()}, chủ yếu đề cập đến {vi_detail}."
        en = f"{en_lead.capitalize()}, mostly touching on {en_detail}."
    return {
        "summary_vi": clean_summary_text(vi, 1600),
        "summary_en": clean_summary_text(en, 1600),
    }


def local_final_sentiment_summary(item: dict[str, Any]) -> str:
    aspect = str(item.get("aspect", ""))
    sentiment = normalize_sentiment(item.get("sentiment", ""))
    count = int(item.get("count", 0) or 0)
    samples = [clean_text(sample) for sample in item.get("samples", []) if clean_text(sample)]
    sentiment_label = {
        "positive": "tích cực",
        "neutral": "bình thường hoặc chưa rõ cảm xúc",
        "negative": "tiêu cực",
    }.get(sentiment, sentiment)
    if count <= 0:
        return f"Không có đủ nhận xét {sentiment_label} trong nhóm dữ liệu này."
    evidence = "; ".join(samples[:3])
    if evidence:
        return f"Các ý {sentiment_label} thường gặp gồm: {evidence}."
    return f"Có {count:,} nhận xét {sentiment_label}."


def summarize_evidence_themes(samples: list[str], aspect: str) -> str:
    text = normalize_for_cache(" ".join(samples))
    if aspect == "branding" and not has_branding_evidence(text):
        return ""
    theme_rules = {
        "facility": [
            ("phòng, nội thất và trạng thái vật lý của không gian", ["phong", "phòng", "room", "clean", "sach", "sạch", "dep", "đẹp"]),
            ("giường, đệm và mức độ thoải mái của phòng", ["bed", "giuong", "giường", "dem", "đệm", "comfortable"]),
            ("ban công, cửa sổ, cách âm hoặc thiết kế vật lý", ["balcony", "window", "soundproof", "ban cong", "cửa sổ", "cach am", "cách âm", "design", "thiet ke", "thiết kế"]),
        ],
        "amenity": [
            ("sự hiện diện của bữa sáng, nhà hàng và tiện ích ăn uống", ["breakfast", "nha hang", "nhà hàng", "an sang", "ăn sáng"]),
            ("vị trí thuận tiện, gần điểm tham quan hoặc dễ di chuyển", ["location", "near", "nearby", "vi tri", "vị trí", "gan", "gần", "center", "trung tam", "trung tâm", "checkin"]),
            ("wifi, bãi đỗ xe, thanh toán, an ninh hoặc tiện ích đi kèm", ["wifi", "parking", "payment", "security", "thanh toan", "thanh toán", "an ninh"]),
        ],
        "service": [
            ("sự nhiệt tình và thái độ hỗ trợ của nhân viên", ["staff", "nhan vien", "nhân viên", "chu", "chủ", "helpful", "ho tro", "hỗ trợ", "nhiệt tình"]),
            ("chất lượng phục vụ và mức độ chu đáo", ["service", "phuc vu", "phục vụ", "chu dao", "chu đáo", "kind"]),
            ("chất lượng món ăn hoặc tốc độ phục vụ đồ ăn", ["food", "meal", "breakfast", "do an", "đồ ăn", "mon an", "món ăn", "ngon", "dở", "slow", "chậm"]),
            ("khả năng phản hồi và giao tiếp với khách", ["call", "message", "goi dien", "gọi điện", "nhan tin", "nhắn tin"]),
        ],
        "experience": [
            ("trải nghiệm lưu trú, cảm giác thư giãn và sự thoải mái", ["experience", "trai nghiem", "trải nghiệm", "retreat", "chill", "relax", "thu gian", "thư giãn"]),
            ("không khí, sự yên bình và cảm giác dễ chịu", ["atmosphere", "khong khi", "không khí", "peaceful", "yen binh", "yên bình", "de chiu", "dễ chịu"]),
            ("cảnh quan hoặc view như một cảm nhận tổng thể", ["scenery", "landscape", "khung canh", "khung cảnh", "canh quan", "cảnh quan", "view"]),
            ("những bất tiện ảnh hưởng đến cảm giác lưu trú", ["noise", "noisy", "on", "ồn", "khó chịu", "kho chiu"]),
        ],
        "branding": [
            ("mức độ khớp giữa trải nghiệm thực tế và ảnh, mô tả hoặc thông tin đặt phòng", ["description", "photo", "picture", "image", "booking", "listing", "mo ta", "mô tả", "hinh", "hình", "anh", "ảnh", "thong tin", "thông tin"]),
        ],
        "loyalty": [
            ("ý định quay lại hoặc giới thiệu cho người khác", ["return", "come back", "recommend", "gioi thieu"]),
        ],
    }
    selected = [
        label
        for label, keywords in theme_rules.get(aspect, [])
        if any(keyword in text for keyword in keywords)
    ]
    if selected:
        return ", ".join(selected[:2])
    fallback = [clean_text(sample).rstrip(".") for sample in samples[:2] if clean_text(sample)]
    if fallback:
        return " và ".join(fallback)[:220]
    return "các ý kiến được ghi nhận trong dữ liệu"


def summarize_cluster_themes(clusters: Any, aspect: str) -> str:
    if not isinstance(clusters, list):
        return ""
    labels = []
    for cluster in sorted(
        (item for item in clusters if isinstance(item, dict)),
        key=lambda item: (-int(item.get("count", 0) or 0), clean_text(item.get("label", ""))),
    ):
        label = clean_text(cluster.get("label", ""))
        if label and label not in labels:
            labels.append(label)
    if aspect == "branding" and labels:
        text = normalize_for_cache(" ".join(labels))
        if not has_branding_evidence(text):
            return ""
    return ", ".join(labels[:3])


def cluster_business_insight(
    *,
    aspect: str,
    sentiment: str,
    label: str,
    descriptors: list[str],
    samples: list[str],
) -> str:
    """Collapse repetitive cluster descriptors into one manager-facing insight."""
    joined = normalize_for_cache(" ".join([label, *descriptors, *samples]))
    sentiment = normalize_sentiment(sentiment)
    label_clean = clean_text(label)

    if label_clean == "Revisit Intention":
        if sentiment == "negative":
            return "ý định quay lại bị ảnh hưởng bởi các điểm chưa đáp ứng kỳ vọng trong trải nghiệm lưu trú"
        if sentiment == "positive":
            return "khách thể hiện ý định quay lại nhờ trải nghiệm lưu trú đủ thuyết phục"
        return "ý định quay lại được nhắc như một khả năng phụ thuộc vào việc cải thiện trải nghiệm"

    if label_clean == "Recommendation Intention":
        if sentiment == "negative":
            return "ý định giới thiệu bị hạn chế khi trải nghiệm chưa đủ nhất quán để khách khuyến nghị"
        if sentiment == "positive":
            return "khách sẵn sàng giới thiệu khách sạn cho người khác"
        return "ý định giới thiệu được nhắc ở mức mô tả hoặc điều kiện"

    if label_clean == "Customer Preference":
        if sentiment == "negative":
            return "khách có dấu hiệu cân nhắc lựa chọn khác khi trải nghiệm chưa phù hợp sở thích"
        if sentiment == "positive":
            return "khách thể hiện xu hướng ưu tiên lựa chọn khách sạn trong các lần lưu trú tương tự"
        return "sở thích lựa chọn của khách được nhắc như bối cảnh đánh giá"

    if label_clean == "Loyalty Behavior":
        if sentiment == "negative":
            return "hành vi trung thành chưa rõ vì trải nghiệm còn điểm làm giảm cam kết của khách"
        if sentiment == "positive":
            return "khách thể hiện hành vi ủng hộ hoặc gắn bó sau trải nghiệm lưu trú"
        return "hành vi trung thành được nhắc như một tín hiệu bối cảnh"

    if label_clean == "Brand Reputation":
        if sentiment == "negative":
            return "danh tiếng thương hiệu bị ảnh hưởng bởi trải nghiệm chưa đạt kỳ vọng"
        if sentiment == "positive":
            return "danh tiếng thương hiệu được củng cố nhờ các nhận xét tích cực"
        return "danh tiếng thương hiệu được nhắc như bối cảnh nhận diện khách sạn"

    if label_clean == "Brand Trust":
        if sentiment == "negative":
            return "niềm tin thương hiệu bị giảm khi trải nghiệm thực tế chưa nhất quán"
        if sentiment == "positive":
            return "khách thể hiện niềm tin vào chất lượng hoặc uy tín của khách sạn"
        return "niềm tin thương hiệu được nhắc ở mức mô tả"

    if label_clean == "Luxury Perception":
        if sentiment == "negative":
            return "cảm nhận cao cấp bị suy giảm khi trải nghiệm chưa tương xứng kỳ vọng"
        if sentiment == "positive":
            return "khách ghi nhận cảm giác sang trọng hoặc cao cấp của khách sạn"
        return "cảm nhận sang trọng được nhắc như một thuộc tính hình ảnh khách sạn"

    if label_clean == "Brand Consistency":
        if sentiment == "negative":
            return "tính nhất quán thương hiệu bị ảnh hưởng khi trải nghiệm thực tế chưa khớp kỳ vọng"
        if sentiment == "positive":
            return "trải nghiệm thực tế được nhìn nhận là phù hợp với hình ảnh hoặc tiêu chuẩn khách sạn"
        return "tính nhất quán thương hiệu được nhắc qua mức khớp giữa kỳ vọng và trải nghiệm"

    if label_clean == "Expectation Fulfillment":
        if sentiment == "negative":
            return "khách cho rằng trải nghiệm chưa đáp ứng đúng kỳ vọng ban đầu"
        if sentiment == "positive":
            return "trải nghiệm lưu trú đáp ứng hoặc vượt kỳ vọng của khách"
        return "mức đáp ứng kỳ vọng được nhắc như bối cảnh đánh giá"

    if aspect in {"branding", "loyalty"}:
        return ""

    if label_clean == "Breakfast Quality" or any(term in joined for term in ("breakfast", "bua sang", "an sang", "buffet")):
        if sentiment == "negative":
            if any(term in joined for term in ("variety", "diverse", "selection", "it lua chon", "ít lựa chọn", "it mon", "ít món", "don dieu", "đơn điệu", "phong phu", "phong phú", "da dang", "đa dạng")):
                return "khách cho rằng thực đơn bữa sáng còn đơn điệu và thiếu sự đa dạng"
            if any(term in joined for term in ("cold", "nguoi", "nguội", "khong tuoi", "không tươi", "not fresh", "quality", "kem", "kém", "bad")):
                return "chất lượng bữa sáng chưa ổn định, đặc biệt ở độ tươi và cách phục vụ món ăn"
            return "bữa sáng là nguồn phàn nàn cần kiểm tra về chất lượng và mức đáp ứng kỳ vọng"
        if sentiment == "positive":
            if any(term in joined for term in ("delicious", "ngon", "tasty", "hop khau vi", "excellent", "great")):
                return "bữa sáng được đánh giá tốt về hương vị và mức độ hợp khẩu vị"
            return "bữa sáng đóng góp tích cực vào trải nghiệm tiện ích của khách"

    if any(term in joined for term in ("wifi", "wi fi", "internet", "mang", "ket noi")):
        if sentiment == "negative":
            return "chất lượng kết nối wifi chưa ổn định, với các phản ánh về tín hiệu yếu hoặc mất kết nối"
        if sentiment == "positive":
            return "wifi đáp ứng tốt nhu cầu kết nối cơ bản của khách"

    if any(term in joined for term in ("pool", "ho boi", "be boi", "swimming")):
        if sentiment == "negative":
            if any(term in joined for term in ("small", "nho", "nhỏ", "dirty", "ban", "bẩn", "duc", "đục", "clean", "sach", "sạch")):
                return "trải nghiệm hồ bơi chưa đồng đều, chủ yếu do quy mô nhỏ hoặc độ sạch chưa đạt kỳ vọng"
            return "hồ bơi tạo phản hồi trái chiều và cần kiểm tra lại điều kiện vận hành"
        if sentiment == "positive":
            return "hồ bơi là tiện ích được ghi nhận tích cực nhờ tạo thêm không gian thư giãn"

    if any(term in joined for term in ("room", "phong", "bed", "giuong", "bathroom", "phong tam")):
        if sentiment == "negative":
            if any(term in joined for term in ("noise", "noisy", "on", "ồn", "soundproof", "cach am", "cách âm")):
                return "chất lượng cách âm và kiểm soát tiếng ồn trong phòng chưa đáp ứng kỳ vọng"
            if any(term in joined for term in ("dirty", "ban", "bẩn", "clean", "sach", "sạch", "smell", "mui", "mùi", "mold", "am", "ẩm")):
                return "vệ sinh và bảo trì phòng chưa đồng đều giữa các trải nghiệm lưu trú"
            if any(term in joined for term in ("old", "cu", "cũ", "broken", "hong", "hỏng", "maintenance", "bao tri", "bảo trì")):
                return "tình trạng phòng và trang thiết bị cần được bảo trì nhất quán hơn"
        if sentiment == "positive":
            if any(term in joined for term in ("clean", "sach", "sạch", "spacious", "rong", "rộng", "comfortable", "thoai mai", "thoải mái")):
                return "phòng được đánh giá tốt về độ sạch, độ rộng hoặc mức độ thoải mái"

    if any(term in joined for term in ("staff", "nhan vien", "manager", "quan ly", "host", "chu nha")):
        if sentiment == "negative":
            return "chất lượng tương tác của nhân viên chưa nhất quán và cần chuẩn hóa lại cách phục vụ"
        if sentiment == "positive":
            return "nhân viên là điểm mạnh nhờ thái độ thân thiện, hỗ trợ và tạo cảm giác được chăm sóc"

    if any(term in joined for term in ("food", "restaurant", "nha hang", "do an", "thuc an", "mon an", "beverage")):
        if sentiment == "negative":
            return "chất lượng đồ ăn thức uống hoặc trải nghiệm nhà hàng chưa ổn định"
        if sentiment == "positive":
            return "đồ ăn thức uống được ghi nhận tích cực về hương vị hoặc trải nghiệm phục vụ"

    return ""


def summarize_cluster_evidence(
    clusters: Any,
    aspect: str,
    max_clusters: int = 5,
    sentiment: str = "",
) -> str:
    if not isinstance(clusters, list):
        return ""
    parts: list[str] = []
    for cluster in sorted(
        (item for item in clusters if isinstance(item, dict)),
        key=lambda item: (-int(item.get("count", 0) or 0), clean_text(item.get("label", ""))),
    ):
        label = clean_text(cluster.get("label", ""))
        descriptors = unique_preserve_order(
            [str(value) for value in cluster.get("descriptors", []) if clean_text(value)],
            12,
        )
        samples = unique_preserve_order(
            [clean_text(sample).rstrip(".") for sample in cluster.get("samples", []) if clean_text(sample)],
            3,
        )
        cluster_sentiment = (
            normalize_sentiment(cluster.get("sentiment", ""))
            if cluster.get("sentiment")
            else normalize_sentiment(sentiment) if sentiment else ""
        )
        business_insight = cluster_business_insight(
            aspect=aspect,
            sentiment=cluster_sentiment,
            label=label,
            descriptors=descriptors,
            samples=samples,
        )
        if business_insight:
            parts.append(business_insight)
            if len(parts) >= max_clusters:
                break
            continue
        detail_values = descriptors[:]
        for sample in samples:
            if not any(normalize_for_cache(sample) == normalize_for_cache(value) for value in detail_values):
                detail_values.append(sample)
        detail = ", ".join(unique_preserve_order(detail_values, 14))
        if not detail:
            detail = label
        if not detail:
            continue
        if label:
            parts.append(f"{label}: {detail}")
        else:
            parts.append(detail)
        if len(parts) >= max_clusters:
            break
    evidence = "; ".join(parts)
    if aspect == "branding" and evidence and not has_branding_evidence(normalize_for_cache(evidence)):
        return ""
    return evidence[:900]


def local_final_sentiment_summary(item: dict[str, Any]) -> str:
    aspect = str(item.get("aspect", ""))
    sentiment = normalize_sentiment(item.get("sentiment", ""))
    count = int(item.get("count", 0) or 0)
    samples = [clean_text(sample) for sample in item.get("samples", []) if clean_text(sample)]
    sentiment_label = {
        "positive": "tích cực",
        "neutral": "trung lập hoặc chưa rõ cảm xúc",
        "negative": "tiêu cực",
    }.get(sentiment, sentiment)
    if count <= 0:
        return f"Không có phản hồi {sentiment_label} rõ ràng trong nhóm dữ liệu này."
    theme = (
        summarize_cluster_evidence(item.get("clusters"), aspect, sentiment=sentiment)
        or summarize_cluster_themes(item.get("clusters"), aspect)
        or summarize_evidence_themes(samples, aspect)
    )
    if aspect == "branding" and not theme:
        return "Chưa có đủ bằng chứng để kết luận về thương hiệu theo tiêu chí khớp giữa trải nghiệm thực tế và thông tin khách sạn đã cung cấp."
    if sentiment == "positive":
        return f"Phản hồi tích cực nổi bật ở {theme}."
    if sentiment == "negative":
        return f"Phản hồi tiêu cực chủ yếu liên quan đến {theme}."
    return f"Phản hồi trung lập hoặc chưa rõ cảm xúc xoay quanh {theme}."


def business_final_sentiment_summary(item: dict[str, Any]) -> str:
    aspect = str(item.get("aspect", ""))
    sentiment = normalize_sentiment(item.get("sentiment", ""))
    count = int(item.get("count", 0) or 0)
    samples = [clean_text(sample) for sample in item.get("samples", []) if clean_text(sample)]
    if count <= 0:
        return ""
    theme = (
        summarize_cluster_evidence(item.get("clusters"), aspect, sentiment=sentiment)
        or summarize_cluster_themes(item.get("clusters"), aspect)
        or summarize_evidence_themes(samples, aspect)
    )
    if aspect == "branding" and not theme:
        return ""
    return finish_sentence(theme)


def strip_business_summary_lead(summary: str, aspect_label: str) -> str:
    text = clean_text(summary).rstrip(".")
    prefixes = [
        f"Điểm được khen nhiều nhất về {aspect_label} là ",
        f"Phần phàn nàn chính về {aspect_label} tập trung vào ",
        f"Ý kiến trung lập về {aspect_label} chủ yếu xoay quanh ",
        "Điểm được khen nhiều nhất là ",
        "Phần phàn nàn chính tập trung vào ",
        "Ý kiến trung lập chủ yếu xoay quanh ",
        "Các lời khen được ghi nhận gồm ",
        "Các phàn nàn được ghi nhận gồm ",
        "Các ghi nhận trung lập gồm ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def normalize_insight_paragraph(text: str, max_chars: int = 0) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\b(với|gồm|bao gồm)\s+\d+\s+(tích cực|tiêu cực|trung lập)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(có|gồm)\s+\d+\s+(câu|ý kiến|phản hồi)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
    if max_chars and max_chars > 0 and len(cleaned) > max_chars:
        clipped = cleaned[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,.;")
        cleaned = clipped or cleaned[:max_chars]
    return cleaned.rstrip(".") + "."


def finish_sentence(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def lowercase_first(text: str) -> str:
    text = clean_text(text).strip()
    if not text:
        return ""
    return text[0].lower() + text[1:]


def aspect_sentiment_relation(
    positive_count: int,
    negative_count: int,
    neutral_count: int,
) -> str:
    total = positive_count + negative_count + neutral_count
    if total <= 0:
        return "thiếu tín hiệu rõ để kết luận"
    if positive_count > 0 and negative_count > 0:
        smaller = min(positive_count, negative_count)
        larger = max(positive_count, negative_count)
        if smaller >= 5 and smaller / max(1, larger) >= 0.55:
            return "phân cực rõ"
        if negative_count >= positive_count * 1.45:
            return "nghiêng tiêu cực và cần ưu tiên xử lý"
        if positive_count >= negative_count * 1.8:
            return "nghiêng tích cực nhưng vẫn có rủi ro chất lượng"
        return "trái chiều"
    if negative_count > 0:
        return "là vấn đề cần ưu tiên cải thiện"
    if positive_count > 0:
        return "là điểm mạnh tương đối ổn định"
    return "chủ yếu là ghi nhận trung lập"


def action_clause_from_negative(negative: str) -> str:
    negative = lowercase_first(negative).rstrip(".")
    if not negative:
        return ""
    replacements = [
        ("khách hàng phàn nàn rằng ", ""),
        ("khách hàng phàn nàn về ", ""),
        ("khách hàng chỉ trích ", ""),
        ("khách phàn nàn rằng ", ""),
        ("khách phàn nàn về ", ""),
        ("khách chê ", ""),
        ("các phàn nàn được ghi nhận gồm ", ""),
        ("phản hồi tiêu cực chủ yếu liên quan đến ", ""),
        ("phản hồi tiêu cực nổi bật ở ", ""),
        ("phản hồi tiêu cực tập trung vào ", ""),
    ]
    for prefix, replacement in replacements:
        if negative.startswith(prefix):
            negative = replacement + negative[len(prefix):]
            break
    return negative.strip(" ,.;")


def retain_clause_from_positive(positive: str) -> str:
    positive = lowercase_first(positive).rstrip(".")
    if not positive:
        return ""
    replacements = [
        ("khách hàng đánh giá rất cao ", ""),
        ("khách hàng đánh giá cao ", ""),
        ("khách hàng ca ngợi ", ""),
        ("khách hàng khen ngợi ", ""),
        ("khách đánh giá cao ", ""),
        ("khách ca ngợi ", ""),
        ("khách khen ngợi ", ""),
        ("khách khen ", ""),
        ("các lời khen được ghi nhận gồm ", ""),
        ("phản hồi tích cực nổi bật ở ", ""),
    ]
    for prefix, replacement in replacements:
        if positive.startswith(prefix):
            positive = replacement + positive[len(prefix):]
            break
    return positive.strip(" ,.;")


def build_overall_aspect_summary(row: dict[str, Any], summary_language: str = "vi") -> str:
    aspect = str(row.get("aspect", ""))
    aspect_code = ASPECT_SUMMARY_CODES.get(aspect, aspect.upper()[:3] or "ASP")
    aspect_label = ASPECT_LABELS_VI.get(aspect, aspect)
    positive = strip_business_summary_lead(str(row.get("positive_summary", "")), aspect_label)
    negative = strip_business_summary_lead(str(row.get("negative_summary", "")), aspect_label)
    neutral = strip_business_summary_lead(str(row.get("neutral_summary", "")), aspect_label)
    positive_count = int(row.get("positive_count", 0) or 0)
    negative_count = int(row.get("negative_count", 0) or 0)
    neutral_count = int(row.get("neutral_count", 0) or 0)
    if summary_language == "en":
        fallback_en = "There is no clear signal strong enough for a separate conclusion."
        positive_line = lowercase_first(positive).rstrip(".") or fallback_en
        negative_line = lowercase_first(negative).rstrip(".") or fallback_en
        neutral_line = lowercase_first(neutral).rstrip(".") or fallback_en
        positive_line = (
            fallback_en
            if has_vietnamese_summary_leak(positive_line)
            or has_raw_cluster_dump(positive_line)
            or has_incomplete_english_summary_line(positive_line)
            else positive_line
        )
        negative_line = (
            fallback_en
            if has_vietnamese_summary_leak(negative_line)
            or has_raw_cluster_dump(negative_line)
            or has_incomplete_english_summary_line(negative_line)
            else negative_line
        )
        neutral_line = (
            fallback_en
            if has_vietnamese_summary_leak(neutral_line)
            or has_raw_cluster_dump(neutral_line)
            or has_incomplete_english_summary_line(neutral_line)
            else neutral_line
        )
        return "\n".join(
            [
                f"{aspect_code}: {positive_count:,} positive sentences, {negative_count:,} negative sentences, {neutral_count:,} neutral sentences",
                f"Pos: {finish_sentence(positive_line)}",
                f"Neg: {finish_sentence(negative_line)}",
                f"Neu: {finish_sentence(neutral_line)}",
            ]
        )
    fallback = "Không có tín hiệu nổi bật đủ rõ để rút ra nhận định riêng."
    positive_line = retain_clause_from_positive(positive) or lowercase_first(positive).rstrip(".") or fallback
    negative_line = action_clause_from_negative(negative) or lowercase_first(negative).rstrip(".") or fallback
    neutral_line = lowercase_first(neutral).rstrip(".") or fallback
    return "\n".join(
        [
            f"{aspect_code}: {positive_count:,} câu tích cực, {negative_count:,} câu tiêu cực, {neutral_count:,} câu trung lập",
            f"Pos: {finish_sentence(positive_line)}",
            f"Neg: {finish_sentence(negative_line)}",
            f"Neu: {finish_sentence(neutral_line)}",
        ]
    )


def truncate_summary_fragment(text: str, max_chars: int = 220) -> str:
    text = clean_text(text).rstrip(".")
    if len(text) <= max_chars:
        return text
    clipped = text[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;")
    return f"{clipped}..."


def top_aspect_summary_fragments(
    detail_rows: list[dict[str, Any]],
    sentiment: str,
    *,
    limit: int = 3,
) -> list[str]:
    entries: list[tuple[int, str, str]] = []
    for row in detail_rows:
        aspect = str(row.get("aspect", ""))
        count = int(row.get(f"{sentiment}_count", 0) or 0)
        summary = clean_text(row.get(f"{sentiment}_summary", ""))
        if aspect not in ASPECT_NAMES or count <= 0 or not summary:
            continue
        aspect_label = ASPECT_LABELS_VI.get(aspect, aspect)
        fragment = strip_business_summary_lead(summary, aspect_label)
        if sentiment == "positive":
            fragment = retain_clause_from_positive(fragment) or fragment
        elif sentiment == "negative":
            fragment = action_clause_from_negative(fragment) or fragment
        else:
            fragment = lowercase_first(fragment)
        entries.append((count, aspect, fragment))
    entries.sort(key=lambda item: item[0], reverse=True)
    return [
        f"{ASPECT_LABELS_VI.get(aspect, aspect)}: {truncate_summary_fragment(fragment)}"
        for count, aspect, fragment in entries[:limit]
    ]


def build_all_aspects_sentiment_summary(detail_rows: list[dict[str, Any]], sentiment: str) -> str:
    fragments = top_aspect_summary_fragments(detail_rows, sentiment)
    if not fragments:
        return ""
    prefix = {
        "positive": "Điểm được khen rõ nhất nằm ở",
        "negative": "Điểm cần xử lý rõ nhất nằm ở",
        "neutral": "Bối cảnh trung lập đáng chú ý nằm ở",
    }.get(sentiment, "Tín hiệu chính nằm ở")
    return f"{prefix}: {'; '.join(fragments)}."


def normalize_one_sentence_summary(text: str, max_chars: int = 0) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\s*[.!?。！？]+\s+", ", ", cleaned)
    cleaned = re.sub(r"\s*[;!?]+\s*", ", ", cleaned)
    cleaned = re.sub(r"\b(với|gồm|bao gồm)\s+\d+\s+(tích cực|tiêu cực|trung lập)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(có|gồm)\s+\d+\s+(câu|ý kiến|phản hồi)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.")
    if not cleaned:
        return ""
    if max_chars and max_chars > 0 and len(cleaned) > max_chars:
        clipped = cleaned[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,")
        cleaned = clipped or cleaned[:max_chars]
    return cleaned.rstrip(".") + "."


def build_all_aspects_overall_summary(detail_rows: list[dict[str, Any]], summary_language: str = "vi") -> str:
    counts = {
        sentiment: sum(int(row.get(f"{sentiment}_count", 0) or 0) for row in detail_rows)
        for sentiment in SENTIMENTS
    }
    total_count = sum(counts.values())
    if total_count <= 0:
        return ""
    dominant = max(SENTIMENTS, key=lambda sentiment: counts[sentiment])
    dominant_label = {
        "positive": "nghiêng tích cực",
        "negative": "nghiêng tiêu cực",
        "neutral": "khá trung tính",
    }[dominant]
    if summary_language == "en":
        dominant_label_en = {
            "positive": "leans positive",
            "negative": "leans negative",
            "neutral": "is mostly neutral",
        }[dominant]
        sorted_rows = sorted(
            (row for row in detail_rows if str(row.get("aspect", "")) in ASPECT_NAMES),
            key=lambda row: sum(int(row.get(f"{sentiment}_count", 0) or 0) for sentiment in SENTIMENTS),
            reverse=True,
        )
        parts = [f"The overall guest picture {dominant_label_en}."]
        for row in sorted_rows[:3]:
            aspect = str(row.get("aspect", ""))
            summary = clean_overall_summary_text(row.get("overall_aspect_summary", ""), 260)
            if summary:
                parts.append(f"{ASPECT_LABELS_EN.get(aspect, aspect).title()}: {summary}")
        return normalize_one_sentence_summary(" ".join(parts), 1800)
    positive = "; ".join(top_aspect_summary_fragments(detail_rows, "positive", limit=2))
    negative = "; ".join(top_aspect_summary_fragments(detail_rows, "negative", limit=2))
    neutral = "; ".join(top_aspect_summary_fragments(detail_rows, "neutral", limit=1))
    opening = f"Bức tranh chung {dominant_label}."
    parts = [opening]
    if positive:
        parts.append(f"Điểm mạnh nằm ở {positive}.")
    if negative:
        parts.append(f"Điểm cần xử lý nằm ở {negative}.")
    if neutral:
        parts.append(f"Bối cảnh trung lập nằm ở {neutral}.")
    return normalize_insight_paragraph(" ".join(parts))


def parse_all_aspects_summary_items(content: str, items: list[dict[str, Any]]) -> list[str]:
    try:
        payload = extract_json_payload(content)
    except json.JSONDecodeError:
        return [
            build_all_aspects_overall_summary(
                item.get("detail_rows", []),
                str(item.get("summary_language", "vi")),
            )
            for item in items
        ]

    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, str] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(items):
                by_id[idx] = clean_overall_summary_text(item.get("summary", ""), 1800)
    return [
        by_id.get(idx)
        or build_all_aspects_overall_summary(
            item.get("detail_rows", []),
            str(item.get("summary_language", "vi")),
        )
        for idx, item in enumerate(items)
    ]


class QwenAllAspectsOverallWriter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.client = OpenAI(
            base_url=args.qwen_base_url,
            api_key=args.qwen_api_key,
            timeout=args.timeout_sec,
        )

    def summarize_batch(self, items: list[dict[str, Any]]) -> list[str]:
        if self.args.skip_qwen:
            return [
                build_all_aspects_overall_summary(
                    item.get("detail_rows", []),
                    getattr(self.args, "summary_language", "vi"),
                )
                for item in items
            ]

        payload_items: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            detail_rows = item.get("detail_rows", [])
            aspects_payload = []
            for row in detail_rows:
                aspect = str(row.get("aspect", ""))
                if aspect not in ASPECT_NAMES:
                    continue
                aspects_payload.append(
                    {
                        "aspect": aspect,
                        "overall_aspect_summary": clean_text(row.get("overall_aspect_summary", "")),
                        "positive_count": int(row.get("positive_count", 0) or 0),
                        "negative_count": int(row.get("negative_count", 0) or 0),
                        "neutral_count": int(row.get("neutral_count", 0) or 0),
                    }
                )
            payload_items.append({"id": idx, "hotel_id": item.get("hotel_id", ""), "aspects": aspects_payload})

        payload = {"items": payload_items}
        output_language = "English" if getattr(self.args, "summary_language", "vi") == "en" else "Vietnamese"
        system = (
            f"You rewrite all hotel aspects into one {output_language} overall insight paragraph. Return strict JSON only. "
            "Each input already contains per-aspect summaries. Synthesize them into two to four concise sentences, "
            "not a list and not one mini-sentence per aspect. "
            "Start directly with the overall reading, then identify the strongest positive driver, the clearest weakness, and the practical priority. "
            "Do not mention number of reviews, number of sentences, or any counting phrases. "
            "Use counts only as hidden ranking context, never as the main wording. "
            "Do not use bullets or patterns like 'Tổng hợp phản hồi', 'gồm 3 ý', 'có N phản hồi', 'Trong các phản hồi', "
            "'overall feedback mentions', or 'there are N reviews'. "
            "Do not include aspect codes, XML tags, cluster IDs, or raw metadata. "
            "No fabricated details."
        )
        user = (
            "Return exactly this compact schema:\n"
            '{"items":[{"id":0,"summary":"..."}]}\n\n'
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        last_error = ""
        for attempt in range(self.args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not self.args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = self.client.chat.completions.create(
                    model=self.args.qwen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=self.args.final_summary_max_output_tokens,
                    extra_body=extra_body,
                )
                content = (rsp.choices[0].message.content or "").strip()
                if not content:
                    finish = getattr(rsp.choices[0], "finish_reason", "")
                    raise ValueError(f"empty Qwen all-aspects summary content finish_reason={finish}")
                return parse_all_aspects_summary_items(content, items)
            except Exception as exc:  # noqa: BLE001 - endpoint failures need fallback.
                last_error = str(exc)
                if attempt < self.args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        print(
            f"[WARN] Qwen all-aspects summary failed; using local summary fallback. Error: {last_error}",
            file=sys.stderr,
        )
        return [build_all_aspects_overall_summary(item.get("detail_rows", [])) for item in items]


def parse_final_summary_items(content: str, items: list[dict[str, Any]]) -> list[str]:
    try:
        payload = extract_json_payload(content)
    except json.JSONDecodeError:
        repaired = repair_final_summary_items(content, items)
        if repaired:
            return repaired
        raise
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, str] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(items):
                by_id[idx] = strip_formulaic_aspect_intro(item.get("summary", ""))[:1200]
    return [
        by_id.get(idx) or business_final_sentiment_summary(item)
        for idx, item in enumerate(items)
    ]


def parse_qwen_summary_items(content: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    try:
        payload = extract_json_payload(content)
    except json.JSONDecodeError:
        repaired = repair_aspect_summary_items(content, items)
        if repaired:
            return repaired
        raise
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, dict[str, str]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(items):
                by_id[idx] = {
                    "summary_vi": clean_summary_text(item.get("summary_vi", ""), 1600),
                    "summary_en": clean_summary_text(item.get("summary_en", ""), 1600),
                }
    out = []
    for idx, item in enumerate(items):
        summary = by_id.get(idx)
        if summary and (summary.get("summary_vi") or summary.get("summary_en")):
            out.append(summary)
        else:
            out.append(local_aspect_summary(item))
    return out


def repair_final_summary_items(content: str, items: list[dict[str, Any]]) -> list[str]:
    item_pattern = re.compile(
        r'\{\s*"id"\s*:\s*(\d+).*?"summary"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.DOTALL,
    )
    by_id: dict[int, str] = {}
    for match in item_pattern.finditer(content):
        idx = int(match.group(1))
        if not (0 <= idx < len(items)):
            continue
        try:
            summary_text = json.loads(f'"{match.group(2)}"')
        except json.JSONDecodeError:
            summary_text = match.group(2)
        by_id[idx] = strip_formulaic_aspect_intro(summary_text)[:1200]
    if not by_id:
        return []
    return [by_id.get(idx) or business_final_sentiment_summary(item) for idx, item in enumerate(items)]


def repair_aspect_summary_items(content: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    item_pattern = re.compile(
        r'\{\s*"id"\s*:\s*(\d+)'
        r'(?:.*?"summary_vi"\s*:\s*"((?:[^"\\]|\\.)*)")?'
        r'(?:.*?"summary_en"\s*:\s*"((?:[^"\\]|\\.)*)")?',
        re.DOTALL,
    )
    by_id: dict[int, dict[str, str]] = {}
    for match in item_pattern.finditer(content):
        idx = int(match.group(1))
        if not (0 <= idx < len(items)):
            continue
        try:
            summary_vi = json.loads(f'"{match.group(2)}"') if match.group(2) else ""
        except json.JSONDecodeError:
            summary_vi = match.group(2) or ""
        try:
            summary_en = json.loads(f'"{match.group(3)}"') if match.group(3) else ""
        except json.JSONDecodeError:
            summary_en = match.group(3) or ""
        by_id[idx] = {
            "summary_vi": clean_summary_text(summary_vi, 1600),
            "summary_en": clean_summary_text(summary_en, 1600),
        }
    if not by_id:
        return []
    out: list[dict[str, str]] = []
    for idx, item in enumerate(items):
        summary = by_id.get(idx)
        if summary and (summary.get("summary_vi") or summary.get("summary_en")):
            out.append(summary)
        else:
            out.append(local_aspect_summary(item))
    return out


def parse_qwen_items(content: str, sentences: list[str]) -> list[dict[str, Any]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(sentences):
                by_id[idx] = {
                    "aspect": normalize_aspect(item.get("aspect"), sentences[idx]),
                    "sentiment": normalize_sentiment(item.get("sentiment")),
                    "confidence": clamp_confidence(item.get("confidence", 0.65)),
                    "reason_short": str(item.get("reason_short", ""))[:200],
                }
    out = []
    for idx, sentence in enumerate(sentences):
        out.append(by_id.get(idx, fallback_classify(sentence)))
    return out


def repair_common_json_errors(text: str) -> str:
    """Repair frequent Qwen JSON glitches without changing valid JSON."""
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    key_names = (
        "items",
        "id",
        "aspect",
        "sentiment",
        "unit_ids",
        "cluster_code",
        "cluster_label",
        "descriptors",
        "confidence",
        "source",
        "segment",
        "segments",
        "units",
        "text",
        "source_text",
        "normalized_text_vi",
        "normalized_text_en",
        "summary_vi",
        "summary_en",
    )
    key_re = "|".join(re.escape(key) for key in key_names)
    repaired = re.sub(rf'(?<=[}}\]"0-9])\s+(?="(?:{key_re})"\s*:)', ",", repaired)
    repaired = re.sub(r"}\s+{", "},{", repaired)
    return repaired


def json_loads_lenient(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_common_json_errors(text)
        if repaired != text:
            return json.loads(repaired)
        raise


def extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json_loads_lenient(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json_loads_lenient(text[start : end + 1])
        raise


def fallback_aspect(text: str, allow_branding: bool | None = None) -> str:
    lower = normalize_for_cache(text)
    if allow_branding is None:
        allow_branding = has_branding_evidence(lower)
    scores = {aspect: 0 for aspect in ASPECT_NAMES}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if aspect == "branding" and not allow_branding:
            continue
        if aspect == "loyalty" and not LOYALTY_INTENT_RE.search(lower):
            continue
        for keyword in keywords:
            if contains_keyword(lower, keyword):
                scores[aspect] += 1
    if LOYALTY_INTENT_RE.search(lower):
        scores["loyalty"] += 6
    if PHYSICAL_SECURITY_RE.search(lower):
        scores["facility"] += 5
        scores["amenity"] = max(0, scores["amenity"] - 2)
        scores["experience"] = max(0, scores["experience"] - 1)
    elif SUBJECTIVE_SAFETY_RE.search(lower) and not (
        FOOD_CONTEXT_RE.search(lower)
        or AMENITY_CONTEXT_RE.search(lower)
        or TRANSPORT_AMENITY_RE.search(lower)
    ):
        scores["experience"] += 5
    if VALUE_CONTEXT_RE.search(lower):
        scores["experience"] += 5
        scores["branding"] = max(0, scores["branding"] - 1)
    if INTERIOR_DESIGN_RE.search(lower):
        scores["facility"] += 4
        scores["experience"] = max(0, scores["experience"] - 1)
    if FACILITY_LOCATION_VIEW_RE.search(lower):
        scores["facility"] += 5
        scores["amenity"] = max(0, scores["amenity"] - 2)
        scores["experience"] = max(0, scores["experience"] - 1)
    if FOOD_CONTEXT_RE.search(lower):
        if SERVICE_HUMAN_RE.search(lower) and re.search(
            r"\b(waiter|waitress|restaurant service|served|serve|service|slow|rude|attentive|"
            r"phuc vu|phục vụ|boi ban|bồi bàn|nhan vien|nhân viên|le tan|lễ tân)\b",
            lower,
            re.IGNORECASE,
        ):
            scores["service"] += 4
        else:
            scores["amenity"] += 5
            scores["service"] = max(0, scores["service"] - 2)
    if TRANSPORT_AMENITY_RE.search(lower):
        scores["amenity"] += 5
    if AMENITY_CONTEXT_RE.search(lower):
        scores["amenity"] += 4
    if SERVICE_HUMAN_RE.search(lower) or SERVICE_ACTION_RE.search(lower):
        scores["service"] += 4
    if VIEW_NOISE_CONTEXT_RE.search(lower):
        if FACILITY_PHYSICAL_CONTEXT_RE.search(lower) or FACILITY_LOCATION_VIEW_RE.search(lower):
            scores["facility"] += 3
        elif EXPERIENCE_FEELING_RE.search(lower):
            scores["experience"] += 3
    if STYLE_CONTEXT_RE.search(lower) and not has_branding_evidence(lower):
        scores["facility"] = max(0, scores["facility"] - 2)
        scores["branding"] = 0
        scores["experience"] += 1
    best_aspect, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_aspect if best_score > 0 else "experience"


def fallback_sentiment(text: str) -> tuple[str, float]:
    lower = normalize_for_cache(text)
    pos = len(POSITIVE_RE.findall(lower))
    neg = len(NEGATIVE_RE.findall(lower))
    if pos > neg:
        return "positive", min(0.55 + 0.12 * pos, 0.86)
    if neg > pos:
        return "negative", min(0.55 + 0.12 * neg, 0.86)
    return "neutral", 0.45


def fallback_classify(text: str) -> dict[str, Any]:
    sentiment, confidence = fallback_sentiment(text)
    return {
        "aspect": fallback_aspect(text),
        "sentiment": sentiment,
        "confidence": confidence,
        "reason_short": "rule-based fallback",
    }


def aspect_score_for_text(aspect: str, text: str) -> float:
    lower = normalize_for_cache(text)
    keywords = ASPECT_KEYWORDS.get(aspect, ())
    hits = sum(1 for keyword in keywords if contains_keyword(lower, keyword))
    return min(1.0, 0.35 + 0.18 * hits)


def aspect_total(agg: AspectAggregate) -> int:
    return int(sum(agg.counts.values()))


def aspect_has_sentiment_tie(agg: AspectAggregate) -> bool:
    total = aspect_total(agg)
    if total <= 0:
        return False
    max_count = max(agg.counts.get(sent, 0) for sent in SENTIMENTS)
    return sum(1 for sent in SENTIMENTS if agg.counts.get(sent, 0) == max_count) > 1


def build_profile_item(entity_id: str, hotel: HotelAggregate, args: argparse.Namespace) -> dict[str, Any]:
    aspects: dict[str, Any] = {}
    for aspect in ASPECT_NAMES:
        agg = hotel.aspects[aspect]
        candidates = []
        for sentiment in SENTIMENTS:
            choice = agg.representative.get(sentiment, SentenceChoice())
            if choice.text:
                candidates.append(
                    {
                        "sentiment": sentiment,
                        "text": choice.text[: args.profile_candidate_chars],
                        "score": round(float(choice.score), 4),
                    }
                )
        aspects[aspect] = {
            "count": aspect_total(agg),
            "positive": int(agg.counts.get("positive", 0)),
            "negative": int(agg.counts.get("negative", 0)),
            "neutral": int(agg.counts.get("neutral", 0)),
            "local_sentiment": agg.final_sentiment() if aspect_total(agg) else "",
            "sentiment_tie": aspect_has_sentiment_tie(agg),
            "candidates": candidates,
        }
    return {
        "entity_id": entity_id,
        "data_source": hotel.data_source,
        "hotel_id": hotel.hotel_id,
        "review_count": len(hotel.review_indexes),
        "sentence_count": hotel.sentence_count,
        "aspects": aspects,
    }


def compact_profile_for_qwen(profile: dict[str, Any]) -> dict[str, Any]:
    aspects: dict[str, Any] = {}
    for aspect in ASPECT_NAMES:
        info = profile["aspects"].get(aspect, {})
        if not info.get("sentiment_tie") or int(info.get("count", 0)) <= 0:
            continue
        aspects[aspect] = {
            "count": info.get("count", 0),
            "positive": info.get("positive", 0),
            "negative": info.get("negative", 0),
            "neutral": info.get("neutral", 0),
            "local_sentiment": info.get("local_sentiment", ""),
            "candidates": info.get("candidates", []),
        }
    return {
        "entity_id": profile["entity_id"],
        "data_source": profile["data_source"],
        "hotel_id": profile["hotel_id"],
        "aspects": aspects,
    }


def profile_hash(profile: dict[str, Any]) -> str:
    payload = json.dumps(profile, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def local_profile_decision(profile: dict[str, Any]) -> dict[str, Any]:
    out = {"entity_id": profile["entity_id"], "aspects": {}}
    for aspect in ASPECT_NAMES:
        info = profile["aspects"].get(aspect, {})
        local_sentiment = normalize_sentiment(info.get("local_sentiment", ""))
        representative = ""
        for candidate in info.get("candidates", []):
            if normalize_sentiment(candidate.get("sentiment", "")) == local_sentiment:
                representative = str(candidate.get("text", ""))
                break
        if not representative and info.get("candidates"):
            representative = str(info["candidates"][0].get("text", ""))
        out["aspects"][aspect] = {
            "sentiment": local_sentiment if int(info.get("count", 0)) else "",
            "representative_sentence": representative,
            "note": "local",
        }
    return out


def parse_profile_items(content: str, profiles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payload = extract_json_payload(content)
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id"))
            except Exception:
                continue
            if 0 <= idx < len(profiles):
                decision = local_profile_decision(profiles[idx])
                raw_aspects = item.get("aspects", {})
                if not isinstance(raw_aspects, dict):
                    raw_aspects = {}
                for aspect in ASPECT_NAMES:
                    if aspect not in raw_aspects:
                        continue
                    raw_aspect = raw_aspects.get(aspect, {})
                    if not isinstance(raw_aspect, dict):
                        raw_aspect = {}
                    decision["aspects"][aspect] = {
                        "sentiment": normalize_sentiment(raw_aspect.get("sentiment", "")),
                        "representative_sentence": clean_text(raw_aspect.get("representative_sentence", "")),
                        "note": str(raw_aspect.get("note", ""))[:200],
                    }
                by_id[idx] = decision
    out: dict[str, dict[str, Any]] = {}
    for idx, profile in enumerate(profiles):
        out[profile["entity_id"]] = by_id.get(idx, local_profile_decision(profile))
    return out


def deserialize_profile_decision(raw: dict[str, Any]) -> dict[str, ProfileAspectDecision]:
    aspects: dict[str, ProfileAspectDecision] = {}
    raw_aspects = raw.get("aspects", {}) if isinstance(raw, dict) else {}
    if not isinstance(raw_aspects, dict):
        raw_aspects = {}
    for aspect in ASPECT_NAMES:
        item = raw_aspects.get(aspect, {})
        if not isinstance(item, dict):
            item = {}
        aspects[aspect] = ProfileAspectDecision(
            sentiment=normalize_sentiment(item.get("sentiment", "")) if item.get("sentiment", "") else "",
            representative_sentence=clean_text(item.get("representative_sentence", "")),
            note=str(item.get("note", ""))[:200],
        )
    return aspects


def classify_pending(
    pending: list[tuple[str, str]],
    cache: ClassificationCache,
    classifier: QwenClassifier,
) -> dict[str, dict[str, Any]]:
    """Classify uncached sentences.

    pending contains (hash, original sentence text). Duplicate hashes should already
    be removed by the caller.
    """
    if not pending:
        return {}
    out: dict[str, dict[str, Any]] = {}
    batch_size = max(1, int(classifier.args.batch_size))
    total_batches = (len(pending) + batch_size - 1) // batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_num = (start // batch_size) + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
            print(
                f"[qwen] classifying batch {batch_num}/{total_batches} "
                f"sentences={len(batch)}",
                flush=True,
            )
        texts = [item[1] for item in batch]
        items = classifier.classify_batch(texts)
        rows_to_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        for (hash_key, text), item in zip(batch, items):
            normalized = normalize_for_cache(text)
            item = {
                "aspect": normalize_aspect(item.get("aspect"), text),
                "sentiment": normalize_sentiment(item.get("sentiment")),
                "confidence": clamp_confidence(item.get("confidence", 0.45)),
                "reason_short": str(item.get("reason_short", ""))[:200],
            }
            out[hash_key] = item
            rows_to_cache[hash_key] = (normalized, item)
        cache.set_many(rows_to_cache)
    return out


def process_file(
    path: Path,
    args: argparse.Namespace,
    cache: ClassificationCache,
    classifier: QwenClassifier,
    hotels: dict[str, HotelAggregate],
    debug_stats: dict[str, Any],
) -> None:
    configured_skip_rows = max(0, int(getattr(args, "skip_rows_per_source", 0)))
    source_rows = configured_skip_rows
    bad_ref_ids = 0
    empty_reviews = 0
    skipped_min_reviews = 0
    eligible_entity_ids = getattr(args, "eligible_entity_ids", None)
    seen_hotels_for_source: defaultdict[str, set[str]] = defaultdict(set)
    resume_rows_remaining = configured_skip_rows
    if resume_rows_remaining > 0:
        print(f"[skip] {path.name}: skipping {resume_rows_remaining:,} initial rows", flush=True)

    for chunk in pd.read_csv(path, dtype=str, chunksize=args.chunk_size, usecols=["ref_id", "review"]):
        if resume_rows_remaining > 0:
            if resume_rows_remaining >= len(chunk):
                resume_rows_remaining -= len(chunk)
                continue
            chunk = chunk.iloc[resume_rows_remaining:].copy()
            resume_rows_remaining = 0
        if args.max_rows_per_source and source_rows >= args.max_rows_per_source:
            break
        if args.max_rows_per_source:
            chunk = chunk.head(max(0, args.max_rows_per_source - source_rows))

        sentence_records: list[dict[str, Any]] = []
        for _, row in chunk.iterrows():
            source_rows += 1
            parsed = parse_ref_id(str(row.get("ref_id", "")))
            if parsed is None:
                bad_ref_ids += 1
                continue
            data_source, hotel_id, review_index = parsed
            entity_id = make_entity_id(data_source, hotel_id)
            if eligible_entity_ids is not None and entity_id not in eligible_entity_ids:
                skipped_min_reviews += 1
                continue
            if args.max_hotels_per_source:
                source_hotels = seen_hotels_for_source[data_source]
                if hotel_id not in source_hotels and len(source_hotels) >= args.max_hotels_per_source:
                    continue
                source_hotels.add(hotel_id)

            review_text = clean_text(row.get("review", ""))
            if not review_text:
                empty_reviews += 1
                continue
            sentences = split_sentences(review_text, args.min_words)
            if not sentences:
                empty_reviews += 1
                continue
            for local_idx, sentence in enumerate(sentences):
                normalized = normalize_for_cache(sentence)
                hash_key = sentence_hash(normalized)
                sentence_records.append(
                    {
                        "hash": hash_key,
                        "normalized": normalized,
                        "entity_id": entity_id,
                        "data_source": data_source,
                        "hotel_id": hotel_id,
                        "review_index": review_index,
                        "sentence_id": f"{data_source}_{hotel_id}_{review_index}_{local_idx}",
                        "text": sentence,
                    }
                )

        if not sentence_records:
            continue

        unique_hashes = list(dict.fromkeys(record["hash"] for record in sentence_records))
        cached = cache.get_many(unique_hashes)
        pending_map: dict[str, str] = {}
        for record in sentence_records:
            if record["hash"] not in cached and record["hash"] not in pending_map:
                pending_map[record["hash"]] = record["text"]
        classified = {
            **cached,
            **classify_pending(list(pending_map.items()), cache, classifier),
        }

        for record in sentence_records:
            item = classified.get(record["hash"], fallback_classify(record["text"]))
            aspect = normalize_aspect(item.get("aspect"), record["text"])
            sentiment = normalize_sentiment(item.get("sentiment"))
            confidence = clamp_confidence(item.get("confidence", 0.45))
            aggregate = hotels.get(record["entity_id"])
            if aggregate is None:
                aggregate = HotelAggregate(record["data_source"], record["hotel_id"])
                hotels[record["entity_id"]] = aggregate
            aggregate.add_sentence(
                review_index=record["review_index"],
                sentence_id=record["sentence_id"],
                text=record["text"],
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                aspect_score=aspect_score_for_text(aspect, record["text"]),
                keep_debug_samples=args.debug_samples_per_aspect,
                keep_reference_text=not args.disable_bertscore,
                keep_reference_stats=not args.disable_summary_reference_stats,
            )

        if args.progress_every > 0 and source_rows % args.progress_every < len(chunk):
            print(
                f"[progress] {path.name}: rows={source_rows:,} hotels={len(hotels):,} "
                f"cache_hits={len(cached):,} new_sentences={len(pending_map):,}",
                flush=True,
            )

    debug_stats[path.name] = {
        "rows_seen": source_rows,
        "bad_ref_ids": bad_ref_ids,
        "empty_reviews": empty_reviews,
        "skipped_min_reviews": skipped_min_reviews,
    }


def extract_pending_semantic_units(
    pending: list[tuple[str, str]],
    cache: ClassificationCache,
    segmenter: QwenSemanticPreSegmenter,
) -> dict[str, list[dict[str, Any]]]:
    if not pending:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    batch_size = max(1, int(segmenter.args.batch_size))
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    total_batches = len(batches)
    workers = max(1, int(segmenter.args.qwen_extract_workers))

    def run_batch(batch_idx: int, batch: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]], list[list[dict[str, Any]]]]:
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"[qwen-preseg] batch {batch_idx}/{total_batches} reviews={len(batch)}",
                flush=True,
            )
        texts = [item[1] for item in batch]
        local_segmenter = segmenter if workers == 1 else QwenSemanticPreSegmenter(segmenter.args)
        return batch_idx, batch, local_segmenter.segment_batch(texts)

    completed: list[tuple[int, list[tuple[str, str]], list[list[dict[str, Any]]]]] = []
    if workers == 1 or total_batches <= 1:
        for batch_idx, batch in enumerate(batches, start=1):
            completed.append(run_batch(batch_idx, batch))
    else:
        print(
            f"[qwen-preseg] running {total_batches:,} batches with {workers:,} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_batch, batch_idx, batch)
                for batch_idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                completed.append(future.result())

    for _, batch, segmented_items in sorted(completed, key=lambda item: item[0]):
        rows_to_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        for (hash_key, text), units in zip(batch, segmented_items):
            normalized = normalize_for_cache(text)
            raw_unit_dicts = [unit for unit in units if isinstance(unit, dict)]
            cleaned_units = clean_semantic_units(raw_unit_dicts, text, segmenter.args)
            out[hash_key] = cleaned_units
            rows_to_cache[hash_key] = (normalized, cleaned_units)
        cache.set_semantic_segments(rows_to_cache)
    return out


def extract_pending_aspects(
    pending: list[tuple[str, str]],
    cache: ClassificationCache,
    extractor: QwenAspectSegmenter,
) -> dict[str, list[dict[str, Any]]]:
    if not pending:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    batch_size = max(1, int(extractor.args.batch_size))
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    total_batches = len(batches)
    workers = max(1, int(extractor.args.qwen_extract_workers))

    def run_batch(batch_idx: int, batch: list[tuple[str, str]]) -> tuple[int, list[tuple[str, str]], list[list[dict[str, Any]]]]:
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"[qwen-extract] batch {batch_idx}/{total_batches} sentences={len(batch)}",
                flush=True,
            )
        texts = [item[1] for item in batch]
        local_extractor = extractor if workers == 1 else QwenAspectSegmenter(extractor.args)
        return batch_idx, batch, local_extractor.segment_batch(texts)

    completed: list[tuple[int, list[tuple[str, str]], list[list[dict[str, Any]]]]] = []
    if workers == 1 or total_batches <= 1:
        for batch_idx, batch in enumerate(batches, start=1):
            completed.append(run_batch(batch_idx, batch))
    else:
        print(
            f"[qwen-extract] running {total_batches:,} batches with {workers:,} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_batch, batch_idx, batch)
                for batch_idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                completed.append(future.result())

    for _, batch, extracted_items in sorted(completed, key=lambda item: item[0]):
        rows_to_cache: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        for (hash_key, text), segments in zip(batch, extracted_items):
            normalized = normalize_for_cache(text)
            cleaned_segments = parse_qwen_aspect_segments_lenient(
                json.dumps({"items": [{"id": 0, "segments": segments}]}, ensure_ascii=False),
                [text],
            )[0]
            if not cleaned_segments and has_explicit_hotel_signal(text):
                cleaned_segments = fallback_aspect_segments(text)
            out[hash_key] = cleaned_segments
            rows_to_cache[hash_key] = (normalized, cleaned_segments)
        cache.set_aspect_extractions(rows_to_cache)
    return out


def classify_pending_segment_sentiments(
    pending: list[tuple[str, str, str]],
    cache: ClassificationCache,
    classifier: QwenAspectSentimentClassifier,
) -> dict[str, dict[str, Any]]:
    if not pending:
        return {}
    out: dict[str, dict[str, Any]] = {}
    batch_size = max(1, int(classifier.args.batch_size))
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    total_batches = len(batches)
    workers = max(1, int(classifier.args.qwen_sentiment_workers))

    def run_batch(
        batch_idx: int,
        batch: list[tuple[str, str, str]],
    ) -> tuple[int, list[tuple[str, str, str]], list[dict[str, Any]]]:
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"[qwen-sentiment] batch {batch_idx}/{total_batches} segments={len(batch)}",
                flush=True,
            )
        payload = [{"aspect": aspect, "text": text} for _, aspect, text in batch]
        local_classifier = classifier if workers == 1 else QwenAspectSentimentClassifier(classifier.args)
        return batch_idx, batch, local_classifier.classify_batch(payload)

    completed: list[tuple[int, list[tuple[str, str, str]], list[dict[str, Any]]]] = []
    if workers == 1 or total_batches <= 1:
        for batch_idx, batch in enumerate(batches, start=1):
            completed.append(run_batch(batch_idx, batch))
    else:
        print(
            f"[qwen-sentiment] running {total_batches:,} batches with {workers:,} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_batch, batch_idx, batch)
                for batch_idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                completed.append(future.result())

    for _, batch, items in sorted(completed, key=lambda item: item[0]):
        rows_to_cache: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for (hash_key, aspect, text), item in zip(batch, items):
            cleaned = {
                "aspect": normalize_aspect(item.get("aspect", aspect), text),
                "sentiment": normalize_sentiment(item.get("sentiment", "")),
                "confidence": clamp_confidence(item.get("confidence", 0.45)),
            }
            out[hash_key] = cleaned
            rows_to_cache[hash_key] = (aspect, text, cleaned)
        cache.set_segment_sentiments(rows_to_cache)
    return out


def assign_pending_clusters(
    pending: list[tuple[str, str, str, str]],
    cache: ClassificationCache,
    assigner: QwenClusterAssigner,
) -> dict[str, dict[str, Any]]:
    if not pending:
        return {}
    out: dict[str, dict[str, Any]] = {}
    batch_size = max(1, int(assigner.args.batch_size))
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    total_batches = len(batches)
    workers = max(1, int(assigner.args.qwen_cluster_workers))

    def run_batch(
        batch_idx: int,
        batch: list[tuple[str, str, str, str]],
    ) -> tuple[int, list[tuple[str, str, str, str]], list[dict[str, Any]]]:
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"[qwen-cluster] batch {batch_idx}/{total_batches} segments={len(batch)}",
                flush=True,
            )
        payload = [
            {"aspect": aspect, "sentiment": sentiment, "text": text}
            for _, aspect, sentiment, text in batch
        ]
        local_assigner = assigner if workers == 1 else QwenClusterAssigner(assigner.args)
        return batch_idx, batch, local_assigner.assign_batch(payload)

    completed: list[tuple[int, list[tuple[str, str, str, str]], list[dict[str, Any]]]] = []
    if workers == 1 or total_batches <= 1:
        for batch_idx, batch in enumerate(batches, start=1):
            completed.append(run_batch(batch_idx, batch))
    else:
        print(
            f"[qwen-cluster] running {total_batches:,} batches with {workers:,} workers",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(run_batch, batch_idx, batch)
                for batch_idx, batch in enumerate(batches, start=1)
            ]
            for future in as_completed(futures):
                completed.append(future.result())

    for _, batch, items in sorted(completed, key=lambda item: item[0]):
        rows_to_cache: dict[str, tuple[str, str, str, dict[str, Any]]] = {}
        for (hash_key, aspect, sentiment, text), item in zip(batch, items):
            rule_item = rule_cluster_assignment(aspect, sentiment, text)
            cleaned = normalize_cluster_assignment(item, aspect, sentiment, text)
            if (
                assigner.args.cluster_assignment == "llm"
                and clamp_confidence(cleaned.get("confidence", 0.0)) < assigner.args.cluster_assignment_min_confidence
            ):
                cleaned = rule_item
                cleaned["source"] = "rule_low_llm_confidence"
            out[hash_key] = cleaned
            rows_to_cache[hash_key] = (aspect, sentiment, text, cleaned)
        cache.set_cluster_assignments(rows_to_cache)
    return out


def process_file_multistage_qwen(
    path: Path,
    args: argparse.Namespace,
    cache: ClassificationCache,
    pre_segmenter: QwenSemanticPreSegmenter | None,
    extractor: QwenAspectSegmenter,
    sentiment_classifier: QwenAspectSentimentClassifier,
    cluster_assigner: QwenClusterAssigner | None,
    hotels: dict[str, HotelAggregate],
    debug_stats: dict[str, Any],
    aspect_writer: "AspectOutputWriter | None",
    processed_sentence_writer: "ProcessedSentenceOutputWriter | None",
    final_summary: FinalSummaryAggregate | None,
    preseg_metric_writer: PreSegmentationMetricWriter | None = None,
    checkpoint: PipelineCheckpointManager | None = None,
    checkpoint_inputs: list[Path] | None = None,
    profile_decisions: ProfileDecisions | None = None,
) -> None:
    saved_state = checkpoint.get_file_state(path) if checkpoint is not None else {}
    if saved_state.get("completed"):
        debug_stats[path.name] = {
            "rows_seen": int(saved_state.get("rows_seen", 0)),
            "bad_ref_ids": int(saved_state.get("bad_ref_ids", 0)),
            "empty_reviews": int(saved_state.get("empty_reviews", 0)),
            "skipped_min_reviews": int(saved_state.get("skipped_min_reviews", 0)),
            "aspect_segments": int(saved_state.get("aspect_segments", 0)),
        }
        print(f"[resume-skip] {path.name}: already completed in checkpoint", flush=True)
        return

    saved_rows_seen = int(saved_state.get("rows_seen", 0))
    configured_skip_rows = max(0, int(getattr(args, "skip_rows_per_source", 0)))
    source_rows = saved_rows_seen if saved_rows_seen > 0 else configured_skip_rows
    bad_ref_ids = int(saved_state.get("bad_ref_ids", 0))
    empty_reviews = int(saved_state.get("empty_reviews", 0))
    skipped_min_reviews = int(saved_state.get("skipped_min_reviews", 0))
    extracted_segments_total = int(saved_state.get("aspect_segments", 0))
    eligible_entity_ids = getattr(args, "eligible_entity_ids", None)
    seen_hotels_for_source: defaultdict[str, set[str]] = defaultdict(
        set,
        {
            source: set(values)
            for source, values in dict(saved_state.get("seen_hotels_for_source", {})).items()
        },
    )
    resume_rows_remaining = source_rows
    if resume_rows_remaining > 0:
        label = "previously processed" if saved_rows_seen > 0 else "initial"
        print(f"[resume] {path.name}: skipping {resume_rows_remaining:,} {label} rows", flush=True)

    for chunk in pd.read_csv(path, dtype=str, chunksize=args.chunk_size, usecols=["ref_id", "review"]):
        if resume_rows_remaining > 0:
            if resume_rows_remaining >= len(chunk):
                resume_rows_remaining -= len(chunk)
                continue
            chunk = chunk.iloc[resume_rows_remaining:].copy()
            resume_rows_remaining = 0
        if args.max_rows_per_source and source_rows >= args.max_rows_per_source:
            break
        if args.max_rows_per_source:
            remaining_rows = max(0, args.max_rows_per_source - source_rows)
            if remaining_rows <= 0:
                break
            chunk = chunk.head(remaining_rows)
        if chunk.empty:
            continue

        sentence_records: list[dict[str, Any]] = []
        review_records: list[dict[str, Any]] = []
        for _, row in chunk.iterrows():
            source_rows += 1
            parsed = parse_ref_id(str(row.get("ref_id", "")))
            if parsed is None:
                bad_ref_ids += 1
                continue
            data_source, hotel_id, review_index = parsed
            entity_id = make_entity_id(data_source, hotel_id)
            if eligible_entity_ids is not None and entity_id not in eligible_entity_ids:
                skipped_min_reviews += 1
                continue
            if args.max_hotels_per_source:
                source_hotels = seen_hotels_for_source[data_source]
                if hotel_id not in source_hotels and len(source_hotels) >= args.max_hotels_per_source:
                    continue
                source_hotels.add(hotel_id)

            review_text = clean_text(row.get("review", ""))
            if not review_text:
                empty_reviews += 1
                continue

            if args.pre_segmentation == "semantic-qwen":
                normalized_review = normalize_for_cache(review_text)
                review_records.append(
                    {
                        "hash": semantic_review_hash(normalized_review),
                        "normalized": normalized_review,
                        "entity_id": entity_id,
                        "data_source": data_source,
                        "hotel_id": hotel_id,
                        "review_index": review_index,
                        "text": review_text,
                    }
                )
            else:
                sentences = split_sentences(review_text, args.min_words)
                if not sentences:
                    empty_reviews += 1
                    continue
                for local_idx, sentence in enumerate(sentences):
                    normalized = normalize_for_cache(sentence)
                    hash_key = aspect_extraction_hash(normalized)
                    sentence_records.append(
                        {
                            "hash": hash_key,
                            "normalized": normalized,
                            "entity_id": entity_id,
                            "data_source": data_source,
                            "hotel_id": hotel_id,
                            "review_index": review_index,
                            "sentence_id": f"{data_source}_{hotel_id}_{review_index}_{local_idx}",
                            "text": sentence,
                            "source_text": sentence,
                            "source_review": review_text,
                            "pre_segment_confidence": 1.0,
                            "semantic_source_precision": 1.0,
                        }
                    )

        if review_records:
            unique_review_hashes = list(dict.fromkeys(record["hash"] for record in review_records))
            cached_semantic_units = cache.get_semantic_segments(unique_review_hashes)
            pending_semantic_reviews: dict[str, str] = {}
            for record in review_records:
                if record["hash"] not in cached_semantic_units and record["hash"] not in pending_semantic_reviews:
                    pending_semantic_reviews[record["hash"]] = record["text"]
            semantic_map = {
                **cached_semantic_units,
                **(
                    extract_pending_semantic_units(
                        list(pending_semantic_reviews.items()),
                        cache,
                        pre_segmenter,
                    )
                    if pre_segmenter is not None
                    else {}
                ),
            }
            preseg_metric_rows: list[dict[str, Any]] = []
            for record in review_records:
                units = semantic_map.get(record["hash"], fallback_semantic_units(record["text"], args.min_words))
                if not units:
                    empty_reviews += 1
                    continue
                if preseg_metric_writer is not None and preseg_metric_writer.enabled:
                    unit_texts = [clean_text(unit.get("text", "")) for unit in units if clean_text(unit.get("text", ""))]
                    source_precisions = [
                        float(unit.get("source_precision", 0.0) or 0.0)
                        for unit in units
                        if clean_text(unit.get("text", ""))
                    ]
                    confidences = [
                        clamp_confidence(unit.get("confidence", 0.0))
                        for unit in units
                        if clean_text(unit.get("text", ""))
                    ]
                    decomposition_text = clean_text(" ".join(unit_texts))
                    if decomposition_text:
                        preseg_metric_rows.append(
                            {
                                "source_file": path.name,
                                "data_source": record["data_source"],
                                "hotel_id": record["hotel_id"],
                                "review_index": record["review_index"],
                                "review_hash": record["hash"],
                                "unit_count": len(unit_texts),
                                "review_word_count": len(WORD_RE.findall(record["text"])),
                                "decomposition_word_count": len(WORD_RE.findall(decomposition_text)),
                                "avg_unit_source_precision": round(statistics.mean(source_precisions), 6)
                                if source_precisions
                                else 0.0,
                                "min_unit_source_precision": round(min(source_precisions), 6)
                                if source_precisions
                                else 0.0,
                                "avg_pre_segment_confidence": round(statistics.mean(confidences), 6)
                                if confidences
                                else 0.0,
                                "review_text": record["text"],
                                "decomposition_text": decomposition_text,
                            }
                        )
                for local_idx, unit in enumerate(units):
                    unit_text = clean_text(unit.get("text", ""))
                    if not unit_text:
                        continue
                    normalized = normalize_for_cache(unit_text)
                    hash_key = aspect_extraction_hash(normalized)
                    sentence_records.append(
                        {
                            "hash": hash_key,
                            "normalized": normalized,
                            "entity_id": record["entity_id"],
                            "data_source": record["data_source"],
                            "hotel_id": record["hotel_id"],
                            "review_index": record["review_index"],
                            "sentence_id": (
                                f"{record['data_source']}_{record['hotel_id']}_{record['review_index']}_{local_idx}"
                            ),
                            "text": unit_text,
                            "source_text": clean_text(unit.get("source_text", "")) or record["text"],
                            "source_review": record["text"],
                            "pre_segment_confidence": clamp_confidence(unit.get("confidence", 0.65)),
                            "semantic_source_precision": float(unit.get("source_precision", 0.0) or 0.0),
                        }
                    )
            if preseg_metric_rows and preseg_metric_writer is not None:
                preseg_metric_writer.write_review_rows(preseg_metric_rows)

        if not sentence_records:
            continue

        unique_hashes = list(dict.fromkeys(record["hash"] for record in sentence_records))
        if args.skip_qwen:
            cached_extractions: dict[str, list[dict[str, Any]]] = {}
            pending_extractions = {
                record["hash"]: record["text"]
                for record in sentence_records
            }
            extracted_map = {
                record["hash"]: fallback_aspect_segments(record["text"])
                for record in sentence_records
            }
        else:
            cached_extractions = cache.get_aspect_extractions(unique_hashes)
            current_record_by_hash = {record["hash"]: record for record in sentence_records}
            if LEGACY_ASPECT_EXTRACTION_PROMPT_VERSIONS:
                legacy_hash_to_current: dict[str, str] = {}
                legacy_hashes: list[str] = []
                for record in sentence_records:
                    if record["hash"] in cached_extractions:
                        continue
                    for legacy_hash in legacy_aspect_extraction_hashes(record["normalized"]):
                        if legacy_hash not in legacy_hash_to_current:
                            legacy_hash_to_current[legacy_hash] = record["hash"]
                            legacy_hashes.append(legacy_hash)
                legacy_cached = cache.get_aspect_extractions(legacy_hashes)
                reusable_legacy_rows: dict[str, tuple[str, list[dict[str, Any]]]] = {}
                for legacy_hash, legacy_segments in legacy_cached.items():
                    current_hash = legacy_hash_to_current.get(legacy_hash, "")
                    if not current_hash or current_hash in cached_extractions:
                        continue
                    record = current_record_by_hash.get(current_hash)
                    if record is None:
                        continue
                    if should_reuse_legacy_aspect_segments(record["text"], legacy_segments):
                        cached_extractions[current_hash] = legacy_segments
                        reusable_legacy_rows[current_hash] = (record["normalized"], legacy_segments)
                if reusable_legacy_rows:
                    cache.set_aspect_extractions(reusable_legacy_rows)
            pending_extractions: dict[str, str] = {}
            for record in sentence_records:
                if record["hash"] not in cached_extractions and record["hash"] not in pending_extractions:
                    pending_extractions[record["hash"]] = record["text"]
            extracted_map = {
                **cached_extractions,
                **extract_pending_aspects(list(pending_extractions.items()), cache, extractor),
            }

        segment_records: list[dict[str, Any]] = []
        chunk_aspect_rows: list[dict[str, Any]] = []
        for record in sentence_records:
            segments = extracted_map.get(record["hash"], fallback_aspect_segments(record["text"]))
            if not segments and has_explicit_hotel_signal(record["text"]):
                segments = fallback_aspect_segments(record["text"])
            for segment_idx, segment in enumerate(segments):
                segment_text = clean_text(segment.get("segment_text", "")) or record["text"]
                llm_aspect = canonicalize_aspect_label(segment.get("aspect")) or fallback_aspect(segment_text)
                aspect = normalize_aspect(llm_aspect, segment_text)
                classification_text = preferred_text(segment, args.sentiment_language)
                summary_text = preferred_text(segment, args.summary_language)
                if not classification_text:
                    classification_text = record["text"]
                if not summary_text:
                    summary_text = record["text"]
                guardrail = guardrail_aspect_assignment(aspect, summary_text)
                final_aspect = guardrail["final_aspect"]
                rule_base_action = "kept" if final_aspect == llm_aspect else "rerouted"
                rule_base_reason = guardrail["reason"] or (
                    "definition_reroute" if rule_base_action == "rerouted" else ""
                )
                segment_hash_key = segment_cache_hash(final_aspect, classification_text)
                segment_records.append(
                    {
                        **record,
                        "aspect": final_aspect,
                        "llm_aspect": llm_aspect,
                        "aspect_before_guardrail": llm_aspect,
                        "aspect_guardrail_action": rule_base_action,
                        "aspect_guardrail_reason": rule_base_reason,
                        "aspect_guardrail_version": guardrail["version"],
                        "aspect_rule_base_action": rule_base_action,
                        "aspect_rule_base_reason": rule_base_reason,
                        "aspect_rule_base_version": guardrail["version"],
                        "aspect_segment_id": f"{record['sentence_id']}_{segment_idx}_{final_aspect}",
                        "segment_text": segment_text,
                        "detected_language": normalize_language_code(segment.get("detected_language", "")),
                        "normalized_text_vi": clean_text(segment.get("normalized_text_vi", "")) or record["text"],
                        "normalized_text_en": clean_text(segment.get("normalized_text_en", "")) or record["text"],
                        "classification_text": classification_text,
                        "summary_text": summary_text,
                        "source_text": record.get("source_text", record["text"]),
                        "source_review": record.get("source_review", record["text"]),
                        "pre_segment_confidence": record.get("pre_segment_confidence", 1.0),
                        "semantic_source_precision": record.get("semantic_source_precision", 1.0),
                        "extraction_confidence": clamp_confidence(segment.get("confidence", 0.45)),
                        "segment_hash": segment_hash_key,
                    }
                )

        extracted_segments_total += len(segment_records)
        unique_segment_hashes = list(dict.fromkeys(record["segment_hash"] for record in segment_records))
        if args.skip_qwen:
            cached_sentiments: dict[str, dict[str, Any]] = {}
            pending_segment_classifications = {
                record["segment_hash"]: (record["aspect"], record["classification_text"])
                for record in segment_records
            }
            classified_sentiments = {
                record["segment_hash"]: fallback_sentiment_item(record["aspect"], record["classification_text"])
                for record in segment_records
            }
        else:
            cached_sentiments = cache.get_segment_sentiments(unique_segment_hashes)
            pending_segment_classifications: dict[str, tuple[str, str]] = {}
            for record in segment_records:
                if (
                    record["segment_hash"] not in cached_sentiments
                    and record["segment_hash"] not in pending_segment_classifications
                ):
                    pending_segment_classifications[record["segment_hash"]] = (
                        record["aspect"],
                        record["classification_text"],
                    )
            classified_sentiments = {
                **cached_sentiments,
                **classify_pending_segment_sentiments(
                    [
                        (hash_key, values[0], values[1])
                        for hash_key, values in pending_segment_classifications.items()
                    ],
                    cache,
                    sentiment_classifier,
                ),
            }

        for record in segment_records:
            sentiment_item = classified_sentiments.get(
                record["segment_hash"],
                fallback_sentiment_item(record["aspect"], record["classification_text"]),
            )
            record["sentiment"] = normalize_sentiment(sentiment_item.get("sentiment", ""))
            record["sentiment_confidence"] = clamp_confidence(sentiment_item.get("confidence", 0.45))
            if not record.get("aspect_guardrail_version"):
                guardrail = guardrail_aspect_assignment(record["aspect"], record["summary_text"])
                before_rule_base = record.get("llm_aspect") or guardrail["original_aspect"]
                rule_base_action = "kept" if guardrail["final_aspect"] == before_rule_base else "rerouted"
                rule_base_reason = guardrail["reason"] or (
                    "definition_reroute" if rule_base_action == "rerouted" else ""
                )
                record["aspect_before_guardrail"] = before_rule_base
                record["aspect_guardrail_action"] = rule_base_action
                record["aspect_guardrail_reason"] = rule_base_reason
                record["aspect_guardrail_version"] = guardrail["version"]
                record["aspect_rule_base_action"] = rule_base_action
                record["aspect_rule_base_reason"] = rule_base_reason
                record["aspect_rule_base_version"] = guardrail["version"]
                record["aspect"] = guardrail["final_aspect"]
            record["cluster_hash"] = cluster_assignment_hash(
                record["aspect"],
                record["sentiment"],
                record["summary_text"],
            )

        needs_cluster_assignments = (
            not getattr(args, "disable_final_summary_clusters", False)
            and (
                final_summary is not None
                or aspect_writer is not None
                or processed_sentence_writer is not None
            )
        )
        if needs_cluster_assignments:
            unique_cluster_hashes = list(dict.fromkeys(record["cluster_hash"] for record in segment_records))
            if args.cluster_assignment == "rule" or args.skip_qwen or cluster_assigner is None:
                cached_cluster_assignments: dict[str, dict[str, Any]] = {}
                pending_cluster_assignments = {
                    record["cluster_hash"]: (
                        record["aspect"],
                        record["sentiment"],
                        record["summary_text"],
                    )
                    for record in segment_records
                }
                cluster_assignments = {
                    record["cluster_hash"]: rule_cluster_assignment(
                        record["aspect"],
                        record["sentiment"],
                        record["summary_text"],
                    )
                    for record in segment_records
                }
            else:
                cached_cluster_assignments_all = cache.get_cluster_assignments(unique_cluster_hashes)
                cached_cluster_assignments: dict[str, dict[str, Any]] = {}
                pending_cluster_assignments: dict[str, tuple[str, str, str]] = {}
                for record in segment_records:
                    cached_item = cached_cluster_assignments_all.get(record["cluster_hash"])
                    if cached_item is not None and not cluster_needs_llm_refinement(
                        cached_item,
                        args.cluster_assignment_min_confidence,
                    ):
                        cached_cluster_assignments[record["cluster_hash"]] = cached_item
                        continue
                    if (
                        record["cluster_hash"] not in pending_cluster_assignments
                    ):
                        pending_cluster_assignments[record["cluster_hash"]] = (
                            record["aspect"],
                            record["sentiment"],
                            record["summary_text"],
                        )
                cluster_assignments = {
                    **cached_cluster_assignments,
                    **assign_pending_clusters(
                        [
                            (hash_key, values[0], values[1], values[2])
                            for hash_key, values in pending_cluster_assignments.items()
                        ],
                        cache,
                        cluster_assigner,
                    ),
                }
        else:
            cached_cluster_assignments = {}
            pending_cluster_assignments = {}
            cluster_assignments = {}

        for record in segment_records:
            sentiment = record["sentiment"]
            confidence = record["sentiment_confidence"]
            if getattr(args, "disable_final_summary_clusters", False):
                cluster_item = {
                    "cluster_code": "",
                    "cluster_label": "",
                    "descriptors": [],
                    "confidence": 0.0,
                    "source": "disabled",
                }
            else:
                cluster_item = cluster_assignments.get(
                    record.get("cluster_hash", ""),
                    rule_cluster_assignment(record["aspect"], sentiment, record["summary_text"]),
                )
                cluster_item = canonicalize_cluster_assignment_item(
                    cluster_item,
                    record["aspect"],
                    sentiment,
                    record["summary_text"],
                )
            if not args.final_summary_only:
                aggregate = hotels.get(record["entity_id"])
                if aggregate is None:
                    aggregate = HotelAggregate(record["data_source"], record["hotel_id"])
                    hotels[record["entity_id"]] = aggregate
                aggregate.add_sentence(
                    review_index=record["review_index"],
                    sentence_id=record["aspect_segment_id"],
                    text=record["summary_text"],
                    aspect=record["aspect"],
                    sentiment=sentiment,
                    confidence=confidence,
                    aspect_score=max(
                        record["extraction_confidence"],
                        aspect_score_for_text(record["aspect"], record["segment_text"]),
                    ),
                    keep_debug_samples=args.debug_samples_per_aspect,
                    sample_payload={
                        "original_sentence": record["text"],
                        "source_text": record["source_text"],
                        "source_review": record["source_review"],
                        "segment_text": record["segment_text"],
                        "detected_language": record["detected_language"],
                        "normalized_text_vi": record["normalized_text_vi"],
                        "normalized_text_en": record["normalized_text_en"],
                        "classification_text": record["classification_text"],
                        "summary_text": record["summary_text"],
                        "pre_segment_confidence": record["pre_segment_confidence"],
                        "semantic_source_precision": record["semantic_source_precision"],
                        "extraction_confidence": record["extraction_confidence"],
                    },
                    keep_reference_text=not args.disable_bertscore,
                    keep_reference_stats=not args.disable_summary_reference_stats,
                    cluster_assignment_source=clean_text(cluster_item.get("source", "")),
                )
            if final_summary is not None:
                final_summary.add(
                    entity_id=record["entity_id"],
                    data_source=record["data_source"],
                    hotel_id=record["hotel_id"],
                    source_file=path.name,
                    aspect=record["aspect"],
                    sentiment=sentiment,
                    text=record["summary_text"],
                    confidence=confidence,
                    keep_reference_text=not args.disable_bertscore,
                    keep_reference_stats=not args.disable_summary_reference_stats,
                    cluster_code=clean_text(cluster_item.get("cluster_code", "")),
                    cluster_label=clean_text(cluster_item.get("cluster_label", "")),
                    cluster_descriptors=[
                        clean_text(value) for value in cluster_item.get("descriptors", []) if clean_text(value)
                    ],
                )
            chunk_aspect_rows.append(
                {
                    "source_file": path.name,
                    "entity_id": record["entity_id"],
                    "data_source": record["data_source"],
                    "hotel_id": record["hotel_id"],
                    "review_index": record["review_index"],
                    "sentence_id": record["sentence_id"],
                    "aspect_segment_id": record["aspect_segment_id"],
                    "aspect": record["aspect"],
                    "original_sentence": record["text"],
                    "shortened_sentence": record["text"],
                    "processed_sentence": record["text"],
                    "source_text": record["source_text"],
                    "source_sentence": record["source_text"],
                    "source_review": record["source_review"],
                    "segment_text": record["segment_text"],
                    "aspect_segment_text": record["segment_text"],
                    "detected_language": record["detected_language"],
                    "normalized_text_vi": record["normalized_text_vi"],
                    "normalized_text_en": record["normalized_text_en"],
                    "classification_text": record["classification_text"],
                    "summary_text": record["summary_text"],
                    "llm_aspect": record.get("llm_aspect", ""),
                    "sentiment": sentiment,
                    "sentiment_confidence": confidence,
                    "cluster_code": clean_text(cluster_item.get("cluster_code", "")),
                    "cluster_label": clean_text(cluster_item.get("cluster_label", "")),
                    "cluster_descriptors": json.dumps(
                        [clean_text(value) for value in cluster_item.get("descriptors", []) if clean_text(value)],
                        ensure_ascii=False,
                    ),
                    "cluster_assignment_confidence": clamp_confidence(cluster_item.get("confidence", 0.45)),
                    "cluster_assignment_source": clean_text(cluster_item.get("source", "")),
                    "aspect_before_guardrail": record.get("aspect_before_guardrail", ""),
                    "aspect_guardrail_action": record.get("aspect_guardrail_action", ""),
                    "aspect_guardrail_reason": record.get("aspect_guardrail_reason", ""),
                    "aspect_guardrail_version": record.get("aspect_guardrail_version", ""),
                    "aspect_rule_base_action": record.get("aspect_rule_base_action", ""),
                    "aspect_rule_base_reason": record.get("aspect_rule_base_reason", ""),
                    "aspect_rule_base_version": record.get("aspect_rule_base_version", ""),
                    "pre_segment_confidence": record["pre_segment_confidence"],
                    "semantic_source_precision": record["semantic_source_precision"],
                    "extraction_confidence": record["extraction_confidence"],
                }
            )

        if chunk_aspect_rows and aspect_writer is not None:
            aspect_writer.write_rows(chunk_aspect_rows)
        if chunk_aspect_rows and processed_sentence_writer is not None:
            processed_sentence_writer.write_rows(chunk_aspect_rows)

        if args.progress_every > 0 and source_rows % args.progress_every < len(chunk):
            print(
                f"[aspect-pipeline] {path.name}: rows={source_rows:,} hotels={len(hotels):,} "
                f"segments={extracted_segments_total:,} cached_extractions={len(cached_extractions):,} "
                f"new_extractions={len(pending_extractions):,} cached_sentiments={len(cached_sentiments):,} "
                f"new_segment_sentiments={len(pending_segment_classifications):,} "
                f"cached_clusters={len(cached_cluster_assignments):,} "
                f"new_clusters={len(pending_cluster_assignments):,}",
                flush=True,
            )
        debug_stats[path.name] = {
            "rows_seen": source_rows,
            "bad_ref_ids": bad_ref_ids,
            "empty_reviews": empty_reviews,
            "skipped_min_reviews": skipped_min_reviews,
            "aspect_segments": extracted_segments_total,
        }
        if checkpoint is not None:
            checkpoint.update_file_state(
                path,
                build_file_progress_state(
                    rows_seen=source_rows,
                    bad_ref_ids=bad_ref_ids,
                    empty_reviews=empty_reviews,
                    seen_hotels_for_source=seen_hotels_for_source,
                    skipped_min_reviews=skipped_min_reviews,
                    aspect_segments=extracted_segments_total,
                    completed=False,
                ),
            )
            checkpoint.maybe_save(
                inputs=checkpoint_inputs or [path],
                strategy=args.strategy,
                hotels=hotels,
                stats={"files": debug_stats},
                final_summary=final_summary,
                profile_decisions=profile_decisions or {},
            )

    debug_stats[path.name] = {
        "rows_seen": source_rows,
        "bad_ref_ids": bad_ref_ids,
        "empty_reviews": empty_reviews,
        "skipped_min_reviews": skipped_min_reviews,
        "aspect_segments": extracted_segments_total,
    }
    if checkpoint is not None:
        checkpoint.update_file_state(
            path,
            build_file_progress_state(
                rows_seen=source_rows,
                bad_ref_ids=bad_ref_ids,
                empty_reviews=empty_reviews,
                seen_hotels_for_source=seen_hotels_for_source,
                skipped_min_reviews=skipped_min_reviews,
                aspect_segments=extracted_segments_total,
                completed=True,
            ),
        )
        checkpoint.maybe_save(
            inputs=checkpoint_inputs or [path],
            strategy=args.strategy,
            hotels=hotels,
            stats={"files": debug_stats},
            final_summary=final_summary,
            profile_decisions=profile_decisions or {},
            force=True,
        )


def process_file_local(
    path: Path,
    args: argparse.Namespace,
    hotels: dict[str, HotelAggregate],
    debug_stats: dict[str, Any],
    checkpoint: PipelineCheckpointManager | None = None,
    checkpoint_inputs: list[Path] | None = None,
    profile_decisions: ProfileDecisions | None = None,
) -> None:
    saved_state = checkpoint.get_file_state(path) if checkpoint is not None else {}
    if saved_state.get("completed"):
        debug_stats[path.name] = {
            "rows_seen": int(saved_state.get("rows_seen", 0)),
            "bad_ref_ids": int(saved_state.get("bad_ref_ids", 0)),
            "empty_reviews": int(saved_state.get("empty_reviews", 0)),
            "skipped_min_reviews": int(saved_state.get("skipped_min_reviews", 0)),
        }
        print(f"[resume-skip] {path.name}: already completed in checkpoint", flush=True)
        return

    saved_rows_seen = int(saved_state.get("rows_seen", 0))
    configured_skip_rows = max(0, int(getattr(args, "skip_rows_per_source", 0)))
    source_rows = saved_rows_seen if saved_rows_seen > 0 else configured_skip_rows
    bad_ref_ids = int(saved_state.get("bad_ref_ids", 0))
    empty_reviews = int(saved_state.get("empty_reviews", 0))
    skipped_min_reviews = int(saved_state.get("skipped_min_reviews", 0))
    eligible_entity_ids = getattr(args, "eligible_entity_ids", None)
    seen_hotels_for_source: defaultdict[str, set[str]] = defaultdict(
        set,
        {
            source: set(values)
            for source, values in dict(saved_state.get("seen_hotels_for_source", {})).items()
        },
    )
    resume_rows_remaining = source_rows
    if resume_rows_remaining > 0:
        label = "previously processed" if saved_rows_seen > 0 else "initial"
        print(f"[resume] {path.name}: skipping {resume_rows_remaining:,} {label} rows", flush=True)

    for chunk in pd.read_csv(path, dtype=str, chunksize=args.chunk_size, usecols=["ref_id", "review"]):
        if resume_rows_remaining > 0:
            if resume_rows_remaining >= len(chunk):
                resume_rows_remaining -= len(chunk)
                continue
            chunk = chunk.iloc[resume_rows_remaining:].copy()
            resume_rows_remaining = 0
        if args.max_rows_per_source and source_rows >= args.max_rows_per_source:
            break
        if args.max_rows_per_source:
            remaining_rows = max(0, args.max_rows_per_source - source_rows)
            if remaining_rows <= 0:
                break
            chunk = chunk.head(remaining_rows)
        if chunk.empty:
            continue

        for _, row in chunk.iterrows():
            source_rows += 1
            parsed = parse_ref_id(str(row.get("ref_id", "")))
            if parsed is None:
                bad_ref_ids += 1
                continue
            data_source, hotel_id, review_index = parsed
            entity_id = make_entity_id(data_source, hotel_id)
            if eligible_entity_ids is not None and entity_id not in eligible_entity_ids:
                skipped_min_reviews += 1
                continue
            if args.max_hotels_per_source:
                source_hotels = seen_hotels_for_source[data_source]
                if hotel_id not in source_hotels and len(source_hotels) >= args.max_hotels_per_source:
                    continue
                source_hotels.add(hotel_id)

            review_text = clean_text(row.get("review", ""))
            if not review_text:
                empty_reviews += 1
                continue
            sentences = split_sentences(review_text, args.min_words)
            if not sentences:
                empty_reviews += 1
                continue

            aggregate = hotels.get(entity_id)
            if aggregate is None:
                aggregate = HotelAggregate(data_source, hotel_id)
                hotels[entity_id] = aggregate

            for local_idx, sentence in enumerate(sentences):
                item = fallback_classify(sentence)
                aspect = normalize_aspect(item.get("aspect"), sentence)
                sentiment = normalize_sentiment(item.get("sentiment"))
                confidence = clamp_confidence(item.get("confidence", 0.45))
                aggregate.add_sentence(
                    review_index=review_index,
                    sentence_id=f"{data_source}_{hotel_id}_{review_index}_{local_idx}",
                    text=sentence,
                    aspect=aspect,
                    sentiment=sentiment,
                    confidence=confidence,
                    aspect_score=aspect_score_for_text(aspect, sentence),
                    keep_debug_samples=args.debug_samples_per_aspect,
                    keep_reference_text=not args.disable_bertscore,
                    keep_reference_stats=not args.disable_summary_reference_stats,
                    cluster_assignment_source="rule",
                )

        if args.progress_every > 0 and source_rows % args.progress_every < len(chunk):
            print(
                f"[local] {path.name}: rows={source_rows:,} hotels={len(hotels):,}",
                flush=True,
            )
        debug_stats[path.name] = {
            "rows_seen": source_rows,
            "bad_ref_ids": bad_ref_ids,
            "empty_reviews": empty_reviews,
            "skipped_min_reviews": skipped_min_reviews,
        }
        if checkpoint is not None:
            checkpoint.update_file_state(
                path,
                build_file_progress_state(
                    rows_seen=source_rows,
                    bad_ref_ids=bad_ref_ids,
                    empty_reviews=empty_reviews,
                    seen_hotels_for_source=seen_hotels_for_source,
                    skipped_min_reviews=skipped_min_reviews,
                    completed=False,
                ),
            )
            checkpoint.maybe_save(
                inputs=checkpoint_inputs or [path],
                strategy=args.strategy,
                hotels=hotels,
                stats={"files": debug_stats},
                final_summary=None,
                profile_decisions=profile_decisions or {},
            )

    debug_stats[path.name] = {
        "rows_seen": source_rows,
        "bad_ref_ids": bad_ref_ids,
        "empty_reviews": empty_reviews,
        "skipped_min_reviews": skipped_min_reviews,
    }
    if checkpoint is not None:
        checkpoint.update_file_state(
            path,
            build_file_progress_state(
                rows_seen=source_rows,
                bad_ref_ids=bad_ref_ids,
                empty_reviews=empty_reviews,
                seen_hotels_for_source=seen_hotels_for_source,
                skipped_min_reviews=skipped_min_reviews,
                completed=True,
            ),
        )
        checkpoint.maybe_save(
            inputs=checkpoint_inputs or [path],
            strategy=args.strategy,
            hotels=hotels,
            stats={"files": debug_stats},
            final_summary=None,
            profile_decisions=profile_decisions or {},
            force=True,
        )


def collect_aspect_clusters_for_summary(
    final_summary: FinalSummaryAggregate | None,
    entity_id: str,
    aspect: str,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    if getattr(args, "disable_final_summary_clusters", False):
        return {}
    if final_summary is None:
        return {}
    group_buckets = final_summary.buckets.get(entity_id, {})
    aspect_buckets = group_buckets.get(aspect, {}) if isinstance(group_buckets, dict) else {}
    clusters_by_sentiment: dict[str, list[dict[str, Any]]] = {}
    for sentiment in SENTIMENTS:
        bucket = aspect_buckets.get(sentiment) if isinstance(aspect_buckets, dict) else None
        if bucket is None:
            continue
        cluster_items = bucket.cluster_items(
            args.final_summary_max_clusters,
            args.final_summary_sample_chars,
            args.final_summary_cluster_descriptors,
        )
        if cluster_items:
            clusters_by_sentiment[sentiment] = cluster_items
    return clusters_by_sentiment


def generate_aspect_summaries(
    hotels: dict[str, HotelAggregate],
    args: argparse.Namespace,
    final_summary: FinalSummaryAggregate | None = None,
) -> None:
    existing = 0
    needed = 0
    for hotel in hotels.values():
        for aspect in ASPECT_NAMES:
            agg = hotel.aspects[aspect]
            if aspect_total(agg) <= 0:
                continue
            needed += 1
            if has_non_formulaic_summary(agg.summary_vi) or has_non_formulaic_summary(agg.summary_en):
                existing += 1
    if needed > 0 and existing == needed:
        print(f"[aspect-summary] using {existing:,} summaries already present in checkpoint", flush=True)
        return

    summary_items: list[dict[str, Any]] = []
    for entity_id, hotel in sorted(hotels.items(), key=lambda kv: (kv[1].data_source, kv[1].hotel_id)):
        for aspect in ASPECT_NAMES:
            agg = hotel.aspects[aspect]
            if aspect_total(agg) <= 0:
                continue
            rep_vi = ""
            rep_en = ""
            final_sentiment = agg.final_sentiment()
            meta = agg.representative_meta.get(final_sentiment, {})
            if not meta and agg.representative_meta:
                meta = next(iter(agg.representative_meta.values()))
            rep_vi = clean_text(meta.get("normalized_text_vi", "")) or agg.final_representative()
            rep_en = clean_text(meta.get("normalized_text_en", "")) or agg.final_representative()
            summary_items.append(
                {
                    "entity_id": entity_id,
                    "data_source": hotel.data_source,
                    "hotel_id": hotel.hotel_id,
                    "aspect": aspect,
                    "aspect_code": ASPECT_SUMMARY_CODES.get(aspect, aspect.upper()[:3] or "ASP"),
                    "final_sentiment": final_sentiment,
                    "counts": {
                        "positive": int(agg.counts.get("positive", 0)),
                        "negative": int(agg.counts.get("negative", 0)),
                        "neutral": int(agg.counts.get("neutral", 0)),
                    },
                    "representative_vi": rep_vi,
                    "representative_en": rep_en,
                    "clusters_by_sentiment": collect_aspect_clusters_for_summary(
                        final_summary,
                        entity_id,
                        aspect,
                        args,
                    ),
                }
            )

    if not summary_items:
        return

    batch_size = max(1, int(args.summary_batch_size))
    batches = [summary_items[start : start + batch_size] for start in range(0, len(summary_items), batch_size)]
    workers = max(1, int(getattr(args, "summary_workers", 1)))
    progress_every = max(1, int(getattr(args, "summary_progress_every_batches", 100)))
    results_by_batch: list[list[dict[str, str]]] = [[] for _ in batches]

    print(
        f"[aspect-summary] running {len(batches):,} summary batches with {workers:,} workers",
        flush=True,
    )

    def summarize_one(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
        return QwenAspectSummarizer(args).summarize_batch(batch)

    if workers == 1 or len(batches) <= 1:
        summarizer = QwenAspectSummarizer(args)
        for idx, batch in enumerate(batches):
            results_by_batch[idx] = summarize_batch_with_cache(
                args,
                "hotel_aspect_summary",
                FINAL_ASPECT_SUMMARY_PROMPT_VERSION,
                batch,
                summarizer.summarize_batch,
            )
            completed = idx + 1
            if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                print(f"[aspect-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(summarize_one, batch): idx for idx, batch in enumerate(batches)}
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                batch = batches[idx]
                try:
                    results_by_batch[idx] = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad summary batch should not kill the full run.
                    print(
                        f"[WARN] aspect summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                        file=sys.stderr,
                    )
                    results_by_batch[idx] = [local_aspect_summary(item) for item in batch]
                completed += 1
                if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                    print(f"[aspect-summary] completed {completed:,}/{len(batches):,} batches", flush=True)

    for batch, results in zip(batches, results_by_batch):
        for item, summary in zip(batch, results):
            agg = hotels[item["entity_id"]].aspects[item["aspect"]]
            agg.summary_vi = clean_summary_text(summary.get("summary_vi", ""), 1600)
            agg.summary_en = clean_summary_text(summary.get("summary_en", ""), 1600)


class AspectOutputWriter:
    def __init__(self, base_dir: Path, reset_existing: bool = True):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if reset_existing:
            for old_csv in self.base_dir.rglob("*.csv"):
                old_csv.unlink()
        self._initialized_paths: set[Path] = set()
        self.has_rows = False

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.has_rows = True
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            aspect = str(row.get("aspect", ""))
            if aspect not in ASPECT_NAMES:
                continue
            for language in ("vi", "en"):
                payload = dict(row)
                payload["normalized_text"] = payload.get(f"normalized_text_{language}", "")
                grouped[(language, aspect)].append(payload)

        for (language, aspect), payload_rows in grouped.items():
            language_dir = self.base_dir / language
            language_dir.mkdir(parents=True, exist_ok=True)
            output_path = language_dir / f"{aspect}.csv"
            write_header = output_path not in self._initialized_paths and not output_path.exists()
            with output_path.open("a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(payload_rows[0].keys()))
                if write_header:
                    writer.writeheader()
                writer.writerows(payload_rows)
            self._initialized_paths.add(output_path)


PROCESSED_SENTENCE_FIELDNAMES = [
    "source_file",
    "entity_id",
    "data_source",
    "hotel_id",
    "review_index",
    "sentence_id",
    "aspect_segment_id",
    "source_review",
    "source_text",
    "source_sentence",
    "shortened_sentence",
    "processed_sentence",
    "aspect_segment_text",
    "segment_text",
    "normalized_text_vi",
    "normalized_text_en",
    "classification_text",
    "summary_text",
    "llm_aspect",
    "aspect",
    "sentiment",
    "sentiment_confidence",
    "cluster_code",
    "cluster_label",
    "cluster_descriptors",
    "cluster_assignment_confidence",
    "cluster_assignment_source",
    "aspect_before_guardrail",
    "aspect_guardrail_action",
    "aspect_guardrail_reason",
    "aspect_guardrail_version",
    "aspect_rule_base_action",
    "aspect_rule_base_reason",
    "aspect_rule_base_version",
    "pre_segment_confidence",
    "semantic_source_precision",
    "extraction_confidence",
]


class ProcessedSentenceOutputWriter:
    def __init__(self, path: Path, reset_existing: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset_existing and self.path.exists():
            self.path.unlink()
        self._initialized = self.path.exists() and self.path.stat().st_size > 0
        self.has_rows = False

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.has_rows = True
        with self.path.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROCESSED_SENTENCE_FIELDNAMES, extrasaction="ignore")
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in PROCESSED_SENTENCE_FIELDNAMES})


def build_final_summary_rows(final_summary: FinalSummaryAggregate, args: argparse.Namespace) -> list[dict[str, Any]]:
    summarizer = QwenFinalSummaryWriter(args)
    disable_clusters = getattr(args, "disable_final_summary_clusters", False)
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    item_positions: list[tuple[int, str]] = []

    for group_key in final_summary.sorted_group_keys():
        group_meta = final_summary.group_metadata.get(group_key, {})
        for aspect in ASPECT_NAMES:
            buckets = final_summary.buckets[group_key][aspect]
            total_count = sum(buckets[sentiment].count for sentiment in SENTIMENTS)
            if total_count <= 0:
                continue
            row: dict[str, Any] = {
                "_group_key": group_key,
                "hotel_id": group_meta.get("hotel_id", ""),
                "aspect": aspect,
                "overall_aspect_summary": "",
            }
            for sentiment in SENTIMENTS:
                bucket = buckets[sentiment]
                cluster_items = (
                    []
                    if disable_clusters
                    else bucket.cluster_items(
                        args.final_summary_max_clusters,
                        args.final_summary_sample_chars,
                        args.final_summary_cluster_descriptors,
                    )
                )
                row[f"{sentiment}_count"] = bucket.count
                row[f"{sentiment}_avg_confidence"] = round(bucket.average_confidence(), 6)
                row[f"{sentiment}_cluster_count"] = 0 if disable_clusters else len(getattr(bucket, "clusters", {}) or {})
                row[f"{sentiment}_clusters"] = json.dumps(cluster_items, ensure_ascii=False)
                row[f"{sentiment}_summary"] = ""
                if bucket.count > 0:
                    compact_samples = [
                        clean_text(sample)[: args.final_summary_sample_chars]
                        for sample in bucket.samples
                        if clean_text(sample)
                    ]
                    items.append(
                        {
                            "hotel_id": group_meta.get("hotel_id", ""),
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "count": bucket.count,
                            "avg_confidence": round(bucket.average_confidence(), 6),
                            "clusters": cluster_items,
                            "samples": compact_samples,
                        }
                    )
                    item_positions.append((len(rows), sentiment))
            rows.append(row)

    batch_size = max(1, int(args.final_summary_batch_size))
    batches = [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
    workers = max(1, int(args.final_summary_workers))
    progress_every = max(1, int(getattr(args, "final_summary_progress_every_batches", 100)))
    summaries_by_batch: list[list[str]] = [[] for _ in batches]

    if workers == 1 or len(batches) <= 1:
        for idx, batch in enumerate(batches):
            summaries_by_batch[idx] = summarize_batch_with_cache(
                args,
                "final_sentiment_summary",
                FINAL_SENTIMENT_SUMMARY_PROMPT_VERSION,
                batch,
                summarizer.summarize_batch,
            )
            completed = idx + 1
            if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                print(f"[sentiment-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
    else:
        print(
            f"[sentiment-summary] running {len(batches):,} summary batches with {workers:,} workers",
            flush=True,
        )

        def summarize_one(batch: list[dict[str, Any]]) -> list[str]:
            # Use one client per worker call path to avoid sharing transport state across threads.
            return QwenFinalSummaryWriter(args).summarize_batch(batch)

        max_pending = min(len(batches), max(workers * 2, workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Any, tuple[int, list[dict[str, Any]]]] = {}
            next_idx = 0
            completed = 0

            def submit_next() -> None:
                nonlocal next_idx
                if next_idx >= len(batches):
                    return
                batch = batches[next_idx]
                futures[executor.submit(summarize_one, batch)] = (next_idx, batch)
                next_idx += 1

            for _ in range(max_pending):
                submit_next()

            while futures:
                for future in as_completed(futures):
                    idx, batch = futures.pop(future)
                    break
                try:
                    summaries_by_batch[idx] = future.result()
                except Exception as exc:  # noqa: BLE001 - one bad summary batch should not kill the full run.
                    print(
                        f"[WARN] sentiment summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                        file=sys.stderr,
                    )
                    summaries_by_batch[idx] = [business_final_sentiment_summary(item) for item in batch]
                completed += 1
                if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                    print(f"[sentiment-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
                submit_next()

    summaries = [summary for batch_summaries in summaries_by_batch for summary in batch_summaries]

    for (row_idx, sentiment), summary in zip(item_positions, summaries):
        rows[row_idx][f"{sentiment}_summary"] = clean_text(summary)

    aspect_items: list[dict[str, Any]] = []
    for row in rows:
        counts = {sentiment: int(row.get(f"{sentiment}_count", 0) or 0) for sentiment in SENTIMENTS}
        dominant_sentiment = max(SENTIMENTS, key=lambda sentiment: counts[sentiment])
        clusters_by_sentiment: dict[str, list[dict[str, Any]]] = {}
        for sentiment in SENTIMENTS:
            clusters_raw = row.get(f"{sentiment}_clusters", "[]")
            try:
                parsed_clusters = json.loads(clusters_raw) if isinstance(clusters_raw, str) else clusters_raw
            except Exception:
                parsed_clusters = []
            clusters_value = parsed_clusters if isinstance(parsed_clusters, list) else []
            clusters_by_sentiment[sentiment] = [
                {**item, "sentiment": sentiment}
                for item in clusters_value
                if isinstance(item, dict)
            ]
        aspect_items.append(
            {
                "hotel_id": row.get("hotel_id", ""),
                "aspect": row.get("aspect", ""),
                "aspect_code": ASPECT_SUMMARY_CODES.get(str(row.get("aspect", "")), str(row.get("aspect", "")).upper()[:3] or "ASP"),
                "final_sentiment": dominant_sentiment,
                "counts": counts,
                "representative_vi": ""
                if getattr(args, "summary_language", "vi") == "en"
                else " ".join(
                    clean_text(row.get(f"{sentiment}_summary", "")) for sentiment in SENTIMENTS
                ).strip(),
                "representative_en": " ".join(
                    clean_text(row.get(f"{sentiment}_summary", "")) for sentiment in SENTIMENTS
                ).strip()
                if getattr(args, "summary_language", "vi") == "en"
                else "",
                "clusters_by_sentiment": clusters_by_sentiment,
            }
        )

    aspect_summaries = summarize_final_aspect_items_in_batches(aspect_items, args)
    for row, aspect_summary in zip(rows, aspect_summaries):
        if getattr(args, "summary_language", "vi") == "en":
            qwen_summary = clean_summary_text(aspect_summary.get("summary_en", ""), 1800) or clean_summary_text(
                aspect_summary.get("summary_vi", ""), 1800
            )
        else:
            qwen_summary = clean_summary_text(aspect_summary.get("summary_vi", ""), 1800) or clean_summary_text(
                aspect_summary.get("summary_en", ""), 1800
            )
        row["overall_aspect_summary"] = enforce_structured_overall_summary(
            row,
            qwen_summary,
            getattr(args, "summary_language", "vi"),
        )

    rows = add_all_aspects_summary_rows(rows, args)
    return rows


def summarize_final_aspect_items_in_batches(
    aspect_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    if not aspect_items:
        return []

    batch_size = max(1, int(getattr(args, "summary_batch_size", 4) or 4))
    workers = max(1, int(getattr(args, "summary_workers", getattr(args, "final_summary_workers", 1)) or 1))
    progress_every = max(1, int(getattr(args, "summary_progress_every_batches", 100) or 100))
    batches = [aspect_items[start : start + batch_size] for start in range(0, len(aspect_items), batch_size)]
    results_by_batch: list[list[dict[str, str]]] = [[] for _ in batches]

    print(
        f"[final-aspect-summary] running {len(batches):,} final-row aspect summary batches with {workers:,} workers",
        flush=True,
    )

    def summarize_one(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
        return QwenAspectSummarizer(args).summarize_batch(batch)

    if workers == 1 or len(batches) <= 1:
        summarizer = QwenAspectSummarizer(args)
        for idx, batch in enumerate(batches):
            try:
                results_by_batch[idx] = summarize_batch_with_cache(
                    args,
                    "final_aspect_summary",
                    FINAL_ASPECT_SUMMARY_PROMPT_VERSION,
                    batch,
                    summarizer.summarize_batch,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] aspect summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                    file=sys.stderr,
                )
                results_by_batch[idx] = [local_aspect_summary(item) for item in batch]
            completed = idx + 1
            if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                print(f"[final-aspect-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
    else:
        max_pending = min(len(batches), max(workers * 2, workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Any, tuple[int, list[dict[str, Any]]]] = {}
            next_idx = 0
            completed = 0

            def submit_next() -> None:
                nonlocal next_idx
                if next_idx >= len(batches):
                    return
                batch = batches[next_idx]
                futures[executor.submit(summarize_one, batch)] = (next_idx, batch)
                next_idx += 1

            for _ in range(max_pending):
                submit_next()

            while futures:
                for future in as_completed(futures):
                    idx, batch = futures.pop(future)
                    break
                try:
                    results_by_batch[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[WARN] aspect summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                        file=sys.stderr,
                    )
                    results_by_batch[idx] = [local_aspect_summary(item) for item in batch]
                completed += 1
                if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                    print(f"[final-aspect-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
                submit_next()

    return [summary for batch_summaries in results_by_batch for summary in batch_summaries]


def add_all_aspects_summary_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not rows:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_order: list[str] = []
    for row in rows:
        group_key = str(row.get("_group_key", ""))
        if not group_key or str(row.get("aspect", "")) not in ASPECT_NAMES:
            continue
        if group_key not in grouped:
            group_order.append(group_key)
        grouped[group_key].append(row)

    all_aspects_by_group: dict[str, dict[str, Any]] = {}
    all_aspects_items: list[dict[str, Any]] = []
    for group_key in group_order:
        detail_rows = grouped[group_key]
        first = detail_rows[0]
        row: dict[str, Any] = {
            "_group_key": group_key,
            "hotel_id": first.get("hotel_id", ""),
            "aspect": ALL_ASPECTS_SUMMARY_KEY,
            "overall_aspect_summary": "",
        }
        for sentiment in SENTIMENTS:
            count = sum(int(item.get(f"{sentiment}_count", 0) or 0) for item in detail_rows)
            weighted_confidence = sum(
                int(item.get(f"{sentiment}_count", 0) or 0)
                * float(item.get(f"{sentiment}_avg_confidence", 0.0) or 0.0)
                for item in detail_rows
            )
            row[f"{sentiment}_count"] = count
            row[f"{sentiment}_avg_confidence"] = round(weighted_confidence / count, 6) if count else 0.0
            row[f"{sentiment}_cluster_count"] = sum(
                int(item.get(f"{sentiment}_cluster_count", 0) or 0) for item in detail_rows
            )
            row[f"{sentiment}_clusters"] = "[]"
            row[f"{sentiment}_summary"] = build_all_aspects_sentiment_summary(detail_rows, sentiment)
        all_aspects_by_group[group_key] = row
        all_aspects_items.append(
            {
                "hotel_id": first.get("hotel_id", ""),
                "detail_rows": detail_rows,
                "summary_language": getattr(args, "summary_language", "vi"),
            }
        )

    if all_aspects_items:
        all_aspects_summaries = summarize_all_aspects_items_in_batches(all_aspects_items, args)
        for group_key, summary in zip(group_order, all_aspects_summaries):
            all_aspects_by_group[group_key]["overall_aspect_summary"] = clean_overall_summary_text(summary, 1800)

    output_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        output_rows.append(row)
        current_group = str(row.get("_group_key", ""))
        next_group = str(rows[idx + 1].get("_group_key", "")) if idx + 1 < len(rows) else ""
        if current_group != next_group and current_group in all_aspects_by_group:
            output_rows.append(all_aspects_by_group[current_group])
    return output_rows


def summarize_all_aspects_items_in_batches(
    all_aspects_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[str]:
    if not all_aspects_items:
        return []

    batch_size = max(1, int(getattr(args, "all_aspects_batch_size", 8) or 8))
    workers = max(1, int(getattr(args, "summary_workers", getattr(args, "final_summary_workers", 1)) or 1))
    progress_every = max(1, int(getattr(args, "summary_progress_every_batches", 100) or 100))
    batches = [
        all_aspects_items[start : start + batch_size]
        for start in range(0, len(all_aspects_items), batch_size)
    ]
    results_by_batch: list[list[str]] = [[] for _ in batches]

    print(
        f"[all-aspects-summary] running {len(batches):,} summary batches with {workers:,} workers",
        flush=True,
    )

    def summarize_one(batch: list[dict[str, Any]]) -> list[str]:
        return QwenAllAspectsOverallWriter(args).summarize_batch(batch)

    if workers == 1 or len(batches) <= 1:
        writer = QwenAllAspectsOverallWriter(args)
        for idx, batch in enumerate(batches):
            try:
                results_by_batch[idx] = summarize_batch_with_cache(
                    args,
                    "all_aspects_summary",
                    ALL_ASPECTS_SUMMARY_PROMPT_VERSION,
                    batch,
                    writer.summarize_batch,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] all-aspects summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                    file=sys.stderr,
                )
                results_by_batch[idx] = [
                    build_all_aspects_overall_summary(
                        item.get("detail_rows", []),
                        getattr(args, "summary_language", "vi"),
                    )
                    for item in batch
                ]
            completed = idx + 1
            if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                print(f"[all-aspects-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
    else:
        max_pending = min(len(batches), max(workers * 2, workers))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Any, tuple[int, list[dict[str, Any]]]] = {}
            next_idx = 0
            completed = 0

            def submit_next() -> None:
                nonlocal next_idx
                if next_idx >= len(batches):
                    return
                batch = batches[next_idx]
                futures[executor.submit(summarize_one, batch)] = (next_idx, batch)
                next_idx += 1

            for _ in range(max_pending):
                submit_next()

            while futures:
                for future in as_completed(futures):
                    idx, batch = futures.pop(future)
                    break
                try:
                    results_by_batch[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[WARN] all-aspects summary batch {idx + 1} failed; using local summaries. Error: {exc}",
                        file=sys.stderr,
                    )
                    results_by_batch[idx] = [
                        build_all_aspects_overall_summary(
                            item.get("detail_rows", []),
                            getattr(args, "summary_language", "vi"),
                        )
                        for item in batch
                    ]
                completed += 1
                if completed == 1 or completed % progress_every == 0 or completed == len(batches):
                    print(f"[all-aspects-summary] completed {completed:,}/{len(batches):,} batches", flush=True)
                submit_next()

    return [summary for batch_summaries in results_by_batch for summary in batch_summaries]


def merge_final_summary_metrics_into_rows(
    final_summary_rows: list[dict[str, Any]],
    final_metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return final_summary_rows


def strip_final_summary_internal_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.pop("_group_key", None)
    return rows


def build_aspect_summary_report_rows(final_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hotel_id": row.get("hotel_id", ""),
            "aspect": row.get("aspect", ""),
            "aspect_summary": row.get("overall_aspect_summary", ""),
        }
        for row in final_summary_rows
    ]


def write_aspect_summary_json(path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_type": "reader_facing_aspect_summary",
        "stats": stats,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_final_row_metrics(
    final_summary: FinalSummaryAggregate,
    final_summary_rows: list[dict[str, Any]],
    bertscore_language: str = "en",
    enable_bertscore: bool = True,
) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    total = len(final_summary_rows)
    completed = 0
    started_at = time.time()
    if enable_bertscore:
        print(f"[final-summary-metrics] computing {total:,} rows with BERTScore", flush=True)
    for row in final_summary_rows:
        group_key = str(row.get("_group_key", ""))
        aspect = str(row.get("aspect", ""))
        if group_key not in final_summary.buckets or aspect not in ASPECT_NAMES:
            continue
        combined_bucket = FinalSentimentBucket()
        for sentiment in SENTIMENTS:
            bucket = final_summary.buckets[group_key][aspect][sentiment]
            combined_bucket.count += bucket.count
            combined_bucket.confidence_sum += bucket.confidence_sum
            combined_bucket.reference_unigrams.update(bucket.reference_unigrams)
            combined_bucket.reference_bigrams.update(bucket.reference_bigrams)
            combined_bucket.reference_token_count += bucket.reference_token_count
            combined_bucket.reference_sentence_count += bucket.reference_sentence_count
            combined_bucket.reference_token_sequence.extend(bucket.reference_token_sequence)
            combined_bucket.reference_texts.extend(bucket.reference_texts)
        summary_parts = [clean_text(row.get(f"{sentiment}_summary", "")) for sentiment in SENTIMENTS]
        summary_text = " ".join(part for part in summary_parts if part)
        metrics = combined_bucket.compute_summary_metrics(summary_text, bertscore_language, enable_bertscore)
        metric_row = {
            "hotel_id": str(row.get("hotel_id", "")),
            "aspect": aspect,
            "reference_sentence_count": metrics.reference_sentence_count,
            "reference_token_count": metrics.reference_token_count,
            "summary_token_count": metrics.summary_token_count,
            "rouge1_recall": round(metrics.rouge1_recall, 6),
            "rouge2_recall": round(metrics.rouge2_recall, 6),
            "rouge_l_recall": round(metrics.rouge_l_recall, 6),
            "coverage_score": round(metrics.coverage_score, 6),
            "bertscore_precision": round(metrics.bertscore_precision, 6),
            "bertscore_recall": round(metrics.bertscore_recall, 6),
            "bertscore_f1": round(metrics.bertscore_f1, 6),
            "bertscore_available": metrics.bertscore_available,
        }
        metric_rows.append(metric_row)
        completed += 1
        if enable_bertscore and (
            completed == 1 or completed % 100 == 0 or completed == total
        ):
            elapsed = time.time() - started_at
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / rate if rate > 0 else 0.0
            print(
                f"[final-summary-metrics] completed {completed:,}/{total:,} "
                f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m",
                flush=True,
            )
    return metric_rows


def write_final_summary_json(path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_type": "final_aspect_sentiment_summary",
        "notes": {
            "scope": (
                "Each regular row summarizes all review segments from one hotel_id and one aspect. "
                f"Rows with aspect={ALL_ASPECTS_SUMMARY_KEY} summarize all aspects for that hotel_id."
            ),
            "sentiment_paragraphs": "positive_summary, neutral_summary, and negative_summary are final report paragraphs.",
        },
        "stats": stats,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cluster_evidence_rows(final_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in final_summary_rows:
        aspect = str(row.get("aspect", ""))
        if aspect not in ASPECT_NAMES:
            continue
        for sentiment in SENTIMENTS:
            clusters_raw = row.get(f"{sentiment}_clusters", "[]")
            try:
                clusters_value = json.loads(clusters_raw) if isinstance(clusters_raw, str) else clusters_raw
            except Exception:
                clusters_value = []
            if not isinstance(clusters_value, list):
                continue
            for cluster_index, cluster in enumerate((item for item in clusters_value if isinstance(item, dict)), start=1):
                descriptors = unique_preserve_order(
                    [str(value) for value in cluster.get("descriptors", []) if clean_text(value)],
                    18,
                )
                samples = unique_preserve_order(
                    [clean_text(sample).rstrip(".") for sample in cluster.get("samples", []) if clean_text(sample)],
                    5,
                )
                canonical_code, canonical_label = canonical_cluster_fields(
                    aspect,
                    sentiment,
                    " ".join(samples),
                    clean_text(cluster.get("code", "")),
                    clean_text(cluster.get("measurement_scale", "")) or clean_text(cluster.get("label", "")),
                    descriptors,
                )
                evidence_text = summarize_cluster_evidence([cluster], aspect, max_clusters=1) or summarize_cluster_themes(
                    [cluster],
                    aspect,
                )
                rows.append(
                    {
                        "hotel_id": clean_text(row.get("hotel_id", "")),
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "cluster_index": cluster_index,
                        "cluster_label": canonical_label,
                        "cluster_code": canonical_code,
                        "measurement_scale": canonical_label,
                        "count": int(cluster.get("count", 0) or 0),
                        "avg_confidence": round(float(cluster.get("avg_confidence", 0.0) or 0.0), 6),
                        "descriptors": json.dumps(descriptors, ensure_ascii=False),
                        "samples": json.dumps(samples, ensure_ascii=False),
                        "evidence_text": evidence_text,
                    }
                )
    return rows


def processed_row_evidence_text(row: dict[str, Any]) -> str:
    return (
        clean_text(row.get("summary_text", ""))
        or clean_text(row.get("normalized_text_vi", ""))
        or clean_text(row.get("segment_text", ""))
        or clean_text(row.get("aspect_segment_text", ""))
        or clean_text(row.get("preseg_sentence", ""))
        or clean_text(row.get("processed_sentence", ""))
        or clean_text(row.get("shortened_sentence", ""))
        or clean_text(row.get("source_text", ""))
    )


def build_cluster_evidence_rows_from_processed_sentences(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "confidence_sum": 0.0,
            "descriptors": [],
            "evidence_texts": [],
            "segment_texts": [],
            "source_texts": [],
            "review_indexes": [],
            "source_files": [],
        }
    )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_id = clean_text(row.get("entity_id", ""))
            data_source = clean_text(row.get("data_source", ""))
            hotel_id = clean_text(row.get("hotel_id", ""))
            aspect = canonicalize_aspect_label(row.get("aspect", ""))
            if aspect not in ASPECT_NAMES:
                continue
            sentiment = normalize_sentiment(row.get("sentiment", ""))
            cluster_code = clean_text(row.get("cluster_code", ""))
            cluster_label = clean_text(row.get("cluster_label", ""))
            evidence = processed_row_evidence_text(row)
            descriptors = unique_preserve_order(
                [
                    clean_text(item)
                    for item in parse_json_list_field(row.get("cluster_descriptors", "[]"))
                    if clean_text(item)
                ],
                18,
            )
            cluster_code, cluster_label = canonical_cluster_fields(
                aspect,
                sentiment,
                evidence,
                cluster_code,
                cluster_label,
                descriptors,
            )
            key = (entity_id, data_source, hotel_id, aspect, sentiment, cluster_code, cluster_label)
            bucket = grouped[key]
            bucket["count"] += 1
            bucket["confidence_sum"] += clamp_confidence(
                row.get("cluster_assignment_confidence", row.get("sentiment_confidence", 0.45))
            )

            bucket["descriptors"].extend(descriptors)
            if evidence:
                bucket["evidence_texts"].append(evidence)
            segment_text = clean_text(row.get("segment_text", ""))
            if segment_text:
                bucket["segment_texts"].append(segment_text)
            source_text = clean_text(row.get("source_text", ""))
            if source_text:
                bucket["source_texts"].append(source_text)
            review_index = clean_text(row.get("review_index", ""))
            if review_index:
                bucket["review_indexes"].append(review_index)
            source_file = clean_text(row.get("source_file", ""))
            if source_file:
                bucket["source_files"].append(source_file)

    rows: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        entity_id, data_source, hotel_id, aspect, sentiment, cluster_code, cluster_label = key
        descriptors = unique_preserve_order(bucket["descriptors"], None)
        evidence_texts = unique_preserve_order(bucket["evidence_texts"], None)
        segment_texts = unique_preserve_order(bucket["segment_texts"], None)
        source_texts = unique_preserve_order(bucket["source_texts"], None)
        review_indexes = unique_preserve_order(bucket["review_indexes"], None)
        source_files = unique_preserve_order(bucket["source_files"], None)
        count = int(bucket["count"])
        rows.append(
            {
                "entity_id": entity_id,
                "data_source": data_source,
                "hotel_id": hotel_id,
                "aspect": aspect,
                "sentiment": sentiment,
                "cluster_code": cluster_code,
                "measurement_scale": cluster_label,
                "cluster_label": cluster_label,
                "count": count,
                "avg_confidence": round(bucket["confidence_sum"] / count, 6) if count else 0.0,
                "descriptor_count": len(descriptors),
                "descriptors_json": json.dumps(descriptors, ensure_ascii=False),
                "evidence_count": len(evidence_texts),
                "evidence_texts_json": json.dumps(evidence_texts, ensure_ascii=False),
                "segment_texts_json": json.dumps(segment_texts, ensure_ascii=False),
                "source_texts_json": json.dumps(source_texts, ensure_ascii=False),
                "review_indexes_json": json.dumps(review_indexes, ensure_ascii=False),
                "source_files_json": json.dumps(source_files, ensure_ascii=False),
            }
        )
    rows.sort(key=lambda item: (item["hotel_id"], item["aspect"], item["sentiment"], -int(item["count"])))
    return rows


def write_cluster_evidence_json(path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary_type": "cluster_evidence",
        "notes": {
            "scope": (
                "Each row captures one extracted cluster for one hotel/entity, aspect, and sentiment. "
                "Rows built from processed sentences include full deduplicated evidence_texts_json."
            ),
            "evidence_texts_json": "Full deduplicated source evidence for this cluster when available.",
            "evidence_text": "Human-readable cluster evidence synthesized from the cluster label, descriptors, and samples when using final-summary payloads.",
        },
        "stats": stats,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_hotel_profiles(
    hotels: dict[str, HotelAggregate],
    args: argparse.Namespace,
    cache: ClassificationCache,
) -> ProfileDecisions:
    profiles = [
        build_profile_item(entity_id, hotel, args)
        for entity_id, hotel in sorted(hotels.items(), key=lambda kv: (kv[1].data_source, kv[1].hotel_id))
    ]
    profile_hashes = {profile["entity_id"]: profile_hash(profile) for profile in profiles}
    cached = cache.get_profiles([profile["entity_id"] for profile in profiles])
    decisions: ProfileDecisions = {}
    pending: list[dict[str, Any]] = []
    for profile in profiles:
        entity_id = profile["entity_id"]
        cached_item = cached.get(entity_id)
        if cached_item and cached_item[0] == profile_hashes[entity_id]:
            decisions[entity_id] = deserialize_profile_decision(cached_item[1])
        else:
            pending.append(profile)

    if not pending:
        print(f"[profile-qwen] all {len(profiles):,} profiles loaded from cache", flush=True)
        return decisions

    if args.profile_qwen_filter == "tied":
        skipped = 0
        filtered_pending = []
        for profile in pending:
            has_tie = any(
                bool(profile["aspects"].get(aspect, {}).get("sentiment_tie"))
                and int(profile["aspects"].get(aspect, {}).get("count", 0)) > 0
                for aspect in ASPECT_NAMES
            )
            if has_tie:
                filtered_pending.append(profile)
            else:
                raw_decision = local_profile_decision(profile)
                decisions[profile["entity_id"]] = deserialize_profile_decision(raw_decision)
                skipped += 1
        pending = filtered_pending
        print(
            f"[profile-qwen] filter=tied skipped_local={skipped:,} pending_qwen={len(pending):,}",
            flush=True,
        )
        if not pending:
            return decisions

    validator = QwenProfileValidator(args)
    batch_size = max(1, int(args.profile_batch_size))
    total_batches = (len(pending) + batch_size - 1) // batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_num = (start // batch_size) + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
            print(
                f"[profile-qwen] batch {batch_num}/{total_batches} profiles={len(batch)}",
                flush=True,
            )
        raw_decisions = validator.validate_batch(batch)
        rows_to_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        for profile in batch:
            entity_id = profile["entity_id"]
            raw_decision = raw_decisions.get(entity_id, local_profile_decision(profile))
            decisions[entity_id] = deserialize_profile_decision(raw_decision)
            rows_to_cache[entity_id] = (profile_hashes[entity_id], raw_decision)
        cache.set_profiles(rows_to_cache)
    return decisions


def resolve_aspect_output(
    entity_id: str,
    aspect: str,
    hotel: HotelAggregate,
    profile_decisions: ProfileDecisions,
) -> tuple[str, str]:
    agg = hotel.aspects[aspect]
    total = aspect_total(agg)
    local_sentiment = agg.final_sentiment() if total else ""
    decision = profile_decisions.get(entity_id, {}).get(aspect)
    if total and decision and aspect_has_sentiment_tie(agg) and decision.sentiment:
        final_sentiment = decision.sentiment
    else:
        final_sentiment = local_sentiment
    representative = agg.final_representative()
    if total and decision and decision.representative_sentence:
        representative = decision.representative_sentence
    return final_sentiment, representative


def build_output_rows(
    hotels: dict[str, HotelAggregate],
    profile_decisions: ProfileDecisions | None = None,
) -> list[dict[str, Any]]:
    rows = []
    profile_decisions = profile_decisions or {}
    for entity_id, hotel in sorted(hotels.items(), key=lambda kv: (kv[1].data_source, kv[1].hotel_id)):
        row: dict[str, Any] = {
            "hotel_id": hotel.hotel_id,
            "review_count": len(hotel.review_indexes),
            "sentence_count": hotel.sentence_count,
            "processed_sentence_count": hotel.sentence_count,
            "cluster_llm_sentence_count": 0,
            "cluster_rule_fallback_sentence_count": 0,
            "cluster_unknown_source_sentence_count": 0,
            "cluster_assignment_source_counts": "",
        }
        hotel_cluster_source_counts: Counter[str] = Counter()
        for aspect in ASPECT_NAMES:
            agg = hotel.aspects[aspect]
            final_sentiment, representative = resolve_aspect_output(entity_id, aspect, hotel, profile_decisions)
            positive_sentence_count = int(agg.counts.get("positive", 0))
            negative_sentence_count = int(agg.counts.get("negative", 0))
            neutral_sentence_count = int(agg.counts.get("neutral", 0))
            aspect_sentence_count = positive_sentence_count + negative_sentence_count + neutral_sentence_count
            source_counts = getattr(agg, "cluster_assignment_source_counts", Counter()) or Counter()
            source_counts = Counter(
                {normalize_cluster_assignment_source(source): int(count) for source, count in source_counts.items()}
            )
            hotel_cluster_source_counts.update(source_counts)
            row[f"{aspect}_sentiment"] = final_sentiment
            row[f"{aspect}_count"] = aspect_sentence_count
            row[f"{aspect}_positive_count"] = positive_sentence_count
            row[f"{aspect}_negative_count"] = negative_sentence_count
            row[f"{aspect}_neutral_count"] = neutral_sentence_count
            row[f"{aspect}_sentence_count"] = aspect_sentence_count
            row[f"{aspect}_positive_sentence_count"] = positive_sentence_count
            row[f"{aspect}_negative_sentence_count"] = negative_sentence_count
            row[f"{aspect}_neutral_sentence_count"] = neutral_sentence_count
            row[f"{aspect}_sentiment_sentence_counts"] = (
                f"positive={positive_sentence_count}; "
                f"negative={negative_sentence_count}; "
                f"neutral={neutral_sentence_count}; "
                f"total={aspect_sentence_count}"
            )
            row[f"{aspect}_representative_sentence"] = representative
            row[f"{aspect}_summary_vi"] = agg.summary_vi
            row[f"{aspect}_summary_en"] = agg.summary_en
        (
            row["cluster_llm_sentence_count"],
            row["cluster_rule_fallback_sentence_count"],
            row["cluster_unknown_source_sentence_count"],
        ) = cluster_assignment_source_group_counts(hotel_cluster_source_counts, hotel.sentence_count)
        row["cluster_assignment_source_counts"] = format_cluster_assignment_source_counts(
            hotel_cluster_source_counts,
            hotel.sentence_count,
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_debug_json(
    path: Path,
    hotels: dict[str, HotelAggregate],
    stats: dict[str, Any],
    profile_decisions: ProfileDecisions | None = None,
) -> None:
    if not str(path):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stats": stats,
        "hotels": {},
    }
    profile_decisions = profile_decisions or {}
    for entity_id, hotel in sorted(hotels.items(), key=lambda kv: kv[0]):
        payload["hotels"][entity_id] = {
            "data_source": hotel.data_source,
            "hotel_id": hotel.hotel_id,
            "review_count": len(hotel.review_indexes),
            "sentence_count": hotel.sentence_count,
            "aspects": {
                aspect: {
                    "counts": dict(agg.counts),
                    "confidence_sums": dict(agg.confidence_sums),
                    "final_sentiment": agg.final_sentiment() if sum(agg.counts.values()) else "",
                    "representative_sentence": agg.final_representative(),
                    "summary_vi": agg.summary_vi,
                    "summary_en": agg.summary_en,
                    "reference_sentence_count": agg.reference_sentence_count,
                    "reference_token_count": agg.reference_token_count,
                    "qwen_profile_decision": (
                        {
                            "sentiment": profile_decisions[entity_id][aspect].sentiment,
                            "representative_sentence": profile_decisions[entity_id][aspect].representative_sentence,
                            "note": profile_decisions[entity_id][aspect].note,
                        }
                        if entity_id in profile_decisions and aspect in profile_decisions[entity_id]
                        else None
                    ),
                    "samples": agg.samples,
                }
                for aspect, agg in hotel.aspects.items()
            },
        }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_summary_metric_rows(
    hotels: dict[str, HotelAggregate],
    profile_decisions: ProfileDecisions | None = None,
    summary_language: str = "vi",
    bertscore_language: str = "en",
    enable_bertscore: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_decisions = profile_decisions or {}
    total = sum(
        1
        for hotel in hotels.values()
        for aspect in ASPECT_NAMES
        if aspect_total(hotel.aspects[aspect]) > 0
    )
    completed = 0
    started_at = time.time()
    if enable_bertscore:
        print(f"[summary-metrics] computing {total:,} rows with BERTScore", flush=True)
    for entity_id, hotel in sorted(hotels.items(), key=lambda kv: (kv[1].data_source, kv[1].hotel_id)):
        for aspect in ASPECT_NAMES:
            agg = hotel.aspects[aspect]
            if aspect_total(agg) <= 0:
                continue
            final_sentiment, representative = resolve_aspect_output(entity_id, aspect, hotel, profile_decisions)
            if summary_language == "vi":
                summary_text = agg.summary_vi or agg.summary_en or representative
            else:
                summary_text = agg.summary_en or agg.summary_vi or representative
            metrics = agg.compute_summary_metrics(summary_text, bertscore_language, enable_bertscore)
            rows.append(
                {
                    "hotel_id": hotel.hotel_id,
                    "aspect": aspect,
                    "final_sentiment": final_sentiment,
                    "summary_text": summary_text,
                    "reference_sentence_count": metrics.reference_sentence_count,
                    "reference_token_count": metrics.reference_token_count,
                    "summary_token_count": metrics.summary_token_count,
                    "rouge1_recall": round(metrics.rouge1_recall, 6),
                    "rouge2_recall": round(metrics.rouge2_recall, 6),
                    "rouge_l_recall": round(metrics.rouge_l_recall, 6),
                    "coverage_score": round(metrics.coverage_score, 6),
                    "bertscore_precision": round(metrics.bertscore_precision, 6),
                    "bertscore_recall": round(metrics.bertscore_recall, 6),
                    "bertscore_f1": round(metrics.bertscore_f1, 6),
                    "bertscore_available": metrics.bertscore_available,
                }
            )
            completed += 1
            if enable_bertscore and (
                completed == 1 or completed % 100 == 0 or completed == total
            ):
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (total - completed) / rate if rate > 0 else 0.0
                print(
                    f"[summary-metrics] completed {completed:,}/{total:,} "
                    f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m",
                    flush=True,
                )
    return rows


def build_final_summary_metric_rows(
    final_summary: FinalSummaryAggregate,
    final_summary_rows: list[dict[str, Any]],
    bertscore_language: str = "en",
    enable_bertscore: bool = True,
) -> list[dict[str, Any]]:
    return compute_final_row_metrics(
        final_summary,
        final_summary_rows,
        bertscore_language,
        enable_bertscore,
    )


def aggregate_metric_values(rows: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
    values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 6),
        "median": round(float(statistics.median(values)), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def write_summary_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_metrics_json(path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aspect[str(row["aspect"])].append(row)

    metric_names = [
        "rouge1_recall",
        "rouge2_recall",
        "rouge_l_recall",
        "coverage_score",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    payload = {
        "metric_type": "aspect_summary_metrics",
        "notes": {
            "scope": "Metrics evaluate each generated summary against the source sentences already assigned to the same hotel/aspect.",
            "not_extraction_benchmark": (
                "These are summary-quality metrics, not multi-aspect extraction metrics. "
                "They assume the upstream aspect assignment is already fixed."
            ),
            "rouge1_recall": "Unigram recall of summary tokens against the hotel/aspect source sentence corpus.",
            "rouge2_recall": "Bigram recall of summary tokens against the hotel/aspect source sentence corpus.",
            "rouge_l_recall": "Longest common subsequence recall between summary and concatenated hotel/aspect source sentences.",
            "coverage_score": "Share of unique source tokens covered by the summary.",
            "bertscore": "Best BERTScore between the summary and individual source sentences. Values are 0 when bert-score is not installed or cannot run.",
        },
        "stats": stats,
        "overall": {
            "rows": len(rows),
            **{metric_name: aggregate_metric_values(rows, metric_name) for metric_name in metric_names},
        },
        "by_aspect": {
            aspect: {
                "rows": len(aspect_rows),
                **{metric_name: aggregate_metric_values(aspect_rows, metric_name) for metric_name in metric_names},
            }
            for aspect, aspect_rows in sorted(by_aspect.items())
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_summary_metrics_json(path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aspect[str(row["aspect"])].append(row)

    metric_names = [
        "rouge1_recall",
        "rouge2_recall",
        "rouge_l_recall",
        "coverage_score",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    payload = {
        "metric_type": "final_aspect_sentiment_summary_metrics",
        "notes": {
            "scope": (
                "Metrics evaluate each generated final summary paragraph against source segments already "
                "assigned to the same hotel_id, aspect, and sentiment bucket."
            ),
            "not_extraction_benchmark": (
                "These metrics do not test whether the pipeline found every gold ABSA quad. "
                "For that, group gold and predictions by review/sentence and compare sets of aspect-sentiment units."
            ),
            "rouge1_recall": "Unigram recall of summary tokens against the hotel/aspect/sentiment source segment corpus.",
            "rouge2_recall": "Bigram recall of summary tokens against the hotel/aspect/sentiment source segment corpus.",
            "rouge_l_recall": "Longest common subsequence recall between summary and concatenated hotel/aspect/sentiment source segments.",
            "coverage_score": "Share of unique source tokens covered by the summary.",
            "bertscore": "Best BERTScore between the summary and individual source segments. Values are 0 when bert-score is not installed or cannot run.",
        },
        "stats": stats,
        "overall": {
            "rows": len(rows),
            **{metric_name: aggregate_metric_values(rows, metric_name) for metric_name in metric_names},
        },
        "by_aspect": {
            aspect: {
                "rows": len(aspect_rows),
                **{metric_name: aggregate_metric_values(aspect_rows, metric_name) for metric_name in metric_names},
            }
            for aspect, aspect_rows in sorted(by_aspect.items())
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    inputs = [Path(p) for p in args.inputs]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    eligible_entity_ids, min_review_filter_stats = build_min_review_filter(inputs, args.min_reviews_per_hotel)
    setattr(args, "eligible_entity_ids", eligible_entity_ids)
    if min_review_filter_stats:
        print(
            "[filter] min_reviews_per_hotel="
            f"{min_review_filter_stats['min_reviews_per_hotel']:,} "
            f"eligible_hotels={min_review_filter_stats['eligible_hotels']:,}/"
            f"{min_review_filter_stats['total_hotels_before_filter']:,} "
            f"eligible_rows={min_review_filter_stats['eligible_review_rows']:,}",
            flush=True,
        )

    resume_payload: dict[str, Any] | None = None
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    if args.resume_from_checkpoint:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
        resume_payload = PipelineCheckpointManager.load(checkpoint_path)
        saved_inputs = resume_payload.get("inputs", [])
        current_inputs = [str(path.resolve()) for path in inputs]
        if saved_inputs and saved_inputs != current_inputs:
            print(
                "[WARN] Checkpoint inputs differ from the current run. Resume will use the saved aggregates as-is.",
                file=sys.stderr,
            )
        saved_strategy = resume_payload.get("strategy", "")
        if saved_strategy and saved_strategy != args.strategy:
            print(
                f"[WARN] Checkpoint strategy={saved_strategy} differs from current strategy={args.strategy}.",
                file=sys.stderr,
            )
        saved_min_reviews = int(dict(resume_payload.get("stats", {})).get("min_reviews_per_hotel", 0) or 0)
        if saved_min_reviews != args.min_reviews_per_hotel:
            print(
                "[WARN] Checkpoint min_reviews_per_hotel="
                f"{saved_min_reviews} differs from current={args.min_reviews_per_hotel}. "
                "Use a fresh checkpoint when changing hotel filters.",
                file=sys.stderr,
            )
        saved_cluster_assignment = str(dict(resume_payload.get("stats", {})).get("cluster_assignment", "rule"))
        if saved_cluster_assignment != args.cluster_assignment:
            print(
                "[WARN] Checkpoint cluster_assignment="
                f"{saved_cluster_assignment} differs from current={args.cluster_assignment}. "
                "Use a fresh checkpoint when changing cluster assignment mode.",
                file=sys.stderr,
            )

    cache = ClassificationCache(Path(args.cache_db))
    setattr(args, "_summary_cache", cache)
    checkpoint = PipelineCheckpointManager(args.checkpoint_path, args.checkpoint_every_chunks, resume_payload)
    hotels: dict[str, HotelAggregate] = dict((resume_payload or {}).get("hotels", {}))
    stats: dict[str, Any] = {
        "inputs": [str(path) for path in inputs],
        "strategy": args.strategy,
        "qwen_base_url": args.qwen_base_url,
        "qwen_model": args.qwen_model,
        "skip_qwen": bool(args.skip_qwen),
        "pre_segmentation": args.pre_segmentation,
        "cluster_assignment": args.cluster_assignment,
        "cluster_assignment_min_confidence": args.cluster_assignment_min_confidence,
        "semantic_min_source_precision": args.semantic_min_source_precision,
        "final_summary_cluster_threshold": args.final_summary_cluster_threshold,
        "final_summary_max_clusters": args.final_summary_max_clusters,
        "final_summary_cluster_samples": args.final_summary_cluster_samples,
        "final_summary_cluster_descriptors": args.final_summary_cluster_descriptors,
        "min_reviews_per_hotel": args.min_reviews_per_hotel,
        "min_review_filter": min_review_filter_stats,
        "aspect_output_dir": str(Path(args.aspect_output_dir).resolve()),
        "aspect_output_enabled": not bool(args.disable_aspect_output),
        "processed_sentences_csv": (
            str(Path(args.processed_sentences_csv).resolve()) if args.processed_sentences_csv else ""
        ),
        "sentiment_language": args.sentiment_language,
        "summary_language": args.summary_language,
        "bertscore_language": args.bertscore_language,
        "bertscore_enabled": not bool(args.disable_bertscore),
        "summary_metrics_enabled": bool(args.summary_metrics_csv or args.summary_metrics_json),
        "final_summary_metrics_enabled": bool(args.final_summary_metrics_csv or args.final_summary_metrics_json),
        "cluster_evidence_enabled": bool(args.cluster_evidence_csv or args.cluster_evidence_json),
        "preseg_metrics_enabled": bool(
            args.strategy == "sentence-qwen"
            and args.pre_segmentation == "semantic-qwen"
            and (args.preseg_metrics_csv or args.preseg_metrics_json)
        ),
        "preseg_metrics_csv": str(Path(args.preseg_metrics_csv).resolve()) if args.preseg_metrics_csv else "",
        "preseg_metrics_json": str(Path(args.preseg_metrics_json).resolve()) if args.preseg_metrics_json else "",
        "information_coverage_model": args.information_coverage_model,
        "final_summary_only": bool(args.final_summary_only),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else "",
        "checkpoint_enabled": bool(checkpoint.enabled),
        "resumed_from_checkpoint": bool(args.resume_from_checkpoint),
        "final_summary_csv": str(Path(args.final_summary_csv).resolve()) if args.final_summary_csv else "",
        "final_summary_json": str(Path(args.final_summary_json).resolve()) if args.final_summary_json else "",
        "final_summary_metrics_csv": (
            str(Path(args.final_summary_metrics_csv).resolve()) if args.final_summary_metrics_csv else ""
        ),
        "final_summary_metrics_json": (
            str(Path(args.final_summary_metrics_json).resolve()) if args.final_summary_metrics_json else ""
        ),
        "cluster_evidence_csv": str(Path(args.cluster_evidence_csv).resolve()) if args.cluster_evidence_csv else "",
        "cluster_evidence_json": str(Path(args.cluster_evidence_json).resolve()) if args.cluster_evidence_json else "",
        "files": {},
    }
    if resume_payload:
        saved_stats = dict(resume_payload.get("stats", {}))
        if isinstance(saved_stats.get("files"), dict):
            stats["files"] = dict(saved_stats["files"])
    profile_decisions: ProfileDecisions = dict((resume_payload or {}).get("profile_decisions", {}))
    aspect_writer = (
        AspectOutputWriter(Path(args.aspect_output_dir), reset_existing=not args.resume_from_checkpoint)
        if args.strategy == "sentence-qwen" and not args.disable_aspect_output
        else None
    )
    processed_sentence_writer = (
        ProcessedSentenceOutputWriter(Path(args.processed_sentences_csv), reset_existing=not args.resume_from_checkpoint)
        if args.strategy == "sentence-qwen" and args.processed_sentences_csv
        else None
    )
    preseg_metric_writer = (
        PreSegmentationMetricWriter(
            Path(args.preseg_metrics_csv) if args.preseg_metrics_csv else None,
            Path(args.preseg_metrics_json) if args.preseg_metrics_json else None,
            InformationCoverageScorer(
                args.information_coverage_model,
                args.information_coverage_batch_size,
            ),
            reset_existing=not args.resume_from_checkpoint,
        )
        if (
            args.strategy == "sentence-qwen"
            and args.pre_segmentation == "semantic-qwen"
            and (args.preseg_metrics_csv or args.preseg_metrics_json)
        )
        else None
    )
    final_summary: FinalSummaryAggregate | None = None
    if (
        args.final_summary_csv
        or args.final_summary_json
        or args.final_summary_metrics_csv
        or args.final_summary_metrics_json
        or args.cluster_evidence_csv
        or args.cluster_evidence_json
        or args.aspect_summary_csv
        or args.aspect_summary_json
    ):
        saved_final_summary = (resume_payload or {}).get("final_summary")
        if isinstance(saved_final_summary, FinalSummaryAggregate):
            final_summary = saved_final_summary
            final_summary.enable_clusters = not getattr(args, "disable_final_summary_clusters", False)
            final_summary.configure_clustering(
                args.final_summary_cluster_threshold,
                args.final_summary_cluster_samples,
            )
        else:
            final_summary = FinalSummaryAggregate(
                args.final_summary_samples_per_sentiment,
                args.final_summary_sample_chars,
                args.final_summary_cluster_threshold,
                args.final_summary_cluster_samples,
                enable_clusters=not getattr(args, "disable_final_summary_clusters", False),
            )

    try:
        if args.strategy == "sentence-qwen":
            pre_segmenter = QwenSemanticPreSegmenter(args) if args.pre_segmentation == "semantic-qwen" else None
            extractor = QwenAspectSegmenter(args)
            sentiment_classifier = QwenAspectSentimentClassifier(args)
            cluster_assigner = (
                QwenClusterAssigner(args)
                if args.cluster_assignment == "llm" and not args.skip_qwen
                else None
            )
            for path in inputs:
                print(f"[start] processing {path}", flush=True)
                process_file_multistage_qwen(
                    path,
                    args,
                    cache,
                    pre_segmenter,
                    extractor,
                    sentiment_classifier,
                    cluster_assigner,
                    hotels,
                    stats["files"],
                    aspect_writer,
                    processed_sentence_writer,
                    final_summary,
                    preseg_metric_writer,
                    checkpoint,
                    inputs,
                    profile_decisions,
                )
            if not args.final_summary_only:
                generate_aspect_summaries(hotels, args, final_summary)
                checkpoint.maybe_save(
                    inputs=inputs,
                    strategy=args.strategy,
                    hotels=hotels,
                    stats=stats,
                    final_summary=final_summary,
                    profile_decisions=profile_decisions,
                    force=True,
                )
        else:
            for path in inputs:
                print(f"[start-local] processing {path}", flush=True)
                process_file_local(path, args, hotels, stats["files"], checkpoint, inputs, profile_decisions)
            if args.strategy == "profile-qwen":
                profile_decisions = validate_hotel_profiles(hotels, args, cache)
                checkpoint.maybe_save(
                    inputs=inputs,
                    strategy=args.strategy,
                    hotels=hotels,
                    stats=stats,
                    final_summary=final_summary,
                    profile_decisions=profile_decisions,
                    force=True,
                )

        if not args.final_summary_only:
            rows = build_output_rows(hotels, profile_decisions)
            write_csv(Path(args.output_csv), rows)
            if args.debug_json:
                write_debug_json(Path(args.debug_json), hotels, stats, profile_decisions)
            if args.summary_metrics_csv or args.summary_metrics_json:
                metric_rows = build_summary_metric_rows(
                    hotels,
                    profile_decisions,
                    args.summary_language,
                    args.bertscore_language,
                    not args.disable_bertscore,
                )
                if args.summary_metrics_csv:
                    write_summary_metrics_csv(Path(args.summary_metrics_csv), metric_rows)
                if args.summary_metrics_json:
                    write_summary_metrics_json(Path(args.summary_metrics_json), metric_rows, stats)
        if final_summary is not None:
            final_summary_rows = build_final_summary_rows(final_summary, args)
            if args.cluster_evidence_csv or args.cluster_evidence_json:
                processed_path = Path(args.processed_sentences_csv) if args.processed_sentences_csv else None
                if processed_path and processed_path.exists():
                    cluster_evidence_rows = build_cluster_evidence_rows_from_processed_sentences(processed_path)
                    stats["cluster_evidence_source"] = "processed_sentences"
                else:
                    cluster_evidence_rows = build_cluster_evidence_rows(final_summary_rows)
                    stats["cluster_evidence_source"] = "final_summary_clusters"
                stats["cluster_evidence_rows"] = len(cluster_evidence_rows)
                if args.cluster_evidence_csv:
                    write_csv(Path(args.cluster_evidence_csv), cluster_evidence_rows)
                if args.cluster_evidence_json:
                    write_cluster_evidence_json(Path(args.cluster_evidence_json), cluster_evidence_rows, stats)
            final_metric_rows: list[dict[str, Any]] = []
            if args.final_summary_metrics_csv or args.final_summary_metrics_json:
                final_metric_rows = build_final_summary_metric_rows(
                    final_summary,
                    final_summary_rows,
                    args.bertscore_language,
                    not args.disable_bertscore,
                )
            final_summary_rows = strip_final_summary_internal_columns(final_summary_rows)
            if args.aspect_summary_csv or args.aspect_summary_json:
                aspect_summary_rows = build_aspect_summary_report_rows(final_summary_rows)
                stats["aspect_summary_rows"] = len(aspect_summary_rows)
                if args.aspect_summary_csv:
                    write_csv(Path(args.aspect_summary_csv), aspect_summary_rows)
                if args.aspect_summary_json:
                    write_aspect_summary_json(Path(args.aspect_summary_json), aspect_summary_rows, stats)
            if args.final_summary_csv:
                write_csv(Path(args.final_summary_csv), final_summary_rows)
            if args.final_summary_json:
                write_final_summary_json(Path(args.final_summary_json), final_summary_rows, stats)
            if args.final_summary_metrics_csv or args.final_summary_metrics_json:
                if args.final_summary_metrics_csv:
                    write_summary_metrics_csv(Path(args.final_summary_metrics_csv), final_metric_rows)
                if args.final_summary_metrics_json:
                    write_final_summary_metrics_json(
                        Path(args.final_summary_metrics_json),
                        final_metric_rows,
                        stats,
                    )
        if preseg_metric_writer is not None:
            preseg_metric_writer.write_json(stats)
        checkpoint.maybe_save(
            inputs=inputs,
            strategy=args.strategy,
            hotels=hotels,
            stats=stats,
            final_summary=final_summary,
            profile_decisions=profile_decisions,
            force=True,
            run_completed=True,
        )
    finally:
        cache.close()

    print(
        json.dumps(
            {
                "hotels": len(hotels),
                "strategy": args.strategy,
                "output_csv": "" if args.final_summary_only else str(Path(args.output_csv).resolve()),
                "debug_json": str(Path(args.debug_json).resolve()) if args.debug_json else "",
                "summary_metrics_csv": (
                    str(Path(args.summary_metrics_csv).resolve()) if args.summary_metrics_csv else ""
                ),
                "summary_metrics_json": (
                    str(Path(args.summary_metrics_json).resolve()) if args.summary_metrics_json else ""
                ),
                "aspect_summary_csv": (
                    str(Path(args.aspect_summary_csv).resolve()) if args.aspect_summary_csv else ""
                ),
                "aspect_summary_json": (
                    str(Path(args.aspect_summary_json).resolve()) if args.aspect_summary_json else ""
                ),
                "final_summary_csv": (
                    str(Path(args.final_summary_csv).resolve()) if args.final_summary_csv else ""
                ),
                "final_summary_json": (
                    str(Path(args.final_summary_json).resolve()) if args.final_summary_json else ""
                ),
                "final_summary_metrics_csv": (
                    str(Path(args.final_summary_metrics_csv).resolve()) if args.final_summary_metrics_csv else ""
                ),
                "final_summary_metrics_json": (
                    str(Path(args.final_summary_metrics_json).resolve()) if args.final_summary_metrics_json else ""
                ),
                "cluster_evidence_csv": (
                    str(Path(args.cluster_evidence_csv).resolve()) if args.cluster_evidence_csv else ""
                ),
                "cluster_evidence_json": (
                    str(Path(args.cluster_evidence_json).resolve()) if args.cluster_evidence_json else ""
                ),
                "preseg_metrics_csv": (
                    str(Path(args.preseg_metrics_csv).resolve())
                    if preseg_metric_writer is not None and args.preseg_metrics_csv
                    else ""
                ),
                "preseg_metrics_json": (
                    str(Path(args.preseg_metrics_json).resolve())
                    if preseg_metric_writer is not None and args.preseg_metrics_json
                    else ""
                ),
                "cache_db": str(Path(args.cache_db).resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else "",
                "aspect_output_dir": (
                    str(Path(args.aspect_output_dir).resolve())
                    if aspect_writer is not None and aspect_writer.has_rows
                    else ""
                ),
                "processed_sentences_csv": (
                    str(Path(args.processed_sentences_csv).resolve())
                    if processed_sentence_writer is not None and processed_sentence_writer.has_rows
                    else ""
                ),
                "stats": stats["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
