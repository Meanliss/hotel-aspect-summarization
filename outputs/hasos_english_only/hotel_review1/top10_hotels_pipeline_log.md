# Top-10 Hotels Pipeline Log &mdash; `hotel_review1.csv`

Refactored view of the SemAE HASOS English-only scoring run. All numbers below are pulled from the sibling JSON log.

## 1. Pipeline Overview

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

Input: `C:\Users\dso3hc\Downloads\Implement\hotel_review1.csv`

Language detector: English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.

ROUGE: not available (no human gold summaries in the source CSV).


## 2. Pipeline Issues Found

Auto-detected from JSON log. **HIGH: 36 | MEDIUM: 64 | LOW: 17**

| Severity | Hotel(s) | Aspect(s) | Issue | Failing Step |
| :---: | --- | --- | --- | --- |
| **HIGH** | booking_02157 | `FAC_BATH, FAC_ROOM` | Identical representative sentence reused: "The room was comfortable and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02157 | `BRA_LUXURY, FAC_INTERIOR` | Identical representative sentence reused: "The pool is lovely - a beautiful view of the city and the room was so luxurious." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02157 | `AM_ROOM_UTIL, FAC_ENV, SER_COMM` | Identical representative sentence reused: "the pool was so good but must be mantained -elevator time (very slow) with 1 lif..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02157 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was very comfortable and clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02157 | `AM_POOL, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The room was very comfortable and the rooftop pool is amazing." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02757 | `AM_ENT, AM_TRANSPORT, AM_WELLNESS, EXP_EMOTION, FAC_ENV` | Identical representative sentence reused: "The staff were amazing and went above and beyond at every chance The breakfast w..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02757 | `AM_FOOD, EXP_OVERALL` | Identical representative sentence reused: "The rooftop bar was amazing, the breakfast was great and staff were very friendl..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02757 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "Location of the hotel was very convenient." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02787 | `FAC_INTERIOR, FAC_ROOM` | Identical representative sentence reused: "It is in a fantastic location, the hotel is Beautiful and the rooms are clean an..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02787 | `BRA_LUXURY, BRA_REPUTE` | Identical representative sentence reused: "The room was a high standard, well-equipped and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `AM_POOL, FAC_BATH` | Identical representative sentence reused: "The room was nice and comfortable and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `EXP_EMOTION, FAC_INTERIOR` | Identical representative sentence reused: "The room is clean and cozy." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `AM_WELLNESS, FAC_SECURITY` | Identical representative sentence reused: "The rooms were modern , very clean , and a safe location (you have to go down a ..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `AM_UTILITY, AM_WIFI` | Identical representative sentence reused: "Facilities were nice and clean, staff are really friendly and helpful, the Wi Fi..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02798 | `BRA_LUXURY, BRA_REPUTE` | Identical representative sentence reused: "Finally the hotel is very clean and high standard and it is so quiet for a downt..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02806 | `AM_POOL, FAC_BATH, FAC_ROOM` | Identical representative sentence reused: "The room was very clean and comfortable!" | 8 (TF-IDF centroid) |
| **HIGH** | booking_02806 | `FAC_BUILDING, FAC_ENV` | Identical representative sentence reused: "The location was in the heart of the old quarter, but the hotel was quiet." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02806 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was very clean and the bed was very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02806 | `AM_WELLNESS, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "- The staff were very friendly and willing to help with any questions and discom..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02806 | `EXP_OVERALL, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The amazing staff and the great location." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02810 | `EXP_SAFETY, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The room was comfortable and clean!" | 8 (TF-IDF centroid) |
| **HIGH** | booking_02810 | `AM_POOL, FAC_BATH` | Identical representative sentence reused: "The hotel is very new and room was very comfortable and clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02810 | `AM_WELLNESS, FAC_INTERIOR` | Identical representative sentence reused: "The hotel is clean and modern." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02810 | `AM_FOOD, EXP_OVERALL, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The hotel was amazing, the location is great, near everything The staff very hel..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02813 | `AM_POOL, AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room was very clean and spacious." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02813 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was very comfortable and clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02813 | `AM_ENT, BRA_REPUTE, FAC_SECURITY, SER_OPERATION` | Identical representative sentence reused: "I booked a room at Hanoi Silk hotel center but when we arrive at around 10-11 pm..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02813 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The location is very convenient." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02840 | `AM_FOOD, FAC_ROOM, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The bed is comfortable, the room has a good size and was very clean, the staff i..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02840 | `AM_POOL, EXP_SAFETY, FAC_BATH` | Identical representative sentence reused: "The room was very comfortable and clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02840 | `AM_ENT, AM_ROOM_UTIL, AM_UTILITY, BRA_LUXURY, BRA_REPUTE, EXP_EMOTION, FAC_SECURITY, SER_OPERATION` | Identical representative sentence reused: "The location and the staff was very good, the rooms were of a decent size and ha..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02886 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The room was great, clean and very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02886 | `AM_ROOM_UTIL, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The hotel was great, location is very convenient, the staff was very kind and he..." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02897 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The bed was very comfortable and the room was very clean." | 8 (TF-IDF centroid) |
| **HIGH** | booking_02897 | `AM_UTILITY, BRA_REPUTE` | Identical representative sentence reused: "Great location Nice staff Comfortable room The hotel is good, but we were previo..." | 8 (TF-IDF centroid) |
| **MEDIUM** | 6 hotels | `SER_ATTITUDE` | Generic boilerplate repeated: "the staff were very friendly and helpful." | 8 (no novelty filter) |
| **MEDIUM** | booking_02157 | `FAC_INTERIOR` | Summary mixes 3 topics (location, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `FAC_ENV` | Summary mixes 5 topics (bath, food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `AM_FOOD` | Summary mixes 3 topics (food, location, pool) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `AM_ROOM_UTIL` | Summary mixes 5 topics (bath, food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `SER_COMM` | Summary mixes 5 topics (bath, food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `EXP_EMOTION` | Summary mixes 4 topics (food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02157 | `BRA_LUXURY` | Summary mixes 3 topics (location, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `FAC_ENV` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_WIFI` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_FOOD` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_POOL` | Summary mixes 5 topics (food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_WELLNESS` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_ENT` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `AM_TRANSPORT` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `SER_OPERATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `EXP_OVERALL` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `EXP_EMOTION` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02757 | `BRA_LUXURY` | Summary mixes 6 topics (bath, food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02787 | `AM_ENT` | Summary mixes 4 topics (bath, food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02798 | `FAC_SECURITY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02798 | `AM_WELLNESS` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02798 | `AM_ENT` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `FAC_INTERIOR` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `FAC_SECURITY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `AM_WELLNESS` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `AM_ROOM_UTIL` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `SER_OPERATION` | Summary mixes 4 topics (bath, food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02806 | `BRA_REPUTE` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `FAC_VIEW_LOCATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `AM_FOOD` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `AM_ENT` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `SER_OPERATION` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `EXP_OVERALL` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02810 | `EXP_EMOTION` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `FAC_INTERIOR` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `FAC_SECURITY` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `AM_FOOD` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `AM_ENT` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `SER_OPERATION` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `BRA_REPUTE` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02813 | `BRA_LUXURY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `FAC_ROOM` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `FAC_INTERIOR` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `FAC_SECURITY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `FAC_VIEW_LOCATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `AM_WIFI` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `AM_FOOD` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `AM_ENT` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `AM_ROOM_UTIL` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `AM_UTILITY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `SER_OPERATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `EXP_EMOTION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `BRA_REPUTE` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02840 | `BRA_LUXURY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02886 | `FAC_INTERIOR` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02886 | `AM_UTILITY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02886 | `SER_OPERATION` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `AM_WELLNESS` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `AM_UTILITY` | Summary mixes 4 topics (location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `SER_OPERATION` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `EXP_OVERALL` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `EXP_EMOTION` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | booking_02897 | `BRA_REPUTE` | Summary mixes 4 topics (location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **LOW** | 10 hotels | `SER_OPERATION` | CEC below 0.50 (avg 0.439) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 3 hotels | `LOY_RETURN` | CEC below 0.50 (avg 0.483) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 3 hotels | `AM_TRANSPORT` | CEC below 0.50 (avg 0.486) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `FAC_ENV` | CEC below 0.50 (avg 0.480) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `EXP_VALUE` | CEC below 0.50 (avg 0.476) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `EXP_OVERALL` | CEC below 0.50 (avg 0.467) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 5 hotels | `SER_SUPPORT` | CEC below 0.50 (avg 0.477) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 5 hotels | `BRA_REPUTE` | CEC below 0.50 (avg 0.459) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 5 hotels | `AM_WELLNESS` | CEC below 0.50 (avg 0.463) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 7 hotels | `AM_ROOM_UTIL` | CEC below 0.50 (avg 0.453) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `FAC_SECURITY` | CEC below 0.50 (avg 0.449) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `SER_COMM` | CEC below 0.50 (avg 0.459) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `AM_WIFI` | CEC below 0.50 (avg 0.459) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `AM_ENT` | CEC below 0.50 (avg 0.455) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `AM_UTILITY` | CEC below 0.50 (avg 0.462) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `EXP_EMOTION` | CEC below 0.50 (avg 0.456) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `BRA_LUXURY` | CEC below 0.50 (avg 0.466) — weak evidence coverage | 7+8 (clustering/centroid) |


## 3. Selected Hotels

Thresholds — PASS: ASC >= 0.73 AND CEC >= 0.54; FAIL: ASC < 0.7 AND CEC < 0.5; WARN otherwise.

| # | Hotel ID | Reviews | Sentences | Aspects | ASC | Macro CEC | Weighted CEC | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `booking_02157` | 1,327 | 4,015 | 29/29 | 0.7358 | 0.5153 | 0.5225 | [WARN] |
| 2 | `booking_02806` | 1,091 | 3,385 | 29/29 | 0.7446 | 0.5499 | 0.5608 | [PASS] |
| 3 | `booking_02757` | 985 | 3,126 | 29/29 | 0.7416 | 0.5263 | 0.5340 | [WARN] |
| 4 | `booking_02798` | 942 | 2,603 | 29/29 | 0.7473 | 0.5279 | 0.5380 | [WARN] |
| 5 | `booking_02840` | 933 | 3,003 | 29/29 | 0.7682 | 0.5371 | 0.5488 | [WARN] |
| 6 | `booking_02787` | 906 | 3,795 | 29/29 | 0.7301 | 0.5400 | 0.5497 | [PASS] |
| 7 | `booking_02813` | 884 | 2,562 | 29/29 | 0.7141 | 0.5488 | 0.5548 | [WARN] |
| 8 | `booking_02886` | 821 | 2,349 | 29/29 | 0.7296 | 0.5429 | 0.5547 | [WARN] |
| 9 | `booking_02810` | 815 | 2,502 | 29/29 | 0.7400 | 0.5199 | 0.5304 | [WARN] |
| 10 | `booking_02897` | 812 | 2,477 | 29/29 | 0.7589 | 0.5163 | 0.5268 | [WARN] |


## 4. Step-by-Step Trace

Single example: hotel #1 `booking_02157` (top aspect `AM_POOL`).

| Step | Transform | Input | Output | Sample evidence | Health |
| ---: | --- | --- | --- | --- | :---: |
| 1 | Load CSV | Raw rows | 1,327 reviews kept | Fantastic pool, nice view | OK |
| 2 | Group by hotel_id | ref_id without numeric suffix | Hotel `booking_02157` | 1,327 rows | OK |
| 3 | Sentence split | Per-review splitter | 4,015 sentences | Fantastic pool, nice view | OK |
| 4 | Normalize | lowercase, strip accents | Normalized text | fantastic pool, nice view | OK |
| 5 | Aspect match | Keyword vs HASOS taxonomy | 29/29 aspects hit | Top: `AM_POOL` | WARN: over-matches |
| 6 | Dedup | Unique opinions per aspect | 1,170 unique / 1,184 matched | — | OK |
| 7 | Cluster weight | log(1+n) normalized | weight = 0.0476 | — | OK |
| 8 | Representative | TF-IDF centroid pick | 1 sentence per aspect | The room was very comfortable and the rooftop pool is amazing. | FAIL: not discriminative |
| 9 | Metrics | CEC / ASC | ASC 0.7358 / CEC 0.5153 | — | WARN: inherits step-8 noise |


## 5. Per-Hotel Details

_Click a hotel to expand._

<details>
<summary><strong>1. <code>booking_02157</code></strong> &mdash; ASC 0.7358 &middot; CEC 0.5153 &middot; 1,327 reviews &middot; [WARN]</summary>

- Sentences: 4,015
- Matched aspects: 29/29
- Weighted CEC: 0.5225
- First review sample: Fantastic pool, nice view

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_POOL` | 1,170 | 0.0476 | 0.5636 | 0.0381 | The room was very comfortable and the rooftop pool is amazing. |
| `FAC_VIEW_LOCATION` | 784 | 0.0449 | 0.5838 | 0.0355 | The room was very comfortable and the rooftop pool is amazing. |
| `AM_FOOD` | 980 | 0.0464 | 0.5665 | 0.0350 | The pool was big and had amazing views of the city and the breakfast was amazing. |
| `SER_ATTITUDE` | 719 | 0.0443 | 0.6261 | 0.0323 | The staff were very friendly and helpful. |
| `EXP_OVERALL` | 627 | 0.0434 | 0.5741 | 0.0320 | The pool was amazing! |

</details>

<details>
<summary><strong>2. <code>booking_02806</code></strong> &mdash; ASC 0.7446 &middot; CEC 0.5499 &middot; 1,091 reviews &middot; [PASS]</summary>

- Sentences: 3,385
- Matched aspects: 29/29
- Weighted CEC: 0.5608
- First review sample: Good location, helpful and friendly staffs, lovely room. I will come back. Highly recommended this place See u next time Scent Vị trí tốt, nhân viên cực kì thân thiện và đáng yêu, phòng rất đẹp và sạc...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,096 | 0.0503 | 0.6923 | 0.0402 | The staff were very friendly and helpful. |
| `FAC_VIEW_LOCATION` | 762 | 0.0477 | 0.6395 | 0.0369 | The amazing staff and the great location. |
| `FAC_ROOM` | 865 | 0.0486 | 0.6057 | 0.0363 | The room was very clean and comfortable! |
| `FAC_BUILDING` | 281 | 0.0406 | 0.5871 | 0.0338 | The location was in the heart of the old quarter, but the hotel was quiet. |
| `AM_FOOD` | 695 | 0.0471 | 0.5971 | 0.0323 | The staff were very friendly and helpful and the location is excellent. |

</details>

<details>
<summary><strong>3. <code>booking_02757</code></strong> &mdash; ASC 0.7416 &middot; CEC 0.5263 &middot; 985 reviews &middot; [WARN]</summary>

- Sentences: 3,126
- Matched aspects: 29/29
- Weighted CEC: 0.5340
- First review sample: Khi khách vào check in mà lễ tân lại ngồi để làm việc thì không ổn chút nào. Cảm giác không thoải mái thật sự khi vào khách sạn đánh giá 5 sao.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 774 | 0.0474 | 0.5281 | 0.0383 | The room was very comfortable & clean. |
| `AM_POOL` | 433 | 0.0433 | 0.5548 | 0.0368 | The hotel itself was beautiful - rooftop pool and bar were excellent, food and drinks were amazing, the room was super c... |
| `AM_FOOD` | 869 | 0.0482 | 0.6336 | 0.0363 | The rooftop bar was amazing, the breakfast was great and staff were very friendly and helpful. |
| `SER_ATTITUDE` | 769 | 0.0474 | 0.6305 | 0.0361 | The staff were very friendly and helpful. |
| `FAC_VIEW_LOCATION` | 676 | 0.0464 | 0.6105 | 0.0338 | The breakfast was amazing and the location was great. |

</details>

<details>
<summary><strong>4. <code>booking_02798</code></strong> &mdash; ASC 0.7473 &middot; CEC 0.5279 &middot; 942 reviews &middot; [WARN]</summary>

- Sentences: 2,603
- Matched aspects: 29/29
- Weighted CEC: 0.5380
- First review sample: nothing Total scam! Everything on the site is inaccurate. False advertising on booking site. It's not 4star, it's not worth for the cost. Room- very small, bad smell from the bathroom and the bathroom...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 725 | 0.0495 | 0.6882 | 0.0395 | The staff was very helpful and friendly . |
| `FAC_VIEW_LOCATION` | 497 | 0.0467 | 0.5781 | 0.0377 | And the hotel is in a great location. |
| `FAC_BATH` | 449 | 0.0459 | 0.6130 | 0.0350 | The room was nice and comfortable and very clean. |
| `AM_FOOD` | 422 | 0.0454 | 0.5502 | 0.0343 | The room was clean and the breakfast was good. |
| `FAC_BUILDING` | 217 | 0.0404 | 0.6002 | 0.0343 | The hotel is in the old quarter. |

</details>

<details>
<summary><strong>5. <code>booking_02840</code></strong> &mdash; ASC 0.7682 &middot; CEC 0.5371 &middot; 933 reviews &middot; [WARN]</summary>

- Sentences: 3,003
- Matched aspects: 29/29
- Weighted CEC: 0.5488
- First review sample: The hotel is well located in the Old Quarter. The breakfast was good with many options. The air-conditioner was noisy - The lift can be slow, in particular in the morning when people are going for bre...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_VIEW_LOCATION` | 640 | 0.0480 | 0.6149 | 0.0407 | The bed is comfortable, the room has a good size and was very clean, the staff is super nice and helpful, the breakfast ... |
| `AM_FOOD` | 802 | 0.0497 | 0.5895 | 0.0392 | The bed is comfortable, the room has a good size and was very clean, the staff is super nice and helpful, the breakfast ... |
| `FAC_ROOM` | 875 | 0.0503 | 0.6198 | 0.0392 | The bed is comfortable, the room has a good size and was very clean, the staff is super nice and helpful, the breakfast ... |
| `SER_ATTITUDE` | 818 | 0.0498 | 0.6852 | 0.0389 | The staff was very friendly and helpful. |
| `FAC_BATH` | 448 | 0.0454 | 0.6215 | 0.0359 | The room was very comfortable and clean. |

</details>

<details>
<summary><strong>6. <code>booking_02787</code></strong> &mdash; ASC 0.7301 &middot; CEC 0.5400 &middot; 906 reviews &middot; [PASS]</summary>

- Sentences: 3,795
- Matched aspects: 29/29
- Weighted CEC: 0.5497
- First review sample: Staff verry good

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_VIEW_LOCATION` | 681 | 0.0454 | 0.5502 | 0.0401 | This hotel is in a great location in the heart of the old quarter and a great place to explore Hanoi from. |
| `SER_ATTITUDE` | 849 | 0.0469 | 0.6866 | 0.0362 | The staff were very friendly and helpful. |
| `FAC_ROOM` | 1,069 | 0.0485 | 0.5804 | 0.0358 | It is in a fantastic location, the hotel is Beautiful and the rooms are clean and spacious and very comfortable. |
| `FAC_BATH` | 486 | 0.0430 | 0.6022 | 0.0352 | The hotel is very clean and the room is lovely and comfortable. |
| `FAC_BUILDING` | 350 | 0.0408 | 0.6040 | 0.0346 | The location in the heart of the Old Quarter of Hanoi is perfect. |

</details>

<details>
<summary><strong>7. <code>booking_02813</code></strong> &mdash; ASC 0.7141 &middot; CEC 0.5488 &middot; 884 reviews &middot; [WARN]</summary>

- Sentences: 2,562
- Matched aspects: 29/29
- Weighted CEC: 0.5548
- First review sample: Great location and amazing staffs. the hotel is under construction so it smells bad and a little messy.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 729 | 0.0507 | 0.7159 | 0.0388 | The staff was very friendly and helpful! |
| `FAC_VIEW_LOCATION` | 502 | 0.0478 | 0.5579 | 0.0387 | Very helpful staff and the location was great! |
| `FAC_ROOM` | 781 | 0.0512 | 0.5540 | 0.0326 | the bed was comfortable the staff was nice and the room was clean. |
| `FAC_BUILDING` | 227 | 0.0417 | 0.6153 | 0.0316 | The hotel is right in the Old quarter, in the heart of the city! |
| `AM_POOL` | 199 | 0.0407 | 0.6911 | 0.0307 | The room was very clean and spacious. |

</details>

<details>
<summary><strong>8. <code>booking_02886</code></strong> &mdash; ASC 0.7296 &middot; CEC 0.5429 &middot; 821 reviews &middot; [WARN]</summary>

- Sentences: 2,349
- Matched aspects: 29/29
- Weighted CEC: 0.5547
- First review sample: room was very comfortable and clean. executive room very quite. junior suite bigger and nice balcony but of course less quite. breakfast very goodstaff very professional and kind. great location

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 682 | 0.0505 | 0.7417 | 0.0399 | The staff were very friendly and helpful. |
| `FAC_VIEW_LOCATION` | 495 | 0.0480 | 0.5636 | 0.0397 | The hotel was great, location is very convenient, the staff was very kind and helpful. |
| `FAC_ROOM` | 737 | 0.0511 | 0.5882 | 0.0378 | The room was great, clean and very comfortable. |
| `AM_FOOD` | 474 | 0.0477 | 0.5907 | 0.0362 | The breakfast was very good. |
| `FAC_BATH` | 343 | 0.0452 | 0.6198 | 0.0318 | The room was great, clean and very comfortable. |

</details>

<details>
<summary><strong>9. <code>booking_02810</code></strong> &mdash; ASC 0.7400 &middot; CEC 0.5199 &middot; 815 reviews &middot; [WARN]</summary>

- Sentences: 2,502
- Matched aspects: 29/29
- Weighted CEC: 0.5304
- First review sample: Good breakfast, great location, easy for shopping and food. Clean and comfortable hotel, especially the receptionist Ms Anna, very helpful and friendly.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 715 | 0.0494 | 0.5800 | 0.0388 | The room was comfortable and clean! |
| `AM_FOOD` | 516 | 0.0469 | 0.5727 | 0.0372 | The hotel was amazing, the location is great, near everything The staff very helpful and nice The room was big, clean an... |
| `FAC_VIEW_LOCATION` | 486 | 0.0465 | 0.5424 | 0.0371 | The hotel was amazing, the location is great, near everything The staff very helpful and nice The room was big, clean an... |
| `SER_ATTITUDE` | 701 | 0.0492 | 0.6386 | 0.0361 | The staff very helpful and friendly. |
| `FAC_BATH` | 350 | 0.0440 | 0.5964 | 0.0361 | The hotel is very new and room was very comfortable and clean. |

</details>

<details>
<summary><strong>10. <code>booking_02897</code></strong> &mdash; ASC 0.7589 &middot; CEC 0.5163 &middot; 812 reviews &middot; [WARN]</summary>

- Sentences: 2,477
- Matched aspects: 29/29
- Weighted CEC: 0.5268
- First review sample: Excellent and widespread breakfast with many choice. I really love the breakfast here. Room is squeaky clean and the design is lovely. Thank you for a great stay! nothing. this hotel is literally the ...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_FOOD` | 498 | 0.0477 | 0.4845 | 0.0388 | The hotel is nice and the breakfast is very good. |
| `SER_ATTITUDE` | 723 | 0.0506 | 0.6971 | 0.0376 | The staff were very friendly and helpful. |
| `EXP_OVERALL` | 336 | 0.0447 | 0.5276 | 0.0364 | The hotel was in the perfect location, The staff at the hotel were the best we have had and very helpful and very welcom... |
| `FAC_ROOM` | 694 | 0.0503 | 0.5968 | 0.0359 | The bed was very comfortable and the room was very clean. |
| `FAC_VIEW_LOCATION` | 457 | 0.0471 | 0.5388 | 0.0334 | And the location is very good. |

</details>


## 6. Re-run Command

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\scripts
python .\log_10_hotels_pipeline.py --input-csv ..\..\hotel_review1.csv --limit 10
```
