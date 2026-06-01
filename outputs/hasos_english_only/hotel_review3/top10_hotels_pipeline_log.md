# Top-10 Hotels Pipeline Log &mdash; `hotel_review3.csv`

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

Input: `C:\Users\dso3hc\Downloads\Implement\hotel_review3.csv`

Language detector: English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.

ROUGE: not available (no human gold summaries in the source CSV).


## 2. Pipeline Issues Found

Auto-detected from JSON log. **HIGH: 38 | MEDIUM: 41 | LOW: 16**

| Severity | Hotel(s) | Aspect(s) | Issue | Failing Step |
| :---: | --- | --- | --- | --- |
| **HIGH** | tripadvisor_01306 | `AM_POOL, AM_TRANSPORT, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The hotel is very comfortable and the rooms are spacious and clean." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_01306 | `BRA_LUXURY, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The room we stayed in was lovely and the cleanliness was of a very high standard..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_01306 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The hotel is in a very convenient location in the Old Quarter." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_10987 | `AM_POOL, AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room was spacious and clean and very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_10987 | `AM_WELLNESS, FAC_INTERIOR` | Identical representative sentence reused: "The hotel is in a great location and the rooms were clean and modern." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_10987 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was very clean and spacious and the bed was so comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_10987 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The location of the hotel is located in the Old Quarter in Hanoi, this was very ..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_10987 | `EXP_EMOTION, EXP_OVERALL` | Identical representative sentence reused: "We had a wonderful stay in this hotel and the staff went above and beyond to mak..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_123184 | `EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The room was clean and very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_123184 | `AM_WELLNESS, BRA_LUXURY, FAC_INTERIOR, FAC_SECURITY` | Identical representative sentence reused: "The hotel is clean and modern and the rooms are all to a very high standard." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_123184 | `AM_POOL, AM_TRANSPORT` | Identical representative sentence reused: "The room was spacious and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_125833 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The hotel was very nice and the room was very clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_125833 | `AM_ENT, BRA_LUXURY, BRA_REPUTE` | Identical representative sentence reused: "The rooms are of a high standard and the breakfast, which is included in the pri..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_125833 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The hotel is located in a very convenient location and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_140411 | `AM_POOL, FAC_BATH` | Identical representative sentence reused: "The hotel is well appointed and the room was clean and very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_140411 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The rooms were clean and the bed was very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_140411 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The hotel is in great condition as you'd expect from a Sofitel and the restauran..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_140411 | `AM_UTILITY, AM_WELLNESS` | Identical representative sentence reused: "All of the staff were very helpful and friendly and the service was very profess..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_429195 | `EXP_SAFETY, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The hotel and room was very clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_429195 | `AM_POOL, AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room is very spacious and clean." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_429195 | `AM_ENT, AM_ROOM_UTIL, AM_UTILITY, BRA_LUXURY, BRA_REPUTE` | Identical representative sentence reused: "Nikko is a fantastic hotel with excellent staff, the hotel itself is in a great ..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_440716 | `EXP_SAFETY, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The hotel is very clean and the rooms comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_440716 | `AM_WELLNESS, FAC_BUILDING` | Identical representative sentence reused: "The hotel is modern and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_440716 | `EXP_EMOTION, EXP_OVERALL, FAC_ENV` | Identical representative sentence reused: "Stayed at this hotel and from the moment I walked through the doors I was impres..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_440716 | `BRA_LUXURY, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The breakfast was of a very high standard." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_440716 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The hotel location is very convenient and in the heart of the city." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_458627 | `AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The rooms were very clean and comfortable and spacious." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_458627 | `AM_FOOD, FAC_INTERIOR` | Identical representative sentence reused: "The hotel is beautiful, the service was excellent and the restaurants were very ..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_458627 | `BRA_LUXURY, FAC_BUILDING` | Identical representative sentence reused: "My first visit to saigon and stayed at the hotel top location in city centre nex..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_458627 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room and bed was very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_458627 | `AM_ENT, BRA_REPUTE` | Identical representative sentence reused: "The food was well above standard for a hotel and this applied to all the restaur..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_89558 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was clean and bed very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_89558 | `BRA_LUXURY, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The first night we were in a standard room which was comfortable and clean and h..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_89558 | `AM_ENT, EXP_EMOTION` | Identical representative sentence reused: "My new husband and I chose this hotel for a night during our honeymoon after rea..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_89558 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "We had a very nice room, the staff was very kind and the hotel is located very c..." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_96021 | `FAC_ROOM, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The hotel is in a great location in the old quarter." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_96021 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE` | Identical representative sentence reused: "The room was very clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | tripadvisor_96021 | `AM_ENT, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "We ate in the restaurant one night and the food was of a very high standard as w..." | 8 (TF-IDF centroid) |
| **MEDIUM** | tripadvisor_01306 | `AM_ENT` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_01306 | `SER_OPERATION` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_01306 | `EXP_EMOTION` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_10987 | `AM_ENT` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_123184 | `AM_ROOM_UTIL` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_123184 | `AM_UTILITY` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_123184 | `SER_OPERATION` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_123184 | `EXP_EMOTION` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_140411 | `AM_FOOD` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_140411 | `AM_ENT` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_140411 | `EXP_EMOTION` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `AM_ENT` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `AM_ROOM_UTIL` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `AM_UTILITY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `EXP_OVERALL` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `BRA_REPUTE` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_429195 | `BRA_LUXURY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `FAC_ENV` | Summary mixes 4 topics (location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `AM_FOOD` | Summary mixes 5 topics (food, location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `AM_ENT` | Summary mixes 4 topics (food, location, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `EXP_OVERALL` | Summary mixes 4 topics (location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `EXP_EMOTION` | Summary mixes 4 topics (location, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_440716 | `EXP_VALUE` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `FAC_ENV` | Summary mixes 3 topics (location, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `AM_WIFI` | Summary mixes 3 topics (room, staff, wifi) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `AM_WELLNESS` | Summary mixes 4 topics (food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `AM_ROOM_UTIL` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `AM_UTILITY` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_458627 | `EXP_EMOTION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `FAC_ROOM` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `AM_WIFI` | Summary mixes 3 topics (bath, room, wifi) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `AM_FOOD` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `AM_ENT` | Summary mixes 3 topics (bath, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `AM_ROOM_UTIL` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `AM_UTILITY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `EXP_OVERALL` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_89558 | `EXP_EMOTION` | Summary mixes 3 topics (bath, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_96021 | `AM_FOOD` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_96021 | `SER_OPERATION` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_96021 | `EXP_EMOTION` | Summary mixes 4 topics (bath, food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | tripadvisor_96021 | `EXP_VALUE` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **LOW** | 10 hotels | `FAC_SECURITY` | CEC below 0.50 (avg 0.467) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 10 hotels | `AM_ENT` | CEC below 0.50 (avg 0.451) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 10 hotels | `BRA_LUXURY` | CEC below 0.50 (avg 0.458) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `FAC_ENV` | CEC below 0.50 (avg 0.476) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `SER_SUPPORT` | CEC below 0.50 (avg 0.478) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 5 hotels | `AM_WELLNESS` | CEC below 0.50 (avg 0.477) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `AM_ROOM_UTIL` | CEC below 0.50 (avg 0.476) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `EXP_VALUE` | CEC below 0.50 (avg 0.462) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `EXP_OVERALL` | CEC below 0.50 (avg 0.487) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 7 hotels | `LOY_PREFERENCE` | CEC below 0.50 (avg 0.423) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `AM_WIFI` | CEC below 0.50 (avg 0.450) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `AM_UTILITY` | CEC below 0.50 (avg 0.470) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `SER_OPERATION` | CEC below 0.50 (avg 0.460) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `EXP_EMOTION` | CEC below 0.50 (avg 0.455) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `BRA_REPUTE` | CEC below 0.50 (avg 0.463) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `SER_COMM` | CEC below 0.50 (avg 0.464) — weak evidence coverage | 7+8 (clustering/centroid) |


## 3. Selected Hotels

Thresholds — PASS: ASC >= 0.73 AND CEC >= 0.54; FAIL: ASC < 0.7 AND CEC < 0.5; WARN otherwise.

| # | Hotel ID | Reviews | Sentences | Aspects | ASC | Macro CEC | Weighted CEC | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `tripadvisor_96021` | 4,541 | 27,703 | 29/29 | 0.7278 | 0.5349 | 0.5400 | [WARN] |
| 2 | `tripadvisor_140411` | 3,923 | 23,074 | 29/29 | 0.7415 | 0.4950 | 0.4970 | [WARN] |
| 3 | `tripadvisor_10987` | 2,891 | 17,691 | 29/29 | 0.7343 | 0.5240 | 0.5303 | [WARN] |
| 4 | `tripadvisor_01306` | 2,594 | 14,607 | 29/29 | 0.7457 | 0.5341 | 0.5422 | [WARN] |
| 5 | `tripadvisor_429195` | 2,500 | 14,161 | 29/29 | 0.7209 | 0.5269 | 0.5323 | [WARN] |
| 6 | `tripadvisor_89558` | 2,487 | 14,897 | 29/29 | 0.7410 | 0.5385 | 0.5456 | [WARN] |
| 7 | `tripadvisor_440716` | 2,462 | 14,287 | 29/29 | 0.7258 | 0.5364 | 0.5426 | [WARN] |
| 8 | `tripadvisor_123184` | 2,347 | 14,039 | 29/29 | 0.7406 | 0.5347 | 0.5404 | [WARN] |
| 9 | `tripadvisor_125833` | 2,337 | 13,706 | 29/29 | 0.7236 | 0.5251 | 0.5329 | [WARN] |
| 10 | `tripadvisor_458627` | 2,305 | 12,159 | 29/29 | 0.7303 | 0.5080 | 0.5103 | [WARN] |


## 4. Step-by-Step Trace

Single example: hotel #1 `tripadvisor_96021` (top aspect `FAC_ROOM`).

| Step | Transform | Input | Output | Sample evidence | Health |
| ---: | --- | --- | --- | --- | :---: |
| 1 | Load CSV | Raw rows | 4,541 reviews kept | What makes a good hotel great? The people of course! La Siesta has a great team of staff who make your stay a memorable ... | OK |
| 2 | Group by hotel_id | ref_id without numeric suffix | Hotel `tripadvisor_96021` | 4,541 rows | OK |
| 3 | Sentence split | Per-review splitter | 27,703 sentences | What makes a good hotel great? The people of course! La Siesta has a great team of staff who make your stay a memorable  | OK |
| 4 | Normalize | lowercase, strip accents | Normalized text | what makes a good hotel great? the people of course! la siesta has a great team of staff who make your stay a memorable ... | OK |
| 5 | Aspect match | Keyword vs HASOS taxonomy | 29/29 aspects hit | Top: `FAC_ROOM` | WARN: over-matches |
| 6 | Dedup | Unique opinions per aspect | 6,776 unique / 6,817 matched | — | OK |
| 7 | Cluster weight | log(1+n) normalized | weight = 0.0443 | — | OK |
| 8 | Representative | TF-IDF centroid pick | 1 sentence per aspect | The hotel is in a great location in the old quarter. | FAIL: not discriminative |
| 9 | Metrics | CEC / ASC | ASC 0.7278 / CEC 0.5349 | — | WARN: inherits step-8 noise |


## 5. Per-Hotel Details

_Click a hotel to expand._

<details>
<summary><strong>1. <code>tripadvisor_96021</code></strong> &mdash; ASC 0.7278 &middot; CEC 0.5349 &middot; 4,541 reviews &middot; [WARN]</summary>

- Sentences: 27,703
- Matched aspects: 29/29
- Weighted CEC: 0.5400
- First review sample: What makes a good hotel great? The people of course! La Siesta has a great team of staff who make your stay a memorable one! They are attentive, friendly & so supportive! I would thoroughly recommend ...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 6,776 | 0.0443 | 0.5411 | 0.0346 | The hotel is in a great location in the old quarter. |
| `FAC_BUILDING` | 2,582 | 0.0395 | 0.6715 | 0.0328 | The hotel is in the heart of the Old Quarter. |
| `SER_ATTITUDE` | 5,961 | 0.0437 | 0.5934 | 0.0314 | The hotel staff was very friendly and helpful to us. |
| `EXP_OVERALL` | 5,185 | 0.0430 | 0.5131 | 0.0310 | We had an amazing trip to Hanoi and our stay at the La Siesta made in all the more memorable. |
| `AM_WELLNESS` | 1,793 | 0.0376 | 0.5133 | 0.0299 | ....and the Spa...... |

</details>

<details>
<summary><strong>2. <code>tripadvisor_140411</code></strong> &mdash; ASC 0.7415 &middot; CEC 0.4950 &middot; 3,923 reviews &middot; [WARN]</summary>

- Sentences: 23,074
- Matched aspects: 29/29
- Weighted CEC: 0.4970
- First review sample: The most beautiful hotel with the best service. We enjoyed our stay at Sofitel Metropole and came and went three times over our Vietnam holiday. The location is excellent and places the hotel in easy ...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 5,261 | 0.0439 | 0.4844 | 0.0358 | We stayed in the old wing, the room was well appointed, the bathroom large and the bed was very comfortable. |
| `AM_FOOD` | 4,296 | 0.0428 | 0.4914 | 0.0349 | The food in the bar by the pool was very good, as was the service, and they gave us a great breakfast in the hotel limo ... |
| `FAC_BUILDING` | 2,378 | 0.0398 | 0.4835 | 0.0333 | This is a beautiful hotel in an excellent location; the original part of the hotel is full of history and the rooms have... |
| `FAC_INTERIOR` | 2,399 | 0.0399 | 0.4909 | 0.0302 | The hotel is beautiful. |
| `FAC_VIEW_LOCATION` | 2,793 | 0.0406 | 0.5516 | 0.0286 | The hotel is beautiful and the staff is amazing. |

</details>

<details>
<summary><strong>3. <code>tripadvisor_10987</code></strong> &mdash; ASC 0.7343 &middot; CEC 0.5240 &middot; 2,891 reviews &middot; [WARN]</summary>

- Sentences: 17,691
- Matched aspects: 29/29
- Weighted CEC: 0.5303
- First review sample: Had a great time there in Bespoke Trendy Hotel, room was clean, its in a convenient area, breakfast was good and the staff were really friendly. In particular, David who was amazing and really went ou...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `EXP_OVERALL` | 3,322 | 0.0434 | 0.4832 | 0.0357 | We had a wonderful stay in this hotel and the staff went above and beyond to make our stay pleasant. |
| `AM_FOOD` | 3,797 | 0.0442 | 0.5019 | 0.0340 | The room was clean and the food in the restaurant was very good. |
| `FAC_BUILDING` | 1,561 | 0.0394 | 0.6172 | 0.0338 | The rooms are very clean and modern and the hotel is in the heart of old quarter!! |
| `FAC_BATH` | 1,815 | 0.0402 | 0.6041 | 0.0330 | The room was spacious and clean and very comfortable. |
| `FAC_VIEW_LOCATION` | 2,132 | 0.0411 | 0.5431 | 0.0296 | The location of the hotel is great! |

</details>

<details>
<summary><strong>4. <code>tripadvisor_01306</code></strong> &mdash; ASC 0.7457 &middot; CEC 0.5341 &middot; 2,594 reviews &middot; [WARN]</summary>

- Sentences: 14,607
- Matched aspects: 29/29
- Weighted CEC: 0.5422
- First review sample: Being near food and attractions, this conveniently located euro-styled boutique hotel was a gem to stay for my recent anniversary. With spacious & comfortable bedrooms plus boasting of a reputable res...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_FOOD` | 2,977 | 0.0441 | 0.5345 | 0.0335 | The food in the restaurant was delicious and a good selection at the breakfast. |
| `EXP_OVERALL` | 2,719 | 0.0436 | 0.4844 | 0.0332 | My husband and I were staying in Hanoi for our honeymoon and we would have to say that the experience we had at this hot... |
| `FAC_ROOM` | 3,468 | 0.0450 | 0.5913 | 0.0331 | The hotel is very comfortable and the rooms are spacious and clean. |
| `FAC_VIEW_LOCATION` | 1,918 | 0.0417 | 0.5493 | 0.0327 | The hotel was very clean and in a great location. |
| `AM_POOL` | 1,272 | 0.0394 | 0.6476 | 0.0320 | The hotel is very comfortable and the rooms are spacious and clean. |

</details>

<details>
<summary><strong>5. <code>tripadvisor_429195</code></strong> &mdash; ASC 0.7209 &middot; CEC 0.5269 &middot; 2,500 reviews &middot; [WARN]</summary>

- Sentences: 14,161
- Matched aspects: 29/29
- Weighted CEC: 0.5323
- First review sample: Your team gave me a souvenir for my 100th night, but I estimate I have stayed a total of >130 nights over the last ten years. There is not enough space to write all the good things about Nikko Saigon,...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_FOOD` | 2,768 | 0.0429 | 0.5157 | 0.0334 | The breakfast buffet was delicious and the dinner seafood buffet was amazing. |
| `EXP_OVERALL` | 2,069 | 0.0413 | 0.4993 | 0.0316 | I would like to thank all the staff at the Nikko hotel, we only stayed for 3 nights but the room and overall experience ... |
| `AM_POOL` | 1,591 | 0.0399 | 0.6394 | 0.0305 | The room is very spacious and clean. |
| `FAC_ROOM` | 3,465 | 0.0441 | 0.5930 | 0.0305 | The hotel and room was very clean and comfortable. |
| `FAC_BATH` | 1,867 | 0.0408 | 0.6488 | 0.0302 | The room is very spacious and clean. |

</details>

<details>
<summary><strong>6. <code>tripadvisor_89558</code></strong> &mdash; ASC 0.7410 &middot; CEC 0.5385 &middot; 2,487 reviews &middot; [WARN]</summary>

- Sentences: 14,897
- Matched aspects: 29/29
- Weighted CEC: 0.5456
- First review sample: The location was perfect and right in the centre of the old town, making it easy for us to walk everywhere. The staff were super friendly and Hana was in regular contact a while before our stay, to he...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 4,048 | 0.0455 | 0.5613 | 0.0354 | The hotel is in a great location within the Old Quarter of Hanoi, the rooms are clean and the breakfast is tasty. |
| `EXP_OVERALL` | 2,604 | 0.0431 | 0.4972 | 0.0350 | Our Halong Bay trip was cancelled and the Oriental Suites Hotel was full so they directed us to their Sister Hotel, The ... |
| `FAC_VIEW_LOCATION` | 2,387 | 0.0426 | 0.6075 | 0.0329 | The hotel is in a great location in the Old Quarter. |
| `FAC_BUILDING` | 1,383 | 0.0396 | 0.6879 | 0.0326 | The hotel is in a great location - right in the heart of the old quarter. |
| `AM_POOL` | 1,293 | 0.0392 | 0.6264 | 0.0324 | The breakfast was nice and the rooms were clean and comfortable. |

</details>

<details>
<summary><strong>7. <code>tripadvisor_440716</code></strong> &mdash; ASC 0.7258 &middot; CEC 0.5364 &middot; 2,462 reviews &middot; [WARN]</summary>

- Sentences: 14,287
- Matched aspects: 29/29
- Weighted CEC: 0.5426
- First review sample: Good location as there are many places to go around. The only set back was that the pool could not be used. Hence we would need to use the other Liberty hotel, ie riverside. Great team there, especial...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_FOOD` | 3,010 | 0.0433 | 0.5259 | 0.0362 | I was very impressed with this hotel in all respects and the service level from all the staff was always at the highest ... |
| `EXP_OVERALL` | 2,018 | 0.0412 | 0.5052 | 0.0351 | Stayed at this hotel and from the moment I walked through the doors I was impressed The staff were so friendly and helpf... |
| `AM_POOL` | 2,048 | 0.0413 | 0.5941 | 0.0337 | The hotel is clean and modern with a great rooftop pool and bar. |
| `FAC_VIEW_LOCATION` | 2,669 | 0.0427 | 0.5496 | 0.0323 | The location of the hotel is very central. |
| `FAC_BATH` | 1,902 | 0.0409 | 0.6212 | 0.0305 | The room was very clean and modern. |

</details>

<details>
<summary><strong>8. <code>tripadvisor_123184</code></strong> &mdash; ASC 0.7406 &middot; CEC 0.5347 &middot; 2,347 reviews &middot; [WARN]</summary>

- Sentences: 14,039
- Matched aspects: 29/29
- Weighted CEC: 0.5404
- First review sample: Such a lovely bouquet hotel, clean, good foods with a great service mind. it is located in the foods and shopping area. Very convenient and easy to go everywhere. If I've chance, I will revisit sure!

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 3,662 | 0.0453 | 0.5242 | 0.0352 | The room was clean and very comfortable. |
| `EXP_OVERALL` | 2,466 | 0.0432 | 0.5153 | 0.0347 | In the heart of the Old Quarter in Hanoi, Oriental Suites was our first hotel experience in Vietnam, and the hotel was s... |
| `FAC_BUILDING` | 1,237 | 0.0393 | 0.6560 | 0.0342 | The hotel is located in the heart of the old quarter. |
| `FAC_VIEW_LOCATION` | 1,909 | 0.0417 | 0.5885 | 0.0339 | The hotel is right in the middle of the old quarter and a perfect location. |
| `SER_ATTITUDE` | 3,138 | 0.0445 | 0.5783 | 0.0334 | The staff were very friendly and helpful. |

</details>

<details>
<summary><strong>9. <code>tripadvisor_125833</code></strong> &mdash; ASC 0.7236 &middot; CEC 0.5251 &middot; 2,337 reviews &middot; [WARN]</summary>

- Sentences: 13,706
- Matched aspects: 29/29
- Weighted CEC: 0.5329
- First review sample: The staff at O Gallery are very friendly and made me feel like an A-list celebrity, especially the receptionist Bora. She double-checked with my tour company that they were picking me up for my tour t...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `EXP_OVERALL` | 2,398 | 0.0436 | 0.4774 | 0.0332 | The O’Gallery hotel was the first hotel we stayed in for our honeymoon and it was a great way to start our trip. |
| `FAC_BUILDING` | 959 | 0.0384 | 0.5929 | 0.0322 | The hotel is located in the old quarter of Hanoi. |
| `FAC_VIEW_LOCATION` | 1,536 | 0.0411 | 0.5735 | 0.0321 | The hotel is in a great location and the rooms were very nice. |
| `AM_POOL` | 1,081 | 0.0391 | 0.6516 | 0.0316 | The hotel was very nice and the room was very clean and comfortable. |
| `AM_FOOD` | 2,444 | 0.0437 | 0.5282 | 0.0314 | The hotel was very clean, the food at the breakfast buffet was delicious and it was located in a great location. |

</details>

<details>
<summary><strong>10. <code>tripadvisor_458627</code></strong> &mdash; ASC 0.7303 &middot; CEC 0.5080 &middot; 2,305 reviews &middot; [WARN]</summary>

- Sentences: 12,159
- Matched aspects: 29/29
- Weighted CEC: 0.5103
- First review sample: The Hotel is located is a very nice area, everything just walking distance to locals treats and Mall. The Staff is the best! Breakfast was delicious and has many options with the locals food. I would ...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `EXP_OVERALL` | 1,866 | 0.0421 | 0.5227 | 0.0336 | Our stay at the Park Hyatt Saigon was amazing, this is the best hotel we have stayed in. |
| `AM_FOOD` | 2,465 | 0.0437 | 0.5420 | 0.0320 | The hotel is beautiful, the service was excellent and the restaurants were very good. |
| `AM_POOL` | 1,409 | 0.0405 | 0.5387 | 0.0318 | And the pool! |
| `FAC_VIEW_LOCATION` | 1,739 | 0.0417 | 0.5468 | 0.0316 | The location is perfect and the hotel is beautiful. |
| `FAC_BUILDING` | 860 | 0.0378 | 0.4749 | 0.0308 | My first visit to saigon and stayed at the hotel top location in city centre next to the opera house and the property it... |

</details>


## 6. Re-run Command

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\scripts
python .\log_10_hotels_pipeline.py --input-csv ..\..\hotel_review3.csv --limit 10
```
