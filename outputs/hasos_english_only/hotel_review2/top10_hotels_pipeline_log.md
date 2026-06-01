# Top-10 Hotels Pipeline Log &mdash; `hotel_review2.csv`

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

Input: `C:\Users\dso3hc\Downloads\Implement\hotel_review2.csv`

Language detector: English-like if repaired text has at least 8 alphabetic chars, ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either 2 common English marker hits, or 1 marker plus 1 hotel-domain hit, or 2 hotel-domain hits with ASCII ratio >= 0.94.

ROUGE: not available (no human gold summaries in the source CSV).


## 2. Pipeline Issues Found

Auto-detected from JSON log. **HIGH: 36 | MEDIUM: 42 | LOW: 15**

| Severity | Hotel(s) | Aspect(s) | Issue | Failing Step |
| :---: | --- | --- | --- | --- |
| **HIGH** | bookingnew_265478 | `AM_POOL, AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room was very clean and spacious." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_265478 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was very clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_265478 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "I booked a room at Hanoi Silk hotel center but when we arrive at around 10-11 pm..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_265478 | `AM_ENT, AM_WIFI` | Identical representative sentence reused: "The other issue was, (and this was partially out fault for not doing our researc..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_265478 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "the location is on the beer street, and it is very convenient." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_268022 | `EXP_SAFETY, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The room was clean and very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_268022 | `AM_POOL, AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room was very spacious and clean." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_268022 | `AM_ENT, AM_ROOM_UTIL, AM_UTILITY, AM_WELLNESS, AM_WIFI, FAC_ENV` | Identical representative sentence reused: "The laundry service was great and convenient, the spa was really nice and the ma..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_268022 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The hotel and staff are all of a high quality standard." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_367391 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The room was very clean and comfortable!" | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_367391 | `FAC_BUILDING, FAC_ENV` | Identical representative sentence reused: "The location was in the heart of the old quarter, but the hotel was quiet." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_367391 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The hotel was clean and of high standard." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_367391 | `AM_ENT, AM_WELLNESS` | Identical representative sentence reused: "The hotel was very comfortable with great breakfast and the staff did everything..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_393345 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE` | Identical representative sentence reused: "The room was very clean and comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_393345 | `EXP_EMOTION, FAC_INTERIOR` | Identical representative sentence reused: "The room was very cozy and the location is perfect for trips to the old town of ..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_393345 | `BRA_LUXURY, BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The staff were friendly and helpful It was a little rundown, old and not up to t..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_393345 | `AM_FOOD, EXP_OVERALL` | Identical representative sentence reused: "The staff were amazing and the location was great." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_393345 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "The location in the heart of the Old Quarter was very convenient." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_4040357 | `AM_TRANSPORT, FAC_INTERIOR, FAC_ROOM` | Identical representative sentence reused: "It is in a fantastic location, the hotel is Beautiful and the rooms are clean an..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_4040357 | `FAC_BATH, FAC_ENV` | Identical representative sentence reused: "The hotel is very clean and quiet.The staff are very friendly and informative,Th..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_4040357 | `AM_POOL, EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was comfortable and clean." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_4040357 | `AM_ENT, BRA_REPUTE` | Identical representative sentence reused: "The room was perfect they upgraded to a suite with jacuzzi so fabulous and the b..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5433446 | `AM_TRANSPORT, FAC_BATH` | Identical representative sentence reused: "The room was clean and spacious." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5937937 | `BRA_LUXURY, FAC_INTERIOR` | Identical representative sentence reused: "The pool is lovely - a beautiful view of the city and the room was so luxurious." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5937937 | `EXP_SAFETY, FAC_CLIMATE` | Identical representative sentence reused: "The room was comfortable and very clean." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5937937 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "Breakfast wasn't up to the standard of a 5 star hotel." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5937937 | `EXP_OVERALL, FAC_VIEW_LOCATION` | Identical representative sentence reused: "And the pool was amazing !" | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_5937937 | `AM_UTILITY, AM_WIFI` | Identical representative sentence reused: "The lifts are very slow." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_6105345 | `EXP_SAFETY, FAC_BATH, FAC_CLIMATE, FAC_ROOM` | Identical representative sentence reused: "The rooms were clean and the bed was very comfortable." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_6105345 | `BRA_REPUTE, FAC_SECURITY` | Identical representative sentence reused: "The food at all restaurants is of a very high standard and served by amazing sta..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_6105345 | `AM_FOOD, EXP_OVERALL` | Identical representative sentence reused: "The rooftop bar was amazing, the breakfast was great and staff were very friendl..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_6105345 | `AM_ROOM_UTIL, AM_UTILITY` | Identical representative sentence reused: "Location of the hotel was very convenient." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_7384631 | `EXP_SAFETY, FAC_BATH, FAC_CLIMATE` | Identical representative sentence reused: "The room was comfortable and clean!" | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_7384631 | `AM_FOOD, AM_TRANSPORT, AM_UTILITY, FAC_VIEW_LOCATION` | Identical representative sentence reused: "The rooms were clean and comfortable, and the staff was very friendly and helpfu..." | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_9074874 | `AM_POOL, EXP_SAFETY, FAC_BATH, FAC_CLIMATE` | Identical representative sentence reused: "The room was clean and very comfortable!" | 8 (TF-IDF centroid) |
| **HIGH** | bookingnew_9074874 | `AM_ROOM_UTIL, AM_TRANSPORT, AM_UTILITY` | Identical representative sentence reused: "The staff was lovely and the location was very convenient." | 8 (TF-IDF centroid) |
| **MEDIUM** | 4 hotels | `SER_ATTITUDE` | Generic boilerplate repeated: "the staff were very friendly and helpful." | 8 (no novelty filter) |
| **MEDIUM** | bookingnew_265478 | `FAC_INTERIOR` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_265478 | `FAC_SECURITY` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_265478 | `BRA_REPUTE` | Summary mixes 3 topics (bath, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_265478 | `BRA_LUXURY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `FAC_ENV` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_WIFI` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_FOOD` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_WELLNESS` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_ENT` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_ROOM_UTIL` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `AM_UTILITY` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_268022 | `EXP_OVERALL` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_367391 | `FAC_INTERIOR` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_367391 | `AM_FOOD` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_393345 | `AM_ENT` | Summary mixes 4 topics (bath, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_4040357 | `FAC_BATH` | Summary mixes 4 topics (bath, food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_4040357 | `FAC_ENV` | Summary mixes 4 topics (bath, food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_4040357 | `SER_OPERATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_4040357 | `EXP_EMOTION` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5433446 | `FAC_ROOM` | Summary mixes 3 topics (food, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5433446 | `AM_POOL` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5937937 | `FAC_INTERIOR` | Summary mixes 3 topics (location, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5937937 | `FAC_BUILDING` | Summary mixes 4 topics (food, pool, staff, wifi) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5937937 | `AM_ROOM_UTIL` | Summary mixes 3 topics (food, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5937937 | `BRA_LUXURY` | Summary mixes 3 topics (location, pool, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_5937937 | `LOY_PREFERENCE` | Summary mixes 3 topics (food, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `AM_WIFI` | Summary mixes 3 topics (food, location, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `AM_FOOD` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `AM_ENT` | Summary mixes 3 topics (pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `SER_OPERATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `EXP_OVERALL` | Summary mixes 3 topics (food, pool, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_6105345 | `BRA_LUXURY` | Summary mixes 4 topics (food, pool, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `FAC_VIEW_LOCATION` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `AM_FOOD` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `AM_ENT` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `AM_TRANSPORT` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `AM_UTILITY` | Summary mixes 4 topics (food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_7384631 | `SER_OPERATION` | Summary mixes 5 topics (bath, food, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_9074874 | `AM_WIFI` | Summary mixes 4 topics (bath, location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_9074874 | `AM_ENT` | Summary mixes 3 topics (bath, location, room) | 5+8 (over-match + non-discriminative centroid) |
| **MEDIUM** | bookingnew_9074874 | `BRA_LUXURY` | Summary mixes 3 topics (location, room, staff) | 5+8 (over-match + non-discriminative centroid) |
| **LOW** | 10 hotels | `AM_ENT` | CEC below 0.50 (avg 0.441) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 10 hotels | `SER_OPERATION` | CEC below 0.50 (avg 0.429) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 3 hotels | `FAC_INTERIOR` | CEC below 0.50 (avg 0.461) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 3 hotels | `SER_SUPPORT` | CEC below 0.50 (avg 0.482) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `FAC_ENV` | CEC below 0.50 (avg 0.472) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 4 hotels | `AM_WELLNESS` | CEC below 0.50 (avg 0.465) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 5 hotels | `AM_ROOM_UTIL` | CEC below 0.50 (avg 0.460) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `AM_UTILITY` | CEC below 0.50 (avg 0.468) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `SER_COMM` | CEC below 0.50 (avg 0.431) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 6 hotels | `BRA_LUXURY` | CEC below 0.50 (avg 0.440) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 7 hotels | `LOY_PREFERENCE` | CEC below 0.50 (avg 0.474) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `EXP_EMOTION` | CEC below 0.50 (avg 0.457) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 8 hotels | `BRA_REPUTE` | CEC below 0.50 (avg 0.464) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `FAC_SECURITY` | CEC below 0.50 (avg 0.448) — weak evidence coverage | 7+8 (clustering/centroid) |
| **LOW** | 9 hotels | `AM_WIFI` | CEC below 0.50 (avg 0.464) — weak evidence coverage | 7+8 (clustering/centroid) |


## 3. Selected Hotels

Thresholds — PASS: ASC >= 0.73 AND CEC >= 0.54; FAIL: ASC < 0.7 AND CEC < 0.5; WARN otherwise.

| # | Hotel ID | Reviews | Sentences | Aspects | ASC | Macro CEC | Weighted CEC | Status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `bookingnew_5937937` | 3,276 | 9,681 | 29/29 | 0.7302 | 0.5295 | 0.5397 | [WARN] |
| 2 | `bookingnew_367391` | 2,464 | 8,030 | 29/29 | 0.7186 | 0.5575 | 0.5670 | [WARN] |
| 3 | `bookingnew_265478` | 1,785 | 5,196 | 29/29 | 0.6938 | 0.5457 | 0.5571 | [WARN] |
| 4 | `bookingnew_5433446` | 1,753 | 6,079 | 29/29 | 0.7287 | 0.5230 | 0.5337 | [WARN] |
| 5 | `bookingnew_393345` | 1,646 | 4,730 | 29/29 | 0.7225 | 0.5542 | 0.5662 | [WARN] |
| 6 | `bookingnew_268022` | 1,641 | 5,828 | 29/29 | 0.7547 | 0.5549 | 0.5696 | [PASS] |
| 7 | `bookingnew_4040357` | 1,636 | 6,768 | 29/29 | 0.7514 | 0.5434 | 0.5528 | [PASS] |
| 8 | `bookingnew_7384631` | 1,634 | 4,926 | 29/29 | 0.7391 | 0.5373 | 0.5516 | [WARN] |
| 9 | `bookingnew_6105345` | 1,593 | 5,201 | 29/29 | 0.7030 | 0.5287 | 0.5387 | [WARN] |
| 10 | `bookingnew_9074874` | 1,582 | 4,577 | 29/29 | 0.7202 | 0.5424 | 0.5530 | [WARN] |


## 4. Step-by-Step Trace

Single example: hotel #1 `bookingnew_5937937` (top aspect `AM_POOL`).

| Step | Transform | Input | Output | Sample evidence | Health |
| ---: | --- | --- | --- | --- | :---: |
| 1 | Load CSV | Raw rows | 3,276 reviews kept | Air conditioned rooms kept clean daily, pool with amazing view, very friendly staff Location was a bit far out of town f... | OK |
| 2 | Group by hotel_id | ref_id without numeric suffix | Hotel `bookingnew_5937937` | 3,276 rows | OK |
| 3 | Sentence split | Per-review splitter | 9,681 sentences | Air conditioned rooms kept clean daily, pool with amazing view, very friendly staff Location was a bit far out of town f | OK |
| 4 | Normalize | lowercase, strip accents | Normalized text | air conditioned rooms kept clean daily, pool with amazing view, very friendly staff location was a bit far out of town f... | OK |
| 5 | Aspect match | Keyword vs HASOS taxonomy | 29/29 aspects hit | Top: `AM_POOL` | WARN: over-matches |
| 6 | Dedup | Unique opinions per aspect | 2,663 unique / 2,716 matched | — | OK |
| 7 | Cluster weight | log(1+n) normalized | weight = 0.0462 | — | OK |
| 8 | Representative | TF-IDF centroid pick | 1 sentence per aspect | The rooftop pool and the amazing. | FAIL: not discriminative |
| 9 | Metrics | CEC / ASC | ASC 0.7302 / CEC 0.5295 | — | WARN: inherits step-8 noise |


## 5. Per-Hotel Details

_Click a hotel to expand._

<details>
<summary><strong>1. <code>bookingnew_5937937</code></strong> &mdash; ASC 0.7302 &middot; CEC 0.5295 &middot; 3,276 reviews &middot; [WARN]</summary>

- Sentences: 9,681
- Matched aspects: 29/29
- Weighted CEC: 0.5397
- First review sample: Air conditioned rooms kept clean daily, pool with amazing view, very friendly staff Location was a bit far out of town from central tourist locations

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_POOL` | 2,663 | 0.0462 | 0.5909 | 0.0377 | The rooftop pool and the amazing. |
| `AM_FOOD` | 2,419 | 0.0457 | 0.6458 | 0.0354 | The rooftop pool and the buffet breakfast was amazing! |
| `FAC_VIEW_LOCATION` | 1,802 | 0.0439 | 0.6347 | 0.0329 | And the pool was amazing ! |
| `FAC_ROOM` | 2,110 | 0.0449 | 0.5948 | 0.0323 | The room was very comfortable and clean. |
| `SER_ATTITUDE` | 1,769 | 0.0438 | 0.6677 | 0.0316 | The staff were very friendly and helpful. |

</details>

<details>
<summary><strong>2. <code>bookingnew_367391</code></strong> &mdash; ASC 0.7186 &middot; CEC 0.5575 &middot; 2,464 reviews &middot; [WARN]</summary>

- Sentences: 8,030
- Matched aspects: 29/29
- Weighted CEC: 0.5670
- First review sample: The staff was lovely and very accommodating. The breakfast was outstanding - lots of choices, all of which were absolutely delicious. The room was clean and well maintained, and the beds were super co...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 2,465 | 0.0479 | 0.6747 | 0.0371 | The staff were very helpful and friendly. |
| `FAC_ROOM` | 1,910 | 0.0463 | 0.6154 | 0.0357 | The room was very clean and comfortable! |
| `FAC_BUILDING` | 647 | 0.0397 | 0.5892 | 0.0339 | The location was in the heart of the old quarter, but the hotel was quiet. |
| `FAC_VIEW_LOCATION` | 1,634 | 0.0454 | 0.6386 | 0.0336 | The amazing staff and the great location. |
| `AM_FOOD` | 1,740 | 0.0458 | 0.5935 | 0.0330 | The staff, the breakfast and the location. |

</details>

<details>
<summary><strong>3. <code>bookingnew_265478</code></strong> &mdash; ASC 0.6938 &middot; CEC 0.5457 &middot; 1,785 reviews &middot; [WARN]</summary>

- Sentences: 5,196
- Matched aspects: 29/29
- Weighted CEC: 0.5571
- First review sample: Great location and amazing staffs. the hotel is under construction so it smells bad and a little messy.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,389 | 0.0483 | 0.7440 | 0.0383 | The staff were friendly and very helpful. |
| `FAC_VIEW_LOCATION` | 1,054 | 0.0465 | 0.5738 | 0.0357 | Very helpful staff and the location was great! |
| `FAC_BUILDING` | 468 | 0.0411 | 0.6432 | 0.0324 | location of the hotel in the old quarter |
| `FAC_ROOM` | 1,641 | 0.0495 | 0.5597 | 0.0317 | The room was very comfortable and clean. |
| `FAC_ENV` | 359 | 0.0393 | 0.5210 | 0.0311 | The location of the hotel was very busy and loud at night. |

</details>

<details>
<summary><strong>4. <code>bookingnew_5433446</code></strong> &mdash; ASC 0.7287 &middot; CEC 0.5230 &middot; 1,753 reviews &middot; [WARN]</summary>

- Sentences: 6,079
- Matched aspects: 29/29
- Weighted CEC: 0.5337
- First review sample: Clean , very welcoming staff. Great room and versatile breakfast. 10 minutes walk from the city center 
Great experience Nothing

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,442 | 0.0468 | 0.6636 | 0.0373 | The staff were very friendly and helpful. |
| `FAC_VIEW_LOCATION` | 1,061 | 0.0448 | 0.5791 | 0.0353 | The location and the staff is great. |
| `FAC_ROOM` | 1,613 | 0.0475 | 0.5580 | 0.0347 | The rooms were spacious, the breakfast was great and the hotel was in a great location! |
| `FAC_BUILDING` | 481 | 0.0397 | 0.5704 | 0.0334 | The hotel is in a great location and a short walk to the center of Old Town. |
| `AM_FOOD` | 1,248 | 0.0459 | 0.5499 | 0.0334 | The breakfast was amazing ! |

</details>

<details>
<summary><strong>5. <code>bookingnew_393345</code></strong> &mdash; ASC 0.7225 &middot; CEC 0.5542 &middot; 1,646 reviews &middot; [WARN]</summary>

- Sentences: 4,730
- Matched aspects: 29/29
- Weighted CEC: 0.5662
- First review sample: It’s a great location with very friendly staff.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,289 | 0.0492 | 0.7298 | 0.0394 | The staff was very friendly and helpful. |
| `AM_FOOD` | 941 | 0.0470 | 0.5974 | 0.0360 | The staff were amazing and the location was great. |
| `FAC_VIEW_LOCATION` | 919 | 0.0469 | 0.6346 | 0.0352 | The staff and the location is great. |
| `FAC_ROOM` | 1,357 | 0.0496 | 0.5963 | 0.0323 | The room was very clean and the bed was comfortable. |
| `AM_POOL` | 363 | 0.0405 | 0.6843 | 0.0315 | The room was very clean and comfortable. |

</details>

<details>
<summary><strong>6. <code>bookingnew_268022</code></strong> &mdash; ASC 0.7547 &middot; CEC 0.5549 &middot; 1,641 reviews &middot; [PASS]</summary>

- Sentences: 5,828
- Matched aspects: 29/29
- Weighted CEC: 0.5696
- First review sample: Cảm ơn các bạn rất nhiều 
Fantastic hotel and service !!!

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,391 | 0.0472 | 0.7242 | 0.0370 | The staff were very friendly and helpful. |
| `AM_FOOD` | 1,107 | 0.0458 | 0.6120 | 0.0363 | The breakfast was very good, the staff were helpful and friendly and the room was very comfortable! |
| `AM_POOL` | 898 | 0.0444 | 0.6664 | 0.0353 | The room was very spacious and clean. |
| `FAC_BUILDING` | 475 | 0.0402 | 0.5279 | 0.0351 | The location of the hotel is well located between the beach and the old town. |
| `FAC_VIEW_LOCATION` | 775 | 0.0434 | 0.5472 | 0.0345 | the staff and location was great. |

</details>

<details>
<summary><strong>7. <code>bookingnew_4040357</code></strong> &mdash; ASC 0.7514 &middot; CEC 0.5434 &middot; 1,636 reviews &middot; [PASS]</summary>

- Sentences: 6,768
- Matched aspects: 29/29
- Weighted CEC: 0.5528
- First review sample: Excellent hotel located in the best possible location in Hanoi. The rooms were outstanding, the restaurant were so good but the best aspect was the service. The staff treated you like you were a VIP i...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `FAC_ROOM` | 1,875 | 0.0471 | 0.5905 | 0.0381 | It is in a fantastic location, the hotel is Beautiful and the rooms are clean and spacious and very comfortable. |
| `FAC_BATH` | 811 | 0.0419 | 0.6245 | 0.0365 | The hotel is very clean and quiet.The staff are very friendly and informative,The room was very clean and quite and a go... |
| `AM_POOL` | 631 | 0.0403 | 0.6235 | 0.0356 | The room was comfortable and clean. |
| `FAC_VIEW_LOCATION` | 1,228 | 0.0445 | 0.5741 | 0.0348 | Breakfast was very good and the location of the hotel was perfect for walking around in the Old Quarter. |
| `SER_ATTITUDE` | 1,556 | 0.0460 | 0.6892 | 0.0343 | The staff were friendly and very helpful. |

</details>

<details>
<summary><strong>8. <code>bookingnew_7384631</code></strong> &mdash; ASC 0.7391 &middot; CEC 0.5373 &middot; 1,634 reviews &middot; [WARN]</summary>

- Sentences: 4,926
- Matched aspects: 29/29
- Weighted CEC: 0.5516
- First review sample: We had a great experience. Especially Anna und Emma were so lovely and the helped us a lot during our stay! 5 stars and 100% recommondation.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,399 | 0.0479 | 0.6850 | 0.0362 | The staff was very friendly and helpful! |
| `FAC_VIEW_LOCATION` | 959 | 0.0454 | 0.5866 | 0.0358 | The rooms were clean and comfortable, and the staff was very friendly and helpful: Anna, Mila.The location was convenien... |
| `AM_FOOD` | 932 | 0.0452 | 0.5693 | 0.0355 | The rooms were clean and comfortable, and the staff was very friendly and helpful: Anna, Mila.The location was convenien... |
| `FAC_BATH` | 628 | 0.0426 | 0.6390 | 0.0349 | The room was comfortable and clean! |
| `FAC_ROOM` | 1,357 | 0.0477 | 0.6214 | 0.0347 | The room was clean and the location was great. |

</details>

<details>
<summary><strong>9. <code>bookingnew_6105345</code></strong> &mdash; ASC 0.7030 &middot; CEC 0.5287 &middot; 1,593 reviews &middot; [WARN]</summary>

- Sentences: 5,201
- Matched aspects: 29/29
- Weighted CEC: 0.5387
- First review sample: Schon beim Empfang im Hotel bekamen wir eine Vorahnung wie schön unser Urlaub werden sollte, doch wurde dieses Gefühl im weiteren Verlauf noch weit übertroffen. Während der ganzen Zeit wurden uns quas...

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `AM_FOOD` | 1,415 | 0.0475 | 0.6365 | 0.0364 | The rooftop bar was amazing, the breakfast was great and staff were very friendly and helpful. |
| `FAC_VIEW_LOCATION` | 1,140 | 0.0461 | 0.6251 | 0.0353 | the staff was amazing, the hotel is beautiful |
| `SER_ATTITUDE` | 1,211 | 0.0465 | 0.6620 | 0.0348 | The staff were very friendly and helpful. |
| `AM_POOL` | 722 | 0.0431 | 0.5576 | 0.0325 | the room was lovely and the rooftop pool and bar was amazing! |
| `FAC_BUILDING` | 480 | 0.0404 | 0.6188 | 0.0320 | Location of hotel in the Old Quarter. |

</details>

<details>
<summary><strong>10. <code>bookingnew_9074874</code></strong> &mdash; ASC 0.7202 &middot; CEC 0.5424 &middot; 1,582 reviews &middot; [WARN]</summary>

- Sentences: 4,577
- Matched aspects: 29/29
- Weighted CEC: 0.5530
- First review sample: Very nice large room and comfy beds.

**Top 5 aspects by ASC contribution**

| Aspect | Unique opinions | Weight | CEC | ASC contribution | Representative summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `SER_ATTITUDE` | 1,323 | 0.0487 | 0.7314 | 0.0386 | The staff was very helpful and friendly. |
| `FAC_ROOM` | 1,391 | 0.0491 | 0.5969 | 0.0360 | The room was clean and comfortable, and the staff was very helpful and friendly. |
| `FAC_VIEW_LOCATION` | 855 | 0.0458 | 0.6193 | 0.0357 | The hotel is in a great location in the old quarter and the staff are so friendly and helpful. |
| `FAC_BUILDING` | 417 | 0.0409 | 0.5996 | 0.0335 | It is in a great location in the heart of the Old Quarter. |
| `FAC_BATH` | 663 | 0.0441 | 0.6265 | 0.0324 | The room was clean and very comfortable! |

</details>


## 6. Re-run Command

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\scripts
python .\log_10_hotels_pipeline.py --input-csv ..\..\hotel_review2.csv --limit 10
```
