# Abstractive Aspect Synthesis Report - `space_hasos_full_e20_ranked_abstractive_ranked`

- Source run: `space_hasos_full_e20_ranked`
- Source mode: `ranked_evidence`
- Model: `google/flan-t5-base`
- Aspect/entity files processed: 1370
- Generated this run: 1368
- Loaded from cache/existing output: 0
- Empty evidence files: 0
- Fallback rows: 2
- Exact copy detections: 148
- Aspects: 29

## Notes

This stage rewrites top-ranked SemAE evidence into short abstractive summaries. In `ranked_evidence` mode, the evidence is captured before extractive summary truncation.

Each row keeps the evidence records used for synthesis so the generated summary can be audited back to ranked SemAE sentences.

## Samples

- `AM_ENT` `dev_100585`: The hotel is a great place to relax and have a good time.
- `AM_ENT` `dev_100597`: The hotel has a good TV, and a nice bathroom with thick towels and plenty of hot water.
- `AM_ENT` `dev_1029276`: Not a bad hotel but not a great place for a business trip.
- `AM_ENT` `dev_1064610`: The location is excellent, next to the financial district, entertainment district, and at the foot of McGill's campus. The rooms are spacious, clean and have all the amenities you need (safe, coffee maker, iron, wifi,...
- `AM_ENT` `dev_1113787`: Our tweens loved the salt water heated infinity pool, the Charlie Palmer bar, awesome room service and gorgeous room......we live in Carrollton, but felt we had traveled to another luxurious country!
- `AM_ENT` `dev_111509`: The room was very clean, tastefully decorated and had great amenities such as Wi-Fi, a flat screen TV, DVD player, etc. It has not been changed since the building was built 20+ years ago, the bathroom tiles are chippe...
- `AM_ENT` `dev_112429`: The room was OK, typical hotel room with queen bed, desk, TV, clock radio, coffee maker. The room was very well decorated, with a big TV, a sleek marble bathroom, a small closet, and a large desk in the room.
- `AM_ENT` `dev_1176198`: The hotel is located in the entertainment district, close to clubs, bars, restaurants, theater, Rogers Center and the CN Tower.
- `AM_ENT` `dev_120274`: Good location on the 192 at the end so nice and quiet (with a mega short cut to Animal Kingdom if anyone wants to know it mail me) right near all the good buffet restaurants, close to Disney and a short drive to Unive...
- `AM_ENT` `dev_121241`: The free breakfast was very good and the entertainment while you eat was a nice touch.
- `AM_ENT` `dev_124849`: The room was very spacious with a separate sitting area with its own dedicated TV in addition to the 24" LCD adjacent to the queen size bed.
- `AM_ENT` `dev_150241`: The hotel is a great place to stay.
- `AM_ENT` `dev_1510471`: The hotel has a great breakfast buffet.
- `AM_ENT` `dev_156564`: The hotel has a lot of amenities, including a flat-screen TV, free Wi-Fi, and a large spa bath.
- `AM_ENT` `dev_1722910`: Good amenities except only 2/3 tv channels n like all hotels in italy the tvs were like 8 inches!
- `AM_ENT` `dev_182002`: The hotel has a pool, a steam room, a gym, a TV, a microwave, a fridge, a coffee maker, a hairdryer, and a Tim Horton's donut shop.
- `AM_ENT` `dev_182519`: It was our first time in Vancouver, and our impression of the entertainment district downtown was a little gritty.
- `AM_ENT` `dev_182542`: The room was very noisy and the TV was not working properly.
- `AM_ENT` `dev_183092`: The decor was tasteful, flat screen tv, VERY soft King bed, marble counter in the bathroom...we were quite impressed.
- `AM_ENT` `dev_183645`: I like watching tv before bed and was very happy that there are not one, but two, count em, two tv's and a balcony.
