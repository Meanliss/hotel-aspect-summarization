# Trained SemAE Aspect Inference Report &mdash; `space_hasos_full_e20_abstractive`

Auto-generated summary of the **trained SemAE model** outputs (VQ-VAE + KL-divergence ranking).
These replace the earlier TF-IDF baseline scoring run in `outputs/hasos_english_only/`.

## 1. Run overview

- Source data: `data/hasos/hasos_summ.json` (50 entities)
- Model: trained over 10 epochs on GPU (see `logs/train_hasos_run1.log`)
- Aspects: 29
- Total summary files: 1450
- Empty outputs: 80
- Generation wall time: ~22 min on 4 parallel shards (RTX 3500 Ada 12GB)

## 2. Per-aspect statistics

Aspects ordered by `first_sent_uniqueness` (lower = more boilerplate, higher = more diverse).

| Group | Aspect | Entities | Empty | Avg words | Unique first sentences | Uniqueness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SERVICE | `SER_SUPPORT` | 50 | 0 | 22.5 | 46/50 | 0.92 |
| EXPERIENCE | `EXP_SAFETY` | 50 | 2 | 20.5 | 45/48 | 0.94 |
| AMENITY | `AM_ENT` | 50 | 0 | 23.9 | 47/50 | 0.94 |
| SERVICE | `SER_OPERATION` | 50 | 0 | 24.6 | 47/50 | 0.94 |
| SERVICE | `SER_COMM` | 50 | 9 | 19.0 | 39/41 | 0.95 |
| BRANDING | `BRA_LUXURY` | 50 | 4 | 28.6 | 44/46 | 0.96 |
| FACILITY | `FAC_CLIMATE` | 50 | 3 | 23.7 | 45/47 | 0.96 |
| BRANDING | `BRA_REPUTE` | 50 | 0 | 23.7 | 48/50 | 0.96 |
| FACILITY | `FAC_ENV` | 50 | 0 | 28.5 | 48/50 | 0.96 |
| LOYALTY | `LOY_RECOMMEND` | 50 | 0 | 14.7 | 48/50 | 0.96 |
| SERVICE | `SER_ATTITUDE` | 50 | 0 | 18.4 | 48/50 | 0.96 |
| FACILITY | `FAC_SECURITY` | 50 | 12 | 22.5 | 37/38 | 0.97 |
| EXPERIENCE | `EXP_VALUE` | 50 | 0 | 21.0 | 49/50 | 0.98 |
| FACILITY | `FAC_BUILDING` | 50 | 0 | 27.1 | 49/50 | 0.98 |
| AMENITY | `AM_FOOD` | 50 | 0 | 22.5 | 50/50 | 1.00 |
| AMENITY | `AM_POOL` | 50 | 10 | 25.6 | 40/40 | 1.00 |
| AMENITY | `AM_ROOM_UTIL` | 50 | 0 | 25.6 | 50/50 | 1.00 |
| AMENITY | `AM_TRANSPORT` | 50 | 0 | 22.8 | 50/50 | 1.00 |
| AMENITY | `AM_UTILITY` | 50 | 18 | 22.0 | 32/32 | 1.00 |
| AMENITY | `AM_WELLNESS` | 50 | 12 | 22.2 | 38/38 | 1.00 |
| AMENITY | `AM_WIFI` | 50 | 0 | 23.4 | 50/50 | 1.00 |
| EXPERIENCE | `EXP_EMOTION` | 50 | 2 | 21.7 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_OVERALL` | 50 | 0 | 17.1 | 50/50 | 1.00 |
| FACILITY | `FAC_BATH` | 50 | 0 | 25.0 | 50/50 | 1.00 |
| FACILITY | `FAC_INTERIOR` | 50 | 0 | 23.1 | 50/50 | 1.00 |
| FACILITY | `FAC_ROOM` | 50 | 0 | 25.9 | 50/50 | 1.00 |
| FACILITY | `FAC_VIEW_LOCATION` | 50 | 0 | 26.3 | 50/50 | 1.00 |
| LOYALTY | `LOY_PREFERENCE` | 50 | 8 | 20.3 | 42/42 | 1.00 |
| LOYALTY | `LOY_RETURN` | 50 | 0 | 17.3 | 50/50 | 1.00 |

## 3. Cross-aspect duplicate first sentences

_Same first sentence reused by ≥2 aspects across entities. With the trained model this is much rarer than the TF-IDF baseline._

Total cross-aspect duplicate sentences: **67**

| # | First sentence (truncated) | Aspects |
| ---: | --- | --- |
| 1 | this hotel is a great place to stay | `AM_ENT`, `AM_FOOD`, `AM_ROOM_UTIL`, `AM_UTILITY`, `AM_WIFI`, `EXP_OVERALL`, `EXP_SAFETY`, `EXP_VALUE`, `FAC_BUILDING`, `FAC_CLIMATE`, `FAC_ENV`, `FAC_INTERIOR`, `SER_COMM`, `SER_OPERATION`, `SER_SUPPORT` |
| 2 | this hotel is a great place to stay. | `AM_ENT`, `AM_WELLNESS`, `EXP_EMOTION`, `FAC_BUILDING`, `FAC_INTERIOR`, `LOY_RETURN` |
| 3 | the hotel is a good value for money. | `AM_ENT`, `AM_ROOM_UTIL`, `AM_WIFI`, `FAC_BUILDING`, `FAC_CLIMATE`, `FAC_INTERIOR` |
| 4 | i'm a big fan of this hotel | `AM_POOL`, `BRA_REPUTE`, `FAC_CLIMATE`, `FAC_SECURITY`, `SER_COMM`, `SER_OPERATION` |
| 5 | this is a very good hotel | `AM_ROOM_UTIL`, `EXP_SAFETY`, `FAC_SECURITY`, `SER_COMM`, `SER_OPERATION`, `SER_SUPPORT` |
| 6 | the hotel is a great place to stay. | `AM_ENT`, `AM_FOOD`, `AM_WIFI`, `EXP_SAFETY` |
| 7 | i'm not a fan of this hotel | `AM_ENT`, `FAC_CLIMATE`, `SER_COMM`, `SER_OPERATION` |
| 8 | we had a great time at the hotel. | `AM_FOOD`, `EXP_EMOTION`, `EXP_SAFETY`, `SER_SUPPORT` |
| 9 | the hotel is a good place to stay | `EXP_SAFETY`, `FAC_ENV`, `SER_COMM`, `SER_SUPPORT` |
| 10 | we stayed at the venetian | `EXP_SAFETY`, `FAC_BUILDING`, `FAC_CLIMATE`, `SER_SUPPORT` |
| 11 | we stayed at this hotel for a couple of nights | `FAC_BATH`, `FAC_BUILDING`, `FAC_CLIMATE`, `FAC_ENV` |
| 12 | the hotel is a good value for the money. | `AM_ROOM_UTIL`, `AM_TRANSPORT`, `AM_WIFI` |
| 13 | this is a summary of the hotel review evidence. | `AM_TRANSPORT`, `EXP_SAFETY`, `LOY_PREFERENCE` |
| 14 | i'm not a big fan of this hotel. | `AM_WIFI`, `SER_OPERATION`, `SER_SUPPORT` |
| 15 | the hotel is a great place to stay | `BRA_REPUTE`, `EXP_SAFETY`, `SER_SUPPORT` |
| 16 | the staff was very friendly and helpful. | `SER_ATTITUDE`, `SER_OPERATION`, `SER_SUPPORT` |
| 17 | the tv in the bedroom was situated high in the corner of the bedroom on the type of platform you would find in a hospita… | `AM_ENT`, `AM_ROOM_UTIL` |
| 18 | good amenities except only 2/3 tv channels n like all hotels in italy the tvs were like 8 inches! | `AM_ENT`, `AM_ROOM_UTIL` |
| 19 | we had a small balcony but had plastic furniture and latice on the floor (not nice) the dresser, desk and minibar are st… | `AM_ENT`, `AM_ROOM_UTIL` |
| 20 | very poor in every respect, bar closed early, barman had to look up how to make a cocktail, cold water in the shower, tv… | `AM_ENT`, `AM_ROOM_UTIL` |
| … | _47 more_ | |

## 4. Sample summaries

<details><summary><code>AM_ENT</code> (AMENITY)</summary>

- `dev_100585`: Each room is named after a Washington winery, and one of
- `dev_100597`: We stayed at this hotel for the first time, and it was a good experience.
- `dev_1029276`: Not the best area at night TV cable kept going out Conference rooms are cold

</details>

<details><summary><code>AM_FOOD</code> (AMENITY)</summary>

- `dev_100585`: We had a great time at the hotel.
- `dev_100597`: The food was good and the service was efficient.
- `dev_1029276`: The buffet breakfast was very good.

</details>

<details><summary><code>AM_POOL</code> (AMENITY)</summary>

- `dev_100585`: You can purchase a pass for a great fitness center with steam room and lap pool
- `dev_100597`: The swimming pool looked nice but we didn't get the chance to swim
- `dev_1029276`: The pool area was actually a nice surprise as it was spacious and well furnished with a very friendly full service pool bar and grill

</details>

<details><summary><code>AM_ROOM_UTIL</code> (AMENITY)</summary>

- `dev_100585`: I filled out one of their comment cards with a minor complaint (we found the
- `dev_100597`: We had a large room with a balcony, comfortable beds, an easy chair and ottoman, and a coffee maker - all things we don't always find.
- `dev_1029276`: Room was great, very large corner room, nice flat panel TV, fridge, the 2 outer walls were basically floor to ceiling windows with amazing views

</details>

<details><summary><code>AM_TRANSPORT</code> (AMENITY)</summary>

- `dev_100585`: The parking was very tight to get in off the street and into the underground where the valet was.
- `dev_100597`: - The shuttle will drop you at the airport train station if you are going to Seattle, but on your way back you must walk from the station to the shuttle bus pick up area at the airport(about a 5 to
- `dev_1029276`: We didn't stay there long, it was

</details>

<details><summary><code>AM_UTILITY</code> (AMENITY)</summary>

- `dev_100585`: I was able to do laundry.
- `dev_100597`: A few weeks ago I visited this beautiful hotel and I stayed only one night because I was taking a cruise.
- `dev_1029276`: The hotel is not a good place to stay.

</details>

<details><summary><code>AM_WELLNESS</code> (AMENITY)</summary>

- `dev_100585`: They made a good effort to find a late yoga class for me, but without luck (Seattle shuts down early).
- `dev_100597`: Very nice hotel, very close to the airport, walkable in fact to the terminals in less than 10 minutes or one can take a
- `dev_1029276`: I still think this hotel is very nice with the size of the room, lobby area, swimming pool and the gym.

</details>

<details><summary><code>AM_WIFI</code> (AMENITY)</summary>

- `dev_100585`: This hotel is a great place to stay, especially if you are a member of InTouch.
- `dev_100597`: Internet available in the room for a fee or free wifi in lobby
- `dev_1029276`: The executive lounge had new Dell computers, wifi internet access was available throughout the hotel (including the pool area) -- free for gold and diamond members

</details>

<details><summary><code>BRA_LUXURY</code> (BRANDING)</summary>

- `dev_100585`: I probably wouldn't have stayed at someplace so upscale had I not lucked out with a great rate on Expedia ($112), but I was thrilled that I was able to because the location can't be beat.
- `dev_100597`: I asked for a room in their Tower, but was told that Priceline customers had to pay a ten dollar a night premium, which I GLADLY paid
- `dev_1029276`: So, if you are staying one night and then heading to the cruise terminal it is fine...anything more than that look at a more upscale place in a better area of town

</details>

<details><summary><code>BRA_REPUTE</code> (BRANDING)</summary>

- `dev_100585`: The main reason we chose this hotel was Kimpton has a great reputation and their Seattle hotel was beyond our expectations.
- `dev_100597`: It's an ok hotel by any other standards but nice compared to what's close by
- `dev_1029276`: We felt that an establishment of this nature in this location and carrying the Hilton brand should have made staff available to man the lounge at the times stated.

</details>

<details><summary><code>EXP_EMOTION</code> (EXPERIENCE)</summary>

- `dev_100585`: The staff is friendly and helpful.
- `dev_100597`: The hotel itself is a bit outdated, but the room we stayed in (in the tower) was very pleasant, relaxing and modern.
- `dev_1029276`: I liked the comfort of the beds, the flow of water in the shower, and the helpfulness

</details>

<details><summary><code>EXP_OVERALL</code> (EXPERIENCE)</summary>

- `dev_100585`: I will stay here again without hesitation and recommend it to anyone traveling to Seattle.
- `dev_100597`: Stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I

</details>

<details><summary><code>EXP_SAFETY</code> (EXPERIENCE)</summary>

- `dev_100585`: I didn't like the room
- `dev_100597`: This is a summary of the hotel review evidence, not the instructions.
- `dev_1029276`: You get a safe way to and from the SOBE.

</details>

<details><summary><code>EXP_VALUE</code> (EXPERIENCE)</summary>

- `dev_100585`: I would be prepared to pay full price for this hotel, considering the service, location, and overall quality.
- `dev_100597`: This hotel is a good value for money
- `dev_1029276`: Great value for the money we spent and not that far from airport

</details>

<details><summary><code>FAC_BATH</code> (FACILITY)</summary>

- `dev_100585`: The room was a little small but clean, and very nicely done with shiny black tile floor, granite countertop, fluffy towels and those nice Aveda toiletries.
- `dev_100597`: We stayed at this hotel for a couple of nights
- `dev_1029276`: The hotel room is well-appointed, clean, and contains the new amenities at Hilton - Lavazza in-room coffee,

</details>

<details><summary><code>FAC_BUILDING</code> (FACILITY)</summary>

- `dev_100585`: The decor was very "posh" it is probably liked by most that
- `dev_100597`: The elevators were quite cold
- `dev_1029276`: A huge group of teenagers were staying in the hotel, and kids were all over the place until late at night...flooding the lobby, elevators, and public space.

</details>

<details><summary><code>FAC_CLIMATE</code> (FACILITY)</summary>

- `dev_100585`: The bed was comfortable but between the noise of the traffic ( not the jets) and the room fan, we did not sleep well
- `dev_100597`: The air conditioning system (not the ac) was loud and could not be turned off...I am not the type that needs total silence to sleep but I was counting the days to get the he'll outta here.
- `dev_1029276`: Usually stay at Sheraton

</details>

<details><summary><code>FAC_ENV</code> (FACILITY)</summary>

- `dev_100585`: The hotel and rooms are gorgeous, and the location is perfect--downtown but on a fairly quiet street
- `dev_100597`: The room was very clean, cozy and very quiet even though it was so close to the airport
- `dev_1029276`: I was 5 rooms away from the elevator and the noise in the hall of just the elevator was annoying

</details>

<details><summary><code>FAC_INTERIOR</code> (FACILITY)</summary>

- `dev_100585`: The wine tasting was lovely every evening, great location, very confortable beds/pillows, very clean, homey atmosphere, helpful
- `dev_100597`: The hotel has seen its newer days: the carpets are threadbare in some spots and the cement balconies are noticeably empty of furniture
- `dev_1029276`: The hotel was full and jumping with lively but good natured Colts fans and had a fantastic atmosphere.

</details>

<details><summary><code>FAC_ROOM</code> (FACILITY)</summary>

- `dev_100585`: I had a room with a view of the parking lot, I didn't like it.
- `dev_100597`: We had a balcony, the beds were super comfortable, room was a good size and clean.
- `dev_1029276`: We had a room with a great view, and my room was very clean and equipped appropriately

</details>

<details><summary><code>FAC_SECURITY</code> (FACILITY)</summary>

- `dev_100585`: This hotel was not a good idea for a single female traveler.
- `dev_100597`: The area is not safe to walk.
- `dev_1029276`: Security was very good as you need to use your room key key to use the elevators and to enter the parking lot.

</details>

<details><summary><code>FAC_VIEW_LOCATION</code> (FACILITY)</summary>

- `dev_100585`: Hotel Vintage Park is centrally located in downtown Seattle, it was very easy to get to the shopping areas/Pike Place Market and a few good restaurants.
- `dev_100597`: The view was great.
- `dev_1029276`: The Downtown Miami Hilton is on the fringe of the downtown area but very close to the light rail and a short drive to S

</details>

<details><summary><code>LOY_PREFERENCE</code> (LOYALTY)</summary>

- `dev_100585`: I travel to Seattle for business once or twice a month
- `dev_100597`: The Doubletrees is one of my favorite hotels
- `dev_1029276`: The room was a bit dusty and the coffee station did not have stir sticks.

</details>

<details><summary><code>LOY_RECOMMEND</code> (LOYALTY)</summary>

- `dev_100585`: I highly recommend this as your destination hotel any time you travel to Seattle.
- `dev_100597`: I would highly recommend the Doubletree and plan on staying here again on our next trip.
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here again.

</details>

<details><summary><code>LOY_RETURN</code> (LOYALTY)</summary>

- `dev_100585`: I will stay here again without hesitation and recommend it to anyone traveling to Seattle
- `dev_100597`: We stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here again.

</details>

<details><summary><code>SER_ATTITUDE</code> (SERVICE)</summary>

- `dev_100585`: The staff is extremely friendly, takes initiative, and helpful
- `dev_100597`: The front desk staff were grouchy, uninterested and not pleasant to deal with.
- `dev_1029276`: Staff was friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there.

</details>

<details><summary><code>SER_COMM</code> (SERVICE)</summary>

- `dev_100585`: I could not recommend this hotel
- `dev_100597`: This is a very large hotel and very busy with many international guest (we overheard many languages spoken) and eventhough the staff was VERY busy they all had a great attitude and friendly demeanor.
- `dev_1029276`: We waited and waited for almost 3 hours and the whole meal took almost 3 hours.

</details>

<details><summary><code>SER_OPERATION</code> (SERVICE)</summary>

- `dev_100585`: We arrived at the hotel quite early after an Alaskan cruise and the very friendly hotel.
- `dev_100597`: The staff was friendly and helpful and we enjoyed the warm, chocolate chip cookie we were given at check-in.
- `dev_1029276`: I'm a big fan of this hotel

</details>

<details><summary><code>SER_SUPPORT</code> (SERVICE)</summary>

- `dev_100585`: I'm not a big fan of this hotel.
- `dev_100597`: I had to leave early the next morning, I was put into this property by an airline due to a maintenance issue, although I checked in late at night and had to leave late at night, I saw enough that I wa…
- `dev_1029276`: The hotel is a good place to stay

</details>

## 5. How to reproduce

```powershell
$env:PYTHONIOENCODING='utf-8'
cd SemAE\scripts
# 1. Prepare data + seeds
python .\prepare_hasos.py
# 2. Train SemAE (10 epochs on GPU 0)
.\train_hasos.ps1 -Gpu 0 -Epochs 10 -RunId hasos_run1
# 3. Run aspect inference on 4 parallel shards
python .\run_aspect_inference_parallel.py `
    --model ..\models\hasos_run1_10_model.pt `
    --run_id hasos_aspects_run1 `
    --num_shards 4 --gpu 0
```
