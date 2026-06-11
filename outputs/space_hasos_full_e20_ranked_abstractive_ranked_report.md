# Trained SemAE Aspect Inference Report &mdash; `space_hasos_full_e20_ranked_abstractive_ranked`

Auto-generated summary of the **trained SemAE model** outputs (VQ-VAE + KL-divergence ranking).
These replace the earlier TF-IDF baseline scoring run in `outputs/hasos_english_only/`.

## 1. Run overview

- Source data: `data/hasos/hasos_summ.json` (50 entities)
- Model: trained over 10 epochs on GPU (see `logs/train_hasos_run1.log`)
- Aspects: 29
- Total summary files: 1370
- Empty outputs: 0
- Generation wall time: ~22 min on 4 parallel shards (RTX 3500 Ada 12GB)

## 2. Per-aspect statistics

Aspects ordered by `first_sent_uniqueness` (lower = more boilerplate, higher = more diverse).

| Group | Aspect | Entities | Empty | Avg words | Unique first sentences | Uniqueness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AMENITY | `AM_ROOM_UTIL` | 50 | 0 | 9.8 | 15/50 | 0.30 |
| AMENITY | `AM_WIFI` | 50 | 0 | 15.5 | 23/50 | 0.46 |
| AMENITY | `AM_WELLNESS` | 38 | 0 | 17.4 | 23/38 | 0.61 |
| SERVICE | `SER_ATTITUDE` | 50 | 0 | 11.2 | 32/50 | 0.64 |
| SERVICE | `SER_SUPPORT` | 50 | 0 | 23.1 | 42/50 | 0.84 |
| FACILITY | `FAC_ROOM` | 50 | 0 | 32.2 | 45/50 | 0.90 |
| BRANDING | `BRA_LUXURY` | 46 | 0 | 16.8 | 42/46 | 0.91 |
| LOYALTY | `LOY_RECOMMEND` | 50 | 0 | 19.9 | 46/50 | 0.92 |
| SERVICE | `SER_COMM` | 41 | 0 | 20.3 | 38/41 | 0.93 |
| EXPERIENCE | `EXP_SAFETY` | 48 | 0 | 20.0 | 45/48 | 0.94 |
| EXPERIENCE | `EXP_VALUE` | 50 | 0 | 21.9 | 48/50 | 0.96 |
| SERVICE | `SER_OPERATION` | 50 | 0 | 29.4 | 48/50 | 0.96 |
| FACILITY | `FAC_BATH` | 50 | 0 | 27.3 | 49/50 | 0.98 |
| AMENITY | `AM_ENT` | 50 | 0 | 25.1 | 50/50 | 1.00 |
| AMENITY | `AM_FOOD` | 50 | 0 | 22.4 | 50/50 | 1.00 |
| AMENITY | `AM_POOL` | 40 | 0 | 23.3 | 40/40 | 1.00 |
| AMENITY | `AM_TRANSPORT` | 50 | 0 | 25.7 | 50/50 | 1.00 |
| AMENITY | `AM_UTILITY` | 32 | 0 | 24.3 | 32/32 | 1.00 |
| BRANDING | `BRA_REPUTE` | 50 | 0 | 26.6 | 50/50 | 1.00 |
| EXPERIENCE | `EXP_EMOTION` | 48 | 0 | 30.1 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_OVERALL` | 50 | 0 | 16.5 | 50/50 | 1.00 |
| FACILITY | `FAC_BUILDING` | 50 | 0 | 31.5 | 50/50 | 1.00 |
| FACILITY | `FAC_CLIMATE` | 47 | 0 | 28.3 | 47/47 | 1.00 |
| FACILITY | `FAC_ENV` | 50 | 0 | 30.3 | 50/50 | 1.00 |
| FACILITY | `FAC_INTERIOR` | 50 | 0 | 24.4 | 50/50 | 1.00 |
| FACILITY | `FAC_SECURITY` | 38 | 0 | 29.2 | 38/38 | 1.00 |
| FACILITY | `FAC_VIEW_LOCATION` | 50 | 0 | 24.2 | 50/50 | 1.00 |
| LOYALTY | `LOY_PREFERENCE` | 42 | 0 | 26.2 | 42/42 | 1.00 |
| LOYALTY | `LOY_RETURN` | 50 | 0 | 16.9 | 50/50 | 1.00 |

## 3. Cross-aspect duplicate first sentences

_Same first sentence reused by ≥2 aspects across entities. With the trained model this is much rarer than the TF-IDF baseline._

Total cross-aspect duplicate sentences: **32**

| # | First sentence (truncated) | Aspects |
| ---: | --- | --- |
| 1 | the staff was very friendly and helpful. | `AM_UTILITY`, `EXP_EMOTION`, `EXP_SAFETY`, `FAC_BATH`, `FAC_CLIMATE`, `SER_ATTITUDE`, `SER_COMM`, `SER_OPERATION`, `SER_SUPPORT` |
| 2 | this hotel is a great place to stay | `AM_WIFI`, `FAC_CLIMATE`, `SER_OPERATION`, `SER_SUPPORT` |
| 3 | the marquesa is one of the best hotels in key west. | `EXP_OVERALL`, `FAC_BUILDING`, `FAC_INTERIOR`, `FAC_SECURITY` |
| 4 | the staff was very friendly and helpful | `FAC_SECURITY`, `SER_ATTITUDE`, `SER_SUPPORT` |
| 5 | the room was very clean, tastefully decorated and had great amenities such as wi-fi, a flat screen tv, dvd player, etc | `AM_ENT`, `AM_ROOM_UTIL` |
| 6 | this is a 4 star hotel on the north side of the river, about 10 minutes walk from vatican city | `AM_ENT`, `EXP_VALUE` |
| 7 | the roman spa offers mud baths and massages and the helpful staff gives tips on restaurants and shops on calistoga's mai… | `AM_FOOD`, `SER_COMM` |
| 8 | the location of the hotel is very good | `AM_TRANSPORT`, `FAC_VIEW_LOCATION` |
| 9 | this hotel is a great value for money | `AM_UTILITY`, `EXP_VALUE` |
| 10 | the hotel is a bit pricey for what you get. | `AM_UTILITY`, `BRA_LUXURY` |
| 11 | the stone castle hotel is a must to see and stay at,from its three castles and two indoor heated pools and two spas it a… | `AM_WELLNESS`, `AM_WIFI` |
| 12 | the hotel is a good value for money. | `BRA_REPUTE`, `EXP_VALUE` |
| 13 | the staff was friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time t… | `EXP_EMOTION`, `FAC_BUILDING` |
| 14 | i would not stay at this motel again for the price. | `EXP_OVERALL`, `LOY_RETURN` |
| 15 | the wedgewood is a great hotel. | `EXP_OVERALL`, `SER_OPERATION` |
| 16 | i would stay here again, but will ask for a room close to the elevator. | `EXP_OVERALL`, `LOY_RETURN` |
| 17 | i would stay here again and would highly recommend it. | `EXP_OVERALL`, `LOY_RETURN` |
| 18 | i would highly recommend this hotel and plan to stay here on our next trip to venice. | `EXP_OVERALL`, `LOY_RETURN` |
| 19 | i would definitely stay here again and recommend to anyone who does not require fancy accommodations. | `EXP_OVERALL`, `LOY_RETURN` |
| 20 | i would definately recommend the murray hill east and wouldn't hesitate to stay there in the future. | `EXP_OVERALL`, `LOY_RETURN` |
| … | _12 more_ | |

## 4. Sample summaries

<details><summary><code>AM_ENT</code> (AMENITY)</summary>

- `dev_100585`: The hotel is a great place to relax and have a good time.
- `dev_100597`: The hotel has a good TV, and a nice bathroom with thick towels and plenty of hot water.
- `dev_1029276`: Not a bad hotel but not a great place for a business trip.

</details>

<details><summary><code>AM_FOOD</code> (AMENITY)</summary>

- `dev_100585`: The Vintage Park is a great hotel with a very creative restaurant in the lobby.
- `dev_100597`: The hotel restaurant was nice enough, but bland; the food was nicely presented but had no taste.
- `dev_1029276`: The Hilton Honors is a great hotel

</details>

<details><summary><code>AM_POOL</code> (AMENITY)</summary>

- `dev_100585`: The hotel has a swimming pool, a fitness center, and a winery.
- `dev_100597`: The hotel has a swimming pool and a hot tub.
- `dev_1029276`: The Hilton Miami Downtown is a great place to stay.

</details>

<details><summary><code>AM_ROOM_UTIL</code> (AMENITY)</summary>

- `dev_100585`: In-room Utilities (AM_ROOM_UTIL).
- `dev_100597`: In-room Utilities (AM_ROOM_UTIL).
- `dev_1029276`: This hotel is a bit noisy and not as good as other Hilton hotels.

</details>

<details><summary><code>AM_TRANSPORT</code> (AMENITY)</summary>

- `dev_100585`: The parking was very tight and the internet speed was slow.
- `dev_100597`: The hotel has a convenient airport shuttle that picks up guests on level 3 of the airport parking garage at stops #1 and 3.
- `dev_1029276`: Parking is a bit high for the price, but it's worth it.

</details>

<details><summary><code>AM_UTILITY</code> (AMENITY)</summary>

- `dev_100597`: The hotel was clean, the beds were fine, and it was close to a Denny's, Jack in the Box, and a steak place.
- `dev_1029276`: This hotel is a great place to stay for a short stay.
- `dev_111509`: The staff was very friendly and helpful.

</details>

<details><summary><code>AM_WELLNESS</code> (AMENITY)</summary>

- `dev_100585`: Gym, Spa & Wellness Facilities (AM_WELLNESS).
- `dev_100597`: The hotel is close to the airport and has a nice lounge with free internet access.
- `dev_1029276`: This hotel was ok for the money and acceptable while on a business trip, but if I stayed in the hotel for holidays, it would be very disappointing.

</details>

<details><summary><code>AM_WIFI</code> (AMENITY)</summary>

- `dev_100585`: Wifi & Connectivity (AM_WIFI) is one of the most important aspects of a hotel.
- `dev_100597`: Internet available in the room for a fee or free wifi in lobby
- `dev_1029276`: Wifi & Connectivity (AM_WIFI) is one of the most important aspects of a hotel.

</details>

<details><summary><code>BRA_LUXURY</code> (BRANDING)</summary>

- `dev_100585`: BRA_LUXURY
- `dev_100597`: BRA_LUXURY is a luxury hotel with a premium perception.
- `dev_1029276`: BRA_LUXURY is one of the best hotels I have ever stayed in.

</details>

<details><summary><code>BRA_REPUTE</code> (BRANDING)</summary>

- `dev_100585`: I highly recommend this hotel and the Kimpton brand if you want to get away from all the generic chain hotels.
- `dev_100597`: It's an average hotel by any other standards but nice compared to what's close by
- `dev_1029276`: The room was very sub-standard for cleanliness...floor looked like it never gets vacuumed

</details>

<details><summary><code>EXP_EMOTION</code> (EXPERIENCE)</summary>

- `dev_100585`: The staff was very friendly and helpful.
- `dev_100597`: The hotel is a bit outdated, but the room we stayed in (in the tower) was very pleasant, relaxing and modern.
- `dev_1029276`: The staff was friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there.

</details>

<details><summary><code>EXP_OVERALL</code> (EXPERIENCE)</summary>

- `dev_100585`: Hotel Vintage Park is a great place to stay in Seattle.
- `dev_100597`: I would highly recommend the Doubletree and plan on staying here again when flying out of Seattle.
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here again.

</details>

<details><summary><code>EXP_SAFETY</code> (EXPERIENCE)</summary>

- `dev_100585`: This is a nice hotel to stay at
- `dev_100597`: This hotel is by a small lake and at dusk there are misquitos and misc.
- `dev_1029276`: There is no fridge or in-room safe in the room.

</details>

<details><summary><code>EXP_VALUE</code> (EXPERIENCE)</summary>

- `dev_100585`: It's not luxurious, but for the price, it's worth it.
- `dev_100597`: It's a good value for money, but it's not the best value for money.
- `dev_1029276`: This is a very nice hotel, especially for the cost, if you have to be downtown, it's an OK choice.

</details>

<details><summary><code>FAC_BATH</code> (FACILITY)</summary>

- `dev_100585`: The room was decent, but the bathroom was quite small (old tiles, old fixtures, just a new piece of granite on the sink)
- `dev_100597`: The hotel has a good TV, and a nice bathroom with thick towels and plenty of hot water
- `dev_1029276`: The room was spacious and bathroom clean with great views over the bay and the breakfast was awesome, the staff throughout the hotel were very polite and helpful

</details>

<details><summary><code>FAC_BUILDING</code> (FACILITY)</summary>

- `dev_100585`: We stayed at the Venetian for a weekend
- `dev_100597`: The hotel is located on a busy street and I didn't see a restaurant within walking distance so I ate in the lobby.
- `dev_1029276`: The staff was friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there.

</details>

<details><summary><code>FAC_CLIMATE</code> (FACILITY)</summary>

- `dev_100597`: The room was fine, but I am not sure that this was worth it
- `dev_1029276`: The air conditioning system was messed up and the pool sucked
- `dev_1064610`: The Sheraton is one of the best hotels I've ever stayed in.

</details>

<details><summary><code>FAC_ENV</code> (FACILITY)</summary>

- `dev_100585`: The hotel is a bit noisy, but the rooms are very comfortable and the location is perfect.
- `dev_100597`: The hotel is close to the airport and has a nice view of the city.
- `dev_1029276`: The room was a bit noisy as it was across the hall from the vending machine, so they moved me, gladly, to the 18th floor

</details>

<details><summary><code>FAC_INTERIOR</code> (FACILITY)</summary>

- `dev_100585`: I love Kimpton Hotels for their cleanliness, service, decor, and small/mid-size hotel feel.
- `dev_100597`: The design of the hotel is a little maze-like but still beautiful and the pool definitely makes you feel like you're somewhere else rather than beside the Sea-Tac Airport.
- `dev_1029276`: The hotel has a tropical theme, colorful decor, and a great view of the Colts.

</details>

<details><summary><code>FAC_ROOM</code> (FACILITY)</summary>

- `dev_100585`: Room, Bed & Sleep Quality (FAC_ROOM): The room was well appointed, with an excellent mini bar, soft robes, slipper socks for chilly feet (purchase), and quality toiletry items
- `dev_100597`: Room, Bed & Sleep Quality (FAC_ROOM) is one of the most important aspects of a hotel.
- `dev_1029276`: Room, Bed & Sleep Quality (FAC_ROOM) is one of the most important aspects of a hotel.

</details>

<details><summary><code>FAC_SECURITY</code> (FACILITY)</summary>

- `dev_100597`: This hotel was not a good choice for a single female traveler.
- `dev_1029276`: The area is not safe to walk around with panhandlers lurking around, if the burger king needs security at night, this is a clear indication that you better stay inside and order in.
- `dev_1064610`: You need your room key to enter different sections of the hotel,including to be able to operate the elevators.

</details>

<details><summary><code>FAC_VIEW_LOCATION</code> (FACILITY)</summary>

- `dev_100585`: Hotel Vintage Park is located in downtown Seattle.
- `dev_100597`: This hotel provides an eagle's eye view of the airport (unless you're on the east side of the tower, in which case you're stuck looking at the snowcapped Cascades and/or Mt Rainier).
- `dev_1029276`: Our room was very comfortable and had a great view of the city and of the bay.

</details>

<details><summary><code>LOY_PREFERENCE</code> (LOYALTY)</summary>

- `dev_100585`: I travel to Seattle for business once or twice a month
- `dev_100597`: The Doubletree is one of my favorite hotels.
- `dev_1029276`: The room was a bit dusty and the hair dryer had a broken attachment

</details>

<details><summary><code>LOY_RECOMMEND</code> (LOYALTY)</summary>

- `dev_100585`: I highly recommend this hotel and the Kimpton brand.
- `dev_100597`: I would not recommend this hotel.
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here anytime.

</details>

<details><summary><code>LOY_RETURN</code> (LOYALTY)</summary>

- `dev_100585`: I will stay here again without hesitation and recommend it to anyone traveling to Seattle.
- `dev_100597`: I stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here again

</details>

<details><summary><code>SER_ATTITUDE</code> (SERVICE)</summary>

- `dev_100585`: The staff at the reception were very friendly and helpful.
- `dev_100597`: The staff was friendly and helpful.
- `dev_1029276`: The staff was very friendly and helpful.

</details>

<details><summary><code>SER_COMM</code> (SERVICE)</summary>

- `dev_100597`: The front desk staff was friendly and helpful
- `dev_1029276`: This is a very large hotel and very busy with many international guest (we overheard many languages spoken) and eventhough the staff was VERY busy they all had a great attitude and friendly demeanor
- `dev_1064610`: The staff was very nice, but the food took almost 3 hours, we waited and waited and eventually got our 'fillet mignon' which was tiny and very over decorated, infact the meal looked a mess (what we en…

</details>

<details><summary><code>SER_OPERATION</code> (SERVICE)</summary>

- `dev_100585`: I arrived early at the hotel and was given the choice to check into a room that was already ready on a lower floor or wait about an hour for a room on a higher floor
- `dev_100597`: The staff was friendly and helpful and we were given a nice corner room with a balcony on the 9th floor (accompanied by the delicious chocolate chip cookies).
- `dev_1029276`: This hotel is one of the nicest hotels that I have stayed in, we flew to Miami from Vegas and arrived at 7.30am we were able to check in which was great staff were very helpfull

</details>

<details><summary><code>SER_SUPPORT</code> (SERVICE)</summary>

- `dev_100585`: The staff was very friendly and helpful.
- `dev_100597`: The only complaint I have is that the parking for overnight was $16, and on top of that, we had $1.40 of tax.
- `dev_1029276`: The one issue I must stress is do not take the overhead train cars to or from the hotel area after 6pm.

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
