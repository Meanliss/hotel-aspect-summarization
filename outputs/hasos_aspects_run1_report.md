# Trained SemAE Aspect Inference Report &mdash; `hasos_aspects_run1`

Auto-generated summary of the **trained SemAE model** outputs (VQ-VAE + KL-divergence ranking).
These replace the earlier TF-IDF baseline scoring run in `outputs/hasos_english_only/`.

## 1. Run overview

- Source data: `data/hasos/hasos_summ.json` (50 entities)
- Model: trained over 10 epochs on GPU (see `logs/train_hasos_run1.log`)
- Aspects: 29
- Total summary files: 1450
- Empty outputs: 78
- Generation wall time: ~22 min on 4 parallel shards (RTX 3500 Ada 12GB)

## 2. Per-aspect statistics

Aspects ordered by `first_sent_uniqueness` (lower = more boilerplate, higher = more diverse).

| Group | Aspect | Entities | Empty | Avg words | Unique first sentences | Uniqueness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LOYALTY | `LOY_RETURN` | 50 | 0 | 40.1 | 49/50 | 0.98 |
| SERVICE | `SER_ATTITUDE` | 50 | 0 | 40.0 | 49/50 | 0.98 |
| AMENITY | `AM_ENT` | 50 | 1 | 40.2 | 49/49 | 1.00 |
| AMENITY | `AM_FOOD` | 50 | 0 | 40.1 | 50/50 | 1.00 |
| AMENITY | `AM_POOL` | 50 | 10 | 40.1 | 40/40 | 1.00 |
| AMENITY | `AM_ROOM_UTIL` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| AMENITY | `AM_TRANSPORT` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| AMENITY | `AM_UTILITY` | 50 | 18 | 40.2 | 32/32 | 1.00 |
| AMENITY | `AM_WELLNESS` | 50 | 12 | 40.2 | 38/38 | 1.00 |
| AMENITY | `AM_WIFI` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| BRANDING | `BRA_LUXURY` | 50 | 1 | 40.2 | 49/49 | 1.00 |
| BRANDING | `BRA_REPUTE` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| EXPERIENCE | `EXP_EMOTION` | 50 | 2 | 39.8 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_OVERALL` | 50 | 0 | 40.1 | 50/50 | 1.00 |
| EXPERIENCE | `EXP_SAFETY` | 50 | 2 | 40.1 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_VALUE` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| FACILITY | `FAC_BATH` | 50 | 0 | 40.0 | 50/50 | 1.00 |
| FACILITY | `FAC_BUILDING` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| FACILITY | `FAC_CLIMATE` | 50 | 3 | 40.1 | 47/47 | 1.00 |
| FACILITY | `FAC_ENV` | 50 | 0 | 39.9 | 50/50 | 1.00 |
| FACILITY | `FAC_INTERIOR` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| FACILITY | `FAC_ROOM` | 50 | 0 | 40.0 | 50/50 | 1.00 |
| FACILITY | `FAC_SECURITY` | 50 | 12 | 40.3 | 38/38 | 1.00 |
| FACILITY | `FAC_VIEW_LOCATION` | 50 | 0 | 39.6 | 50/50 | 1.00 |
| LOYALTY | `LOY_PREFERENCE` | 50 | 8 | 40.4 | 42/42 | 1.00 |
| LOYALTY | `LOY_RECOMMEND` | 50 | 0 | 40.1 | 50/50 | 1.00 |
| SERVICE | `SER_COMM` | 50 | 9 | 40.0 | 41/41 | 1.00 |
| SERVICE | `SER_OPERATION` | 50 | 0 | 40.4 | 50/50 | 1.00 |
| SERVICE | `SER_SUPPORT` | 50 | 0 | 40.2 | 50/50 | 1.00 |

## 3. Cross-aspect duplicate first sentences

_Same first sentence reused by ≥2 aspects across entities. With the trained model this is much rarer than the TF-IDF baseline._

Total cross-aspect duplicate sentences: **83**

| # | First sentence (truncated) | Aspects |
| ---: | --- | --- |
| 1 | first, the tv's 500 channels could afford to have one or two more english channels than cnn and bbc | `AM_ENT`, `AM_ROOM_UTIL`, `SER_COMM` |
| 2 | the bathroom was big, with plenty of fluffy towels and the bedroom had a large tv with a good selection of channels | `AM_ENT`, `AM_ROOM_UTIL`, `FAC_BATH` |
| 3 | the first impression when i arrived it was a cosy entrance and friendly staff (well done manuel) the room was ok for sin… | `AM_ENT`, `AM_ROOM_UTIL`, `FAC_BATH` |
| 4 | modern items like tv, mini bar and safe are built into antique furnishings; towelling robes and slippers are free; and t… | `AM_ENT`, `AM_ROOM_UTIL`, `EXP_SAFETY` |
| 5 | as a minus point i would say the breakfast could be better and maybe a spa/pool to better justify the price (the hotel d… | `AM_POOL`, `AM_WELLNESS`, `BRA_LUXURY` |
| 6 | there's no pool or spa -- i sort of expect a pool for a four star hotel | `AM_POOL`, `AM_WELLNESS`, `BRA_LUXURY` |
| 7 | the room itself had a retro 70's feel with a comfortable living room and kitchen area, a separate bedroom with a nice ki… | `AM_ROOM_UTIL`, `FAC_BATH`, `FAC_ROOM` |
| 8 | the little work out room is kind of a joke, but it is a clean and well appointed joke...the tv (w/headphones) on the exe… | `AM_ENT`, `FAC_BATH` |
| 9 | room with 2 x double beds, shower over (tiny) bath, toilet, sink in room, hair dryer, tv, iron and board, sitting table … | `AM_ENT`, `AM_UTILITY` |
| 10 | we got a ground floor room which had tv with cable(bbc world news was nice for us brits), huge bed, free wi fi, coffee m… | `AM_ENT`, `AM_ROOM_UTIL` |
| 11 | the tv in the bedroom was situated high in the corner of the bedroom on the type of platform you would find in a hospita… | `AM_ENT`, `AM_ROOM_UTIL` |
| 12 | there was a safe and tv | `AM_ENT`, `AM_ROOM_UTIL` |
| 13 | i had a huge bed, flat screen tv, comfortable chairs, coffee maker (and real cream is provided!) | `AM_ENT`, `AM_ROOM_UTIL` |
| 14 | * small rooms * we had a small balcony but had plastic furniture and latice on the floor (not nice) * the dresser, desk … | `AM_ENT`, `AM_ROOM_UTIL` |
| 15 | coffee shop open 24hrs where you can get just about anything to eat such as bagels, donuts, pizza, ice cream, sandwiches… | `AM_ENT`, `AM_TRANSPORT` |
| 16 | only thing wrong was no tea or coffee making facilities, but they had satellite tv | `AM_ENT`, `AM_ROOM_UTIL` |
| 17 | the next room next door was listening to the tv very loud | `AM_ENT`, `AM_ROOM_UTIL` |
| 18 | another thing is that if you want to watch tv and don't speak italian you'll have to pay extra to get additional channel… | `AM_ENT`, `AM_ROOM_UTIL` |
| 19 | the bathroom was very clean and modern (i can't stand fusty old bathrooms), the bed was comfy, you get a desk with wirel… | `AM_ENT`, `AM_ROOM_UTIL` |
| 20 | small kitchenette with stove and fridge (no microwave), large dinning area, living area (comfortable couch and chair - a… | `AM_ENT`, `AM_FOOD` |
| … | _63 more_ | |

## 4. Sample summaries

<details><summary><code>AM_ENT</code> (AMENITY)</summary>

- `dev_100585`: The little work out room is kind of a joke, but it is a clean and well appointed joke...the TV (w/headphones) on the exercise machines is nice
- `dev_100597`: Decent size with flat panel TV, and normal size bathroom
- `dev_1029276`: Room was great, very large corner room, nice flat panel TV, fridge, the 2 outer walls were basically floor to ceiling windows with amazing views

</details>

<details><summary><code>AM_FOOD</code> (AMENITY)</summary>

- `dev_100585`: Another wonderful feature is the free wine service in the lobby in the evenings
- `dev_100597`: We had dinner in the attached restaurant, and the food was absolutely delicious
- `dev_1029276`: To further my aggravation, the hotel information stated that a continental breakfast buffet was included, however when we went down the dining area and got the kids some food and sat down to eat is wh…

</details>

<details><summary><code>AM_POOL</code> (AMENITY)</summary>

- `dev_100585`: You can purchase a pass for a great fitness center with steam room and lap pool
- `dev_100597`: Hotel pool is outside, heated to a nice temp and there is a hot tub nearby with a nice seating area all around
- `dev_1029276`: The pool area was actually a nice surprise as it was spacious and well furnished with a very friendly full service pool bar and grill

</details>

<details><summary><code>AM_ROOM_UTIL</code> (AMENITY)</summary>

- `dev_100585`: We were then given a corner room - the only demonstrated make up - that was bright, fairly large and airy (at $260 US per night, it shoudl have been)
- `dev_100597`: I had a spacious and clean suite with fridge, microwave, nice sized table and a comfortable bed
- `dev_1029276`: That having been said that have updated the rooms and the corner Jr

</details>

<details><summary><code>AM_TRANSPORT</code> (AMENITY)</summary>

- `dev_100585`: The mandatory car parking service was good, but expensive ($28/day)
- `dev_100597`: Hotel shuttle picked us up at the airport and took us to the light rail station to go into Seattle the next day
- `dev_1029276`: When I booked the hotel, the hotel info said that parking was available, it was not stated that I would have to pay $28 to park, so when we went out for the day and came back $28, and when

</details>

<details><summary><code>AM_UTILITY</code> (AMENITY)</summary>

- `dev_100585`: The room was equipped with a coffee maker, iron & ironing board and blow dryer and there were 2 sinks - one inside the bathroom and one outside
- `dev_100597`: As I was only there for one night I didn't utility all the services the hotel had to offer
- `dev_1029276`: Empty water bottles were never thrown out and no one put the iron and ironing board away

</details>

<details><summary><code>AM_WELLNESS</code> (AMENITY)</summary>

- `dev_100585`: You can purchase a pass for a great fitness center with steam room and lap pool
- `dev_100597`: We enjoyed the spa/pool area and they have a nice lounge with free internet access terminals
- `dev_1029276`: However, I still think this hotel is very nice with the size of the room, lobby area, swimming pool and the gym

</details>

<details><summary><code>AM_WIFI</code> (AMENITY)</summary>

- `dev_100585`: Joining is free and you get perks like said free wifi, $10 towards mini bar snack, your choice of paper in the morning, and choice of pillows
- `dev_100597`: Internet available in the room for a fee or free wifi in lobby
- `dev_1029276`: Free internet is available only in lobby downstairs and if you want internet in your room, you have to pay for it

</details>

<details><summary><code>BRA_LUXURY</code> (BRANDING)</summary>

- `dev_100585`: This was billed as a 4 star hotel and it is not even a 3
- `dev_100597`: This Doubletree is very tired, and more like a 2 star property than a 3 star
- `dev_1029276`: So, if you are staying one night and then heading to the cruise terminal it is fine...anything more than that look at a more upscale place in a better area of town

</details>

<details><summary><code>BRA_REPUTE</code> (BRANDING)</summary>

- `dev_100585`: I highly recommend this hotel and the Kimpton brand
- `dev_100597`: The fitness room is small but the bedrooms are standard
- `dev_1029276`: We felt that an establishment of this nature in this location and carrying the Hilton brand should have made staff available to man the lounge at the times stated

</details>

<details><summary><code>EXP_EMOTION</code> (EXPERIENCE)</summary>

- `dev_100585`: Trulio, the attached restaurant is outstanding and has a comfortable clubby atmosphere
- `dev_100597`: The hotel interior itself is a bit outdated, but the room we stayed in (in the tower) was very pleasant, relaxing and modern
- `dev_1029276`: The staff were friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there

</details>

<details><summary><code>EXP_OVERALL</code> (EXPERIENCE)</summary>

- `dev_100585`: I will stay here again without hesitation and recommend it to anyone traveling to Seattle
- `dev_100597`: I would highly recommend the Doubletree and plan on staying here again on our next trip
- `dev_1029276`: I was not looking forward to staying at this hotel after reading the bad reviews ,but i have to say that we really enjoyed our two night stay

</details>

<details><summary><code>EXP_SAFETY</code> (EXPERIENCE)</summary>

- `dev_100585`: The in-room safe was a nice feature
- `dev_100597`: We felt safe and secure with our small children
- `dev_1029276`: No safe or frig in the rooms

</details>

<details><summary><code>EXP_VALUE</code> (EXPERIENCE)</summary>

- `dev_100585`: Much better and/or less expensive than many hotels in Downtown Seattle
- `dev_100597`: Unfortunately there was no wifi and the parking was $16 per night - expensive for the area
- `dev_1029276`: The breakfast buffet is expensive but worth it

</details>

<details><summary><code>FAC_BATH</code> (FACILITY)</summary>

- `dev_100585`: The little work out room is kind of a joke, but it is a clean and well appointed joke...the TV (w/headphones) on the exercise machines is nice
- `dev_100597`: The beds were very comfortable, room large enough for wheelchair, and bathroom clean
- `dev_1029276`: First, the toilet paper holder is 2.5 ft

</details>

<details><summary><code>FAC_BUILDING</code> (FACILITY)</summary>

- `dev_100585`: Downside is the small lobby, which can get quite crowded during the wine social and the lobby's close proximity to the outside garage (and smoking area), where cigar smoke drifted in each time the doo…
- `dev_100597`: --The doors to the rooms have a huge gap at the bottom - allows a lot of light and sound in from the hallway
- `dev_1029276`: Ä±t has an interesting entrance you take a lift to 3 floor above to enter the lobby from the valet service but still not tiring

</details>

<details><summary><code>FAC_CLIMATE</code> (FACILITY)</summary>

- `dev_100585`: The bed was comfortable but between the noise of the traffic ( not the jets) and the room fan, we did not sleep well
- `dev_100597`: The ventilation system (not the ac) was loud and could not be turned off...I am not the type that needs total silence to sleep but I was counting the days to get the he'll outta here
- `dev_1029276`: The pool was also very clean, not too cold, and the hot tub was the perfect temperature

</details>

<details><summary><code>FAC_ENV</code> (FACILITY)</summary>

- `dev_100585`: We arrived after midnight and were assigned a street-level room the first night, which I would avoid in the future due to street noise
- `dev_100597`: Even though the hotel was close to the airport, you couldn't haer the planes taking off, it was quiet and very relaxing
- `dev_1029276`: I was 5 rooms away from the elevator and the noise in the hall of just the elevator was annoying

</details>

<details><summary><code>FAC_INTERIOR</code> (FACILITY)</summary>

- `dev_100585`: My bargain-priced room was small but perfectly clean and gleaming, modern design, well lighted, elegant furnishings with professional decor and a super-comfortable bed
- `dev_100597`: The design of the hotel is a little maze-like but still beautiful and the pool definitely makes you feel like you're somewhere else rather than beside the Sea-Tac Airport
- `dev_1029276`: Some fun decor and some oldish

</details>

<details><summary><code>FAC_ROOM</code> (FACILITY)</summary>

- `dev_100585`: The minus was for the slightly worn out chair in the room, a bath towel that had seen better days, and the rather high parking rate - $30/night
- `dev_100597`: Bed's in the room were the heavenly beds and I slept very well all 3 nights
- `dev_1029276`: The room is a big corner room with a balcony as big as the room itself :P floor to ceiling window provide great view and bright natrual light

</details>

<details><summary><code>FAC_SECURITY</code> (FACILITY)</summary>

- `dev_100585`: Due to security, I did not think this was a good idea for a single female traveler, however, I did not want to 'make waves' and request another room as I was only there for one night
- `dev_100597`: Not only did the three of us feel unsafe but we were scared to stay in the room for fear the guard might have access to our room
- `dev_1029276`: Security was very good as you need to use your room card key to use the elevators and to enter the parking lot

</details>

<details><summary><code>FAC_VIEW_LOCATION</code> (FACILITY)</summary>

- `dev_100585`: small)--the unique feel, attentive staff, wonderful food and great downtown location make it a great place to stay in Seattle
- `dev_100597`: If you don't want to wait, a good alternative is the nearby taxi stand; fare to the Doubletree should be less than $10 because of the hotel's proximity to the airport
- `dev_1029276`: I stayed two nights in a room with two beds and floor to ceiling windows with a beautiful view of the city

</details>

<details><summary><code>LOY_PREFERENCE</code> (LOYALTY)</summary>

- `dev_100585`: I travel to Seattle for business once or twice a month.I have been trying different hotels each time visit, and this is my favorite thus far
- `dev_100597`: Pros: - cookies upon check-in - one of my favorite things about the Doubletrees - complimentary airport shuttle every 15 mins - nice pools, rooms are very comfortable with excellent beds and Neutrogen…
- `dev_1029276`: The room was immaculate (although the hair dryer had a broken attachment)

</details>

<details><summary><code>LOY_RECOMMEND</code> (LOYALTY)</summary>

- `dev_100585`: I highly recommend this as your destination hotel any time you travel to Seattle
- `dev_100597`: I would highly recommend the Doubletree and plan on staying here again on our next trip
- `dev_1029276`: I only could recommend a stay in this wonderful hotel to everybody

</details>

<details><summary><code>LOY_RETURN</code> (LOYALTY)</summary>

- `dev_100585`: We do not normally stay in such a nice place but now I am forever spoiled and will not want to go anywhere else should I get to visit Seattle again
- `dev_100597`: I stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I normally stay in South Beach when I go to Miami, but this time I stayed at the Hilton in Downtown Miami

</details>

<details><summary><code>SER_ATTITUDE</code> (SERVICE)</summary>

- `dev_100585`: Exceptionally friendly and knowledgeable staff who went out of their way to assist with any need
- `dev_100597`: The hotel staff was courteous and very helpful with our questions
- `dev_1029276`: The staff were friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there

</details>

<details><summary><code>SER_COMM</code> (SERVICE)</summary>

- `dev_100585`: Unfortunately I hope front desk could have better communication
- `dev_100597`: Fridge didnt work Staff was rude and all spanish speaking
- `dev_1029276`: There were plenty of waiting staff and they were all very nice, but hadnt a clue what to do, they all stood around, fell over each other and the whole meal took almost 3 hours, we waited and waited an…

</details>

<details><summary><code>SER_OPERATION</code> (SERVICE)</summary>

- `dev_100585`: We arrived an hour before check-in and after waiting 15 minutes in the lobby to check in, were given the key to our room
- `dev_100597`: The staff was friendly and helpful and we enjoyed the warm, chocolate chip cookie we were given at check-in
- `dev_1029276`: Absolutely nothing to complain about rooms (modern and with huge widescreen tv) and service: the express check out on the phone, in a busy day of meeting and conferences, saved me a lot of time

</details>

<details><summary><code>SER_SUPPORT</code> (SERVICE)</summary>

- `dev_100585`: The staff wascourteous and helpful and rooms and beds were well appointed and for the most part comfortable ( I personally had a bed that was uneven/slanted- but of the 4 couples there- I was the exce…
- `dev_100597`: They not only are wheelchair accessible, but have an ADA van and very courteous staff that doesn't act put-out at having to help disabled guests
- `dev_1029276`: When we arrived for a one night stay, (we were catching a cruise ship the next morning) the one bell hop at the entrance did not greet us, or offer to help, or open a door

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
