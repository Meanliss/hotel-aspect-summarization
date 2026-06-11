# Trained SemAE Aspect Inference Report &mdash; `space_hasos_full_e20`

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
| LOYALTY | `LOY_RECOMMEND` | 50 | 0 | 40.0 | 48/50 | 0.96 |
| EXPERIENCE | `EXP_OVERALL` | 50 | 0 | 40.0 | 49/50 | 0.98 |
| SERVICE | `SER_ATTITUDE` | 50 | 0 | 40.0 | 49/50 | 0.98 |
| AMENITY | `AM_ENT` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| AMENITY | `AM_FOOD` | 50 | 0 | 39.8 | 50/50 | 1.00 |
| AMENITY | `AM_POOL` | 50 | 10 | 40.0 | 40/40 | 1.00 |
| AMENITY | `AM_ROOM_UTIL` | 50 | 0 | 40.4 | 50/50 | 1.00 |
| AMENITY | `AM_TRANSPORT` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| AMENITY | `AM_UTILITY` | 50 | 18 | 40.2 | 32/32 | 1.00 |
| AMENITY | `AM_WELLNESS` | 50 | 12 | 40.3 | 38/38 | 1.00 |
| AMENITY | `AM_WIFI` | 50 | 0 | 40.4 | 50/50 | 1.00 |
| BRANDING | `BRA_LUXURY` | 50 | 4 | 40.0 | 46/46 | 1.00 |
| BRANDING | `BRA_REPUTE` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| EXPERIENCE | `EXP_EMOTION` | 50 | 2 | 39.6 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_SAFETY` | 50 | 2 | 40.1 | 48/48 | 1.00 |
| EXPERIENCE | `EXP_VALUE` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| FACILITY | `FAC_BATH` | 50 | 0 | 40.2 | 50/50 | 1.00 |
| FACILITY | `FAC_BUILDING` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| FACILITY | `FAC_CLIMATE` | 50 | 3 | 40.2 | 47/47 | 1.00 |
| FACILITY | `FAC_ENV` | 50 | 0 | 39.9 | 50/50 | 1.00 |
| FACILITY | `FAC_INTERIOR` | 50 | 0 | 40.3 | 50/50 | 1.00 |
| FACILITY | `FAC_ROOM` | 50 | 0 | 40.1 | 50/50 | 1.00 |
| FACILITY | `FAC_SECURITY` | 50 | 12 | 40.3 | 38/38 | 1.00 |
| FACILITY | `FAC_VIEW_LOCATION` | 50 | 0 | 39.9 | 50/50 | 1.00 |
| LOYALTY | `LOY_PREFERENCE` | 50 | 8 | 40.3 | 42/42 | 1.00 |
| LOYALTY | `LOY_RETURN` | 50 | 0 | 40.0 | 50/50 | 1.00 |
| SERVICE | `SER_COMM` | 50 | 9 | 40.0 | 41/41 | 1.00 |
| SERVICE | `SER_OPERATION` | 50 | 0 | 40.5 | 50/50 | 1.00 |
| SERVICE | `SER_SUPPORT` | 50 | 0 | 40.1 | 50/50 | 1.00 |

## 3. Cross-aspect duplicate first sentences

_Same first sentence reused by ≥2 aspects across entities. With the trained model this is much rarer than the TF-IDF baseline._

Total cross-aspect duplicate sentences: **98**

| # | First sentence (truncated) | Aspects |
| ---: | --- | --- |
| 1 | the tv had a number of english speaking channels and the mini-bar well stocked and not hugely overpriced | `AM_ENT`, `AM_ROOM_UTIL`, `SER_COMM` |
| 2 | the room was spacious with two tvs (one in the bedroom and another in the sitting area), a small couch and chair as well… | `AM_ENT`, `AM_ROOM_UTIL`, `AM_UTILITY` |
| 3 | for instance: the tv is older, smaller screen variant, the security lock on the door was broken, the sink stop was missi… | `AM_ENT`, `FAC_BATH`, `FAC_SECURITY` |
| 4 | the roof top pool and lounge area with bar was our favorite place to meet for a drink and the views of the city at the e… | `AM_FOOD`, `AM_POOL`, `LOY_PREFERENCE` |
| 5 | our room had a brand new air conditioning unit which was quiter than the ones we are used to, a microwave, fridge, freez… | `AM_ROOM_UTIL`, `BRA_REPUTE`, `FAC_CLIMATE` |
| 6 | the location on the borghese gardens means that the setting is peaceful yet still only a short drive from the centre of … | `EXP_EMOTION`, `FAC_ENV`, `FAC_VIEW_LOCATION` |
| 7 | you will not pay a massive sum to stay here but what you get is great value for money | `EXP_OVERALL`, `EXP_VALUE`, `LOY_RETURN` |
| 8 | overall i would highly recommend this hotel and will definately stay here again on my next visit to venice | `EXP_OVERALL`, `LOY_RECOMMEND`, `LOY_RETURN` |
| 9 | we would definately stay here again, and would recommend this hotel to our friends | `EXP_OVERALL`, `LOY_RECOMMEND`, `LOY_RETURN` |
| 10 | our room had two sinks,a flat screen tv, and two comfortable beds | `AM_ENT`, `AM_ROOM_UTIL` |
| 11 | room was great, very large corner room, nice flat panel tv, fridge, the 2 outer walls were basically floor to ceiling wi… | `AM_ENT`, `AM_ROOM_UTIL` |
| 12 | the location is great, right near the entertainment district, within walking distance of the old port and lots of shoppi… | `AM_ENT`, `FAC_VIEW_LOCATION` |
| 13 | the room was spacious enough, with a big glass wall on the outside, a 42" flat screen tv, and terrific lighting arrangem… | `AM_ENT`, `AM_ROOM_UTIL` |
| 14 | we got a ground floor room which had tv with cable(bbc world news was nice for us brits), huge bed, free wi fi, coffee m… | `AM_ENT`, `AM_ROOM_UTIL` |
| 15 | nice flat screen tv also with about 40 channels but only one in english bbc world | `AM_ENT`, `AM_ROOM_UTIL` |
| 16 | the tv in the bedroom was situated high in the corner of the bedroom on the type of platform you would find in a hospita… | `AM_ENT`, `AM_ROOM_UTIL` |
| 17 | good amenities except only 2/3 tv channels n like all hotels we stayed at in italy the tvs were like 8 inches | `AM_ENT`, `AM_ROOM_UTIL` |
| 18 | none of the 'perks' of the hotel were available-pool, steam room, gym, and the tv was old and hardly worked | `AM_ENT`, `AM_WELLNESS` |
| 19 | internet and flatscreen tvs | `AM_ENT`, `AM_ROOM_UTIL` |
| 20 | i had a huge bed, flat screen tv, comfortable chairs, coffee maker (and real cream is provided!) | `AM_ENT`, `AM_ROOM_UTIL` |
| … | _78 more_ | |

## 4. Sample summaries

<details><summary><code>AM_ENT</code> (AMENITY)</summary>

- `dev_100585`: I filled out one of their comment cards with a minor complaint (we found the tv system cumbersome) and I actualy received an e mail from the hotel management
- `dev_100597`: Our room had two sinks,a flat screen TV, and two comfortable beds
- `dev_1029276`: Room was great, very large corner room, nice flat panel TV, fridge, the 2 outer walls were basically floor to ceiling windows with amazing views

</details>

<details><summary><code>AM_FOOD</code> (AMENITY)</summary>

- `dev_100585`: The wine and cheese evening in the small but attractive lobby as well as a lovely dinner in Tulio (their Italian restaurant on-site) made for an enjoyable and relaxing weekend
- `dev_100597`: The service in both the casual restaurant at breakfast and the dining room in the evening was efficient
- `dev_1029276`: The pool area was actually a nice surprise as it was spacious and well furnished with a very friendly full service pool bar and grill

</details>

<details><summary><code>AM_POOL</code> (AMENITY)</summary>

- `dev_100585`: You can purchase a pass for a great fitness center with steam room and lap pool
- `dev_100597`: The swimming pool looked nice but we didn't get the chance to swim
- `dev_1029276`: The pool area was actually a nice surprise as it was spacious and well furnished with a very friendly full service pool bar and grill

</details>

<details><summary><code>AM_ROOM_UTIL</code> (AMENITY)</summary>

- `dev_100585`: Only complaint: there are no vending machines so we were forced to buy overpriced minibar drinks the first night (not much of a complaint, huh?)
- `dev_100597`: Our room had two sinks,a flat screen TV, and two comfortable beds
- `dev_1029276`: Room was great, very large corner room, nice flat panel TV, fridge, the 2 outer walls were basically floor to ceiling windows with amazing views

</details>

<details><summary><code>AM_TRANSPORT</code> (AMENITY)</summary>

- `dev_100585`: The staff are friendly, whether you are talking to the parking valet or the Front desk personnel
- `dev_100597`: The shuttle will drop you at the airport train station if you are going to Seattle, but on your way back you must walk from the station to the shuttle bus pick up area at the airport(about a 5 to
- `dev_1029276`: $28/day for valet and $25/day self parking...very expensive Blood stains on pillow cases...contacted front desk about the issue and they reacted like it was a normal thing and would mention it to hous…

</details>

<details><summary><code>AM_UTILITY</code> (AMENITY)</summary>

- `dev_100585`: The room was equipped with a coffee maker, iron & ironing board and blow dryer and there were 2 sinks - one inside the bathroom and one outside
- `dev_100597`: As I was only there for one night I didn't utility all the services the hotel had to offer
- `dev_1029276`: Empty water bottles were never thrown out and no one put the iron and ironing board away

</details>

<details><summary><code>AM_WELLNESS</code> (AMENITY)</summary>

- `dev_100585`: But there was in-room yoga on the TV and they gave me a yoga mat
- `dev_100597`: We enjoyed the spa/pool area and they have a nice lounge with free internet access terminals
- `dev_1029276`: However, I still think this hotel is very nice with the size of the room, lobby area, swimming pool and the gym

</details>

<details><summary><code>AM_WIFI</code> (AMENITY)</summary>

- `dev_100585`: Free internet for InTouch members, and you also get $10 minibar credit per stay
- `dev_100597`: Internet available in the room for a fee or free wifi in lobby
- `dev_1029276`: The executive lounge had new Dell computers, wifi internet access was available throughout the hotel (including the pool area) -- free for gold and diamond members

</details>

<details><summary><code>BRA_LUXURY</code> (BRANDING)</summary>

- `dev_100585`: I probably wouldn't have stayed at someplace so upscale had I not lucked out with a great rate on Expedia ($112), but I was thrilled that I was able to because the location can't be beat
- `dev_100597`: I asked for a room in their Tower, but was told that Priceline customers had to pay a ten dollar a night premium, which I GLADLY paid
- `dev_1029276`: So, if you are staying one night and then heading to the cruise terminal it is fine...anything more than that look at a more upscale place in a better area of town

</details>

<details><summary><code>BRA_REPUTE</code> (BRANDING)</summary>

- `dev_100585`: This hotel largely meets the standards of other Kimpton Hotels, as indicated in many of the other reviews
- `dev_100597`: It's an ok hotel by any other standards but nice compared to what's close by
- `dev_1029276`: We felt that an establishment of this nature in this location and carrying the Hilton brand should have made staff available to man the lounge at the times stated

</details>

<details><summary><code>EXP_EMOTION</code> (EXPERIENCE)</summary>

- `dev_100585`: The wine and cheese evening in the small but attractive lobby as well as a lovely dinner in Tulio (their Italian restaurant on-site) made for an enjoyable and relaxing weekend
- `dev_100597`: The hotel interior itself is a bit outdated, but the room we stayed in (in the tower) was very pleasant, relaxing and modern
- `dev_1029276`: The staff were friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there

</details>

<details><summary><code>EXP_OVERALL</code> (EXPERIENCE)</summary>

- `dev_100585`: We had a wonderful stay, and look forward to staying again the next time we're in Seattle
- `dev_100597`: I stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I accepted since the only reason I stayed at this Hilton was for a good nights sleep while starting a family vacation

</details>

<details><summary><code>EXP_SAFETY</code> (EXPERIENCE)</summary>

- `dev_100585`: - In room safe - Efforts in being eco-friendly Things I didn't like: - Mattress was awfully uncomfortable
- `dev_100597`: We felt safe and secure with our small children
- `dev_1029276`: No room safe or fridge - ice down the hall

</details>

<details><summary><code>EXP_VALUE</code> (EXPERIENCE)</summary>

- `dev_100585`: A high quality for the value
- `dev_100597`: :) The only complaint is - parking for overnight cost $16, and on top of that, we had $1.40 of tax
- `dev_1029276`: The cost of this hotel should be 1/2 of existing for the value

</details>

<details><summary><code>FAC_BATH</code> (FACILITY)</summary>

- `dev_100585`: The bathroom was a little small but clean, and well-appointed with Aveda bath products
- `dev_100597`: There was a good TV, and nice bathroom with thick towels and plenty of hot water
- `dev_1029276`: Beds were very comfortble with excellent creature comforts, towels pillows and bedding and I felt that it was a very comfortable sleeping room

</details>

<details><summary><code>FAC_BUILDING</code> (FACILITY)</summary>

- `dev_100585`: While they don't have coffee service in the room, it's available down in the lobby and they even provided us with "to go" cups for our morning walk
- `dev_100597`: The tower elevators, which run on the exterior of the building, were quite cold
- `dev_1029276`: A huge group of teenagers were staying in the hotel, and kids were all over the place until late at night...flooding the lobby, elevators, and public space

</details>

<details><summary><code>FAC_CLIMATE</code> (FACILITY)</summary>

- `dev_100585`: The bed was comfortable but between the noise of the traffic ( not the jets) and the room fan, we did not sleep well
- `dev_100597`: The ventilation system (not the ac) was loud and could not be turned off...I am not the type that needs total silence to sleep but I was counting the days to get the he'll outta here
- `dev_1029276`: Usually stay at Sheraton

</details>

<details><summary><code>FAC_ENV</code> (FACILITY)</summary>

- `dev_100585`: We were given lower floor and had to deal with construction noise across the street and the on-ramp to the freeway
- `dev_100597`: The room was very clean, cozy and very quiet even though it was so close to the airport
- `dev_1029276`: I was 5 rooms away from the elevator and the noise in the hall of just the elevator was annoying

</details>

<details><summary><code>FAC_INTERIOR</code> (FACILITY)</summary>

- `dev_100585`: My bargain-priced room was small but perfectly clean and gleaming, modern design, well lighted, elegant furnishings with professional decor and a super-comfortable bed
- `dev_100597`: The hotel has seen its newer days: the carpets are threadbare in some spots and the cement balconies are noticeably empty of furniture
- `dev_1029276`: Tropical theme, colorful decor

</details>

<details><summary><code>FAC_ROOM</code> (FACILITY)</summary>

- `dev_100585`: The minus was for the slightly worn out chair in the room, a bath towel that had seen better days, and the rather high parking rate - $30/night
- `dev_100597`: We had a balcony, the beds were super comfortable, room was a good size and clean
- `dev_1029276`: They provided me a room with a great view, and my room was very clean and equipped appropriately

</details>

<details><summary><code>FAC_SECURITY</code> (FACILITY)</summary>

- `dev_100585`: Due to security, I did not think this was a good idea for a single female traveler, however, I did not want to 'make waves' and request another room as I was only there for one night
- `dev_100597`: During our stay we were sexually harassed by the security guard on duty by the pool
- `dev_1029276`: Security was very good as you need to use your room card key to use the elevators and to enter the parking lot

</details>

<details><summary><code>FAC_VIEW_LOCATION</code> (FACILITY)</summary>

- `dev_100585`: Hotel Vintage Park is centrally located in downtown Seattle, it was very easy to get to the shopping areas/Pike Place Market and a few good restaurants
- `dev_100597`: It had easy access to the light rail located at the airport (we took the DT shuttle over), and was next to an Enterprise car rental
- `dev_1029276`: The Downtown Miami Hilton is on the fringe of the downtown area but very close to the light rail and a short drive to S

</details>

<details><summary><code>LOY_PREFERENCE</code> (LOYALTY)</summary>

- `dev_100585`: I travel to Seattle for business once or twice a month.I have been trying different hotels each time visit, and this is my favorite thus far
- `dev_100597`: Pros: - cookies upon check-in - one of my favorite things about the Doubletrees - complimentary airport shuttle every 15 mins - nice pools, rooms are very comfortable with excellent beds and Neutrogen…
- `dev_1029276`: The room was immaculate (although the hair dryer had a broken attachment)

</details>

<details><summary><code>LOY_RECOMMEND</code> (LOYALTY)</summary>

- `dev_100585`: I highly recommend this as your destination hotel any time you travel to Seattle
- `dev_100597`: I would not recommend this hotel
- `dev_1029276`: I would recommend this hotel and I would stay myself there anytime

</details>

<details><summary><code>LOY_RETURN</code> (LOYALTY)</summary>

- `dev_100585`: I will stay here again without hesitation and recommend it to anyone traveling to Seattle
- `dev_100597`: I stayed here for a week in June, and thoroughly enjoyed my stay
- `dev_1029276`: I would recommend this hotel to anyone looking for a hotel close to the Miami port and I would definitely stay here again

</details>

<details><summary><code>SER_ATTITUDE</code> (SERVICE)</summary>

- `dev_100585`: The staff is incredibly friendly, takes initiative, and helpful
- `dev_100597`: The breakfast in the restaurant was amazing, and the staff was very attentive and friendly
- `dev_1029276`: The staff were friendly and helpful, and the lobby and bars downstairs were very relaxing and we did spend alot of time there

</details>

<details><summary><code>SER_COMM</code> (SERVICE)</summary>

- `dev_100585`: Unfortunately I hope front desk could have better communication
- `dev_100597`: This is a very large hotel and very busy with many international guest (we overheard many languages spoken) and eventhough the staff was VERY busy they all had a great attitude and friendly demeanor
- `dev_1029276`: There were plenty of waiting staff and they were all very nice, but hadnt a clue what to do, they all stood around, fell over each other and the whole meal took almost 3 hours, we waited and waited an…

</details>

<details><summary><code>SER_OPERATION</code> (SERVICE)</summary>

- `dev_100585`: We arrived an hour before check-in and after waiting 15 minutes in the lobby to check in, were given the key to our room
- `dev_100597`: The staff was friendly and helpful and we enjoyed the warm, chocolate chip cookie we were given at check-in
- `dev_1029276`: Absolutely nothing to complain about rooms (modern and with huge widescreen tv) and service: the express check out on the phone, in a busy day of meeting and conferences, saved me a lot of time

</details>

<details><summary><code>SER_SUPPORT</code> (SERVICE)</summary>

- `dev_100585`: I begged and pleaded with the staff to help me correct the issues and all I got was "Oh, we're so sorry, we don't know what's wrong"
- `dev_100597`: I was put into this property by an airline due to a maintenance issue, although I checked in late at night and had to leave early the next morning, I saw enough that I want to return
- `dev_1029276`: One issue with the executive lounge access - hotel literature stated the lounge should have been open from 6.00 am - this was not the case and when we arrived on one occasion to grab breakfast before …

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
