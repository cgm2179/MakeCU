# Step 8 report: live session (personal model vs popularity vs random), adi_v2

- Generated: 2026-11-08 03:16:24  
- User: adi_v2

## 1. Unit tests

- Ran 218: 218 passed, 0 failed, 0 errors, 0 skipped.

## 2. Result

```
random      liked   1 of  40: 2% (95% interval 0% to 13%)
popularity  liked  25 of  40: 62% (95% interval 47% to 76%)
personal    liked  21 of  40: 52% (95% interval 37% to 67%)
personal minus popularity: -10 points (95% range -32 to +12; gate needs +15)
personal minus random: +50 points (95% range +34 to +66; gate needs +25)
Overall: FAIL  (needs 40+ songs per condition; at that size a share is only known to about +/-15 points)
```

## 3. Method

- Three conditions mixed in random blocks of three; the person was not told which condition a song came from.
- random: any song in the pool. popularity: one of the 10 most popular left. personal: one of the 10 songs that the Step 7 model: the frozen Last.fm base (base-v2-lastfm1k) plus the person's personal layer scores highest (frozen; it does not learn during the test).
- Pool: songs from the gym/party/chill/happy/sad/study moods, minus every song already rated, thumbed in the camera sessions, or thumbed in an earlier live session (the personal layer was fitted on those thumbs), one song per artist.
- Label: the explicit thumb after each song. The song stops as soon as the thumb is given.
- Share liked = thumbs up / songs. 95% interval: Wilson. Difference range: normal approximation.
- Gate fixed beforehand: personal >= popularity + 15 points and >= random + 25 points, 40+ songs per condition.

## 4. Limits

- One person, 40 songs per condition: a share is known only to about +/-15 points, so a 15-point gap is suggestive.
- The camera is not used here. The thumb is the label.
- The personal layer was fitted on every earlier thumb, so this session is its first test on songs it never saw.


## 5. Analysis done after seeing the result (not part of the pre-registered gate)
**Verdict: FAIL on the pre-registered gate.** The fine-tuned model (C) beat random by +50 points (needs +25) but did not beat popularity: -10 points (95% range -32 to +12; needs +15).
The gate is missed, and the data cannot say that popularity is actually better either: the range includes zero. As written in advance, popularity was expected to win.

- **Big improvement over the first live session.** The old ratings-trained profile got 30% liked (12 of 40). The Step 7 model (frozen base + personal layer) got **52% (21 of 40)**. Popularity was 72% then and 62% now.
- **It found a real niche.** Of the 40 songs picked for C, 19 were genre "dance" and **16 of those 19 (84%) were liked**; rock was 3 of 8 liked and punk-rock 1 of 5. The popularity condition liked every pop (6 of 6) and hip-hop (5 of 5) song.
  C's picks were less mainstream than B's: mean popularity 78.8 vs 83.6, and 18 of 40 C songs had popularity 80 or more (all 40 B songs did).
- **No drift:** first 60 songs random 1/20, popularity 13/20, personal 11/20; last 60: 0/20, 12/20, 10/20.
- **Offline ranking over all 120 songs:** popularity AUC 0.851, fine-tuned 0.755, base-only 0.702. Popularity ranks Adi's thumbs better, consistent with Step 7.

### The random condition looks unusually low
Random songs were liked **1 of 40 (2%)**, against **11 of 40 (28%)** in the first live session, although their mean popularity was the same (59 vs 61). The "+50 over random" is inflated by this, and I cannot fully explain it.
Likely contributors: the pool now excludes every song Adi has rated or thumbed (453 songs, which include most of the songs Adi liked in earlier sessions), so fewer liked songs are left to be drawn at random; and Adi may have been stricter this time.
A drop from 28% to 2% by chance alone is very unlikely, so something differed between the sessions. Against the first session's random level (28%), C's margin would be about +24 points, just under the +25 mark.
Treat the random comparison as unreliable; the comparison with popularity is the informative one.

### Limits
One person, 40 songs per condition (a share is known to about +/-15 points), the pool differs between the two sessions, and the thumb is a quick judgement after hearing only part of a song.
