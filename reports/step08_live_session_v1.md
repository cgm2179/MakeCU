# Step 8 report (first live session, v1): old ratings-trained profile vs popularity vs random

- Generated: 2026-11-07 23:40:47  
- User: adi

## 1. Unit tests

- Ran 177: 177 passed, 0 failed, 0 errors, 0 skipped.

## 2. Result

```
random      liked  11 of  40: 28% (95% interval 16% to 43%)
popularity  liked  29 of  40: 72% (95% interval 57% to 84%)
personal    liked  12 of  40: 30% (95% interval 18% to 45%)
personal minus popularity: -42 points (95% range -62 to -23; gate needs +15)
personal minus random: +2 points (95% range -17 to +22; gate needs +25)
Overall: FAIL  (needs 40+ songs per condition; at that size a share is only known to about +/-15 points)
```

## 3. Method

- Three conditions mixed in random blocks of three; the person was not told which condition a song came from.
- random: any song in the pool. popularity: one of the 10 most popular left. personal: one of the 10 songs the saved profile predicts highest (profile frozen; it does not learn during the test).
- Pool: songs from the gym/party/chill/happy/sad/study moods, minus every song already rated (the profile was trained on those), one song per artist.
- Label: the explicit thumb after each song. The song stops as soon as the thumb is given.
- Share liked = thumbs up / songs. 95% interval: Wilson. Difference range: normal approximation.
- Gate fixed beforehand: personal >= popularity + 15 points and >= random + 25 points, 40+ songs per condition.

## 4. Limits

- One person, 40 songs per condition: a share is known only to about +/-15 points, so a 15-point gap is suggestive.
- The camera is not used here. The thumb is the label.
- The profile was trained on every earlier label, so this session is its first test on songs it never saw.


## 5. Why it failed (analysis done after seeing the result; not part of the pre-set gate)
- **Popularity won clearly:** 72% liked (29 of 40) versus 28% for random and 30% for the personal model. Personal minus popularity
  was -42 points (95% range -62 to -23): the personal model is **worse** than just playing hits, and no better than random.
- **Same in both halves of the session:** first 60 songs popularity 13/20, personal 7/20, random 6/20; last 60: 16/20, 5/20, 5/20.
  So it is not tiredness or a drift.
- **Popularity predicts the thumbs; the saved profile does not.** Over all 120 songs, popularity ranks liked above disliked with AUC 0.76;
  the profile gets 0.44 (below chance). On the 40 random songs only (an unselected sample): popularity 0.70, profile 0.62 (noisy).
- **The personal picks were narrow and missed:** about half were in a few genres the profile favoured (electro 1 of 4 liked,
  pagode 0 of 4, folk 1 of 3, j-pop 0 of 2). The profile had only 35 liked songs, and the earlier cross-validation (0.915) was
  inflated by near-duplicate liked songs picked by the hunt (step04b). It learned that region, not a general taste.
- **Different bars for "liked":** the profile was trained on preference 4-5 ("want to hear it a lot", 12% of songs), but a thumb up is
  looser ("I liked it", 37% of all songs here, 72% of hits). Hits clear the looser bar often even though most are not top-rated.
  In Step 4 the hit band also had the highest preference ratings (2.3 vs 1.7 for unknown songs).
- **Limits:** 40 songs per condition, one person, the explicit thumb after a few seconds of each song (familiar hits are easy to like quickly).

## 6. What this means
Layer-3 gate: **FAIL**. The personal model, as trained, does not beat the popularity-only behaviour. The robot's earlier
"mainstream bias" default (a popularity prior that fades as it learns) is better supported by this data than a profile alone.
Next options (untested, to be judged on a fresh session): train on thumbs instead of 4-5 ratings; add the 120 live thumbs as labels;
combine the profile with a popularity prior.
