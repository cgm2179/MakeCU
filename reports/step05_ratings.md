# Step 4 report: two-question ratings (quality vs preference)

- Generated: 2026-11-07 22:37:06  
- Code: rl/eval-and-profiles @ f4f91a8  
- Python 3.9.0 on Darwin  
- User: adi

## 1. Unit tests: ALL PASSED

- Ran 163 tests in 59.2s: 163 passed, 0 failed, 0 errors, 0 skipped.
- Skipped tests need the camera libraries or the Spotify CSV and are skipped on a plain Python install; a skip is not a pass.

| test file | passed | failed | skipped |
|---|---|---|---|
| integration.tests.test_camera_loop | 13 | 0 | 0 |
| integration.tests.test_core_loop | 2 | 0 | 0 |
| integration.tests.test_cv_to_rl | 1 | 0 | 0 |
| motion.tests.test_arms | 2 | 0 | 0 |
| music.tests.test_audio | 16 | 0 | 0 |
| music.tests.test_dataset_pipeline | 4 | 0 | 0 |
| music.tests.test_gemini_lookup | 21 | 0 | 0 |
| music.tests.test_mainstream_bias | 11 | 0 | 0 |
| music.tests.test_playlist | 3 | 0 | 0 |
| rl.tests.test_agreement | 15 | 0 | 0 |
| rl.tests.test_bandit | 6 | 0 | 0 |
| rl.tests.test_eval | 11 | 0 | 0 |
| rl.tests.test_profile | 6 | 0 | 0 |
| rl.tests.test_ratings | 22 | 0 | 0 |
| rl.tests.test_report | 4 | 0 | 0 |
| rl.tests.test_signals | 23 | 0 | 0 |

## 2. What the ratings show

Data: this person's own ratings of 30-second previews.

**Overall: FAIL**

| gate | status | detail |
|---|---|---|
| 150 songs rated, 30 held out | PASS | 150 rated, 30 held out |
| the two ratings really differ | FAIL | 1% of songs differ by 2+ points (need 15%) |
| re-rating is consistent | PASS | within 1 point: quality 95%, preference 100% (need 80% each) |

| rating | 1 | 2 | 3 | 4 | 5 | mean |
|---|---|---|---|---|---|---|
| quality | 16 | 85 | 38 | 8 | 3 | 2.31 |
| preference | 41 | 80 | 17 | 10 | 2 | 2.01 |

- **They differ by 2+ points on 2 of 150 songs (1%; need 15%).** 2 are good-but-not-for-me (quality above preference) and 0 are guilty pleasures (preference above quality). These are the songs that teach the model the difference.
- Rank correlation between quality and preference: 0.70 (1 = always the same ranking, 0 = unrelated).
- Preference as yes/no labels for the evaluation tool: 12 liked (4-5), 121 disliked (1-2), 17 undecided (3, left out).
- Warning: one rating value makes up 57% of an answer set; a narrow range gives a model little to learn from.

| coverage | songs |
|---|---|
| mood: aggressive | 41 |
| mood: calm | 33 |
| mood: melancholy | 39 |
| mood: upbeat | 37 |
| popularity: hit (65 and over) | 54 |
| popularity: known (35 to 65) | 50 |
| popularity: unknown (popularity under 35) | 46 |
| distinct genres | 78 |
| most common genre | ambient (4) |

- **Does dataset popularity stand in for 'quality'?** Rank correlation of your quality rating with the song's popularity: 0.19; of your preference with popularity: 0.28. If quality tracks popularity much more closely than preference does, popularity is a fair stand-in for 'objective'; if neither does, it is not, and the plan's 'popularity as the objective proxy' needs rethinking.

- **Re-rating (test-retest), 20 songs:** within 1 point: quality 95%, preference 100%, both 95%; identical on both questions 30%. Need 80% on each question.

## 3. How the numbers were produced

- **Collection:** songs are drawn evenly across mood (upbeat, aggressive, calm, melancholy) and three popularity bands (under 35, 35 to 65, 65 and over), at most one song per artist and 6 per genre. Each plays as a 30-second preview; the questions appear after 15 seconds and the clip keeps playing. Every 5th rated song is saved as held out (the final exam), so 30 of 150.
- **Disagreement:** share of songs where |quality − preference| >= 2.
- **Rank correlation:** Spearman, the correlation of the two ratings' ranks (ties share the average rank).
- **Test-retest:** the same 20 songs are rated again at least 12 hours later without showing the earlier answers; a song is 'within 1 point' when the new rating is 1 or less away from the old.
- **Gates (proposed, fixed before any ratings):** 150 songs with 30 held out; disagreement >= 15%; re-rating within 1 point on >= 80% for each question.

## 4. What these numbers do and do not show

- One person. Quality as one person rates it is that person's opinion of quality, not an objective fact.
- 30-second previews are not whole songs, and the order of songs, tiredness and mood on the day all move ratings. The re-rating measures part of that noise.
- This step only collects and checks labels. Whether a model can learn from them is tested in later steps (base model, fine-tune, live test).
- Reports contain counts and aggregates only, never song titles or the raw ratings.

## Notes added after the run
- The re-rating happened about 1.3 hours after the first pass (`--force`), not 12+ hours, so remembered answers may help.
- Exact repeats: quality 13 of 20, preference 10 of 20; within 1 point is much looser.
- None of the 20 songs had a first preference of 4-5, so the liked songs the bandit learns from were not re-tested.
