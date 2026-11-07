# Step 3 report: camera vs thumbs agreement (P0.3)

- Generated: 2026-11-07 17:51:20  
- Code: rl/eval-and-profiles @ a55ac16 (+ uncommitted changes)  
- Python 3.9.0 on Darwin  
- User: adi

## 1. Unit tests: ALL PASSED

- Ran 113 tests in 47.7s: 113 passed, 0 failed, 0 errors, 0 skipped.
- Skipped tests need the camera libraries or the Spotify CSV and are skipped on a plain Python install; a skip is not a pass.

| test file | passed | failed | skipped |
|---|---|---|---|
| integration.tests.test_camera_loop | 13 | 0 | 0 |
| integration.tests.test_core_loop | 2 | 0 | 0 |
| music.tests.test_audio | 16 | 0 | 0 |
| music.tests.test_dataset_pipeline | 4 | 0 | 0 |
| music.tests.test_gemini_lookup | 21 | 0 | 0 |
| music.tests.test_mainstream_bias | 11 | 0 | 0 |
| music.tests.test_playlist | 3 | 0 | 0 |
| rl.tests.test_agreement | 15 | 0 | 0 |
| rl.tests.test_bandit | 6 | 0 | 0 |
| rl.tests.test_eval | 11 | 0 | 0 |
| rl.tests.test_profile | 6 | 0 | 0 |
| rl.tests.test_report | 4 | 0 | 0 |

## 2. How well the camera agrees with the person

Data: this person's own thumbs up/down next to the camera's reading for the same songs.

**Result: FAIL: the camera matched the thumb on 54% of 78 songs (gate 75%). Action: use the explicit thumbs as the training label and treat the camera only as a secondary signal.**

- Songs rated: 80. Usable (a face was seen with confidence >= 0.1): 78. Excluded: 2.

|  | camera says like | camera says dislike |
|---|---|---|
| thumb UP | 14 | 6 |
| thumb DOWN | 30 | 28 |

| measure | value | what it means |
|---|---|---|
| agreement (accuracy) | 54% | share of songs where the camera's sign matched the thumb; 95% interval 43% to 64% |
| always-guess-majority | 74% | what a camera that ignored your face and always said the more common thumb would score |
| balanced accuracy | 59% | average of 'liked songs the camera caught' and 'disliked songs the camera caught'; fair when thumbs are lopsided |
| kappa | 0.13 | agreement beyond chance: 0 = chance, 1 = perfect |
| AUC of liking vs thumb | 0.674 | chance a liked song got a higher camera score than a disliked one; ignores the 0 cut-off |
| clear reactions (liking at least 0.2 from 0, either way) | 71% of 14 | agreement when the camera was not near neutral |
| mean liking, thumb UP | 0.09 | should be clearly above 0 |
| mean liking, thumb DOWN | -0.01 | should be clearly below the thumb-UP mean |

## 2b. Re-test with a per-person calibration (Step 3b)

**Result: FAIL: per-person calibration, each song called by a rule that never saw it: accuracy 63% (need 75%), balanced accuracy 64% (need 65%), AUC 0.67 (need 0.65).**

The general rule ('liking > 0.0025 = liked') was replaced by a rule fitted to this person's thumbs, starting from the general rule and held close to it when data is thin. Each song is predicted by a model that never saw that song (leave-one-out), so the score is not flattered by fitting and testing on the same songs.

| measure | value | notes |
|---|---|---|
| usable songs | 78 | 20 thumb up, 58 thumb down |
| accuracy (leave-one-out) | 63% | need 75%; always-majority scores 74% |
| balanced accuracy | 64% | need 65%; catches liked: 65%, catches disliked: 62% |
| AUC | 0.674 | need 0.65; 0.5 = chance |
| personal cut-off | 0.04 | the general rule is 0.0025; a song counts as liked when the camera's calibrated liking is above this |

## 3. How the numbers were produced

- **Collection:** songs from the gym and party moods play as 10-second iTunes previews. Before the first song the camera measures a resting baseline (neutral face, guarded against smiling or moving). At the end of each clip the camera reading is the average liking over the last 3 seconds, adjusted relative to that baseline so a neutral face is about 0. Then the person presses up or down.
- **Camera's call:** like if calibrated liking > 0.0025, otherwise dislike. A reading at or below that counts as dislike.
- **Excluded:** a song is not judged if no face was seen or mean confidence < 0.1; those songs are counted above, not hidden.
- **Agreement:** (camera-like and thumb-up) + (camera-dislike and thumb-down), divided by usable songs.
- **95% interval:** Wilson score interval for a proportion, centre (p + z²/2n)/(1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²)/(1 + z²/n), z = 1.96.
- **Kappa:** (agreement − expected) / (1 − expected), where expected is the agreement two independent raters would reach by chance given how often each says like.
- **Gate (proposed, fixed beforehand):** at least 40 usable songs and agreement >= 75%.
- **Step 3b (per-person calibration):** the cut-off on the camera's calibrated liking starts at the general rule (0). It is moved to the value that best catches both this person's liked and disliked songs (best balanced accuracy on the training songs), then shrunk toward 0.0025 by n / (n + 10) so a handful of songs barely moves it. Each song is called by a cut-off fitted without it (leave-one-out). Gate set before it was run on any data: accuracy >= 75%, balanced accuracy >= 65%, AUC >= 0.65, and at least 10 songs of the rarer thumb (otherwise nothing can be judged). Moving a cut-off cannot change the ranking of songs, so AUC stays low if the camera has no signal.

## 4. What these numbers do and do not show

- With 78 songs the 95% interval is about ±11 points; a score a few points either side of 75% is not decisive.
- One person, one room, gym and party songs only. It does not show the camera works for other people or other music.
- The thumb is treated as the truth, but it is a noisy label too (people press the wrong button, or rate the song rather than how they felt).
- If thumbs are lopsided (mostly up or mostly down), compare agreement with 'always-guess-majority' and look at balanced accuracy and kappa.
- If this fails, the plan is to train on thumbs and use the camera only as a secondary signal; the base model and personal layers are not affected.
