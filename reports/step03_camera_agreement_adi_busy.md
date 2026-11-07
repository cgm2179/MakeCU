# Step 3 report: camera vs thumbs agreement (P0.3)

- Generated: 2026-11-07 18:41:43  
- Code: rl/eval-and-profiles @ e47cf87 (+ uncommitted changes)  
- Python 3.9.0 on Darwin  
- User: adi_busy

## 1. Unit tests: ALL PASSED

- Ran 137 tests in 60.0s: 137 passed, 0 failed, 0 errors, 0 skipped.
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
| rl.tests.test_signals | 23 | 0 | 0 |

## 2. How well the camera agrees with the person

Data: this person's own thumbs up/down next to the camera's reading for the same songs.

**Result: FAIL: the camera matched the thumb on 50% of 40 songs (gate 75%). Action: use the explicit thumbs as the training label and treat the camera only as a secondary signal.**

- Songs rated: 40. Usable (a face was seen with confidence >= 0.1): 40. Excluded: 0.

|  | camera says like | camera says dislike |
|---|---|---|
| thumb UP | 7 | 9 |
| thumb DOWN | 11 | 13 |

| measure | value | what it means |
|---|---|---|
| agreement (accuracy) | 50% | share of songs where the camera's sign matched the thumb; 95% interval 35% to 65% |
| always-guess-majority | 60% | what a camera that ignored your face and always said the more common thumb would score |
| balanced accuracy | 49% | average of 'liked songs the camera caught' and 'disliked songs the camera caught'; fair when thumbs are lopsided |
| kappa | -0.02 | agreement beyond chance: 0 = chance, 1 = perfect |
| AUC of liking vs thumb | 0.523 | chance a liked song got a higher camera score than a disliked one; ignores the 0 cut-off |
| clear reactions (liking at least 0.2 from 0, either way) | 50% of 14 | agreement when the camera was not near neutral |
| mean liking, thumb UP | 0.08 | should be clearly above 0 |
| mean liking, thumb DOWN | 0.05 | should be clearly below the thumb-UP mean |

## 2b. Re-test with a per-person calibration (Step 3b)

**Result: FAIL: per-person calibration, each song called by a rule that never saw it: accuracy 48% (need 75%), balanced accuracy 52% (need 65%), AUC 0.52 (need 0.65).**

The general rule ('liking > 0.0025 = liked') was replaced by a rule fitted to this person's thumbs, starting from the general rule and held close to it when data is thin. Each song is predicted by a model that never saw that song (leave-one-out), so the score is not flattered by fitting and testing on the same songs.

| measure | value | notes |
|---|---|---|
| usable songs | 40 | 16 thumb up, 24 thumb down |
| accuracy (leave-one-out) | 48% | need 75%; always-majority scores 60% |
| balanced accuracy | 52% | need 65%; catches liked: 75%, catches disliked: 29% |
| AUC | 0.523 | need 0.65; 0.5 = chance |
| personal cut-off | -0.09 | the general rule is 0.0025; a song counts as liked when the camera's calibrated liking is above this |

## 2c. Do movement and reaction-dynamics features help? (Step 3c)

**Result: NOT ENOUGH DATA: 40 songs with features (16 up, 24 down); need 60 songs with at least 15 of each thumb. So far (cross-validated): old reading (last 3 s liking): AUC 0.44, reaction dynamics (peak, range, build...): AUC 0.58, movement (nodding, head motion): AUC 0.84, all features: AUC 0.81.**

| camera measurement used | features | AUC (5-fold, repeated 3 times: each song scored by a model that never saw it) |
|---|---|---|
| old reading (last 3 s liking) | 1 | 0.444 |
| reaction dynamics (peak, range, build...) | 6 | 0.581 |
| movement (nodding, head motion) | 4 | 0.840 |
| all features | 11 | 0.809 |

Gain of 'all features' over the old reading: +0.365; 95% resampling range +0.115 to +0.510 (if the range includes 0, the gain could be noise).

| feature (standardised) | weight | reading |
|---|---|---|
| bob_share | +0.64 | raises P(liked) |
| low_liking | +0.18 | raises P(liked) |
| motion | -0.17 | lowers P(liked) |
| build_liking | -0.17 | lowers P(liked) |
| mean_liking | +0.14 | raises P(liked) |
| vert_motion | -0.14 | lowers P(liked) |

Weights come from one fit on all songs and are descriptive only: with this few songs a single weight is not reliable.

| setting | songs | AUC (all features) |
|---|---|---|
| busy | 40 | 0.802 |

Checks on the 'all features' result (the primary comparison fixed beforehand):

| check | result |
|---|---|
| thumbs shuffled at random (100 times) | models reach AUC 0.68 or more only 5% of the time by luck; real result 0.81; chance of doing this well by luck p = 0.020 |
| bob_share alone, first half of the session (20 songs, 11 liked) | AUC 0.859 |
| bob_share alone, second half (20 songs, 5 liked) | AUC 0.947 |
| the songs' own audio (tempo, danceability, energy...) alone | AUC 0.496; if this were high the movement could just be the song |

| single feature | AUC on its own (0.5 = chance) |
|---|---|
| bob_share | 0.880 |
| motion | 0.365 |
| low_liking | 0.630 |
| vert_motion | 0.370 |
| mean_liking | 0.602 |

Single features were looked at after the fact, so the best one is likely to look better than it is. That is why it is re-tested on new songs below.

**Camera weight for the learner:** 0.62 (from cross-validated AUC 0.809 on 40 songs; 0 = ignore the camera, 1 = trust it like a thumb). A weight needs at least 30 songs with features and 8 of each thumb; below that the camera is ignored.

## 2d. Confirmation on new songs

Rule fixed before the quiet-room session: **bob_share** (share of up-and-down head movement in the 1 to 3.5 Hz nodding band, picked because it stood out in the busy-group session) must reach AUC >= 0.70 by itself on at least 40 songs from other sessions, 12 or more of each thumb. Nothing is fitted, so there is nothing to over-fit.

**Result: NOT ENOUGH DATA: 0 songs outside the busy session (0 up, 0 down); need 40 with at least 12 of each thumb.**

## 3. How the numbers were produced

- **Collection:** songs from the gym and party moods play as 10-second iTunes previews. Before the first song the camera measures a resting baseline (neutral face, guarded against smiling or moving). At the end of each clip the camera reading is the average liking over the last 3 seconds, adjusted relative to that baseline so a neutral face is about 0. Then the person presses up or down.
- **Camera's call:** like if calibrated liking > 0.0025, otherwise dislike. A reading at or below that counts as dislike.
- **Excluded:** a song is not judged if no face was seen or mean confidence < 0.1; those songs are counted above, not hidden.
- **Agreement:** (camera-like and thumb-up) + (camera-dislike and thumb-down), divided by usable songs.
- **95% interval:** Wilson score interval for a proportion, centre (p + z²/2n)/(1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²)/(1 + z²/n), z = 1.96.
- **Kappa:** (agreement − expected) / (1 − expected), where expected is the agreement two independent raters would reach by chance given how often each says like.
- **Gate (proposed, fixed beforehand):** at least 40 usable songs and agreement >= 75%.
- **Step 3b (per-person calibration):** the cut-off on the camera's calibrated liking starts at the general rule (0). It is moved to the value that best catches both this person's liked and disliked songs (best balanced accuracy on the training songs), then shrunk toward 0.0025 by n / (n + 10) so a handful of songs barely moves it. Each song is called by a cut-off fitted without it (leave-one-out). Gate set before it was run on any data: accuracy >= 75%, balanced accuracy >= 65%, AUC >= 0.65, and at least 10 songs of the rarer thumb (otherwise nothing can be judged). Moving a cut-off cannot change the ranking of songs, so AUC stays low if the camera has no signal.

- **Step 3c (richer features):** per clip the tool records the reaction's mean, peak, low, range, build (last third minus first third) and last-3-seconds average; the smile probability minus the resting smile; and face movement: speed of the face centre, vertical (nodding) speed, and the share of vertical movement in the 1 to 3.5 Hz head-bob band with its amplitude. Movement and smile are measured against the person's own resting baseline from calibration. A logistic model per person (strong pull toward 'no effect', equal to 10 songs of evidence) turns the features into P(thumb up); every song is scored by a model fitted without it (stratified 5-fold cross-validation repeated 3 times; leave-one-out is avoided because it pushes weak-signal AUC toward 0), and AUC is computed on those scores. Gate set before any feature data existed: at least 60 songs with features and 15 of each thumb, all-features AUC >= 0.70, and at least 0.05 better than the old reading.
- **Camera weight:** 2 × (cross-validated AUC − 0.5), between 0 and 1. It is the share of a thumb's influence the camera gets when the learner combines thumbs, skips and the camera.

## 4. What these numbers do and do not show

- With 40 songs the 95% interval is about ±15 points; a score a few points either side of 75% is not decisive.
- One person, one room, gym and party songs only. It does not show the camera works for other people or other music.
- The thumb is treated as the truth, but it is a noisy label too (people press the wrong button, or rate the song rather than how they felt).
- If thumbs are lopsided (mostly up or mostly down), compare agreement with 'always-guess-majority' and look at balanced accuracy and kappa.
- If this fails, the plan is to train on thumbs and use the camera only as a secondary signal; the base model and personal layers are not affected.
