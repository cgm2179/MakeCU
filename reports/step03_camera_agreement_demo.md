# Step 3 report: camera vs thumbs agreement (P0.3)

- Generated: 2026-11-07 17:05:38  
- Code: rl/eval-and-profiles @ e81b3d6 (+ uncommitted changes)  
- Python 3.9.0 on Darwin  
- User: demo

## 1. Unit tests: ALL PASSED

- Ran 106 tests in 47.8s: 106 passed, 0 failed, 0 errors, 0 skipped.
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
| rl.tests.test_agreement | 8 | 0 | 0 |
| rl.tests.test_bandit | 6 | 0 | 0 |
| rl.tests.test_eval | 11 | 0 | 0 |
| rl.tests.test_profile | 6 | 0 | 0 |
| rl.tests.test_report | 4 | 0 | 0 |

## 2. How well the camera agrees with the person

**SYNTHETIC data: a scripted fake face and fake thumbs, not a real person.**

**Result: PASS: the camera matched the thumb on 95% of 40 songs (gate 75%).**

- Songs rated: 40. Usable (a face was seen with confidence >= 0.1): 40. Excluded: 0.

|  | camera says like | camera says dislike |
|---|---|---|
| thumb UP | 29 | 0 |
| thumb DOWN | 2 | 9 |

| measure | value | what it means |
|---|---|---|
| agreement (accuracy) | 95% | share of songs where the camera's sign matched the thumb; 95% interval 83% to 99% |
| always-guess-majority | 72% | what a camera that ignored your face and always said the more common thumb would score |
| balanced accuracy | 91% | average of 'liked songs the camera caught' and 'disliked songs the camera caught'; fair when thumbs are lopsided |
| kappa | 0.87 | agreement beyond chance: 0 = chance, 1 = perfect |
| AUC of liking vs thumb | 0.962 | chance a liked song got a higher camera score than a disliked one; ignores the 0 cut-off |
| clear reactions (liking at least 0.2 from 0, either way) | 97% of 36 | agreement when the camera was not near neutral |
| mean liking, thumb UP | 0.75 | should be clearly above 0 |
| mean liking, thumb DOWN | -0.46 | should be clearly below the thumb-UP mean |

## 3. How the numbers were produced

- **Collection:** songs from the gym and party moods play as 10-second iTunes previews. Before the first song the camera measures a resting baseline (neutral face, guarded against smiling or moving). At the end of each clip the camera reading is the average liking over the last 3 seconds, adjusted relative to that baseline so a neutral face is about 0. Then the person presses up or down.
- **Camera's call:** like if calibrated liking > 0, otherwise dislike. A reading exactly at the baseline counts as dislike.
- **Excluded:** a song is not judged if no face was seen or mean confidence < 0.1; those songs are counted above, not hidden.
- **Agreement:** (camera-like and thumb-up) + (camera-dislike and thumb-down), divided by usable songs.
- **95% interval:** Wilson score interval for a proportion, centre (p + z²/2n)/(1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²)/(1 + z²/n), z = 1.96.
- **Kappa:** (agreement − expected) / (1 − expected), where expected is the agreement two independent raters would reach by chance given how often each says like.
- **Gate (proposed, fixed beforehand):** at least 40 usable songs and agreement >= 75%.

## 4. What these numbers do and do not show

- With 40 songs the 95% interval is about ±8 points; a score a few points either side of 75% is not decisive.
- One person, one room, gym and party songs only. It does not show the camera works for other people or other music.
- The thumb is treated as the truth, but it is a noisy label too (people press the wrong button, or rate the song rather than how they felt).
- If thumbs are lopsided (mostly up or mostly down), compare agreement with 'always-guess-majority' and look at balanced accuracy and kappa.
- If this fails, the plan is to train on thumbs and use the camera only as a secondary signal; the base model and personal layers are not affected.
