# Step 2 report: Evaluation harness and profile save/load

- Generated: 2026-11-07 17:01:20  
- Code: rl/eval-and-profiles @ 68af41c (+ uncommitted changes)  
- Python 3.9.0 on Darwin

## 1. Unit tests: ALL PASSED

- Ran 98 tests in 9.7s: 89 passed, 0 failed, 0 errors, 9 skipped.
- Skipped tests need the camera libraries or the Spotify CSV and are skipped on a plain Python install; a skip is not a pass.

| test file | passed | failed | skipped |
|---|---|---|---|
| integration.tests.test_camera_loop | 5 | 0 | 9 |
| integration.tests.test_core_loop | 2 | 0 | 0 |
| music.tests.test_audio | 16 | 0 | 0 |
| music.tests.test_dataset_pipeline | 4 | 0 | 0 |
| music.tests.test_gemini_lookup | 21 | 0 | 0 |
| music.tests.test_mainstream_bias | 11 | 0 | 0 |
| music.tests.test_playlist | 3 | 0 | 0 |
| rl.tests.test_bandit | 6 | 0 | 0 |
| rl.tests.test_eval | 11 | 0 | 0 |
| rl.tests.test_profile | 6 | 0 | 0 |
| rl.tests.test_report | 4 | 0 | 0 |

## 2. How well the algorithm performs

Data: **SYNTHETIC** labels from a fake listener (likes jazz, classical, acoustic, piano; 10% flipped).
- 150 labeled songs: 108 liked (72%), 42 disliked. A model that always says 'liked' would be right 72% of the time, so read precision@10 against that, and prefer AUC (it ignores the like/dislike ratio).

| model | mean AUC | sd across folds | mean precision@10 | AUC per fold |
|---|---|---|---|---|
| random | 0.473 | 0.016 | 0.740 | 0.47, 0.47, 0.45, 0.49, 0.49 |
| popularity-only | 0.523 | 0.117 | 0.740 | 0.61, 0.52, 0.38, 0.67, 0.45 |
| bandit (from scratch) | 0.820 | 0.141 | 0.940 | 0.85, 0.89, 0.89, 0.57, 0.90 |

Bandit minus popularity-only, per fold: +0.24, +0.37, +0.52, -0.10, +0.45. Mean +0.297; the bandit is ahead in 4 of 5 folds.

Test-fold sizes: 31 (22 liked), 31 (22 liked), 30 (22 liked), 29 (21 liked), 29 (21 liked).

### Learning curve (mean AUC when only the first n training labels are used)

| model | n=10 | n=20 | n=40 | n=80 | n=120 |
|---|---|---|---|---|---|
| random | 0.473 | 0.473 | 0.473 | 0.473 | 0.473 |
| popularity-only | 0.523 | 0.523 | 0.523 | 0.523 | 0.523 |
| bandit (from scratch) | 0.695 | 0.743 | 0.799 | 0.797 | 0.796 |

## 3. How the numbers were produced

- **Labels:** each song has one label, 1 = liked, 0 = disliked. Songs come from the 80,834-song Spotify dataset.
- **Split:** 5-fold cross-validation, stratified (likes and dislikes are dealt out separately so every fold has both). Each fold is tested once, using a model trained only on the other 4 folds. Shuffle seed = 0, so rerunning gives identical folds. The final held-out songs (if any) are never used here.
- **AUC:** take every (liked, disliked) pair in the test fold; count 1 if the model scored the liked song higher, 0.5 for a tie, 0 otherwise; divide by the number of pairs. 0.5 = no better than chance, 1.0 = perfect.
- **precision@10:** rank the test-fold songs by model score, take the top 10, and report the share that were liked. Ties are broken randomly with a fixed seed, so a model that scores everything equally gets chance level.
- **random:** a random score per song. **popularity-only:** the dataset's popularity (0-100) as the score. **bandit (from scratch):** the live learner (rl/agent/bandit.py) starting at zero weights and learning from the training labels once each, in shuffled order, thumbs up = +1 and thumbs down = -1 at full confidence; a song's score is the sum of its learned feature weights (genre, artist, mood, energy/valence/danceability/tempo buckets).
- **Learning curve:** same test folds, but the model sees only the first n training labels, so you can see how fast it learns.
- **Worked example of AUC.** Take scores 0.9, 0.8, 0.3, 0.1 with labels liked, disliked, liked, disliked. Liked songs scored 0.9 and 0.3; disliked scored 0.8 and 0.1. Of the 4 liked-vs-disliked pairs (0.9 vs 0.8, 0.9 vs 0.1, 0.3 vs 0.8, 0.3 vs 0.1) the model orders 3 correctly, so AUC = 3/4 = 0.75. (Checked by a unit test.)


## 4. What these numbers do and do not show

- **The labels are synthetic.** They come from a fake listener who likes a fixed set of genres (with 10% random flips). This shows the measuring tool works and that the bandit can learn a genre taste. It says nothing about how well the robot will do for a real person.
- Real labels (Step 4) will replace this data; rerun `python3 -m rl.report` then.
- With 150 songs and 5 folds, each fold tests only about 30 songs, so a single fold's AUC is noisy (the 'sd across folds' column shows how noisy). Treat differences smaller than about 0.05 as unproven.
- Folds share training data, so the fold scores are not independent; the standard deviation is a rough guide, not a confidence interval.
- Not yet covered: the base model (Layer 1), personal fine-tuning (Layer 2) and a live test (Layer 3).
