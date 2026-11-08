# Step 4c: one-time final held-out test (adi)

Date: 2026-11-07. Lock file: data/eval/final_adi.json (git-ignored); this test cannot be re-run.

## Method
Train on all 137 development labels (31 liked, 106 disliked; includes the 40 hunt songs). Score the 30 held-out songs
(all with a yes/no label: **4 liked, 26 disliked**). Labels: preference 4-5 liked, 1-2 disliked.

## Result
| model | AUC | precision@10 |
|---|---|---|
| random | 0.721 | 0.30 |
| popularity-only | 0.827 | 0.30 |
| bandit (from scratch) | 0.745 | 0.30 |

## How to read it
- **No claim of a win.** With only 4 liked songs the AUC rests on 4 x 26 = 104 pairs; one song moves it by several points.
  Random scoring reached 0.72 by luck, which shows how wide the noise is. The three models cannot be told apart here.
- **Cross-validation said 0.915, this says 0.745.** Cross-validation was inflated by near-duplicate liked songs picked by the
  hunt (see step04b). The held-out songs came from the unbiased stratified sample, so this is the more honest, if noisy, figure:
  the bandit is modestly above chance but not shown to beat popularity.
- **Leak:** hunt songs were chosen by a bandit trained on all 150 ratings including the held-out 30 (fixed in code afterwards).
  The effect would favour the bandit, and it still did not beat popularity-only.
- Consistency of ratings (Step 5) was waived, not measured.

## Conclusion
Layer-1 taste learning is not demonstrated on held-out data. It is not refuted either: the test is too small. More liked
songs in an unbiased held-out set (for example 30 more stratified songs with a fresh lock) would be needed for a real answer.
