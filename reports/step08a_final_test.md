# Step 8a: final held-out test (L3.1)

Date: 2026-11-08. Decision (user, 2026-11-08): **report the existing once-only result; do not read the held-out songs a second time.**
No code or model was run for this report; the numbers are copied from the lock file `data/eval/final_adi.json` (written 2026-11-07, Step 4c; see `step04c_final_test.md`).

## Result (30 held-out songs: 4 liked, 26 disliked; models trained on 137 development labels)
| model | AUC | precision@10 |
|---|---|---|
| random | 0.721 | 0.30 |
| popularity-only | 0.827 | 0.30 |
| bandit (from scratch) | 0.745 | 0.30 |
| base-only | **not scored** | |
| from-scratch logistic | **not scored** | |
| fine-tuned (base + personal layer) | **not scored** | |

## What this does and does not show
- L3.1 asks for a table of random / popularity-only / base-only / from-scratch / fine-tuned. Only three of the five were scored, because the held-out set was read once in Step 4c,
  before the base model and the personal layer existed. The checklist says the test is run exactly once, so base-only and fine-tuned were deliberately **not** scored on it. L3.1 is therefore **incomplete, not passed or failed**.
- With 4 liked songs the AUC rests on 4 x 26 = 104 pairs; one re-ranked song moves it several points, and random scoring reached 0.72 by luck. The three models cannot be told apart on this set.
- The held-out set was not fully independent even for the first read: the Step 4b hunt picked songs using a bandit trained on all 150 ratings, including the held-out ones (fixed in code afterwards; see `step04c_final_test.md`).

## What stands in for it
- Step 7 cross-validation on 264 thumbs (every song scored by a model that never saw it): popularity-only 0.712, from-scratch logistic 0.694, fine-tuned 0.686, base-only 0.672, bandit 0.578. See `step07_personal_layer.md`.
- The Step 8b live session on never-rated songs is the clean test of the fine-tuned model (`step08_preregistration.md`).

## Layer-3 status
L3.1 incomplete by decision; L3.2 pending the live session; L3.3 tooling exists (PR #16), not run; L3.4 write-up exists (`step09_writeup.md`) and needs updating with Steps 6 to 8.
