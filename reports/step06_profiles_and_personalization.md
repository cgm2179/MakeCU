# Step 6: layer-1 decision, saved profiles (P0.4) and personal training (L2.2, partial)

Date: 2026-11-07. Tests: 170 ran, 0 failed, 0 skipped.

## Decision (layer 1)
The Step 4 gate "quality and preference ratings really differ" failed (1% of songs, need 15%). The checkpoint plan says that
then the objective-vs-preference idea has nothing to learn from for this user, and gives a fallback: **popularity as the
objective proxy, the user's own labels as the only preference data**, and say plainly that the general-preference part is thin.
External many-person data (Last.fm, Million Song Dataset) was not pursued: large download, licence checks, uncertain join
rate to the 80k tracks, and limited time. Consequence: **there is no base model** (`base_model_id = none-v0`); the only
"base" is the agent's mainstream-popularity prior.

## What was built
- `run(..., user=...)` / `run_loop --user NAME`: load the saved profile at the start (fresh if none), save at the end.
  A damaged or wrong-base profile raises an error instead of being replaced.
- `rl/personalize.py`: trains a profile from `data/ratings/<user>.csv` (preference 4-5 = liked, 1-2 = disliked, 3 left out;
  hunt songs included; held-out songs only with `--include-heldout`). `python3 -m rl.personalize --user adi --include-heldout`.
- 7 new tests (`rl/tests/test_personalize.py`): label rules, trained profile survives save/load with identical predictions
  and prefers the liked genres, loop saves then continues from the saved state, damaged profile is not overwritten, no user = no file.

## Checkpoint status
| Item | Status | Evidence |
|---|---|---|
| P0.4 save / load profile | **PASS** | restart gives identical predictions for 20 songs (test_profile), different base refused, loop wiring tested |
| L1.1 data decision | **DECIDED (fallback)** | above |
| L1.3 train the base | **SKIPPED** | no base; the separability test (q vs quality > q vs preference) cannot be run |
| L2.1 user layer over a frozen base | **NOT APPLICABLE** | with no base the user layer is the whole model |
| L2.2 fine-tune beats base-only / needs half the labels | **NOT TESTABLE** | no base to compare against; only the from-scratch bandit exists |

What can be said: the personal bandit trained on 167 labels (35 liked) was saved as `data/profiles/adi.json`.
Its strongest liked features were genre electro (+0.16) and soul (+0.11); strongest dislikes were slow/mid tempo, high
danceability, calm mood, low energy. Cross-validated AUC 0.915 (inflated by hunt selection, see step04b) and final held-out
AUC 0.745 vs popularity 0.827 on only 4 liked songs (inconclusive, step04c).

## Limits
One user, 167 labels, 35 liked. The final held-out test is spent. This profile was trained on every label including the held-out
ones, so it cannot be scored on them again. The next honest number is a live session (L3.2).
