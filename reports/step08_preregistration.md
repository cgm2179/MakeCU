# Step 8b pre-registration (committed before the new live session was built or run)

Date: 2026-11-08. Checklist Step 8b / checkpoint L3.2: blind live session with three hidden conditions.

## What changes compared with the first live session (reports/step08_live_session_v1.md)
The first session tested the old ratings-trained profile as condition C. It is replaced by the **Step 7 model**: the frozen base (`models/base_model_v2.json`,
hash `87b6175d...`) plus Adi's personal layer (`data/profiles/adi_personal_v2.json`, fitted on Adi's 264 thumbs). Nothing is retrained for the session.

## Conditions (same as L3.2; mixed in random blocks of three, the person is not told which)
- A: a random song from the candidate pool.
- B: one of the 10 most popular songs left in the pool.
- C: one of the 10 songs the fine-tuned model (base + personal layer) scores highest.
Label = the explicit thumb after each song; the song stops as soon as the thumb is given. The model is frozen: it does not learn during the session.

## No leakage
The pool excludes every song Adi has already rated, thumbed in the camera sessions, or thumbed in the first live session, because the personal layer was fitted on those.
One song per artist in the session.

## Pass marks (unchanged from L3.2)
C liked-share >= B + 15 points AND >= A + 25 points, with 40+ songs per condition. At 40 songs the margin of error on a share is about +/-15 points,
so a pass is suggestive and a fail is not proof either. The result is reported as it comes out; no marks are changed.

## Expectation stated in advance
Offline cross-validation (Step 7) found popularity-only (AUC 0.712) above the fine-tuned model (0.686) on Adi's thumbs, and the first live session had popularity at 72% liked.
The most likely outcome is that condition B beats condition C. A fail would not be surprising.

## Step 8a (final held-out test)
The 30-song set was read once in Step 4c (lock file `data/eval/final_adi.json`); the checklist says "exactly once". The table in Step 4c has random, popularity-only and the bandit only.
Any further read of the held-out songs needs the user's explicit decision, and is reported as a second read if it happens.
