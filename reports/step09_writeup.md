> **Superseded by `step10_writeup.md`** (the final L3.4 write-up, with Steps 6 to 9, the camera replication and the new-user check). Kept as the earlier draft; the numbers below for the first live session are still correct.

# Learning a listener's music taste from their reactions: results, including what failed

One main user (Adi), about 700 labelled interactions, 24-hour hackathon build. Every threshold below was fixed before the data
was collected (docs/THREE_LAYER_CHECKPOINTS.md); step reports in `reports/` hold the full numbers and methods.

## Bottom line
The robot, its learner, the evaluation tools and the data collection all work and are tested (180 unit tests). On the question
that matters, **a model of this user's taste did not beat simply playing popular songs.** In the live test, popular songs were
liked 72% of the time; the personal model's picks were liked 30%, the same as random (28%). The one promising result is that the
shape of a face's reaction over a clip predicted the thumb with AUC about 0.80 in a quiet room; it has not been replicated.

## Results against the thresholds set in advance
| Check | Threshold | Result | Verdict |
|---|---|---|---|
| Camera call matches thumb (148 songs, quiet room) | >= 75% | 59% (always guessing "dislike" scores 65%) | **FAIL** |
| Camera features beat the old reading (Step 3c) | AUC >= 0.70 and gain >= +0.05 | all features 0.79, gain +0.10 (range -0.01 to +0.19) | PASS, not replicated |
| Nodding signal holds in a new session | AUC >= 0.70 on 40+ new songs | 0.56 on 70 songs | **FAIL** |
| 150 songs rated, 30 held out | 150 / 30 | 150 / 30 | PASS |
| Quality and preference ratings really differ | >= 15% of songs differ by 2+ points | 1% | **FAIL** |
| Re-rating is consistent | >= 80% within 1 point | quality 95%, preference 100% (re-rated only 1.3 h later; none of the liked songs were retested) | PASS, weak evidence |
| Final held-out test (30 songs, 4 liked) | beat baselines | bandit 0.745, popularity 0.827, random 0.721 AUC | inconclusive |
| Live session, 40 songs per condition | personal >= popularity + 15 pts and >= random + 25 pts | personal 30%, popularity 72%, random 28% | **FAIL** |
| New-user check (L3.3), 40 songs/condition | base >= popularity + 15 pts and >= random + 25 pts | personal 42%, popularity 48%, random 25% (`step09_new_user_check.md`) | **FAIL** (-5 pts vs. popularity, +18 vs. random) |

Offline cross-validation of the bandit reached AUC 0.915, but that number was inflated: the "liked-song hunt" picked songs that
resemble each other, so folds contained near-duplicates. The held-out and live tests are the honest figures.

## What the failures taught us
- **Popularity predicts this user's thumbs (AUC 0.71 on 264 thumbs); nothing we added improved on it.** The bandit trained on
  thumbs scored 0.58, blends with popularity 0.62 to 0.71, audio features alone 0.60.
- **"Liked" means different things at different bars.** The profile was trained on ratings of 4-5 (12% of songs); a thumb up is
  looser (37% of songs, 72% of hits). A model of the strict bar did not transfer to the casual one.
- **Objective quality and personal preference were not separable for this user:** the two ratings moved together (rank
  correlation 0.70), so the planned "base model that knows the difference" was dropped for the fallback (popularity as the
  objective proxy, the user's labels as the only preference data). Layers 1 and 2 as designed could not be tested without a base.
- **The camera's like/dislike sign is not usable as a label.** Thumbs stay the training label; the camera is a secondary signal.

## What works
- A contextual bandit that learns live from thumbs and reactions, with a fading popularity prior (default `MAINSTREAM_BIAS` 0.5).
- Evaluation harness with a once-only final test, resumable collection tools (agreement check, two-question ratings, hunt, live
  session), saved per-user profiles, and a report for every step. These are what allowed the negative results to be measured
  instead of guessed.

## Limits
- One main user; the labels come from one person at one time, a few hundred per test, so shares carry about +/-15 points.
- The final held-out test is spent and had only 4 liked songs. The profile was trained on all labels afterwards.
- The 0.80 camera result was found by looking at features after the fact; it needs a fresh session before it is trusted or used
  as a reward. The nodding feature did not replicate.
- Song features are Spotify-dataset numbers; Gemini lookups, where used, are rough estimates.
- The new-user check (`step09_new_user_check.md`) is one person; still not done is a next-day
  re-rating that includes the liked songs.

## Recommended next steps
1. Replicate the camera reaction-shape result on a new day (40+ songs, 12+ per thumb) before turning the camera reward on.
2. Keep the popularity-leaning default and let the bandit learn within a session; judge it on a live session with thumbs.
3. Run the new-user check on a third person (`step09_new_user_check.md`: base alone failed the gate for
   Cameron too, though it tracked taste better there than the fine-tuned model did for Adi) to see
   whether that pattern holds.
