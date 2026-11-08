# Policy replay pre-registration (committed before the replay was run)

Date: 2026-11-08. Question: does a policy that starts from the frozen base model and learns online from thumbs pick songs Adi likes more often than popularity-greedy,
within a 40-song session? (The robot's real job, which the offline AUC comparisons do not measure.)

## Data and method
- **Logged thumbs:** every explicit thumb Adi gave in the camera sessions (`data/agreement/adi.csv`, `adi_rep.csv`) and the live sessions (`data/live/adi.csv`, `adi_v2.csv`); one label per song (later overrides earlier); busy-group sessions excluded.
- **Replay:** each simulated session draws a random pool of 150 of these songs. A policy picks 40 songs one by one without repeats; when a song is picked its logged thumb is revealed and the policy may learn from it.
  Repeated for 300 sessions with different random pools (the same pools for every policy).
- **Metric:** share of the 40 picks that were liked, also for the first 20 and last 20 picks (does it improve while learning).
- **No leakage:** the base model was trained on Last.fm only. The Step 7 personal layer was fitted on these thumbs and is **not** used in the replay.

## Policies (fixed in advance; no tuning on the replay results)
1. random.
2. popularity-greedy: the most popular unplayed song in the pool.
3. base-only: highest frozen base-model logit; never learns.
4. old bandit: the existing `BanditAgent` (mainstream prior 0.5, epsilon-greedy), learning from thumbs.
5. base + online learner, no exploration bonus (beta = 0): offset = base logit; a penalised logistic correction (L2 1.0, changed to 0.05 by the addendum below) refitted after every thumb.
6. base + online learner with exploration (beta = 0.3): the same, scored as logit + 0.3 * uncertainty.
7. popularity prior + online learner (beta = 0.3): offset = (popularity - 50) / 25, same learner.

## Expectation stated in advance
Popularity-greedy is likely the strongest or tied (popularity predicted Adi's thumbs best offline, AUC 0.71); a learner starting from the base (AUC 0.67) is unlikely to overtake it within 40 songs.
The learner on a popularity prior (7) is the one most likely to match or beat popularity-greedy late in the session. The old bandit should be weakest.
Reported as it comes out; the policy actually shipped is chosen on this plus the live evidence, and a non-win is a non-win.

## Limits
Closed-world replay: the pools are songs Adi already rated, chosen earlier by several procedures (so the share liked is higher than on random songs and popularity-heavy), 300 sessions share one set of labels, and the thumbs are one person's quick judgements.

## Addendum (before the replay was run on any real data): learner penalty changed from 1.0 to 0.05
The policies above said "L2 1.0". While testing the learner on a **synthetic** world (the person likes jazz and folk, the prior favours rock), a penalty of 1.0 barely learned:
after 60 labels the liked-genre gap on unseen songs was -0.07 (it still ranked the prior's favourite above the person's), against +0.18 at 0.1, +0.35 at 0.05 and +0.73 at 0.01.
Step 7's own cross-validation had chosen penalties of 0.01 (13 of 25 folds) and 0.1 (6 of 25), so 1.0 was a poor choice made without consulting it. The penalty is now **0.05**, fixed for all learner policies (5, 6, 7).
No real-data replay result had been seen when this was written. Nothing else in this file changes.
