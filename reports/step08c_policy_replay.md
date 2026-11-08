# Policy replay: a learner that starts from the base model vs the old bandit vs popularity

Date: 2026-11-08. Tests: 259 ran, 0 failed. Pre-registered in `step08c_policy_replay_preregistration.md` (with one dated addendum changing the learner penalty from 1.0 to 0.05
before any real-data result; the 1.0 failed to learn on a synthetic test world).
Method: 300 simulated 40-song sessions, each from a random pool of 150 songs Adi already rated; a picked song reveals its logged thumb (444 thumbs, 163 liked).
No leakage: the base model was trained on Last.fm only; the Step 7 personal layer is not used.

## Result
| policy | liked share (mean +/- s.e.) | first 20 picks | last 20 picks |
|---|---|---|---|
| random | 37.1% +/- 0.4 | 36.7% | 37.5% |
| **popularity-greedy** | **68.5% +/- 0.4** | 74.5% | 62.4% |
| base-only (never learns) | 56.6% +/- 0.4 | 49.3% | 63.9% |
| old bandit | 49.4% +/- 0.5 | 49.4% | 49.5% |
| base + online learner (beta 0) | 58.1% +/- 0.6 | 53.1% | 63.1% |
| base + online learner (beta 0.3) | 56.9% +/- 0.6 | 52.3% | 61.5% |
| popularity prior + online learner (beta 0.3) | 62.7% +/- 0.5 | 61.8% | 63.7% |

## What it says
- **The new learner learns inside a session and the old bandit does not.** The old bandit is flat at 49.4% from the first picks to the last; the learner starting from the base goes from 53% to 63% between the first and second halves.
  Overall the new learners are 7 to 13 points above the old bandit.
- **Popularity-greedy is still the best overall (68.5%)**, as expected and written in advance. The best learner (popularity prior + learning) is 5.8 points lower overall, but equal in the last 20 picks (63.7% vs 62.4%).
- **Do not read that second-half tie as a win.** Popularity-greedy takes the 40 most popular songs of a 150-song pool, so its supply of liked hits runs out; its drop from 74.5% to 62.4% is largely this closed-world artefact, which a catalog of thousands of songs would not have (live, popularity held up at 62 to 72%).
- **The exploration bonus did not help** (56.9% vs 58.1%, about two standard errors, so a small harm at most).
- **Offline replay cannot settle popularity vs the learner;** only a live comparison can.

## What was built (all behind the old default; nothing switched on)
- `rl/agent/prior_learner.py`: online learner = prior (base logit, popularity, or both) + a penalised logistic correction refitted after each feedback, with an optional uncertainty bonus. Feedback goes through `combine()`: thumbs (1.0) and behaviour (0.7) always count; the **camera counts only if `camera_validated=True`** (default False).
- `rl/agent/prior_agent.py` + `run_loop --policy prior-learner`: plugs it into the existing loop; it refuses `--live` (the camera reward is not validated) and `--user` (no profile saving yet). The default stays the old bandit.
- `rl/replay.py`: this harness. `fit_logistic` now accepts sample weights (unit weights give identical results; Step 7 numbers are unchanged).

## Limits
Closed-world replay on songs Adi already rated (chosen earlier by several procedures; popularity-heavy), one person, 444 labels, 300 sessions sharing one label set, thumbs are quick judgements.
The learner penalty (0.05) was set from a synthetic test and Step 7's cross-validated penalties, not tuned on these results.

## Recommendation
Keep popularity-led picking as the safe default; adopt "popularity prior + online learner" as the candidate adaptive policy and test it live against popularity-greedy and the old bandit in a blinded session (3 conditions, sequential within each) before it replaces anything.
