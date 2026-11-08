# Step 9: new-user check (L3.3)

Date: 2026-11-08. Subject: Cameron, a second person who contributed none of the base model's training
data (frozen `base_model_v2.json`, trained only on Last.fm listeners; see `step06_base_model.md`).
Tool: `integration.orchestrator.run_new_user_check` (built in PR #16, not run until now). It reuses the
L3.2 live-session gate (`step08_preregistration.md`): personal >= popularity + 15 points and
personal >= random + 25 points, 40+ songs per condition, "personal" here being the frozen base model's
preference head instead of a trained profile, since Cameron has no ratings.

    python3 -m integration.orchestrator.run_new_user_check --user cameron

## Result (45 songs so far, 15 per condition; gate wants 40+ per condition)
| condition | liked | share | 95% interval |
|---|---|---|---|
| random | 3/15 | 20% | 7% to 45% |
| popularity | 7/15 | 47% | 25% to 70% |
| personal (base model) | 4/15 | 27% | 11% to 52% |

personal minus popularity: -20 points (95% range -54 to +14; gate needs +15)
personal minus random: +7 points (95% range -24 to +37; gate needs +25)

Tool's own verdict: **IN PROGRESS** (needs 40+ songs per condition before the gate can be read).

## Reading this honestly
- At this sample size, even the optimistic end of the confidence interval for personal vs. popularity
  (+14 points) falls just short of the +15 gate, so there is no scenario in this batch where the base
  model is ahead of popularity.
- Personal vs. random is genuinely undecided: the point estimate (+7) is well under the +25 gate, but
  the interval (-24 to +37) is wide enough that more songs could move it either way.
- This matches the pattern already seen on Adi (`step06b_base_model.md`, `step07_personal_layer.md`,
  `step09_writeup.md`): popularity is hard to beat, and the base model so far sits between popularity
  and random, closer to random.
- Not a pass or a fail: 15 of the needed 40+ songs per condition is a first look, not the pre-registered
  read. No part of the gate or the data collected is being changed because of this result.

## Status
L3.3: **IN PROGRESS**, not pass or fail. Resume with the same command (progress is saved, keyed by
`--user cameron`) to add roughly 25 more songs per condition before reading the gate for real.
