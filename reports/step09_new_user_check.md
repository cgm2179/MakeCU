# Step 9: new-user check (L3.3)

Date: 2026-11-08. Subject: Cameron, a second person who contributed none of the base model's training
data (frozen `base_model_v2.json`, trained only on Last.fm listeners; see `step06_base_model.md`).
Tool: `integration.orchestrator.run_new_user_check` (built in PR #16, not run until now). It reuses the
L3.2 live-session gate (`step08_preregistration.md`): personal >= popularity + 15 points and
personal >= random + 25 points, 40+ songs per condition, "personal" here being the frozen base model's
preference head instead of a trained profile, since Cameron has no ratings.

    python3 -m integration.orchestrator.run_new_user_check --user cameron --per-condition 40

## Result (120 songs, 40 per condition — the full pre-registered size)
| condition | liked | share | 95% interval |
|---|---|---|---|
| random | 10/40 | 25% | 14% to 40% |
| popularity | 19/40 | 48% | 33% to 63% |
| personal (base model) | 17/40 | 42% | 29% to 58% |

personal minus popularity: -5 points (95% range -27 to +17; gate needs +15)
personal minus random: +18 points (95% range -3 to +38; gate needs +25)

Tool's own verdict: **FAIL**.

## Reading this honestly
- The base model does not beat popularity: it is 5 points behind on the point estimate, and even the
  optimistic end of the interval (+17) falls short of the +15 gate only by a hair in the other
  direction — this is a near-miss, not a clean rejection, but the gate was fixed before the session and
  it is not met.
- The base model clearly beats random on point estimate (+18 of the +25 needed), and the interval's
  lower bound (-3) just barely includes "no better than random," so a larger sample could plausibly
  still fail this half too. It does not reach the pre-registered +25 bar.
- Compared to the pattern on Adi (`step09_writeup.md`: personal 30%, popularity 72%, random 28% — a
  47-point gap to popularity), the base model does noticeably better here (42% vs. 48%, a 6-point gap)
  and further above random (42% vs. 25% vs. Adi's 30% vs. 28%, which was statistically indistinguishable
  from random). On a person it has never seen, the frozen base's preference head tracks taste better
  than it tracked Adi's fine-tuned personal layer did against popularity — but it still does not clear
  either pre-registered bar.
- 120 songs (40/condition) matches the design in `step08_preregistration.md`; at this size a share is
  known to about +/-15 points, consistent with the width of the intervals above.

## Status
L3.3: **FAIL** (base alone does not beat popularity, and does not clear the random gate either), on the
full pre-registered 40 songs/condition. The base model is directionally better than chance and closer to
popularity on a brand-new person than it was for Adi's fine-tuned model, which is evidence the base
generalizes somewhat, but it does not pass the bar set in advance.
