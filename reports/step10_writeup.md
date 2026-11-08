# Learning a listener's music taste from their reactions: one-page summary

24-hour hackathon. Evidence: Adi (190 ratings, several hundred thumbs) and Cameron (120 live thumbs). All marks were fixed before the data was collected. Full table and reasons: `step10_writeup_full.md`.

**Bottom line.** No model of a person's taste beat "play popular songs" in a live test. The robot, learner, tools and tests work; personalisation and facial-emotion control are not yet shown to help.

| Check | Mark set in advance | Result | Verdict |
|---|---|---|---|
| Camera's like/dislike call matches the thumb | >= 75% | 59% (always "dislike" scores 65%) | FAIL |
| Camera reaction shape replicates on new songs | AUC >= 0.70 | 0.556 (earlier session: 0.80) | FAIL |
| Movement/stillness features hold across sessions | AUC >= 0.65 on every session | 0.74 inside session A; session B not run | PENDING |
| Base model beats popularity on strangers' likes | +0.03 AUC | +0.054 (first run +0.025; test users read twice) | PASS, caveat |
| Base data matches our catalog | >= 30% | 4.9% | FAIL |
| Quality and preference ratings differ | >= 15% of songs | 1% | FAIL |
| Fine-tuned model beats base-only / from-scratch | +0.05 / better | +0.021 / -0.008 | FAIL |
| Final held-out test (read once) | all models scored | only 3 scored; 4 liked songs; base-only, fine-tuned never scored | INCOMPLETE |
| Live, Adi: personal vs popular vs random | +15 / +25 points | 52% / 62% / 2% (-10; random unusually low) | FAIL |
| Live, new user Cameron: base vs popular vs random | +15 / +25 points | 42% / 48% / 25% (-5; +18) | FAIL |

Live results are 40 songs per condition (a share is known to about +/-15 points; the gaps to popularity include zero, so popularity is not proven better).

**Why it failed.** Popularity predicts thumbs about as well as anything we built (AUC 0.71; fine-tuned 0.686, base-only 0.672). The camera measurement drifted between sessions (inside session 1 AUC 0.795, inside session 2 0.461). For Adi, "how good" and "do I want it" were the same rating. The base data covers few of our songs.

**What worked.** The base beats popularity on other people's likes; the Step 7 model beat the old profile (52% vs 30%); the new online learner improves inside a session in replay (53% to 63%) while the old bandit stays flat (49%); models are frozen and hashed; 259 unit tests pass; the deployable base is 1,477 bytes (stack 1,914 bytes) against an ESP32's roughly 520 KB RAM.

**Handing it to another user on a playlist.** Carries over: the frozen base, popularity prior, online learner and tools. Does not: Adi's personal layer or the camera's weights. Not built or not validated: facial emotion deciding what plays (camera reward is off by default), the playlist source (not implemented; our results are on songs people had not chosen), songs outside the 80,834-song catalog (no popularity value), and mid-song changes (each clip plays fully).

**Limits.** Two people; the held-out set had 4 liked songs and is spent; thumbs follow short clips; Last.fm data is from 2005-2009; the camera was tested in one room.

**Before handover.** (1) Ship base + popularity prior + online learner with the camera off. (2) Run the second camera session and repeat it on the new user's face. (3) Build and test the playlist source. (4) Tell audiences plainly that personalised taste learning did not beat popularity for the two people tested.
