# DJ Bot: results in one page

DJ Bot is a robot speaker that watches listeners' reactions, moves its arms with the emotion it sees, and learns which songs to play next. Built in 24 hours. Evidence: Adi (190 song ratings, several hundred thumbs) and Cameron (120 live thumbs). Full table and reasons: `step10_writeup_full.md`.

**Bottom line.** DJ Bot works end to end, and its starting model is useful and tiny. In live tests with two listeners, a plain popularity list matched or beat our personalised picks, and the facial-emotion signal is not yet reliable enough to control playback. We report both sides below.

## What works
- **A complete system:** camera, emotion reading, learning, music and moving arms run together in the live demo, with music and motion as independent processes.
- **A useful start for new listeners.** A base model trained on 992 Last.fm listeners (19 million plays) predicted held-out strangers' likes better than a popularity ranking: +0.054 AUC on 126 users (pre-set mark +0.03; first run +0.025; test users read twice).
- **Learning within a session.** In replay on logged thumbs the new online learner rises from 53% liked in the first half to 63% in the second; the original bandit stays flat at 49%. In live testing, Adi's liked share rose from 30% (earlier profile) to 52% (Step 7 model).
- **Tiny and rigorous:** the base model is 1,477 bytes (1,914 with a personal layer) against an ESP32's roughly 520 KB of RAM; pass marks were fixed before the data; models are frozen and hashed; 259 unit tests pass.

## How it did against the marks set in advance
| Check | Mark | Result | Outcome |
|---|---|---|---|
| Base model beats popularity on strangers' likes | +0.03 AUC | +0.054 (first run +0.025; test users read twice) | Met, with caveat |
| Quality/preference ratings repeat when rated again | >= 80% within 1 point | 95% / 100% (re-rated 1.3 h later) | Met, weak evidence |
| Frozen base, zero-data layer, reset, wrong-base refusal | exact | all hold | Met |
| Live, Adi: personal vs popular vs random | +15 / +25 points | 52% / 62% / 2% | Not met |
| Live, new user Cameron: base vs popular vs random | +15 / +25 points | 42% / 48% / 25% | Not met |
| Camera's like/dislike call matches the thumb | >= 75% | 59% | Not met |
| Camera reaction shape replicates on new songs | AUC >= 0.70 | 0.556 (earlier session 0.80) | Not met |
| Movement/stillness features hold across sessions | AUC >= 0.65 on every session | 0.74 inside session A; session B not run | Pending |
| Fine-tuned model beats base-only / from-scratch | +0.05 / better | +0.021 / -0.008 | Not met |
| Base data matches our catalog | >= 30% | 4.9% | Not met |
| Quality and preference ratings differ | >= 15% of songs | 1% | Not met |
| Final held-out test (read once) | all models scored | 3 scored on 4 liked songs; base-only, fine-tuned never scored | Incomplete |

Live results are 40 songs per condition (a share is known to about +/-15 points); the gaps to popularity include zero, so popularity is not proven better, only that our target was not reached.

## What the misses taught us
- **Popularity is a strong baseline** (AUC 0.71 on 264 thumbs; fine-tuned 0.686, base-only 0.672), so a learner must add something on top of it from a few hundred labels.
- **The camera measurement drifted between sessions** (inside session 1 AUC 0.795, inside session 2 0.461). That led to the per-song baseline and movement-and-beat features, built and awaiting their cross-session test.
- **For Adi, "how good" and "do I want it" were the same rating,** and the base data covers few of our songs.

## Handing it to another user on a playlist
**Carries over:** the frozen base, popularity prior, online learner and tools. **Does not:** Adi's personal layer or the camera's weights. **Not yet built or validated:** facial emotion deciding what plays (camera reward off by default), the playlist source, songs outside the 80,834-song catalog, and mid-song changes.

## Limits
Two people; the held-out set had 4 liked songs and is spent; thumbs follow short clips; Last.fm data is from 2005-2009; the camera was tested in one room.

## Next
(1) Ship the base with a popularity prior and the online learner, camera off. (2) Run the second camera session and repeat it on the new user's face. (3) Build and test the playlist source.
