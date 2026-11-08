> Full version (about two pages) of the final write-up. The one-page summary is `step10_writeup.md`; this file keeps every row, the reasons for each failure and the transfer details.

# Learning a listener's music taste from their reactions: results against the marks set in advance, and what transfers to a new user

Built in a 24-hour hackathon. Evidence: Adi (190 song ratings, several hundred camera-session and live thumbs) and Cameron (120 live thumbs). Every threshold below was fixed before the data was collected (pre-registrations in `reports/`). Failures are listed as failures.

## Bottom line
- **No model of a person's taste beat "play popular songs" in a live test.** Adi: popular 62%, fine-tuned personal model 52%, random 2% (40 songs each). Cameron, who contributed no training data: popular 48%, the frozen base model 42%, random 25%.
- **The one clear win is over random, and over the old learner.** The Step 7 model lifted Adi's liked share from 30% (old profile) to 52%; in offline replay the new online learner improves inside a session (53% to 63%) and the old bandit stays flat (49%).
- **Facial emotion is not yet a trustworthy signal for changing songs.** The only strong camera result (reaction shape, AUC about 0.80) did not replicate (0.556). A new measurement built around movement and stillness looks promising inside one session (AUC 0.74) but has not passed its cross-session test.
- **Everything still works as a system:** 259 unit tests pass, models are frozen and hashed, and the deployable base model is 1,477 bytes (the whole stack, with the personal layer, 1,914 bytes; an ESP32 has about 520 KB of RAM).

## Results against the pre-set marks
| Check | Mark | Result | Verdict |
|---|---|---|---|
| Camera's like/dislike call matches the thumb (148 songs, quiet) | >= 75% | 59% (always saying "dislike" scores 65%) | **FAIL** |
| Camera reaction shape replicates on new songs (Step 3d, 57 songs) | AUC >= 0.70, +0.05 over old reading | 0.556 (range 0.387 to 0.720); gain -0.041. Earlier session 0.80 | **FAIL** |
| Nodding signal holds on new songs | AUC >= 0.70 | 0.56 | **FAIL** |
| Movement/stillness (v2) holds across sessions (Step 3e) | AUC >= 0.65 on every held-out session | session A alone: 0.74 inside the session; session B not run | **PENDING** |
| L1.1 base data matches our tracks | >= 30% | 4.9% | **FAIL** |
| L1.2 quality and preference ratings differ | >= 15% of songs | 1% | **FAIL** |
| L1.2 ratings repeat when rated again | >= 80% within 1 point | 95% quality, 100% preference (1.3 h later; liked songs not retested) | PASS, weak |
| L1.3 base beats popularity on other people's likes | gain >= +0.03 AUC | +0.054 on 126 held-out users (first 600-epoch run: +0.025; test users read twice) | PASS with caveat |
| L1.3 objective quality tracks quality more than preference | >= 0.30 and above preference | 0.149 vs 0.230 | **FAIL** |
| L2.1 zero data = base; base file unchanged; reset; wrong base refused | exact | all hold | PASS |
| L2.2 fine-tuned vs base-only; vs from-scratch; labels needed | +0.05; beat both; reach it with <= 80 labels | +0.021; beats bandit, not logistic (-0.008); 0.669 vs 0.702 at 80 labels | **FAIL** |
| L3.1 final held-out test, read once | models scored once | Step 4c: random 0.721, popularity 0.827, bandit 0.745 (4 liked songs); base-only and fine-tuned never scored on it | **INCOMPLETE** |
| L3.2 live, old ratings profile (Adi) | +15 over popular, +25 over random | 30% vs 72% vs 28% | **FAIL** |
| L3.2 live, Step 7 model (Adi) | same | 52% vs 62% vs 2%: -10 vs popular (range -32 to +12); +50 vs random but random was unusually low | **FAIL** |
| L3.3 new user (Cameron), base model | checklist: base beats popular on first 10 recs; the run used the L3.2 marks | 42% vs 48% vs 25%: -5 vs popular (range -27 to +17); +18 vs random (range -3 to +38). First-10 not computed | **FAIL** |

Note on L3.3: Cameron's report compares against Adi's first session (old profile). The like-for-like comparison is Adi's second session: the gap to popularity was -10 points there against about -5 for Cameron's base model. Both ranges include zero, so neither says popularity is actually better, only that the +15 target was missed.

## Why it failed (what the data supports)
- **Popularity predicts thumbs about as well as anything we built** (AUC 0.71 on 264 thumbs; fine-tuned 0.686, base-only 0.672, from-scratch logistic 0.694, old bandit 0.578). A personal model has to add something on top of a strong baseline using a few hundred labels.
- **The camera measurement was not stable between sessions.** Inside session 1 the reaction model scored 0.795; inside session 2 it scored 0.461, and the calibrated readings had shifted by about +0.4.
- **"How good" and "do I want it" were the same question for Adi** (ratings differ on 1% of songs), so the planned objective-versus-preference split had nothing to learn.
- **The base's catalog coverage was too thin** (4.9% of commonly played Last.fm tracks match our 80,834 songs) and its taste is rock-heavy 2005-2009 listening.

## What worked
- A reproducible pipeline with pre-registration, so the negative results are credible; models are frozen with hashes; raw camera samples are now saved for re-analysis.
- The base model beats popularity on strangers' likes (+0.054) and is tiny.
- The new online learner (`--policy prior-learner`) learns inside a session; the old bandit does not (replay, 300 sessions: popularity-greedy 68.5%, popularity prior + learner 62.7%, base + learner 58.1%, old bandit 49.4%, random 37.1%). Popularity-greedy is still best overall; the replay cannot settle it against the learner.
- Inside one session the new movement-and-stillness features were mostly independent of tempo, energy and popularity (|r| <= 0.20) and separated liked from disliked songs (activity AUC 0.76, stillness 0.77). Smile did not.

## Transferring this to another user who changes songs on a playlist from their facial emotion
**What carries over:** the frozen base model (trained on other people, never on Adi), the popularity-leaning prior, the online learner that learns from thumbs and phone overrides, and the tools and tests. **What does not:** Adi's personal layer (it is Adi's taste; do not ship it), the camera's weights and calibration (measured on one person and not replicated), and the thresholds tuned on Adi.
**What is not built or not validated for that use:**
1. **Facial emotion deciding what plays.** No camera reward has passed validation, so the new learner ignores the camera unless `camera_validated=True` (default off) and refuses `--live`. Treat emotion as a display and a weak hint for now; thumbs and phone changes are the dependable signal.
2. **Playlists.** The `playlist` source is not implemented (`get_candidates` handles `event` and `random` only), and the loop picks from a mood-filtered catalog sample. Every result here is on songs the person had not chosen; a playlist is songs they already like, so the share liked will be much higher, popularity means something different, and the job becomes ordering and skipping, not discovery. None of our numbers apply to that.
3. **Songs outside the 80,834-song catalog** get Gemini-estimated features without popularity, so the base model's popularity input falls back to its average.
4. **The loop never changes a song mid-clip:** every song plays its 30 seconds, then the emotion is context for the next pick.
5. **The new user's own result will differ.** Cameron's base-only result (42%) was close to popularity but not above it; a new person should be checked with 40 thumbs before trusting the recommendations.

## Limits
One main user and one second person; 40 songs per condition (a share is known only to about +/-15 points); thumbs given after a short clip; the final held-out set had 4 liked songs and is spent; the base model's test users were read twice (disclosed); the Last.fm data is from 2005-2009 and non-commercial; the camera was tested in one room by one person.

## Recommended before handing it over
1. Ship the base model with a popularity prior and the online learner (opt-in), camera reward off; keep the old bandit as the default until a live test of the learner against popularity-greedy is run.
2. Run the two quiet-room sessions for the movement-and-stillness features (`rl/camera_validation.py`; pass = AUC >= 0.65 on both) before emotion is allowed to change songs, then repeat the check on the new user's own face and room.
3. Implement and evaluate the playlist source on playlist-style data (songs the person already picked).
4. Say plainly in the demo that personalised taste learning did not beat popularity for the two people tested.
