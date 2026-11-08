# Step 3e pre-registration: do movement-and-smile (engagement) features work across sessions? (committed before any session with the new measurements)

Date: 2026-11-08. Why: the first camera result (reaction shape, AUC about 0.80) did not replicate (Step 3d: 0.556; within the second session 0.461).
The old measurement compared everything to one calibration taken at the start of a session, treated "liked" mainly as a positive expression, and measured movement without reference to the music.

## What changed in the measurement (implemented before this was written; no data with the new features exists yet)
- **Pre-roll baseline per song.** About 4 s of relaxed face before each song (the first second ignored) is that clip's baseline, so slow changes in light, distance or posture cancel.
- **Engagement features** (7): `activity` (head movement relative to the pre-roll), `stillness` (share of 1-second windows quieter than the person's relaxed pre-roll level),
  `beat_power` and `beat_amp` (head movement locked to the song's tempo or its half/double time, from the catalog's tempo), `lean` (face size change: leaning in), `smile_frac` and `smile_peak` (smile above the pre-roll level).
- **Raw samples are saved for every clip** (face positions and emotion probabilities, no images), so any feature can be recomputed later without a new session.

## Data to collect
Two or more quiet-room sessions with `--preroll 4`, each **on a different occasion** (different day or at least a re-seated, re-lit setup), 50+ usable songs each, at least 12 of each thumb.
Songs never rated or thumbed before; same 10-second clips; one person alone.

## Test (fixed)
- Model: logistic regression, L2 penalty 1.0, standardised features, on the 7 engagement features only. Standardisation and weights come from the training sessions only.
- **Leave-one-session-out:** train on all other sessions, test on the held-out one; repeat for every session.
- **Pass: AUC >= 0.65 on EVERY held-out session**, with each session having >= 40 usable songs and >= 12 of each thumb. Otherwise "not enough data".
- Reported but not part of the mark: within-session cross-validated AUC; the same test with the old reaction-shape features and with `last3_liking` alone (for contrast);
  popularity AUC on the same songs; 95% bootstrap ranges; which engagement features carry the signal.

## Reading the outcome
- PASS: movement/smile engagement is a measurement that transfers between sessions; it may then be tried as a weak secondary reward with a validation gate (reports/step03e_*), thumbs remaining the ground truth.
- FAIL on every held-out session or within-session AUC near 0.5 as well: the face is not giving a usable like/dislike signal with this camera and setup; drop it as a reward.
- Mixed (one session passes, one not): the measurement is not stable enough to trust; also reported as a fail.
- Limits stated in advance: one person, two or three sessions, an AUC is known to about +/-0.1 at this size; stillness can also mean concentration or attention elsewhere.
