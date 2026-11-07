# Three-Layer Model: Checkpoints

Goal: a **base model** that knows the difference between *objective* opinion ("this is a good song")
and *preferential* opinion ("I like this song"), a **personal layer** fine-tuned on one user (Adi) on
top of it, and a **test** showing the personalised model beats simpler options for a real user.

Status key: **[DONE]** built and tested · **[PARTIAL]** exists but a gate is unmet · **[TODO]** not started.
Thresholds marked *(proposed)* are starting points: agree on them **before** running the test, not after.
Effort is a rough guess: S under 1 h, M 1-3 h, L 3-6 h.

## At a glance

| # | Checkpoint | Layer | Status | Gate (short) |
|---|---|---|---|---|
| **P0** | Foundations | all | mostly done | see below |
| P0.1 | Song features (dataset + Gemini) | - | [DONE] | 77 tests pass |
| P0.2 | Online learner (contextual bandit) | 2 | [DONE] | learns a fake taste in tests |
| P0.3 | Reaction signal from a person (camera, calibrated) | all | [PARTIAL] | agreement with explicit labels **unmeasured** |
| P0.4 | Save / load a user profile | 2 | [TODO] | restart keeps predictions identical |
| P0.5 | Offline evaluation harness | 3 | [TODO] | one command prints metrics on labeled data |
| **L1** | **Base model** | 1 | [TODO] | beats popularity on other people's likes; separates objective from preference |
| **L2** | **Personal fine-tune** | 2 | [TODO] | beats base-only and from-scratch on Adi's held-out songs |
| **L3** | **Test on a user** | 3 | [TODO] | live session, 3 conditions, personal model wins |

**Honest summary today:** the plumbing (P0.1, P0.2, most of P0.3) exists. None of L1, L2, L3 exists.
The bandit starts from zero every run, saves nothing, and nothing separates "good" from "liked".

**Terminology note.** Training the base is mostly *offline learning from logged feedback* (supervised on
like/dislike records). The *reinforcement-learning* part is the online stage: the bandit chooses songs
and learns from the reward. That is a legitimate pipeline, but call the base "pretrained from logged
feedback" so nobody expects a full RL environment.

---

## P0 - Foundations

### P0.1 Song features [DONE]
- 80,834 songs with valence/energy/danceability/tempo/popularity; Gemini lookup for songs outside it.
- **Pass (already green):** `python3 -m unittest music.tests.test_dataset_pipeline music.tests.test_gemini_lookup`

### P0.2 Online learner [DONE]
- `rl/agent/bandit.py`: weights over genre, artist, mood and audio buckets at three levels (global, per filter, per filter+emotion).
- **Pass (already green):** `python3 -m unittest rl.tests.test_bandit`
- Limitation to remove in L2: starts from zero every run.

### P0.3 Reaction signal [PARTIAL]
- Built: webcam + DeepFace, resting-baseline calibration with a guard, windowed read, skip when nobody in view.
- **Missing gate: do the camera scores agree with what Adi actually thinks?** We only saw that smiling songs
  scored higher than a neutral one in a 4-song run. That is not agreement.
- **Pass:** Adi gives an explicit thumbs up/down on **>= 40 songs** while the camera runs. The camera's sign
  (calibrated liking > 0) matches the thumb on **>= 75%** *(proposed)*; report the confusion table.
- **If it fails:** use explicit thumbs as the label and the camera only as a secondary signal. Do not train on a signal that disagrees with Adi.
- Effort: M (needs a small tool that records thumbs next to the camera reading). Owner: perception + integration.

### P0.4 Save / load a user profile [TODO]
- Save the personal weights to `data/profiles/<user>.json` with a version and the id of the base model it sits on.
- **Pass:** train on N interactions, save, restart the process, load: predictions for 20 fixed songs are **identical**; loading a profile made against a different base model is refused with a clear error.
- Effort: S.

### P0.5 Offline evaluation harness [TODO]
- Input: a table of `(user, song, label)`. Output: AUC for like/dislike, precision@10 (of the 10 songs the model ranks highest from a held-out pool, share the user liked), and a learning curve (metric vs number of labels used).
- Use **5-fold cross-validation** over a small user's labels, plus a **final test split that is touched once**.
- **Pass:** one command (`python3 -m rl.eval ...`) prints the metrics for three baselines: random, popularity-only, and the current from-scratch bandit. A unit test checks the metrics on a toy case with known answers.
- Effort: M. **Build this before L1**, so every later step has a number.

---

## L1 - Base model (general knowledge + Adi's validation)

What it is: a model with two parts, **objective quality** `q(song)` (what people in general rate highly) and
**population preference** (how features relate to liking across many people). The *personal* part is added in L2.

### L1.1 Decide the data and define "objective" [TODO, needs a decision]
- **Objective signal.** Options: (a) dataset popularity (available now, but it measures how much a song is *played*, not how good it is); (b) Adi's own "quality" rating; (c) both. *Recommended: (a) as the large-scale proxy, (b) to validate it.*
- **Preference signal from many people.** The Spotify dataset has **no** per-person likes. Candidates to **verify availability and licence before relying on them**: Last.fm user listening histories (e.g. the Celma 360K/1K sets), the Million Song Dataset Taste Profile, LFM-2b. Joining them to our 80k tracks needs title+artist matching (we have `music/catalog/names.py`).
- **Pass:** a written decision (data source, how "objective" is defined, match rate to our tracks **>= 30%** *(proposed)* so the join isn't mostly empty). If no suitable external data exists, fall back to: popularity as the objective proxy + Adi's labels as the only preference data, and say plainly that the "general preference" part is then thin.
- Effort: M (mostly reading and a download). Owner: needs Adi's call.

### L1.2 Adi's validation labels [TODO]
- Build a small labeling tool. For each song (30 s iTunes preview), Adi answers **two** questions:
  1. **Quality:** "ignoring my taste, how well-made / widely appreciated is this?" (1-5)
  2. **Preference:** "how much do I personally want to hear this?" (1-5)
- Sample **~150 songs** spread across moods, genres and popularity (not only hits), so the two ratings can disagree.
- **Pass:**
  - >= 150 songs labeled; 30 held out as the **final test set, never used for tuning**.
  - **Test-retest:** re-rate 20 songs a day later; **>= 80%** within 1 point *(proposed)*. If lower, the labels are too noisy to evaluate against.
  - **The two ratings really differ:** they disagree by >= 2 points on **>= 15%** of songs *(proposed)*. If not, the objective-vs-preference idea has nothing to learn from for Adi.
- Effort: M for the tool + about 30-40 min of Adi's time. Owner: integration + Adi.

### L1.3 Train the base [TODO]
- Fit `q(song)` to the objective signal; fit population feature weights to the preference signal (L1.1 data); keep both **frozen** afterwards.
- **Pass (the base "knows the difference"):**
  1. **Beats popularity** at predicting *other people's* likes on held-out users: AUC at least **+0.03** over popularity-only *(proposed)*.
  2. **Objective and preference are separable:** on Adi's labels, correlation of `q` with Adi's *quality* rating is **higher** than its correlation with Adi's *preference* rating, and `q` vs quality is **>= 0.3** *(proposed)*. This is the direct test of "objective vs preferential".
  3. The base is saved with a version id and a hash, and a test proves it is unchanged by any later fine-tuning.
- Effort: L.

**Layer 1 gate:** all of L1.1-L1.3 pass.

---

## L2 - Personal fine-tune

### L2.1 User layer on top of the frozen base [TODO]
- The user's weights start at zero and are **regularised toward the base's prediction**. With no user data the model equals the base.
- Replace the bandit's "start from zero" with "start from the base"; keep the popularity prior only as a cold-start fallback if L1 fails.
- **Pass:** (a) zero user data -> output identical to the base (unit test); (b) the base's weights are bit-for-bit unchanged after fine-tuning (hash check); (c) resetting the profile returns to the base.
- Effort: M.

### L2.2 Fine-tune on Adi [TODO]
- Train the user layer on Adi's training labels (explicit thumbs, plus camera reactions only if P0.3 passed).
- **Pass, on Adi's held-out songs (5-fold CV, then the final test split once):**
  - fine-tuned beats **base-only** by AUC **>= +0.05** *(proposed)*;
  - fine-tuned beats a **from-scratch bandit given the same number of labels** (this is what shows the base helps);
  - **learning curve:** it reaches the from-scratch model's final quality with **half the labels or fewer** *(proposed)*.
- If it fails: check whether the base pulls Adi toward the crowd (too strong a prior) before changing anything else.
- Effort: M-L.

**Layer 2 gate:** P0.4 + L2.1 + L2.2 pass.

---

## L3 - Test on a user

### L3.1 Offline final test [TODO]
- Run the final 30-song test split **once**; report AUC and precision@10 for random / popularity-only / base-only / from-scratch / fine-tuned. No tuning afterwards.

### L3.2 Live session [TODO]
- Adi plays **>= 40 songs per condition** (two or three sessions, order randomised, Adi not told which condition is running):
  A. random within the mood · B. popularity-only (today's behaviour) · C. personal model.
- Ground truth = Adi's **explicit thumb** after each clip; the camera reading is logged alongside, not used as the label.
- **Pass:** C's liked-share is **>= B + 15 points** and **>= A + 25 points** *(proposed)*.
- **Caution on the numbers:** at 40 songs per condition the margin of error is about 15 points, so a 15-point gap is only suggestive. Say so in the write-up, or run more songs.
- Effort: M, plus Adi's time.

### L3.3 New-user check [TODO]
- A second person (a teammate) rates ~40 songs. **Pass:** the *base alone* beats popularity-only on their first 10 recommended songs. This is what shows the base model helps someone it has never seen.

### L3.4 Write-up [TODO]
- One page: the numbers from L3.1-L3.3, the thresholds that were set in advance, what failed, and the limits (one main user, a few hundred labels, the camera is noisy, Gemini estimates are rough).

**Layer 3 gate:** L3.1-L3.4 done, with the thresholds fixed beforehand.

---

## Order of work and shortest path

```
P0.5 eval harness --> P0.4 save/load --> P0.3 camera-vs-thumbs check
        \                                        |
         L1.1 decide data --> L1.2 Adi labels --> L1.3 train base --> L2.1 --> L2.2 --> L3.1 --> L3.2 --> L3.3 --> L3.4
```

**If time is very short (a hackathon demo), the minimum honest version is:**
1. P0.5 + P0.4 (so there is a number and a saved profile).
2. L1.2 labels from Adi (the two ratings) - this alone gives real validation data.
3. Base = popularity as `q` + Adi's two ratings to show the gap (skip external data; say so).
4. L2.1-L2.2 on Adi's labels, L3.2 live session with 2 conditions (B vs C).
Skipping the external dataset means the "general preference" half is unproven. Say that in the demo.

## Decisions needed before building
1. **Data:** external per-user preference data (and which), or the popularity-only fallback?
2. **"Objective":** popularity, Adi's quality rating, or both?
3. **Who labels:** can Adi rate ~150 songs on two questions (about 40 minutes)?
4. **Thresholds:** are the *(proposed)* numbers acceptable, or should any be tightened?

## Risks
- **One main user.** Results show the pipeline works for Adi; they do not prove it works for everyone (L3.3 helps a little).
- **Label noise.** Human ratings and the camera are both noisy; test-retest (L1.2) and the camera check (P0.3) exist to measure that.
- **Offline wins can fail live.** Hence L3.2 as a separate gate.
- **External data may not exist in a usable form** (coverage, licence). L1.1 is a decision point for that reason.
- **Gemini features are rough and vary between calls**; songs looked up that way are weaker evidence than dataset songs.
