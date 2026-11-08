# Step 3d: camera reaction-shape replication on new songs

- Generated: 2026-11-08 03:39:40  
- Rule fixed beforehand: `step03d_preregistration.md`

## Unit tests

- Ran 225: 0 failed, 0 errors, 0 skipped.

## Result

```
New songs: 57 (16 liked, 41 disliked); trained on 70 earlier songs.
Reaction-shape model AUC 0.556  (95% range 0.387 to 0.720; mark >= 0.7)
Old reading (last 3 s) AUC 0.598;  gain -0.041  (95% range -0.141 to +0.047; mark >= +0.05)
peak_liking alone 0.535;  first half 0.567, second half 0.526
context: popularity AUC 0.907
context: fine-tuned model (Step 7) AUC 0.803
Overall: FAIL
```

Weights (standardised features; descriptive only): mean_liking +0.04, peak_liking +0.17, low_liking +0.02, range_liking +0.15, build_liking +0.14, last3_liking +0.09

## Limits

- One person, one room, a few dozen new songs: an AUC is known to about +/-0.1.
- The thumb is given right after a 10-second clip.
- Part of any camera AUC can come from the songs themselves (popular, upbeat songs get liked and may also get a stronger reaction); the context lines show how well popularity alone does on the same songs.


## Verdict
**NOT REPLICATED (FAIL against the pre-registered marks).** On 57 new songs (16 liked, 41 disliked) the reaction-shape model scored **AUC 0.556** (95% range 0.387 to 0.720; mark >= 0.70) and was **0.041 below** the old reading (0.598; mark: at least +0.05 above).
The first-session result (cross-validated AUC about 0.80) does not hold on new songs. With only 16 liked songs the range is wide and its top end touches 0.72, so a modest signal cannot be ruled out; the point estimate is near chance.

## What differed between the two sessions (analysis after seeing the result, not part of the mark)
| feature | session 1 mean (sd) | session 2 mean (sd) | AUC in session 1 | AUC in session 2 |
|---|---|---|---|---|
| mean_liking | -0.098 (0.185) | +0.312 (0.216) | 0.602 | 0.599 |
| peak_liking | 0.490 (0.356) | 0.699 (0.262) | 0.817 | 0.535 |
| low_liking | -0.632 (0.196) | -0.026 (0.150) | 0.535 | 0.604 |
| range_liking | 1.122 (0.375) | 0.725 (0.224) | 0.791 | 0.489 |
| build_liking | 0.084 (0.267) | 0.016 (0.226) | 0.804 | 0.564 |
| last3_liking (old reading) | -0.075 (0.239) | +0.325 (0.274) | 0.700 | 0.598 |
| smile | 0.054 (0.079) | 0.203 (0.214) | 0.771 | 0.544 |

- **The camera's calibrated readings shifted a lot between sessions.** The old rule called 98% of session 2 songs "liked" (37% in session 1), which is why the old 75% agreement gate scored only 26%. Mean liking moved by about +0.4 and the spread of peak and range shrank.
  The most likely causes are a different resting baseline at calibration and different lighting or camera position; the data cannot say which.
- **Re-scaling does not rescue it.** Standardising each session on its own and retraining gives AUC 0.552. The best session-1 features (peak, range, build, smile) fall to 0.49 to 0.56.
- **The session-1 signal was not a drift over time.** Its features hardly correlate with song position (corr -0.21 to +0.07) and their AUCs are unchanged after removing the position trend (0.68 to 0.80). So it was a real pattern in that session, but it did not transfer.
- **The songs themselves predict the thumbs better than the face does.** On these 57 songs popularity alone scores AUC 0.907 and the Step 7 fine-tuned model 0.803 (inflated: half the songs were chosen by that model, and the other half were random, so popular songs were far likelier to be liked). The camera adds nothing visible on top.

## Conclusion
The camera reaction-shape signal is **not a reliable training signal at present**. Most likely explanations, none proven: the first-session result was partly chance (70 songs, best features picked after looking), or it depends on a calibration and setup that was not reproduced.
The camera reward must stay off. If the camera is to be used at all, it would need a fixed physical setup, a calibration check before each session, and a larger session (150+ songs).
Thumbs remain the ground truth and popularity-leaning prior the robust default, as in Steps 7 and 8.
