# Step 3d pre-registration: replicate the camera reaction-shape result on new songs (committed before the session)

Date: 2026-11-08. Question: does the shape of the face's reaction over a clip predict Adi's thumb on songs the model has never seen?
Background: in the first quiet-room session (70 songs with features) the reaction-dynamics group scored cross-validated AUC 0.80 and beat the old reading
(last 3 s of liking, 0.69) by +0.096, but the range on the gain included zero, the best single features were picked after looking, and nodding (`bob_share`) failed to replicate (0.56).

## Fixed in advance
- **Training data (never changes after this):** the 70 quiet-session songs with features in `data/agreement/adi.csv`.
- **New data:** a fresh quiet-room session saved to `data/agreement/adi_rep.csv`: songs Adi has never rated, thumbed or heard in an earlier session; same 10-second clips, same camera, same calibration, `--setting quiet`, one person alone.
  Songs are ordered to alternate the fine-tuned model's top picks with random songs so that both thumbs occur.
- **Model:** logistic regression, L2 penalty 1.0 on standardised features, on the six reaction-dynamics features
  `mean_liking, peak_liking, low_liking, range_liking, build_liking, last3_liking` (the group defined in Step 3c, not a hand-picked subset). Standardisation and weights come from the training data only. No tuning on the new songs.
- **Baselines on the same new songs:** the old reading (`last3_liking` alone), the single feature `peak_liking`, and, for context, Spotify popularity and the Step 7 fine-tuned model.
- **Rows used:** songs with features and confidence >= 0.1.

## Pass marks (all must hold)
1. At least **40** usable new songs, with at least **12** of each thumb. Otherwise the result is "not enough data", not a pass or fail.
2. The dynamics model's AUC on the new songs is **>= 0.70**.
3. It beats the old reading (`last3_liking` alone) by AUC **>= +0.05**.

Reported but not part of the mark: 95% bootstrap range (resampling songs, 2,000 times) for the AUC and for the gain; AUC in the first and second half of the session; AUC of `peak_liking` alone.

## Reading the outcome
- PASS: the reaction-shape signal replicates; it is then reasonable to try the camera as a secondary reward in a live loop (suggested weight about 0.5), with thumbs still the ground truth.
- FAIL: the 0.80 was luck or specific to the first session; the camera is dropped as a learning signal.
- If the camera model only matches popularity's AUC on the same songs, the camera adds little beyond knowing which songs are popular; this is reported either way.
- Limits stated in advance: one person, one room, a few dozen songs (a share or AUC is known to roughly +/-0.1), thumbs given right after a 10-second clip.
