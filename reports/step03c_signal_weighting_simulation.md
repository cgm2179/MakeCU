# Step 3c simulation: does weighting the camera by its reliability help the learner?

**SYNTHETIC.** Made-up listeners (like about 25% of songs by genre; thumbs right 92% of the time) and a made-up weak camera (old reading AUC 0.67, one extra feature AUC 0.60, ten features with no signal). It shows what the weighting does under these assumptions; it says nothing about how real people or Adi's camera behave.

- 8 random worlds; the bandit learns online for 120 songs per condition; scores are on 300 unseen songs. Mean ± standard deviation across worlds.
- 'camera as a thumb' is the old behaviour: the camera's call (reading above 0 = liked) counted as much as a thumb.

# Camera model fitted on 80 earlier songs with thumbs

Measured reliability: cross-validated AUC 0.523 ± 0.062, weight 0.078 ± 0.090 (0 = ignored, 1 = trusted like a thumb).

## Ranking of unseen songs (AUC, 0.5 = chance)

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.500 ± 0.000 | 0.570 ± 0.035 | 0.589 ± 0.028 |
| camera as a thumb | 0.545 ± 0.035 | 0.535 ± 0.058 | 0.589 ± 0.028 |
| camera weighted (ours) | 0.525 ± 0.031 | 0.575 ± 0.028 | 0.626 ± 0.073 |

## Share of its last 20 picks the listener really liked

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.237 ± 0.108 | 0.575 ± 0.071 | 0.887 ± 0.093 |
| camera as a thumb | 0.450 ± 0.166 | 0.569 ± 0.240 | 0.887 ± 0.093 |
| camera weighted (ours) | 0.244 ± 0.133 | 0.475 ± 0.179 | 0.856 ± 0.151 |

# Camera model fitted on 150 earlier songs with thumbs

Measured reliability: cross-validated AUC 0.570 ± 0.073, weight 0.154 ± 0.123 (0 = ignored, 1 = trusted like a thumb).

## Ranking of unseen songs (AUC, 0.5 = chance)

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.500 ± 0.000 | 0.570 ± 0.035 | 0.589 ± 0.028 |
| camera as a thumb | 0.545 ± 0.035 | 0.535 ± 0.058 | 0.589 ± 0.028 |
| camera weighted (ours) | 0.533 ± 0.056 | 0.573 ± 0.038 | 0.621 ± 0.093 |

## Share of its last 20 picks the listener really liked

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.237 ± 0.108 | 0.575 ± 0.071 | 0.887 ± 0.093 |
| camera as a thumb | 0.450 ± 0.166 | 0.569 ± 0.240 | 0.887 ± 0.093 |
| camera weighted (ours) | 0.269 ± 0.109 | 0.344 ± 0.169 | 0.750 ± 0.232 |

## History of this simulation

- Version 1 turned the camera into a reward of 2p - 1. For a listener who likes 25% of songs that is about -0.5 on nearly every song, a constant push down. On its worlds (seeds 0 to 7), with thumbs on 25% of songs, the weighted camera's last-20 picks were liked 0.31 of the time against 0.52 for 'thumbs only' (spread about ±0.2). Version 2 measured the camera relative to the person's usual rate but scaled the two sides differently, which left a negative average (seeds 100 to 107: weighted 0.41, thumbs only 0.53). Version 3 used one symmetric scale so the camera's reward averages 0, and is run on fresh worlds (seeds 200 to 207) so it is not judged on the data it was designed on. Version 3 was judged by leave-one-out scoring, which turned out to push weak-signal reliability toward 0 (a held-out liked song lowers the training base rate, so its own score falls). Version 4 (this report) uses stratified 5-fold cross-validation, found by the real busy-session data (an AUC of 0.000 for a signal-free reading) and confirmed by a unit test, and runs on fresh worlds (seeds 300 to 307). Four versions is already some tuning to a made-up world: treat the result as a check that nothing is broken, not as proof.

## How to read this

- 'camera weighted' scales the camera by its measured reliability, so a noisy camera cannot outvote a thumb; its rewards are small when the camera is weak.
- Differences smaller than the ± spread are not real. The listener, noise levels and camera strength are assumptions.
