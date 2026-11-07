# Step 3c simulation: does weighting the camera by its reliability help the learner?

**SYNTHETIC.** Made-up listeners (like about 25% of songs by genre; thumbs right 92% of the time) and a made-up weak camera (old reading AUC 0.67, one extra feature AUC 0.60, ten features with no signal). It shows what the weighting does under these assumptions; it says nothing about how real people or Adi's camera behave.

- 8 random worlds; the bandit learns online for 120 songs per condition; scores are on 300 unseen songs. Mean ± standard deviation across worlds.
- 'camera as a thumb' is the old behaviour: the camera's call (reading above 0 = liked) counted as much as a thumb.

# Camera model fitted on 80 earlier songs with thumbs

Measured reliability: leave-one-out AUC 0.477 ± 0.078, weight 0.053 ± 0.093 (0 = ignored, 1 = trusted like a thumb).

## Ranking of unseen songs (AUC, 0.5 = chance)

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.500 ± 0.000 | 0.591 ± 0.041 | 0.552 ± 0.052 |
| camera as a thumb | 0.521 ± 0.045 | 0.570 ± 0.047 | 0.552 ± 0.052 |
| camera weighted (ours) | 0.507 ± 0.014 | 0.580 ± 0.050 | 0.612 ± 0.087 |

## Share of its last 20 picks the listener really liked

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.250 ± 0.087 | 0.487 ± 0.162 | 0.919 ± 0.066 |
| camera as a thumb | 0.438 ± 0.141 | 0.431 ± 0.169 | 0.919 ± 0.066 |
| camera weighted (ours) | 0.256 ± 0.085 | 0.412 ± 0.176 | 0.800 ± 0.200 |

# Camera model fitted on 150 earlier songs with thumbs

Measured reliability: leave-one-out AUC 0.562 ± 0.044, weight 0.125 ± 0.088 (0 = ignored, 1 = trusted like a thumb).

## Ranking of unseen songs (AUC, 0.5 = chance)

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.500 ± 0.000 | 0.591 ± 0.041 | 0.552 ± 0.052 |
| camera as a thumb | 0.521 ± 0.045 | 0.570 ± 0.047 | 0.552 ± 0.052 |
| camera weighted (ours) | 0.543 ± 0.034 | 0.563 ± 0.073 | 0.573 ± 0.064 |

## Share of its last 20 picks the listener really liked

| how signals are used | thumbs on 0% of songs | thumbs on 25% of songs | thumbs on 100% of songs |
|---|---|---|---|
| thumbs only | 0.250 ± 0.087 | 0.487 ± 0.162 | 0.919 ± 0.066 |
| camera as a thumb | 0.438 ± 0.141 | 0.431 ± 0.169 | 0.919 ± 0.066 |
| camera weighted (ours) | 0.206 ± 0.098 | 0.319 ± 0.134 | 0.925 ± 0.066 |

## History of this simulation

- Version 1 turned the camera into a reward of 2p - 1. For a listener who likes 25% of songs that is about -0.5 on nearly every song, a constant push down. On its worlds (seeds 0 to 7), with thumbs on 25% of songs, the weighted camera's last-20 picks were liked 0.31 of the time against 0.52 for 'thumbs only' (spread about ±0.2). Version 2 measured the camera relative to the person's usual rate but scaled the two sides differently, which left a negative average (seeds 100 to 107: weighted 0.41, thumbs only 0.53). Version 3 (this report) uses one symmetric scale so the camera's reward averages 0, and is run on fresh worlds (seeds 200 to 207) so it is not judged on the data it was designed on. Three versions is already some tuning to a made-up world: treat the result as a check that nothing is broken, not as proof.

## How to read this

- 'camera weighted' scales the camera by its measured reliability, so a noisy camera cannot outvote a thumb; its rewards are small when the camera is weak.
- Differences smaller than the ± spread are not real. The listener, noise levels and camera strength are assumptions.
