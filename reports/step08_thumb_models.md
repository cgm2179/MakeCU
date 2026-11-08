# Step 8: can anything beat "play popular songs" on Adi's thumbs?

Date: 2026-11-08. Tests: 180 ran, 0 failed. Follows the failed live test (reports/step07_live_session.md).

## Data
264 thumb labels (100 liked, 36%): 150 from the camera-agreement sessions (data/agreement/adi.csv) and 120 from the live session
(data/live/adi.csv); 6 songs appear in both and the live thumb is kept. Songs come from different selection rules (gym/party/smart
order, plus random / popular / profile-picked), so this is a mixed sample, not one clean random draw.

## Method
Stratified 5-fold cross-validation repeated 5 times with different splits; every song is scored by a model that never saw it;
AUC (0.5 = chance). Models: popularity only; the bandit trained on thumbs; a blend `w*z(popularity) + (1-w)*z(bandit)`
(`rl/blend.py`); and a plain logistic regression on audio numbers (valence, energy, danceability, tempo, acousticness,
instrumentalness, popularity), run once as an exploratory check.

## Results (AUC)
| model | AUC |
|---|---|
| **popularity only** | **0.712** |
| bandit trained on thumbs | 0.580 +/- 0.039 |
| blend w=0.2 / 0.4 / 0.5 | 0.623 / 0.665 / 0.682 |
| blend w=0.6 / 0.7 / 0.8 / 0.9 | 0.696 / 0.705 / 0.709 / 0.711 |
| logistic regression, audio only (no popularity) | 0.599 |
| logistic regression, audio + popularity | 0.689 to 0.698 depending on regularisation |

Correlation of the thumb with each song feature: popularity +0.36, energy +0.17, valence +0.12, danceability +0.10, tempo +0.01,
acousticness -0.22, instrumentalness -0.19.

## Reading
- Popularity is the best single predictor available (0.71), and every addition **lowers** the score. With 264 labels, the personal
  signal is too weak to add anything beyond it; the blend only approaches popularity as its weight on popularity goes to 1.
- The audio numbers carry a little signal on their own (0.60, mostly less acoustic/instrumental, more energetic), but nothing that
  popularity does not already cover.
- This does not show that Adi has no individual taste: it shows that the taste signal is not recoverable from 264 thumbs with these
  features, and that the earlier ratings-based profile did not transfer to thumbs.

## Caveats
- Weights were compared on the same 264 labels, so even the best blend weight is slightly optimistic; popularity-only has no tuning.
- The exploratory logistic regression had its regularisation looked at after the fact.
- One person. No fresh test session has been run on these models; none of them beat popularity here, so none is proposed for one.

## Recommendation
Keep the popularity-leaning default (`MAINSTREAM_BIAS` 0.5, fading prior) and let the bandit learn online from live thumbs and
the camera, which is where an individual signal could still appear within a session. In the write-up say plainly that offline
personalization did not beat popularity for this user.
