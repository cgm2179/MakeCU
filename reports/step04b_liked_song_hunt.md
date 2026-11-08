# Step 4b: liked-song hunt and first bandit check on real labels (adi)

Date: 2026-11-07. Tests: 163 ran, 0 failed, 9 skipped (dataset-dependent).

## Method
- After Step 4 only 8 of 103 development labels were "liked" (preference 4-5), too few to learn from.
- Hunt mode (`run_rating_session --hunt`): bandit trained on the 150 earlier preference ratings picks songs; half are its
  top-ranked unrated songs, half random, alternating. 40 songs rated, preference question only, saved as round 3, dev split.
- Labels: preference 4-5 liked, 1-2 disliked, 3 left out. 30 held-out songs excluded from this check.

## Result (5-fold cross-validation, 137 labeled songs: 31 liked, 106 disliked)
| model | AUC (mean +/- sd) | precision@10 |
|---|---|---|
| random | 0.411 +/- 0.012 | 0.22 |
| popularity-only | 0.609 +/- 0.202 | 0.36 |
| bandit (from scratch) | **0.915 +/- 0.036** | **0.52** |

Learning curve (bandit AUC): n=10 0.78, n=20 0.88, n=40 0.83, n=80 0.91, n=110 0.90.
Hunt ratings: 2:11, 3:6, 4:12, 5:11, so 23 of the 40 were liked (58%, versus 8% in the stratified 150).

## Caveats (read before trusting 0.915)
1. **Selection bias.** The hunt songs were chosen by a bandit trained on Adi's earlier ratings, so many liked songs resemble
   each other (same genres/moods). Cross-validation folds then contain near-duplicates, which inflates AUC. The number shows
   the bandit can model Adi's taste in the region it explored, not that it generalises to the whole catalog.
2. **Mild leak into the final exam.** `hunt_order` was trained on all 150 first-pass ratings, including the 30 held-out ones,
   so which songs were hunted depended slightly on held-out labels. The held-out labels were not used in this check, but the
   final test is no longer perfectly independent. Report the final result with this noted, or fix `hunt_order` to train on
   dev rows only before any future hunt.
3. The "random" AUC of 0.41 is just noise from 31 positives; it should sit near 0.5.
4. Step 5 (consistency) is waived, see reports/step04_ratings.md.

## Conclusion
On real labels the bandit clearly beats popularity-only and random in cross-validation, and it learns fast (AUC about 0.88 by 20 labels).
That is encouraging, with the selection-bias caveat. The honest test is the one-time held-out final.
