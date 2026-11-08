# Step 7 pre-registration (committed before any Step 7 model was fitted)

Date: 2026-11-08. Checkpoints L2.1 (user layer on a frozen base) and L2.2 (fine-tune on Adi). Base = `models/base_model_v2.json`
(hash `87b6175d...`, trained on Last.fm 1K). Marks are those of docs/THREE_LAYER_CHECKPOINTS.md.

## Two changes to the Step 7 prompt, agreed with the user
1. The final 30-song test set was already read once (Step 4c, lock file `data/eval/final_adi.json`), so it cannot be used again. Evaluation is repeated
   stratified 5-fold cross-validation on the training labels: every song is scored by a model that never saw it. The honest final test is a fresh live session later.
2. Labels are Adi's explicit **thumbs** (264 songs: camera-agreement sessions + live session, 100 liked), not the 4-5 ratings.

## Models compared (all trained only on the training part of each fold)
- **popularity-only** (Spotify popularity).
- **base-only**: the frozen base's preference head, no Adi data.
- **bandit (from scratch)**: the current online bandit (`BanditModel`).
- **logistic from scratch**: logistic regression on the same features as the base (+ genres with 5+ training songs), L2-penalised, no base.
- **fine-tuned**: logit = base logit + x . delta + b, with delta (one number per base feature, plus one per extra genre with 5+ training songs) starting at zero and
  L2-penalised toward zero (so it is pulled toward the base). The bias b is not penalised (it only recalibrates).
- Penalty strength for the two logistic models is chosen inside each training fold by inner 5-fold CV AUC from {0.001, 0.01, 0.1, 1, 10, 100}; ties go to the larger penalty.
  Fitting: Newton's method (IRLS), deterministic.

## Evaluation
- 5-fold stratified CV repeated 5 times (seeds 0-4). Reported AUC = pooled out-of-fold AUC per repeat, averaged over repeats.
- Confidence range for a difference: bootstrap over songs (2,000 resamples) of the AUC difference of out-of-fold scores averaged over repeats.
- Learning curve: train on the first n of each shuffled training fold, n in {10, 20, 40, 80, all}; penalty fixed at the one most often chosen in the main CV.

## L2.2 pass marks (all must hold)
a. fine-tuned beats base-only by AUC **>= +0.05**.
b. fine-tuned beats **both** from-scratch models (bandit and logistic) at full labels (point estimate; range reported).
c. fine-tuned at **n = 80 labels or fewer (at most half of the ~211 training labels)** reaches the from-scratch logistic model's AUC with all its labels.

## L2.1 pass marks
a. With zero user data (delta = 0, b = 0) the output is **identical** to the base (unit test, exact equality of logits).
b. The base file is **bit-for-bit unchanged** by fine-tuning (SHA-256 before and after; the file is read-only).
c. Resetting the personal layer returns exactly the base's output; a personal profile made on a different base (different hash) is refused to load.

## Layer 2 gate
P0.4 (profile save/load, done in Step 6) + L2.1 + L2.2 all pass.

## Device size
The personal layer is also float32 numbers; the total (base 1,477 bytes + personal layer) must stay under the 32 KB budget, hard cap 520 KB (ESP32 SRAM).

## Known risks stated in advance
264 labels from mixed selection rules; the base leans rock/indie while Adi's strongest likes were electro/soul; earlier steps found nothing beating popularity on these thumbs.
A fail is a real possibility and will be reported as a fail.
