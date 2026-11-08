# Step 6 pre-registration (written and committed before the Last.fm data was opened)

Date: 2026-11-08. Data approved by the user: Last.fm 1K (Celma), Zenodo record 6090214, non-commercial licence.
Nothing below may be changed after the results are seen. Pass marks are those of docs/THREE_LAYER_CHECKPOINTS.md.

## L1.1 Data decision and match rate
- Decision: objective signal = Last.fm "how many different listeners play this song" (log), fitted by a model from song features;
  preference signal = Last.fm users' repeat listening. Adi's quality rating is used only to validate the objective signal.
- Normalisation: `clean_title` / `clean_artist` from `music/catalog/names.py`; a Last.fm (artist, track) matches a catalog song when
  both cleaned strings are equal (also tried: the catalog's primary artist). If a key matches several catalog songs, the most popular wins.
- **Match rate = share of distinct (artist, track) pairs played by at least 5 different users that match a catalog song.
  Pass mark: >= 30%.** The share of listening events that land on a matched song is also reported but is not the pass mark.

## Preference data
- Per user, a matched song is **liked** if that user played it at least 3 times. A user is used only if they like at least 10 matched songs.
- The **known set K** = matched songs played by at least 5 different users. All train/test work uses K only (so era and "is on Last.fm"
  cannot be learned from absence).
- Users are split at random by id (seed 0) into train 70%, validation 15%, test 15%. The test users are read once, at the end.

## L1.3 pass marks (all must hold)
1. **Beats popularity on other people's likes.** For each test user, AUC of ranking their liked songs above their other songs in K
   (never-played or played fewer than 3 times); mean over test users. Model = logistic regression on song features (audio numbers,
   mood, genre, Spotify popularity), trained on train users only. Baseline = Spotify popularity alone. **Pass: model mean AUC >= baseline + 0.03.**
   Also reported, not part of the mark: 95% bootstrap range over test users, and the in-data listener-count ranking as a ceiling reference.
2. **Objective and preference are separable.** q(song) = ridge regression of log(number of distinct listeners) on the same features
   (no Adi data used). On Adi's 150 first-pass ratings: Spearman(q, Adi's quality) **>= 0.30 and higher than** Spearman(q, Adi's preference).
3. **Saved with a version id and a hash, and unchanged by later fine-tuning**: shown by a test.

## L1 gate
L1.1 + L1.2 + L1.3 all pass. L1.2 is already known: 150 rated / 30 held out PASS; re-rating consistency PASS (weak, 1.3 h gap);
quality and preference differ on >= 15% of songs **FAIL (1%)**. So the L1 gate cannot pass overall unless that result is reconsidered with the user.

## Training protocol
Weighted logistic regression over the songs in K (each song's weight = how many train users like it / do not like it), gradient descent,
checkpoints saved periodically, best checkpoint chosen by validation loss on validation users, test users read once.

## Addendum (user request, added before any Last.fm result was computed): fits in an ESP32's RAM
- The ESP32 has about 520 KB of SRAM in total (Wi-Fi/Bluetooth and the program use much of it). Requirement: the deployed base model must
  be smaller than that. Pre-set limits: **hard cap 520 KB; working budget 32 KB** (about 6% of RAM, leaving room for everything else).
- Counted as the in-memory size on the device: every number stored as a 4-byte float32, plus the genre-name table.
- Deliverables: `models/base_model_v2.bin` (compact binary) and `models/base_model_v2.h` (C arrays) built from the same numbers, with a test
  that decodes the binary and reproduces the host model's scores. Checkpoints and the JSON stay on the host computer, not the device.
