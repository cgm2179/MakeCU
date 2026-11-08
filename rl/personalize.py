"""Layer 2: fine-tune a personal profile on one person's own ratings (checkpoints L2.2 and P0.4).

    python3 -m rl.personalize --user adi                 # train on development ratings, save data/profiles/adi.json
    python3 -m rl.personalize --user adi --include-heldout   # also use the 30 held-out songs (after the final test)

Base model: none yet (BASE_NONE). Layer 1 fell back to "popularity as the objective proxy + this person's labels as the
only preference data" because the quality and preference ratings did not separate (reports/step04_ratings.md). So the
base is just the mainstream prior inside the agent, and this step learns everything personal from the ratings.
Labels: preference 4-5 = liked (+1), 1-2 = disliked (-1), 3 left out; hunt ratings (round 3) are included.
"""
from __future__ import annotations

import argparse

from rl import ratings as rt
from rl.agent.profile import BASE_NONE, profile_path, save_profile
from rl.eval import BanditModel


def labels_from_ratings(rows, catalog_by_id, include_heldout=False):
    """[(Song, 0/1)] from first-pass and hunt ratings. Held-out songs only when include_heldout."""
    out = []
    for r in rt.first_pass(rows) + rt.hunt_rows(rows):
        if r["split"] == "test" and not include_heldout:
            continue
        song = catalog_by_id.get(r["song_id"])
        if song is None:
            continue
        if r["preference"] >= rt.LIKE_AT:
            out.append((song, 1))
        elif r["preference"] <= rt.DISLIKE_AT:
            out.append((song, 0))
    return out


def train_profile(user, rows, catalog_by_id, include_heldout=False, seed=0, directory=None, base_model_id=BASE_NONE):
    """Train a fresh bandit on the labels and save it as the user's profile. Returns (agent, n_labels, path)."""
    items = labels_from_ratings(rows, catalog_by_id, include_heldout)
    if not any(y for _, y in items) or not any(not y for _, y in items):
        raise SystemExit("Need at least one liked and one disliked song to train a profile.")
    model = BanditModel(seed)
    model.fit(items)
    path = save_profile(model.agent, user, base_model_id, directory)
    return model.agent, len(items), path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from music.catalog.spotify_catalog import load_spotify
    rows = rt.load_rows(rt.ratings_path(a.user), a.user)
    if not rows:
        raise SystemExit(f"No ratings for '{a.user}'. Run the rating session first.")
    by_id = {s.song_id: s for s in load_spotify()}
    agent, n, path = train_profile(a.user, rows, by_id, a.include_heldout, a.seed)
    liked = sum(1 for _, y in labels_from_ratings(rows, by_id, a.include_heldout) if y)
    print(f"Trained on {n} labels ({liked} liked) and saved {path}")
    best, worst = agent.top_features("global", 5)
    print("Likes:   " + ", ".join(f"{f} ({w:+.2f})" for f, w in best))
    print("Dislikes: " + ", ".join(f"{f} ({w:+.2f})" for f, w in worst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
