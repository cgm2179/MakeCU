"""Offline evaluation harness (checkpoint P0.5).

Input : a table of (user, song_id, label). label 1 = liked, 0 = disliked.
Output: AUC, precision@10 and a learning curve for each model, by 5-fold cross-validation,
        plus a final held-out test that can be run only ONCE per user.

Run from the repo root:
    python3 -m rl.eval --labels data/ratings/adi.csv --user adi        # cross-validation
    python3 -m rl.eval --labels data/ratings/adi.csv --user adi --final  # final test, once
    python3 -m rl.eval --demo                                            # synthetic labels, to see it work

CSV columns: user, song_id, label   (optional: split = "test" marks the final held-out rows).
Models compared: random, popularity-only, and the current from-scratch bandit. Later models
(base-only, fine-tuned) plug in by implementing fit(train) and score(song).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from shared.schemas.messages import GroupReaction, PersonReaction, Song

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
FINAL_SIZE = 30
LEARNING_SIZES = (10, 20, 40, 80, 160)


# ---------------------------------------------------------------- metrics

def auc(scores, labels):
    """Probability that a random liked song outscores a random disliked one (ties count 0.5).
    Returns None when only one class is present (AUC is undefined)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def precision_at_k(scores, labels, k=10, seed=0):
    """Of the k songs the model ranks highest, the share that were liked.
    Ties are broken randomly (fixed seed) so a model that scores everything equally gets
    chance level instead of whatever order the file happens to be in."""
    k = min(k, len(scores))
    if k == 0:
        return None
    rng = random.Random(seed)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], rng.random()))
    return sum(1 for i in order[:k] if labels[i]) / k


# ---------------------------------------------------------------- models
# Each model: fit(train: list[(Song, int)]) then score(song) -> float (higher = more likely liked).

class RandomModel:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def fit(self, train):
        pass

    def score(self, song):
        return self.rng.random()


class PopularityModel:
    name = "popularity-only"

    def fit(self, train):
        pass

    def score(self, song):
        return song.audio_features.get("popularity", 0.0)


class BanditModel:
    """The current online learner trained from scratch: one pass over the labels, shuffled,
    exactly as it would learn live (thumbs up = +1, thumbs down = -1, full confidence)."""
    name = "bandit (from scratch)"

    def __init__(self, seed=0, make_agent=None):
        self.seed = seed
        self.make_agent = make_agent or (lambda: BanditAgent(seed=seed))
        self.agent = None

    def fit(self, train):
        self.agent = self.make_agent()
        order = list(train)
        random.Random(self.seed).shuffle(order)
        for song, y in order:
            liking = 1.0 if y else -1.0
            reaction = GroupReaction(0.0, song.song_id, (PersonReaction(0, "happy" if y else "sad", liking, 1.0),),
                                     liking, {})
            self.agent.update(song, reaction, Context())

    def score(self, song):
        return self.agent.predict(song, Context())


def default_models(seed=0):
    return {"random": lambda: RandomModel(seed), "popularity-only": PopularityModel,
            "bandit (from scratch)": lambda: BanditModel(seed)}


# ---------------------------------------------------------------- data

def make_folds(items, n_folds=5, seed=0):
    """Stratified: likes and dislikes are dealt out separately so every fold has both."""
    rng = random.Random(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in (1, 0):
        group = [it for it in items if bool(it[1]) == bool(cls)]
        rng.shuffle(group)
        for i, it in enumerate(group):
            folds[i % n_folds].append(it)
    return folds


def evaluate_fold(factory, train, test, k=10):
    model = factory()
    model.fit(train)
    scores = [model.score(s) for s, _ in test]
    labels = [y for _, y in test]
    return auc(scores, labels), precision_at_k(scores, labels, k)


def cross_validate(items, factory, n_folds=5, seed=0, k=10):
    """items: list of (Song, label). Returns {'auc': [...per fold], 'p@k': [...], 'skipped': n}."""
    folds = make_folds(items, n_folds, seed)
    aucs, precs, skipped = [], [], 0
    for i, test in enumerate(folds):
        train = [it for j, f in enumerate(folds) if j != i for it in f]
        a, p = evaluate_fold(factory, train, test, k)
        if a is None:
            skipped += 1
            continue
        aucs.append(a)
        precs.append(p)
    return {"auc": aucs, "p@k": precs, "skipped": skipped}


def learning_curve(items, factory, sizes=LEARNING_SIZES, n_folds=5, seed=0, k=10):
    """Mean AUC when only the first n training labels (random order) are used."""
    folds = make_folds(items, n_folds, seed)
    rng = random.Random(seed + 1)
    curve = {}
    for n in sizes:
        vals = []
        for i, test in enumerate(folds):
            train = [it for j, f in enumerate(folds) if j != i for it in f]
            rng.shuffle(train)
            a, _ = evaluate_fold(factory, train[:n], test, k)
            if a is not None:
                vals.append(a)
        if vals:
            curve[n] = statistics.mean(vals)
    return curve


def split_final(rows, n=FINAL_SIZE, seed=0):
    """Rows marked split=test are the final set; otherwise n rows are held out by seed."""
    marked = [r for r in rows if r.get("split") == "test"]
    if marked:
        dev = [r for r in rows if r.get("split") != "test"]
        return dev, marked
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    final = set(idx[:n])
    return [r for i, r in enumerate(rows) if i not in final], [r for i, r in enumerate(rows) if i in final]


def parse_label(text):
    t = str(text).strip().lower()
    if t in ("1", "1.0", "like", "liked", "up", "true", "yes"):
        return 1
    if t in ("0", "-1", "0.0", "-1.0", "dislike", "disliked", "down", "false", "no"):
        return 0
    raise ValueError(f"unreadable label {text!r} (use 1 for liked, 0 for disliked)")


def load_labels(path, user=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if user and r.get("user") != user:
                continue
            rows.append({"user": r.get("user", ""), "song_id": r["song_id"].strip(),
                         "label": parse_label(r["label"]), "split": (r.get("split") or "").strip()})
    return rows


def attach_songs(rows, catalog):
    """Join rows to Song objects. Returns (items, n_missing)."""
    by_id = {s.song_id: s for s in catalog}
    items, missing = [], 0
    for r in rows:
        song = by_id.get(r["song_id"])
        if song is None:
            missing += 1
        else:
            items.append((song, r["label"]))
    return items, missing


# ---------------------------------------------------------------- reporting

def _fmt(vals):
    if not vals:
        return "n/a"
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):.3f} +/- {sd:.3f}"


def report_cv(items, models, n_folds=5, seed=0, k=10, out=print):
    likes = sum(1 for _, y in items if y)
    out(f"\n{len(items)} labeled songs ({likes} liked, {len(items) - likes} disliked), {n_folds}-fold cross-validation")
    out(f"{'model':<24}{'AUC (mean +/- sd)':<22}{'precision@' + str(k):<14}")
    results = {}
    for name, factory in models.items():
        r = cross_validate(items, factory, n_folds, seed, k)
        results[name] = r
        out(f"{name:<24}{_fmt(r['auc']):<22}{_fmt(r['p@k']):<14}")
    out("\nLearning curve (mean AUC using only the first n training labels):")
    full = len(items) - len(items) // n_folds  # labels available to train on in each fold
    sizes = sorted({n for n in LEARNING_SIZES if n < full} | {full})
    curves = {name: learning_curve(items, f, sizes, n_folds, seed, k) for name, f in models.items()}
    ns = sorted({n for c in curves.values() for n in c})
    out(f"{'model':<24}" + "".join(f"{('n=' + str(n)):>9}" for n in ns))
    for name, c in curves.items():
        out(f"{name:<24}" + "".join(f"{c[n]:>9.3f}" if n in c else f"{'-':>9}" for n in ns))
    return results, curves


def run_final(dev_items, final_items, models, user, directory=DEFAULT_DIR, k=10, force=False, out=print):
    """Train on all development labels, score the held-out set. Refuses a second run."""
    directory = Path(directory)
    lock = directory / f"final_{user}.json"
    if lock.exists() and not force:
        raise SystemExit(f"Final test for '{user}' was already run ({lock}). It can be used once; "
                         "tuning after seeing it would make the result meaningless.")
    out(f"\nFINAL TEST for '{user}': train on {len(dev_items)} labels, test on {len(final_items)} held-out songs")
    out(f"{'model':<24}{'AUC':<10}{'precision@' + str(k):<14}")
    result = {}
    for name, factory in models.items():
        a, p = evaluate_fold(factory, dev_items, final_items, k)
        result[name] = {"auc": a, "p@k": p}
        out(f"{name:<24}{('n/a' if a is None else f'{a:.3f}'):<10}{('n/a' if p is None else f'{p:.3f}'):<14}")
    directory.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"user": user, "time": time.time(), "n_train": len(dev_items),
                                "n_test": len(final_items), "result": result}, indent=2))
    out(f"(saved to {lock}; this test cannot be run again)")
    return result


# ---------------------------------------------------------------- demo data

def synthetic_labels(catalog, n=150, liked_genres=("jazz", "classical", "acoustic", "piano"),
                     noise=0.1, seed=0):
    """SYNTHETIC: a fake listener who likes certain genres. Only to show the harness working."""
    rng = random.Random(seed)
    pool = [s for s in catalog if s.genre in liked_genres] + rng.sample(catalog, min(len(catalog), 4 * n))
    rng.shuffle(pool)
    seen, items = set(), []
    for s in pool:
        if s.song_id in seen:
            continue
        seen.add(s.song_id)
        y = int(s.genre in liked_genres)
        if rng.random() < noise:
            y = 1 - y
        items.append((s, y))
        if len(items) == n:
            break
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline evaluation of song recommenders.")
    ap.add_argument("--labels", help="CSV with columns user, song_id, label")
    ap.add_argument("--user", help="only use this user's rows (also names the final-test lock)")
    ap.add_argument("--final", action="store_true", help="run the one-time final held-out test")
    ap.add_argument("--demo", action="store_true", help="use SYNTHETIC labels to show the harness working")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available, load_spotify
    if not dataset_available():
        sys.exit("The Spotify dataset is missing (data/spotify/dataset.csv). Run music/catalog/download_spotify.py.")
    catalog = load_spotify()
    models = default_models(args.seed)

    if args.demo:
        print("DEMO: labels are SYNTHETIC (a fake listener who likes jazz/classical/acoustic/piano), not a real person.")
        items = synthetic_labels(catalog, seed=args.seed)
        report_cv(items, models, args.folds, args.seed)
        return 0
    if not args.labels:
        ap.error("give --labels FILE.csv (or --demo)")
    rows = load_labels(args.labels, args.user)
    if not rows:
        sys.exit("No rows found" + (f" for user '{args.user}'." if args.user else "."))
    dev_rows, final_rows = split_final(rows, seed=args.seed)
    dev, miss_d = attach_songs(dev_rows, catalog)
    final, miss_f = attach_songs(final_rows, catalog)
    if miss_d or miss_f:
        print(f"Note: {miss_d + miss_f} rows skipped because their song_id is not in the dataset.")
    if args.final:
        run_final(dev, final, models, args.user or "default")
    else:
        print(f"({len(final)} songs are held out for the final test and are not used here.)")
        report_cv(dev, models, args.folds, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
