"""Base model v1: a logistic-regression taste model trained step by step, with checkpoints and an exportable file.

What it is: P(thumb up | song) from the song's audio numbers, popularity, mood and genre, trained on explicit thumbs.
What it is not: a model of many people. Layer 1 needs per-person likes from a crowd and none is available, so this base is
trained on ONE user's thumbs (reports/step06_profiles_and_personalization.md, step08). Its job here: a frozen, exportable file
with a documented training run, which later layers can sit on.

    python3 -m rl.base_model --user adi            # trains, writes models/base_model_v1.json, checkpoints, graph, report

Pure Python (no numpy) so the exported JSON can be scored anywhere: see predict_from_file().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

from rl.eval import auc, make_folds

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
FORMAT = "makecu-base-model"
VERSION = 1
NUMERIC = ["valence", "energy", "danceability", "tempo", "acousticness", "instrumentalness", "popularity"]
MOODS = ["upbeat", "aggressive", "calm", "melancholy"]
MIN_GENRE_COUNT = 5


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def split_items(items, fractions=(0.70, 0.15, 0.15), seed=0):
    """Stratified train / validation / test split. Test is for one final read only."""
    rng = random.Random(seed)
    parts = ([], [], [])
    for cls in (1, 0):
        group = [it for it in items if bool(it[1]) == bool(cls)]
        rng.shuffle(group)
        a = round(len(group) * fractions[0])
        b = a + round(len(group) * fractions[1])
        parts[0].extend(group[:a])
        parts[1].extend(group[a:b])
        parts[2].extend(group[b:])
    for p in parts:
        rng.shuffle(p)
    return parts


class Featurizer:
    """Turns a Song into a number vector. Everything it needs (means, scales, genre list) is stored with the model."""

    def __init__(self, means=None, sds=None, genres=None):
        self.means, self.sds, self.genres = means or {}, sds or {}, genres or []

    @property
    def names(self):
        return NUMERIC + [f"mood:{m}" for m in MOODS] + [f"genre:{g}" for g in self.genres]

    @classmethod
    def fit(cls, songs):
        means, sds = {}, {}
        for c in NUMERIC:
            vals = [s.audio_features[c] for s in songs if c in s.audio_features]
            m = sum(vals) / len(vals) if vals else 0.0
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) if vals else 1.0
            means[c], sds[c] = m, sd or 1.0
        counts = {}
        for s in songs:
            counts[s.genre] = counts.get(s.genre, 0) + 1
        return cls(means, sds, sorted(g for g, n in counts.items() if n >= MIN_GENRE_COUNT and g))

    def vector(self, song):
        v = [((song.audio_features.get(c, self.means[c]) - self.means[c]) / self.sds[c]) for c in NUMERIC]
        v += [1.0 if song.mood == m else 0.0 for m in MOODS]
        v += [1.0 if song.genre == g else 0.0 for g in self.genres]
        return v


def log_loss(ps, ys):
    return -sum(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)) for p, y in zip(ps, ys)) / len(ys)


class BaseModel:
    def __init__(self, featurizer, weights=None, bias=0.0, meta=None):
        self.f, self.bias, self.meta = featurizer, bias, meta or {}
        self.w = list(weights) if weights is not None else [0.0] * len(featurizer.names)

    def prob(self, song):
        return _sigmoid(sum(a * b for a, b in zip(self.w, self.f.vector(song))) + self.bias)

    def score(self, song):          # the interface rl.eval expects
        return self.prob(song)

    def payload(self):
        return {"feature_names": self.f.names, "weights": [round(x, 8) for x in self.w], "bias": round(self.bias, 8),
                "means": self.f.means, "sds": self.f.sds, "genres": self.f.genres}

    def copy(self):
        return BaseModel(Featurizer(dict(self.f.means), dict(self.f.sds), list(self.f.genres)), self.w, self.bias, dict(self.meta))


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_model(model, path, extra=None):
    """One self-contained JSON file: weights, scaling, genre list, training facts and a hash of the weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.payload()
    doc = {"format": FORMAT, "version": VERSION, "saved_at": time.time(), "model": payload,
           "weights_sha256": payload_hash(payload), "training": {**model.meta, **(extra or {})}}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    return path


def load_model(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"{path} is not a {FORMAT} v{VERSION} file")
    m = doc["model"]
    if payload_hash(m) != doc["weights_sha256"]:
        raise ValueError(f"{path}: weights do not match their hash (file changed or damaged)")
    return BaseModel(Featurizer(m["means"], m["sds"], m["genres"]), m["weights"], m["bias"], doc.get("training", {}))


def predict_from_file(path, song):
    return load_model(path).prob(song)


def train(items, seed=0, epochs=400, lr=0.1, lam=0.05, checkpoint_every=40, directory=None, log=print, log_every=50):
    """Full-batch gradient descent on log loss with an L2 penalty.

    Returns dict(model_best, model_last, history, checkpoints, splits). `history` has one row per epoch. The model
    with the lowest validation loss is `model_best`. The test split is never touched here."""
    directory = Path(directory or MODEL_DIR)
    train_i, val_i, test_i = split_items(items, seed=seed)
    f = Featurizer.fit([s for s, _ in train_i])
    Xtr = [f.vector(s) for s, _ in train_i]
    ytr = [y for _, y in train_i]
    Xva = [f.vector(s) for s, _ in val_i]
    yva = [y for _, y in val_i]
    model = BaseModel(f)
    d = len(model.w)
    history, checkpoints, best = [], [], (float("inf"), None, 0)
    t0 = time.time()
    for ep in range(0, epochs + 1):
        ptr = [_sigmoid(sum(a * b for a, b in zip(model.w, x)) + model.bias) for x in Xtr]
        pva = [_sigmoid(sum(a * b for a, b in zip(model.w, x)) + model.bias) for x in Xva]
        row = {"epoch": ep, "train_loss": log_loss(ptr, ytr), "val_loss": log_loss(pva, yva),
               "train_auc": auc(ptr, ytr), "val_auc": auc(pva, yva)}
        history.append(row)
        if row["val_loss"] < best[0]:
            best = (row["val_loss"], model.copy(), ep)
        if ep % checkpoint_every == 0:
            c = model.copy()
            c.meta = {"epoch": ep, **{k: row[k] for k in ("train_loss", "val_loss", "train_auc", "val_auc")}}
            path = save_model(c, directory / "checkpoints" / f"base_epoch_{ep:04d}.json")
            checkpoints.append((ep, path))
        if log and (ep % log_every == 0 or ep == epochs):
            log(f"  epoch {ep:>4}/{epochs}  train loss {row['train_loss']:.4f}  val loss {row['val_loss']:.4f}  "
                f"val AUC {row['val_auc']:.3f}  ({time.time() - t0:.1f}s)")
        if ep == epochs:
            break
        grad = [0.0] * d
        gb = 0.0
        for x, y, p in zip(Xtr, ytr, ptr):
            e = p - y
            gb += e
            for j in range(d):
                grad[j] += e * x[j]
        n = len(ytr)
        for j in range(d):
            model.w[j] -= lr * (grad[j] / n + lam * model.w[j])
        model.bias -= lr * gb / n
    best[1].meta = {**best[1].meta, "best_epoch": best[2]}
    return {"model_best": best[1], "best_epoch": best[2], "model_last": model, "history": history,
            "checkpoints": checkpoints, "splits": (train_i, val_i, test_i)}


def evaluate(model, items):
    scores = [model.score(s) for s, _ in items]
    ys = [y for _, y in items]
    return {"n": len(items), "liked": sum(ys), "auc": auc(scores, ys), "log_loss": log_loss(scores, ys)}


def popularity_auc(items):
    return auc([s.audio_features.get("popularity", 0.0) for s, _ in items], [y for _, y in items])


# ---------------------------------------------------------------- graph (plain SVG, no plotting library)

def training_graph_svg(history, checkpoint_epochs, best_epoch, pop_val_auc=None, title="Base model training"):
    W, H, ml, mr, mt, mb = 920, 400, 60, 20, 78, 50
    pw = (W - ml - mr - 60) / 2
    ph = H - mt - mb
    xs = [r["epoch"] for r in history]
    x0, x1 = min(xs), max(xs)

    def panel(ox, series, ylo, yhi, ylabel, hline=None):
        def X(v):
            return ox + (v - x0) / (x1 - x0) * pw

        def Y(v):
            return mt + (1 - (v - ylo) / (yhi - ylo)) * ph
        out = [f'<rect x="{ox}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
        for i in range(5):
            v = ylo + (yhi - ylo) * i / 4
            out.append(f'<line x1="{ox}" x2="{ox + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>'
                       f'<text x="{ox - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end">{v:.2f}</text>')
        for i in range(5):
            v = x0 + (x1 - x0) * i / 4
            out.append(f'<text x="{X(v):.1f}" y="{mt + ph + 16}" font-size="11" text-anchor="middle">{v:.0f}</text>')
        out.append(f'<text x="{ox + pw / 2}" y="{H - 10}" font-size="12" text-anchor="middle">epoch</text>')
        out.append(f'<text x="{ox + pw / 2}" y="{mt - 10}" font-size="13" font-weight="bold" text-anchor="middle">{ylabel}</text>')
        for e in checkpoint_epochs:
            out.append(f'<line x1="{X(e):.1f}" x2="{X(e):.1f}" y1="{mt + ph}" y2="{mt + ph - 8}" stroke="#555" stroke-width="2"/>')
        out.append(f'<line x1="{X(best_epoch):.1f}" x2="{X(best_epoch):.1f}" y1="{mt}" y2="{mt + ph}" stroke="#2a9d3f" stroke-dasharray="4 3"/>')
        if hline is not None and ylo <= hline <= yhi:
            out.append(f'<line x1="{ox}" x2="{ox + pw}" y1="{Y(hline):.1f}" y2="{Y(hline):.1f}" stroke="#888" stroke-dasharray="6 4"/>'
                       f'<text x="{ox + pw - 4}" y="{Y(hline) - 5:.1f}" font-size="11" text-anchor="end" fill="#555">popularity-only</text>')
        for key, color, label in series:
            pts = " ".join(f"{X(r['epoch']):.1f},{Y(min(max(r[key], ylo), yhi)):.1f}" for r in history)
            out.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
        lx = ox + 8
        for i, (key, color, label) in enumerate(series):
            out.append(f'<rect x="{lx}" y="{mt + 8 + i * 16}" width="10" height="10" fill="{color}"/>'
                       f'<text x="{lx + 15}" y="{mt + 17 + i * 16}" font-size="11">{label}</text>')
        return out

    lo_l = min(min(r["train_loss"], r["val_loss"]) for r in history)
    hi_l = max(max(r["train_loss"], r["val_loss"]) for r in history)
    lo_a = min(min(r["train_auc"], r["val_auc"]) for r in history + [{"train_auc": 0.5, "val_auc": 0.5}])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="22" font-size="15" font-weight="bold" text-anchor="middle">{title}</text>',
             f'<text x="{W / 2}" y="38" font-size="11" text-anchor="middle" fill="#555">grey ticks = checkpoints saved, green dashed = best checkpoint (epoch {best_epoch}, lowest validation loss)</text>']
    parts += panel(ml, [("train_loss", "#d1495b", "training loss"), ("val_loss", "#1f6fb2", "validation loss")],
                   max(0.0, lo_l - 0.02), hi_l + 0.02, "loss (lower is better)")
    parts += panel(ml + pw + 60, [("train_auc", "#d1495b", "training AUC"), ("val_auc", "#1f6fb2", "validation AUC")],
                   max(0.4, lo_a - 0.02), 1.0, "AUC (higher is better)", hline=pop_val_auc)
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- data and CLI

def load_thumb_items(user, by_id):
    """Explicit thumbs from the camera-agreement sessions and the live session (the live thumb wins on a repeat)."""
    from rl import agreement as ag
    from rl import live_test as lt
    labels = {}
    p = Path(__file__).resolve().parents[1] / "data" / "agreement" / f"{user}.csv"
    for r in (ag.load_rows(p, user) if p.exists() else []):
        labels[r["song_id"]] = r["thumb"]
    for r in lt.load_rows(lt.live_path(user), user):
        labels[r["song_id"]] = r["thumb"]
    return [(by_id[k], v) for k, v in labels.items() if k in by_id]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="model folder (default models/)")
    a = ap.parse_args(argv)
    from music.catalog.spotify_catalog import load_spotify
    by_id = {s.song_id: s for s in load_spotify()}
    items = load_thumb_items(a.user, by_id)
    if len(items) < 50 or not any(y for _, y in items):
        raise SystemExit(f"Only {len(items)} thumb labels for '{a.user}': need at least 50 with some liked.")
    out = Path(a.out or MODEL_DIR)
    print(f"Training on {len(items)} thumbs ({sum(y for _, y in items)} liked)")
    res = train(items, a.seed, a.epochs, a.lr, a.lam, a.checkpoint_every, out)
    tr, va, te = res["splits"]
    best, last = res["model_best"], res["model_last"]
    ev = {name: evaluate(m, te) for name, m in (("best", best), ("last", last))}
    pop = {"val": popularity_auc(va), "test": popularity_auc(te)}
    print(f"Best checkpoint: epoch {res['best_epoch']}")
    print(f"TEST ({len(te)} songs, {sum(y for _, y in te)} liked): best AUC {ev['best']['auc']:.3f}, last AUC {ev['last']['auc']:.3f}, "
          f"popularity-only {pop['test']:.3f}")
    extra = {"user": a.user, "seed": a.seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "epochs": a.epochs, "lr": a.lr, "lam": a.lam, "test_auc": ev["best"]["auc"], "test_log_loss": ev["best"]["log_loss"],
             "popularity_test_auc": pop["test"]}
    path = save_model(best, out / "base_model_v1.json", extra)
    print(f"Saved {path}")
    import csv
    with open(out / "training_history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["history"][0]))
        w.writeheader()
        w.writerows(res["history"])
    svg = training_graph_svg(res["history"], [e for e, _ in res["checkpoints"]], res["best_epoch"], pop["val"])
    (out / "training_curve.svg").write_text(svg)
    print(f"Saved {out / 'training_curve.svg'} and {out / 'training_history.csv'}; {len(res['checkpoints'])} checkpoints in {out / 'checkpoints'}")
    return {"res": res, "eval": ev, "pop": pop, "extra": extra, "path": path}


if __name__ == "__main__":
    main()
