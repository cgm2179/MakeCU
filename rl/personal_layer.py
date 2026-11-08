"""Layer 2: a personal layer on top of the frozen base (checkpoints L2.1 and L2.2). Design fixed in reports/step07_preregistration.md.

logit(song) = base_logit(song) + x . delta + b
  base_logit : the frozen base's preference head (models/base_model_v2.json), never changed
  delta      : one number per base feature plus one per extra genre (genres with 5+ of the user's training songs), starts at 0
  b          : a free bias (recalibration only)
delta is L2-penalised toward zero, so with no data the model IS the base, and with little data it stays close to it.

    python3 -m rl.personal_layer --user adi       # cross-validated comparison, learning curve, saves the personal profile
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rl import base_lastfm as bl
from rl.base_model import MOODS, NUMERIC
from rl.eval import BanditModel, make_folds

ROOT = Path(__file__).resolve().parents[1]
LAMS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)    # pre-registered grid
MIN_GENRE = 5
REPEATS = 5
CURVE_SIZES = (10, 20, 40, 80)
GAIN_OVER_BASE = 0.05                           # pre-registered
PERSONAL_VERSION = 1
PROFILE_DIR = ROOT / "data" / "profiles"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fast_auc(scores, y):
    y = np.asarray(y)
    pos = y == 1
    p, n = int(pos.sum()), int((~pos).sum())
    if p == 0 or n == 0:
        return float("nan")
    r = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((r[pos].sum() - p * (p + 1) / 2) / (p * n))


def fit_logistic(X, y, offset, lam, iters=60, sample_weight=None):
    """Newton's method (IRLS) for L2-penalised logistic regression with an offset and optional sample weights (soft labels allowed).
    X's LAST column is the bias (all ones) and is not penalised. Deterministic. With no weights this is the original Step 7 fit."""
    n, d = X.shape
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    tot = float(sw.sum())
    pen = np.full(d, float(lam))
    pen[-1] = 0.0
    w = np.zeros(d)

    def loss(w_):
        z = offset + X @ w_
        return float(np.sum(sw * (np.logaddexp(0, z) - y * z)) / tot + 0.5 * np.sum(pen * w_ * w_))
    cur = loss(w)
    for _ in range(iters):
        p = sigmoid(offset + X @ w)
        g = X.T @ (sw * (p - y)) / tot + pen * w
        W = np.maximum(p * (1 - p), 1e-6) * sw
        H = (X.T * W) @ X / tot + np.diag(pen) + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4 and loss(w - t * step) > cur + 1e-12:
            t /= 2
        w = w - t * step
        new = loss(w)
        if abs(cur - new) < 1e-12 and np.abs(t * step).max() < 1e-7:
            break
        cur = new
    return w


# ---------------------------------------------------------------- matrices for fast cross-validation

def build_matrices(songs, base):
    """Everything the models need, computed once for a list of songs."""
    xb = np.array([base.feat.vector(s) for s in songs])                      # base features (26)
    off = np.array([base.preference_logit(s) for s in songs])                # frozen base logit
    num = np.array([[s.audio_features.get(c, np.nan) for c in NUMERIC] for s in songs], dtype=float)
    mood = np.array([[1.0 if s.mood == m else 0.0 for m in MOODS] for s in songs])
    genres = sorted({s.genre for s in songs if s.genre})
    g = np.array([[1.0 if s.genre == x else 0.0 for x in genres] for s in songs])
    pop = np.array([s.audio_features.get("popularity", 0.0) for s in songs])
    return {"xb": xb, "off": off, "num": num, "mood": mood, "genres": genres, "g": g, "pop": pop}


def extra_genre_cols(M, tr_idx, base_genres, min_count=MIN_GENRE):
    counts = M["g"][tr_idx].sum(axis=0)
    return [j for j, name in enumerate(M["genres"]) if counts[j] >= min_count and name not in set(base_genres)]


def ft_design(M, idx, cols):
    return np.hstack([M["xb"][idx], M["g"][idx][:, cols], np.ones((len(idx), 1))])


def scratch_design(M, tr_idx, idx, min_count=MIN_GENRE):
    """Features for the no-base logistic model: numbers standardised on the TRAIN songs, moods, genres with 5+ train songs."""
    num_tr = M["num"][tr_idx]
    ok = ~np.isnan(num_tr)
    cnt = ok.sum(axis=0)
    safe = np.where(ok, num_tr, 0.0)
    means = np.where(cnt > 0, safe.sum(axis=0) / np.maximum(cnt, 1), 0.0)      # a feature missing everywhere becomes 0
    var = np.where(cnt > 0, (np.where(ok, num_tr - means, 0.0) ** 2).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    sds = np.sqrt(var)
    sds[sds == 0] = 1.0
    gcols = [j for j in range(len(M["genres"])) if M["g"][tr_idx][:, j].sum() >= min_count]
    z = np.where(np.isnan(M["num"][idx]), means, M["num"][idx])
    return np.hstack([(z - means) / sds, M["mood"][idx], M["g"][idx][:, gcols], np.ones((len(idx), 1))])


def inner_choose_lam(design_fn, y, off, tr, seed, lams=LAMS):
    """Pick the penalty by inner 5-fold CV AUC on the training part only. Ties go to the larger penalty."""
    items = [(int(i), int(y[i])) for i in tr]
    folds = make_folds(items, 5, seed)
    best, best_auc = lams[-1], -1.0
    for lam in lams:
        oof = {}
        for k, te in enumerate(folds):
            tri = np.array([i for j, f in enumerate(folds) if j != k for i, _ in f])
            tei = np.array([i for i, _ in te])
            Xtr, Xte = design_fn(tri, tri), design_fn(tri, tei)
            w = fit_logistic(Xtr, y[tri], off[tri], lam)
            for i, s in zip(tei, off[tei] + Xte @ w):
                oof[i] = s
        a = fast_auc([oof[i] for i in tr], y[tr])
        if a >= best_auc - 1e-12:           # >= so ties move to the larger penalty (grid is ascending)
            best, best_auc = lam, a
    return best


# ---------------------------------------------------------------- the personal layer object (L2.1)

class PersonalLayer:
    def __init__(self, base, extra_genres=(), delta=None, bias=0.0, lam=None):
        self.base, self.extra_genres, self.lam = base, list(extra_genres), lam
        d = len(base.feat.names) + len(self.extra_genres)
        self.delta = np.zeros(d) if delta is None else np.array(delta, dtype=float)
        self.bias = float(bias)

    def _x(self, song):
        return np.array(self.base.feat.vector(song) + [1.0 if song.genre == g else 0.0 for g in self.extra_genres])

    def logit(self, song):
        return float(self.base.preference_logit(song) + self._x(song) @ self.delta + self.bias)

    def prob(self, song):
        return float(sigmoid(self.logit(song)))

    score = prob

    def reset(self):
        """Forget the person: back to exactly the base."""
        self.extra_genres, self.delta, self.bias = [], np.zeros(len(self.base.feat.names)), 0.0

    @classmethod
    def fit(cls, base, items, lam):
        songs = [s for s, _ in items]
        y = np.array([float(v) for _, v in items])
        M = build_matrices(songs, base)
        idx = np.arange(len(songs))
        cols = extra_genre_cols(M, idx, base.feat.genres)
        w = fit_logistic(ft_design(M, idx, cols), y, M["off"], lam)
        return cls(base, [M["genres"][j] for j in cols], w[:-1], w[-1], lam)

    def to_dict(self):
        return {"format": "makecu-personal-layer", "version": PERSONAL_VERSION, "base_model_id": self.base.id,
                "base_sha256": self.base.sha256, "saved_at": time.time(), "lam": self.lam,
                "extra_genres": self.extra_genres, "delta": [round(float(x), 8) for x in self.delta],
                "bias": round(self.bias, 8)}


def save_personal(layer, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(layer.to_dict(), indent=1))
    os.replace(tmp, path)
    return path


def load_personal(path, base):
    """Load a personal layer; refuses one that was made on a different base (different hash)."""
    d = json.loads(Path(path).read_text())
    if d.get("format") != "makecu-personal-layer" or d.get("version") != PERSONAL_VERSION:
        raise ValueError(f"{path} is not a personal layer file")
    if d["base_sha256"] != base.sha256:
        raise ValueError(f"personal layer was fitted on base {d['base_model_id']} ({d['base_sha256'][:8]}...), "
                         f"but the loaded base is {base.id} ({base.sha256[:8]}...). Refusing to load it.")
    return PersonalLayer(base, d["extra_genres"], d["delta"], d["bias"], d.get("lam"))


def personal_device_bytes(layer):
    """Size on the device: delta + bias as float32, plus extra genre names (NUL-terminated)."""
    return 4 * (len(layer.delta) + 1) + sum(len(g) + 1 for g in layer.extra_genres)


# ---------------------------------------------------------------- cross-validation (L2.2)

MODELS = ("popularity-only", "base-only", "bandit (scratch)", "logistic (scratch)", "fine-tuned")


def run_cv(items, base, repeats=REPEATS, log=print):
    songs = [s for s, _ in items]
    y = np.array([float(v) for _, v in items])
    M = build_matrices(songs, base)
    n = len(songs)
    oof = {m: np.zeros((repeats, n)) for m in MODELS}
    chosen = {"logistic (scratch)": [], "fine-tuned": []}
    for r in range(repeats):
        folds = make_folds([(i, int(y[i])) for i in range(n)], 5, seed=r)
        for k, te in enumerate(folds):
            tei = np.array([i for i, _ in te])
            tri = np.array([i for j, f in enumerate(folds) if j != k for i, _ in f])
            oof["popularity-only"][r, tei] = M["pop"][tei]
            oof["base-only"][r, tei] = M["off"][tei]
            bm = BanditModel(r)
            bm.fit([(songs[i], int(y[i])) for i in tri])
            oof["bandit (scratch)"][r, tei] = [bm.score(songs[i]) for i in tei]
            # logistic from scratch
            sd = lambda a, b: scratch_design(M, a, b)
            lam = inner_choose_lam(sd, y, np.zeros(n), tri, seed=100 * r + k)
            chosen["logistic (scratch)"].append(lam)
            w = fit_logistic(sd(tri, tri), y[tri], np.zeros(len(tri)), lam)
            oof["logistic (scratch)"][r, tei] = sd(tri, tei) @ w
            # fine-tuned on the base
            cols = extra_genre_cols(M, tri, base.feat.genres)
            fd = lambda a, b, cols=cols: ft_design(M, b, cols)
            lam = inner_choose_lam(fd, y, M["off"], tri, seed=100 * r + k)
            chosen["fine-tuned"].append(lam)
            w = fit_logistic(ft_design(M, tri, cols), y[tri], M["off"][tri], lam)
            oof["fine-tuned"][r, tei] = M["off"][tei] + ft_design(M, tei, cols) @ w
        if log:
            log(f"  repeat {r + 1}/{repeats}: " + ", ".join(f"{m} {fast_auc(oof[m][r], y):.3f}" for m in MODELS))
    per_rep = {m: [fast_auc(oof[m][r], y) for r in range(repeats)] for m in MODELS}
    mean_auc = {m: float(np.mean(per_rep[m])) for m in MODELS}
    avg = {m: oof[m].mean(axis=0) for m in MODELS}
    return {"y": y, "M": M, "per_rep": per_rep, "mean_auc": mean_auc, "avg_oof": avg, "chosen": chosen, "oof": oof}


def per_fold_auc(oof, y, repeats):
    """Mean AUC computed inside each test fold (scores from one fold's model only), so a model whose bias is refitted per fold
    is not penalised for scores that sit on different scales in different folds. Robustness check; the pre-registered metric is pooled."""
    n = len(y)
    out = {m: [] for m in oof}
    for r in range(repeats):
        for te in make_folds([(i, int(y[i])) for i in range(n)], 5, seed=r):
            idx = np.array([i for i, _ in te])
            for m in oof:
                out[m].append(fast_auc(oof[m][r, idx], y[idx]))
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in out.items()}


def boot_diff(a_scores, b_scores, y, n=2000, seed=0):
    """Point difference and 95% range of AUC(a) - AUC(b), resampling songs."""
    rng = np.random.RandomState(seed)
    point = fast_auc(a_scores, y) - fast_auc(b_scores, y)
    ds = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(fast_auc(a_scores[i], y[i]) - fast_auc(b_scores[i], y[i]))
    return float(point), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def modal(xs):
    vals, counts = np.unique(xs, return_counts=True)
    return float(vals[np.argmax(counts)])


def learning_curve(items, base, lams, repeats=REPEATS, sizes=CURVE_SIZES, log=print):
    songs = [s for s, _ in items]
    y = np.array([float(v) for _, v in items])
    M = build_matrices(songs, base)
    n = len(songs)
    out = {m: {} for m in ("base-only", "bandit (scratch)", "logistic (scratch)", "fine-tuned")}
    full_train = int(round(n * 0.8))
    for size in list(sizes) + [full_train]:
        accum = {m: [] for m in out}
        for r in range(repeats):
            folds = make_folds([(i, int(y[i])) for i in range(n)], 5, seed=r)
            oofs = {m: np.zeros(n) for m in out}
            for k, te in enumerate(folds):
                tei = np.array([i for i, _ in te])
                tri = np.array([i for j, f in enumerate(folds) if j != k for i, _ in f])
                rng = np.random.RandomState(1000 * r + k)
                use = rng.permutation(tri)[:size]
                oofs["base-only"][tei] = M["off"][tei]
                bm = BanditModel(r)
                bm.fit([(songs[i], int(y[i])) for i in use])
                oofs["bandit (scratch)"][tei] = [bm.score(songs[i]) for i in tei]
                if len(set(y[use])) < 2:
                    oofs["logistic (scratch)"][tei] = 0.0
                    oofs["fine-tuned"][tei] = M["off"][tei]
                    continue
                w = fit_logistic(scratch_design(M, use, use), y[use], np.zeros(len(use)), lams["logistic (scratch)"])
                oofs["logistic (scratch)"][tei] = scratch_design(M, use, tei) @ w
                cols = extra_genre_cols(M, use, base.feat.genres)
                w = fit_logistic(ft_design(M, use, cols), y[use], M["off"][use], lams["fine-tuned"])
                oofs["fine-tuned"][tei] = M["off"][tei] + ft_design(M, tei, cols) @ w
            for m in out:
                accum[m].append(fast_auc(oofs[m], y))
        for m in out:
            out[m][size] = float(np.mean(accum[m]))
        if log:
            log(f"  n={size:>3}: " + ", ".join(f"{m} {out[m][size]:.3f}" for m in out))
    return out, full_train


def curve_svg(curve, full_train, pop_auc, title="Learning curve: AUC on songs the model never saw"):
    W, H, ml, mr, mt, mb = 720, 380, 60, 160, 50, 50
    sizes = sorted(next(iter(curve.values())))
    allv = [v for m in curve.values() for v in m.values()] + [pop_auc]
    ylo, yhi = min(allv) - 0.03, max(allv) + 0.03
    pw, ph = W - ml - mr, H - mt - mb
    import math
    X = lambda s: ml + (math.log10(s) - math.log10(sizes[0])) / (math.log10(sizes[-1]) - math.log10(sizes[0])) * pw
    Y = lambda v: mt + (1 - (v - ylo) / (yhi - ylo)) * ph
    colors = {"base-only": "#888888", "bandit (scratch)": "#d1495b", "logistic (scratch)": "#edae49", "fine-tuned": "#1f6fb2"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W / 2}" y="24" font-size="15" font-weight="bold" text-anchor="middle">{title}</text>',
         f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
    for i in range(5):
        v = ylo + (yhi - ylo) * i / 4
        o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>'
                 f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" font-size="11" text-anchor="end">{v:.2f}</text>')
    for s in sizes:
        o.append(f'<text x="{X(s):.1f}" y="{mt + ph + 16}" font-size="11" text-anchor="middle">{s}</text>')
    o.append(f'<text x="{ml + pw / 2}" y="{H - 10}" font-size="12" text-anchor="middle">training labels used (log scale; last = all)</text>')
    o.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(pop_auc):.1f}" y2="{Y(pop_auc):.1f}" stroke="#444" stroke-dasharray="6 4"/>'
             f'<text x="{ml + pw + 6}" y="{Y(pop_auc) + 4:.1f}" font-size="11">popularity-only</text>')
    label_y = {}
    for m, pts in curve.items():
        coords = " ".join(f"{X(s):.1f},{Y(pts[s]):.1f}" for s in sizes)
        o.append(f'<polyline fill="none" stroke="{colors[m]}" stroke-width="2.5" points="{coords}"/>')
        label_y[m] = Y(pts[sizes[-1]]) + 4
    placed = [Y(pop_auc) + 4]                      # keep end labels at least 14 px apart (also from the popularity label)
    for m in sorted(label_y, key=lambda k: label_y[k]):
        y = label_y[m]
        while any(abs(y - q) < 14 for q in placed):
            y += 14
        placed.append(y)
        o.append(f'<text x="{ml + pw + 6}" y="{y:.1f}" font-size="11" fill="{colors[m]}">{m}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--base", default=str(bl.MODEL_DIR / "base_model_v2.json"))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    from music.catalog.spotify_catalog import load_spotify
    from rl.base_model import load_thumb_items
    by_id = {s.song_id: s for s in load_spotify()}
    items = load_thumb_items(a.user, by_id)
    base = bl.load_base(a.base)
    print(f"{len(items)} thumbs ({sum(v for _, v in items)} liked); base {base.id} {base.sha256[:12]}")
    sha_before = hashlib.sha256(Path(a.base).read_bytes()).hexdigest()
    res = run_cv(items, base, a.repeats)
    y, avg = res["y"], res["avg_oof"]
    ft = "fine-tuned"
    diffs = {m: boot_diff(avg[ft], avg[m], y) for m in MODELS if m != ft}
    lams = {m: modal(v) for m, v in res["chosen"].items()}
    print("chosen penalties (most often):", lams)
    print("learning curve:")
    curve, full_train = learning_curve(items, base, lams, a.repeats)
    layer = PersonalLayer.fit(base, items, lams[ft])
    ppath = save_personal(layer, PROFILE_DIR / f"{a.user}_personal_v2.json")
    sha_after = hashlib.sha256(Path(a.base).read_bytes()).hexdigest()
    check_reset = (lambda l: (l.reset(), all(abs(l.logit(s) - base.preference_logit(s)) < 1e-12 for s, _ in items[:50]))[1])(
        PersonalLayer(base, layer.extra_genres, layer.delta, layer.bias))
    out = {"n": len(items), "liked": int(y.sum()), "mean_auc": res["mean_auc"], "per_rep": res["per_rep"], "diffs": diffs,
           "chosen_lams": lams, "chosen_all": res["chosen"], "curve": curve, "full_train": full_train,
           "base_hash_unchanged": sha_before == sha_after, "reset_equals_base": check_reset,
           "personal_bytes": personal_device_bytes(layer), "base_bytes": 1477, "profile": str(ppath),
           "extra_genres": layer.extra_genres,
           "top_delta": sorted(zip(base.feat.names + [f"genre:{g}" for g in layer.extra_genres], layer.delta),
                               key=lambda x: -abs(x[1]))[:10]}
    pop = res["mean_auc"]["popularity-only"]
    s = curve_svg(curve, full_train, pop)
    (bl.MODEL_DIR / "step07_learning_curve.svg").write_text(s)
    (bl.MODEL_DIR / "step07_results.json").write_text(json.dumps(out, indent=1, default=str))
    print("\nMean pooled out-of-fold AUC (%d repeats of 5-fold CV):" % a.repeats)
    for m in MODELS:
        print(f"  {m:<20} {res['mean_auc'][m]:.3f}  (sd over repeats {np.std(res['per_rep'][m]):.3f})")
    for m, (d, lo, hi) in diffs.items():
        print(f"  fine-tuned minus {m:<20} {d:+.3f}  (95% range {lo:+.3f} to {hi:+.3f})")
    return out


if __name__ == "__main__":
    main()
