"""How much to trust one person's camera: a small per-person model of P(thumb up | camera features).

The camera is a weak, noisy signal (on Adi's first 78 songs the reading ranked liked above disliked songs
with AUC 0.67, where 0.5 is chance). So it should count for a small fraction of a thumb, not equal to it.
This module (1) learns, from a person's own thumbs, what their camera features say about liking, and
(2) measures how reliable that is, on songs the model did not see, and turns it into a weight in [0, 1].

Plain-Python logistic regression with a strong pull toward 'no effect', so few songs cannot produce
confident nonsense. With too little data the weight is 0 and the camera is ignored.
"""
from __future__ import annotations

import math

from perception.emotion.clip_features import FEATURES
from rl import eval as ev

MIN_SONGS = 30         # fewer labelled songs than this: no reliability claimed, camera weight 0
MIN_PER_CLASS = 8      # and at least this many of the rarer thumb
STRENGTH = 10.0        # the prior 'no feature matters' counts as this many songs of evidence


def _sigmoid(z):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


class CameraReliability:
    def __init__(self, feature_names=None, strength=STRENGTH, iterations=300, lr=0.3):
        self.names = list(feature_names or FEATURES)
        self.strength, self.iterations, self.lr = strength, iterations, lr
        self.mean = {n: 0.0 for n in self.names}
        self.std = {n: 1.0 for n in self.names}
        self.w = {n: 0.0 for n in self.names}
        self.b = 0.0
        self.base_rate = 0.5

    def _x(self, feats):
        return [(feats.get(n, self.mean[n]) - self.mean[n]) / self.std[n] for n in self.names]

    def fit(self, rows):
        """rows: dicts with 'features' (dict) and 'thumb' (1/0)."""
        rows = [r for r in rows if r.get("features")]
        n = len(rows)
        if n == 0:
            return self
        for name in self.names:
            vals = [r["features"].get(name, 0.0) for r in rows]
            m = sum(vals) / n
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
            self.mean[name], self.std[name] = m, sd if sd > 1e-9 else 1.0
        xs = [self._x(r["features"]) for r in rows]
        ys = [r["thumb"] for r in rows]
        self.base_rate = min(0.99, max(0.01, sum(ys) / n))
        b = math.log(self.base_rate / (1 - self.base_rate))
        w = [0.0] * len(self.names)
        for _ in range(self.iterations):
            gb, gw = 0.0, [0.0] * len(w)
            for x, y in zip(xs, ys):
                err = _sigmoid(b + sum(wi * xi for wi, xi in zip(w, x))) - y
                gb += err
                for j, xi in enumerate(x):
                    gw[j] += err * xi
            b -= self.lr * gb / n
            w = [wi - self.lr * (g + self.strength * wi) / (n + self.strength) for wi, g in zip(w, gw)]
        self.b, self.w = b, dict(zip(self.names, w))
        return self

    def prob(self, feats):
        return _sigmoid(self.b + sum(self.w[n] * xi for n, xi in zip(self.names, self._x(feats))))

    def soft_reward(self, feats):
        """What the camera adds beyond this person's usual rate, as a reward in [-1, 1]:
        (p - usual rate) / max(usual rate, 1 - usual rate). It is 0 when the face says nothing new and averages 0
        over the songs the model was fitted on. (2p - 1 would be about -0.5 on almost every song for someone who
        likes 25% of songs, pushing every song down by the same amount; scaling the two sides of the rate
        differently also leaves a lean, because the dislike side is much steeper.) Someone who rarely likes songs
        can only be pushed a little below zero, because the camera cannot make a rare event rarer."""
        b = self.base_rate
        return max(-1.0, min(1.0, (self.prob(feats) - b) / max(b, 1 - b)))


def cv_probs(rows, feature_names=None, folds=5, seed=0, **kw):
    """Out-of-fold probability for every song: stratified k-fold, each song scored by a model that never saw it.

    Not leave-one-out: leaving a single song out shifts the training base rate by a fixed amount that depends
    on that song's own label (a liked song lowers it), so when the features carry little signal the held-out
    scores slide the WRONG way and AUC collapses toward 0. Stratified folds keep the training base rate the
    same for every held-out song, so a model with no signal scores about 0.5, as it should."""
    import random
    rows = [r for r in rows if r.get("features")]
    rng = random.Random(seed)
    idx = {1: [], 0: []}
    for i, r in enumerate(rows):
        idx[int(bool(r["thumb"]))].append(i)
    fold_of = {}
    for cls in idx:
        rng.shuffle(idx[cls])
        for k, i in enumerate(idx[cls]):
            fold_of[i] = k % folds
    probs = [0.0] * len(rows)
    for f in range(folds):
        train = [rows[i] for i in range(len(rows)) if fold_of[i] != f]
        model = CameraReliability(feature_names, **kw).fit(train)
        for i in range(len(rows)):
            if fold_of[i] == f:
                probs[i] = model.prob(rows[i]["features"])
    return rows, probs


def cv_auc(rows, feature_names=None, repeats=3, **kw):
    """Mean AUC over several different fold assignments (and the rows used)."""
    aucs = []
    for rep in range(repeats):
        use, probs = cv_probs(rows, feature_names, seed=rep, **kw)
        a = ev.auc(probs, [r["thumb"] for r in use])
        if a is not None:
            aucs.append(a)
    return (sum(aucs) / len(aucs) if aucs else None), use


def reliability(rows, feature_names=None, **kw):
    """(weight in [0, 1], cross-validated AUC or None, number of songs used).
    weight = 2 * (AUC - 0.5), clipped: AUC 0.5 (chance) -> 0, AUC 0.75 -> 0.5, AUC 1.0 -> 1."""
    use = [r for r in rows if r.get("features")]
    ups = sum(r["thumb"] for r in use)
    if len(use) < MIN_SONGS or min(ups, len(use) - ups) < MIN_PER_CLASS:
        return 0.0, None, len(use)
    auc, use = cv_auc(use, feature_names, **kw)
    if auc is None:
        return 0.0, None, len(use)
    return max(0.0, min(1.0, 2 * (auc - 0.5))), auc, len(use)
