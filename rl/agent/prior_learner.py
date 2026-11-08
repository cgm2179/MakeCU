"""An online learner that starts from a prior instead of from nothing (replaces learning from scratch).

score(song) = offset(song) + x . delta + beta * uncertainty(song)
  offset       the prior: the frozen base model's logit, or popularity, or both
  delta        a correction learned from this person's feedback: L2-penalised logistic regression, refitted after every label (starts at 0, so
               with no feedback the policy follows the prior exactly)
  uncertainty  sqrt(x' H^-1 x) from the Laplace approximation of the fit: large for songs unlike anything rated yet (optimism, so it explores
               where it knows least; beta = 0 turns this off)
Feedback goes through rl.agent.reward.combine: thumbs (weight 1) and behaviour (0.7) always count; the camera counts only if `camera_validated`
is True, i.e. only after a validation test (reports/step03e_*) has passed. Default False.
"""
from __future__ import annotations

import random

import numpy as np

from rl.agent.reward import combine
from rl.personal_layer import fit_logistic, sigmoid

OFFSETS = ("base", "popularity", "base+popularity")


class PriorLearner:
    def __init__(self, base, offset="base", lam=0.05, beta=0.3, seed=None, camera_validated=False):
        if offset not in OFFSETS:
            raise ValueError(f"offset must be one of {OFFSETS}")
        self.base, self.offset_kind, self.lam, self.beta = base, offset, lam, beta
        self.camera_validated = camera_validated
        self.rng = random.Random(seed)
        self.items = []                        # (song, soft label in [0, 1], weight)
        d = len(base.feat.names) + 1
        self.delta = np.zeros(d)               # last entry = bias
        self.H = np.eye(d) * lam               # posterior precision (prior only until data arrives)
        self.n_updates = 0
        self._vec = base.__dict__.setdefault("_vec_cache", {})          # feature vectors and base logits never change, so share them

    # ---- scoring ----
    def _x(self, song):
        hit = self._vec.get(song.song_id)
        if hit is None:
            x = np.array(self.base.feat.vector(song) + [1.0])
            hit = self._vec[song.song_id] = (x, float(x[:-1] @ self.base._w + self.base.bias))
        return hit[0]

    def prior(self, song):
        pop = (song.audio_features.get("popularity", 50.0) - 50.0) / 25.0
        self._x(song)
        b = self._vec[song.song_id][1]
        return {"base": b, "popularity": pop, "base+popularity": b + pop}[self.offset_kind]

    def logit(self, song):
        return float(self.prior(song) + self._x(song) @ self.delta)

    def uncertainty(self, song):
        x = self._x(song)
        return float(np.sqrt(max(x @ np.linalg.solve(self.H, x), 0.0)))

    def score(self, song):
        return self.logit(song) + self.beta * self.uncertainty(song)

    def prob(self, song):
        return float(sigmoid(self.logit(song)))

    # ---- learning ----
    def update(self, song, y, weight=1.0):
        """y in [0, 1] (1 = liked). Refits the correction on everything seen so far."""
        self.items.append((song, float(y), float(weight)))
        self._refit()
        self.n_updates += 1

    def update_from_signals(self, song, thumb=None, behaviour=None, camera=None, camera_weight=0.0):
        """Learn from whatever signals exist for this song. Returns the combined reward in [-1, 1], or None if there was nothing to learn from."""
        reward, parts = combine(thumb, behaviour, camera if self.camera_validated else None, camera_weight if self.camera_validated else 0.0)
        if reward is None:
            return None
        self.update(song, (reward + 1.0) / 2.0, min(1.0, sum(w for _, w, _ in parts)))
        return reward

    def _refit(self):
        songs, ys, ws = zip(*self.items)
        X = np.array([self._x(s) for s in songs])
        off = np.array([self.prior(s) for s in songs])
        y, w = np.array(ys), np.array(ws)
        coef = fit_logistic(X, y, off, self.lam, sample_weight=w)
        self.delta = coef
        p = sigmoid(off + X @ coef)
        tot = max(float(w.sum()), 1.0)
        self.H = (X.T * (w * np.maximum(p * (1 - p), 1e-6))) @ X + np.eye(X.shape[1]) * self.lam * tot

    # ---- choosing ----
    def choose(self, pool, recent_ids=()):
        """The highest-scoring song not recently played (random among exact ties). Returns (song, score)."""
        skip = set(recent_ids)
        cand = [s for s in pool if s.song_id not in skip] or list(pool)
        scores = [self.score(s) for s in cand]
        best = max(scores)
        pick = self.rng.choice([s for s, v in zip(cand, scores) if v >= best - 1e-12])
        return pick, best

    def reset(self):
        """Forget the person: back to following the prior."""
        self.items, self.n_updates = [], 0
        self.delta = np.zeros_like(self.delta)
        self.H = np.eye(len(self.delta)) * self.lam
