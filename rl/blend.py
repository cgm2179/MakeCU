"""Popularity + personal taste blended into one score, trained on thumbs (live fix after the Step 7 failure).

score(song) = w * z(popularity) + (1 - w) * z(personal bandit score), where z() standardises over the training songs.
w = 1 is popularity-only; w = 0 is the personal bandit alone. The bandit is trained on thumbs (liked = 1, disliked = 0).
"""
from __future__ import annotations

import statistics

from rl.agent.context import Context
from rl.eval import BanditModel, auc, make_folds


def _z(values):
    sd = statistics.pstdev(values) or 1.0
    return statistics.mean(values), sd


class BlendModel:
    def __init__(self, w=0.5, seed=0):
        self.w, self.seed = w, seed
        self.name = f"blend w={w:g}"
        self.bandit = None

    def fit(self, train):
        self.bandit = BanditModel(self.seed)
        self.bandit.fit(train)
        pops = [s.audio_features.get("popularity", 0.0) for s, _ in train]
        pers = [self.bandit.score(s) for s, _ in train]
        self.pm, self.ps = _z(pops)
        self.bm, self.bs = _z(pers)

    def score(self, song):
        p = (song.audio_features.get("popularity", 0.0) - self.pm) / self.ps
        b = (self.bandit.score(song) - self.bm) / self.bs
        return self.w * p + (1 - self.w) * b


def repeated_cv_auc(items, factory, repeats=5, n_folds=5, seed=0):
    """Mean AUC over several stratified k-fold splits; each song is scored by a model that never saw it."""
    out = []
    for r in range(repeats):
        folds = make_folds(items, n_folds, seed + r)
        scores, labels = [], []
        for i, test in enumerate(folds):
            train = [it for j, f in enumerate(folds) if j != i for it in f]
            m = factory()
            m.fit(train)
            scores += [m.score(s) for s, _ in test]
            labels += [y for _, y in test]
        out.append(auc(scores, labels))
    return statistics.mean(out), statistics.pstdev(out)
