"""Adapter: gives PriorLearner the BanditAgent interface that integration/orchestrator/run_loop.py expects, so
`run(..., agent=PriorLearnerAgent(PriorLearner(base)))` swaps the learner without touching the loop."""
from __future__ import annotations

import collections

from rl.agent.bandit import MIN_CONFIDENCE, mean_confidence
from shared.schemas.messages import GroupReaction, Song


class PriorLearnerAgent:
    mainstream_bias = 0.0
    mainstream_active = False              # the prior is part of the learner; the loop's separate popularity switch is not used

    def __init__(self, learner):
        self.learner = learner
        self.play_counts = collections.Counter()

    @property
    def n_updates(self):
        return self.learner.n_updates

    # ---- what the loop prints ----
    def scopes(self, ctx=None):
        return ["global", "global", "global"]

    def top_features(self, scope="global", k=3):
        return [], []

    def predict(self, song: Song, context=None) -> float:
        return 2.0 * self.learner.prob(song) - 1.0           # in [-1, 1] like the bandit's predicted reward

    # ---- choosing ----
    def choose_next_song(self, candidate_pool, last_song_id=None, context=None, recent_ids=()):
        song, _ = self.learner.choose(candidate_pool, set(recent_ids) | {last_song_id})
        return song, "learned"

    def recommend(self, pool, context=None, k=3, unplayed_only=False):
        cand = [s for s in pool if not (unplayed_only and self.play_counts[s.song_id])]
        return sorted(((s, self.predict(s)) for s in cand), key=lambda x: -x[1])[:k]

    # ---- learning ----
    def update(self, song: Song, reaction: GroupReaction, context=None) -> dict:
        confidence = mean_confidence(reaction)
        predicted = self.predict(song)
        if confidence < MIN_CONFIDENCE:
            return {"reward": 0.0, "predicted": predicted, "error": 0.0, "skipped": True, "confidence": confidence}
        r = reaction.group_liking * confidence                # same reward definition as the bandit
        self.learner.update(song, (r + 1.0) / 2.0, weight=confidence)
        self.play_counts[song.song_id] += 1
        return {"reward": r, "predicted": predicted, "error": r - predicted, "skipped": False, "confidence": confidence}

    def record_manual_override(self, chosen: Song, replaced: Song | None, context=None) -> None:
        """A phone song change: weaker than a thumb (behaviour weight 0.7)."""
        self.learner.update_from_signals(chosen, behaviour="chosen")
        if replaced is not None:
            self.learner.update_from_signals(replaced, behaviour="replaced")
