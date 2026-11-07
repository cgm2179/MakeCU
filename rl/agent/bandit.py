"""Basic RL: epsilon-greedy contextual bandit over song features.

Each song is described by features (its genre and artist). The agent keeps a
weight per feature = how much the user likes it. A song's predicted reward is the
sum of its feature weights. After each reaction the weights move toward the
observed reward (a simple delta-rule / TD update), so liking one pop song also
raises other pop songs ("similar songs").
"""
from __future__ import annotations
import random

from shared.schemas.messages import GroupReaction, Song


def features(song: Song) -> list[str]:
    return [f"genre:{song.genre}", f"artist:{song.artist}"]


class BanditAgent:
    def __init__(self, epsilon=0.2, alpha=0.5, seed=None):
        self.epsilon = epsilon  # chance to explore a random song
        self.alpha = alpha  # learning rate
        self.weights: dict[str, float] = {}
        self.rng = random.Random(seed)

    def predict(self, song: Song) -> float:
        return sum(self.weights.get(f, 0.0) for f in features(song))

    def choose_next_song(self, candidate_pool: list[Song], last_song_id=None) -> tuple[Song, str]:
        """Return (song, mode) where mode is 'explore' or 'exploit'."""
        pool = [s for s in candidate_pool if s.song_id != last_song_id] or candidate_pool
        if self.rng.random() < self.epsilon:
            return self.rng.choice(pool), "explore"
        best = max(self.predict(s) for s in pool)
        return self.rng.choice([s for s in pool if self.predict(s) == best]), "exploit"

    @staticmethod
    def reward(reaction: GroupReaction) -> float:
        conf = [p.confidence for p in reaction.people] or [0.0]
        return reaction.group_liking * (sum(conf) / len(conf))

    def update(self, song: Song, reaction: GroupReaction) -> dict:
        """Learn from a reaction. Returns details for printing."""
        r = self.reward(reaction)
        predicted = self.predict(song)
        error = r - predicted
        fs = features(song)
        for f in fs:
            self.weights[f] = self.weights.get(f, 0.0) + self.alpha * error / len(fs)
        return {"reward": r, "predicted": predicted, "error": error}
