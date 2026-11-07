"""Simulated listener so the loop can run with no keyboard (for tests)."""
import random

from perception.emotion.fake_reaction import _reaction


class SimulatedListener:
    def __init__(self, liked_genres=("jazz",), seed=0):
        self.liked = set(liked_genres)
        self.rng = random.Random(seed)

    def react(self, song):
        likes = song.genre in self.liked
        if self.rng.random() < 0.1:  # a little noise
            likes = not likes
        return _reaction(song.song_id, 1.0 if likes else -1.0, "happy" if likes else "sad")
