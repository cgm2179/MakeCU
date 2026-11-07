"""Fake listener with a simple hidden taste: it likes certain genres.

Use set_liked() mid-run to change its taste and check the agent adapts.
Reports "happy" when it likes a song and "sad" when it doesn't (like the keyboard).
"""
import random

from perception.emotion.fake_reaction import _reaction


class TasteListener:
    def __init__(self, liked_genres, seed=0, noise=0.1):
        self.liked = set(liked_genres)
        self.rng = random.Random(seed)
        self.noise = noise

    def set_liked(self, genres) -> None:
        self.liked = set(genres)

    def react(self, song):
        likes = song.genre in self.liked
        if self.rng.random() < self.noise:
            likes = not likes
        return _reaction(song.song_id, 1.0 if likes else -1.0, "happy" if likes else "sad")
