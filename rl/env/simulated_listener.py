"""Simulated listener so the loop can run with no keyboard (for tests).

Taste depends on the listener's emotion, which drifts over time, so a
context-aware agent has something to learn:
  - always likes jazz
  - happy: also likes high-energy songs
  - sad:   dislikes high-energy songs, likes calm ones
"""
import random

from perception.emotion.fake_reaction import _reaction

EMOTIONS = ("happy", "sad", "neutral")


class SimulatedListener:
    def __init__(self, liked_genres=("jazz",), seed=0, stay=0.8, noise=0.1):
        self.liked = set(liked_genres)
        self.rng = random.Random(seed)
        self.stay, self.noise = stay, noise
        self.emotion = "neutral"

    def taste(self, song, emotion) -> float:
        energy = song.audio_features.get("energy", 0.5)
        score = 1.0 if song.genre in self.liked else 0.0
        if emotion == "happy" and energy >= 0.7:
            score += 1.0
        if emotion == "sad" and energy >= 0.7:
            score -= 1.0
        if emotion == "sad" and song.mood == "calm":
            score += 1.0
        return score

    def react(self, song):
        emotion = self.emotion
        likes = self.taste(song, emotion) > 0
        if self.rng.random() < self.noise:
            likes = not likes
        if self.rng.random() > self.stay:  # emotion drifts for next time
            self.emotion = self.rng.choice(EMOTIONS)
        return _reaction(song.song_id, 1.0 if likes else -1.0, emotion)
