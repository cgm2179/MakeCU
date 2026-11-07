"""Test aid: a mock emotion model whose faces react to the song that is playing.

Plugs into the real perception pipeline (tracker, smoothing, scoring) in place of DeepFace.
Two fake people; each independently likes the song when likes(song) is true (flipped with
probability `noise`), showing happy vs angry/sad faces. present=False means no faces in view.
"""
from __future__ import annotations

import random

from perception.emotion.camera_reaction import import_perception_cv

LIKED = {"happy": 0.80, "neutral": 0.12, "surprise": 0.08}
DISLIKED = {"angry": 0.45, "sad": 0.35, "neutral": 0.20}
RESTING = {"sad": 0.45, "neutral": 0.35, "angry": 0.20}  # like a real resting face: reads mildly negative


class TasteMockModel:
    def __init__(self, likes, n_faces=2, noise=0.1, seed=0):
        import_perception_cv()
        from perception_cv.emotion.base import Face

        self._Face = Face
        self.likes, self.n, self.noise = likes, n_faces, noise
        self.rng = random.Random(seed)
        self.person_likes = [False] * n_faces
        self.present = True
        self.resting = True  # until the first song starts
        self.resting_probs = dict(RESTING)  # tests can swap in a smiling face to trigger the guard

    def set_song(self, song, present=True) -> None:
        self.present = present
        self.resting = False
        liked = bool(self.likes(song))
        self.person_likes = [liked != (self.rng.random() < self.noise) for _ in range(self.n)]

    def predict(self, frame_bgr):
        if not self.present:
            return []
        h, w = frame_bgr.shape[:2]
        fw = int(w * 0.18)
        faces = []
        for i in range(self.n):
            cx = w * (0.3 + 0.4 * i / max(self.n - 1, 1))
            probs = dict(self.resting_probs if self.resting else LIKED if self.person_likes[i] else DISLIKED)
            faces.append(self._Face((int(cx - fw / 2), int(h * 0.4), fw, fw), probs, 0.9))
        return faces
