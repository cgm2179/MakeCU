from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

# Raw classes produced by DeepFace / FER-style models.
RAW_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


@dataclass
class Face:
    box: tuple[int, int, int, int]   # x, y, w, h in pixels
    probs: dict[str, float]          # keys from RAW_EMOTIONS, sums to 1
    face_confidence: float           # detector confidence in [0, 1]


class EmotionModel(Protocol):
    def predict(self, frame_bgr: np.ndarray) -> list[Face]:
        """Detect every face in a BGR frame and return an emotion distribution for each."""
        ...
