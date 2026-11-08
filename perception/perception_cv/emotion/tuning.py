"""Editable filters for DeepFace's raw emotion probabilities.

Probabilities and minimums are fractions from 0.0 to 1.0. Disabled emotions
are removed before normalization; predictions below their minimum are removed
as well. If no enabled emotion passes its minimum, that face is omitted.
"""
from __future__ import annotations

ENABLED_RAW_EMOTIONS = {
    "happy",
    "sad",
    "neutral",
    # "angry",
    # "disgust",
    # "fear",
    # "surprise",
}

MIN_PROBABILITY = {
    "happy": 0.0,
    "sad": 0.0,
    "neutral": 0.0,
    # "angry": 0.0,
    # "disgust": 0.0,
    # "fear": 0.0,
    # "surprise": 0.0,
}


def tune_probabilities(raw: dict[str, float]) -> dict[str, float]:
    """Keep enabled classes above their configured minimum and renormalize."""
    kept = {
        emotion: probability
        for emotion, probability in raw.items()
        if emotion in ENABLED_RAW_EMOTIONS
        and probability >= MIN_PROBABILITY.get(emotion, 0.0)
    }
    total = sum(kept.values())
    if total <= 0.0:
        return {}
    return {emotion: probability / total for emotion, probability in kept.items()}
