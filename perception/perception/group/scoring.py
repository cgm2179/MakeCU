from __future__ import annotations
from ..config import PerceptionConfig
from ..types import EMOTIONS, PersonReaction, GroupReaction
from .tracker import Track

# How much each emotion says about "I like this song". Tune these.
LIKING_WEIGHTS = {"happy": 1.0, "surprised": 0.4, "neutral": 0.0, "bored": -0.3, "sad": -0.6, "angry": -1.0}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def person_reaction(track: Track, person_id: int, now: float, cfg: PerceptionConfig) -> PersonReaction:
    probs = dict(track.probs)
    probs["bored"] = 0.0
    if cfg.ENABLE_BORED:
        frac = _clamp((now - track.still_since) / cfg.BORED_AFTER_S, 0.0, 1.0)
        shift = probs["neutral"] * frac
        probs["neutral"] -= shift
        probs["bored"] = shift
    emotion = max(probs, key=probs.get)
    liking = _clamp(sum(probs[k] * LIKING_WEIGHTS[k] for k in probs), -1.0, 1.0)
    stability = min(1.0, track.frames / cfg.FRAMES_FOR_FULL_CONFIDENCE)
    confidence = _clamp(track.face_conf * stability, 0.0, 1.0)
    return PersonReaction(person_id, emotion, liking, confidence)


def aggregate(song_id: str, people: list[PersonReaction], now: float, max_people: int) -> GroupReaction:
    people = sorted(people, key=lambda p: p.confidence, reverse=True)[:max_people]
    if not people:
        # nobody visible: neutral, and reward = group_liking * mean(confidence) will be 0 downstream
        return GroupReaction(now, song_id, [], 0.0, {e: float(e == "neutral") for e in EMOTIONS})
    n = len(people)
    group_liking = sum(p.liking for p in people) / n
    shares = {e: 0.0 for e in EMOTIONS}
    for p in people:
        shares[p.emotion] += 1.0 / n
    return GroupReaction(now, song_id, people, group_liking, shares)
