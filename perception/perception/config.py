"""Perception tuning knobs. MAX_PEOPLE should come from shared/config once it exists."""
from dataclasses import dataclass


@dataclass
class PerceptionConfig:
    MAX_PEOPLE: int = 6              # shared key (AGENTS.md), default 6
    STALE_SECONDS: float = 2.0       # ignore a phone that has not sent a frame this recently
    EMA_ALPHA: float = 0.4           # smoothing of per-person emotion probabilities
    TRACK_MAX_AGE_S: float = 2.0     # forget a face unseen for this long
    TRACK_MATCH_DIST: float = 0.20   # max centroid jump (fraction of frame) to keep an identity
    STILL_EPS: float = 0.03          # centroid movement below this counts as "still"
    BORED_AFTER_S: float = 10.0      # stillness duration at which neutral becomes fully "bored"
    ENABLE_BORED: bool = True        # "bored" is a stillness heuristic, not a trained class
    FRAMES_FOR_FULL_CONFIDENCE: int = 5
