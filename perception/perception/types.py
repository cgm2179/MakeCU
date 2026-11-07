"""Mirrors the message shapes in AGENTS.md. Swap for shared/schemas imports once they exist."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field

EMOTIONS = ("happy", "sad", "neutral", "angry", "bored", "surprised")


@dataclass
class PersonReaction:
    person_id: int
    emotion: str
    liking: float        # [-1, 1]
    confidence: float    # [0, 1]


@dataclass
class GroupReaction:
    timestamp: float
    song_id: str
    people: list[PersonReaction] = field(default_factory=list)
    group_liking: float = 0.0
    group_emotion: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
