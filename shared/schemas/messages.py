"""Shared message types. Names follow AGENTS.md."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Song:
    song_id: str
    title: str
    artist: str
    genre: str
    audio_features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FilterSource:
    source_type: str  # "event" | "random" | "playlist" | "artists" | "themes"
    source_value: str = ""


@dataclass(frozen=True)
class Filter:
    sources: tuple = ()


@dataclass(frozen=True)
class PersonReaction:
    person_id: int
    emotion: str
    liking: float  # [-1, 1]
    confidence: float  # [0, 1]


@dataclass(frozen=True)
class GroupReaction:
    timestamp: float
    song_id: str
    people: tuple
    group_liking: float  # [-1, 1]
    group_emotion: dict = field(default_factory=dict)
