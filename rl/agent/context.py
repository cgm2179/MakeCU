"""Bandit context = the first filter + the user's current emotion."""
from __future__ import annotations

from dataclasses import dataclass

from shared.schemas.messages import Filter, GroupReaction


@dataclass(frozen=True)
class Context:
    filter_key: str = "none"
    emotion: str = "neutral"


def filter_key(flt: Filter | None) -> str:
    """Turn a Filter into a stable string, e.g. 'event:party'."""
    if flt is None or not flt.sources:
        return "none"
    return "+".join(sorted(f"{s.source_type}:{s.source_value}" for s in flt.sources))


def dominant_emotion(reaction: GroupReaction | None) -> str:
    """The most common emotion in a reaction (the user's current emotional state)."""
    if reaction is None or not reaction.group_emotion:
        return "neutral"
    return max(reaction.group_emotion.items(), key=lambda kv: kv[1])[0]
