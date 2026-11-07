"""Fake feedback: keyboard stands in for the cameras (Person A replaces this)."""
from __future__ import annotations
import time

from shared.schemas.messages import GroupReaction, PersonReaction


def _reaction(song_id: str, liking: float, emotion: str) -> GroupReaction:
    person = PersonReaction(person_id=0, emotion=emotion, liking=liking, confidence=1.0)
    return GroupReaction(time.time(), song_id, (person,), liking, {emotion: 1.0})


def get_group_reaction(song_id: str) -> GroupReaction | None:
    """Press l = like, d = dislike (then Enter). Returns None if the user quits (q)."""
    while True:
        key = input("  Reaction? [l]ike / [d]islike / [q]uit: ").strip().lower()
        if key == "l":
            return _reaction(song_id, 1.0, "happy")
        if key == "d":
            return _reaction(song_id, -1.0, "sad")
        if key == "q":
            return None
        print("  Please press l, d, or q.")
