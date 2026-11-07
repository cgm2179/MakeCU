"""Stand-in for the filtered music database (Person C replaces this)."""
from __future__ import annotations
from shared.schemas.messages import Filter, Song

SONGS = [
    Song("s1", "Levitating", "Dua Lipa", "pop", {"energy": 0.8}),
    Song("s2", "Blinding Lights", "The Weeknd", "pop", {"energy": 0.8}),
    Song("s3", "Take Five", "Dave Brubeck", "jazz", {"energy": 0.3}),
    Song("s4", "So What", "Miles Davis", "jazz", {"energy": 0.3}),
    Song("s5", "Lose Yourself", "Eminem", "hip-hop", {"energy": 0.9}),
    Song("s6", "HUMBLE.", "Kendrick Lamar", "hip-hop", {"energy": 0.8}),
    Song("s7", "Clair de Lune", "Debussy", "classical", {"energy": 0.1}),
    Song("s8", "Gymnopedie No.1", "Erik Satie", "classical", {"energy": 0.1}),
]


def get_candidates(filter: Filter | None = None) -> list[Song]:
    """Return the candidate pool. The fake version ignores the filter."""
    return list(SONGS)
