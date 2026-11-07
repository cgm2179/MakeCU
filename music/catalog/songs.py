"""Stand-in for the filtered music database (Person C replaces this).

Each song has: genre, artist, energy (0-1, in audio_features) and mood.
"""
from __future__ import annotations

from shared.schemas.messages import Filter, Song


def _s(sid, title, artist, genre, energy, mood):
    return Song(sid, title, artist, genre, {"energy": energy}, mood)


SONGS = [
    _s("s1", "Levitating", "Dua Lipa", "pop", 0.8, "upbeat"),
    _s("s2", "Blinding Lights", "The Weeknd", "pop", 0.8, "upbeat"),
    _s("s3", "Someone Like You", "Adele", "pop", 0.3, "melancholy"),
    _s("s4", "Take Five", "Dave Brubeck", "jazz", 0.3, "calm"),
    _s("s5", "So What", "Miles Davis", "jazz", 0.3, "calm"),
    _s("s6", "Cantaloupe Island", "Herbie Hancock", "jazz", 0.6, "upbeat"),
    _s("s7", "Fly Me to the Moon", "Frank Sinatra", "jazz", 0.4, "dreamy"),
    _s("s8", "Lose Yourself", "Eminem", "hip-hop", 0.9, "aggressive"),
    _s("s9", "HUMBLE.", "Kendrick Lamar", "hip-hop", 0.8, "aggressive"),
    _s("s10", "Alright", "Kendrick Lamar", "hip-hop", 0.7, "upbeat"),
    _s("s11", "Clair de Lune", "Debussy", "classical", 0.1, "calm"),
    _s("s12", "Gymnopedie No.1", "Erik Satie", "classical", 0.1, "melancholy"),
    _s("s13", "Summer (Four Seasons)", "Vivaldi", "classical", 0.8, "upbeat"),
    _s("s14", "Mr. Brightside", "The Killers", "rock", 0.9, "upbeat"),
    _s("s15", "Creep", "Radiohead", "rock", 0.4, "melancholy"),
    _s("s16", "Hotel California", "Eagles", "rock", 0.5, "dreamy"),
]


def get_candidates(filter: Filter | None = None) -> list[Song]:
    """Return the candidate pool. The fake version ignores the filter."""
    return list(SONGS)
