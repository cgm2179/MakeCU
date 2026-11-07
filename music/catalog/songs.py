"""Stand-in for the filtered music database (Person C replaces this).

Each song has: genre, artist, energy (0-1, in audio_features) and mood.
"""
from __future__ import annotations

import random

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


def get_candidates(filter: Filter | None = None, pool_size: int = 500) -> list[Song]:
    """Return the candidate pool for the first filter.

    Uses the downloaded Spotify dataset when data/spotify/dataset.csv exists
    (see music/catalog/download_spotify.py); otherwise falls back to the hardcoded
    SONGS above (filter ignored). Supported filter sources so far:
      event  : a mood/activity name from music/filter/mood_targets.py (e.g. "gym")
      random : a random sample
    Other source types (playlist, artists, themes) are not implemented yet.
    """
    from music.catalog.spotify_catalog import dataset_available, load_spotify

    if not dataset_available():
        return list(SONGS)
    songs = load_spotify()
    for src in (filter.sources if filter else ()):
        if src.source_type == "event":
            from music.filter.mood_targets import profile_for, select_pool
            profile = profile_for(src.source_value)
            if profile:
                return select_pool(songs, profile, pool_size)
    return random.Random().sample(songs, pool_size)  # "random" or no usable filter
