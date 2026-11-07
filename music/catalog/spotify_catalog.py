"""Loads the downloaded Spotify Tracks CSV into our Song schema (offline).

Column mapping:
    track_id     -> song_id          track_name -> title
    artists      -> artist (first artist)        track_genre -> genre
    valence, energy, danceability, tempo, acousticness, instrumentalness, popularity
                 -> audio_features
    mood         -> derived from the valence/energy quadrant (see mood_from_audio)
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from shared.schemas.messages import Song

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "spotify" / "dataset.csv"
FEATURE_COLUMNS = ("valence", "energy", "danceability", "tempo", "acousticness",
                   "instrumentalness", "popularity")

_cache: dict = {}


def csv_path() -> Path:
    return Path(os.environ.get("MAKECU_SPOTIFY_CSV", DEFAULT_CSV))


def dataset_available() -> bool:
    return csv_path().exists()


def mood_from_audio(valence: float, energy: float) -> str:
    """Coarse mood label from the valence/energy quadrant."""
    if energy >= 0.5:
        return "upbeat" if valence >= 0.5 else "aggressive"
    return "calm" if valence >= 0.5 else "melancholy"


def load_spotify(path=None) -> list[Song]:
    """Read the CSV once (cached). Drops rows missing a title/artist or with tempo 0,
    and duplicate tracks (the same track appears under several genres)."""
    path = Path(path) if path else csv_path()
    if path in _cache:
        return _cache[path]
    songs, seen_ids, seen_names = [], set(), set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            title, artists = row["track_name"].strip(), row["artists"].strip()
            if not title or not artists or row["track_id"] in seen_ids:
                continue
            try:
                feats = {c: float(row[c]) for c in FEATURE_COLUMNS}
            except ValueError:
                continue
            if feats["tempo"] <= 0:
                continue
            artist = artists.split(";")[0].strip()
            name_key = (title.lower(), artist.lower())
            if name_key in seen_names:  # same song, different album/release
                continue
            seen_ids.add(row["track_id"])
            seen_names.add(name_key)
            songs.append(Song(row["track_id"], title, artist, row["track_genre"], feats,
                              mood_from_audio(feats["valence"], feats["energy"])))
    _cache[path] = songs
    return songs
