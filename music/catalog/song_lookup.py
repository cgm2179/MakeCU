"""Resolve a (title, artist) to a Song with features, for songs that may not be in the dataset.

Order: 1) the Spotify dataset (fast path), 2) the on-disk cache, 3) Gemini (only for a song in
neither). Gemini results, including "not identified", are cached so a song is never looked up twice.
Failures of the call itself (network, rate limit, no key) are NOT cached, so a retry can work.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from music.catalog.gemini_features import GeminiClient, GeminiError
from music.catalog.names import artist_keys, clean_title
from shared.schemas.messages import CandidatePool, Song

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "gemini_cache.json"
FUZZY_TITLE_RATIO = 0.9


@dataclass
class LookupResult:
    song: Song | None
    source: str  # "dataset" | "cache" | "gemini" | "unidentified" | "error" | "gemini-disabled"
    reason: str = ""


def _key(title: str, artist: str) -> str:
    return f"{clean_title(title)}|{artist_keys(artist)[-1] if artist_keys(artist) else ''}"


def song_from_features(title: str, artist: str, f: dict) -> Song:
    feats = {k: f[k] for k in ("valence", "energy", "danceability", "tempo") if k in f}
    sid = "gem:" + hashlib.sha1(_key(title, artist).encode()).hexdigest()[:12]
    return Song(sid, title, artist, f["genre"], feats, f["mood"])


class SongLookup:
    def __init__(self, gemini: GeminiClient | None = None, cache_path: Path | str | None = DEFAULT_CACHE,
                 dataset: list[Song] | None = None):
        self.gemini = gemini
        self.cache_path = Path(cache_path) if cache_path else None
        self._dataset = dataset
        self._index: dict | None = None
        self._cache: dict | None = None

    # ---- dataset (fast path) ----
    def _build_index(self) -> dict:
        if self._index is None:
            songs = self._dataset
            if songs is None:
                from music.catalog.spotify_catalog import dataset_available, load_spotify

                songs = load_spotify() if dataset_available() else []
            exact, by_artist = {}, {}
            for s in songs:
                for a in artist_keys(s.artist)[:1]:
                    exact.setdefault((clean_title(s.title), a), s)
                    by_artist.setdefault(a, []).append((clean_title(s.title), s))
            self._index = {"exact": exact, "by_artist": by_artist}
        return self._index

    def find_in_dataset(self, title: str, artist: str) -> Song | None:
        idx, t = self._build_index(), clean_title(title)
        for a in artist_keys(artist):
            if (t, a) in idx["exact"]:
                return idx["exact"][(t, a)]
        for a in artist_keys(artist):  # same artist, near-identical title ("Antihero" vs "Anti-Hero")
            titles = idx["by_artist"].get(a, [])
            best = max(titles, key=lambda x: difflib.SequenceMatcher(None, t, x[0]).ratio(), default=None)
            if best and difflib.SequenceMatcher(None, t, best[0]).ratio() >= FUZZY_TITLE_RATIO:
                return best[1]
        return None

    # ---- cache ----
    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
            if self.cache_path and self.cache_path.exists():
                try:
                    self._cache = json.loads(self.cache_path.read_text())
                except (OSError, ValueError):
                    self._cache = {}  # unreadable cache: start fresh rather than crash
        return self._cache

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.cache_path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(self._cache, fh, indent=1, sort_keys=True)
        os.replace(tmp, self.cache_path)

    # ---- the public call ----
    def lookup(self, title: str, artist: str) -> LookupResult:
        hit = self.find_in_dataset(title, artist)
        if hit is not None:
            return LookupResult(hit, "dataset")
        cache, key = self._load_cache(), _key(title, artist)
        if key in cache:
            entry = cache[key]
            if entry is None:
                return LookupResult(None, "unidentified", "cached: Gemini could not identify this song")
            return LookupResult(song_from_features(title, artist, entry), "cache")
        if self.gemini is None:
            return LookupResult(None, "gemini-disabled", "not in the dataset and no Gemini client is configured")
        try:
            features = self.gemini.lookup(title, artist)
        except GeminiError as err:
            return LookupResult(None, "error", str(err))  # not cached: may work next time
        # None is cached too: don't ask again about a song Gemini doesn't know
        cache[key] = dict(features, fetched=int(time.time()), model=self.gemini.model) if features else None
        self._save_cache()
        if features is None:
            return LookupResult(None, "unidentified", "Gemini could not identify this song")
        return LookupResult(song_from_features(title, artist, features), "gemini")

    def lookup_many(self, pairs) -> list[LookupResult]:
        return [self.lookup(t, a) for t, a in pairs]


def extend_pool(pool, songs) -> CandidatePool:
    """Add looked-up songs to a candidate pool (skipping ones already there), keeping its source tag."""
    out = CandidatePool(pool, getattr(pool, "source", "unknown"))
    have = {s.song_id for s in out}
    for s in songs:
        if s is not None and s.song_id not in have:
            out.append(s)
            have.add(s.song_id)
    return out
