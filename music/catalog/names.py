"""Normalising song titles and artist names so the same song matches across sources."""
from __future__ import annotations

import re


def clean_title(title: str) -> str:
    title = re.sub(r"[\(\[].*?[\)\]]", " ", title.lower())  # (feat. X), [Remix]
    title = re.sub(r"\s+-\s+.*$", "", title)  # " - Radio Edit", " - Remastered 2011"
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def clean_artist(artist: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", artist.lower()).strip()


def primary_artist(artist: str) -> str:
    """The first credited artist: 'Dua Lipa feat. X' / 'A, B' / 'A; B' / 'A x B' -> 'Dua Lipa'."""
    return re.split(r"\s*(?:;|,|\bfeat\b\.?|\bft\b\.?|\bfeaturing\b|\bx\b|\bwith\b)\s*", artist.strip(), maxsplit=1,
                    flags=re.IGNORECASE)[0].strip()


def artist_keys(artist: str) -> list[str]:
    """Normalised artist names to try, most specific first ('Simon & Garfunkel' before 'Simon')."""
    keys = [clean_artist(artist), clean_artist(primary_artist(artist))]
    return [k for i, k in enumerate(keys) if k and k not in keys[:i]]
