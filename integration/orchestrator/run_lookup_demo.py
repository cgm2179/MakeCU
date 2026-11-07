"""Look up songs that may not be in the 80k dataset, using Gemini for the ones that are not.

    export GEMINI_API_KEY=...        # or put GEMINI_API_KEY=... in a .env file in the repo root
    python3 -m integration.orchestrator.run_lookup_demo
    python3 -m integration.orchestrator.run_lookup_demo --songs "Billie Jean|Michael Jackson" "Flowers|Miley Cyrus"

For each song: where the answer came from (dataset / cache / gemini), the features, the feature
strings the bandit learns from, and how well it fits each mood filter (distance to the mood's
target; lower is closer). Results are cached in data/gemini_cache.json, so reruns make no API calls.
"""
from __future__ import annotations

import argparse
import sys

from music.catalog.gemini_features import GeminiClient, GeminiError, GeminiNotConfigured
from music.catalog.song_lookup import DEFAULT_CACHE, SongLookup
from music.filter.mood_targets import PROFILES, distance
from rl.agent.bandit import features

DEFAULT_SONGS = [
    ("Levitating", "Dua Lipa"),                      # control: in the dataset
    ("Billie Jean", "Michael Jackson"),              # classic not in the dataset
    ("Anti-Hero", "Taylor Swift"),                   # 2022
    ("Flowers", "Miley Cyrus"),                      # 2023
    ("Not Like Us", "Kendrick Lamar"),               # 2024
    ("Espresso", "Sabrina Carpenter"),               # 2024
    ("Zzyzx Quibble Marmalade", "The Fictional Hedgehogs"),  # control: should not be identified
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", metavar="'Title|Artist'", help="songs to look up (default: a built-in list)")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write data/gemini_cache.json")
    a = ap.parse_args()
    songs = [tuple(x.split("|", 1)) for x in a.songs] if a.songs else DEFAULT_SONGS

    try:
        gemini = GeminiClient()
        print(f"Gemini model: {gemini.model}")
    except GeminiNotConfigured as err:
        gemini = None
        print(f"NOTE: {err}\n      Songs in the dataset still resolve; the rest will be reported as not looked up.\n")
    lookup = SongLookup(gemini, cache_path=None if a.no_cache else DEFAULT_CACHE)

    counts = {}
    for title, artist in songs:
        r = lookup.lookup(title, artist)
        counts[r.source] = counts.get(r.source, 0) + 1
        print(f"\n{title} - {artist}\n  source: {r.source}" + (f"  ({r.reason})" if r.reason else ""))
        if r.song is None:
            print("  -> skipped")
            continue
        s = r.song
        af = {k: round(v, 2) for k, v in s.audio_features.items()}
        print(f"  genre={s.genre}  mood={s.mood}  features={af}")
        print(f"  bandit features: {features(s)}")
        fit = sorted((distance(s, p), m) for m, p in PROFILES.items())
        print("  mood-filter fit (distance, lower = closer): " + ", ".join(f"{m} {d:.2f}" for d, m in fit[:4]))
    print(f"\nSummary: {counts}; Gemini API calls this run: {gemini.calls if gemini else 0}")
    if gemini is None and counts.get("gemini-disabled"):
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except GeminiError as err:
        raise SystemExit(f"Gemini error: {err}")
