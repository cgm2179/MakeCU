"""One-time download of the Spotify Tracks Dataset (maharshipandya, ~114k tracks, 20 MB).

    python3 -m music.catalog.download_spotify

Saved to data/spotify/dataset.csv (git-ignored). After that everything runs offline.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset"
       "/resolve/main/dataset.csv")
DEST = Path(__file__).resolve().parents[2] / "data" / "spotify" / "dataset.csv"

if __name__ == "__main__":
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
    else:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, DEST)
        print(f"Saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
