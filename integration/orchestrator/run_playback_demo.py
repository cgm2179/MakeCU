"""Live demo: the bandit picks songs for a mood and their iTunes previews actually play.

    python3 -m integration.orchestrator.run_playback_demo --mood gym --songs 3 --clip-seconds 10

Needs internet (iTunes Search API, no key) and the Spotify CSV (see README). Songs with
no matching preview are skipped. A fake listener likes/dislikes after each clip.
"""
from __future__ import annotations

import argparse

from integration.orchestrator.run_loop import run
from music.catalog.songs import get_candidates
from music.catalog.spotify_catalog import dataset_available
from music.playback.audio import AudioPlayer
from rl.env.taste_listener import TasteListener
from shared.schemas.messages import Filter, FilterSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mood", default="gym")
    ap.add_argument("--songs", type=int, default=3, help="how many songs to actually play")
    ap.add_argument("--clip-seconds", type=float, default=10, help="play only the first N seconds (0 = full 30 s)")
    ap.add_argument("--likes", default="", help="genres the fake listener likes")
    ap.add_argument("--mainstream-bias", type=float, default=None, metavar="0..1",
                    help="lean toward well-known songs on the database path (default 0.5; 0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")

    pool = get_candidates(Filter((FilterSource("event", a.mood),)), mainstream_bias=a.mainstream_bias)
    player = AudioPlayer(clip_seconds=a.clip_seconds or None)
    try:
        likes = [g for g in a.likes.split(",") if g] or ["pop"]
        out = run(a.songs, event=a.mood, seed=a.seed, pool=pool, player=player, wait_for_playback=True,
                  listener=TasteListener(likes, seed=a.seed), overrides={}, round_delay=0, mainstream_bias=a.mainstream_bias)
    finally:
        player.close()
    by_id = {x.song_id: x for x in pool}
    print("\nPlayed in order: " + " -> ".join(f"{by_id[i].title} ({by_id[i].artist})"
                                              for i in out["controller"].history))


if __name__ == "__main__":
    main()
