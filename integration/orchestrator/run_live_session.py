"""Live session (checkpoint L3.2): play songs from three hidden conditions and thumb each one.

    python3 -m integration.orchestrator.run_live_session --user adi

Each song starts playing at once and asks for your thumb straight away. The moment you answer, it stops and the next
song starts, so you listen only as long as you need. u = liked, d = disliked, s = skip this song, q = quit (resumes).
You are not told which condition (random / most popular / your personal model) a song came from.
Needs a saved profile: python3 -m rl.personalize --user adi --include-heldout
"""
from __future__ import annotations

import argparse
import random
import time

from rl import live_test as lt

PROMPT = "  Did you like it?  u = up, d = down, s = skip song, q = quit: "


def read_thumb(input_fn, prompt=PROMPT):
    while True:
        t = input_fn(prompt).strip().lower()
        if t in ("u", "up", "y", "1"):
            return 1
        if t in ("d", "down", "n", "0"):
            return 0
        if t in ("s", "q"):
            return t
        print("  Please type u, d, s or q.")


def run_session(pool, agent, user, csv_path, player=None, input_fn=input, say=print, per_condition=lt.PER_CONDITION,
                seed=0, now=time.time):
    """Returns the number of songs rated this run. Condition for each song comes from a fixed random sequence."""
    rows = lt.load_rows(csv_path, user)
    sequence = lt.make_sequence(per_condition, seed)
    used = {r["song_id"] for r in rows}
    artists = set()
    remaining = [s for s in pool if s.song_id not in used]
    rng = random.Random(seed + len(rows))
    rated = 0
    while len(rows) + rated < len(sequence):
        seq = len(rows) + rated
        condition = sequence[seq]
        candidates = [s for s in remaining if s.artist not in artists]
        song = lt.pick_song(condition, candidates, agent, rng)
        if song is None:
            say("No songs left in the pool.")
            break
        remaining.remove(song)
        if player is not None and not player.play(song):
            continue                      # no preview for this song; try another for the same slot
        say(f"\n[{seq + 1}/{len(sequence)}] {song.title} - {song.artist}")
        try:
            thumb = read_thumb(input_fn)
        finally:
            if player is not None:
                player.stop()
        if thumb == "q":
            say("Quit. Your progress is saved; run the same command to resume.")
            break
        if thumb == "s":
            say("  skipped (not saved)")
            continue
        artists.add(song.artist)
        lt.append_row(csv_path, {"user": user, "seq": seq, "condition": condition, "song_id": song.song_id,
                                 "thumb": thumb, "time": now()})
        rated += 1
    return rated


def build_pool(moods, rated_ids):
    from music.catalog.songs import get_candidates
    from shared.schemas.messages import Filter, FilterSource
    seen, pool = set(rated_ids), []
    for mood in moods:
        for s in get_candidates(Filter((FilterSource("event", mood),)), pool_size=500):
            if s.song_id not in seen:
                seen.add(s.song_id)
                pool.append(s)
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--per-condition", type=int, default=lt.PER_CONDITION)
    ap.add_argument("--moods", default="gym,party,chill,happy,sad,study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available
    from rl import ratings as rt
    from rl.agent.profile import ProfileError, load_profile
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    try:
        agent = load_profile(a.user)
    except ProfileError as e:
        raise SystemExit(f"{e}\nCreate one first: python3 -m rl.personalize --user {a.user} --include-heldout")
    rated_ids = {r["song_id"] for r in rt.load_rows(rt.ratings_path(a.user), a.user)}
    pool = build_pool([m.strip() for m in a.moods.split(",") if m.strip()], rated_ids)
    print(f"{len(pool)} candidate songs (none you have rated before). Conditions are hidden.")
    player = None
    if not a.no_audio:
        from music.playback.audio import AudioPlayer
        player = AudioPlayer(clip_seconds=30)
    path = lt.live_path(a.user)
    try:
        n = run_session(pool, agent, a.user, path, player, per_condition=a.per_condition, seed=a.seed)
    finally:
        if player:
            player.close()
    rows = lt.load_rows(path, a.user)
    print(f"\nRated {n} songs this run; {len(rows)} saved in {path}\n")
    if rows:
        print(lt.format_text(lt.analyse(rows)))
        print(f"\nNext: python3 -m rl.live_test --user {a.user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
