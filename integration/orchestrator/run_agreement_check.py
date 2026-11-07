"""Camera vs thumbs check (checkpoint P0.3): does your face agree with what you say you liked?

Plays short clips from the gym and party moods, reads the camera at the end of each clip, then asks
for your thumb. Saves one row per song to data/agreement/<user>.csv (git-ignored) and prints how often
the camera's sign matched your thumb. You can quit with q and resume later; rated songs are not repeated.

    python3 -m integration.orchestrator.run_agreement_check --user adi            # real webcam + audio
    python3 -m integration.orchestrator.run_agreement_check --demo                # scripted fake face/thumbs

Needs the camera libraries (see perception/perception_cv/requirements.txt) for anything but unit tests.
At each prompt type  u + Enter = liked,  d + Enter = disliked,  s + Enter = skip this song,  q + Enter = quit.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from rl import agreement as ag
from rl.agent.bandit import mean_confidence
from rl.agent.context import dominant_emotion

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "agreement"
MOODS = ("gym", "party")


def rated_ids(path, user):
    if not Path(path).exists():
        return set()
    return {r["song_id"] for r in ag.load_rows(path, user)}


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ag.FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_thumb(input_fn, prompt):
    """Returns 1 (up), 0 (down), 's' (skip) or 'q' (quit). Asks again on anything else."""
    while True:
        t = input_fn(prompt).strip().lower()
        if t in ("u", "up", "y", "1"):
            return 1
        if t in ("d", "down", "n", "0"):
            return 0
        if t in ("s", "q"):
            return t
        print("  Please type u, d, s or q.")


def collect(feedback, songs, user, out_csv, target=40, player=None, listen_seconds=10.0,
            input_fn=input, say=print):
    """Play each song, read the camera, ask for the thumb. Returns the number of songs rated this run."""
    done = rated_ids(out_csv, user)
    have = len(done)
    rated = 0
    for song in songs:
        if have + rated >= target:
            break
        if song.song_id in done:
            continue
        if player is not None and not player.play(song):
            continue  # no preview for this song; try another
        say(f"\n[{have + rated + 1}/{target}] {song.title} - {song.artist}")
        feedback.begin_song(song)
        if player is not None:
            player.wait()
        else:
            time.sleep(listen_seconds)
        reaction = feedback.get_group_reaction(song.song_id)
        conf = mean_confidence(reaction)
        liking = reaction.group_liking if reaction.people else None
        thumb = read_thumb(input_fn, "  Did you like it?  u = up, d = down, s = skip song, q = quit: ")
        if thumb == "q":
            break
        if thumb == "s":
            continue
        append_row(out_csv, {"user": user, "song_id": song.song_id, "thumb": thumb,
                             "liking": "" if liking is None else round(liking, 4),
                             "confidence": round(conf, 4), "emotion": dominant_emotion(reaction), "time": time.time()})
        rated += 1
        cam = "no face seen" if liking is None else f"camera {liking:+.2f}"
        say(f"  saved: thumb {'UP' if thumb else 'DOWN'}, {cam}")
    return rated


def build_songs(seed=0):
    from music.catalog.songs import get_candidates
    from shared.schemas.messages import Filter, FilterSource
    rng = random.Random(seed)
    lists = []
    for mood in MOODS:
        pool = list(get_candidates(Filter((FilterSource("event", mood),))))
        rng.shuffle(pool)
        lists.append(pool)
    out, seen = [], set()
    for i in range(max(len(p) for p in lists)):  # alternate gym, party, gym, ...
        for p in lists:
            if i < len(p) and p[i].song_id not in seen:
                seen.add(p[i].song_id)
                out.append(p[i])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="adi")
    ap.add_argument("--songs", type=int, default=40, help="total usable songs to collect (resumes)")
    ap.add_argument("--clip-seconds", type=float, default=10)
    ap.add_argument("--calibrate-seconds", type=float, default=8)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--detector", default="yunet")
    ap.add_argument("--window", type=float, default=3.0)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--demo", action="store_true", help="scripted fake face and thumbs, no camera or audio (writes to a demo file)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available
    from perception.emotion.camera_reaction import CameraFeedback
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    songs = build_songs(a.seed)
    user = a.user
    if a.demo:
        from perception.emotion.taste_mock_model import TasteMockModel
        liked = lambda s: s.audio_features["valence"] >= 0.6
        feedback = CameraFeedback(model=TasteMockModel(liked, seed=a.seed))
        rng = random.Random(a.seed)
        playing = {}
        begin = feedback.begin_song

        def begin_and_remember(song):
            playing["song"] = song
            begin(song)
        feedback.begin_song = begin_and_remember

        def input_fn(prompt):  # a fake person who likes happy songs and presses the wrong button 10% of the time
            return "u" if liked(playing["song"]) != (rng.random() < 0.1) else "d"
        user, player, listen = "demo", None, 1.0
        out_csv = DEFAULT_DIR / "demo.csv"
        out_csv.unlink(missing_ok=True)
    else:
        feedback = CameraFeedback(mock=False, camera_index=a.camera, detector=a.detector,
                                  calibrate_seconds=a.calibrate_seconds, read_window_seconds=a.window)
        input_fn, listen = input, a.clip_seconds
        player = None
        if not a.no_audio:
            from music.playback.audio import AudioPlayer
            player = AudioPlayer(clip_seconds=a.clip_seconds)
        out_csv = DEFAULT_DIR / f"{user}.csv"

    feedback.start()
    try:
        if not a.demo and a.calibrate_seconds > 0:
            feedback.calibrate(say=print)
        n = collect(feedback, songs, user, out_csv, target=a.songs, player=player,
                    listen_seconds=listen, input_fn=input_fn)
    finally:
        if player:
            player.close()
        feedback.stop()
    rows = ag.load_rows(out_csv, user)
    print(f"\nRated {n} songs this run; {len(rows)} saved in {out_csv}\n")
    if rows:
        print(ag.format_text(ag.analyse(rows)))
        print(f"\nNext: python3 -m rl.report --step 3 --agreement {out_csv} --user {user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
