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
import json
import random
import time
from pathlib import Path

from rl import agreement as ag
from rl.agent.bandit import mean_confidence
from rl.agent.context import dominant_emotion

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data" / "agreement"
MOODS = ("gym", "party")


def rated_ids(path, user):
    if not Path(path).exists():
        return set()
    return {r["song_id"] for r in ag.load_rows(path, user)}


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # an older file without the newer columns: rewrite it once with the full header
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ag.FIELDS:
                old = list(reader)
                with open(path, "w", newline="") as g:
                    w = csv.DictWriter(g, fieldnames=ag.FIELDS, restval="")
                    w.writeheader()
                    w.writerows(old)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ag.FIELDS, restval="")
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


def read_count(input_fn, prompt, n):
    """How many of n people liked the song: an integer 0..n, or 's' / 'q'."""
    while True:
        t = input_fn(prompt).strip().lower()
        if t in ("s", "q"):
            return t
        if t.isdigit() and 0 <= int(t) <= n:
            return int(t)
        print(f"  Please type a number from 0 to {n}, or s or q.")


def save_raw(raw_dir, song_id, thumb, raw):
    """One JSON per clip with the raw face samples (no images), so features can be recomputed later without a new session."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{song_id}.json").write_text(json.dumps({"song_id": song_id, "thumb": thumb, **raw}))


def collect(feedback, songs, user, out_csv, target=40, player=None, listen_seconds=10.0,
            input_fn=input, say=print, setting="quiet", group_size=1, preroll=0.0, raw_dir=None):
    """Play each song, read the camera, ask for the thumb. Returns the number of songs rated this run."""
    done = rated_ids(out_csv, user)
    have = len(done)
    rated = 0
    for song in songs:
        if have + rated >= target:
            break
        if song.song_id in done:
            continue
        if preroll > 0 and hasattr(feedback, "begin_preroll"):
            say("\n  ...relax, look ahead, hands off the keyboard: the song starts in a moment")
            feedback.begin_preroll()
            time.sleep(preroll)
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
        feats = feedback.clip_features() if hasattr(feedback, "clip_features") else None
        raw = feedback.clip_raw() if raw_dir and hasattr(feedback, "clip_raw") else None
        likes = ""
        if group_size > 1:
            likes = read_count(input_fn, f"  How many of the {group_size} people liked it?  0-{group_size}, s = skip song, q = quit: ", group_size)
            thumb = likes if likes in ("s", "q") else (1 if 2 * likes > group_size else 0 if 2 * likes < group_size else "tie")
            if thumb == "tie":
                say("  an even split: not saved (it says nothing either way)")
                continue
        else:
            thumb = read_thumb(input_fn, "  Did you like it?  u = up, d = down, s = skip song, q = quit: ")
        if thumb == "q":
            break
        if thumb == "s":
            continue
        append_row(out_csv, {"user": user, "song_id": song.song_id, "thumb": thumb,
                             "liking": "" if liking is None else round(liking, 4),
                             "confidence": round(conf, 4), "emotion": dominant_emotion(reaction), "time": time.time(),
                             "features": json.dumps({k: round(v, 5) for k, v in feats.items()}) if feats else "",
                             "setting": setting, "group_size": group_size, "likes": likes})
        if raw is not None:
            save_raw(raw_dir, song.song_id, thumb, raw)
        rated += 1
        cam = "no face seen" if liking is None else f"camera {liking:+.2f}"
        say(f"  saved: thumb {'UP' if thumb else 'DOWN'}, {cam}")
    return rated


def smart_order(songs, rated_rows, by_id, seed=0):
    """Order unrated songs so the camera test sees BOTH liked and disliked songs.

    Step 3 found only 4 liked songs out of 39 (the gym and party picks were mostly not Adi's taste), which
    leaves nothing to test the camera on. So: train the bandit on the thumbs so far, then alternate
    its top-ranked songs (likely liked) with random ones (likely disliked, and a check on the model)."""
    from rl.eval import BanditModel
    rated = {r["song_id"] for r in rated_rows}
    train = [(by_id[r["song_id"]], r["thumb"]) for r in rated_rows if r["song_id"] in by_id]
    fresh = [x for x in songs if x.song_id not in rated]
    if not train or not any(y for _, y in train):
        return fresh
    model = BanditModel(seed)
    model.fit(train)
    best = sorted(fresh, key=lambda x: -model.score(x))
    rest = list(fresh)
    random.Random(seed).shuffle(rest)
    out, seen = [], set()
    for a, b in zip(best, rest):
        for x in (a, b):
            if x.song_id not in seen:
                seen.add(x.song_id)
                out.append(x)
    return out


def ft_order(songs, layer, seed=0):
    """Order songs to alternate the fine-tuned model's top picks with random ones, so a session gets both liked and disliked songs
    (used for the camera replication, where at least 12 of each thumb are needed)."""
    from rl.agent.context import Context
    best = sorted(songs, key=lambda x: -layer.logit(x))
    rest = list(songs)
    random.Random(seed).shuffle(rest)
    out, seen = [], set()
    for a, b in zip(best, rest):
        for x in (a, b):
            if x.song_id not in seen:
                seen.add(x.song_id)
                out.append(x)
    return out


def build_songs(seed=0, moods=MOODS):
    from music.catalog.songs import get_candidates
    from shared.schemas.messages import Filter, FilterSource
    rng = random.Random(seed)
    lists = []
    for mood in moods:
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
    ap.add_argument("--moods", default=",".join(MOODS),
                    help="comma-separated moods to draw songs from (gym, party, chill, happy, sad, study)")
    ap.add_argument("--smart", action="store_true",
                    help="use your earlier thumbs to pick songs you are likely to like (half) and random ones (half)")
    ap.add_argument("--setting", choices=("quiet", "busy"), default="quiet",
                    help="tag this session: quiet room alone, or busy (several people, music up, movement)")
    ap.add_argument("--group-size", type=int, default=1,
                    help="people in front of the camera; with more than 1 you enter how many liked each song")
    ap.add_argument("--exclude-seen", metavar="SUBJECT", default=None,
                    help="skip every song SUBJECT has already rated or thumbed (ratings, camera sessions, live sessions)")
    ap.add_argument("--rank-with", metavar="SUBJECT", default=None,
                    help="order songs to alternate SUBJECT's fine-tuned model's top picks with random ones (gets both thumbs)")
    ap.add_argument("--preroll", type=float, default=0.0,
                    help="seconds of relaxed face recorded before each song (its baseline); 4 is recommended. 0 = use the session calibration only")
    ap.add_argument("--raw-dir", default=None,
                    help="save the raw face samples of every clip here (default data/camera_raw/<user> when --preroll is on)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--demo", action="store_true", help="scripted fake face and thumbs, no camera or audio (writes to a demo file)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available
    from perception.emotion.camera_reaction import CameraFeedback
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    moods = tuple(m.strip() for m in a.moods.split(",") if m.strip())
    songs = build_songs(a.seed, moods)
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

    if a.smart and not a.demo:
        from music.catalog.spotify_catalog import load_spotify
        prior = ag.load_rows(out_csv, user) if Path(out_csv).exists() else []
        by_id = {x.song_id: x for x in load_spotify()}
        songs = smart_order(songs, prior, by_id, a.seed)
        print(f"Smart picking from {len(prior)} earlier thumbs ({sum(r['thumb'] for r in prior)} liked).")
    if a.exclude_seen:
        from integration.orchestrator.run_live_session import seen_song_ids
        seen = seen_song_ids(a.exclude_seen)
        before = len(songs)
        songs = [x for x in songs if x.song_id not in seen]
        print(f"Excluding {before - len(songs)} songs {a.exclude_seen} has already rated or thumbed; {len(songs)} left.")
    if a.rank_with:
        from rl import base_lastfm as bl
        from rl import personal_layer as pl
        try:
            base = bl.load_base(bl.MODEL_DIR / "base_model_v2.json")
            layer = pl.load_personal(pl.PROFILE_DIR / f"{a.rank_with}_personal_v2.json", base)
        except (OSError, ValueError) as e:
            raise SystemExit(f"{e}\nNeeds models/base_model_v2.json and data/profiles/{a.rank_with}_personal_v2.json")
        songs = ft_order(songs, layer, a.seed)
        print(f"Songs ordered to alternate {a.rank_with}'s model's top picks with random ones.")
    feedback.start()
    try:
        if not a.demo and a.calibrate_seconds > 0:
            feedback.calibrate(say=print)
        raw_dir = a.raw_dir or (str(ROOT / "data" / "camera_raw" / user) if a.preroll > 0 and not a.demo else None)
        n = collect(feedback, songs, user, out_csv, target=a.songs, player=player,
                    listen_seconds=listen, input_fn=input_fn, setting=a.setting, group_size=a.group_size,
                    preroll=0.0 if a.demo else a.preroll, raw_dir=raw_dir)
    finally:
        if player:
            player.close()
        feedback.stop()
    rows = ag.load_rows(out_csv, user)
    print(f"\nRated {n} songs this run; {len(rows)} saved in {out_csv}\n")
    if rows:
        print(ag.format_text(ag.analyse(rows)))
        if a.exclude_seen:
            print(f"\nNext (replication): python3 -m rl.camera_replication --train {a.exclude_seen} --new {user}")
        else:
            print(f"\nNext: python3 -m rl.report --step 3 --agreement {out_csv} --user {user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
