"""Rate songs on two questions (checkpoint L1.2). Step 4 of docs/ADI_CHECKLIST.txt.

    python3 -m integration.orchestrator.run_rating_session --user adi            # rate until 150 songs (resumes)
    python3 -m integration.orchestrator.run_rating_session --user adi --retest   # Step 5: re-rate 20 songs, a day later

Each song plays as a 30-second iTunes preview. After --listen seconds (default 15) you are asked, while it keeps
playing, TWO different questions, each 1 to 5:
   QUALITY     "Ignoring my own taste, how well-made or widely appreciated is this song?"
   PREFERENCE  "How much do I personally want to hear this?"
Keys: 1-5 answer, r = replay the clip, s = skip this song, q = quit (progress is saved, run again to resume).

Songs are spread across mood, popularity (unknown songs and hits) and genre. Every 5th rated song is held out as
the final exam (30 of 150); the tool never shows you which. Saved to data/ratings/<user>.csv (git-ignored).
Song titles and artists go to Apple (to find the preview); your ratings never leave this computer.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from rl import ratings as rt

QUALITY_Q = "QUALITY    - ignoring your own taste, how well-made or widely appreciated is this song? (1 poor .. 5 excellent)"
PREFERENCE_Q = "PREFERENCE - how much do YOU personally want to hear this? (1 not at all .. 5 a lot)"
GENRE_CAP = 6   # at most this many songs of one genre in the plan, so one genre cannot dominate


def build_plan(catalog, n=400, seed=0, genre_cap=GENRE_CAP):
    """Songs in the order they will be offered, spread evenly over mood x popularity band, one per artist,
    at most genre_cap per genre. More than 150 are returned because some songs have no playable preview."""
    rng = random.Random(seed)
    cells = {}
    for s in catalog:
        pop = s.audio_features.get("popularity")
        if pop is None or not s.mood:
            continue
        cells.setdefault((s.mood, rt.pop_band(pop)), []).append(s)
    for pool in cells.values():
        rng.shuffle(pool)
    order = sorted(cells)
    plan, artists, genres = [], set(), {}
    while len(plan) < n and any(cells.values()):
        rng.shuffle(order)
        for cell in order:
            pool = cells[cell]
            while pool:
                s = pool.pop()
                if s.artist in artists or genres.get(s.genre, 0) >= genre_cap:
                    continue
                artists.add(s.artist)
                genres[s.genre] = genres.get(s.genre, 0) + 1
                plan.append(s)
                break
            if len(plan) >= n:
                break
    return plan


def ask(input_fn, prompt, keys="12345"):
    """A rating 1-5 (as int), or 'r' replay, 's' skip, 'q' quit. Asks again on anything else."""
    while True:
        t = input_fn(prompt).strip().lower()
        if t in tuple(keys):
            return int(t)
        if t in ("r", "s", "q"):
            return t
        print("  Please type 1-5, or r (replay), s (skip), q (quit).")


def _rate_one(song, player, input_fn, say, listen_seconds, wait_fn, blind, pref_only=False):
    """Play a song and collect (quality, preference). Returns (q, p) or 's' / 'q'.
    With pref_only only the preference question is asked and quality is returned equal to it (hunt mode)."""
    if player is not None and not player.play(song):
        return "nopreview"
    say("\n" + ("(title hidden)" if blind else f"{song.title}  -  {song.artist}"))
    if listen_seconds > 0:
        say(f"  listening for {listen_seconds:.0f} s ...")
        wait_fn(listen_seconds)
    try:
        while True:
            if pref_only:
                say("  " + PREFERENCE_Q)
                p = ask(input_fn, "  preference (1-5): ")
                if p == "r":
                    if player is not None:
                        player.stop()
                        player.play(song)
                    continue
                return p if p in ("s", "q") else (p, p)
            say("  " + QUALITY_Q)
            q = ask(input_fn, "  quality (1-5): ")
            if q == "r":
                if player is not None:
                    player.stop()
                    player.play(song)
                continue
            if q in ("s", "q"):
                return q
            say("  " + PREFERENCE_Q)
            p = ask(input_fn, "  preference (1-5): ")
            if p == "r":
                if player is not None:
                    player.stop()
                    player.play(song)
                continue
            if p in ("s", "q"):
                return p
            return q, p
    finally:
        if player is not None:
            player.stop()


def rate_session(plan, user, csv_path, player=None, input_fn=input, target=rt.TARGET_SONGS, listen_seconds=15.0,
                 say=print, wait_fn=time.sleep, blind=False, now=time.time):
    """Rate songs from the plan until `target` songs are rated in total. Returns the number rated this run."""
    done = {r["song_id"] for r in rt.first_pass(rt.load_rows(csv_path, user))}
    count = len(done)
    rated = 0
    for song in plan:
        if count >= target:
            break
        if song.song_id in done:
            continue
        res = _rate_one(song, player, input_fn, say, listen_seconds, wait_fn, blind)
        if res == "nopreview":
            continue
        if res == "q":
            say("Quit. Your progress is saved; run the same command to resume.")
            break
        if res == "s":
            say("  skipped (not saved)")
            continue
        q, p = res
        count += 1
        rated += 1
        split = "test" if count % 5 == 0 else "dev"      # exactly 30 held out of 150, spread through the session
        rt.append_row(csv_path, {"user": user, "song_id": song.song_id, "quality": q, "preference": p,
                                 "split": split, "round": 1, "time": now()})
        done.add(song.song_id)
        say(f"  saved ({count}/{target})")
        if count % 30 == 0 and count < target:
            say("  --- 30 songs done: take a short break if you like. Press Enter to continue. ---")
            input_fn("")
    return rated


def hunt_order(plan_pool, rows, by_id, seed=0):
    """Unrated songs ordered so about half are ones the bandit expects you to like (trained on your preferences
    so far) and half are random, alternating. Same idea as run_agreement_check.smart_order."""
    from rl.eval import BanditModel
    seen = {r["song_id"] for r in rows}
    train = [(by_id[r["song_id"]], 1 if r["preference"] >= rt.LIKE_AT else 0) for r in rows
             if r["split"] != "test" and r["song_id"] in by_id and (r["preference"] >= rt.LIKE_AT or r["preference"] <= rt.DISLIKE_AT)]
    fresh = [x for x in plan_pool if x.song_id not in seen]
    rest = list(fresh)
    random.Random(seed).shuffle(rest)
    if not any(y for _, y in train):
        return rest
    model = BanditModel(seed)
    model.fit(train)
    best = sorted(fresh, key=lambda x: -model.score(x))
    out, used = [], set()
    for a, b in zip(best, rest):
        for x in (a, b):
            if x.song_id not in used:
                used.add(x.song_id)
                out.append(x)
    return out


def hunt_session(songs, user, csv_path, player=None, input_fn=input, target=40, listen_seconds=15.0, say=print,
                 wait_fn=time.sleep, blind=False, now=time.time):
    """Liked-song hunt: rate `target` more songs (preference only). Saved as round 3, dev only, outside the 150."""
    done = {r["song_id"] for r in rt.hunt_rows(rt.load_rows(csv_path, user))}
    count = len(done)
    rated = 0
    for song in songs:
        if count >= target:
            break
        if song.song_id in done:
            continue
        res = _rate_one(song, player, input_fn, say, listen_seconds, wait_fn, blind, pref_only=True)
        if res == "nopreview":
            continue
        if res == "q":
            say("Quit. Your progress is saved; run the same command to resume.")
            break
        if res == "s":
            say("  skipped (not saved)")
            continue
        p = res[1]
        count += 1
        rated += 1
        rt.append_row(csv_path, {"user": user, "song_id": song.song_id, "quality": p, "preference": p,
                                 "split": "dev", "round": rt.HUNT_ROUND, "time": now()})
        done.add(song.song_id)
        say(f"  saved ({count}/{target})")
    return rated


def retest_set(rows, n=rt.RETEST_SONGS, seed=0):
    """The fixed set of songs to re-rate (same set every time for a given file)."""
    ids = sorted(r["song_id"] for r in rt.first_pass(rows))
    random.Random(seed).shuffle(ids)
    return ids[:n]


def retest_session(user, csv_path, songs_by_id, player=None, input_fn=input, n=rt.RETEST_SONGS, listen_seconds=15.0,
                   say=print, wait_fn=time.sleep, blind=False, min_hours=12.0, force=False, now=time.time, seed=0):
    """Step 5: rate the same 20 songs again, at least min_hours after the first pass, without showing the old ratings."""
    rows = rt.load_rows(csv_path, user)
    first = rt.first_pass(rows)
    if len(first) < n:
        raise SystemExit(f"Rate at least {n} songs first (you have {len(first)}).")
    gap = (now() - max(r["time"] for r in first)) / 3600
    if gap < min_hours and not force:
        raise SystemExit(f"The last first-pass rating was only {gap:.1f} h ago. A re-rating is meant to be at least "
                         f"{min_hours:.0f} h later (sleep on it); come back later, or use --force to ignore this.")
    chosen = retest_set(rows, n, seed)
    done = {r["song_id"] for r in rows if r["round"] == 2}
    split_of = {r["song_id"]: r["split"] for r in first}
    rated = 0
    for sid in chosen:
        if sid in done or sid not in songs_by_id:
            continue
        res = _rate_one(songs_by_id[sid], player, input_fn, say, listen_seconds, wait_fn, blind)
        if res == "nopreview":
            continue
        if res == "q":
            say("Quit. Your progress is saved; run the same command to resume.")
            break
        if res == "s":
            say("  skipped (not saved)")
            continue
        q, p = res
        rt.append_row(csv_path, {"user": user, "song_id": sid, "quality": q, "preference": p,
                                 "split": split_of[sid], "round": 2, "time": now()})
        rated += 1
        say(f"  saved re-rating ({len(done) + rated}/{len(chosen)})")
    return rated


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--songs", type=int, default=rt.TARGET_SONGS)
    ap.add_argument("--listen", type=float, default=15.0, help="seconds of listening before the questions appear")
    ap.add_argument("--clip-seconds", type=float, default=30.0)
    ap.add_argument("--retest", action="store_true", help="Step 5: re-rate 20 songs you already rated")
    ap.add_argument("--hunt", action="store_true",
                    help="liked-song hunt: rate more songs (preference only), half picked as likely-liked; extra data, not part of the 150")
    ap.add_argument("--hunt-songs", type=int, default=40)
    ap.add_argument("--min-hours", type=float, default=12.0)
    ap.add_argument("--force", action="store_true", help="allow a re-rating sooner than --min-hours")
    ap.add_argument("--blind", action="store_true", help="hide titles and artists")
    ap.add_argument("--no-audio", action="store_true", help="debug: no sound")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available, load_spotify
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    catalog = load_spotify()
    path = rt.ratings_path(a.user)
    player = None
    if not a.no_audio:
        from music.playback.audio import AudioPlayer
        player = AudioPlayer(clip_seconds=a.clip_seconds)
    try:
        if a.retest:
            n = retest_session(a.user, path, {s.song_id: s for s in catalog}, player, listen_seconds=a.listen,
                               blind=a.blind, min_hours=a.min_hours, force=a.force, seed=a.seed)
            print(f"\nRe-rated {n} songs this run.")
        elif a.hunt:
            rows0 = rt.load_rows(path, a.user)
            by_id = {s.song_id: s for s in catalog}
            songs = hunt_order(catalog, [r for r in rows0], by_id, a.seed)
            print(f"Hunting from {len(rt.first_pass(rows0))} earlier ratings "
                  f"({sum(1 for r in rows0 if r['preference'] >= rt.LIKE_AT)} liked).")
            n = hunt_session(songs, a.user, path, player, target=a.hunt_songs, listen_seconds=a.listen, blind=a.blind)
            print(f"\nRated {n} hunt songs this run.")
            print(f"Next: python3 -m rl.ratings --user {a.user} --export data/eval/{a.user}_pref_labels.csv --include-hunt")
            return 0
        else:
            plan = build_plan(catalog, seed=a.seed)
            n = rate_session(plan, a.user, path, player, target=a.songs, listen_seconds=a.listen, blind=a.blind)
            print(f"\nRated {n} songs this run.")
    finally:
        if player:
            player.close()
    rows = rt.load_rows(path, a.user)
    if rows:
        print("\n" + rt.format_text(rt.analyse(rows, {s.song_id: s for s in catalog})))
        print(f"\nNext: python3 -m rl.report --step 4 --ratings {path} --user {a.user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
