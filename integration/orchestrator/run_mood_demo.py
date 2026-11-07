"""End-to-end demo on the real Spotify dataset (offline, fake like/dislike feedback).

    python3 -m integration.orchestrator.run_mood_demo --mood gym --rounds 40 \\
        --likes punk-rock,emo --then-likes ska,anime --shift-at 25

Steps: load CSV -> mood filter picks the pool near the target -> contextual bandit
plays songs, a fake listener with a hidden genre taste likes/dislikes them -> summary.
"""
from __future__ import annotations

import argparse
import statistics as st
from collections import Counter

from integration.orchestrator.run_loop import describe, run
from music.catalog.songs import get_candidates
from music.catalog.spotify_catalog import dataset_available, load_spotify
from music.filter.mood_targets import profile_for
from rl.env.taste_listener import TasteListener
from shared.schemas.messages import Filter, FilterSource

KEYS = ("valence", "energy", "tempo", "instrumentalness")


def means(songs):
    return {k: round(st.mean(s.audio_features[k] for s in songs), 2) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mood", default="gym")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--pool-size", type=int, default=500)
    ap.add_argument("--likes", default="", help="genres the fake listener likes (comma separated)")
    ap.add_argument("--then-likes", default="", help="genres liked after --shift-at")
    ap.add_argument("--shift-at", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", type=int, default=0, metavar="N",
                    help="repeat the run over N seeds and print aggregate results only")
    ap.add_argument("--quiet", action="store_true", help="hide the per-round learning printout")
    a = ap.parse_args()

    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    profile = profile_for(a.mood)
    if not profile:
        raise SystemExit(f"Unknown mood '{a.mood}'")

    allsongs = load_spotify()
    flt = Filter((FilterSource("event", a.mood),))
    pool = get_candidates(flt, a.pool_size)
    genres = Counter(s.genre for s in pool)
    print(f"Dataset: {len(allsongs):,} unique tracks loaded from CSV")
    print(f"First filter: event='{a.mood}' -> target { {k: v[0] for k, v in profile.items()} }")
    print(f"Pool: {len(pool)} nearest songs")
    print(f"  pool means   : {means(pool)}")
    print(f"  dataset means: {means(allsongs)}")
    print(f"  top genres in pool: {genres.most_common(8)}")
    print("  closest 5 to target:")
    for s in pool[:5]:
        print("    -", describe(s))

    likes = [g for g in a.likes.split(",") if g] or [g for g, _ in genres.most_common(4)[1:3]]
    shift = {a.shift_at: [g for g in a.then_likes.split(",") if g]} if a.then_likes else None
    print(f"\nFake listener secretly likes genres: {likes}"
          + (f", switching to {shift[a.shift_at]} at round {a.shift_at}" if shift else ""))
    print("=" * 78)

    if a.sweep:
        base = sum(genres[g] for g in likes) / len(pool)
        early, late = [], []
        for sd in range(a.sweep):
            lg = run(a.rounds, event=a.mood, seed=sd, verbose=False, round_delay=0.0, pool=pool,
                     listener=TasteListener(likes, seed=sd), overrides={})["log"]
            hit = lambda es: sum(e["song"].genre in likes for e in es) / len(es)
            early.append(hit(lg[:15]))
            late.append(hit(lg[-20:]))
        print(f"\nSweep over {a.sweep} seeds, {a.rounds} rounds each (random would be {base:.0%}):")
        print(f"  liked-genre share, first 15 plays: mean {st.mean(early):.0%}")
        print(f"  liked-genre share, last 20 plays : mean {st.mean(late):.0%}, "
              f"worst seed {min(late):.0%}, seeds under 40%: {sum(x < 0.4 for x in late)}/{a.sweep}")
        return

    out = run(a.rounds, event=a.mood, seed=a.seed, verbose=not a.quiet, round_delay=0.0,
              pool=pool, listener=TasteListener(likes, seed=a.seed), taste_shift=shift,
              overrides={})

    log = out["log"]
    played = [e["song"] for e in log]
    print("\n" + "=" * 78 + "\nSUMMARY")
    print(f"Songs played ({len(played)}): pool means were {means(pool)}")
    print(f"  played means: {means(played)}   <- close to the '{a.mood}' target")
    print(f"  distinct songs: {len({s.song_id for s in played})} of {len(played)} plays")
    print(f"  genres played: {Counter(s.genre for s in played).most_common(6)}")

    def like_rate(entries, liked):
        return sum(e["song"].genre in liked for e in entries) / max(1, len(entries))

    third = max(1, len(log) // 3)
    if shift:
        k = a.shift_at - 1
        phases = [("before shift, first third", log[:third], set(likes)),
                  ("before shift, last 8 rounds", log[max(0, k - 8):k], set(likes)),
                  ("right after shift (8 rounds)", log[k:k + 8], set(shift[a.shift_at])),
                  ("end (last 8 rounds)", log[-8:], set(shift[a.shift_at]))]
    else:
        phases = [("first third", log[:third], set(likes)), ("last third", log[-third:], set(likes))]
    print("Share of plays in the listener's currently-liked genres:")
    for name, entries, liked in phases:
        print(f"  {name:<30} {like_rate(entries, liked):4.0%}   "
              f"(mean reward {st.mean(e['reward'] for e in entries):+.2f})")
    base = sum(genres[g] for g in likes) / len(pool)
    print(f"  (random picks from the pool would hit liked genres {base:.0%} of the time)")

    agent = out["agent"]
    best, worst = agent.top_features("global", 6)
    print("\nWhat the agent learned overall:")
    print("  likes   :", ", ".join(f"{f} {w:+.2f}" for f, w in best))
    print("  dislikes:", ", ".join(f"{f} {w:+.2f}" for f, w in worst))


if __name__ == "__main__":
    main()
