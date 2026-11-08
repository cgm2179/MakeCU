"""Offline replay of whole sessions on logged thumbs: which policy picks songs Adi likes, and does it improve while it learns?
Design and expectation fixed in reports/step08c_policy_replay_preregistration.md.

    python3 -m rl.replay --user adi
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("random", "popularity-greedy", "base-only", "old bandit", "base + online learner (beta 0)",
            "base + online learner (beta 0.3)", "popularity prior + online learner (beta 0.3)")
POOL_SIZE = 150
STEPS = 40
SESSIONS = 300


def load_logged_thumbs(user="adi", root=None):
    """{song_id: thumb} from the camera-agreement sessions (not the busy-group one) and the live sessions; later files override earlier ones."""
    from rl import agreement as ag
    from rl import live_test as lt
    root = Path(root) if root else ROOT / "data"
    labels = {}
    for name in (user, f"{user}_rep"):
        p = root / "agreement" / f"{name}.csv"
        if p.exists():
            for r in ag.load_rows(p, name):
                labels[r["song_id"]] = r["thumb"]
    for name in (user, f"{user}_v2"):
        p = root / "live" / f"{name}.csv"
        if p.exists():
            for r in lt.load_rows(p):
                labels[r["song_id"]] = r["thumb"]
    return labels


def _reaction(song, thumb):
    from shared.schemas.messages import GroupReaction, PersonReaction
    v = 1.0 if thumb else -1.0
    return GroupReaction(0.0, song.song_id, (PersonReaction(0, "happy" if thumb else "sad", v, 1.0),), v, {})


def run_policy(name, pool, labels, base, steps=STEPS, seed=0):
    """Play one session; returns the 0/1 thumbs of the picked songs, in the order picked."""
    rng = random.Random(seed)
    left, picked, thumbs = list(pool), [], []
    agent = learner = None
    if name == "old bandit":
        from rl.agent.bandit import BanditAgent
        from rl.agent.context import Context
        agent = BanditAgent(seed=seed)
        agent.mainstream_active = True
        ctx, last = Context(), None
    elif name.startswith(("base +", "popularity prior")):
        from rl.agent.prior_learner import PriorLearner
        beta = 0.0 if "beta 0)" in name else 0.3
        learner = PriorLearner(base, "popularity" if name.startswith("popularity prior") else "base", beta=beta, seed=seed)
    for _ in range(min(steps, len(left))):
        if name == "random":
            song = rng.choice(left)
        elif name == "popularity-greedy":
            song = max(left, key=lambda s: (s.audio_features.get("popularity", 0.0), rng.random()))
        elif name == "base-only":
            song = max(left, key=lambda s: (base.preference_logit(s), rng.random()))
        elif agent is not None:
            song, _ = agent.choose_next_song(left, last, ctx, recent_ids=[x.song_id for x in picked[-4:]])
        elif learner is not None:
            song, _ = learner.choose(left)
        else:
            raise ValueError(f"unknown policy {name!r}")
        left.remove(song)
        picked.append(song)
        thumb = labels[song.song_id]
        thumbs.append(int(thumb))
        if agent is not None:
            agent.update(song, _reaction(song, thumb), ctx)
            last = song.song_id
        elif learner is not None:
            learner.update_from_signals(song, thumb=thumb)
    return thumbs


def replay(songs_by_id, labels, base, sessions=SESSIONS, pool_size=POOL_SIZE, steps=STEPS, policies=POLICIES, seed=0, log=print):
    ids = sorted(i for i in labels if i in songs_by_id)
    out = {p: {"all": [], "first": [], "last": []} for p in policies}
    for k in range(sessions):
        rng = random.Random(seed * 100003 + k)
        pool = [songs_by_id[i] for i in rng.sample(ids, min(pool_size, len(ids)))]
        for p in policies:
            t = run_policy(p, pool, labels, base, steps, seed=k)
            h = len(t) // 2
            out[p]["all"].append(float(np.mean(t)))
            out[p]["first"].append(float(np.mean(t[:h])))
            out[p]["last"].append(float(np.mean(t[h:])))
        if log and (k + 1) % 50 == 0:
            log(f"  {k + 1}/{sessions} sessions")
    return out


def summarize(res):
    rows = {}
    for p, d in res.items():
        a = np.array(d["all"])
        rows[p] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
                   "first": float(np.mean(d["first"])), "last": float(np.mean(d["last"]))}
    return rows


def format_table(rows):
    lines = [f"{'policy':<48} {'liked share':>12} {'first half':>11} {'second half':>12}"]
    for p, r in rows.items():
        lines.append(f"{p:<48} {r['mean']:>7.1%} +/-{r['se'] * 100:3.1f} {r['first']:>11.1%} {r['last']:>12.1%}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="adi")
    ap.add_argument("--sessions", type=int, default=SESSIONS)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from music.catalog.spotify_catalog import load_spotify
    from rl import base_lastfm as bl
    by_id = {s.song_id: s for s in load_spotify()}
    labels = load_logged_thumbs(a.user)
    base = bl.load_base(bl.MODEL_DIR / "base_model_v2.json")
    print(f"{len(labels)} logged thumbs ({sum(labels.values())} liked); {a.sessions} sessions of {STEPS} picks from pools of {POOL_SIZE}")
    t0 = time.time()
    res = replay(by_id, labels, base, a.sessions, seed=a.seed)
    rows = summarize(res)
    print(format_table(rows))
    print(f"({time.time() - t0:.0f}s)")
    import json
    (bl.MODEL_DIR / "policy_replay.json").write_text(json.dumps({"n_labels": len(labels), "liked": int(sum(labels.values())), "rows": rows}, indent=1))
    return rows


if __name__ == "__main__":
    main()
