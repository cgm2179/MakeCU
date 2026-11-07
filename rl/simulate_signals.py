"""Does weighting the camera by its reliability help the learner? A simulation on SYNTHETIC listeners.

    python3 -m rl.simulate_signals            # prints the table and writes reports/step03c_signal_weighting_simulation.md

A made-up listener likes a few genres (about 25% of songs, so, like Adi, mostly dislikes). Thumbs are right
92% of the time and available on a chosen share of songs. The camera is a weak signal made to match what was
measured on real data (AUC about 0.67 for the old reading; two features carry a little signal, the rest
none). The bandit learns online for 60 songs under each way of using the signals:

  thumbs only            learns only from thumbs; songs without a thumb teach it nothing
  camera as a thumb      uses a thumb when there is one, else treats the camera's call as a full-strength thumb
  camera weighted (ours) rl.agent.reward: thumb 1.0, camera scaled by its measured reliability

It reports how well the learned scores rank 300 unseen songs (AUC) and what share of its last 20 picks the
listener truly liked. The numbers say what the weighting does under these assumptions, not how any real person behaves.
"""
from __future__ import annotations

import random
import statistics as st
import sys
import time
from pathlib import Path
from statistics import NormalDist

from perception.emotion.clip_features import FEATURES
from rl import eval as ev
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from rl.agent.reward import CameraRewardModel, combine
from shared.schemas.messages import GroupReaction, PersonReaction

THUMB_ACCURACY = 0.92
STRATEGIES = ["thumbs only", "camera as a thumb", "camera weighted (ours)"]


def d_prime(auc):
    return 2 ** 0.5 * NormalDist().inv_cdf(auc)


class World:
    def __init__(self, catalog, seed, like_share=0.25, auc_main=0.67, auc_second=0.60):
        rng = random.Random(seed)
        self.rng = rng
        sample = rng.sample(catalog, 1500)
        genres = sorted({s.genre for s in sample})
        rng.shuffle(genres)
        counts = {g: sum(1 for s in sample if s.genre == g) for g in genres}
        liked, total = set(), 0
        for g in genres:
            if total / len(sample) >= like_share:
                break
            liked.add(g)
            total += counts[g]
        self.liked = liked
        self.d = {"last3_liking": d_prime(auc_main), "bob_amp": d_prime(auc_second)}
        self.songs = sample

    def likes(self, song):
        return int(song.genre in self.liked)

    def thumb(self, song):
        y = self.likes(song)
        return y if self.rng.random() < THUMB_ACCURACY else 1 - y

    def camera(self, song):
        y = self.likes(song)
        return {n: self.rng.gauss(self.d.get(n, 0.0) * y, 1.0) for n in FEATURES}


def reaction(song, liking=0.0):
    return GroupReaction(0.0, song.song_id, (PersonReaction(0, "neutral", liking, 1.0),), liking, {})


def warm_rows(world, n=80):
    rows = []
    for s in world.rng.sample(world.songs[:600], n):
        rows.append({"user": "sim", "song_id": s.song_id, "thumb": world.thumb(s), "features": world.camera(s),
                     "liking": 0.0, "confidence": 1.0, "emotion": "neutral", "group_size": 1, "setting": ""})
    return rows


def play(world, strategy, p_thumb, rm, seed, rounds=120):
    rng = random.Random(seed * 7 + 1)
    world.rng = random.Random(seed * 13 + 5)
    agent = BanditAgent(seed=seed)
    pool = world.songs[:400]
    ctx = Context()
    recent, picks = [], []
    for _ in range(rounds):
        song, _mode = agent.choose_next_song(pool, context=ctx, recent_ids=recent[-4:])
        recent.append(song.song_id)
        picks.append(world.likes(song))
        thumb = world.thumb(song) if rng.random() < p_thumb else None
        feats = world.camera(song)
        if strategy == "thumbs only":
            reward, _ = combine(thumb=thumb)
        elif strategy == "camera as a thumb":
            reward = (1.0 if thumb else -1.0) if thumb is not None else (1.0 if feats["last3_liking"] > 0 else -1.0)   # the old rule: reading above 0 = liked
        else:
            cam = rm.model.soft_reward(feats) if rm.model is not None else None
            reward, _ = combine(thumb=thumb, camera=cam, camera_weight=rm.weight)
        if reward is not None:
            agent.update(song, reaction(song, reward), ctx)
    held = world.songs[600:900]
    auc = ev.auc([agent.predict(s, ctx) for s in held], [world.likes(s) for s in held])
    return auc, sum(picks[-20:]) / 20


def simulate(catalog, seeds=8, p_thumbs=(0.0, 0.25, 1.0), warm=80, progress=print, seed0=300):
    out = {(s, p): {"auc": [], "picks": []} for s in STRATEGIES for p in p_thumbs}
    weights, cam_aucs = [], []
    for seed in range(seed0, seed0 + seeds):
        world = World(catalog, seed)
        rm = CameraRewardModel().fit(warm_rows(world, warm), iterations=100)
        weights.append(rm.weight)
        cam_aucs.append(rm.auc)
        for p in p_thumbs:
            for strat in STRATEGIES:
                a, pk = play(world, strat, p, rm, seed)
                if a is not None:
                    out[(strat, p)]["auc"].append(a)
                out[(strat, p)]["picks"].append(pk)
        progress(f"  warm={warm}: seed {seed - seed0 + 1}/{seeds} done")
    return out, weights, cam_aucs


def to_markdown(runs, seeds, p_thumbs, rounds=120):
    """runs: {warm: (out, weights, cam_aucs)}"""
    def ms(v):
        v = [x for x in v if x is not None]
        return f"{st.mean(v):.3f} ± {st.pstdev(v):.3f}" if v else "n/a"
    lines = ["# Step 3c simulation: does weighting the camera by its reliability help the learner?", "",
             "**SYNTHETIC.** Made-up listeners (like about 25% of songs by genre; thumbs right 92% of the time) and a made-up weak camera "
             "(old reading AUC 0.67, one extra feature AUC 0.60, ten features with no signal). It shows what the weighting does under these assumptions; "
             "it says nothing about how real people or Adi's camera behave.", "",
             f"- {seeds} random worlds; the bandit learns online for {rounds} songs per condition; scores are on 300 unseen songs. Mean ± standard deviation across worlds.",
             "- 'camera as a thumb' is the old behaviour: the camera's call (reading above 0 = liked) counted as much as a thumb.", ""]
    for warm, (out, weights, cam_aucs) in sorted(runs.items()):
        lines += [f"# Camera model fitted on {warm} earlier songs with thumbs", "",
                  f"Measured reliability: cross-validated AUC {ms(cam_aucs)}, weight {ms(weights)} (0 = ignored, 1 = trusted like a thumb).", ""]
        for metric, title in (("auc", "Ranking of unseen songs (AUC, 0.5 = chance)"), ("picks", "Share of its last 20 picks the listener really liked")):
            lines += [f"## {title}", "", "| how signals are used | " + " | ".join(f"thumbs on {int(p * 100)}% of songs" for p in p_thumbs) + " |",
                      "|---|" + "---|" * len(p_thumbs)]
            for strat in STRATEGIES:
                lines.append(f"| {strat} | " + " | ".join(ms(out[(strat, p)][metric]) for p in p_thumbs) + " |")
            lines.append("")
    lines += ["## History of this simulation", "",
              "- Version 1 turned the camera into a reward of 2p - 1. For a listener who likes 25% of songs that is about -0.5 on nearly every song, a constant push down. "
              "On its worlds (seeds 0 to 7), with thumbs on 25% of songs, the weighted camera's last-20 picks were liked 0.31 of the time against 0.52 for 'thumbs only' (spread about ±0.2). "
              "Version 2 measured the camera relative to the person's usual rate but scaled the two sides differently, which left a negative average (seeds 100 to 107: weighted 0.41, thumbs only 0.53). "
              "Version 3 used one symmetric scale so the camera's reward averages 0, and is run on fresh worlds (seeds 200 to 207) so it is not judged on the data it was designed on. "
              "Version 3 was judged by leave-one-out scoring, which turned out to push weak-signal reliability toward 0 (a held-out liked song lowers the training base rate, so its own score falls). "
              "Version 4 (this report) uses stratified 5-fold cross-validation, found by the real busy-session data (an AUC of 0.000 for a signal-free reading) and confirmed by a unit test, "
              "and runs on fresh worlds (seeds 300 to 307). Four versions is already some tuning to a made-up world: treat the result as a check that nothing is broken, not as proof.", "",
              "## How to read this", "",
              "- 'camera weighted' scales the camera by its measured reliability, so a noisy camera cannot outvote a thumb; its rewards are small when the camera is weak.",
              "- Differences smaller than the ± spread are not real. The listener, noise levels and camera strength are assumptions."]
    return "\n".join(lines) + "\n"


def main():
    from music.catalog.spotify_catalog import dataset_available, load_spotify
    if not dataset_available():
        sys.exit("Dataset missing (data/spotify/dataset.csv).")
    seeds, p_thumbs = 8, (0.0, 0.25, 1.0)
    t0 = time.time()
    catalog = load_spotify()
    runs = {warm: simulate(catalog, seeds, p_thumbs, warm) for warm in (80, 150)}
    text = to_markdown(runs, seeds, p_thumbs)
    path = Path(__file__).resolve().parents[1] / "reports" / "step03c_signal_weighting_simulation.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    print(text)
    print(f"(written to {path}; {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
