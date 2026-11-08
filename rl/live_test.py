"""Live session test (checkpoint L3.2): does the personal model pick songs you like more often?

Three conditions, mixed at random inside the same session so mood, tiredness and the order cannot favour one:
  random      songs drawn at random from the candidate pool
  popularity  one of the 10 most popular songs left in the pool (what the robot did before it learned)
  personal    one of the 10 songs the saved profile predicts you will like most (frozen: it does not learn here)
The person is never told which condition a song came from. Ground truth is the explicit thumb after each song.

File: data/live/<user>.csv, columns user, seq, condition, song_id, thumb, time (git-ignored).
Gate (fixed beforehand): personal liked-share >= popularity + 15 points AND >= random + 25 points, with 40+ songs per condition.
At 40 songs per condition the margin of error on a share is about +/-15 points, so a pass is suggestive, not proof.
"""
from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

from rl.agent.context import Context
from rl.agreement import wilson

CONDITIONS = ("random", "popularity", "personal")
FIELDS = ["user", "seq", "condition", "song_id", "thumb", "time"]
TOP_K = 10
PER_CONDITION = 40
GATE_VS_POPULARITY = 0.15
GATE_VS_RANDOM = 0.25
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "live"


def live_path(user, directory=None):
    import re
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", user or ""):
        raise ValueError(f"bad user name {user!r}")
    return Path(directory or DEFAULT_DIR) / f"{user}.csv"


def load_rows(path, user=None):
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return [{"user": r["user"], "seq": int(r["seq"]), "condition": r["condition"], "song_id": r["song_id"],
                 "thumb": int(r["thumb"]), "time": float(r["time"] or 0)}
                for r in csv.DictReader(f) if not user or r["user"] == user]


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def make_sequence(per_condition=PER_CONDITION, seed=0):
    """Conditions in random blocks of three, so the counts stay balanced all the way through."""
    rng = random.Random(seed)
    out = []
    for _ in range(per_condition):
        block = list(CONDITIONS)
        rng.shuffle(block)
        out += block
    return out


def pick_song(condition, pool, agent, rng, k=TOP_K):
    """One song from the pool for this condition. Returns None if the pool is empty."""
    if not pool:
        return None
    if condition == "random":
        return rng.choice(pool)
    if condition == "popularity":
        ranked = sorted(pool, key=lambda s: -s.audio_features.get("popularity", 0.0))
    elif condition == "personal":
        ctx = Context()
        ranked = sorted(pool, key=lambda s: -agent.predict(s, ctx))
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return rng.choice(ranked[:k])


# ---------------------------------------------------------------- statistics

def analyse(rows):
    out = {"conditions": {}}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c]
        n, k = len(sub), sum(r["thumb"] for r in sub)
        lo, hi = wilson(k, n)
        out["conditions"][c] = {"n": n, "liked": k, "share": k / n if n else None, "lo": lo, "hi": hi}
    out["diffs"] = {}
    p = out["conditions"]["personal"]
    for other, need in (("popularity", GATE_VS_POPULARITY), ("random", GATE_VS_RANDOM)):
        o = out["conditions"][other]
        if p["n"] and o["n"]:
            d = p["share"] - o["share"]
            se = math.sqrt(p["share"] * (1 - p["share"]) / p["n"] + o["share"] * (1 - o["share"]) / o["n"])
            out["diffs"][other] = {"diff": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se, "need": need}
    enough = all(out["conditions"][c]["n"] >= PER_CONDITION for c in CONDITIONS)
    if not enough:
        out["verdict"] = "IN PROGRESS"
    elif all(out["diffs"][o]["diff"] >= out["diffs"][o]["need"] for o in ("popularity", "random")):
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "FAIL"
    return out


def format_text(a):
    lines = []
    for c in CONDITIONS:
        s = a["conditions"][c]
        share = "n/a" if s["share"] is None else f"{s['share']:.0%} (95% interval {s['lo']:.0%} to {s['hi']:.0%})"
        lines.append(f"{c:<11} liked {s['liked']:>3} of {s['n']:>3}: {share}")
    for o, d in a["diffs"].items():
        lines.append(f"personal minus {o}: {d['diff']*100:+.0f} points (95% range {d['lo']*100:+.0f} to {d['hi']*100:+.0f}; "
                     f"gate needs {d['need']*100:+.0f})")
    lines.append(f"Overall: {a['verdict']}  (needs {PER_CONDITION}+ songs per condition; at that size a share is only "
                 f"known to about +/-15 points)")
    return "\n".join(lines)


def build_live_report(rows, tests, user, profile_note=""):
    a = analyse(rows)
    parts = [f"# Step 8 report: live session (personal model vs popularity vs random), {user}", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ", f"- User: {user}", "",
             "## 1. Unit tests", "",
             f"- Ran {tests['ran']}: {tests['ran'] - tests['failed'] - tests['errors'] - tests['skipped']} passed, "
             f"{tests['failed']} failed, {tests['errors']} errors, {tests['skipped']} skipped.", "",
             "## 2. Result", "", "```", format_text(a), "```", "",
             "## 3. Method", "",
             "- Three conditions mixed in random blocks of three; the person was not told which condition a song came from.",
             f"- random: any song in the pool. popularity: one of the {TOP_K} most popular left. personal: one of the {TOP_K} songs "
             "the saved profile predicts highest (profile frozen; it does not learn during the test).",
             "- Pool: songs from the gym/party/chill/happy/sad/study moods, minus every song already rated (the profile was trained "
             "on those), one song per artist.",
             "- Label: the explicit thumb after each song. The song stops as soon as the thumb is given.",
             "- Share liked = thumbs up / songs. 95% interval: Wilson. Difference range: normal approximation.",
             "- Gate fixed beforehand: personal >= popularity + 15 points and >= random + 25 points, 40+ songs per condition.", "",
             "## 4. Limits", "",
             "- One person, 40 songs per condition: a share is known only to about +/-15 points, so a 15-point gap is suggestive.",
             "- The camera is not used here. The thumb is the label.",
             "- The profile was trained on every earlier label, so this session is its first test on songs it never saw.",
             profile_note]
    return "\n".join(p for p in parts if p is not None) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Summarise a live session and write reports/step08_live_session[_USER].md.")
    ap.add_argument("--user", default="adi")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    rows = load_rows(live_path(a.user), a.user)
    if not rows:
        raise SystemExit("No live session rows yet. Run: python3 -m integration.orchestrator.run_live_session")
    print(format_text(analyse(rows)))
    if not a.no_report:
        from rl.report import run_unit_tests, write_report
        tests = run_unit_tests()
        path = write_report(8, "live_session" if a.user == "adi" else f"live_session_{a.user}", build_live_report(rows, tests, a.user))
        print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
        print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
