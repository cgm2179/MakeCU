"""Adi's two-question ratings (checkpoint L1.2): quality vs personal preference.

For every song the person answers two different questions, each 1 to 5:
  quality    : "Ignoring my taste, how well-made or widely appreciated is this song?"
  preference : "How much do I personally want to hear this?"
The disagreements between the two are what teach a model the difference between "good" and "liked".

File: data/ratings/<user>.csv, columns user, song_id, quality, preference, split, round, time.
  round 1 = the first rating of the song, round 2 = a re-rating later (test-retest, Step 5).
  split   = "test" for the 30 songs held out as the final exam (never used for tuning), else "dev".

Gates (proposed in docs/THREE_LAYER_CHECKPOINTS.md, fixed before any data):
  * at least 150 songs rated, 30 of them held out
  * the two ratings differ by 2 or more points on at least 15% of songs
  * re-rating 20 songs a day later: at least 80% within 1 point, for each question
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "ratings"
FIELDS = ["user", "song_id", "quality", "preference", "split", "round", "time"]
TARGET_SONGS = 150
HELD_OUT = 30
MIN_DISAGREE_SHARE = 0.15
DISAGREE_POINTS = 2
RETEST_SONGS = 20
RETEST_WITHIN = 1
RETEST_MIN_SHARE = 0.80
LIKE_AT = 4           # preference 4 or 5 = liked, 1 or 2 = disliked, 3 = undecided (left out of yes/no labels)
DISLIKE_AT = 2
POP_BANDS = ((0, 35, "unknown (popularity under 35)"), (35, 65, "known (35 to 65)"), (65, 101, "hit (65 and over)"))


def ratings_path(user, directory=None):
    import re
    if not re.match(r"^[A-Za-z0-9_-]{1,40}$", user or ""):
        raise ValueError(f"bad user name {user!r}")
    return Path(directory or DEFAULT_DIR) / f"{user}.csv"


def load_rows(path, user=None):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if user and r["user"] != user:
                continue
            rows.append({"user": r["user"], "song_id": r["song_id"], "quality": int(r["quality"]),
                         "preference": int(r["preference"]), "split": r["split"], "round": int(r["round"]),
                         "time": float(r["time"] or 0)})
    return rows


def append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def first_pass(rows):
    return [r for r in rows if r["round"] == 1]


def retest_pairs(rows):
    """[(first rating, second rating)] for songs rated twice."""
    first = {r["song_id"]: r for r in rows if r["round"] == 1}
    return [(first[r["song_id"]], r) for r in rows if r["round"] == 2 and r["song_id"] in first]


# ---------------------------------------------------------------- statistics

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    """Rank correlation in [-1, 1]; None if either side is constant."""
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    sa = sum((x - ma) ** 2 for x in ra)
    sb = sum((x - mb) ** 2 for x in rb)
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) ** 0.5


def pop_band(popularity):
    for lo, hi, name in POP_BANDS:
        if lo <= popularity < hi:
            return name
    return POP_BANDS[-1][2]


def analyse(rows, catalog_by_id=None):
    """Everything the report and the gates need. catalog_by_id (song_id -> Song) adds coverage and popularity checks."""
    one = first_pass(rows)
    n = len(one)
    out = {"n": n, "n_test": sum(1 for r in one if r["split"] == "test"), "n_dev": sum(1 for r in one if r["split"] != "test")}
    if n == 0:
        return {**out, "disagree": None, "corr_qp": None, "q_counts": {}, "p_counts": {}, "retest": retest_stats(rows),
                "gates": gates(out, None, retest_stats(rows))}
    q = [r["quality"] for r in one]
    p = [r["preference"] for r in one]
    dis = sum(1 for a, b in zip(q, p) if abs(a - b) >= DISAGREE_POINTS)
    out.update({
        "disagree": dis / n, "disagree_n": dis,
        "q_higher": sum(1 for a, b in zip(q, p) if a - b >= DISAGREE_POINTS),   # good but not for me
        "p_higher": sum(1 for a, b in zip(q, p) if b - a >= DISAGREE_POINTS),   # guilty pleasure
        "corr_qp": spearman(q, p),
        "q_counts": {v: q.count(v) for v in range(1, 6)}, "p_counts": {v: p.count(v) for v in range(1, 6)},
        "q_mean": statistics.mean(q), "p_mean": statistics.mean(p),
        "liked": sum(1 for v in p if v >= LIKE_AT), "disliked": sum(1 for v in p if v <= DISLIKE_AT),
        "undecided": sum(1 for v in p if DISLIKE_AT < v < LIKE_AT),
    })
    if catalog_by_id:
        have = [(r, catalog_by_id[r["song_id"]]) for r in one if r["song_id"] in catalog_by_id]
        pops = [s.audio_features.get("popularity", 0.0) for _, s in have]
        out["corr_q_pop"] = spearman([r["quality"] for r, _ in have], pops)
        out["corr_p_pop"] = spearman([r["preference"] for r, _ in have], pops)
        out["by_mood"] = _count(s.mood or "unknown" for _, s in have)
        out["by_band"] = _count(pop_band(x) for x in pops)
        out["n_genres"] = len({s.genre for _, s in have})
        gc = _count(s.genre for _, s in have)
        out["top_genre"] = max(gc.items(), key=lambda kv: kv[1]) if gc else None
    rt = retest_stats(rows)
    out["retest"] = rt
    out["gates"] = gates(out, out["disagree"], rt)
    return out


def _count(items):
    d = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return d


def retest_stats(rows):
    pairs = retest_pairs(rows)
    n = len(pairs)
    if n == 0:
        return {"n": 0, "q_within": None, "p_within": None, "both_within": None}
    qs = sum(1 for a, b in pairs if abs(a["quality"] - b["quality"]) <= RETEST_WITHIN)
    ps = sum(1 for a, b in pairs if abs(a["preference"] - b["preference"]) <= RETEST_WITHIN)
    both = sum(1 for a, b in pairs if abs(a["quality"] - b["quality"]) <= RETEST_WITHIN
               and abs(a["preference"] - b["preference"]) <= RETEST_WITHIN)
    exact = sum(1 for a, b in pairs if a["quality"] == b["quality"] and a["preference"] == b["preference"])
    return {"n": n, "q_within": qs / n, "p_within": ps / n, "both_within": both / n, "exact_both": exact / n}


def gates(out, disagree, rt):
    """Each gate is 'PASS', 'FAIL' or 'PENDING' (not enough done yet to judge)."""
    g = {}
    g["150 songs rated, 30 held out"] = ("PASS" if out["n"] >= TARGET_SONGS and out["n_test"] >= HELD_OUT else "PENDING",
                                         f"{out['n']} rated, {out['n_test']} held out")
    if out["n"] < TARGET_SONGS or disagree is None:
        g["the two ratings really differ"] = ("PENDING", f"judged once {TARGET_SONGS} songs are rated"
                                              + ("" if disagree is None else f" (so far {disagree:.0%})"))
    else:
        g["the two ratings really differ"] = ("PASS" if disagree >= MIN_DISAGREE_SHARE else "FAIL",
                                              f"{disagree:.0%} of songs differ by {DISAGREE_POINTS}+ points (need {MIN_DISAGREE_SHARE:.0%})")
    if rt["n"] < RETEST_SONGS:
        g["re-rating is consistent"] = ("PENDING", f"{rt['n']} of {RETEST_SONGS} songs re-rated")
    else:
        ok = rt["q_within"] >= RETEST_MIN_SHARE and rt["p_within"] >= RETEST_MIN_SHARE
        g["re-rating is consistent"] = ("PASS" if ok else "FAIL",
                                        f"within {RETEST_WITHIN} point: quality {rt['q_within']:.0%}, preference {rt['p_within']:.0%} (need {RETEST_MIN_SHARE:.0%} each)")
    return g


def overall(gates_):
    vals = [v for v, _ in gates_.values()]
    if "FAIL" in vals:
        return "FAIL"
    return "PASS" if all(v == "PASS" for v in vals) else "IN PROGRESS"


def export_labels(rows, path, user, target="preference"):
    """Yes/no labels for rl.eval: preference >= 4 -> 1, <= 2 -> 0; 3s are left out as undecided.
    The split column carries the 30 held-out songs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user", "song_id", "label", "split"])
        for r in first_pass(rows):
            v = r[target]
            if v >= LIKE_AT:
                lab = 1
            elif v <= DISLIKE_AT:
                lab = 0
            else:
                continue
            w.writerow([user, r["song_id"], lab, "test" if r["split"] == "test" else ""])
            n += 1
    return n


def format_text(a):
    lines = [f"Songs rated: {a['n']} ({a['n_dev']} for development, {a['n_test']} held out for the final exam)"]
    if a["n"]:
        lines += [f"Quality    ratings 1-5: {a['q_counts']}  mean {a['q_mean']:.2f}",
                  f"Preference ratings 1-5: {a['p_counts']}  mean {a['p_mean']:.2f}",
                  f"They differ by {DISAGREE_POINTS}+ points on {a['disagree_n']} songs ({a['disagree']:.0%}): "
                  f"{a['q_higher']} good-but-not-for-me, {a['p_higher']} guilty-pleasure; rank correlation {a['corr_qp'] if a['corr_qp'] is None else round(a['corr_qp'], 2)}"]
    for name, (v, why) in a["gates"].items():
        lines.append(f"  [{v:^11}] {name}: {why}")
    lines.append(f"Overall: {overall(a['gates'])}")
    return "\n".join(lines)


def build_ratings_report(rows, tests, user="", catalog_by_id=None, synthetic=False):
    from rl import report as rp
    a = analyse(rows, catalog_by_id)
    f = lambda x, d=2: "n/a" if x is None else f"{x:.{d}f}"
    parts = ["# Step 4 report: two-question ratings (quality vs preference)", "",
             f"- Generated: {rp.time.strftime('%Y-%m-%d %H:%M:%S')}  ", f"- Code: {rp.git_info()}  ",
             f"- Python {rp.platform.python_version()} on {rp.platform.system()}  ", f"- User: {user or 'n/a'}", "",
             rp.test_section(tests), "", "## 2. What the ratings show", "",
             "**SYNTHETIC ratings (scripted, not a person).**" if synthetic else "Data: this person's own ratings of 30-second previews.", "",
             f"**Overall: {overall(a['gates'])}**", "",
             rp.md_table(["gate", "status", "detail"], [[k, v, why] for k, (v, why) in a["gates"].items()]), ""]
    if a["n"]:
        parts += [rp.md_table(["rating", "1", "2", "3", "4", "5", "mean"],
                              [["quality"] + [a["q_counts"][v] for v in range(1, 6)] + [f(a["q_mean"])],
                               ["preference"] + [a["p_counts"][v] for v in range(1, 6)] + [f(a["p_mean"])]]), "",
                  f"- **They differ by {DISAGREE_POINTS}+ points on {a['disagree_n']} of {a['n']} songs ({a['disagree']:.0%}; need {MIN_DISAGREE_SHARE:.0%}).** "
                  f"{a['q_higher']} are good-but-not-for-me (quality above preference) and {a['p_higher']} are guilty pleasures (preference above quality). These are the songs that teach the model the difference.",
                  f"- Rank correlation between quality and preference: {f(a['corr_qp'])} (1 = always the same ranking, 0 = unrelated).",
                  f"- Preference as yes/no labels for the evaluation tool: {a['liked']} liked (4-5), {a['disliked']} disliked (1-2), {a['undecided']} undecided (3, left out)."]
        top = max(max(a["q_counts"].values()), max(a["p_counts"].values())) / a["n"]
        if top > 0.5:
            parts.append(f"- Warning: one rating value makes up {top:.0%} of an answer set; a narrow range gives a model little to learn from.")
        if "by_mood" in a:
            parts += ["", rp.md_table(["coverage", "songs"],
                                      [[f"mood: {k}", v] for k, v in sorted(a["by_mood"].items())] +
                                      [[f"popularity: {k}", v] for k, v in sorted(a["by_band"].items())] +
                                      [["distinct genres", a["n_genres"]],
                                       ["most common genre", f"{a['top_genre'][0]} ({a['top_genre'][1]})" if a["top_genre"] else "n/a"]]), "",
                      f"- **Does dataset popularity stand in for 'quality'?** Rank correlation of your quality rating with the song's popularity: {f(a.get('corr_q_pop'))}; "
                      f"of your preference with popularity: {f(a.get('corr_p_pop'))}. If quality tracks popularity much more closely than preference does, popularity is a fair stand-in for 'objective'; "
                      "if neither does, it is not, and the plan's 'popularity as the objective proxy' needs rethinking."]
        rtst = a["retest"]
        if rtst["n"]:
            parts += ["", f"- **Re-rating (test-retest), {rtst['n']} songs:** within {RETEST_WITHIN} point: quality {rtst['q_within']:.0%}, preference {rtst['p_within']:.0%}, both {rtst['both_within']:.0%}; "
                          f"identical on both questions {rtst['exact_both']:.0%}. Need {RETEST_MIN_SHARE:.0%} on each question."]
    parts += ["", "## 3. How the numbers were produced", "",
              "- **Collection:** songs are drawn evenly across mood (upbeat, aggressive, calm, melancholy) and three popularity bands (under 35, 35 to 65, 65 and over), at most one song per artist and "
              f"{6} per genre. Each plays as a 30-second preview; the questions appear after 15 seconds and the clip keeps playing. Every 5th rated song is saved as held out (the final exam), so 30 of 150.",
              f"- **Disagreement:** share of songs where |quality − preference| >= {DISAGREE_POINTS}.",
              "- **Rank correlation:** Spearman, the correlation of the two ratings' ranks (ties share the average rank).",
              f"- **Test-retest:** the same {RETEST_SONGS} songs are rated again at least 12 hours later without showing the earlier answers; a song is 'within 1 point' when the new rating is 1 or less away from the old.",
              f"- **Gates (proposed, fixed before any ratings):** {TARGET_SONGS} songs with {HELD_OUT} held out; disagreement >= {MIN_DISAGREE_SHARE:.0%}; re-rating within 1 point on >= {RETEST_MIN_SHARE:.0%} for each question.", "",
              "## 4. What these numbers do and do not show", "",
              "- One person. Quality as one person rates it is that person's opinion of quality, not an objective fact.",
              "- 30-second previews are not whole songs, and the order of songs, tiredness and mood on the day all move ratings. The re-rating measures part of that noise.",
              "- This step only collects and checks labels. Whether a model can learn from them is tested in later steps (base model, fine-tune, live test).",
              "- Reports contain counts and aggregates only, never song titles or the raw ratings."]
    return "\n".join(parts) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarise a ratings file.")
    ap.add_argument("--user", default="adi")
    ap.add_argument("--file")
    ap.add_argument("--export", help="write yes/no labels for rl.eval to this CSV")
    args = ap.parse_args(argv)
    path = args.file or ratings_path(args.user)
    rows = load_rows(path, args.user)
    if not rows:
        raise SystemExit(f"No ratings found in {path}.")
    print(format_text(analyse(rows)))
    if args.export:
        print(f"Wrote {export_labels(rows, args.export, args.user)} labels to {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
