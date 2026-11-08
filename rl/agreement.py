"""Does the camera agree with what the person says they liked? (checkpoint P0.3)

Input : one row per song: the person's thumb (1 = up, 0 = down) and the camera's calibrated liking.
Rule  : the camera "says like" when calibrated liking > LIKE_CUTOFF (0.0025), otherwise "says dislike".
Gate  : at least 40 usable songs and the camera's sign matches the thumb on >= 75% of them (proposed).

Run (after collecting with integration/orchestrator/run_agreement_check.py):
    python3 -m rl.agreement --file data/agreement/adi.csv --user adi          # print the result
    python3 -m rl.report --step 3 --agreement data/agreement/adi.csv --user adi   # + tests + report file
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics

from rl import eval as ev

GATE_ACCURACY = 0.75
GATE_MIN_SONGS = 40
LIKE_CUTOFF = 0.0025   # the camera counts as "liked" above this calibrated liking (was 0; set by Adi: detect the faintest reaction)
MIN_CONFIDENCE = 0.1   # same cut-off the bandit uses: below it nobody was really seen
STRONG = 0.2           # a "clear" camera reaction: |calibrated liking| >= this
FIELDS = ["user", "song_id", "thumb", "liking", "confidence", "emotion", "time",
          "features", "setting", "group_size", "likes"]


def load_rows(path, user=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if user and r.get("user") != user:
                continue
            rows.append({"user": r["user"], "song_id": r["song_id"], "thumb": int(r["thumb"]),
                         "liking": float(r["liking"]) if r["liking"] not in ("", "None") else None,
                         "confidence": float(r["confidence"] or 0), "emotion": r.get("emotion", ""),
                         "features": json.loads(r["features"]) if r.get("features") else None,
                         "setting": r.get("setting") or "", "group_size": int(r.get("group_size") or 1)})
    return rows


def usable(rows):
    """Rows where the camera really saw someone. Others are counted but cannot be judged."""
    return [r for r in rows if r["liking"] is not None and r["confidence"] >= MIN_CONFIDENCE]


def wilson(successes, n, z=1.96):
    """95% confidence interval for a proportion (Wilson score interval)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def analyse(rows):
    use = usable(rows)
    n = len(use)
    tp = sum(1 for r in use if r["thumb"] and r["liking"] > LIKE_CUTOFF)    # thumb up, camera says like
    fn = sum(1 for r in use if r["thumb"] and r["liking"] <= LIKE_CUTOFF)   # thumb up, camera says dislike
    fp = sum(1 for r in use if not r["thumb"] and r["liking"] > LIKE_CUTOFF)
    tn = sum(1 for r in use if not r["thumb"] and r["liking"] <= LIKE_CUTOFF)
    ups, downs = tp + fn, fp + tn
    out = {"n_rated": len(rows), "n_usable": n, "n_excluded": len(rows) - n,
           "tp": tp, "fn": fn, "fp": fp, "tn": tn, "ups": ups, "downs": downs}
    if n == 0:
        return {**out, "accuracy": None, "ci": (0.0, 1.0), "majority": None, "balanced": None,
                "kappa": None, "auc": None, "strong_n": 0, "strong_acc": None,
                "mean_up": None, "mean_down": None, "verdict": "NO DATA"}
    acc = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)   # agreement expected by chance
    kappa = (acc - pe) / (1 - pe) if pe < 1 else None
    tpr = tp / ups if ups else None
    tnr = tn / downs if downs else None
    balanced = statistics.mean(x for x in (tpr, tnr) if x is not None)
    strong = [r for r in use if abs(r["liking"]) >= STRONG]
    strong_ok = sum(1 for r in strong if (r["liking"] > LIKE_CUTOFF) == bool(r["thumb"]))
    up_l = [r["liking"] for r in use if r["thumb"]]
    dn_l = [r["liking"] for r in use if not r["thumb"]]
    out.update({
        "accuracy": acc, "ci": wilson(tp + tn, n), "majority": max(ups, downs) / n, "balanced": balanced,
        "tpr": tpr, "tnr": tnr, "kappa": kappa,
        "auc": ev.auc([r["liking"] for r in use], [r["thumb"] for r in use]),
        "strong_n": len(strong), "strong_acc": strong_ok / len(strong) if strong else None,
        "mean_up": statistics.mean(up_l) if up_l else None,
        "mean_down": statistics.mean(dn_l) if dn_l else None,
    })
    if n < GATE_MIN_SONGS:
        out["verdict"] = "NOT ENOUGH SONGS"
    else:
        out["verdict"] = "PASS" if acc >= GATE_ACCURACY else "FAIL"
    return out


# ---------------------------------------------------------------- per-person calibration (Step 3b)
# The general rule says "camera liking > LIKE_CUTOFF means liked". People differ: some show a liked song with a
# tiny smile, some barely react. PersonCalibrator keeps the general rule as the starting point and shifts
# the cut-off to where THIS person's thumbs say it belongs (the cut-off that best catches both their liked
# and disliked songs), but only as far as their data justifies: the shift is shrunk toward 0 when there are
# few songs. A shifted cut-off can change accuracy but never the ranking of songs, so it cannot create signal.
PRIOR_STRENGTH = 10.0  # the general rule counts as this many songs of evidence
# Gate for the per-person version, fixed BEFORE it was run on any data:
B_GATE_ACCURACY = 0.75     # same as the original gate
B_GATE_BALANCED = 0.65     # and it must catch both liked and disliked songs, not just the common one
B_GATE_AUC = 0.65          # and rank liked songs above disliked ones clearly better than chance
B_MIN_PER_CLASS = 10       # the rarer thumb must have at least this many songs, or nothing can be judged


def _balanced(rows, cut):
    ups = [r for r in rows if r["thumb"]]
    downs = [r for r in rows if not r["thumb"]]
    parts = []
    if ups:
        parts.append(sum(1 for r in ups if r["liking"] > cut) / len(ups))
    if downs:
        parts.append(sum(1 for r in downs if r["liking"] <= cut) / len(downs))
    return statistics.mean(parts) if parts else 0.0


class PersonCalibrator:
    """A personal cut-off on the camera's calibrated liking, starting from the general rule (cut-off 0)."""

    def __init__(self, strength=PRIOR_STRENGTH):
        self.strength, self.cut, self.raw_cut = strength, LIKE_CUTOFF, LIKE_CUTOFF

    def fit(self, rows):
        vals = sorted({r["liking"] for r in rows})
        cands = [LIKE_CUTOFF] + [(a + b) / 2 for a, b in zip(vals, vals[1:])]
        # best balanced accuracy; ties go to the candidate nearest the general rule
        self.raw_cut = max(cands, key=lambda c: (round(_balanced(rows, c), 9), -abs(c - LIKE_CUTOFF)))
        self.cut = LIKE_CUTOFF + (self.raw_cut - LIKE_CUTOFF) * len(rows) / (len(rows) + self.strength)
        return self

    def says_like(self, liking):
        return liking > self.cut


def leave_one_out(rows):
    """Fit on all but one song, call that song; repeat for every usable song."""
    use = usable(rows)
    calls = [PersonCalibrator().fit(use[:i] + use[i + 1:]).says_like(r["liking"]) for i, r in enumerate(use)]
    return use, calls


def analyse_personal(rows):
    use, calls = leave_one_out(rows)
    n = len(use)
    ups = sum(r["thumb"] for r in use)
    downs = n - ups
    out = {"n_usable": n, "ups": ups, "downs": downs}
    if n == 0:
        return {**out, "verdict": "NO DATA", "accuracy": None, "balanced": None, "auc": None}
    ok = [bool(c) == bool(r["thumb"]) for c, r in zip(calls, use)]
    tpr = sum(1 for o, r in zip(ok, use) if o and r["thumb"]) / ups if ups else None
    tnr = sum(1 for o, r in zip(ok, use) if o and not r["thumb"]) / downs if downs else None
    out.update({"accuracy": sum(ok) / n, "balanced": statistics.mean(x for x in (tpr, tnr) if x is not None),
                "tpr": tpr, "tnr": tnr, "auc": ev.auc([r["liking"] for r in use], [r["thumb"] for r in use]),
                "majority": max(ups, downs) / n, "cut": PersonCalibrator().fit(use).cut})
    if min(ups, downs) < B_MIN_PER_CLASS:
        out["verdict"] = "NOT ENOUGH DATA"
    elif out["accuracy"] >= B_GATE_ACCURACY and out["balanced"] >= B_GATE_BALANCED and (out["auc"] or 0) >= B_GATE_AUC:
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "FAIL"
    return out


def personal_text(b):
    v = b["verdict"]
    if v == "NO DATA":
        return "No usable songs."
    nums = (f"accuracy {_pct(b['accuracy'])} (need {B_GATE_ACCURACY:.0%}), balanced accuracy {_pct(b['balanced'])} "
            f"(need {B_GATE_BALANCED:.0%}), AUC {_num(b['auc'])} (need {B_GATE_AUC:.2f})")
    if v == "NOT ENOUGH DATA":
        return (f"NOT ENOUGH DATA: only {min(b['ups'], b['downs'])} songs of the rarer thumb "
                f"({b['ups']} up, {b['downs']} down); need {B_MIN_PER_CLASS} to judge. For reference, on this small sample: {nums}.")
    return f"{v}: per-person calibration, each song called by a rule that never saw it: {nums}."


# ---------------------------------------------------------------- do richer camera features help? (Step 3c)
# Gate fixed BEFORE any feature data was collected. Primary comparison = "all features" vs the old reading.
C_MIN_SONGS = 60          # songs that have features
C_MIN_PER_CLASS = 15      # of each thumb
C_GATE_AUC = 0.70         # cross-validated AUC of the all-features model
C_GATE_GAIN = 0.05        # and at least this much better than the old reading (last 3 s of liking)


def feature_sets():
    from perception.emotion import clip_features as cf
    return [("old reading (last 3 s liking)", ["last3_liking"]),
            ("reaction dynamics (peak, range, build...)", cf.LIKING_FEATURES),
            ("movement (nodding, head motion)", cf.MOTION_FEATURES),
            ("all features", cf.FEATURES)]


def compare_features(rows, iterations=200, boot=500, seed=0, perm=100):
    import random
    from rl.agent import camera_model as cm
    use = [r for r in rows if r.get("features") and r["liking"] is not None and r["confidence"] >= MIN_CONFIDENCE]
    ups = sum(r["thumb"] for r in use)
    out = {"n": len(use), "ups": ups, "downs": len(use) - ups, "sets": []}
    if not use:
        return {**out, "verdict": "NO FEATURE DATA"}
    labels = [r["thumb"] for r in use]
    probs_by = {}
    for name, names in feature_sets():
        auc_, _ = cm.cv_auc(use, names, iterations=iterations)
        _, probs = cm.cv_probs(use, names, iterations=iterations)      # one assignment, for the resampling range
        probs_by[name] = probs
        out["sets"].append({"name": name, "auc": auc_, "n_features": len(names)})
    a, d = out["sets"][0]["auc"], out["sets"][-1]["auc"]
    out["gain"] = None if a is None or d is None else d - a
    rng = random.Random(seed)
    diffs = []
    pa, pd = probs_by[out["sets"][0]["name"]], probs_by[out["sets"][-1]["name"]]
    for _ in range(boot):
        idx = [rng.randrange(len(use)) for _ in use]
        yy = [labels[i] for i in idx]
        if 0 < sum(yy) < len(yy):
            diffs.append(ev.auc([pd[i] for i in idx], yy) - ev.auc([pa[i] for i in idx], yy))
    diffs.sort()
    out["gain_range"] = (diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]) if len(diffs) > 20 else None
    from perception.emotion import clip_features as cf
    out["single"] = sorted(((n_, ev.auc([r["features"].get(n_, 0.0) for r in use], labels)) for n_ in cf.FEATURES),
                           key=lambda kv: -abs((kv[1] or 0.5) - 0.5))
    rng2 = random.Random(seed + 1)
    real = out["sets"][-1]["auc"]
    null = []
    for _ in range(perm):
        sh = labels[:]
        rng2.shuffle(sh)
        a_, _ = cm.cv_auc([{**r, "thumb": t} for r, t in zip(use, sh)], None, repeats=1, iterations=iterations // 2 or 50)
        null.append(a_)
    null = [x for x in null if x is not None]
    out["perm_p"] = (sum(1 for x in null if x >= real) / len(null)) if null and real is not None else None
    out["null_95"] = sorted(null)[int(0.95 * len(null)) - 1] if null else None
    half = len(use) // 2
    out["halves"] = []
    for nm, part in (("first half of the session", use[:half]), ("second half", use[half:])):
        yy = [r["thumb"] for r in part]
        out["halves"].append((nm, len(part), sum(yy), ev.auc([r["features"].get("bob_share", 0.0) for r in part], yy)))
    full = cm.CameraReliability().fit(use)
    out["weights"] = sorted(full.w.items(), key=lambda kv: -abs(kv[1]))
    by_setting = {}
    for r, p in zip(use, probs_by["all features"]):
        by_setting.setdefault(r.get("setting") or "untagged", []).append((p, r["thumb"]))
    out["by_setting"] = {k: (len(v), ev.auc([x for x, _ in v], [y for _, y in v])) for k, v in by_setting.items()}
    if len(use) < C_MIN_SONGS or min(out["ups"], out["downs"]) < C_MIN_PER_CLASS:
        out["verdict"] = "NOT ENOUGH DATA"
    elif d is not None and d >= C_GATE_AUC and out["gain"] is not None and out["gain"] >= C_GATE_GAIN:
        out["verdict"] = "PASS"
    else:
        out["verdict"] = "FAIL"
    return out


# Confirmation test, written down BEFORE the quiet-room session, after 'bob_share' stood out in the busy-group session.
# It was picked by looking at that data, so it only counts if it holds on songs it was not picked from.
CONFIRM_FEATURE = "bob_share"
CONFIRM_MIN_AUC = 0.70
CONFIRM_MIN_SONGS = 40
CONFIRM_MIN_PER_CLASS = 12


def confirm_feature(rows, exclude_setting_of_discovery="busy"):
    """Does the feature found in the busy session predict thumbs in NEW, separately collected songs (quiet room)?
    Uses the raw feature value (nothing is fitted), so there is nothing to over-fit."""
    use = [r for r in rows if r.get("features") and r["liking"] is not None and r["confidence"] >= MIN_CONFIDENCE
           and r.get("setting") != exclude_setting_of_discovery]
    ups = sum(r["thumb"] for r in use)
    out = {"n": len(use), "ups": ups, "downs": len(use) - ups}
    if len(use) < CONFIRM_MIN_SONGS or min(out["ups"], out["downs"]) < CONFIRM_MIN_PER_CLASS:
        return {**out, "auc": None, "verdict": "NOT ENOUGH DATA"}
    a = ev.auc([r["features"].get(CONFIRM_FEATURE, 0.0) for r in use], [r["thumb"] for r in use])
    return {**out, "auc": a, "verdict": "CONFIRMED" if a >= CONFIRM_MIN_AUC else "NOT CONFIRMED"}


def confirm_text(c):
    if c["verdict"] == "NOT ENOUGH DATA":
        return (f"NOT ENOUGH DATA: {c['n']} songs outside the busy session ({c['ups']} up, {c['downs']} down); need {CONFIRM_MIN_SONGS} "
                f"with at least {CONFIRM_MIN_PER_CLASS} of each thumb.")
    return f"{c['verdict']}: {CONFIRM_FEATURE} AUC {c['auc']:.3f} on {c['n']} new songs (need {CONFIRM_MIN_AUC:.2f})."


def features_text(c):
    v = c["verdict"]
    if v == "NO FEATURE DATA":
        return "No songs with movement/dynamics features yet. Run the collection tool again (Step 3c)."
    sets = ", ".join(f"{s['name']}: AUC {_num(s['auc'])}" for s in c["sets"])
    if v == "NOT ENOUGH DATA":
        return (f"NOT ENOUGH DATA: {c['n']} songs with features ({c['ups']} up, {c['downs']} down); need {C_MIN_SONGS} songs "
                f"with at least {C_MIN_PER_CLASS} of each thumb. So far (cross-validated): {sets}.")
    return (f"{v}: all-features AUC {_num(c['sets'][-1]['auc'])} (need {C_GATE_AUC:.2f}), "
            f"gain over the old reading {c['gain']:+.3f} (need {C_GATE_GAIN:+.2f}). {sets}.")


def verdict_text(a):
    v = a["verdict"]
    if v == "NO DATA":
        return "No usable songs: the camera never saw a face with enough confidence."
    pct = f"{a['accuracy']:.0%}"
    if v == "NOT ENOUGH SONGS":
        return (f"Only {a['n_usable']} usable songs (need {GATE_MIN_SONGS}); the result so far is {pct}. "
                "Collect more before judging.")
    if v == "PASS":
        s = (f"PASS: the camera matched the thumb on {pct} of {a['n_usable']} songs (gate {GATE_ACCURACY:.0%}).")
        if a["accuracy"] - a["majority"] < 0.05:
            s += (f" Warning: always guessing '{'like' if a['ups'] >= a['downs'] else 'dislike'}' would already score "
                  f"{a['majority']:.0%}, so this pass says little.")
        return s
    return (f"FAIL: the camera matched the thumb on {pct} of {a['n_usable']} songs (gate {GATE_ACCURACY:.0%}). "
            "Action: use the explicit thumbs as the training label and treat the camera only as a secondary signal.")


def _pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def _num(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def format_text(a):
    lo, hi = a["ci"]
    return "\n".join([
        f"Songs rated: {a['n_rated']}  usable (face seen): {a['n_usable']}  excluded: {a['n_excluded']}",
        "                      camera: like   camera: dislike",
        f"  thumb UP        {a['tp']:>14}   {a['fn']:>15}",
        f"  thumb DOWN      {a['fp']:>14}   {a['tn']:>15}",
        f"Agreement: {_pct(a['accuracy'])}  (95% interval {lo:.0%} to {hi:.0%}); "
        f"always-guess-majority would score {_pct(a['majority'])}",
        verdict_text(a)])


def build_agreement_report(rows, tests, user="", synthetic=False):
    from rl import report as rp
    a = analyse(rows)
    b = analyse_personal(rows)
    c = compare_features(rows)
    try:
        from music.catalog.spotify_catalog import dataset_available, load_spotify
        if c["sets"] and dataset_available():
            from rl.agent import camera_model as cm
            by = {x.song_id: x for x in load_spotify()}
            keys = ("tempo", "danceability", "energy", "valence", "popularity")
            sr = [{**r, "features": {k: by[r["song_id"]].audio_features.get(k, 0.0) for k in keys}}
                  for r in rows if r.get("features") and r["song_id"] in by and r["liking"] is not None]
            c["song_audio_auc"] = cm.cv_auc(sr, list(keys), iterations=200)[0] if len(sr) >= 20 else None
    except Exception:
        c["song_audio_auc"] = None
    lo, hi = a["ci"]
    parts = [f"# Step 3 report: camera vs thumbs agreement (P0.3)", "",
             f"- Generated: {rp.time.strftime('%Y-%m-%d %H:%M:%S')}  ", f"- Code: {rp.git_info()}  ",
             f"- Python {rp.platform.python_version()} on {rp.platform.system()}  ",
             f"- User: {user or 'n/a'}", "", rp.test_section(tests), "",
             "## 2. How well the camera agrees with the person", "",
             "**SYNTHETIC data: a scripted fake face and fake thumbs, not a real person.**" if synthetic else
             "Data: this person's own thumbs up/down next to the camera's reading for the same songs.", "",
             f"**Result: {verdict_text(a)}**", "",
             f"- Songs rated: {a['n_rated']}. Usable (a face was seen with confidence >= {MIN_CONFIDENCE}): {a['n_usable']}. "
             f"Excluded: {a['n_excluded']}.", "",
             rp.md_table(["", "camera says like", "camera says dislike"],
                         [["thumb UP", a["tp"], a["fn"]], ["thumb DOWN", a["fp"], a["tn"]]]), "",
             rp.md_table(["measure", "value", "what it means"], [
                 ["agreement (accuracy)", _pct(a["accuracy"]), f"share of songs where the camera's sign matched the thumb; 95% interval {lo:.0%} to {hi:.0%}"],
                 ["always-guess-majority", _pct(a["majority"]), "what a camera that ignored your face and always said the more common thumb would score"],
                 ["balanced accuracy", _pct(a["balanced"]), "average of 'liked songs the camera caught' and 'disliked songs the camera caught'; fair when thumbs are lopsided"],
                 ["kappa", _num(a["kappa"]), "agreement beyond chance: 0 = chance, 1 = perfect"],
                 ["AUC of liking vs thumb", _num(a["auc"], 3), "chance a liked song got a higher camera score than a disliked one; ignores the 0 cut-off"],
                 [f"clear reactions (liking at least {STRONG} from 0, either way)", f"{_pct(a['strong_acc'])} of {a['strong_n']}", "agreement when the camera was not near neutral"],
                 ["mean liking, thumb UP", _num(a["mean_up"]), "should be clearly above 0"],
                 ["mean liking, thumb DOWN", _num(a["mean_down"]), "should be clearly below the thumb-UP mean"]]), "",
             "## 2b. Re-test with a per-person calibration (Step 3b)", "",
             f"**Result: {personal_text(b)}**", "",
             f"The general rule ('liking > {LIKE_CUTOFF} = liked') was replaced by a rule fitted to this person's thumbs, "
             "starting from the general rule and held close to it when data is thin. Each song is "
             "predicted by a model that never saw that song (leave-one-out), so the score is not flattered "
             "by fitting and testing on the same songs.", "",
             rp.md_table(["measure", "value", "notes"], [
                 ["usable songs", b["n_usable"], f"{b['ups']} thumb up, {b['downs']} thumb down"],
                 ["accuracy (leave-one-out)", _pct(b["accuracy"]), f"need {B_GATE_ACCURACY:.0%}; always-majority scores {_pct(b.get('majority'))}"],
                 ["balanced accuracy", _pct(b["balanced"]), f"need {B_GATE_BALANCED:.0%}; catches liked: {_pct(b.get('tpr'))}, catches disliked: {_pct(b.get('tnr'))}"],
                 ["AUC", _num(b["auc"], 3), f"need {B_GATE_AUC:.2f}; 0.5 = chance"],
                 ["personal cut-off", _num(b.get("cut")), f"the general rule is {LIKE_CUTOFF}; a song counts as liked when the camera's calibrated liking is above this"]]), "",
             "## 2c. Do movement and reaction-dynamics features help? (Step 3c)", "",
             f"**Result: {features_text(c)}**", ""]
    if c["sets"]:
        parts += [rp.md_table(["camera measurement used", "features", "AUC (5-fold, repeated 3 times: each song scored by a model that never saw it)"],
                              [[s_["name"], s_["n_features"], _num(s_["auc"], 3)] for s_ in c["sets"]]), ""]
        if c.get("gain_range"):
            parts += [f"Gain of 'all features' over the old reading: {c['gain']:+.3f}; 95% resampling range {c['gain_range'][0]:+.3f} to {c['gain_range'][1]:+.3f} "
                      "(if the range includes 0, the gain could be noise).", ""]
        parts += [rp.md_table(["feature (standardised)", "weight", "reading"],
                              [[n_, f"{w:+.2f}", "raises P(liked)" if w > 0 else "lowers P(liked)"] for n_, w in c["weights"][:6]]),
                  "", "Weights come from one fit on all songs and are descriptive only: with this few songs a single weight is not reliable.", "",
                  rp.md_table(["setting", "songs", "AUC (all features)"],
                              [[k, n_, _num(a_, 3)] for k, (n_, a_) in sorted(c["by_setting"].items())]), ""]
        parts += ["Checks on the 'all features' result (the primary comparison fixed beforehand):", "",
                  rp.md_table(["check", "result"], [
                      ["thumbs shuffled at random (100 times)", f"models reach AUC {_num(c.get('null_95'))} or more only 5% of the time by luck; real result {_num(c['sets'][-1]['auc'])}; chance of doing this well by luck p = {_num(c.get('perm_p'), 3)}"],
                      *[[f"{CONFIRM_FEATURE} alone, {nm} ({n_} songs, {u_} liked)", f"AUC {_num(a_, 3)}"] for nm, n_, u_, a_ in c["halves"]],
                      *([["the songs' own audio (tempo, danceability, energy...) alone", f"AUC {_num(c['song_audio_auc'], 3)}; if this were high the movement could just be the song"]] if c.get("song_audio_auc") is not None else [])]), "",
                  rp.md_table(["single feature", "AUC on its own (0.5 = chance)"], [[n_, _num(a_, 3)] for n_, a_ in c["single"][:5]]), "",
                  "Single features were looked at after the fact, so the best one is likely to look better than it is. That is why it is re-tested on new songs below.", ""]
        from rl.agent import camera_model as cm
        wt, auc_, n_ = cm.reliability(rows)
        parts += [f"**Camera weight for the learner:** {wt:.2f} (from cross-validated AUC {_num(auc_, 3)} on {n_} songs; 0 = ignore the camera, 1 = trust it like a thumb). "
                  f"A weight needs at least {cm.MIN_SONGS} songs with features and {cm.MIN_PER_CLASS} of each thumb; below that the camera is ignored.", ""]
    cf_ = confirm_feature(rows)
    parts += ["## 2d. Confirmation on new songs", "",
              f"Rule fixed before the quiet-room session: **{CONFIRM_FEATURE}** (share of up-and-down head movement in the 1 to 3.5 Hz nodding band, picked because it stood out in the busy-group session) "
              f"must reach AUC >= {CONFIRM_MIN_AUC:.2f} by itself on at least {CONFIRM_MIN_SONGS} songs from other sessions, {CONFIRM_MIN_PER_CLASS} or more of each thumb. Nothing is fitted, so there is nothing to over-fit.", "",
              f"**Result: {confirm_text(cf_)}**", "",
              "## 3. How the numbers were produced", "",
             "- **Collection:** songs from the gym and party moods play as 10-second iTunes previews. Before the first song the camera measures a resting baseline "
             "(neutral face, guarded against smiling or moving). At the end of each clip the camera reading is the average liking over the last 3 seconds, "
             "adjusted relative to that baseline so a neutral face is about 0. Then the person presses up or down.",
             f"- **Camera's call:** like if calibrated liking > {LIKE_CUTOFF}, otherwise dislike. A reading at or below that counts as dislike.",
             f"- **Excluded:** a song is not judged if no face was seen or mean confidence < {MIN_CONFIDENCE}; those songs are counted above, not hidden.",
             "- **Agreement:** (camera-like and thumb-up) + (camera-dislike and thumb-down), divided by usable songs.",
             "- **95% interval:** Wilson score interval for a proportion, centre (p + z²/2n)/(1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²)/(1 + z²/n), z = 1.96.",
             "- **Kappa:** (agreement − expected) / (1 − expected), where expected is the agreement two independent raters would reach by chance given how often each says like.",
             f"- **Gate (proposed, fixed beforehand):** at least {GATE_MIN_SONGS} usable songs and agreement >= {GATE_ACCURACY:.0%}.",
             "- **Step 3b (per-person calibration):** the cut-off on the camera's calibrated liking starts at the general rule (0). It is moved to the value that best catches both this person's liked and disliked songs (best balanced accuracy on the training songs), "
             f"then shrunk toward {LIKE_CUTOFF} by n / (n + {PRIOR_STRENGTH:.0f}) so a handful of songs barely moves it. Each song is called by a cut-off fitted without it (leave-one-out). "
             f"Gate set before it was run on any data: accuracy >= {B_GATE_ACCURACY:.0%}, balanced accuracy >= {B_GATE_BALANCED:.0%}, AUC >= {B_GATE_AUC:.2f}, "
             f"and at least {B_MIN_PER_CLASS} songs of the rarer thumb (otherwise nothing can be judged). Moving a cut-off cannot change the ranking of songs, so AUC stays low if the camera has no signal.", "",
             "- **Step 3c (richer features):** per clip the tool records the reaction's mean, peak, low, range, build (last third minus first third) and last-3-seconds average; "
             "the smile probability minus the resting smile; and face movement: speed of the face centre, vertical (nodding) speed, and the share of vertical movement in the 1 to 3.5 Hz head-bob band with its amplitude. "
             "Movement and smile are measured against the person's own resting baseline from calibration. A logistic model per person (strong pull toward 'no effect', equal to "
             f"{10} songs of evidence) turns the features into P(thumb up); every song is scored by a model fitted without it (stratified 5-fold cross-validation repeated 3 times; leave-one-out is avoided because it pushes weak-signal AUC toward 0), and AUC is computed on those scores. "
             f"Gate set before any feature data existed: at least {C_MIN_SONGS} songs with features and {C_MIN_PER_CLASS} of each thumb, all-features AUC >= {C_GATE_AUC:.2f}, and at least {C_GATE_GAIN:.2f} better than the old reading.",
             "- **Camera weight:** 2 × (cross-validated AUC − 0.5), between 0 and 1. It is the share of a thumb's influence the camera gets when the learner combines thumbs, skips and the camera.", "",
             "## 4. What these numbers do and do not show", "",
             f"- With {a['n_usable']} songs the 95% interval is about ±{(hi - lo) / 2 * 100:.0f} points; a score a few points either side of {GATE_ACCURACY:.0%} is not decisive.",
             "- One person, one room, gym and party songs only. It does not show the camera works for other people or other music.",
             "- The thumb is treated as the truth, but it is a noisy label too (people press the wrong button, or rate the song rather than how they felt).",
             "- If thumbs are lopsided (mostly up or mostly down), compare agreement with 'always-guess-majority' and look at balanced accuracy and kappa.",
             "- If this fails, the plan is to train on thumbs and use the camera only as a secondary signal; the base model and personal layers are not affected."]
    return "\n".join(parts) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Camera vs thumbs agreement.")
    ap.add_argument("--file", required=True)
    ap.add_argument("--user")
    args = ap.parse_args(argv)
    rows = load_rows(args.file, args.user)
    if not rows:
        raise SystemExit("No rows found.")
    print(format_text(analyse(rows)))
    print("\nPer-person calibration (leave-one-out):\n" + personal_text(analyse_personal(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
