"""Does the camera agree with what the person says they liked? (checkpoint P0.3)

Input : one row per song: the person's thumb (1 = up, 0 = down) and the camera's calibrated liking.
Rule  : the camera "says like" when calibrated liking > 0, otherwise "says dislike".
Gate  : at least 40 usable songs and the camera's sign matches the thumb on >= 75% of them (proposed).

Run (after collecting with integration/orchestrator/run_agreement_check.py):
    python3 -m rl.agreement --file data/agreement/adi.csv --user adi          # print the result
    python3 -m rl.report --step 3 --agreement data/agreement/adi.csv --user adi   # + tests + report file
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics

from rl import eval as ev

GATE_ACCURACY = 0.75
GATE_MIN_SONGS = 40
MIN_CONFIDENCE = 0.1   # same cut-off the bandit uses: below it nobody was really seen
STRONG = 0.2           # a "clear" camera reaction: |calibrated liking| >= this
FIELDS = ["user", "song_id", "thumb", "liking", "confidence", "emotion", "time"]


def load_rows(path, user=None):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if user and r.get("user") != user:
                continue
            rows.append({"user": r["user"], "song_id": r["song_id"], "thumb": int(r["thumb"]),
                         "liking": float(r["liking"]) if r["liking"] not in ("", "None") else None,
                         "confidence": float(r["confidence"] or 0), "emotion": r.get("emotion", "")})
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
    tp = sum(1 for r in use if r["thumb"] and r["liking"] > 0)    # thumb up, camera says like
    fn = sum(1 for r in use if r["thumb"] and r["liking"] <= 0)   # thumb up, camera says dislike
    fp = sum(1 for r in use if not r["thumb"] and r["liking"] > 0)
    tn = sum(1 for r in use if not r["thumb"] and r["liking"] <= 0)
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
    strong_ok = sum(1 for r in strong if (r["liking"] > 0) == bool(r["thumb"]))
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
# The general rule says "camera liking > 0 means liked". People differ: some show a liked song with a
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
        self.strength, self.cut, self.raw_cut = strength, 0.0, 0.0

    def fit(self, rows):
        vals = sorted({r["liking"] for r in rows})
        cands = [0.0] + [(a + b) / 2 for a, b in zip(vals, vals[1:])]
        # best balanced accuracy; ties go to the candidate nearest the general rule
        self.raw_cut = max(cands, key=lambda c: (round(_balanced(rows, c), 9), -abs(c)))
        self.cut = self.raw_cut * len(rows) / (len(rows) + self.strength)
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
             "The general rule ('liking > 0 = liked') was replaced by a rule fitted to this person's thumbs, "
             "starting from the general rule and held close to it when data is thin. Each song is "
             "predicted by a model that never saw that song (leave-one-out), so the score is not flattered "
             "by fitting and testing on the same songs.", "",
             rp.md_table(["measure", "value", "notes"], [
                 ["usable songs", b["n_usable"], f"{b['ups']} thumb up, {b['downs']} thumb down"],
                 ["accuracy (leave-one-out)", _pct(b["accuracy"]), f"need {B_GATE_ACCURACY:.0%}; always-majority scores {_pct(b.get('majority'))}"],
                 ["balanced accuracy", _pct(b["balanced"]), f"need {B_GATE_BALANCED:.0%}; catches liked: {_pct(b.get('tpr'))}, catches disliked: {_pct(b.get('tnr'))}"],
                 ["AUC", _num(b["auc"], 3), f"need {B_GATE_AUC:.2f}; 0.5 = chance"],
                 ["personal cut-off", _num(b.get("cut")), "the general rule is 0.00; a song counts as liked when the camera's calibrated liking is above this"]]), "",
             "## 3. How the numbers were produced", "",
             "- **Collection:** songs from the gym and party moods play as 10-second iTunes previews. Before the first song the camera measures a resting baseline "
             "(neutral face, guarded against smiling or moving). At the end of each clip the camera reading is the average liking over the last 3 seconds, "
             "adjusted relative to that baseline so a neutral face is about 0. Then the person presses up or down.",
             "- **Camera's call:** like if calibrated liking > 0, otherwise dislike. A reading exactly at the baseline counts as dislike.",
             f"- **Excluded:** a song is not judged if no face was seen or mean confidence < {MIN_CONFIDENCE}; those songs are counted above, not hidden.",
             "- **Agreement:** (camera-like and thumb-up) + (camera-dislike and thumb-down), divided by usable songs.",
             "- **95% interval:** Wilson score interval for a proportion, centre (p + z²/2n)/(1 + z²/n), half-width z·sqrt(p(1-p)/n + z²/4n²)/(1 + z²/n), z = 1.96.",
             "- **Kappa:** (agreement − expected) / (1 − expected), where expected is the agreement two independent raters would reach by chance given how often each says like.",
             f"- **Gate (proposed, fixed beforehand):** at least {GATE_MIN_SONGS} usable songs and agreement >= {GATE_ACCURACY:.0%}.",
             "- **Step 3b (per-person calibration):** the cut-off on the camera's calibrated liking starts at the general rule (0). It is moved to the value that best catches both this person's liked and disliked songs (best balanced accuracy on the training songs), "
             f"then shrunk toward 0 by n / (n + {PRIOR_STRENGTH:.0f}) so a handful of songs barely moves it. Each song is called by a cut-off fitted without it (leave-one-out). "
             f"Gate set before it was run on any data: accuracy >= {B_GATE_ACCURACY:.0%}, balanced accuracy >= {B_GATE_BALANCED:.0%}, AUC >= {B_GATE_AUC:.2f}, "
             f"and at least {B_MIN_PER_CLASS} songs of the rarer thumb (otherwise nothing can be judged). Moving a cut-off cannot change the ranking of songs, so AUC stays low if the camera has no signal.", "",
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
