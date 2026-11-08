"""Step 3d: does the camera's reaction-shape signal replicate on new songs? Rule fixed in reports/step03d_preregistration.md.

    python3 -m rl.camera_replication                 # trains on data/agreement/adi.csv, tests on data/agreement/adi_rep.csv

Model: logistic regression (L2 1.0, standardised features, trained on the first session only) on the six reaction-dynamics features.
Pass: >= 40 usable new songs with >= 12 of each thumb, AUC >= 0.70, and >= +0.05 over the old reading (last3_liking alone).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rl import agreement as ag
from rl.personal_layer import fast_auc, fit_logistic

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS = ["mean_liking", "peak_liking", "low_liking", "range_liking", "build_liking", "last3_liking"]  # pre-registered
LAM = 1.0
PASS_AUC = 0.70
PASS_GAIN = 0.05
MIN_SONGS = 40
MIN_PER_CLASS = 12
MIN_CONF = 0.1


def usable_rows(rows, exclude_setting=None):
    """Rows with all six features, a camera reading and enough confidence. Optionally drop one setting (the busy sessions)."""
    return [r for r in rows if r["features"] and all(k in r["features"] for k in DYNAMICS) and r["liking"] is not None
            and r["confidence"] >= MIN_CONF and (exclude_setting is None or r["setting"] != exclude_setting)]


def matrix(rows, keys=DYNAMICS):
    return np.array([[r["features"][k] for k in keys] for r in rows], dtype=float)


def fit_model(train_rows, keys=DYNAMICS):
    """Standardise on the training songs and fit the penalised logistic model. Returns (weights incl. bias, mean, sd)."""
    X = matrix(train_rows, keys)
    y = np.array([float(r["thumb"]) for r in train_rows])
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mean) / sd, np.ones((len(X), 1))])
    return fit_logistic(Z, y, np.zeros(len(X)), LAM), mean, sd


def score_rows(model, rows, keys=DYNAMICS):
    w, mean, sd = model
    Z = np.hstack([(matrix(rows, keys) - mean) / sd, np.ones((len(rows), 1))])
    return Z @ w


def boot_auc(scores, y, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fast_auc(scores[i], y[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_gain(a, b, y, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fast_auc(a[i], y[i]) - fast_auc(b[i], y[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(train_rows, new_rows, context=None, boot=2000):
    """Apply the pre-registered rule. `context` = optional {name: scores aligned with usable new rows} (popularity, fine-tuned...)."""
    tr = usable_rows(train_rows, exclude_setting="busy")
    new = usable_rows(new_rows)
    y = np.array([r["thumb"] for r in new], dtype=int)
    out = {"n_train": len(tr), "n_new": len(new), "n_liked": int(y.sum()), "n_disliked": int(len(y) - y.sum()),
           "n_dropped": len(new_rows) - len(new)}
    if len(tr) < 20 or len(set(r["thumb"] for r in tr)) < 2:
        return {**out, "verdict": "NOT ENOUGH TRAINING DATA", "reason": f"{len(tr)} training songs with features"}
    if len(new) < MIN_SONGS or out["n_liked"] < MIN_PER_CLASS or out["n_disliked"] < MIN_PER_CLASS:
        return {**out, "verdict": "NOT ENOUGH DATA",
                "reason": f"need >= {MIN_SONGS} songs with >= {MIN_PER_CLASS} of each thumb; have {len(new)} "
                          f"({out['n_liked']} liked, {out['n_disliked']} disliked)"}
    model = fit_model(tr)
    s_dyn = score_rows(model, new)
    s_old = np.array([r["features"]["last3_liking"] for r in new])
    s_peak = np.array([r["features"]["peak_liking"] for r in new])
    auc_dyn, auc_old, auc_peak = fast_auc(s_dyn, y), fast_auc(s_old, y), fast_auc(s_peak, y)
    gain = auc_dyn - auc_old
    half = len(new) // 2
    out.update({"auc": auc_dyn, "auc_lo_hi": boot_auc(s_dyn, y, boot), "auc_old_reading": auc_old, "auc_peak_only": auc_peak,
                "gain": gain, "gain_lo_hi": boot_gain(s_dyn, s_old, y, boot),
                "auc_first_half": fast_auc(s_dyn[:half], y[:half]), "auc_second_half": fast_auc(s_dyn[half:], y[half:]),
                "weights": dict(zip(DYNAMICS, [float(x) for x in model[0][:-1]])),
                "context": {k: fast_auc(np.asarray(v), y) for k, v in (context or {}).items()}})
    out["pass_auc"], out["pass_gain"] = auc_dyn >= PASS_AUC, gain >= PASS_GAIN
    out["verdict"] = "PASS" if out["pass_auc"] and out["pass_gain"] else "FAIL"
    return out


def format_text(r):
    if r["verdict"].startswith("NOT ENOUGH"):
        return f"{r['verdict']}: {r['reason']}. (dropped for no face/low confidence: {r['n_dropped']})"
    lo, hi = r["auc_lo_hi"]
    glo, ghi = r["gain_lo_hi"]
    lines = [f"New songs: {r['n_new']} ({r['n_liked']} liked, {r['n_disliked']} disliked); trained on {r['n_train']} earlier songs.",
             f"Reaction-shape model AUC {r['auc']:.3f}  (95% range {lo:.3f} to {hi:.3f}; mark >= {PASS_AUC})",
             f"Old reading (last 3 s) AUC {r['auc_old_reading']:.3f};  gain {r['gain']:+.3f}  (95% range {glo:+.3f} to {ghi:+.3f}; mark >= {PASS_GAIN:+})",
             f"peak_liking alone {r['auc_peak_only']:.3f};  first half {r['auc_first_half']:.3f}, second half {r['auc_second_half']:.3f}"]
    for k, v in r["context"].items():
        lines.append(f"context: {k} AUC {v:.3f}")
    lines.append(f"Overall: {r['verdict']}")
    return "\n".join(lines)


def build_report(r, tests, notes=""):
    parts = ["# Step 3d: camera reaction-shape replication on new songs", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
             "- Rule fixed beforehand: `step03d_preregistration.md`", "",
             "## Unit tests", "",
             f"- Ran {tests['ran']}: {tests['failed']} failed, {tests['errors']} errors, {tests['skipped']} skipped.", "",
             "## Result", "", "```", format_text(r), "```", ""]
    if "weights" in r:
        parts += ["Weights (standardised features; descriptive only): " + ", ".join(f"{k} {v:+.2f}" for k, v in r["weights"].items()), ""]
    parts += ["## Limits", "",
              "- One person, one room, a few dozen new songs: an AUC is known to about +/-0.1.",
              "- The thumb is given right after a 10-second clip.",
              "- Part of any camera AUC can come from the songs themselves (popular, upbeat songs get liked and may also get a stronger reaction); "
              "the context lines show how well popularity alone does on the same songs.",
              notes]
    return "\n".join(p for p in parts if p is not None) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="adi")
    ap.add_argument("--new", default="adi_rep")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    d = ROOT / "data" / "agreement"
    if not (d / f"{a.new}.csv").exists():
        raise SystemExit(f"No new session yet at {d / (a.new + '.csv')}. Run run_agreement_check with --user {a.new} first.")
    train = ag.load_rows(d / f"{a.train}.csv", a.train)
    new = ag.load_rows(d / f"{a.new}.csv", a.new)
    context = {}
    try:
        from music.catalog.spotify_catalog import load_spotify
        from rl import base_lastfm as bl
        from rl import personal_layer as pl
        by_id = {s.song_id: s for s in load_spotify()}
        used = usable_rows(new)
        songs = [by_id[r["song_id"]] for r in used if r["song_id"] in by_id]
        if len(songs) == len(used):
            context["popularity"] = [s.audio_features.get("popularity", 0.0) for s in songs]
            base = bl.load_base(bl.MODEL_DIR / "base_model_v2.json")
            layer = pl.load_personal(pl.PROFILE_DIR / f"{a.train}_personal_v2.json", base)
            context["fine-tuned model (Step 7)"] = [layer.logit(s) for s in songs]
    except Exception as e:           # context is optional
        print(f"(context baselines skipped: {e})")
    res = evaluate(train, new, context)
    print(format_text(res))
    if not a.no_report:
        from rl.report import run_unit_tests
        tests = run_unit_tests()
        path = ROOT / "reports" / "step03d_camera_replication.md"
        path.write_text(build_report(res, tests))
        print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
        print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
