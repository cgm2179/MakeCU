"""Step 3e: do the engagement features (movement and smile, measured against a per-song pre-roll) transfer between sessions?
Rule fixed in reports/step03e_preregistration.md.

    python3 -m rl.camera_validation --sessions adi_v2a adi_v2b

Leave-one-session-out: train on the other sessions, test on the held-out one. Pass = AUC >= 0.65 on EVERY held-out session,
each with >= 40 usable songs and >= 12 of each thumb.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from perception.emotion import clip_features as cf
from rl import agreement as ag
from rl import camera_replication as cr
from rl.eval import make_folds
from rl.personal_layer import fast_auc, fit_logistic

ROOT = Path(__file__).resolve().parents[1]
V2 = list(cf.ENGAGEMENT_FEATURES)          # pre-registered feature set
PASS_AUC = 0.65
MIN_SONGS = 40
MIN_PER_CLASS = 12
LAM = 1.0


def usable(rows, keys):
    return [r for r in rows if r["features"] and all(k in r["features"] for k in keys) and r["liking"] is not None
            and r["confidence"] >= cr.MIN_CONF]


def within_session_cv(rows, keys, reps=10):
    X = cr.matrix(rows, keys)
    y = np.array([float(r["thumb"]) for r in rows])
    n = len(y)
    aucs = []
    for r in range(reps):
        oof = np.zeros(n)
        folds = make_folds([(i, int(y[i])) for i in range(n)], 5, seed=r)
        for k, te in enumerate(folds):
            tei = np.array([i for i, _ in te])
            tri = np.array([i for j, f in enumerate(folds) if j != k for i, _ in f])
            mu, sd = X[tri].mean(0), X[tri].std(0)
            sd[sd == 0] = 1.0
            Z = lambda idx: np.hstack([(X[idx] - mu) / sd, np.ones((len(idx), 1))])
            oof[tei] = Z(tei) @ fit_logistic(Z(tri), y[tri], np.zeros(len(tri)), LAM)
        aucs.append(fast_auc(oof, y))
    return float(np.mean(aucs))


def leave_one_session_out(sessions, keys):
    """{session: AUC on that session of a model trained on all the others}."""
    out = {}
    for held in sessions:
        train = [r for n, rows in sessions.items() if n != held for r in rows]
        test = sessions[held]
        y = np.array([r["thumb"] for r in test])
        model = cr.fit_model(train, keys)
        out[held] = fast_auc(cr.score_rows(model, test, keys), y)
    return out


def evaluate(raw_sessions, context=None):
    """raw_sessions: {name: rows from load_rows}. context: optional {name: {label: scores aligned with that session's usable rows}}."""
    sess = {n: usable(rows, V2) for n, rows in raw_sessions.items()}
    info = {n: {"n": len(r), "liked": sum(x["thumb"] for x in r), "disliked": len(r) - sum(x["thumb"] for x in r)} for n, r in sess.items()}
    out = {"sessions": info}
    bad = [n for n, i in info.items() if i["n"] < MIN_SONGS or i["liked"] < MIN_PER_CLASS or i["disliked"] < MIN_PER_CLASS]
    if len(sess) < 2:
        return {**out, "verdict": "NOT ENOUGH DATA", "reason": "need at least two sessions"}
    if bad:
        return {**out, "verdict": "NOT ENOUGH DATA",
                "reason": "; ".join(f"{n}: {info[n]['n']} usable songs ({info[n]['liked']} liked, {info[n]['disliked']} disliked)" for n in bad)
                + f" (need >= {MIN_SONGS} with >= {MIN_PER_CLASS} of each thumb)"}
    v2 = leave_one_session_out(sess, V2)
    old = leave_one_session_out({n: usable(r, cr.DYNAMICS) for n, r in raw_sessions.items()}, cr.DYNAMICS)
    last3 = {n: fast_auc([x["features"]["last3_liking"] for x in r], [x["thumb"] for x in r]) for n, r in sess.items()}
    out.update({"loso_v2": v2, "loso_old_reaction_shape": old, "last3_liking": last3,
                "within_session_v2": {n: within_session_cv(r, V2) for n, r in sess.items()},
                "within_session_old": {n: within_session_cv(usable(raw_sessions[n], cr.DYNAMICS), cr.DYNAMICS) for n in sess},
                "single_feature_auc": {k: {n: fast_auc([x["features"][k] for x in r], [x["thumb"] for x in r]) for n, r in sess.items()} for k in V2}})
    pooled = [x for r in sess.values() for x in r]
    model = cr.fit_model(pooled, V2)
    out["weights_all_sessions"] = dict(zip(V2, [float(w) for w in model[0][:-1]]))
    out["context"] = {n: {k: fast_auc(np.asarray(v), [x["thumb"] for x in sess[n]]) for k, v in (c or {}).items()}
                      for n, c in (context or {}).items()}
    out["pass_each"] = {n: a >= PASS_AUC for n, a in v2.items()}
    out["verdict"] = "PASS" if all(out["pass_each"].values()) else "FAIL"
    return out


def format_text(r):
    lines = [f"{n}: {i['n']} usable songs ({i['liked']} liked, {i['disliked']} disliked)" for n, i in r["sessions"].items()]
    if r["verdict"].startswith("NOT ENOUGH"):
        return "\n".join(lines + [f"{r['verdict']}: {r['reason']}"])
    lines.append("Leave-one-session-out AUC (train on the other sessions, test on this one; mark >= %.2f on every session):" % PASS_AUC)
    for n in r["loso_v2"]:
        lines.append(f"  {n}: engagement features {r['loso_v2'][n]:.3f} | old reaction-shape {r['loso_old_reaction_shape'][n]:.3f} | "
                     f"last 3 s liking alone {r['last3_liking'][n]:.3f} | within-session CV {r['within_session_v2'][n]:.3f}")
        if n in r.get("context", {}) and r["context"][n]:
            lines.append("     context: " + ", ".join(f"{k} {v:.3f}" for k, v in r["context"][n].items()))
    lines.append(f"Overall: {r['verdict']}")
    return "\n".join(lines)


def build_report(r, tests, sessions):
    parts = ["# Step 3e: engagement features across sessions", "", f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
             f"- Sessions: {', '.join(sessions)}  ", "- Rule fixed beforehand: `step03e_preregistration.md`", "",
             "## Unit tests", "", f"- Ran {tests['ran']}: {tests['failed']} failed, {tests['errors']} errors, {tests['skipped']} skipped.", "",
             "## Result", "", "```", format_text(r), "```", ""]
    if "single_feature_auc" in r:
        parts += ["## Single features (AUC in each session; 0.5 = chance)", "", "| feature | " + " | ".join(r["sessions"]) + " |",
                  "|---|" + "---|" * len(r["sessions"])]
        for k, d in r["single_feature_auc"].items():
            parts.append(f"| {k} | " + " | ".join(f"{d[n]:.3f}" for n in r["sessions"]) + " |")
        parts += ["", "Weights of the model fitted on all sessions (descriptive only): "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in r["weights_all_sessions"].items()), ""]
    parts += ["## Limits", "", "- One person, two or three sessions: an AUC is known to about +/-0.1.",
              "- Stillness can mean concentration or attention elsewhere, not dislike.", ""]
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True, help="names of session files in data/agreement (e.g. adi_v2a adi_v2b)")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args(argv)
    d = ROOT / "data" / "agreement"
    raw = {}
    for n in a.sessions:
        if not (d / f"{n}.csv").exists():
            raise SystemExit(f"No session file {d / (n + '.csv')}")
        raw[n] = ag.load_rows(d / f"{n}.csv", n)
    context = {}
    try:
        from music.catalog.spotify_catalog import load_spotify
        by_id = {s.song_id: s for s in load_spotify()}
        for n, rows in raw.items():
            u = usable(rows, V2)
            songs = [by_id[x["song_id"]] for x in u if x["song_id"] in by_id]
            if len(songs) == len(u):
                context[n] = {"popularity": [s.audio_features.get("popularity", 0.0) for s in songs]}
    except Exception as e:
        print(f"(context skipped: {e})")
    res = evaluate(raw, context)
    print(format_text(res))
    if not a.no_report:
        from rl.report import run_unit_tests
        tests = run_unit_tests()
        path = ROOT / "reports" / "step03e_camera_v2_validation.md"
        path.write_text(build_report(res, tests, a.sessions))
        print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
        print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
