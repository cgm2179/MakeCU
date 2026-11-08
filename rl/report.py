"""Per-step performance reports.

After every step of the plan (docs/ADI_CHECKLIST.txt) a markdown file is written to reports/:
  1. the unit-test results for that moment,
  2. how well the algorithm performs (metrics, per-fold values, learning curve),
  3. HOW each number was produced (data, split, seed, formula, a worked example),
  4. what the numbers do and do not show.

Run from the repo root:
    python3 -m rl.report --step 2                          # Step 2 report (SYNTHETIC labels)
    python3 -m rl.report --step 4 --labels data/ratings/adi.csv --user adi   # real labels

The reports hold only counts and aggregate metrics, never song lists or raw ratings.
"""
from __future__ import annotations

import argparse
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

from rl import eval as ev

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
TEST_MODULES = [
    "rl.tests.test_bandit", "rl.tests.test_eval", "rl.tests.test_profile", "rl.tests.test_report",
    "rl.tests.test_agreement", "rl.tests.test_signals", "rl.tests.test_ratings", "rl.tests.test_personalize", "rl.tests.test_live", "rl.tests.test_blend", "rl.tests.test_base_model", "rl.tests.test_base_lastfm", "rl.tests.test_personal_layer", "integration.tests.test_cv_to_rl", "motion.tests.test_arms",
    "music.tests.test_playlist", "music.tests.test_dataset_pipeline", "music.tests.test_audio",
    "music.tests.test_mainstream_bias", "music.tests.test_gemini_lookup",
    "integration.tests.test_core_loop", "integration.tests.test_camera_loop",
]


# ---------------------------------------------------------------- unit tests

def run_unit_tests(modules=TEST_MODULES, runner=None):
    """Run the unit tests. Returns {'ran','failed','errors','skipped','ok','per_module','seconds','failures'}."""
    cmd = runner or [sys.executable, "-m", "unittest", "-v", *modules]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    per = {}
    for line in out.splitlines():
        m = re.match(r"^\S+ \(([\w.]+)\.\w+\).*? \.\.\. (ok|FAIL|ERROR|skipped.*)$", line)
        if m:
            mod = _module_of(m.group(1))
            d = per.setdefault(mod, {"ok": 0, "bad": 0, "skipped": 0})
            res = m.group(2)
            d["ok" if res == "ok" else "skipped" if res.startswith("skipped") else "bad"] += 1
    ran = int((re.search(r"Ran (\d+) test", out) or [0, 0])[1])
    sk = re.search(r"skipped=(\d+)", out)
    fl = re.search(r"failures=(\d+)", out)
    er = re.search(r"errors=(\d+)", out)
    failures = [l for l in out.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
    return {"ran": ran, "skipped": int(sk[1]) if sk else 0, "failed": int(fl[1]) if fl else 0,
            "errors": int(er[1]) if er else 0, "ok": p.returncode == 0 and ran > 0, "per_module": per,
            "seconds": time.time() - t0, "failures": failures}


def _module_of(test_class_path):
    parts = test_class_path.split(".")
    return ".".join(parts[:3])


# ---------------------------------------------------------------- numbers

def git_info():
    def g(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    dirty = bool(g("status", "--porcelain"))
    return f"{g('rev-parse', '--abbrev-ref', 'HEAD')} @ {g('rev-parse', '--short', 'HEAD')}" + (" (+ uncommitted changes)" if dirty else "")


def per_fold_detail(items, models, n_folds=5, seed=0, k=10):
    """Per-fold AUC / precision@k for every model, on identical folds, plus fold sizes."""
    folds = ev.make_folds(items, n_folds, seed)
    detail = {name: [] for name in models}
    for i, test in enumerate(folds):
        train = [it for j, f in enumerate(folds) if j != i for it in f]
        for name, factory in models.items():
            a, p = ev.evaluate_fold(factory, train, test, k)
            detail[name].append({"auc": a, "p@k": p})
    sizes = [{"n_test": len(f), "liked": sum(1 for _, y in f if y)} for f in folds]
    return detail, sizes


def worked_auc_example():
    scores, labels = [0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0]
    return ("Worked example: scores 0.9, 0.8, 0.3, 0.1 with labels liked, disliked, liked, disliked. "
            "Liked songs scored 0.9 and 0.3; disliked scored 0.8 and 0.1. Of the 4 liked-vs-disliked pairs "
            "(0.9 vs 0.8, 0.9 vs 0.1, 0.3 vs 0.8, 0.3 vs 0.1) the model orders 3 correctly, so AUC = 3/4 = "
            f"{ev.auc(scores, labels):.2f}. (Checked by a unit test.)")


def _f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------- report body

def test_section(res):
    status = "ALL PASSED" if res["ok"] else "FAILURES"
    lines = [f"## 1. Unit tests: {status}", "",
             f"- Ran {res['ran']} tests in {res['seconds']:.1f}s: {res['ran'] - res['failed'] - res['errors'] - res['skipped']} passed, "
             f"{res['failed']} failed, {res['errors']} errors, {res['skipped']} skipped.",
             "- Skipped tests need the camera libraries or the Spotify CSV and are skipped on a plain Python install; "
             "a skip is not a pass.", ""]
    if res["per_module"]:
        rows = [(m, d["ok"], d["bad"], d["skipped"]) for m, d in sorted(res["per_module"].items())]
        lines.append(md_table(["test file", "passed", "failed", "skipped"], rows))
    for f in res["failures"]:
        lines.append(f"- {f}")
    return "\n".join(lines)


def performance_section(items, models, source_note, n_folds=5, seed=0, k=10):
    detail, sizes = per_fold_detail(items, models, n_folds, seed, k)
    n, likes = len(items), sum(1 for _, y in items if y)
    base_rate = likes / n
    lines = ["## 2. How well the algorithm performs", "",
             f"Data: {source_note}", f"- {n} labeled songs: {likes} liked ({base_rate:.0%}), {n - likes} disliked. "
             f"A model that always says 'liked' would be right {max(base_rate, 1 - base_rate):.0%} of the time, "
             "so read precision@10 against that, and prefer AUC (it ignores the like/dislike ratio).", ""]
    rows = []
    for name, folds in detail.items():
        aucs = [f["auc"] for f in folds if f["auc"] is not None]
        ps = [f["p@k"] for f in folds if f["p@k"] is not None]
        sd = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        rows.append((name, _f(statistics.mean(aucs)), _f(sd), _f(statistics.mean(ps)),
                     ", ".join(_f(a, 2) for a in aucs)))
    lines.append(md_table(["model", "mean AUC", "sd across folds", f"mean precision@{k}", "AUC per fold"], rows))
    # paired comparison on identical folds
    names = list(detail)
    if "bandit (from scratch)" in detail and "popularity-only" in detail:
        b = [f["auc"] for f in detail["bandit (from scratch)"]]
        p = [f["auc"] for f in detail["popularity-only"]]
        diffs = [x - y for x, y in zip(b, p) if x is not None and y is not None]
        wins = sum(1 for d in diffs if d > 0)
        lines += ["", f"Bandit minus popularity-only, per fold: {', '.join(f'{d:+.2f}' for d in diffs)}. "
                      f"Mean {statistics.mean(diffs):+.3f}; the bandit is ahead in {wins} of {len(diffs)} folds."]
    sizes_txt = ", ".join(f"{s['n_test']} ({s['liked']} liked)" for s in sizes)
    lines += ["", f"Test-fold sizes: {sizes_txt}.", ""]

    # learning curve
    full = n - n // n_folds
    ns = sorted({m for m in ev.LEARNING_SIZES if m < full} | {full})
    curves = {name: ev.learning_curve(items, f, ns, n_folds, seed, k) for name, f in models.items()}
    cols = sorted({m for c in curves.values() for m in c})
    lines += ["### Learning curve (mean AUC when only the first n training labels are used)", "",
              md_table(["model"] + [f"n={m}" for m in cols],
                       [[name] + [_f(c.get(m)) for m in cols] for name, c in curves.items()])]
    return "\n".join(lines), detail


def method_section(n_folds, seed, k):
    return "\n".join([
        "## 3. How the numbers were produced", "",
        "- **Labels:** each song has one label, 1 = liked, 0 = disliked. Songs come from the 80,834-song Spotify dataset.",
        f"- **Split:** {n_folds}-fold cross-validation, stratified (likes and dislikes are dealt out separately so every fold has both). "
        f"Each fold is tested once, using a model trained only on the other {n_folds - 1} folds. Shuffle seed = {seed}, "
        "so rerunning gives identical folds. The final held-out songs (if any) are never used here.",
        "- **AUC:** take every (liked, disliked) pair in the test fold; count 1 if the model scored the liked song higher, "
        "0.5 for a tie, 0 otherwise; divide by the number of pairs. 0.5 = no better than chance, 1.0 = perfect.",
        f"- **precision@{k}:** rank the test-fold songs by model score, take the top {k}, and report the share that were liked. "
        "Ties are broken randomly with a fixed seed, so a model that scores everything equally gets chance level.",
        "- **random:** a random score per song. **popularity-only:** the dataset's popularity (0-100) as the score. "
        "**bandit (from scratch):** the live learner (rl/agent/bandit.py) starting at zero weights and learning from the "
        "training labels once each, in shuffled order, thumbs up = +1 and thumbs down = -1 at full confidence; "
        "a song's score is the sum of its learned feature weights (genre, artist, mood, energy/valence/danceability/tempo buckets).",
        "- **Learning curve:** same test folds, but the model sees only the first n training labels, so you can see how fast it learns.",
        "- **Worked example of AUC.** Take scores " + worked_auc_example().removeprefix("Worked example: scores "), ""])


def caveats_section(synthetic, n):
    lines = ["## 4. What these numbers do and do not show", ""]
    if synthetic:
        lines += ["- **The labels are synthetic.** They come from a fake listener who likes a fixed set of genres (with 10% random flips). "
                  "This shows the measuring tool works and that the bandit can learn a genre taste. It says nothing about how well "
                  "the robot will do for a real person.",
                  "- Real labels (Step 4) will replace this data; rerun `python3 -m rl.report` then."]
    else:
        lines += ["- These labels come from a real person's ratings, so they measure this person's taste only."]
    lines += [f"- With {n} songs and 5 folds, each fold tests only about {n // 5} songs, so a single fold's AUC is noisy "
              "(the 'sd across folds' column shows how noisy). Treat differences smaller than about 0.05 as unproven.",
              "- Folds share training data, so the fold scores are not independent; the standard deviation is a rough guide, not a confidence interval.",
              "- Not yet covered: the base model (Layer 1), personal fine-tuning (Layer 2) and a live test (Layer 3)."]
    return "\n".join(lines)


def build_report(step, title, items, models, source_note, synthetic, test_result, n_folds=5, seed=0, k=10, extra=""):
    perf, _ = performance_section(items, models, source_note, n_folds, seed, k)
    parts = [f"# Step {step} report: {title}", "",
             f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ", f"- Code: {git_info()}  ",
             f"- Python {platform.python_version()} on {platform.system()}", "",
             test_section(test_result), "", perf, "", method_section(n_folds, seed, k), "",
             caveats_section(synthetic, len(items))]
    if extra:
        parts += ["", extra]
    return "\n".join(parts) + "\n"


def write_report(step, slug, text, directory=None):
    d = Path(directory or REPORT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"step{int(step):02d}_{slug}.md"
    path.write_text(text)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write the per-step performance report to reports/.")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--labels")
    ap.add_argument("--user")
    ap.add_argument("--agreement", help="Step 3: CSV written by run_agreement_check")
    ap.add_argument("--ratings", help="Step 4/5: ratings CSV written by run_rating_session")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.ratings:  # Steps 4 and 5: quality vs preference ratings
        from rl import ratings as rtg
        print("Running unit tests...")
        tests = run_unit_tests()
        rows = rtg.load_rows(args.ratings, args.user)
        catalog_by_id = None
        try:
            from music.catalog.spotify_catalog import dataset_available, load_spotify
            if dataset_available():
                catalog_by_id = {x.song_id: x for x in load_spotify()}
        except Exception:
            pass
        text = rtg.build_ratings_report(rows, tests, args.user or "", catalog_by_id, synthetic=Path(args.ratings).stem == "demo")
        stem = Path(args.ratings).stem
        path = write_report(args.step, "ratings" if stem == "adi" else f"ratings_{stem}", text)
        print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
        print(rtg.format_text(rtg.analyse(rows, catalog_by_id)))
        print(f"Report written to {path}")
        return 0 if tests["ok"] else 1

    if args.agreement:  # Step 3 has its own body: camera vs thumbs
        from rl import agreement as ag
        print("Running unit tests...")
        tests = run_unit_tests()
        rows = ag.load_rows(args.agreement, args.user)
        text = ag.build_agreement_report(rows, tests, args.user or "", synthetic=Path(args.agreement).stem == "demo")
        stem = Path(args.agreement).stem
        path = write_report(args.step, "camera_agreement" if stem == "adi" else f"camera_agreement_{stem}", text)
        print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
        print(ag.verdict_text(ag.analyse(rows)))
        print(f"Report written to {path}")
        return 0 if tests["ok"] else 1

    from music.catalog.spotify_catalog import dataset_available, load_spotify
    if not dataset_available():
        sys.exit("The Spotify dataset is missing (data/spotify/dataset.csv).")
    catalog = load_spotify()
    models = ev.default_models(args.seed)
    print("Running unit tests...")
    tests = run_unit_tests()
    if args.labels:
        rows = ev.load_labels(args.labels, args.user)
        dev_rows, final_rows = ev.split_final(rows, seed=args.seed)
        items, missing = ev.attach_songs(dev_rows, catalog)
        note = (f"real ratings from `{Path(args.labels).name}`" + (f" for user '{args.user}'" if args.user else "")
                + f"; {len(final_rows)} held-out final-test rows excluded; {missing} rows skipped (song not in dataset).")
        synthetic = False
    else:
        items = ev.synthetic_labels(catalog, seed=args.seed)
        note = "**SYNTHETIC** labels from a fake listener (likes jazz, classical, acoustic, piano; 10% flipped)."
        synthetic = True
    titles = {2: ("Evaluation harness and profile save/load", "eval_harness")}
    title, slug = titles.get(args.step, (f"Step {args.step}", f"step{args.step}"))
    text = build_report(args.step, title, items, models, note, synthetic, tests, seed=args.seed)
    path = write_report(args.step, slug, text)
    print(f"Tests: ran {tests['ran']}, failed {tests['failed']}, errors {tests['errors']}, skipped {tests['skipped']}")
    print(f"Report written to {path}")
    return 0 if tests["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
