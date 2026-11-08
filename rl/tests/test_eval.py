"""Run from the repo root: python3 -m unittest rl.tests.test_eval -v"""
import csv
import random
import tempfile
import unittest
from pathlib import Path

from rl import eval as ev
from shared.schemas.messages import Song

GENRES = ["jazz", "rock", "pop", "metal", "folk", "techno"]
LIKED = {"jazz", "folk"}


def fake_catalog(n=150, seed=1):
    rng = random.Random(seed)
    return [Song(f"t{i}", f"title{i}", f"artist{i % 25}", GENRES[i % len(GENRES)],
                 {"energy": rng.random(), "valence": rng.random(), "popularity": rng.randint(0, 100)},
                 "calm") for i in range(n)]


def fake_items(n=150, noise=0.0, seed=1):
    rng = random.Random(seed)
    return [(s, int((s.genre in LIKED) != (rng.random() < noise))) for s in fake_catalog(n, seed)]


class TestMetrics(unittest.TestCase):
    def test_auc_known_answers(self):
        self.assertEqual(ev.auc([.9, .8, .3, .1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(ev.auc([.1, .3, .8, .9], [1, 1, 0, 0]), 0.0)
        self.assertEqual(ev.auc([.5, .5, .5, .5], [1, 0, 1, 0]), 0.5)
        # liked: .9,.3  disliked: .8,.1 -> pairs won: .9>.8, .9>.1, .3>.1 = 3 of 4
        self.assertEqual(ev.auc([.9, .8, .3, .1], [1, 0, 1, 0]), 0.75)
        self.assertIsNone(ev.auc([.1, .2], [1, 1]))  # one class: undefined

    def test_precision_at_k_known_answers(self):
        scores = [10, 9, 8, 7, 6, 5]
        self.assertEqual(ev.precision_at_k(scores, [1, 1, 0, 0, 1, 1], k=2), 1.0)
        self.assertEqual(ev.precision_at_k(scores, [1, 0, 1, 0, 0, 0], k=4), 0.5)
        self.assertEqual(ev.precision_at_k([1, 2], [1, 0], k=10), 0.5)  # k larger than the data

    def test_ties_do_not_depend_on_file_order(self):
        same = [1.0] * 100
        labels = [1] * 50 + [0] * 50  # liked songs listed first would win with a stable sort
        self.assertLess(ev.precision_at_k(same, labels, k=10), 1.0)


class TestCrossValidation(unittest.TestCase):
    def test_folds_are_stratified_and_disjoint(self):
        items = fake_items(100)
        folds = ev.make_folds(items, 5)
        self.assertEqual(sorted(s.song_id for f in folds for s, _ in f), sorted(s.song_id for s, _ in items))
        for f in folds:
            self.assertTrue(any(y for _, y in f) and any(not y for _, y in f))

    def test_bandit_beats_random_and_popularity_on_a_genre_taste(self):
        items = fake_items(150, noise=0.05)
        res = {n: ev.cross_validate(items, f) for n, f in ev.default_models().items()}
        mean = {n: sum(r["auc"]) / len(r["auc"]) for n, r in res.items()}
        print("\n  mean AUC:", {n: round(v, 3) for n, v in mean.items()})
        self.assertGreater(mean["bandit (from scratch)"], 0.85)
        self.assertLess(mean["random"], 0.65)
        self.assertLess(mean["popularity-only"], 0.65)

    def test_learning_curve_rises_with_more_labels(self):
        items = fake_items(150, noise=0.05)
        c = ev.learning_curve(items, ev.default_models()["bandit (from scratch)"], sizes=(10, 120))
        self.assertGreater(c[120], c[10])

    def test_report_prints_all_three_baselines(self):
        lines = []
        ev.report_cv(fake_items(60), ev.default_models(), out=lines.append)
        text = "\n".join(lines)
        for name in ("random", "popularity-only", "bandit (from scratch)", "Learning curve"):
            self.assertIn(name, text)


class TestFinalTest(unittest.TestCase):
    def test_runs_once_only(self):
        items = fake_items(80)
        dev, final = items[:50], items[50:]
        with tempfile.TemporaryDirectory() as d:
            ev.run_final(dev, final, ev.default_models(), "adi", directory=d, out=lambda *_: None)
            with self.assertRaises(SystemExit):
                ev.run_final(dev, final, ev.default_models(), "adi", directory=d, out=lambda *_: None)

    def test_split_uses_marked_rows_else_holds_out_n(self):
        rows = [{"song_id": str(i), "label": 1, "split": "test" if i < 5 else ""} for i in range(40)]
        dev, final = ev.split_final(rows)
        self.assertEqual((len(dev), len(final)), (35, 5))
        rows2 = [{"song_id": str(i), "label": 1, "split": ""} for i in range(100)]
        dev, final = ev.split_final(rows2, n=30)
        self.assertEqual((len(dev), len(final)), (70, 30))
        self.assertEqual(ev.split_final(rows2, n=30)[1], final)  # deterministic


class TestLabelFiles(unittest.TestCase):
    def test_load_filters_user_and_parses_labels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["user", "song_id", "label"])
                w.writerows([["adi", "a", "1"], ["adi", "b", "0"], ["bo", "c", "like"]])
            self.assertEqual([(r["song_id"], r["label"]) for r in ev.load_labels(p, "adi")], [("a", 1), ("b", 0)])
            self.assertEqual(len(ev.load_labels(p)), 3)
        with self.assertRaises(ValueError):
            ev.parse_label("maybe")

    def test_attach_counts_unknown_songs(self):
        cat = fake_catalog(3)
        items, missing = ev.attach_songs([{"song_id": "t0", "label": 1}, {"song_id": "zzz", "label": 0}], cat)
        self.assertEqual((len(items), missing), (1, 1))


if __name__ == "__main__":
    unittest.main()
