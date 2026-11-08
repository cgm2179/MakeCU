"""Run from the repo root: python3 -m unittest rl.tests.test_personal_layer -v"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl import base_lastfm as bl
from rl import personal_layer as pl
from rl.base_model import Featurizer
from rl.tests.test_eval import GENRES, fake_catalog

CAT = fake_catalog(240, seed=3)
FEAT = Featurizer.fit(CAT)
D = len(FEAT.names)


def make_base(tmp, favour_genre=None, seed=0, name="b.json"):
    rng = np.random.RandomState(seed)
    w = rng.randn(D) * 0.05
    if favour_genre:
        w[FEAT.names.index(f"genre:{favour_genre}")] = 2.0
    path = bl.save_base(Path(tmp) / name, FEAT, w, -1.0, rng.randn(D) * 0.1, 0.5, {"epoch": 1})
    return bl.load_base(path)


class TestFit(unittest.TestCase):
    def test_gradient_is_zero_at_the_solution_and_bias_is_unpenalised(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.randn(120, 4), np.ones((120, 1))])
        y = (X[:, 0] + 0.5 * rng.randn(120) > 0).astype(float)
        off = rng.randn(120) * 0.3
        w = pl.fit_logistic(X, y, off, lam=0.1)
        p = pl.sigmoid(off + X @ w)
        g = X.T @ (p - y) / 120
        g[:-1] += 0.1 * w[:-1]
        self.assertLess(np.abs(g).max(), 1e-6)
        self.assertGreater(w[0], 0.5)

    def test_strong_penalty_keeps_weights_near_zero(self):
        rng = np.random.RandomState(1)
        X = np.hstack([rng.randn(100, 3), np.ones((100, 1))])
        y = (X[:, 0] > 0).astype(float)
        self.assertLess(np.abs(pl.fit_logistic(X, y, np.zeros(100), lam=1000.0)[:-1]).max(), 0.01)

    def test_fast_auc_matches_brute_force_with_ties(self):
        rng = np.random.RandomState(2)
        s = rng.randint(0, 5, 50).astype(float)
        y = (rng.rand(50) < 0.4).astype(int)
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
        self.assertAlmostEqual(pl.fast_auc(s, y), brute)


class TestL21(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = make_base(self.tmp.name)
        liked = {"jazz", "folk"}
        self.items = [(s, 1 if s.genre in liked else 0) for s in CAT[:120]]

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_zero_user_data_is_exactly_the_base(self):
        layer = pl.PersonalLayer(self.base)
        for s in CAT:
            self.assertEqual(layer.logit(s), self.base.preference_logit(s))
            self.assertEqual(layer.prob(s), self.base.prob(s))

    def test_b_base_file_is_bit_for_bit_unchanged_by_fine_tuning(self):
        path = Path(self.tmp.name) / "b.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        mode = os.stat(path).st_mode
        layer = pl.PersonalLayer.fit(self.base, self.items, lam=0.1)
        pl.save_personal(layer, Path(self.tmp.name) / "p.json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(os.stat(path).st_mode, mode)
        self.assertEqual(self.base.sha256, bl.load_base(path).sha256)
        self.assertTrue(np.any(layer.delta != 0))                       # it did learn something

    def test_c_reset_returns_the_base_and_a_different_base_is_refused(self):
        layer = pl.PersonalLayer.fit(self.base, self.items, lam=0.1)
        self.assertNotEqual(layer.logit(CAT[0]), self.base.preference_logit(CAT[0]))
        layer.reset()
        for s in CAT[:40]:
            self.assertEqual(layer.logit(s), self.base.preference_logit(s))
        layer = pl.PersonalLayer.fit(self.base, self.items, lam=0.1)
        p = pl.save_personal(layer, Path(self.tmp.name) / "p.json")
        again = pl.load_personal(p, self.base)
        for s in CAT[:40]:
            self.assertAlmostEqual(again.logit(s), layer.logit(s), places=6)
        other = make_base(self.tmp.name, seed=9, name="other.json")
        self.assertNotEqual(other.sha256, self.base.sha256)
        with self.assertRaises(ValueError):
            pl.load_personal(p, other)

    def test_layer_is_tiny(self):
        layer = pl.PersonalLayer.fit(self.base, self.items, lam=0.1)
        self.assertLess(pl.personal_device_bytes(layer), 1024)


class TestCV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # a base that loves rock, and a person who loves jazz and folk
        self.base = make_base(self.tmp.name, favour_genre="rock")
        liked = {"jazz", "folk"}
        self.items = [(s, 1 if s.genre in liked else 0) for s in CAT]

    def tearDown(self):
        self.tmp.cleanup()

    def test_fine_tuning_beats_a_base_that_disagrees_with_the_person(self):
        res = pl.run_cv(self.items, self.base, repeats=2, log=None)
        m = res["mean_auc"]
        self.assertGreater(m["fine-tuned"], m["base-only"] + 0.2)
        self.assertGreater(m["fine-tuned"], 0.85)
        self.assertEqual(set(res["mean_auc"]), set(pl.MODELS))

    def test_cv_is_deterministic_and_never_scores_a_song_with_a_model_that_saw_it(self):
        a = pl.run_cv(self.items, self.base, repeats=1, log=None)
        b = pl.run_cv(self.items, self.base, repeats=1, log=None)
        self.assertEqual(a["mean_auc"], b["mean_auc"])
        # random labels: nothing should beat chance by much (a leak would show up as a high AUC)
        rng = np.random.RandomState(0)
        noise = [(s, int(rng.rand() < 0.4)) for s, _ in self.items]
        r = pl.run_cv(noise, self.base, repeats=2, log=None)
        for m in ("logistic (scratch)", "fine-tuned"):
            self.assertLess(r["mean_auc"][m], 0.65, m)

    def test_difference_range_contains_the_point_estimate(self):
        res = pl.run_cv(self.items, self.base, repeats=1, log=None)
        d, lo, hi = pl.boot_diff(res["avg_oof"]["fine-tuned"], res["avg_oof"]["base-only"], res["y"], n=200)
        self.assertLessEqual(lo, d)
        self.assertGreaterEqual(hi, d)

    def test_learning_curve_and_graph(self):
        lams = {"logistic (scratch)": 1.0, "fine-tuned": 1.0}
        curve, full = pl.learning_curve(self.items, self.base, lams, repeats=1, sizes=(10, 20, 40), log=None)
        self.assertEqual(sorted(curve["fine-tuned"]), [10, 20, 40, full])
        self.assertGreater(curve["fine-tuned"][full], curve["fine-tuned"][10] - 0.05)
        svg = pl.curve_svg(curve, full, 0.5)
        self.assertTrue(svg.startswith("<svg") and svg.rstrip().endswith("</svg>"))
        self.assertEqual(svg.count("<polyline"), 4)


if __name__ == "__main__":
    unittest.main()
