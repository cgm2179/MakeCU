"""Run from the repo root: python3 -m unittest rl.tests.test_base_model -v"""
import json
import tempfile
import unittest
from pathlib import Path

from rl import base_model as bm
from rl.tests.test_eval import fake_catalog

CAT = fake_catalog(200)
ITEMS = [(s, 1 if s.audio_features["valence"] > 0.55 else 0) for s in CAT]


class TestSplit(unittest.TestCase):
    def test_disjoint_stratified_and_reproducible(self):
        tr, va, te = bm.split_items(ITEMS, seed=1)
        ids = [s.song_id for s, _ in tr + va + te]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(ITEMS))
        for part in (tr, va, te):
            self.assertTrue(0 < sum(y for _, y in part) < len(part))
        self.assertEqual([s.song_id for s, _ in bm.split_items(ITEMS, seed=1)[2]], [s.song_id for s, _ in te])
        self.assertAlmostEqual(len(tr) / len(ITEMS), 0.70, delta=0.03)


class TestFeaturizer(unittest.TestCase):
    def test_vector_length_matches_names_and_missing_values_become_the_mean(self):
        f = bm.Featurizer.fit(CAT)
        v = f.vector(CAT[0])
        self.assertEqual(len(v), len(f.names))
        from dataclasses import replace
        bare = replace(CAT[0], audio_features={})
        self.assertTrue(all(x == 0.0 for x in f.vector(bare)[:len(bm.NUMERIC)]))

    def test_rare_genres_are_dropped(self):
        f = bm.Featurizer.fit(CAT)
        counts = {}
        for s in CAT:
            counts[s.genre] = counts.get(s.genre, 0) + 1
        self.assertTrue(all(counts[g] >= bm.MIN_GENRE_COUNT for g in f.genres))


class TestTrain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def run_train(self, **kw):
        return bm.train(ITEMS, seed=0, epochs=kw.pop("epochs", 120), checkpoint_every=kw.pop("checkpoint_every", 40),
                        directory=self.tmp.name, log=None, **kw)

    def test_learns_the_rule_and_loss_goes_down(self):
        res = self.run_train(epochs=200)
        h = res["history"]
        self.assertEqual(len(h), 201)
        self.assertLess(h[-1]["train_loss"], h[0]["train_loss"])
        self.assertGreater(h[-1]["val_auc"], 0.9)
        te = res["splits"][2]
        self.assertGreater(bm.evaluate(res["model_best"], te)["auc"], 0.9)

    def test_checkpoints_saved_at_every_interval_and_load(self):
        res = self.run_train(epochs=120, checkpoint_every=40)
        self.assertEqual([e for e, _ in res["checkpoints"]], [0, 40, 80, 120])
        for e, p in res["checkpoints"]:
            m = bm.load_model(p)
            self.assertEqual(m.meta["epoch"], e)
        # checkpoint 0 is untrained: every score is the same
        m0 = bm.load_model(res["checkpoints"][0][1])
        self.assertEqual(len({round(m0.score(s), 9) for s in CAT[:20]}), 1)

    def test_best_model_has_the_lowest_validation_loss(self):
        res = self.run_train(epochs=150)
        best_epoch = min(res["history"], key=lambda r: r["val_loss"])["epoch"]
        self.assertEqual(res["best_epoch"], best_epoch)

    def test_featurizer_is_fitted_on_training_songs_only(self):
        seen = []
        real = bm.Featurizer.fit.__func__

        def spy(cls, songs):
            seen.append({x.song_id for x in songs})
            return real(cls, songs)
        bm.Featurizer.fit = classmethod(spy)
        try:
            res = self.run_train(epochs=5)
        finally:
            bm.Featurizer.fit = classmethod(real)
        tr, va, te = res["splits"]
        self.assertEqual(seen, [{s.song_id for s, _ in tr}])
        self.assertFalse(seen[0] & {s.song_id for s, _ in va + te})

    def test_test_labels_do_not_change_the_weights(self):
        a = self.run_train(epochs=30)
        test_ids = {s.song_id for s, _ in a["splits"][2]}
        # same split, but different (scrambled) test labels: a model that never reads them is unchanged
        scrambled = [(s, 1 - y) if s.song_id in test_ids else (s, y) for s, y in ITEMS]
        orig = bm.split_items
        bm.split_items = lambda items, fractions=(0.70, 0.15, 0.15), seed=0: (
            [it for it in scrambled if it[0].song_id in {t.song_id for t, _ in a["splits"][0]}],
            [it for it in scrambled if it[0].song_id in {t.song_id for t, _ in a["splits"][1]}],
            [it for it in scrambled if it[0].song_id in test_ids])
        try:
            b = bm.train(scrambled, seed=0, epochs=30, checkpoint_every=30, directory=self.tmp.name, log=None)
        finally:
            bm.split_items = orig
        self.assertEqual(a["model_last"].payload()["weights"], b["model_last"].payload()["weights"])


class TestExport(unittest.TestCase):
    def test_round_trip_gives_identical_scores_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            res = bm.train(ITEMS, epochs=40, checkpoint_every=40, directory=d, log=None)
            path = bm.save_model(res["model_best"], Path(d) / "m.json", {"user": "x"})
            loaded = bm.load_model(path)
            for s in CAT[:30]:
                self.assertAlmostEqual(res["model_best"].score(s), loaded.score(s), places=6)
                self.assertAlmostEqual(bm.predict_from_file(path, s), loaded.score(s), places=9)
            doc = json.loads(Path(path).read_text())
            doc["model"]["bias"] += 0.5
            Path(path).write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                bm.load_model(path)
            Path(path).write_text(json.dumps({"format": "other", "version": 1}))
            with self.assertRaises(ValueError):
                bm.load_model(path)


class TestGraph(unittest.TestCase):
    def test_svg_has_both_panels_and_marks_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            res = bm.train(ITEMS, epochs=80, checkpoint_every=40, directory=d, log=None)
        svg = bm.training_graph_svg(res["history"], [e for e, _ in res["checkpoints"]], res["best_epoch"], 0.7)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.rstrip().endswith("</svg>"))
        self.assertEqual(svg.count("<polyline"), 4)
        self.assertIn("popularity-only", svg)


if __name__ == "__main__":
    unittest.main()
