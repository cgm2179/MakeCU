"""Run from the repo root: python3 -m unittest rl.tests.test_camera_replication -v"""
import tempfile
import unittest

import numpy as np

from integration.orchestrator import run_agreement_check as rac
from rl import camera_replication as cr
from rl import personal_layer as pl
from rl.tests.test_eval import fake_catalog
from rl.tests.test_personal_layer import make_base


def rows(n, signal, seed, setting="quiet", conf=0.9):
    """Synthetic camera rows. With signal > 0 the thumb drives peak/build/range liking; with 0 the features are pure noise."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        thumb = int(rng.rand() < 0.4)
        f = {k: float(rng.randn() * 0.3) for k in cr.DYNAMICS}
        f["peak_liking"] += signal * thumb
        f["build_liking"] += signal * thumb * 0.8
        f["range_liking"] += signal * thumb * 0.6
        f["last3_liking"] += signal * thumb * 0.1          # the old reading carries little of it
        out.append({"user": "u", "song_id": f"s{seed}_{i}", "thumb": thumb, "liking": f["last3_liking"], "confidence": conf,
                    "emotion": "happy", "features": f, "setting": setting, "group_size": 1})
    return out


class TestEvaluate(unittest.TestCase):
    def test_real_signal_replicates_and_beats_the_old_reading(self):
        r = cr.evaluate(rows(80, 1.2, 1), rows(70, 1.2, 2), boot=200)
        self.assertEqual(r["verdict"], "PASS")
        self.assertGreaterEqual(r["auc"], cr.PASS_AUC)
        self.assertGreaterEqual(r["gain"], cr.PASS_GAIN)
        self.assertLessEqual(r["auc_lo_hi"][0], r["auc"])
        self.assertGreaterEqual(r["auc_lo_hi"][1], r["auc"])

    def test_noise_fails(self):
        r = cr.evaluate(rows(80, 1.2, 1), rows(70, 0.0, 3), boot=200)        # trained on a real signal, new data has none
        self.assertEqual(r["verdict"], "FAIL")
        self.assertLess(r["auc"], cr.PASS_AUC)

    def test_not_enough_new_data_is_reported_as_such_not_pass_or_fail(self):
        few = rows(30, 1.2, 4)
        r = cr.evaluate(rows(80, 1.2, 1), few)
        self.assertEqual(r["verdict"], "NOT ENOUGH DATA")
        skew = [dict(x, thumb=1) for x in rows(60, 1.2, 5)][:55] + [dict(x, thumb=0) for x in rows(60, 1.2, 6)][:5]
        self.assertEqual(cr.evaluate(rows(80, 1.2, 1), skew)["verdict"], "NOT ENOUGH DATA")
        self.assertEqual(cr.evaluate(rows(5, 1.2, 1), rows(70, 1.2, 2))["verdict"], "NOT ENOUGH TRAINING DATA")

    def test_low_confidence_and_busy_rows_are_dropped(self):
        weak = rows(70, 1.2, 2, conf=0.05)
        self.assertEqual(cr.usable_rows(weak), [])
        busy = rows(80, 1.2, 1, setting="busy")
        self.assertEqual(cr.evaluate(busy, rows(70, 1.2, 2))["verdict"], "NOT ENOUGH TRAINING DATA")

    def test_new_songs_never_influence_the_trained_model(self):
        train = rows(80, 1.2, 1)
        a = cr.fit_model(cr.usable_rows(train))
        cr.evaluate(train, rows(70, 1.2, 2), boot=50)
        cr.evaluate(train, rows(70, 0.0, 9), boot=50)
        b = cr.fit_model(cr.usable_rows(train))
        np.testing.assert_array_equal(a[0], b[0])

    def test_context_scores_are_reported(self):
        new = rows(70, 1.2, 2)
        ctx = {"popularity": [float(i) for i in range(len(new))]}
        r = cr.evaluate(rows(80, 1.2, 1), new, ctx, boot=50)
        self.assertIn("popularity", r["context"])
        self.assertIn("Overall: PASS", cr.format_text(r))


class TestSongOrder(unittest.TestCase):
    def test_ft_order_alternates_model_picks_with_random_and_keeps_every_song_once(self):
        cat = fake_catalog(100, seed=2)
        with tempfile.TemporaryDirectory() as d:
            base = make_base(d, favour_genre="jazz")
            layer = pl.PersonalLayer(base)
            order = rac.ft_order(cat, layer, seed=0)
        self.assertEqual(len(order), 100)
        self.assertEqual({s.song_id for s in order}, {s.song_id for s in cat})
        ranked = sorted(cat, key=lambda s: -layer.logit(s))
        self.assertEqual(order[0].song_id, ranked[0].song_id)               # first slot: the model's top pick
        top10 = {s.song_id for s in ranked[:10]}
        self.assertGreaterEqual(len(top10 & {s.song_id for s in order[:20]}), 8)    # top picks come early
        self.assertEqual([s.song_id for s in order], [s.song_id for s in rac.ft_order(cat, layer, seed=0)])


if __name__ == "__main__":
    unittest.main()
