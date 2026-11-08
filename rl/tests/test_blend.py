"""Run from the repo root: python3 -m unittest rl.tests.test_blend -v"""
import unittest

from rl.blend import BlendModel, repeated_cv_auc
from rl.eval import PopularityModel
from rl.tests.test_eval import fake_catalog

CAT = fake_catalog(120)
ITEMS = [(s, 1 if s.audio_features["popularity"] > 50 else 0) for s in CAT]


class TestBlend(unittest.TestCase):
    def test_w1_ranks_like_popularity_and_w0_like_the_bandit(self):
        pop, b1, b0 = PopularityModel(), BlendModel(1.0), BlendModel(0.0)
        for m in (pop, b1, b0):
            m.fit(ITEMS)
        a = sorted(CAT[:30], key=lambda s: pop.score(s))
        self.assertEqual(a, sorted(CAT[:30], key=lambda s: b1.score(s)))
        c = sorted(CAT[:30], key=lambda s: b0.bandit.score(s))
        self.assertEqual(c, sorted(CAT[:30], key=lambda s: b0.score(s)))

    def test_cv_is_a_fair_number_and_beats_chance_when_popularity_is_the_label(self):
        mean, sd = repeated_cv_auc(ITEMS, lambda: BlendModel(0.9), repeats=2)
        self.assertGreater(mean, 0.9)
        self.assertGreaterEqual(sd, 0.0)

    def test_random_labels_give_chance_level(self):
        import random
        rng = random.Random(0)
        items = [(s, rng.randint(0, 1)) for s in CAT]
        mean, _ = repeated_cv_auc(items, lambda: BlendModel(0.5), repeats=3)
        self.assertLess(abs(mean - 0.5), 0.15)


if __name__ == "__main__":
    unittest.main()
