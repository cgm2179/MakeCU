"""Run from the repo root: python3 -m unittest rl.tests.test_prior_learner -v"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl import live_test as lt
from rl import replay as rp
from rl.agent.prior_learner import PriorLearner
from rl.personal_layer import fit_logistic
from rl.tests.test_eval import fake_catalog
from rl.tests.test_personal_layer import make_base

CAT = fake_catalog(160, seed=5)
LIKED = {"jazz", "folk"}


class TestSampleWeights(unittest.TestCase):
    def test_weight_k_equals_k_copies_and_soft_labels_work(self):
        rng = np.random.RandomState(0)
        X = np.hstack([rng.randn(30, 3), np.ones((30, 1))])
        y = (X[:, 0] + rng.randn(30) * 0.5 > 0).astype(float)
        off = rng.randn(30) * 0.2
        wts = rng.randint(1, 4, 30).astype(float)
        a = fit_logistic(X, y, off, 0.5, sample_weight=wts)
        idx = np.repeat(np.arange(30), wts.astype(int))
        b = fit_logistic(X[idx], y[idx], off[idx], 0.5)
        np.testing.assert_allclose(a, b, atol=1e-8)
        soft = fit_logistic(X, np.clip(y * 0.8 + 0.1, 0, 1), off, 0.5)                    # fractional targets are fine
        self.assertTrue(np.all(np.isfinite(soft)))


class TestPriorLearner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = make_base(self.tmp.name, favour_genre="rock")        # a prior that disagrees with the person

    def tearDown(self):
        self.tmp.cleanup()

    def test_with_no_feedback_it_follows_the_prior_exactly(self):
        pl = PriorLearner(self.base, "base", beta=0.0, seed=1)
        for s in CAT[:30]:
            self.assertEqual(pl.logit(s), self.base.preference_logit(s))
        top = max(CAT, key=lambda s: self.base.preference_logit(s))
        self.assertEqual(pl.choose(CAT)[0].song_id, top.song_id)
        pop = PriorLearner(self.base, "popularity", beta=0.0, seed=1)
        self.assertEqual(pop.choose(CAT)[0].audio_features["popularity"], max(s.audio_features["popularity"] for s in CAT))

    def test_it_learns_a_taste_that_disagrees_with_the_prior(self):
        pl = PriorLearner(self.base, "base", beta=0.0, seed=1)
        before = np.mean([pl.prob(s) for s in CAT if s.genre in LIKED]) - np.mean([pl.prob(s) for s in CAT if s.genre not in LIKED])
        for s in CAT[:60]:
            pl.update_from_signals(s, thumb=1 if s.genre in LIKED else 0)
        rest = CAT[60:]
        gap = np.mean([pl.prob(s) for s in rest if s.genre in LIKED]) - np.mean([pl.prob(s) for s in rest if s.genre not in LIKED])
        self.assertGreater(gap, before + 0.15)
        picks = [pl.choose([s for s in rest if s.song_id not in {p.song_id for p in []}])[0] for _ in range(1)]
        self.assertIn(picks[0].genre, LIKED)

    def test_reset_returns_to_the_prior(self):
        pl = PriorLearner(self.base, "base", beta=0.0)
        for s in CAT[:20]:
            pl.update_from_signals(s, thumb=1)
        self.assertNotEqual(pl.logit(CAT[0]), self.base.preference_logit(CAT[0]))
        pl.reset()
        self.assertEqual(pl.logit(CAT[0]), self.base.preference_logit(CAT[0]))
        self.assertEqual(pl.n_updates, 0)

    def test_signals_thumbs_and_behaviour_count_the_camera_only_when_validated(self):
        song = CAT[0]
        pl = PriorLearner(self.base, "base")
        self.assertEqual(pl.update_from_signals(song, thumb=1), 1.0)
        self.assertAlmostEqual(pl.update_from_signals(song, behaviour="replaced"), -0.7)
        self.assertIsNone(pl.update_from_signals(song))                                  # nothing to learn from
        self.assertEqual(pl.n_updates, 2)
        off = PriorLearner(self.base, "base", camera_validated=False)
        self.assertEqual(off.update_from_signals(song, camera=-1.0, camera_weight=0.8), None)      # camera alone is ignored
        self.assertEqual(off.n_updates, 0)
        on = PriorLearner(self.base, "base", camera_validated=True)
        self.assertAlmostEqual(on.update_from_signals(song, camera=-1.0, camera_weight=0.8), -0.8)
        both = PriorLearner(self.base, "base", camera_validated=True)
        self.assertAlmostEqual(both.update_from_signals(song, thumb=1, camera=-1.0, camera_weight=0.5), (1.0 - 0.5) / 1.5)

    def test_uncertainty_is_higher_for_unfamiliar_songs_and_shrinks_with_feedback(self):
        pl = PriorLearner(self.base, "base", beta=1.0)
        s = CAT[0]
        u0 = pl.uncertainty(s)
        for t in [x for x in CAT if x.genre == s.genre][:15]:
            pl.update_from_signals(t, thumb=1)
        self.assertLess(pl.uncertainty(s), u0)
        other = next(x for x in CAT if x.genre not in {t.genre for t in CAT[:0]} and x.genre != s.genre)
        self.assertGreater(pl.uncertainty(other), pl.uncertainty(s))

    def test_recent_songs_are_skipped_and_bad_offset_rejected(self):
        pl = PriorLearner(self.base, "base", beta=0.0)
        top = pl.choose(CAT)[0]
        self.assertNotEqual(pl.choose(CAT, recent_ids=[top.song_id])[0].song_id, top.song_id)
        with self.assertRaises(ValueError):
            PriorLearner(self.base, "nope")


class _SpyLabels(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.asked = []

    def __getitem__(self, k):
        self.asked.append(k)
        return super().__getitem__(k)


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = make_base(self.tmp.name, favour_genre="rock")
        self.by_id = {s.song_id: s for s in CAT}
        self.labels = {s.song_id: int(s.genre in LIKED) for s in CAT}

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_policy_runs_and_a_label_is_only_revealed_when_a_song_is_picked(self):
        for name in rp.POLICIES:
            spy = _SpyLabels(self.labels)
            t = rp.run_policy(name, CAT[:80], spy, self.base, steps=15, seed=3)
            self.assertEqual(len(t), 15, name)
            self.assertEqual(len(spy.asked), 15, name)
            self.assertEqual(len(set(spy.asked)), 15, name)              # no song picked twice

    def test_learners_beat_random_when_there_is_a_taste_to_learn(self):
        res = rp.replay(self.by_id, self.labels, self.base, sessions=12, pool_size=80, steps=30, log=None,
                        policies=("random", "base + online learner (beta 0)", "base-only"))
        rows = rp.summarize(res)
        self.assertGreater(rows["base + online learner (beta 0)"]["mean"], rows["random"]["mean"] + 0.1)
        self.assertGreater(rows["base + online learner (beta 0)"]["mean"], rows["base-only"]["mean"])       # the prior disagrees; learning fixes it
        self.assertGreater(rows["base + online learner (beta 0)"]["last"], rows["base + online learner (beta 0)"]["first"] - 0.05)
        self.assertIn("liked share", rp.format_table(rows))

    def test_replay_is_reproducible(self):
        kw = dict(sessions=4, pool_size=60, steps=15, log=None, policies=("random", "popularity-greedy"))
        self.assertEqual(rp.replay(self.by_id, self.labels, self.base, **kw), rp.replay(self.by_id, self.labels, self.base, **kw))

    def test_logged_thumbs_are_read_from_agreement_and_live_files_but_not_the_busy_session(self):
        from rl import agreement as ag
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "agreement").mkdir()
            for name, sid, th in (("adi", "a1", 1), ("adi_rep", "a2", 0), ("adi_busy", "busy1", 1)):
                with open(root / "agreement" / f"{name}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=ag.FIELDS, restval="")
                    w.writeheader()
                    w.writerow({"user": name, "song_id": sid, "thumb": th, "liking": 0.1, "confidence": 0.9, "emotion": "happy", "time": 0})
            for name, sid, th in (("adi", "l1", 1), ("adi_v2", "a1", 0)):          # a later live thumb overrides an earlier one
                lt.append_row(root / "live" / f"{name}.csv", {"user": name, "seq": 0, "condition": "random", "song_id": sid, "thumb": th, "time": 0})
            got = rp.load_logged_thumbs("adi", root)
        self.assertEqual(got, {"a1": 0, "a2": 0, "l1": 1})


if __name__ == "__main__":
    unittest.main()


class TestLoopAdapter(unittest.TestCase):
    def test_the_loop_runs_with_the_prior_learner_in_place_of_the_bandit_including_a_phone_override(self):
        from integration.orchestrator import run_loop
        from rl.agent.prior_agent import PriorLearnerAgent
        with tempfile.TemporaryDirectory() as d:
            base = make_base(d, favour_genre="rock")
            agent = PriorLearnerAgent(PriorLearner(base, "popularity", seed=1))
            out = run_loop.run(simulate_rounds=24, seed=2, verbose=False, pool=CAT, agent=agent, overrides={6: CAT[10].song_id}, round_delay=0)
        self.assertIs(out["agent"], agent)
        self.assertGreater(agent.n_updates, 10)
        self.assertEqual(len(out["log"]), 24)
        self.assertIn(CAT[10].song_id, [x for x in out["controller"].history])        # the phone's song was played at the next boundary
        self.assertEqual(agent.predict(CAT[0]), 2 * agent.learner.prob(CAT[0]) - 1)

    def test_update_skips_when_nobody_was_seen_and_recommend_lists_unplayed_songs(self):
        from rl.agent.prior_agent import PriorLearnerAgent
        from shared.schemas.messages import GroupReaction
        with tempfile.TemporaryDirectory() as d:
            agent = PriorLearnerAgent(PriorLearner(make_base(d), "popularity"))
        info = agent.update(CAT[0], GroupReaction(0.0, CAT[0].song_id, (), 0.0, {}))
        self.assertTrue(info["skipped"])
        self.assertEqual(agent.n_updates, 0)
        recs = agent.recommend(CAT, k=3)
        self.assertEqual(len(recs), 3)
        self.assertGreaterEqual(recs[0][1], recs[1][1])
