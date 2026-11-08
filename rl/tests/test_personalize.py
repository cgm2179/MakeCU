"""Run from the repo root: python3 -m unittest rl.tests.test_personalize -v"""
import tempfile
import unittest
from pathlib import Path

from integration.orchestrator import run_loop
from rl import personalize as pz
from rl import ratings as rt
from rl.agent.context import Context
from rl.agent.profile import ProfileError, load_profile
from rl.tests.test_eval import fake_catalog

CAT = fake_catalog(80)
BY_ID = {s.song_id: s for s in CAT}


def row(sid, pref, split="dev", rnd=1):
    return {"user": "a", "song_id": sid, "quality": pref, "preference": pref, "split": split, "round": rnd, "time": 0.0}


class TestLabels(unittest.TestCase):
    def test_labels_exclude_threes_and_heldout_unless_asked(self):
        rows = [row(CAT[0].song_id, 5), row(CAT[1].song_id, 1), row(CAT[2].song_id, 3),
                row(CAT[3].song_id, 5, "test"), row(CAT[4].song_id, 4, "dev", rt.HUNT_ROUND)]
        got = pz.labels_from_ratings(rows, BY_ID)
        self.assertEqual({s.song_id: y for s, y in got}, {CAT[0].song_id: 1, CAT[1].song_id: 0, CAT[4].song_id: 1})
        both = pz.labels_from_ratings(rows, BY_ID, include_heldout=True)
        self.assertIn(CAT[3].song_id, {s.song_id for s, _ in both})

    def test_unknown_songs_are_skipped(self):
        self.assertEqual(pz.labels_from_ratings([row("nope", 5)], BY_ID), [])


class TestTrain(unittest.TestCase):
    def test_train_saves_a_profile_that_prefers_liked_genres(self):
        liked = [s for s in CAT if s.genre in ("jazz", "folk")][:12]
        disliked = [s for s in CAT if s.genre not in ("jazz", "folk")][:30]
        rows = [row(s.song_id, 5) for s in liked] + [row(s.song_id, 1) for s in disliked]
        with tempfile.TemporaryDirectory() as d:
            agent, n, path = pz.train_profile("a", rows, BY_ID, directory=d)
            self.assertEqual(n, 42)
            loaded = load_profile("a", directory=d)
        ctx = Context()
        rest = [s for s in CAT if s.song_id not in {r["song_id"] for r in rows}]
        for s in rest[:20]:
            self.assertAlmostEqual(agent.predict(s, ctx), loaded.predict(s, ctx))
        likely = [s for s in rest if s.genre in ("jazz", "folk")]
        unlikely = [s for s in rest if s.genre not in ("jazz", "folk")]
        if likely and unlikely:
            self.assertGreater(sum(agent.predict(s, ctx) for s in likely) / len(likely),
                               sum(agent.predict(s, ctx) for s in unlikely) / len(unlikely))

    def test_needs_both_classes(self):
        with self.assertRaises(SystemExit):
            pz.train_profile("a", [row(CAT[0].song_id, 5)], BY_ID, directory=tempfile.gettempdir())


class TestLoopProfile(unittest.TestCase):
    def test_loop_saves_then_next_run_loads(self):
        with tempfile.TemporaryDirectory() as d:
            first = run_loop.run(simulate_rounds=6, seed=1, verbose=False, pool=CAT, user="a", profile_dir=d)
            self.assertTrue(Path(first["profile_path"]).exists())
            n1 = first["agent"].n_updates
            second = run_loop.run(simulate_rounds=4, seed=2, verbose=False, pool=CAT, user="a", profile_dir=d)
            self.assertGreater(second["agent"].n_updates, n1)       # it continued from the saved state

    def test_damaged_profile_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.json").write_text("{not json")
            with self.assertRaises(ProfileError):
                run_loop.run(simulate_rounds=2, seed=1, verbose=False, pool=CAT, user="a", profile_dir=d)

    def test_no_user_means_nothing_is_saved(self):
        out = run_loop.run(simulate_rounds=2, seed=1, verbose=False, pool=CAT)
        self.assertIsNone(out["profile_path"])


if __name__ == "__main__":
    unittest.main()
