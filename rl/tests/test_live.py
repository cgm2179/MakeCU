"""Run from the repo root: python3 -m unittest rl.tests.test_live -v"""
import random
import tempfile
import unittest
from pathlib import Path

from integration.orchestrator import run_live_session as rls
from rl import live_test as lt
from rl.agent.profile import load_profile
from rl.personalize import train_profile
from rl.tests.test_eval import fake_catalog

CAT = fake_catalog(300)


def scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


class FakePlayer:
    def __init__(self, no_preview=()):
        self.played, self.stops, self.no_preview = [], 0, set(no_preview)

    def play(self, song):
        if song.song_id in self.no_preview:
            return False
        self.played.append(song.song_id)
        return True

    def stop(self):
        self.stops += 1

    def wait(self):
        pass

    def close(self):
        pass


def trained_agent(tmp):
    from rl.tests.test_personalize import row
    by_id = {s.song_id: s for s in CAT}
    liked = [s for s in CAT if s.genre in ("jazz", "folk")][:15]
    other = [s for s in CAT if s.genre not in ("jazz", "folk")][:40]
    rows = [row(s.song_id, 5) for s in liked] + [row(s.song_id, 1) for s in other]
    train_profile("a", rows, by_id, directory=tmp)
    return load_profile("a", directory=tmp), {r["song_id"] for r in rows}


class TestSequence(unittest.TestCase):
    def test_balanced_in_every_block_and_reproducible(self):
        seq = lt.make_sequence(40, seed=3)
        self.assertEqual(len(seq), 120)
        for i in range(0, 120, 3):
            self.assertEqual(sorted(seq[i:i + 3]), sorted(lt.CONDITIONS))
        self.assertEqual(seq, lt.make_sequence(40, seed=3))
        self.assertNotEqual(seq, lt.make_sequence(40, seed=4))


class TestPick(unittest.TestCase):
    def test_conditions_pick_what_they_claim(self):
        with tempfile.TemporaryDirectory() as d:
            agent, rated = trained_agent(d)
        pool = [s for s in CAT if s.song_id not in rated]
        rng = random.Random(0)
        top_pop = sorted(p.audio_features["popularity"] for p in pool)[-lt.TOP_K:]
        for _ in range(20):
            s = lt.pick_song("popularity", pool, agent, rng)
            self.assertGreaterEqual(s.audio_features["popularity"], top_pop[0])
        from rl.agent.context import Context
        ranked = sorted((agent.predict(p, Context()) for p in pool))[-lt.TOP_K:]
        for _ in range(20):
            s = lt.pick_song("personal", pool, agent, rng)
            self.assertGreaterEqual(agent.predict(s, Context()), ranked[0])
        self.assertIsNone(lt.pick_song("random", [], agent, rng))
        with self.assertRaises(ValueError):
            lt.pick_song("nope", pool, agent, rng)


class TestSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a.csv"
        self.agent, self.rated = trained_agent(self.tmp.name)
        self.pool = [s for s in CAT if s.song_id not in self.rated]

    def tearDown(self):
        self.tmp.cleanup()

    def run_s(self, answers, player=None, per=4):
        return rls.run_session(self.pool, self.agent, "a", self.path, player, scripted(answers),
                               say=lambda *x: None, per_condition=per, seed=1)

    def test_full_session_balanced_unique_and_stops_each_song(self):
        player = FakePlayer()
        n = self.run_s(["u", "d"] * 6, player, per=4)
        rows = lt.load_rows(self.path)
        self.assertEqual(n, 12)
        self.assertEqual({c: sum(1 for r in rows if r["condition"] == c) for c in lt.CONDITIONS},
                         {c: 4 for c in lt.CONDITIONS})
        self.assertEqual(len({r["song_id"] for r in rows}), 12)                 # no song twice
        self.assertFalse({r["song_id"] for r in rows} & self.rated)             # never a song the profile trained on
        self.assertEqual(player.stops, 12)                                       # each song stopped on the answer

    def test_resume_continues_the_same_sequence(self):
        self.run_s(["u", "d", "q"], per=4)
        first = lt.load_rows(self.path)
        self.assertEqual(len(first), 2)
        self.run_s(["u", "d"] * 6, per=4)
        rows = lt.load_rows(self.path)
        self.assertEqual(len(rows), 12)
        self.assertEqual([r["condition"] for r in rows], lt.make_sequence(4, 1))

    def test_skip_is_not_saved_and_no_preview_is_retried_with_another_song(self):
        n = self.run_s(["s", "u", "u", "u"], FakePlayer(), per=1)
        self.assertEqual(n, 3)
        n = rls.run_session(self.pool, self.agent, "a", Path(self.tmp.name) / "b.csv", FakePlayer(no_preview={self.pool[0].song_id}),
                            scripted(["u"] * 3), say=lambda *x: None, per_condition=1, seed=1)
        self.assertEqual(n, 3)


class TestAnalysis(unittest.TestCase):
    def rows(self, liked):
        out, i = [], 0
        for c in lt.CONDITIONS:
            for j in range(40):
                out.append({"user": "a", "seq": i, "condition": c, "song_id": str(i), "thumb": 1 if j < liked[c] else 0, "time": 0.0})
                i += 1
        return out

    def test_pass_fail_and_in_progress(self):
        self.assertEqual(lt.analyse(self.rows({"random": 8, "popularity": 14, "personal": 24}))["verdict"], "PASS")
        self.assertEqual(lt.analyse(self.rows({"random": 8, "popularity": 14, "personal": 18}))["verdict"], "FAIL")   # +10 vs popularity
        self.assertEqual(lt.analyse(self.rows({"random": 10, "popularity": 8, "personal": 18}))["verdict"], "FAIL")   # only +20 vs random
        self.assertEqual(lt.analyse(self.rows({"random": 8, "popularity": 14, "personal": 24})[:60])["verdict"], "IN PROGRESS")
        self.assertEqual(lt.analyse([])["verdict"], "IN PROGRESS")

    def test_diff_interval_contains_the_difference(self):
        d = lt.analyse(self.rows({"random": 8, "popularity": 14, "personal": 24}))["diffs"]["popularity"]
        self.assertAlmostEqual(d["diff"], 0.25)
        self.assertLess(d["lo"], d["diff"])
        self.assertGreater(d["hi"], d["diff"])


if __name__ == "__main__":
    unittest.main()
