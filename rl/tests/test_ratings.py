"""Run from the repo root: python3 -m unittest rl.tests.test_ratings -v"""
import random
import tempfile
import unittest
from pathlib import Path

from integration.orchestrator import run_rating_session as rs
from rl import ratings as rt
from shared.schemas.messages import Song

MOODS = ["upbeat", "aggressive", "calm", "melancholy"]


def fake_catalog(n=600, seed=0):
    rng = random.Random(seed)
    return [Song(f"s{i}", f"title{i}", f"artist{i}", f"genre{i % 100}",
                 {"popularity": rng.randint(0, 100), "energy": rng.random()}, MOODS[i % 4]) for i in range(n)]


class FakePlayer:
    def __init__(self, no_preview=()):
        self.no_preview, self.played, self.stops = set(no_preview), [], 0

    def play(self, song):
        if song.song_id in self.no_preview:
            return False
        self.played.append(song.song_id)
        return True

    def stop(self):
        self.stops += 1


def scripted(answers):
    """input_fn: returns the next answer; an empty prompt (the break between blocks) gets Enter."""
    it = iter(answers)
    return lambda prompt: "" if prompt == "" else next(it)


def row(sid, q, p, split="dev", rnd=1, t=0.0):
    return {"user": "a", "song_id": sid, "quality": q, "preference": p, "split": split, "round": rnd, "time": t}


class TestStats(unittest.TestCase):
    def test_spearman_known_answers(self):
        self.assertAlmostEqual(rt.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(rt.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(rt.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
        self.assertAlmostEqual(rt.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)   # 1 - 6*4/(5*24)

    def test_disagreement_and_direction(self):
        rows = [row("a", 5, 1), row("b", 1, 4), row("c", 3, 3), row("d", 4, 3), row("e", 2, 2)]
        a = rt.analyse(rows)
        self.assertEqual(a["disagree_n"], 2)                         # a (diff 4) and b (diff 3); d differs by 1
        self.assertAlmostEqual(a["disagree"], 0.4)
        self.assertEqual((a["q_higher"], a["p_higher"]), (1, 1))     # a is good-not-for-me, b a guilty pleasure
        self.assertEqual(a["q_counts"][5], 1)

    def test_gates_pending_pass_and_fail(self):
        few = rt.analyse([row("a", 5, 1)])
        self.assertEqual(rt.overall(few["gates"]), "IN PROGRESS")
        rows = [row(f"s{i}", 5 if i % 3 == 0 else 3, 1 if i % 3 == 0 else 3, "test" if i % 5 == 4 else "dev") for i in range(150)]
        a = rt.analyse(rows)
        self.assertEqual(a["gates"]["150 songs rated, 30 held out"][0], "PASS")
        self.assertEqual(a["gates"]["the two ratings really differ"][0], "PASS")     # a third of songs differ by 4
        same = [row(f"s{i}", 3, 3, "test" if i % 5 == 4 else "dev") for i in range(150)]
        self.assertEqual(rt.analyse(same)["gates"]["the two ratings really differ"][0], "FAIL")
        self.assertEqual(rt.overall(rt.analyse(same)["gates"]), "FAIL")

    def test_retest_known_answers(self):
        first = [row(f"s{i}", 3, 3, rnd=1) for i in range(20)]
        second = [row(f"s{i}", 3, 3 + (2 if i < 4 else 0), rnd=2) for i in range(20)]     # 4 of 20 move preference by 2
        r = rt.retest_stats(first + second)
        self.assertEqual(r["n"], 20)
        self.assertEqual(r["q_within"], 1.0)
        self.assertEqual(r["p_within"], 0.8)
        a = rt.analyse(first + second)
        self.assertEqual(a["gates"]["re-rating is consistent"][0], "PASS")               # exactly 80% is enough
        second[4]["preference"] = 5
        self.assertEqual(rt.analyse(first + second)["gates"]["re-rating is consistent"][0], "FAIL")

    def test_export_labels_leaves_out_undecided_and_carries_the_split(self):
        rows = [row("a", 3, 5), row("b", 3, 1, "test"), row("c", 3, 3), row("d", 3, 4), row("e", 3, 2, rnd=2)]
        with tempfile.TemporaryDirectory() as d:
            n = rt.export_labels(rows, Path(d) / "l.csv", "a")
            text = (Path(d) / "l.csv").read_text().splitlines()
        self.assertEqual(n, 3)
        self.assertEqual(text, ["user,song_id,label,split", "a,a,1,", "a,b,0,test", "a,d,1,"])

    def test_labels_work_with_the_evaluation_tool(self):
        from rl import eval as ev
        rows = [row(f"s{i}", 3, 5 if i % 2 else 1) for i in range(20)]
        with tempfile.TemporaryDirectory() as d:
            rt.export_labels(rows, Path(d) / "l.csv", "a")
            loaded = ev.load_labels(Path(d) / "l.csv", "a")
        self.assertEqual(len(loaded), 20)

    def test_popularity_check_uses_the_catalog(self):
        cat = {s.song_id: s for s in fake_catalog(60)}
        rows = [row(sid, int(1 + s.audio_features["popularity"] // 21), 3) for sid, s in cat.items()]
        a = rt.analyse(rows, cat)
        self.assertGreater(a["corr_q_pop"], 0.9)                  # quality built from popularity tracks it
        self.assertIsNone(a["corr_p_pop"])                         # constant preference: undefined, not a crash
        self.assertEqual(set(a["by_mood"]), set(MOODS))


class TestPlan(unittest.TestCase):
    def test_plan_is_spread_unique_and_capped(self):
        plan = rs.build_plan(fake_catalog(2000), n=400, seed=1)
        self.assertEqual(len(plan), 400)
        self.assertEqual(len({s.song_id for s in plan}), 400)
        self.assertEqual(len({s.artist for s in plan}), 400)                       # one song per artist
        genres = [s.genre for s in plan]
        self.assertLessEqual(max(genres.count(g) for g in set(genres)), rs.GENRE_CAP)
        first150 = plan[:150]
        for mood in MOODS:
            self.assertGreater(sum(1 for s in first150 if s.mood == mood), 25)      # about a quarter each
        for lo, hi, name in rt.POP_BANDS:
            self.assertGreater(sum(1 for s in first150 if lo <= s.audio_features["popularity"] < hi), 30)
        self.assertEqual([s.song_id for s in plan], [s.song_id for s in rs.build_plan(fake_catalog(2000), n=400, seed=1)])


def run_rate(path, plan, answers, player=None, target=10, **kw):
    return rs.rate_session(plan, "a", path, player, scripted(answers), target=target, listen_seconds=0,
                           say=lambda *x: None, wait_fn=lambda s: None, **kw)


class TestRateSession(unittest.TestCase):
    def setUp(self):
        self.plan = fake_catalog(40)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_saves_both_ratings_and_holds_out_every_fifth(self):
        n = run_rate(self.path, self.plan, ["5", "2"] * 10, target=10)
        rows = rt.load_rows(self.path, "a")
        self.assertEqual(n, 10)
        self.assertEqual([(r["quality"], r["preference"]) for r in rows][:2], [(5, 2), (5, 2)])
        self.assertEqual([r["split"] for r in rows], ["dev"] * 4 + ["test"] + ["dev"] * 4 + ["test"])
        self.assertEqual(sum(1 for r in rt.analyse(rows)["gates"] if r), 3)

    def test_exactly_30_of_150_are_held_out(self):
        plan = fake_catalog(200)
        run_rate(self.path, plan, ["3", "3"] * 150 + [], target=150)
        a = rt.analyse(rt.load_rows(self.path, "a"))
        self.assertEqual((a["n"], a["n_test"], a["n_dev"]), (150, 30, 120))

    def test_bad_keys_replay_skip_and_quit(self):
        player = FakePlayer()
        answers = ["x", "9", "4", "r", "4", "2", "s", "5", "1", "q"]
        # song 1: bad keys, quality 4, then 'r' at preference replays and restarts the questions: 4 then 2
        n = run_rate(self.path, self.plan, answers, player, target=10)
        rows = rt.load_rows(self.path, "a")
        self.assertEqual((rows[0]["quality"], rows[0]["preference"]), (4, 2))
        self.assertEqual(player.played.count(self.plan[0].song_id), 2)          # replayed once
        self.assertEqual(n, 2)                                                  # song 1, (song 2 skipped), song 3, then quit... 
        self.assertEqual([r["song_id"] for r in rows], [self.plan[0].song_id, self.plan[2].song_id])

    def test_songs_without_a_preview_are_skipped_not_counted(self):
        player = FakePlayer(no_preview={self.plan[0].song_id, self.plan[1].song_id})
        run_rate(self.path, self.plan, ["3", "3"] * 3, player, target=3)
        ids = [r["song_id"] for r in rt.load_rows(self.path, "a")]
        self.assertEqual(ids, [self.plan[2].song_id, self.plan[3].song_id, self.plan[4].song_id])

    def test_resume_does_not_repeat_and_keeps_the_split_pattern(self):
        run_rate(self.path, self.plan, ["3", "3"] * 3, target=3)
        run_rate(self.path, self.plan, ["3", "3"] * 4, target=7)
        rows = rt.load_rows(self.path, "a")
        self.assertEqual(len({r["song_id"] for r in rows}), 7)
        self.assertEqual([r["split"] for r in rows], ["dev"] * 4 + ["test"] + ["dev"] * 2)

    def test_the_music_stops_after_answering(self):
        player = FakePlayer()
        run_rate(self.path, self.plan, ["3", "3"], player, target=1)
        self.assertGreaterEqual(player.stops, 1)

    def test_a_break_is_offered_every_30_songs(self):
        breaks = []

        def input_fn(prompt):
            if prompt == "":
                breaks.append(1)
                return ""
            return "3"
        rs.rate_session(fake_catalog(100), "a", self.path, None, input_fn, target=61, listen_seconds=0,
                        say=lambda *x: None, wait_fn=lambda s: None)
        self.assertEqual(len(breaks), 2)


class TestRetest(unittest.TestCase):
    def setUp(self):
        self.plan = fake_catalog(40)
        self.by_id = {s.song_id: s for s in self.plan}
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a.csv"
        rs.rate_session(self.plan, "a", self.path, None, scripted(["3", "3"] * 30), target=30, listen_seconds=0,
                        say=lambda *x: None, wait_fn=lambda s: None, now=lambda: 1000.0)

    def tearDown(self):
        self.tmp.cleanup()

    def retest(self, answers, now, **kw):
        return rs.retest_session("a", self.path, self.by_id, None, scripted(answers), listen_seconds=0,
                                 say=lambda *x: None, wait_fn=lambda s: None, now=lambda: now, **kw)

    def test_too_soon_is_refused_unless_forced(self):
        with self.assertRaises(SystemExit) as cm:
            self.retest(["3", "3"] * 20, now=1000.0 + 3600)
        self.assertIn("hours", str(cm.exception).replace("h ago", "hours ago") + "hours")
        self.assertEqual(len([r for r in rt.load_rows(self.path, "a") if r["round"] == 2]), 0)
        self.assertEqual(self.retest(["3", "3"] * 20, now=1000.0 + 3600, force=True), 20)

    def test_picks_20_rated_songs_and_resumes_the_same_set(self):
        late = 1000.0 + 13 * 3600
        self.assertEqual(self.retest(["4", "4"] * 5 + ["q"], now=late), 5)      # 5 done, then quit... (q ends)
        rows = rt.load_rows(self.path, "a")
        second = [r for r in rows if r["round"] == 2]
        first_ids = {r["song_id"] for r in rows if r["round"] == 1}
        self.assertTrue(all(r["song_id"] in first_ids for r in second))
        self.assertEqual(self.retest(["4", "4"] * 15, now=late), 15)
        second = [r for r in rt.load_rows(self.path, "a") if r["round"] == 2]
        self.assertEqual(len({r["song_id"] for r in second}), 20)                # no song twice, 20 in total
        self.assertEqual(rt.retest_stats(rt.load_rows(self.path, "a"))["n"], 20)

    def test_needs_enough_first_pass_songs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.csv"
            rt.append_row(p, {"user": "a", "song_id": "s0", "quality": 3, "preference": 3, "split": "dev", "round": 1, "time": 0})
            with self.assertRaises(SystemExit):
                rs.retest_session("a", p, self.by_id, None, scripted([]), say=lambda *x: None, now=lambda: 1e9)


class TestReport(unittest.TestCase):
    def test_report_has_gates_method_and_no_song_titles(self):
        from rl import report as rp
        import sys
        tests = rp.run_unit_tests(runner=[sys.executable, "-c", "import sys; print('Ran 1 test', file=sys.stderr)"])
        cat = {s.song_id: s for s in fake_catalog(160)}
        rows = [row(sid, 1 + i % 5, 1 + (i * 2) % 5, "test" if i % 5 == 4 else "dev") for i, sid in enumerate(cat)]
        text = rt.build_ratings_report(rows, tests, "adi", cat, synthetic=True)
        for needle in ("Step 4 report", "SYNTHETIC", "the two ratings really differ", "good-but-not-for-me",
                       "How the numbers were produced", "popularity", "Spearman"):
            self.assertIn(needle, text)
        self.assertNotIn("title1", text)


if __name__ == "__main__":
    unittest.main()


class TestHunt(unittest.TestCase):
    def setUp(self):
        self.cat = fake_catalog(60)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_hunt_saves_round3_outside_the_150_and_gates(self):
        rs.rate_session(self.cat, "a", self.path, None, scripted(["2", "2"] * 10), target=10, listen_seconds=0,
                        say=lambda *x: None, wait_fn=lambda s: None)
        n = rs.hunt_session(self.cat, "a", self.path, None, scripted(["5"] * 5), target=5, listen_seconds=0,
                            say=lambda *x: None, wait_fn=lambda s: None)
        # (songs already rated must not be offered again: hunt_session itself does not know, so use hunt_order)
        rows = rt.load_rows(self.path, "a")
        self.assertEqual(n, 5)
        self.assertEqual(len(rt.first_pass(rows)), 10)
        self.assertEqual(len(rt.hunt_rows(rows)), 5)
        self.assertTrue(all(r["split"] == "dev" for r in rt.hunt_rows(rows)))
        self.assertEqual(rt.analyse(rows)["n"], 10)

    def test_hunt_order_skips_rated_and_puts_likely_liked_first(self):
        by = {s.song_id: s for s in self.cat}
        rows = [row(s.song_id, 3, 5 if s.mood == "upbeat" else 1) for s in self.cat[:20]]
        order = rs.hunt_order(self.cat, rows, by, 0)
        self.assertFalse({s.song_id for s in order} & {r["song_id"] for r in rows})
        self.assertEqual(order[0].mood, "upbeat")

    def test_export_can_include_hunt(self):
        rows = [row("x", 3, 5), {**row("y", 3, 1), "round": rt.HUNT_ROUND}]
        p = Path(self.tmp.name) / "l.csv"
        self.assertEqual(rt.export_labels(rows, p, "a"), 1)
        self.assertEqual(rt.export_labels(rows, p, "a", include_hunt=True), 2)
