"""Run from the repo root: python3 -m unittest rl.tests.test_agreement -v"""
import csv
import tempfile
import unittest
from pathlib import Path

from integration.orchestrator import run_agreement_check as rc
from rl import agreement as ag
from shared.schemas.messages import GroupReaction, PersonReaction, Song


def rows_from(thumbs, likings, conf=0.8):
    return [{"user": "a", "song_id": f"s{i}", "thumb": t, "liking": l, "confidence": conf, "emotion": "happy"}
            for i, (t, l) in enumerate(zip(thumbs, likings))]


TOY = rows_from([1, 1, 1, 1, 0, 0, 0, 0], [.5, .4, -.2, .3, -.5, .2, -.1, -.3])


class TestAnalyse(unittest.TestCase):
    def test_known_answers(self):
        a = ag.analyse(TOY)
        self.assertEqual((a["tp"], a["fn"], a["fp"], a["tn"]), (3, 1, 1, 3))
        self.assertEqual(a["accuracy"], 0.75)
        self.assertEqual(a["kappa"], 0.5)          # po .75, expected by chance .5
        self.assertEqual(a["auc"], 14 / 16)        # 14 of 16 liked/disliked pairs ordered correctly
        self.assertEqual(a["majority"], 0.5)
        lo, hi = a["ci"]                            # Wilson interval for 6 of 8
        self.assertAlmostEqual(lo, 0.409, places=2)
        self.assertAlmostEqual(hi, 0.929, places=2)

    def test_low_confidence_and_missing_faces_are_excluded_not_counted(self):
        rows = TOY + rows_from([1, 0], [None, None]) + rows_from([1], [0.9], conf=0.05)
        a = ag.analyse(rows)
        self.assertEqual((a["n_rated"], a["n_usable"], a["n_excluded"]), (11, 8, 3))
        self.assertEqual(a["accuracy"], 0.75)

    def test_verdicts(self):
        good = rows_from([1, 0] * 20, [.5, -.5] * 20)
        self.assertEqual(ag.analyse(good)["verdict"], "PASS")
        bad = rows_from([1, 0] * 20, [-.5, .5] * 20)
        a = ag.analyse(bad)
        self.assertEqual(a["verdict"], "FAIL")
        self.assertIn("explicit thumbs", ag.verdict_text(a))
        self.assertEqual(ag.analyse(TOY)["verdict"], "NOT ENOUGH SONGS")
        self.assertEqual(ag.analyse(rows_from([1], [None]))["verdict"], "NO DATA")

    def test_pass_that_only_matches_the_majority_is_flagged(self):
        thumbs = [1] * 36 + [0] * 4
        likings = [.5] * 40                        # camera always says like: 90% agreement, learned nothing
        a = ag.analyse(rows_from(thumbs, likings))
        self.assertEqual(a["verdict"], "PASS")
        self.assertIn("always guessing", ag.verdict_text(a))
        self.assertEqual(a["balanced"], 0.5)

    def test_report_text_has_method_and_verdict(self):
        from rl import report as rp
        tests = rp.run_unit_tests(runner=["python3", "-c", "import sys; print('Ran 1 test', file=sys.stderr)"])
        text = ag.build_agreement_report(TOY, tests, "adi")
        for needle in ("Step 3 report", "need 40", "Wilson", "Kappa", "How the numbers were produced", "thumb UP"):
            self.assertIn(needle, text)


class StubFeedback:
    def __init__(self, likings):
        self.likings = iter(likings)
        self.began = []

    def begin_song(self, song):
        self.began.append(song.song_id)

    def get_group_reaction(self, song_id):
        l = next(self.likings)
        if l is None:
            return GroupReaction(0.0, song_id, (), 0.0, {})
        return GroupReaction(0.0, song_id, (PersonReaction(0, "happy" if l > 0 else "sad", l, 0.9),), l, {"happy": 1.0})


SONGS = [Song(f"s{i}", f"t{i}", "a", "pop", {}, "") for i in range(10)]


class TestCollect(unittest.TestCase):
    def run_collect(self, d, answers, likings, target=3):
        it = iter(answers)
        fb = StubFeedback(likings)
        n = rc.collect(fb, SONGS, "adi", Path(d) / "adi.csv", target=target, listen_seconds=0,
                       input_fn=lambda p: next(it), say=lambda *_: None)
        return n, fb

    def test_saves_rows_and_ignores_bad_keys(self):
        with tempfile.TemporaryDirectory() as d:
            n, _ = self.run_collect(d, ["x", "u", "d", "u"], [.6, -.4, .3])
            self.assertEqual(n, 3)
            rows = ag.load_rows(Path(d) / "adi.csv", "adi")
            self.assertEqual([(r["thumb"], r["liking"] > 0) for r in rows], [(1, True), (0, False), (1, True)])

    def test_skip_quit_and_no_face(self):
        with tempfile.TemporaryDirectory() as d:
            n, _ = self.run_collect(d, ["s", "u", "q"], [.5, None, .5], target=5)
            self.assertEqual(n, 1)  # first skipped (not saved), second saved with no face, third quit
            rows = ag.load_rows(Path(d) / "adi.csv", "adi")
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["liking"])
            self.assertEqual(ag.analyse(rows)["n_usable"], 0)

    def test_resume_does_not_repeat_songs(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_collect(d, ["u", "d"], [.5, -.5], target=2)
            n, fb = self.run_collect(d, ["u"], [.5], target=3)
            self.assertEqual(n, 1)
            self.assertEqual(fb.began, ["s2"])  # s0 and s1 were already rated


if __name__ == "__main__":
    unittest.main()
