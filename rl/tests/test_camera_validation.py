"""Run from the repo root: python3 -m unittest rl.tests.test_camera_validation -v"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from integration.orchestrator import run_agreement_check as rac
from perception.emotion import clip_features as cf
from perception.emotion.camera_reaction import CameraFeedback
from rl import camera_validation as cv
from rl.tests.test_engagement import head, liking, probs


def session(n, signal, seed, offset=0.0, conf=0.9):
    """Synthetic session: with signal > 0 liked songs show more activity, more beat power and a bigger smile; `offset` shifts every
    feature (a session whose camera drifted)."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        thumb = int(rng.rand() < 0.45)
        f = {k: float(rng.randn() * 0.5) + offset for k in cf.FEATURES_V2}
        f["activity"] += signal * thumb
        f["beat_power"] += signal * thumb
        f["smile_frac"] += signal * thumb * 0.7
        f["stillness"] -= signal * thumb * 0.8
        rows.append({"user": "u", "song_id": f"s{seed}_{i}", "thumb": thumb, "liking": 0.1, "confidence": conf, "emotion": "neutral",
                     "features": f, "setting": "quiet", "group_size": 1})
    return rows


class TestEvaluate(unittest.TestCase):
    def test_a_real_signal_survives_a_drifted_session(self):
        r = cv.evaluate({"a": session(60, 1.2, 1), "b": session(60, 1.2, 2, offset=0.8), "c": session(60, 1.2, 3, offset=-0.5)})
        self.assertEqual(r["verdict"], "PASS")
        self.assertTrue(all(a >= cv.PASS_AUC for a in r["loso_v2"].values()))
        self.assertEqual(set(r["loso_v2"]), {"a", "b", "c"})
        self.assertIn("Overall: PASS", cv.format_text(r))

    def test_noise_fails(self):
        r = cv.evaluate({"a": session(60, 0.0, 4), "b": session(60, 0.0, 5)})
        self.assertEqual(r["verdict"], "FAIL")

    def test_every_session_must_pass_not_just_the_average(self):
        r = cv.evaluate({"a": session(60, 1.5, 6), "b": session(60, 1.5, 7), "c": session(60, 0.0, 8)})
        self.assertEqual(r["verdict"], "FAIL")
        self.assertFalse(all(r["pass_each"].values()))

    def test_not_enough_data_is_not_a_pass_or_a_fail(self):
        self.assertEqual(cv.evaluate({"a": session(60, 1.2, 1)})["verdict"], "NOT ENOUGH DATA")
        r = cv.evaluate({"a": session(60, 1.2, 1), "b": session(30, 1.2, 2)})
        self.assertEqual(r["verdict"], "NOT ENOUGH DATA")
        self.assertIn("b:", r["reason"])
        skew = [dict(x, thumb=0) for x in session(60, 1.2, 3)][:55] + session(60, 1.2, 4)[:5]
        self.assertEqual(cv.evaluate({"a": session(60, 1.2, 1), "b": skew})["verdict"], "NOT ENOUGH DATA")

    def test_old_rows_without_the_new_features_and_low_confidence_rows_are_not_used(self):
        old = [dict(x, features={k: v for k, v in x["features"].items() if k in cf.FEATURES}) for x in session(60, 1.0, 9)]
        self.assertEqual(cv.usable(old, cv.V2), [])
        self.assertEqual(cv.usable(session(60, 1.0, 9, conf=0.05), cv.V2), [])

    def test_context_scores_are_reported_per_session(self):
        a, b = session(60, 1.2, 1), session(60, 1.2, 2)
        ctx = {"a": {"popularity": list(range(60))}, "b": {"popularity": list(range(60))}}
        r = cv.evaluate({"a": a, "b": b}, ctx)
        self.assertIn("popularity", r["context"]["a"])


class TestRawRecompute(unittest.TestCase):
    def test_features_from_raw_equals_the_live_features(self):
        fb = CameraFeedback(mock=True)
        fb.rest = {"smile": 0.1, "motion": 0.01, "vert_motion": 0.005, "activity": 0.05, "width": 0.3, "still_thr": 0.04}
        for row in head(100, 4, amp=0.003, freq=0.7, seed=1) + head(104, 10, amp=0.015, freq=2.0, seed=2, width=0.24):
            fb._motion.append(row)
        for p in probs(100, 4, 0.03) + probs(104, 10, 0.5):
            fb._probs.append(p)
        for t, v in liking(100, 14):
            fb._samples.append((t, v, "neutral"))
        song = mock.Mock()
        song.audio_features = {"tempo": 120.0}
        with mock.patch("perception.emotion.camera_reaction.time.time", side_effect=[100.0, 104.0]):
            fb.begin_preroll()
            fb.begin_song(song)
        with mock.patch("perception.emotion.camera_reaction.time.time", return_value=114.0):
            live = fb.clip_features()
            raw = fb.clip_raw()
        again = cf.features_from_raw(json.loads(json.dumps(raw)))
        self.assertEqual(set(live), set(again))
        for k in live:
            self.assertAlmostEqual(live[k], again[k], places=9, msg=k)
        self.assertGreater(live["beat_power"], 1.0)                      # 2 Hz nodding at 120 BPM
        self.assertGreater(live["lean"], 0.1)                            # face 0.24 wide vs 0.2 in the pre-roll

    def test_save_raw_writes_one_json_per_clip_with_the_thumb(self):
        with tempfile.TemporaryDirectory() as d:
            rac.save_raw(Path(d) / "u", "song1", 1, {"t0": 1.0, "motion": [[1, 0, 0.5, 0.5, 0.2, 0.2]]})
            data = json.loads((Path(d) / "u" / "song1.json").read_text())
        self.assertEqual((data["song_id"], data["thumb"]), ("song1", 1))


class _FakeFeedback:
    def __init__(self, log):
        self.log = log

    def begin_preroll(self):
        self.log.append("preroll")

    def begin_song(self, song):
        self.log.append("begin_song")

    def get_group_reaction(self, song_id):
        from shared.schemas.messages import GroupReaction, PersonReaction
        return GroupReaction(0.0, song_id, (PersonReaction(0, "happy", 0.2, 0.9),), 0.2, {"happy": 1.0})

    def clip_features(self):
        return {k: 0.1 for k in cf.FEATURES_V2}

    def clip_raw(self):
        return {"t0": 1.0, "tempo": 120, "motion": []}


class _FakePlayer:
    def __init__(self, log):
        self.log = log

    def play(self, song):
        self.log.append("audio")
        return True

    def wait(self):
        self.log.append("wait")


class TestCollectWithPreroll(unittest.TestCase):
    def test_pre_roll_comes_before_the_audio_and_raw_samples_are_saved_for_rated_songs_only(self):
        from rl.tests.test_eval import fake_catalog
        log = []
        answers = iter(["u", "s", "d"])
        with tempfile.TemporaryDirectory() as d:
            n = rac.collect(_FakeFeedback(log), fake_catalog(5), "u", Path(d) / "u.csv", target=2, player=_FakePlayer(log),
                            input_fn=lambda prompt="": next(answers), say=lambda *a: None, preroll=0.01, raw_dir=Path(d) / "raw")
            raws = sorted(p.name for p in (Path(d) / "raw").glob("*.json"))
            rows = (Path(d) / "u.csv").read_text().splitlines()
        self.assertEqual(log[:4], ["preroll", "audio", "begin_song", "wait"])
        self.assertEqual(n, 2)                                   # one skipped
        self.assertEqual(len(raws), 2)
        self.assertEqual(len(rows), 3)                           # header + 2 songs
        self.assertIn("beat_power", rows[1])                     # the v2 features are in the CSV row

    def test_no_pre_roll_and_no_raw_dir_changes_nothing(self):
        from rl.tests.test_eval import fake_catalog
        log = []
        with tempfile.TemporaryDirectory() as d:
            rac.collect(_FakeFeedback(log), fake_catalog(3), "u", Path(d) / "u.csv", target=1, player=_FakePlayer(log),
                        input_fn=lambda prompt="": "u", say=lambda *a: None)
            self.assertFalse((Path(d) / "raw").exists())
        self.assertNotIn("preroll", log)


if __name__ == "__main__":
    unittest.main()
