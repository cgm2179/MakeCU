"""Run from the repo root: python3 -m unittest rl.tests.test_engagement -v"""
import json
import math
import unittest
from unittest import mock

import numpy as np

from perception.emotion import clip_features as cf
from perception.emotion.camera_reaction import CameraFeedback


def head(t0, dur, amp=0.0, freq=2.0, hz=10.0, seed=0, width=0.2, sway=0.0, jitter=0.0005):
    """Synthetic face-box samples (t, person, cx, cy, w, h): nodding at `freq` Hz with amplitude `amp` (frame widths)."""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(int(dur * hz)):
        t = t0 + i / hz
        out.append((t, 0, 0.5 + sway * math.sin(2 * math.pi * 0.3 * (t - t0)) + rng.randn() * jitter,
                    0.5 + amp * math.sin(2 * math.pi * freq * (t - t0)) + rng.randn() * jitter, width, width * 1.3))
    return out


def probs(t0, dur, level, hz=10.0):
    return [(t0 + i / hz, level, 0.0) for i in range(int(dur * hz))]


def liking(t0, dur, hz=10.0):
    return [(t0 + i / hz, 0.1) for i in range(int(dur * hz))]


class TestBeatLocked(unittest.TestCase):
    def test_nodding_on_the_beat_scores_higher_than_the_same_nodding_against_another_tempo(self):
        m = head(0, 10, amp=0.01, freq=2.0)                                # 2 Hz = 120 BPM
        on, amp_on = cf.beat_locked(m, 0, 10, 120)
        off, amp_off = cf.beat_locked(m, 0, 10, 93)                         # 1.55 Hz; halves/doubles 0.78, 3.1: none near 2 Hz
        self.assertGreater(on, off + 1.0)
        self.assertGreater(amp_on, 3 * amp_off)

    def test_half_time_nodding_counts_as_on_the_beat(self):
        m = head(0, 10, amp=0.01, freq=1.0)                                # nods once per two beats at 120 BPM
        half, _ = cf.beat_locked(m, 0, 10, 120)
        wrong, _ = cf.beat_locked(m, 0, 10, 180)                           # candidates 3.0, 1.5 Hz: well away from 1 Hz
        self.assertGreater(half, wrong + 1.0)

    def test_still_head_has_almost_no_beat_amplitude_and_no_tempo_means_zero(self):
        still = head(0, 10, amp=0.0)
        _, amp = cf.beat_locked(still, 0, 10, 120)
        self.assertLess(amp, 0.0005)
        self.assertEqual(cf.beat_locked(head(0, 10, amp=0.01), 0, 10, None), (0.0, 0.0))
        self.assertEqual(cf.beat_locked(head(0, 10, amp=0.01), 0, 10, 0), (0.0, 0.0))
        self.assertEqual(cf.beat_locked([], 0, 10, 120), (0.0, 0.0))

    def test_slow_sway_is_not_a_beat(self):
        sway, _ = cf.beat_locked(head(0, 10, amp=0.0, sway=0.05), 0, 10, 120)
        nod, _ = cf.beat_locked(head(0, 10, amp=0.01, freq=2.0), 0, 10, 120)
        self.assertGreater(nod, sway + 0.5)


class TestEngagement(unittest.TestCase):
    def baseline(self):
        return cf.resting_baseline(probs(0, 5, 0.05), head(0, 5, amp=0.004, freq=0.7, seed=1), 0, 5)

    def test_resting_baseline_has_the_new_keys(self):
        b = self.baseline()
        for k in ("smile", "motion", "vert_motion", "activity", "width", "still_thr"):
            self.assertIn(k, b)
        self.assertAlmostEqual(b["width"], 0.2, places=3)
        self.assertGreater(b["still_thr"], 0)

    def test_moving_more_than_the_pre_roll_raises_activity_and_lowers_stillness(self):
        b = self.baseline()
        calm = cf.engagement(probs(10, 10, 0.05), head(10, 10, amp=0.002, freq=0.7, seed=2), 10, 20, b, 120)
        lively = cf.engagement(probs(10, 10, 0.05), head(10, 10, amp=0.02, freq=2.0, seed=3), 10, 20, b, 120)
        self.assertGreater(lively["activity"], calm["activity"] + 0.01)
        self.assertGreater(calm["stillness"], 0.7)
        self.assertLess(lively["stillness"], 0.2)
        self.assertGreater(lively["beat_power"], calm["beat_power"])

    def test_smile_features_are_relative_to_the_baseline_and_lean_follows_face_size(self):
        b = self.baseline()
        happy = cf.engagement(probs(10, 10, 0.6), head(10, 10, width=0.24, seed=4), 10, 20, b, 120)
        flat = cf.engagement(probs(10, 10, 0.05), head(10, 10, width=0.2, seed=5), 10, 20, b, 120)
        self.assertAlmostEqual(happy["smile_frac"], 1.0)
        self.assertEqual(flat["smile_frac"], 0.0)
        self.assertAlmostEqual(happy["smile_peak"], 0.55, places=2)
        self.assertAlmostEqual(happy["lean"], 0.2, places=2)           # 20% bigger face: leaning in
        self.assertAlmostEqual(flat["lean"], 0.0, places=2)

    def test_no_baseline_threshold_gives_zero_stillness_not_a_crash(self):
        r = cf.engagement(probs(0, 10, 0.1), head(0, 10, amp=0.01), 0, 10, {}, 120)
        self.assertEqual(r["stillness"], 0.0)

    def test_clip_features_has_every_v2_feature_and_stays_backward_compatible(self):
        m = head(10, 10, amp=0.01)
        f = cf.clip_features(liking(10, 10), probs(10, 10, 0.2), m, 10, 20, self.baseline(), tempo_bpm=120)
        for k in cf.FEATURES_V2:
            self.assertIn(k, f, k)
        self.assertEqual(cf.FEATURES, cf.FEATURES_V2[:len(cf.FEATURES)])           # old order untouched
        self.assertTrue(all(math.isfinite(v) for v in f.values()))
        self.assertIsNone(cf.clip_features(liking(10, 0.2), probs(10, 0.2, 0.2), m, 10, 10.2, {}))


class TestCameraFeedbackPreroll(unittest.TestCase):
    def feedback(self):
        fb = CameraFeedback(mock=True)
        fb.rest = {"smile": 0.1, "motion": 0.01, "vert_motion": 0.005, "activity": 0.05, "width": 0.3, "still_thr": 0.04}
        return fb

    def song(self, tempo=120):
        s = mock.Mock()
        s.audio_features = {"tempo": tempo}
        return s

    def test_pre_roll_is_used_only_when_long_enough_and_replaces_the_session_baseline(self):
        fb = self.feedback()
        for t, pid, cx, cy, w, h in head(100, 4, amp=0.003, freq=0.7, width=0.2):
            fb._motion.append((t, pid, cx, cy, w, h))
        for p in probs(100, 4, 0.02):
            fb._probs.append(p)
        with mock.patch("perception.emotion.camera_reaction.time.time", side_effect=[100.0, 104.0]):
            fb.begin_preroll()
            fb.begin_song(self.song(130))
        self.assertEqual(fb._preroll, (100.0, 104.0))
        self.assertEqual(fb._tempo, 130)
        b = fb.clip_baseline()
        self.assertAlmostEqual(b["width"], 0.2, places=2)              # from the pre-roll, not the session's 0.3
        self.assertNotEqual(b["activity"], 0.05)
        with mock.patch("perception.emotion.camera_reaction.time.time", side_effect=[200.0, 201.0]):   # only 1 s: not trusted
            fb.begin_preroll()
            fb.begin_song(self.song(90))
        self.assertIsNone(fb._preroll)
        self.assertEqual(fb.clip_baseline()["width"], 0.3)              # back to the session calibration
        with mock.patch("perception.emotion.camera_reaction.time.time", side_effect=[300.0]):          # never asked for a pre-roll
            fb.begin_song(self.song(None))
        self.assertIsNone(fb._preroll)
        self.assertIsNone(fb._tempo)

    def test_raw_export_is_json_and_contains_the_pre_roll_samples(self):
        fb = self.feedback()
        for row in head(100, 8, amp=0.01):
            fb._motion.append(row)
        for p in probs(100, 8, 0.1):
            fb._probs.append(p)
        for t, v in liking(100, 8):
            fb._samples.append((t, v, "neutral"))
        fb._preroll, fb._clip_start, fb._tempo = (100.0, 104.0), 104.0, 120
        with mock.patch("perception.emotion.camera_reaction.time.time", return_value=108.0):
            raw = fb.clip_raw()
        json.dumps(raw)
        self.assertEqual(raw["preroll"], [100.0, 104.0])
        self.assertEqual(raw["t0"], 100.0)
        self.assertEqual(raw["tempo"], 120)
        self.assertGreater(len(raw["motion"]), 70)
        self.assertEqual(raw["liking"][0][0], 100.0)


if __name__ == "__main__":
    unittest.main()
