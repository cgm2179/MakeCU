"""Music loop fed by the camera package in MOCK mode (no webcam, no DeepFace).

Needs the camera deps (opencv-python, fastapi, uvicorn); skipped without them. Run with a Python that
has them, from the repo root:  python3 -m unittest integration.tests.test_camera_loop -v
"""
import importlib.util
import statistics as st
import unittest

from integration.orchestrator.run_loop import run
from music.catalog.songs import SONGS

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("cv2", "fastapi", "uvicorn", "numpy"))


from perception.emotion.camera_reaction import calibrate_liking, check_baseline


class TestCalibrationMath(unittest.TestCase):
    def test_baseline_maps_to_zero_and_extremes_stay_at_extremes(self):
        for b in (-0.6, -0.3, 0.0, 0.4):
            self.assertAlmostEqual(calibrate_liking(b, b), 0.0)
            self.assertAlmostEqual(calibrate_liking(1.0, b), 1.0)
            self.assertAlmostEqual(calibrate_liking(-1.0, b), -1.0)

    def test_monotonic_and_signs(self):
        b = -0.45  # a resting face reads mildly negative
        xs = [-1, -0.8, -0.45, -0.2, 0.3, 0.9]
        ys = [calibrate_liking(x, b) for x in xs]
        self.assertEqual(ys, sorted(ys))
        self.assertLess(calibrate_liking(-0.8, b), 0)   # worse than resting -> negative
        self.assertGreater(calibrate_liking(-0.2, b), 0)  # better than resting -> positive, even though raw is negative

    def test_guard_thresholds(self):
        self.assertEqual(check_baseline(-0.45, 0.15), [])              # a normal resting face
        self.assertEqual(check_baseline(0.15, 0.25), [])               # exactly on the limits is accepted
        self.assertEqual(len(check_baseline(0.56, 0.27)), 2)           # the smiling run from before: both reasons
        self.assertIn("happy", check_baseline(0.20, 0.10)[0])
        self.assertIn("moving", check_baseline(-0.30, 0.30)[0])
        self.assertEqual(len(check_baseline(0.20, 0.10)), 1)

    def test_extreme_baselines_do_not_divide_by_zero(self):
        for b in (-1.0, 1.0):
            for raw in (-1.0, 0.0, 1.0):
                self.assertTrue(-1.0 <= calibrate_liking(raw, b) <= 1.0)


class Recording:
    """Wraps a CameraFeedback and keeps every reaction the loop read from the perception service."""

    def __init__(self, inner):
        self.inner, self.reactions = inner, []

    def start(self): self.inner.start()
    def stop(self): self.inner.stop()
    def begin_song(self, song): self.inner.begin_song(song)

    def get_group_reaction(self, song_id):
        r = self.inner.get_group_reaction(song_id)
        self.reactions.append(r)
        return r


@unittest.skipUnless(HAVE_DEPS, "camera package dependencies not installed")
class TestCameraLoop(unittest.TestCase):
    def test_bridge_imports_from_repo_root_without_breaking_our_perception_folder(self):
        from perception.emotion.camera_reaction import import_perception_cv
        import perception.emotion.fake_reaction as fake

        pcv = import_perception_cv()
        self.assertTrue(pcv.__file__.replace("\\", "/").endswith("perception/perception_cv/__init__.py"))
        self.assertTrue(callable(fake.get_group_reaction))  # our own module still resolves

    def test_plain_mock_service_feeds_valid_reactions_into_the_loop(self):
        from perception.emotion.camera_reaction import CameraFeedback

        fb = Recording(CameraFeedback(mock=True))  # perception_cv.build_service(mock=True)
        out = run(4, seed=0, pool=SONGS, feedback=fb, listen_seconds=0.6, overrides={}, verbose=False)
        self.assertEqual(len(out["log"]), 4)
        self.assertEqual(len(fb.reactions), 4)
        for r in fb.reactions:
            self.assertEqual(type(r).__module__, "perception_cv.types")  # really from their package
            self.assertGreaterEqual(r.group_liking, -1.0)
            self.assertLessEqual(r.group_liking, 1.0)
            self.assertAlmostEqual(sum(r.group_emotion.values()), 1.0, places=6)
            self.assertGreater(len(r.people), 0)
        self.assertEqual(out["skipped_rounds"], 0)
        self.assertEqual(len(out["rewards"]), 4)
        self.assertIsNone(fb.inner.feeder_error)

    def test_nobody_in_view_skips_learning_for_that_round(self):
        from perception.emotion.camera_reaction import CameraFeedback
        from perception.emotion.taste_mock_model import TasteMockModel

        fb = CameraFeedback(model=TasteMockModel(lambda s: s.genre == "jazz"), empty_rounds={2, 3})
        out = run(5, seed=0, pool=SONGS, feedback=fb, listen_seconds=0.5, overrides={}, verbose=False)
        flags = [e["skipped"] for e in out["log"]]
        self.assertEqual(flags, [False, True, True, False, False])
        for e in out["log"]:
            if e["skipped"]:
                self.assertEqual(e["reaction"].people, [])
                self.assertEqual(out["agent"].play_counts[e["song_id"]], 0)  # nothing was learned from it
        self.assertEqual(out["skipped_rounds"], 2)
        self.assertEqual(len(out["rewards"]), 3)

    def test_bandit_adapts_to_camera_reactions(self):
        from perception.emotion.camera_reaction import CameraFeedback
        from perception.emotion.taste_mock_model import TasteMockModel

        liked = lambda s: s.genre == "jazz"  # the mock faces light up for jazz (4 of 16 songs)
        fb = CameraFeedback(model=TasteMockModel(liked, seed=0))
        out = run(30, seed=0, pool=SONGS, feedback=fb, listen_seconds=0.3, overrides={}, verbose=False)
        share = lambda es: st.mean(liked(e["song"]) for e in es)
        early, late = share(out["log"][:10]), share(out["log"][-10:])
        print(f"\n  songs the faces liked: first 10 {early:.0%} -> last 10 {late:.0%} (random: 25%)")
        self.assertGreater(late, early + 0.2)
        self.assertGreaterEqual(late, 0.6)


@unittest.skipUnless(HAVE_DEPS, "camera package dependencies not installed")
class TestCalibratedSignal(unittest.TestCase):
    def make(self, **kw):
        from perception.emotion.camera_reaction import CameraFeedback
        from perception.emotion.taste_mock_model import TasteMockModel

        liked = lambda s: s.genre == "jazz"
        return CameraFeedback(model=TasteMockModel(liked, noise=0.0), **kw), liked

    def test_defaults_are_yunet_and_bored_disabled(self):
        from perception.emotion.camera_reaction import CameraFeedback

        self.assertEqual(CameraFeedback().detector, "yunet")
        fb, _ = self.make()
        fb.start()
        try:
            self.assertFalse(fb.service.cfg.ENABLE_BORED)  # their stillness-based "bored" rule is off
        finally:
            fb.stop()

    def test_resting_face_calibrates_to_zero_and_reactions_stand_out(self):
        fb, liked = self.make(read_window_seconds=0.6)
        fb.start()
        try:
            base = fb.calibrate(seconds=4, say=lambda *a: None, countdown=0, speak_fn=lambda t: None)
            self.assertTrue(-0.8 < base < -0.2, base)          # a resting face reads negative
            self.assertAlmostEqual(fb._cal(base), 0.0)         # ...and calibrates to zero
            jazz = next(x for x in SONGS if liked(x))
            other = next(x for x in SONGS if not liked(x))
            fb.begin_song(jazz); import time; time.sleep(1.3)
            up = fb.get_group_reaction("up")
            fb.begin_song(other); time.sleep(1.3)
            down = fb.get_group_reaction("down")
        finally:
            fb.stop()
        print(f"\n  baseline {base:+.2f} -> liked song scores {up.group_liking:+.2f}, disliked song {down.group_liking:+.2f}")
        self.assertGreater(up.group_liking, 0.5)
        self.assertLess(down.group_liking, -0.3)
        self.assertGreater(up.group_liking, down.group_liking + 0.8)

    def test_guard_rejects_a_smile_then_accepts_a_neutral_retry_with_countdown(self):
        from perception.emotion.taste_mock_model import LIKED

        fb, _ = self.make()
        fb.model.resting_probs = dict(LIKED)  # person is smiling during the first attempt
        said, spoken = [], []

        def say(msg):
            said.append(msg)
            if msg.startswith("Calibration rejected"):
                from perception.emotion.taste_mock_model import RESTING
                fb.model.resting_probs = dict(RESTING)  # ...then relaxes to a neutral face

        fb.start()
        try:
            base = fb.calibrate(seconds=3, discard=1, say=say, count_step=0.05, speak_fn=spoken.append)
        finally:
            fb.stop()
        text = "\n".join(said)
        self.assertEqual(text.count("Calibration rejected"), 1)
        self.assertIn("happy, not neutral", text)
        self.assertIn("Let's try again", text)
        self.assertIn("accepted", text)
        self.assertLessEqual(base, 0.15)
        # countdown 3, 2, 1 then the hold cue, twice (original attempt + retry); also spoken aloud
        cue = ["3", "2", "1", "hold a neutral face"]
        self.assertEqual(spoken[:4], cue)
        self.assertEqual([x for x in spoken if x in ("3", "2", "1")], ["3", "2", "1", "3", "2", "1"])
        self.assertEqual(spoken.count("hold a neutral face"), 2)
        self.assertIsNotNone(fb.baseline)

    def test_guard_gives_up_after_max_attempts_instead_of_using_a_bad_baseline(self):
        from perception.emotion.taste_mock_model import LIKED

        fb, _ = self.make()
        fb.model.resting_probs = dict(LIKED)  # smiling the whole time
        fb.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "after 2 attempts.*happy, not neutral"):
                fb.calibrate(seconds=3, discard=1, say=lambda *a: None, countdown=0, max_attempts=2,
                             speak_fn=lambda t: None)
        finally:
            fb.stop()
        self.assertIsNone(fb.baseline)  # nothing was stored

    def test_calibration_fails_loudly_if_nobody_is_in_view(self):
        fb, _ = self.make(empty_rounds={1})
        fb.start()
        try:
            fb.begin_song(SONGS[0])  # round 1 = nobody in view
            with self.assertRaisesRegex(RuntimeError, "Calibration failed"):
                fb.calibrate(seconds=3, say=lambda *a: None, countdown=0, max_attempts=1,
                             speak_fn=lambda t: None)
        finally:
            fb.stop()


class TestFakeListenerStillWorks(unittest.TestCase):
    def test_default_path_does_not_need_the_camera_package(self):
        out = run(simulate_rounds=6, seed=1, pool=SONGS, verbose=False, round_delay=0)
        self.assertEqual(len(out["log"]), 6)
        self.assertEqual(out["skipped_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
