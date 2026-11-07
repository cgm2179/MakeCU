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


class TestFakeListenerStillWorks(unittest.TestCase):
    def test_default_path_does_not_need_the_camera_package(self):
        out = run(simulate_rounds=6, seed=1, pool=SONGS, verbose=False, round_delay=0)
        self.assertEqual(len(out["log"]), 6)
        self.assertEqual(out["skipped_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
