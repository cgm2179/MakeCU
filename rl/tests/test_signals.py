"""Run from the repo root: python3 -m unittest rl.tests.test_signals -v
Clip features (movement), the per-person camera model, the combined reward and the loop wiring."""
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from integration.orchestrator import run_agreement_check as rc
from integration.orchestrator.run_loop import run
from perception.emotion import clip_features as cf
from rl import agreement as ag
from rl.agent import camera_model as cm
from rl.agent.bandit import BanditAgent
from rl.agent.reward import CameraRewardModel, combine
from rl.tests.test_agreement import SONGS, StubFeedback, rows_from
from shared.schemas.messages import GroupReaction, PersonReaction, Song


def motion_series(hz_nod=0.0, amp=0.02, jitter=0.0, seconds=8.0, fps=10, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        cy = 0.5 + amp * math.sin(2 * math.pi * hz_nod * t) + rng.uniform(-jitter, jitter)
        out.append((100.0 + t, 1, 0.5, cy, 0.2, 0.3))
    return out


def liking_series(level=0.0, seconds=8.0, rise=0.0):
    n = int(seconds * 10)
    return [(100.0 + i / 10, level + rise * i / n) for i in range(n)]


class TestClipFeatures(unittest.TestCase):
    def test_nodding_on_the_beat_is_detected_and_a_still_face_is_not(self):
        nod = cf.clip_features(liking_series(), [], motion_series(hz_nod=2.0), 100.0, 108.0)
        still = cf.clip_features(liking_series(), [], motion_series(0.0, jitter=0.002), 100.0, 108.0)
        self.assertGreater(nod["bob_share"], 0.7)
        self.assertLess(still["bob_share"], nod["bob_share"] - 0.3)
        self.assertGreater(nod["bob_amp"], 5 * still["bob_amp"])
        self.assertGreater(nod["vert_motion"], still["vert_motion"])

    def test_slow_swaying_is_not_mistaken_for_nodding(self):
        sway = cf.clip_features(liking_series(), [], motion_series(hz_nod=0.15, amp=0.05), 100.0, 108.0)
        self.assertLess(sway["bob_share"], 0.35)

    def test_baseline_is_subtracted(self):
        m = motion_series(hz_nod=2.0)
        base = cf.resting_baseline([(100.0, 0.1, 0.0)] * 5, m, 100.0, 108.0)
        raw = cf.clip_features(liking_series(), [(100.5, 0.5, 0.0)] * 5, m, 100.0, 108.0)
        rel = cf.clip_features(liking_series(), [(100.5, 0.5, 0.0)] * 5, m, 100.0, 108.0, base)
        self.assertAlmostEqual(rel["motion"], raw["motion"] - base["motion"])
        self.assertAlmostEqual(rel["smile"], raw["smile"] - 0.1)

    def test_reaction_dynamics(self):
        f = cf.clip_features(liking_series(-0.1, rise=0.4), [], [], 100.0, 108.0)
        self.assertAlmostEqual(f["peak_liking"], 0.3, delta=0.01)
        self.assertAlmostEqual(f["low_liking"], -0.1, delta=0.01)
        self.assertAlmostEqual(f["range_liking"], f["peak_liking"] - f["low_liking"])
        self.assertGreater(f["build_liking"], 0.2)        # the reaction grew through the clip
        self.assertEqual(set(cf.FEATURES) - set(f), set())

    def test_too_little_data_gives_none(self):
        self.assertIsNone(cf.clip_features(liking_series(seconds=0.3), [], [], 100.0, 108.0))


def synth_rows(n, auc_signal=True, seed=0, rate=0.35):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        y = int(rng.random() < rate)
        f = {name: rng.gauss(0, 1) for name in cf.FEATURES}
        if auc_signal:
            f["bob_amp"] += 1.6 * y     # one feature really carries the thumb
        rows.append({"user": "a", "song_id": f"s{i}", "thumb": y, "liking": f["last3_liking"],
                     "confidence": 0.9, "emotion": "neutral", "features": f, "setting": "quiet" if i % 2 else "busy",
                     "group_size": 1})
    return rows


class TestCameraModel(unittest.TestCase):
    def test_finds_the_informative_feature_and_gives_it_weight(self):
        w, auc, n = cm.reliability(synth_rows(120), iterations=150)
        self.assertGreater(auc, 0.7)
        self.assertGreater(w, 0.3)
        model = cm.CameraReliability(iterations=150).fit(synth_rows(120))
        top = max(model.w, key=lambda k: abs(model.w[k]))
        self.assertEqual(top, "bob_amp")

    def test_noise_earns_almost_no_weight(self):
        w, auc, _ = cm.reliability(synth_rows(120, auc_signal=False), iterations=150)
        self.assertLess(w, 0.25)

    def test_too_few_songs_means_the_camera_is_ignored(self):
        self.assertEqual(cm.reliability(synth_rows(20))[0], 0.0)
        rows = synth_rows(60, rate=0.05)                           # almost no liked songs
        self.assertEqual(cm.reliability(rows)[0], 0.0)

    def test_soft_reward_is_in_range_and_ordered(self):
        model = cm.CameraReliability(iterations=150).fit(synth_rows(120))
        lo = model.soft_reward({**{n: 0.0 for n in cf.FEATURES}, "bob_amp": -2})
        hi = model.soft_reward({**{n: 0.0 for n in cf.FEATURES}, "bob_amp": 3})
        self.assertTrue(-1 <= lo < hi <= 1)

    def test_a_face_that_says_nothing_new_gives_zero_not_a_constant_negative(self):
        rows = synth_rows(200, rate=0.2)                                    # stingy listener: likes 20%
        model = cm.CameraReliability(iterations=150).fit(rows)
        typical = {n: model.mean[n] for n in cf.FEATURES}                   # an average-looking clip
        self.assertLess(abs(model.soft_reward(typical)), 0.15)
        avg = sum(model.soft_reward(r["features"]) for r in rows) / len(rows)
        self.assertLess(abs(avg), 0.2)                                      # no built-in lean toward dislike


class TestCombine(unittest.TestCase):
    def test_known_answers(self):
        self.assertEqual(combine(thumb=1)[0], 1.0)
        self.assertEqual(combine(thumb=0)[0], -1.0)
        self.assertAlmostEqual(combine(camera=1.0, camera_weight=0.2)[0], 0.2)       # weak signal alone stays small
        self.assertAlmostEqual(combine(thumb=1, camera=-1.0, camera_weight=0.2)[0], (1 - 0.2) / 1.2)
        self.assertAlmostEqual(combine(behaviour="replaced")[0], -0.7)
        self.assertAlmostEqual(combine(thumb=0, behaviour="replaced")[0], -1.0)
        self.assertIsNone(combine()[0])
        self.assertIsNone(combine(camera=0.9, camera_weight=0.0)[0])                 # weight 0 = ignored

    def test_a_thumb_outvotes_a_weak_camera(self):
        r, _ = combine(thumb=1, camera=-1.0, camera_weight=0.3)
        self.assertGreater(r, 0.4)

    def test_bad_thumb_value_is_rejected(self):
        with self.assertRaises(ValueError):
            combine(thumb="maybe")


class FeatureFeedback(StubFeedback):
    def __init__(self, likings, feats):
        super().__init__(likings)
        self.feats = iter(feats)

    def clip_features(self):
        return next(self.feats)


class TestRewardModel(unittest.TestCase):
    def reaction(self, liking=0.3):
        return GroupReaction(0.0, "s0", (PersonReaction(0, "happy", liking, 0.9),), liking, {"happy": 1.0})

    def test_no_signal_means_the_bandit_skips_learning(self):
        out = CameraRewardModel()(SONGS[0], self.reaction(), None)       # camera weight 0, no thumb
        self.assertEqual(len(out.people), 0)
        info = BanditAgent(seed=0).update(SONGS[0], out)
        self.assertTrue(info["skipped"])

    def test_thumb_becomes_the_reward(self):
        rm = CameraRewardModel()
        out = rm(SONGS[0], self.reaction(-0.4), None, thumb=1)
        self.assertEqual(out.group_liking, 1.0)
        out = rm(SONGS[0], self.reaction(0.9), None, thumb=0)            # a happy face does not beat a thumb down
        self.assertEqual(out.group_liking, -1.0)

    def test_fitted_camera_adds_a_small_signal(self):
        rm = CameraRewardModel().fit(synth_rows(120))
        self.assertGreater(rm.weight, 0)
        fb = FeatureFeedback([0.1], [{**{n: 0.0 for n in cf.FEATURES}, "bob_amp": 3.0}])
        out = rm(SONGS[0], self.reaction(), fb)
        self.assertTrue(0 < out.group_liking <= rm.weight)

    def test_loop_uses_thumbs_and_behaviour(self):
        pool = SONGS
        fb = FeatureFeedback([0.5] * 6, [None] * 6)
        fb.stop = lambda: None
        thumbs = {0: 1, 1: 0}
        order = []

        def thumb_fn(song):
            order.append(song.song_id)
            return 1 if len(order) % 2 else 0

        out = run(4, pool=pool, feedback=fb, listen_seconds=0, round_delay=0, overrides={2: pool[3].song_id},
                  verbose=False, reward_model=CameraRewardModel(), thumb_fn=thumb_fn)
        rewards = [e["reward"] for e in out["log"]]
        self.assertEqual(len(rewards), 4)
        self.assertTrue(all(abs(abs(r) - 1) < 1e-9 or abs(abs(r) - 0.7) < 1e-9 or r != 0 for r in rewards))
        self.assertEqual(rewards[0], 1.0)           # first song: thumb up


class TestCollectWithFeatures(unittest.TestCase):
    def test_group_mode_stores_features_setting_and_majority(self):
        fb = FeatureFeedback([.3, .1, -.2], [{"mean_liking": .2}, {"mean_liking": .1}, {"mean_liking": 0.0}])
        answers = iter(["2", "1", "x", "3"])     # 2 of 3 -> up; 1 of 3 -> down; bad input then 3 of 3
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.csv"
            n = rc.collect(fb, SONGS, "a", path, target=3, listen_seconds=0, input_fn=lambda p: next(answers),
                           say=lambda *_: None, setting="busy", group_size=3)
            rows = ag.load_rows(path, "a")
        self.assertEqual(n, 3)
        self.assertEqual([r["thumb"] for r in rows], [1, 0, 1])
        self.assertEqual({r["setting"] for r in rows}, {"busy"})
        self.assertEqual(rows[0]["group_size"], 3)
        self.assertEqual(rows[0]["features"], {"mean_liking": .2})

    def test_even_split_is_not_saved(self):
        fb = FeatureFeedback([.3, .3], [None, None])
        answers = iter(["1", "q"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.csv"
            n = rc.collect(fb, SONGS, "a", path, target=3, listen_seconds=0, input_fn=lambda p: next(answers),
                           say=lambda *_: None, group_size=2)
        self.assertEqual(n, 0)

    def test_old_files_are_migrated_not_broken(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.csv"
            path.write_text("user,song_id,thumb,liking,confidence,emotion,time\na,s0,1,0.1,0.9,happy,1.0\n")
            rc.append_row(path, {"user": "a", "song_id": "s1", "thumb": 0, "liking": 0.0, "confidence": .9,
                                 "emotion": "sad", "time": 2.0, "features": json.dumps({"x": 1}), "setting": "busy",
                                 "group_size": 1, "likes": ""})
            rows = ag.load_rows(path, "a")
        self.assertEqual([r["song_id"] for r in rows], ["s0", "s1"])
        self.assertIsNone(rows[0]["features"])
        self.assertEqual(rows[1]["features"], {"x": 1})


class TestCompareFeatures(unittest.TestCase):
    def test_informative_movement_passes_the_preset_gate(self):
        c = ag.compare_features(synth_rows(100), iterations=100, boot=100)
        self.assertEqual(c["verdict"], "PASS")
        self.assertGreater(c["gain"], 0.05)

    def test_pure_noise_does_not_pass(self):
        c = ag.compare_features(synth_rows(100, auc_signal=False), iterations=100, boot=100)
        self.assertNotEqual(c["verdict"], "PASS")

    def test_not_enough_data_and_no_data(self):
        self.assertEqual(ag.compare_features(synth_rows(30), iterations=50, boot=20)["verdict"], "NOT ENOUGH DATA")
        old = rows_from([1, 0] * 20, [.1] * 40)
        self.assertEqual(ag.compare_features(old)["verdict"], "NO FEATURE DATA")


if __name__ == "__main__":
    unittest.main()
