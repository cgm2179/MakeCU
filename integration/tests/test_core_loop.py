"""Run from the repo root: python3 -m unittest integration.tests.test_core_loop -v"""
import unittest

from integration.orchestrator.run_loop import run


class TestCoreLoop(unittest.TestCase):
    def test_manual_override_plays_next_and_motion_is_not_interrupted(self):
        out = run(simulate_rounds=20, seed=3, overrides={6: "s6", 14: "s11"}, verbose=False)
        ids = [e["song_id"] for e in out["log"]]
        modes = [e["mode"] for e in out["log"]]
        # The requested songs play right after the song that was running when asked.
        self.assertEqual((ids[6], modes[6]), ("s6", "phone"))
        self.assertEqual((ids[14], modes[14]), ("s11", "phone"))
        # The song that was playing when the request came was not cut short: it got its reaction.
        self.assertEqual(len(out["rewards"]), 20)
        # Motion ran the whole time, was never restarted, and never stalled.
        motion = out["motion"]
        self.assertEqual(motion.restarts, 0)
        self.assertGreater(len(motion.tick_times), 20)
        self.assertLess(motion.longest_gap(), 0.2)

    def test_agent_learns_over_the_loop(self):
        out = run(simulate_rounds=40, seed=3, verbose=False)
        r = out["rewards"]
        early, late = sum(r[:10]) / 10, sum(r[-10:]) / 10
        print(f"\n  mean reward first 10: {early:+.2f}, last 10: {late:+.2f}")
        self.assertGreater(late, early)


if __name__ == "__main__":
    unittest.main()
