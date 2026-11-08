"""Run from the repo root: python3 -m unittest rl.tests.test_profile -v"""
import tempfile
import unittest

from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from rl.agent.profile import BASE_NONE, ProfileError, load_profile, profile_path, reset_profile, save_profile
from rl.tests.test_eval import fake_catalog
from shared.schemas.messages import GroupReaction, PersonReaction

CTX = Context("event:party", "happy")
SONGS = fake_catalog(60)


def trained_agent(n=40):
    agent = BanditAgent(seed=0)
    for i, s in enumerate(SONGS[:n]):
        liking = 1.0 if s.genre in ("jazz", "folk") else -1.0
        agent.update(s, GroupReaction(0.0, s.song_id, (PersonReaction(0, "happy", liking, 0.8),), liking, {}), CTX)
    agent.choose_next_song(SONGS, context=CTX)  # moves epsilon
    return agent


class TestProfile(unittest.TestCase):
    def test_restart_gives_identical_predictions(self):
        agent = trained_agent()
        fixed = SONGS[:20]
        before = [agent.predict(s, CTX) for s in fixed]
        with tempfile.TemporaryDirectory() as d:
            save_profile(agent, "adi", directory=d)
            loaded = load_profile("adi", directory=d)  # a brand new object, as after a restart
        self.assertEqual(before, [loaded.predict(s, CTX) for s in fixed])  # exactly equal, not approximately
        self.assertEqual(agent.n_updates, loaded.n_updates)
        self.assertEqual(agent.play_counts, loaded.play_counts)
        self.assertEqual(agent.epsilon, loaded.epsilon)
        self.assertEqual([s.song_id for s, _ in agent.recommend(SONGS, CTX, 10)],
                         [s.song_id for s, _ in loaded.recommend(SONGS, CTX, 10)])

    def test_different_base_model_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            save_profile(trained_agent(), "adi", base_model_id=BASE_NONE, directory=d)
            with self.assertRaises(ProfileError) as cm:
                load_profile("adi", base_model_id="base-v1-abc123", directory=d)
            self.assertIn("base model", str(cm.exception))
            self.assertIn("base-v1-abc123", str(cm.exception))

    def test_missing_corrupt_and_wrong_version_files_give_clear_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ProfileError):
                load_profile("nobody", directory=d)
            profile_path("bad", d).write_text("{not json")
            with self.assertRaises(ProfileError):
                load_profile("bad", directory=d)
            profile_path("old", d).write_text('{"version": 0, "base_model_id": "none-v0"}')
            with self.assertRaises(ProfileError):
                load_profile("old", directory=d)

    def test_user_name_cannot_escape_the_folder(self):
        for bad in ("../x", "a/b", "", "a b", "x" * 41):
            with self.assertRaises(ProfileError):
                profile_path(bad)

    def test_reset_forgets_the_user(self):
        with tempfile.TemporaryDirectory() as d:
            save_profile(trained_agent(), "adi", directory=d)
            self.assertTrue(reset_profile("adi", directory=d))
            self.assertFalse(reset_profile("adi", directory=d))
            with self.assertRaises(ProfileError):
                load_profile("adi", directory=d)

    def test_save_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            save_profile(trained_agent(), "adi", directory=d)
            save_profile(trained_agent(10), "adi", directory=d)  # overwrite
            import os
            self.assertEqual(os.listdir(d), ["adi.json"])


if __name__ == "__main__":
    unittest.main()
