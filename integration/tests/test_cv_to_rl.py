"""Run from the repo root: python3 -m unittest integration.tests.test_cv_to_rl -v"""
import unittest

from music.catalog.songs import SONGS
from perception.bridge import to_shared
from perception.emotion.fake_reaction import _reaction
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context

BY_ID = {s.song_id: s for s in SONGS}
CTX = Context("event:party", "neutral")


class _Person:
    def __init__(self, emotion, liking, confidence=1.0):
        self.person_id = 0
        self.emotion = emotion
        self.liking = liking
        self.confidence = confidence


class _Live:
    def __init__(self, liking, emotion):
        self.timestamp = 0.0
        self.song_id = "s8"
        self.people = [_Person(emotion, liking)]
        self.group_liking = liking
        self.group_emotion = {emotion: 1.0}


class TestCameraFeedsTheBandit(unittest.TestCase):
    def test_unhappy_camera_frame_discourages_song_and_genre(self):
        agent = BanditAgent(seed=0)
        song = BY_ID["s8"]  # hip-hop
        reaction = to_shared(_Live(-0.8, "sad"), song.song_id)
        self.assertLessEqual(reaction.group_liking, -0.2)
        self.assertEqual(reaction.people[0].emotion, "sad")

        before = agent.predict(song, CTX)
        before_genre = agent.weights.get(("global", "genre:hip-hop"), 0.0)
        agent.update(song, reaction, CTX)
        self.assertLess(agent.predict(song, CTX), before)
        self.assertLess(agent.weights[("global", "genre:hip-hop")], before_genre)

    def test_keyboard_dislike_uses_the_same_update(self):
        """The fake 'd' key and a sad camera frame are the same kind of reward."""
        song = BY_ID["s1"]
        keyboard = _reaction("s1", -1.0, "sad")
        camera = to_shared(_Live(-1.0, "sad"), "s1")
        a, b = BanditAgent(seed=1), BanditAgent(seed=1)
        a.update(song, keyboard, CTX)
        b.update(song, camera, CTX)
        self.assertAlmostEqual(a.predict(song, CTX), b.predict(song, CTX))


if __name__ == "__main__":
    unittest.main()
