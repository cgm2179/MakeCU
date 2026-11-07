"""Run from the repo root: python3 -m unittest rl.tests.test_bandit -v"""
import unittest

from music.catalog.songs import SONGS
from perception.emotion.fake_reaction import _reaction
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from rl.env.simulated_listener import SimulatedListener

BY_ID = {s.song_id: s for s in SONGS}
PARTY = Context("event:party", "neutral")


def like(song_id, liked=True):
    return _reaction(song_id, 1.0 if liked else -1.0, "happy" if liked else "sad")


class TestBandit(unittest.TestCase):
    def test_recommends_unplayed_songs_from_learned_features(self):
        agent = BanditAgent(seed=0)
        # Train on only 3 songs: likes jazz (s4, s5), dislikes hip-hop (s8).
        for _ in range(5):
            for sid, liked in (("s4", True), ("s5", True), ("s8", False)):
                agent.update(BY_ID[sid], like(sid, liked), PARTY)
        trained = {"s4", "s5", "s8"}
        recs = agent.recommend(SONGS, PARTY, k=16, unplayed_only=True)
        ranked = [s.song_id for s, _ in recs]
        print("\n  Unplayed songs, best first:", [(s.title, round(p, 2)) for s, p in recs[:4]], "...")
        self.assertTrue(trained.isdisjoint(ranked))                    # only unplayed ones
        # Top pick shares features with the liked songs (jazz, or calm/low-energy like them).
        top = BY_ID[ranked[0]]
        self.assertTrue(top.genre == "jazz" or top.mood == "calm", top)
        self.assertEqual(BY_ID[ranked[-1]].genre, "hip-hop")           # worst is like the disliked one
        self.assertLess(ranked.index("s6"), ranked.index("s9"))        # unplayed jazz > unplayed hip-hop
        self.assertLess(ranked.index("s7"), ranked.index("s9"))

    def test_context_changes_preferences(self):
        agent = BanditAgent(seed=0)
        happy, sad = Context("event:party", "happy"), Context("event:party", "sad")
        loud, calm = BY_ID["s9"], BY_ID["s11"]  # HUMBLE. (high energy), Clair de Lune (calm)
        for _ in range(8):
            agent.update(loud, like("s9", True), happy)
            agent.update(calm, like("s11", False), happy)
            agent.update(loud, like("s9", False), sad)
            agent.update(calm, like("s11", True), sad)
        print(f"\n  happy: loud {agent.predict(loud, happy):+.2f} calm {agent.predict(calm, happy):+.2f}"
              f" | sad: loud {agent.predict(loud, sad):+.2f} calm {agent.predict(calm, sad):+.2f}")
        self.assertGreater(agent.predict(loud, happy), agent.predict(calm, happy))
        self.assertGreater(agent.predict(calm, sad), agent.predict(loud, sad))

    def test_like_raises_and_dislike_lowers_score(self):
        agent = BanditAgent(seed=0)
        s = BY_ID["s1"]
        base = agent.predict(s, PARTY)
        agent.update(s, like("s1", True), PARTY)
        up = agent.predict(s, PARTY)
        agent.update(s, like("s1", False), PARTY)
        agent.update(s, like("s1", False), PARTY)
        self.assertGreater(up, base)
        self.assertLess(agent.predict(s, PARTY), up)

    def test_manual_override_is_recorded_as_a_signal(self):
        agent = BanditAgent(seed=0)
        chosen, replaced = BY_ID["s6"], BY_ID["s9"]
        c0, r0 = agent.predict(chosen, PARTY), agent.predict(replaced, PARTY)
        agent.record_manual_override(chosen, replaced, PARTY)
        self.assertAlmostEqual(agent.predict(chosen, PARTY) - c0, 0.3, places=6)
        self.assertAlmostEqual(agent.predict(replaced, PARTY) - r0, -0.15, places=6)
        self.assertGreater(agent.predict(BY_ID["s4"], PARTY), 0)  # other jazz benefits too

    def test_learns_simulated_listener(self):
        agent, listener = BanditAgent(seed=1), SimulatedListener(seed=1)
        picks = []
        for step in range(60):
            ctx = Context("event:party", listener.emotion)
            song, _ = agent.choose_next_song(SONGS, None, ctx, recent_ids=picks[-3:])
            agent.update(song, listener.react(song), ctx)
            picks.append(song.song_id)
        late = [BY_ID[i] for i in picks[-20:]]
        liked_share = sum(s.genre == "jazz" or s.mood == "calm" for s in late) / len(late)
        print(f"\n  late picks that are jazz/calm: {liked_share:.0%}")
        self.assertGreater(liked_share, 0.6)


if __name__ == "__main__":
    unittest.main()
