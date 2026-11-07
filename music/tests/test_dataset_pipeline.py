"""Needs the downloaded CSV (python3 -m music.catalog.download_spotify); skipped otherwise.
Run: python3 -m unittest music.tests.test_dataset_pipeline -v"""
import statistics as st
import unittest

from music.catalog.songs import get_candidates
from music.catalog.spotify_catalog import dataset_available, load_spotify
from music.filter.mood_targets import profile_for
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from rl.env.taste_listener import TasteListener
from shared.schemas.messages import Filter, FilterSource


def pool_for(mood, size=500):
    return get_candidates(Filter((FilterSource("event", mood),)), size)


def mean(pool, key):
    return st.mean(s.audio_features[key] for s in pool)


@unittest.skipUnless(dataset_available(), "Spotify CSV not downloaded")
class TestDatasetPipeline(unittest.TestCase):
    def test_loader_maps_columns_to_song_schema(self):
        songs = load_spotify()
        self.assertGreater(len(songs), 50_000)
        self.assertEqual(len({s.song_id for s in songs}), len(songs))  # no duplicate tracks
        s = songs[0]
        for key in ("valence", "energy", "danceability", "tempo", "acousticness", "instrumentalness"):
            self.assertIn(key, s.audio_features)
        self.assertTrue(s.title and s.artist and s.genre and s.mood)
        self.assertTrue(all(x.audio_features["tempo"] > 0 for x in songs))

    def test_mood_pools_sit_near_their_targets(self):
        gym, study, sad, happy = (pool_for(m) for m in ("gym", "study", "sad", "happy"))
        self.assertEqual(len(gym), 500)
        self.assertGreater(mean(gym, "energy"), 0.85)
        self.assertGreater(mean(gym, "tempo"), 140)
        self.assertGreater(mean(study, "instrumentalness"), 0.7)
        self.assertLess(mean(study, "energy"), 0.35)
        self.assertLess(mean(sad, "valence"), 0.3)
        self.assertLess(mean(sad, "energy"), 0.4)
        self.assertGreater(mean(happy, "valence"), 0.7)
        self.assertGreater(mean(happy, "energy"), 0.6)
        # and the moods are distinct from each other
        self.assertGreater(mean(happy, "valence"), mean(sad, "valence") + 0.5)
        self.assertGreater(mean(gym, "energy"), mean(study, "energy") + 0.5)

    def test_aliases_and_unknown_moods(self):
        self.assertIs(profile_for("Studying"), profile_for("study"))
        self.assertIs(profile_for("workout"), profile_for("gym"))
        self.assertIsNone(profile_for("nonsense"))

    def test_bandit_adapts_to_feedback_on_gym_pool(self):
        pool = pool_for("gym")
        liked = ("punk-rock", "emo")
        base = sum(s.genre in liked for s in pool) / len(pool)
        early, late = [], []
        for seed in range(8):
            agent, listener, ctx, picks = BanditAgent(seed=seed), TasteListener(liked, seed=seed), None, []
            emotion = "neutral"
            for _ in range(60):
                ctx = Context("event:gym", emotion)
                ids = [p.song_id for p in picks]
                song, _ = agent.choose_next_song(pool, ids[-1] if ids else None, ctx, ids[-4:])
                reaction = listener.react(song)
                agent.update(song, reaction, ctx)
                emotion = "happy" if reaction.group_liking > 0 else "sad"
                picks.append(song)
            early.append(sum(s.genre in liked for s in picks[:15]) / 15)
            late.append(sum(s.genre in liked for s in picks[-20:]) / 20)
            # it keeps recommending gym-like songs while it adapts
            self.assertGreater(st.mean(s.audio_features["energy"] for s in picks), 0.85)
        print(f"\n  liked-genre share: first 15 plays {st.mean(early):.0%} -> last 20 {st.mean(late):.0%} "
              f"(random would be {base:.0%})")
        self.assertGreater(st.mean(late), st.mean(early) + 0.2)
        self.assertGreater(st.mean(late), 4 * base)


if __name__ == "__main__":
    unittest.main()
