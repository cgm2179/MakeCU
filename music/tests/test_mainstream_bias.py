"""Mainstream bias: the database path leans toward well-known songs (popularity score).
Needs the Spotify CSV (python3 -m music.catalog.download_spotify); skipped otherwise.
Run: python3 -m unittest music.tests.test_mainstream_bias -v"""
import statistics as st
import unittest

from integration.orchestrator.run_loop import run
from music.catalog.songs import SONGS, get_candidates
from music.catalog.spotify_catalog import dataset_available, load_spotify
from music.filter.mood_targets import PROFILES, select_pool
from perception.emotion.fake_reaction import _reaction
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context
from shared.schemas.messages import CandidatePool, Filter, FilterSource

MOODS = ("gym", "study", "sad", "happy", "party", "chill")
pop = lambda s: s.audio_features["popularity"]


def pool_for(mood, bias):
    return get_candidates(Filter((FilterSource("event", mood),)), mainstream_bias=bias)


@unittest.skipUnless(dataset_available(), "Spotify CSV not downloaded")
class TestPoolBias(unittest.TestCase):
    def test_every_mood_returns_well_known_songs_first(self):
        for mood in MOODS:
            off, on = pool_for(mood, 0.0), pool_for(mood, 0.5)
            first_off, first_on = st.mean(map(pop, off[:10])), st.mean(map(pop, on[:10]))
            self.assertGreaterEqual(first_on, first_off + 10, mood)  # clearly more known
            self.assertGreaterEqual(first_on, 55, mood)
            self.assertGreater(st.mean(map(pop, on)), st.mean(map(pop, off)) + 3, mood)  # whole pool leans too
            # best-first ordering: the head of the pool is better known than its tail
            self.assertGreater(st.mean(map(pop, on[:50])), st.mean(map(pop, on[-50:])), mood)

    def test_pool_still_matches_the_mood(self):
        gym, study, sad, happy = (pool_for(m, 0.5) for m in ("gym", "study", "sad", "happy"))
        m = lambda p, k: st.mean(s.audio_features[k] for s in p)
        self.assertGreater(m(gym, "energy"), 0.85); self.assertGreater(m(gym, "tempo"), 140)
        self.assertGreater(m(study, "instrumentalness"), 0.7); self.assertLess(m(study, "energy"), 0.35)
        self.assertLess(m(sad, "valence"), 0.3); self.assertLess(m(sad, "energy"), 0.4)
        self.assertGreater(m(happy, "valence"), 0.7); self.assertGreater(m(happy, "energy"), 0.6)

    def test_strength_is_configurable_and_monotonic(self):
        firsts = [st.mean(map(pop, pool_for("gym", b)[:10])) for b in (0.0, 0.25, 0.5, 1.0)]
        self.assertEqual(firsts, sorted(firsts))
        self.assertGreater(firsts[-1], firsts[0] + 20)

    def test_bias_zero_is_the_old_behaviour(self):
        songs = load_spotify()
        old = [s.song_id for s in select_pool(songs, PROFILES["gym"], 500)]
        self.assertEqual([s.song_id for s in pool_for("gym", 0.0)], old)

    def test_pool_is_never_short_even_for_niche_moods_or_a_harsh_popularity_floor(self):
        songs = load_spotify()
        for mood in MOODS:  # a floor almost nothing meets: must be topped up, not shrunk
            pool = select_pool(songs, PROFILES[mood], 500, min_popularity=99, mainstream_bias=1.0)
            self.assertEqual(len(pool), 500, mood)
            self.assertEqual(len({s.song_id for s in pool}), 500, mood)  # no duplicates from the top-up
        weird = {"valence": (0.0, 1.0), "energy": (1.0, 1.0), "instrumentalness": (1.0, 1.0)}
        self.assertEqual(len(select_pool(songs, weird, 500, mainstream_bias=1.0)), 500)
        self.assertEqual(len(select_pool(songs[:40], PROFILES["gym"], 500, mainstream_bias=1.0)), 40)  # fewer than asked: all of them

    def test_random_filter_also_leans_but_still_varies(self):
        flt = Filter((FilterSource("random", "any"),))
        on = get_candidates(flt, mainstream_bias=1.0)
        off = get_candidates(flt, mainstream_bias=0.0)
        self.assertEqual(len(on), 500)
        self.assertGreater(st.mean(map(pop, on)), st.mean(map(pop, off)) + 3)
        self.assertNotEqual([s.song_id for s in get_candidates(flt, mainstream_bias=1.0)], [s.song_id for s in on])


@unittest.skipUnless(dataset_available(), "Spotify CSV not downloaded")
class TestAgentPrior(unittest.TestCase):
    def first_picks(self, pool, bias, seeds=60, rounds=3):
        picks = []
        for seed in range(seeds):
            ag = BanditAgent(seed=seed, mainstream_bias=bias)
            ag.mainstream_active = bias > 0
            hist = []
            for _ in range(rounds):
                s, _m = ag.choose_next_song(list(pool), hist[-1] if hist else None, Context("event:gym"), recent_ids=hist[-4:])
                hist.append(s.song_id)
                picks.append(s)
        return picks

    def test_early_picks_lean_well_known_but_still_vary(self):
        for mood in ("gym", "study", "sad"):
            pool = pool_for(mood, 0.5)
            on, off = self.first_picks(pool, 0.5), self.first_picks(pool, 0.0)
            self.assertGreater(st.mean(map(pop, on)), st.mean(map(pop, off)) + 15, mood)
            self.assertGreater(st.mean(map(pop, on)), st.mean(map(pop, pool)) + 5, mood)  # above even the biased pool
            self.assertGreaterEqual(len({s.song_id for s in on}), 25, mood)               # rotates, not one fixed opener
            self.assertTrue(any(pop(s) < st.median(map(pop, pool)) for s in on), mood)    # exploration still reaches less-known songs

    def test_prior_fades_as_the_agent_learns(self):
        pool = pool_for("gym", 0.5)
        ag = BanditAgent(seed=0, mainstream_bias=0.5)
        ag.mainstream_active = True
        top = max(pool, key=pop)
        start = ag.mainstream_prior(top)
        ag.n_updates = 20
        self.assertLess(ag.mainstream_prior(top), start * 0.3)
        self.assertGreater(ag.mainstream_prior(top), 0)  # a lean remains, just a small one

    def test_off_unless_the_pool_is_from_the_database_path(self):
        songs = pool_for("gym", 0.5)
        ag = BanditAgent(seed=0, mainstream_bias=1.0)
        self.assertEqual(ag.mainstream_prior(songs[0]), 0.0)  # off by default
        for source, expect_on in (("database", True), ("playlist", False), ("hardcoded", False), ("unknown", False)):
            out = run(1, seed=0, verbose=False, round_delay=0, overrides={}, mainstream_bias=1.0,
                      pool=CandidatePool(songs[:30], source=source))
            self.assertEqual(out["agent"].mainstream_active, expect_on, source)
        out = run(1, seed=0, verbose=False, round_delay=0, overrides={}, pool=list(songs[:30]), mainstream_bias=1.0)
        self.assertFalse(out["agent"].mainstream_active)  # a plain list (e.g. a user's own playlist) is never biased

    def test_a_less_known_song_the_user_clearly_likes_still_gets_found(self):
        pool = pool_for("gym", 0.5)
        liked = lambda s: s.audio_features["valence"] >= 0.7  # taste unrelated to popularity

        class Listener:
            def __init__(self, seed):
                import random
                self.rng = random.Random(seed)

            def react(self, song):
                like = liked(song) != (self.rng.random() < 0.1)
                return _reaction(song.song_id, 1.0 if like else -1.0, "happy" if like else "sad")

        late = []
        for seed in range(100, 108):
            log = run(50, seed=seed, verbose=False, round_delay=0, pool=pool, listener=Listener(seed),
                      overrides={}, mainstream_bias=0.5)["log"]
            late.append(st.mean(liked(e["song"]) for e in log[-15:]))
        self.assertGreater(st.mean(late), 0.7)


class TestHardcodedFallbackUnaffected(unittest.TestCase):
    def test_hardcoded_pool_is_labelled_and_never_biased(self):
        out = run(3, seed=0, verbose=False, round_delay=0, pool=CandidatePool(SONGS, source="hardcoded"),
                  overrides={}, mainstream_bias=1.0)
        self.assertFalse(out["agent"].mainstream_active)


if __name__ == "__main__":
    unittest.main()
