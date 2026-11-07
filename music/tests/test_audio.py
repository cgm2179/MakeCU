"""Offline tests (no network, no sound). Run: python3 -m unittest music.tests.test_audio -v"""
import json
import unittest
import urllib.error
from unittest import mock

from integration.orchestrator.run_loop import run
from music.catalog.songs import SONGS
from music.playback.audio import AudioPlayer, PreviewFinder, best_match
from music.playback.playlist import PlaylistController
from shared.bus.bus import Bus
from shared.schemas.messages import Song

SONG = Song("x1", "Blinding Lights - Radio Edit (feat. Someone)", "The Weeknd", "pop",
            {"energy": 0.8}, "upbeat")


def result(track, artist, preview="https://audio-ssl.itunes.apple.com/clip.m4a"):
    return {"trackName": track, "artistName": artist, "previewUrl": preview}


class TestMatching(unittest.TestCase):
    def test_matches_despite_edit_and_feature_tags(self):
        r = best_match(SONG, [result("Blinding Lights", "The Weeknd")])
        self.assertEqual(r["trackName"], "Blinding Lights")

    def test_rejects_wrong_artist_cover_versions(self):
        self.assertIsNone(best_match(SONG, [result("Blinding Lights", "Karaoke Band")]))

    def test_rejects_different_song_by_same_artist(self):
        self.assertIsNone(best_match(SONG, [result("Starboy", "The Weeknd")]))

    def test_ignores_results_without_preview(self):
        self.assertIsNone(best_match(SONG, [result("Blinding Lights", "The Weeknd", preview=None)]))

    def test_picks_closest_title(self):
        r = best_match(SONG, [result("Blinding Lights (Remix Pack)", "The Weeknd", "https://a.apple.com/1"),
                              result("Blinding Lights", "The Weeknd", "https://a.apple.com/2")])
        self.assertEqual(r["previewUrl"], "https://a.apple.com/2")


class TestFinder(unittest.TestCase):
    def test_returns_url_and_caches(self):
        fetch = mock.Mock(return_value=json.dumps({"results": [result("Blinding Lights", "The Weeknd")]}).encode())
        finder = PreviewFinder(fetch=fetch)
        self.assertTrue(finder.find(SONG).endswith("clip.m4a"))
        finder.find(SONG)
        self.assertEqual(fetch.call_count, 1)

    def test_no_match_returns_none_and_is_cached(self):
        fetch = mock.Mock(return_value=json.dumps({"results": []}).encode())
        finder = PreviewFinder(fetch=fetch)
        self.assertIsNone(finder.find(SONG))
        self.assertIsNone(finder.find(SONG))
        self.assertEqual(fetch.call_count, 1)

    def test_network_error_returns_none_but_is_retried_later(self):
        fetch = mock.Mock(side_effect=urllib.error.URLError("offline"))
        finder = PreviewFinder(fetch=fetch)
        with mock.patch("sys.stderr"):
            self.assertIsNone(finder.find(SONG))
            self.assertIsNone(finder.find(SONG))
        self.assertEqual(fetch.call_count, 2)  # failure was not cached


class FakeFinder:
    def __init__(self, urls):
        self.urls = urls

    def find(self, song):
        return self.urls.get(song.song_id)


class TestAudioPlayer(unittest.TestCase):
    def make(self, urls, **kw):
        with mock.patch("music.playback.audio.shutil.which", side_effect=lambda n: "/usr/bin/afplay" if n == "afplay" else None):
            return AudioPlayer(finder=FakeFinder(urls), clip_seconds=10, **kw)

    def test_plays_clip_through_player_process(self):
        popen = mock.Mock()
        player = self.make({"x1": "https://a.apple.com/c.m4a"}, popen=popen, download=lambda u, d: None)
        self.assertTrue(player.play(SONG))
        argv = popen.call_args[0][0]
        self.assertEqual(argv[:3], ["afplay", "-t", "10"])
        self.assertTrue(argv[-1].endswith("x1.m4a"))
        player.close()

    def test_no_preview_returns_false_and_starts_nothing(self):
        popen = mock.Mock()
        player = self.make({}, popen=popen, download=lambda u, d: None)
        self.assertFalse(player.play(SONG))
        popen.assert_not_called()
        player.close()

    def test_refuses_to_download_from_unexpected_host(self):
        player = self.make({"x1": "https://evil.example.com/c.m4a"}, popen=mock.Mock())
        with mock.patch("sys.stderr"):
            self.assertFalse(player.play(SONG))
        player.close()

    def test_starting_a_new_song_stops_the_old_one(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        player = self.make({"x1": "https://a.apple.com/c.m4a"}, popen=mock.Mock(return_value=proc),
                           download=lambda u, d: None)
        player.play(SONG)
        player.play(SONG)
        proc.terminate.assert_called_once()
        player.close()


class FakePlayer:
    """Refuses some songs, like a song with no preview."""

    def __init__(self, refuse):
        self.refuse, self.played = set(refuse), []

    def play(self, song):
        if song.song_id in self.refuse:
            return False
        self.played.append(song.song_id)
        return True

    def wait(self): pass
    def stop(self): pass


class TestSkipInLoop(unittest.TestCase):
    def test_controller_does_not_record_unplayable_song(self):
        ctl = PlaylistController(SONGS, Bus(), FakePlayer({"s1"}))
        self.assertFalse(ctl.play(SONGS[0]))
        self.assertEqual(ctl.history, [])
        self.assertIsNone(ctl.now_playing)
        self.assertTrue(ctl.play(SONGS[1]))

    def test_loop_skips_songs_without_preview(self):
        refuse = {"s4", "s5", "s11", "s1", "s2", "s8"}
        fake = FakePlayer(refuse)
        out = run(simulate_rounds=15, seed=2, verbose=False, round_delay=0, pool=SONGS, player=fake, overrides={})
        self.assertEqual(len(out["log"]), 15)          # still got 15 real rounds
        self.assertTrue(refuse.isdisjoint(fake.played))  # never "played" an unplayable song
        self.assertEqual(fake.played, out["controller"].history)

    def test_unplayable_song_is_not_picked_again_later(self):
        class Counting(FakePlayer):
            def __init__(self, refuse):
                super().__init__(refuse)
                self.attempts = []

            def play(self, song):
                self.attempts.append(song.song_id)
                return super().play(song)

        fake = Counting({"s4", "s5", "s11"})
        run(simulate_rounds=20, seed=1, verbose=False, round_delay=0, pool=SONGS, player=fake, overrides={})
        for sid in fake.refuse:
            self.assertLessEqual(fake.attempts.count(sid), 1)  # tried once, then never again

    def test_loop_stops_cleanly_if_nothing_is_playable(self):
        out = run(simulate_rounds=5, seed=0, verbose=False, round_delay=0, pool=SONGS,
                  player=FakePlayer({s.song_id for s in SONGS}), overrides={})
        self.assertEqual(out["log"], [])


if __name__ == "__main__":
    unittest.main()
