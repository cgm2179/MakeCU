"""Run from the repo root: python3 -m unittest music.tests.test_playlist -v"""
import unittest

from app.api.fake_phone import FakePhone
from music.catalog.songs import SONGS
from music.playback.playlist import PlaylistController
from shared.bus.bus import Bus


class TestPlaylist(unittest.TestCase):
    def setUp(self):
        self.bus = Bus()
        self.ctl = PlaylistController(SONGS, self.bus)
        self.phone = FakePhone(self.bus)

    def test_phone_change_does_not_interrupt_current_song(self):
        self.ctl.play(SONGS[0])
        self.phone.change_song("s6")
        self.assertEqual(self.ctl.now_playing.song_id, "s1")  # still playing
        self.assertEqual(self.ctl.history, ["s1"])

    def test_change_applies_at_boundary(self):
        self.ctl.play(SONGS[0])
        self.phone.change_song("s6")
        self.assertEqual(self.ctl.take_manual_change().song_id, "s6")
        self.assertIsNone(self.ctl.take_manual_change())  # consumed once

    def test_latest_valid_request_wins_and_unknown_ids_ignored(self):
        self.phone.change_song("s3")
        self.phone.change_song("nope")
        self.phone.change_song("s7")
        self.phone.change_song("also-nope")
        self.assertEqual(self.ctl.take_manual_change().song_id, "s7")


if __name__ == "__main__":
    unittest.main()
