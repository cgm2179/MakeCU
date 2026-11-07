"""Playlist control (Person C). No audio hardware yet: play() just records state.

Manual override: the phone posts a ManualSongChange to the bus at any time, from
any thread. Nothing reacts to it immediately, so whatever is playing (and whatever
the robot is doing) is not interrupted. The controller picks the change up at the
next song boundary via take_manual_change().
"""
from __future__ import annotations

from shared.bus.bus import MUSIC_MANUAL_CHANGE, Bus
from shared.schemas.messages import Song


class PlaylistController:
    def __init__(self, catalog: list[Song], bus: Bus):
        self.catalog = {s.song_id: s for s in catalog}
        self.bus = bus
        self.now_playing: Song | None = None
        self.history: list[str] = []

    def play(self, song: Song) -> None:
        # Real version: hand `song` to the audio output here.
        self.now_playing = song
        self.history.append(song.song_id)

    def played_ids(self) -> set:
        return set(self.history)

    def take_manual_change(self) -> Song | None:
        """Call at a song boundary. Returns the song the user asked for, if any.

        If the user changed songs several times, the latest valid request wins.
        Unknown song ids are ignored.
        """
        chosen = None
        for msg in self.bus.drain(MUSIC_MANUAL_CHANGE):
            if msg.song_id in self.catalog:
                chosen = self.catalog[msg.song_id]
        return chosen
