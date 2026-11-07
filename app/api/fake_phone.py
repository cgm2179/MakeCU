"""Fake phone (Person C replaces this with the real app link).

The phone only publishes to the bus. It never touches the player or the robot.
"""
from __future__ import annotations

import time

from shared.bus.bus import MUSIC_MANUAL_CHANGE, Bus
from shared.schemas.messages import ManualSongChange


class FakePhone:
    def __init__(self, bus: Bus):
        self.bus = bus

    def change_song(self, song_id: str) -> None:
        self.bus.publish(MUSIC_MANUAL_CHANGE, ManualSongChange(song_id, time.time()))
