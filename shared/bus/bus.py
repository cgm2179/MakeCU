"""Tiny thread-safe in-process message bus (stand-in for the real transport).

Music and motion talk only through topics here; they never import each other.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict

# Music side
MUSIC_MANUAL_CHANGE = "music.manual_change"
# Motion side
MOTION_COMMAND = "motion.command"


class Bus:
    def __init__(self):
        self._queues = defaultdict(queue.SimpleQueue)
        self._lock = threading.Lock()

    def _q(self, topic):
        with self._lock:
            return self._queues[topic]

    def publish(self, topic: str, message) -> None:
        self._q(topic).put(message)

    def drain(self, topic: str) -> list:
        """Return every message waiting on the topic (never blocks)."""
        q, out = self._q(topic), []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out
