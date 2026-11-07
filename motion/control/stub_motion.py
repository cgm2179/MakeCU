"""Stand-in for the robot's movement loop (Person D replaces this).

Runs in its own thread and never imports anything from music/. It records its
tick times so a test can prove music events do not interrupt it.
"""
from __future__ import annotations

import threading
import time


class MotionStub:
    def __init__(self, period: float = 0.01):
        self.period = period
        self.tick_times: list[float] = []
        self.restarts = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            self.restarts += 1  # a second start() would mean motion was reset
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick_times.append(time.monotonic())
            time.sleep(self.period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    def longest_gap(self) -> float:
        t = self.tick_times
        return max((b - a for a, b in zip(t, t[1:])), default=0.0)
