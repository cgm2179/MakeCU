"""Bridge to the camera package (perception/perception_cv) for the music loop.

Same interface as fake_reaction.py: get_group_reaction(song_id, phone=None) -> GroupReaction,
so it swaps in for the fake listener. Their package only imports with perception/ on the
path; import_perception_cv() adds it, so the loop can run from the repo root and none of
their files had to move.

Needs the camera package's dependencies (opencv-python, fastapi, uvicorn; see
perception/perception_cv/README.md), so run the loop with a Python that has them.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

PERCEPTION_DIR = Path(__file__).resolve().parents[1]  # <repo>/perception


def import_perception_cv():
    """Import their package from the repo root. Appends (never inserts) so nothing is shadowed."""
    if str(PERCEPTION_DIR) not in sys.path:
        sys.path.append(str(PERCEPTION_DIR))
    import perception_cv

    return perception_cv


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CameraFeedback:
    """Runs the perception service in a background thread and reads group reactions from it.

    mock=True  : build_service(mock=True) (random fake faces, no model, no camera). In mock mode
                 no phone or webcam is sending frames, so a small thread feeds synthetic frames
                 to the service, standing in for the camera.
    model=...  : use a custom EmotionModel instead (e.g. TasteMockModel) through the real
                 tracker/scoring pipeline.
    mock=False : real DeepFace model; frames then have to come from phones or a webcam
                 (not tested yet).
    empty_rounds : rounds (1-based) in which nobody is "in view", to test the empty case
                 (needs a model that supports it, like TasteMockModel).
    """

    def __init__(self, mock=True, model=None, host="127.0.0.1", port=None, detector="opencv",
                 feed_fps=20, empty_rounds=()):
        self.mock, self.model, self.host, self.port = mock, model, host, port
        self.detector, self.feed_fps = detector, feed_fps
        self.empty_rounds = set(empty_rounds)
        self.service = self.server = self._pcv = None
        self._feeder = None
        self._stop = threading.Event()
        self._round = 0
        self.feeder_error: Exception | None = None

    def start(self) -> None:
        if self.server is not None:  # already started
            return
        pcv = self._pcv = import_perception_cv()
        if self.model is not None:
            from perception_cv.pipeline import PerceptionService

            self.service = PerceptionService(self.model)
        else:
            self.service = pcv.build_service(mock=self.mock, detector=self.detector)
        self.server = pcv.serve(self.service, host=self.host, port=self.port or _free_port(),
                                background=True)
        for _ in range(100):  # wait up to 10 s for the server thread to come up
            if getattr(self.server, "started", False):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("perception service did not start")
        if self.mock or self.model is not None:
            self._feeder = threading.Thread(target=self._feed_frames, daemon=True)
            self._feeder.start()

    def _feed_frames(self) -> None:
        import cv2
        import numpy as np

        ok, jpg = cv2.imencode(".jpg", np.full((240, 320, 3), 127, np.uint8))
        data = jpg.tobytes()
        while not self._stop.is_set():
            try:
                self.service.process_frame("mock-camera", data)
            except Exception as err:  # keep the loop alive; surface it later
                self.feeder_error = err
            time.sleep(1.0 / self.feed_fps)

    def begin_song(self, song) -> None:
        """Called when a song starts (lets a taste-aware mock model know what is playing)."""
        self._round += 1
        if hasattr(self.model, "set_song"):
            self.model.set_song(song, present=self._round not in self.empty_rounds)

    def get_group_reaction(self, song_id: str, phone=None):
        """The group's current reaction (smoothed over the last fraction of a second)."""
        return self._pcv.get_group_reaction(song_id)

    def stop(self) -> None:
        self._stop.set()
        if self.server is not None:
            self.server.should_exit = True
