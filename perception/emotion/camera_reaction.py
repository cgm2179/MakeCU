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
    mock=False : real DeepFace model. With camera_index set, frames come from that local webcam
                 (index 0 first, then fallback_indices); otherwise from phones that connect to
                 the server. If the camera or model cannot be opened, start() raises with the
                 exact error (it never falls back to mock).
    empty_rounds : rounds (1-based) in which nobody is "in view", to test the empty case
                 (needs a model that supports it, like TasteMockModel).
    """

    def __init__(self, mock=True, model=None, host="127.0.0.1", port=None, detector="opencv",
                 feed_fps=20, empty_rounds=(), camera_index=None, fallback_indices=(1, 2, 3),
                 max_width=640):
        self.mock, self.model, self.host, self.port = mock, model, host, port
        self.camera_index, self.fallback_indices = camera_index, tuple(fallback_indices)
        self.max_width = max_width  # frames are shrunk to this width before the model sees them
        self._cap = None
        self.camera_used: int | None = None
        self._frames = 0
        self._ms = 0.0
        self.last_faces = 0
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
        if self.camera_index is not None and not self.mock:
            self._open_webcam()  # fail fast, before loading the model or starting the server
            if self.model is None:
                self._check_detector()
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
        if self._cap is not None:
            self._feeder = threading.Thread(target=self._read_webcam, daemon=True)
            self._feeder.start()
        elif self.mock or self.model is not None:
            self._feeder = threading.Thread(target=self._feed_frames, daemon=True)
            self._feeder.start()

    # ---- real webcam ----
    def _open_webcam(self) -> None:
        import cv2

        errors = []
        for idx in (self.camera_index, *[i for i in self.fallback_indices if i != self.camera_index]):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                errors.append(f"index {idx}: could not open")
                continue
            frame = None
            for _ in range(30):  # cameras often return black frames while they warm up
                ok, f = cap.read()
                if ok and f.mean() > 3:
                    frame = f
                    break
                time.sleep(0.1)
            if frame is None:
                cap.release()
                errors.append(f"index {idx}: opened but only black/no frames")
                continue
            self._cap, self.camera_used = cap, idx
            self._warm_frame = frame
            return
        raise RuntimeError("No usable camera. " + "; ".join(errors) +
                           ". On macOS, allow camera access for the app running this "
                           "(System Settings > Privacy & Security > Camera).")

    def _check_detector(self) -> None:
        """Their model turns every ValueError into "no face found", which would hide a broken
        detector (e.g. missing OpenCV module). Run one frame with enforce_detection=False so a
        configuration error is raised here with its exact message."""
        try:
            from deepface import DeepFace

            DeepFace.analyze(img_path=self._warm_frame, actions=["emotion"], detector_backend=self.detector,
                             enforce_detection=False, silent=True)
        except Exception as err:
            self.stop()
            raise RuntimeError(f"DeepFace detector '{self.detector}' cannot run: {type(err).__name__}: {err}") from err

    def _process(self, phone_id: str, jpg: bytes) -> None:
        t0 = time.time()
        reply = self.service.process_frame(phone_id, jpg)
        self._ms += (time.time() - t0) * 1000
        self._frames += 1
        self.last_faces = len(reply.get("faces", []))

    def _read_webcam(self) -> None:
        import cv2

        phone_id = f"local-camera-{self.camera_used}"
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
                if not ok:
                    raise RuntimeError("could not read a frame from the camera")
                h, w = frame.shape[:2]
                if w > self.max_width:
                    frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)))
                ok, enc = cv2.imencode(".jpg", frame)
                if ok:
                    self._process(phone_id, enc.tobytes())
            except Exception as err:  # keep going; surface it afterwards
                self.feeder_error = err
                time.sleep(0.1)

    def consume_stats(self):
        """(frames analysed, average ms per frame, faces in the latest frame) since the last call."""
        n, ms = self._frames, self._ms
        self._frames, self._ms = 0, 0.0
        return n, (ms / n if n else 0.0), self.last_faces

    def _feed_frames(self) -> None:
        import cv2
        import numpy as np

        ok, jpg = cv2.imencode(".jpg", np.full((240, 320, 3), 127, np.uint8))
        data = jpg.tobytes()
        while not self._stop.is_set():
            try:
                self._process("mock-camera", data)
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
        if self._feeder is not None:
            self._feeder.join(timeout=5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.server is not None:
            self.server.should_exit = True
