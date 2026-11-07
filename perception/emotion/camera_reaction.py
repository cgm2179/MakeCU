"""Bridge to the camera package (perception/perception_cv) for the music loop.

Same interface as fake_reaction.py: get_group_reaction(song_id, phone=None) -> GroupReaction,
so it swaps in for the fake listener. Their package only imports with perception/ on the
path; import_perception_cv() adds it, so the loop can run from the repo root and none of
their files had to move.

Needs the camera package's dependencies (opencv-python, fastapi, uvicorn; see
perception/perception_cv/README.md), so run the loop with a Python that has them.
"""
from __future__ import annotations

import collections
import dataclasses
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

from perception.emotion import clip_features

PERCEPTION_DIR = Path(__file__).resolve().parents[1]  # <repo>/perception


def import_perception_cv():
    """Import their package from the repo root. Appends (never inserts) so nothing is shadowed."""
    if str(PERCEPTION_DIR) not in sys.path:
        sys.path.append(str(PERCEPTION_DIR))
    import perception_cv

    return perception_cv


MIN_SAMPLE_CONFIDENCE = 0.1
MAX_BASELINE = 0.15  # a neutral face reads at or below ~0 on this model; higher means a smile
MAX_SPREAD = 0.25    # a still face gives a steady reading; more spread means it was moving


def check_baseline(baseline: float, spread: float, max_baseline: float = MAX_BASELINE,
                   max_spread: float = MAX_SPREAD) -> list[str]:
    """Reasons a calibration should be rejected (empty list = accept)."""
    problems = []
    if baseline > max_baseline:
        problems.append(f"baseline {baseline:+.2f} is above {max_baseline:+.2f}, so you looked happy, not neutral")
    if spread > max_spread:
        problems.append(f"spread {spread:.2f} is above {max_spread:.2f}, so your face was moving too much")
    return problems


def speak(text: str) -> None:
    """Say a short cue out loud (macOS `say`), because the person is looking at the camera, not
    the terminal. Falls back to the terminal bell. Never blocks."""
    import shutil
    import subprocess

    if shutil.which("say"):
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("\a", end="", flush=True)


def calibrate_liking(raw: float, baseline: float) -> float:
    """Score a liking relative to a resting baseline: baseline -> 0, +1 stays +1, -1 stays -1.

    A resting face reads negative on this model (e.g. -0.4), so subtracting the baseline puts a
    neutral face near 0 and only real changes move the result. Piecewise-linear so the full
    -1..+1 range is still reachable whatever the baseline is."""
    b = max(-0.95, min(0.95, baseline))
    out = (raw - b) / (1 - b) if raw >= b else (raw - b) / (1 + b)
    return max(-1.0, min(1.0, out))


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
    detector   : DeepFace face detector, default "yunet" (the "opencv" Haar detector did not find
                 a face on a laptop webcam; "ssd"/"opencv" also need OpenCV 4, not 5).
    enable_bored : their "bored" label is a stillness timer, not a model output, and it drifts
                 negative the longer someone sits still. Off by default (ENABLE_BORED=False).
    calibrate_seconds : if > 0, run() calls calibrate() before the first song. It measures the
                 person's resting liking and every later reaction is scored relative to it.
    read_window_seconds : if > 0, the reaction handed to the bandit is the average liking over
                 the last N seconds instead of a single instant (a smile that fades just before
                 the clip ends would otherwise be missed).
    """

    def __init__(self, mock=True, model=None, host="127.0.0.1", port=None, detector="yunet",
                 feed_fps=20, empty_rounds=(), camera_index=None, fallback_indices=(1, 2, 3),
                 max_width=640, enable_bored=False, calibrate_seconds=0.0, read_window_seconds=0.0):
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
        self.enable_bored = enable_bored
        self.calibrate_seconds, self.read_window_seconds = calibrate_seconds, read_window_seconds
        self.baseline: float | None = None
        self.baseline_std = 0.0
        self.last_calibration: dict | None = None
        self._samples = collections.deque(maxlen=3000)  # (time, raw group_liking, dominant emotion)
        self._probs = collections.deque(maxlen=3000)    # (time, mean happy prob, mean surprised prob)
        self._motion = collections.deque(maxlen=20000)  # (time, person_id, cx, cy, w, h) per analysed frame
        self.rest: dict = {}                            # resting smile/movement measured at calibration
        self._sampler = None
        self._clip_start = time.time()

    def start(self) -> None:
        if self.server is not None:  # already started
            return
        pcv = self._pcv = import_perception_cv()
        if self.camera_index is not None and not self.mock:
            self._open_webcam()  # fail fast, before loading the model or starting the server
            if self.model is None:
                self._check_detector()
        from perception_cv.config import PerceptionConfig

        cfg = PerceptionConfig(ENABLE_BORED=self.enable_bored)
        if self.model is not None:
            from perception_cv.pipeline import PerceptionService

            self.service = PerceptionService(self.model, cfg)
        else:
            self.service = pcv.build_service(mock=self.mock, detector=self.detector, cfg=cfg)
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
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    # ---- sampling, calibration, calibrated reading ----
    def _sample_loop(self) -> None:
        """~10 samples a second of the raw group reaction, for calibration and windowed reads."""
        while not self._stop.is_set():
            r = self._pcv.get_group_reaction("sample")
            conf = [p.confidence for p in r.people]
            if conf and sum(conf) / len(conf) >= MIN_SAMPLE_CONFIDENCE:
                now = time.time()
                self._samples.append((now, r.group_liking, max(r.group_emotion, key=r.group_emotion.get)))
                hp = self._mean_probs()
                if hp is not None:
                    self._probs.append((now, *hp))
            time.sleep(0.1)

    def _mean_probs(self):
        """Mean (happy, surprised) probability over the faces currently tracked, or None."""
        svc = self.service
        if svc is None or not hasattr(svc, "_phones"):
            return None
        with svc._lock:
            tracks = [t for ph in svc._phones.values() for t in ph.tracker.tracks.values()]
        if not tracks:
            return None
        return (sum(t.probs.get("happy", 0.0) for t in tracks) / len(tracks),
                sum(t.probs.get("surprised", 0.0) for t in tracks) / len(tracks))

    def calibrate(self, seconds: float | None = None, say=print, discard: float = 2.0,
                  countdown: int = 3, count_step: float = 1.0, max_attempts: int = 3,
                  max_baseline: float = MAX_BASELINE, max_spread: float = MAX_SPREAD,
                  speak_fn=speak) -> float:
        """Measure the resting liking while the person holds a relaxed, NEUTRAL face (no music).

        Each attempt: a "3, 2, 1" countdown (printed and spoken), then `seconds` of measuring; the
        first `discard` seconds are thrown away while the tracker's smoothing settles and the
        median of the rest is the baseline. A baseline above `max_baseline` (smiling) or a spread
        above `max_spread` (moving) is rejected and the person is asked to try again, up to
        `max_attempts` times, after which this raises instead of using a bad baseline."""
        seconds = seconds or self.calibrate_seconds
        last_problem = "unknown"
        for attempt in range(1, max_attempts + 1):
            if countdown > 0:
                say(f"Calibration {attempt}/{max_attempts}: get ready...")
                for n in range(countdown, 0, -1):
                    say(f"  {n}...")
                    speak_fn(str(n))
                    time.sleep(count_step)
            say(f"HOLD A NEUTRAL FACE for {seconds:.0f} s, no smiling (first {discard:.0f} s ignored)")
            speak_fn("hold a neutral face")
            start = time.time()
            time.sleep(seconds)
            vals = [lk for t, lk, _ in list(self._samples) if t >= start + discard]
            if len(vals) < 10:
                last_problem = (f"your face was seen in only {len(vals)} samples; face the camera "
                                "and check the lighting")
                problems = [last_problem]
            else:
                baseline, spread = statistics.median(vals), statistics.pstdev(vals)
                problems = check_baseline(baseline, spread, max_baseline, max_spread)
                if not problems:
                    self.baseline, self.baseline_std = baseline, spread
                    self.rest = clip_features.resting_baseline(list(self._probs), list(self._motion),
                                                               start + discard, start + seconds)
                    say(f"Baseline resting liking = {baseline:+.2f}  (spread {spread:.2f}, {len(vals)} samples) - accepted")
                    speak_fn("thanks")
                    return baseline
                last_problem = f"baseline {baseline:+.2f}, spread {spread:.2f}: " + "; ".join(problems)
            say(f"Calibration rejected: {'; '.join(problems)}")
            if attempt < max_attempts:
                say("  Let's try again.")
                speak_fn("let's try again")
        raise RuntimeError(f"Calibration failed after {max_attempts} attempts ({last_problem}). "
                           "Sit still with a relaxed neutral face, facing the camera.")

    def _cal(self, raw: float) -> float:
        return raw if self.baseline is None else calibrate_liking(raw, self.baseline)

    def clip_trace(self) -> dict | None:
        """Calibrated liking over the current clip: peak, low, mean, and the emotion mix."""
        rows = [x for x in list(self._samples) if x[0] >= self._clip_start]
        if not rows:
            return None
        cal = [self._cal(lk) for _, lk, _ in rows]
        mix = collections.Counter(e for _, _, e in rows)
        return {"n": len(rows), "peak": max(cal), "low": min(cal), "mean": sum(cal) / len(cal),
                "emotions": {e: c / len(rows) for e, c in mix.most_common()}}

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
        now = time.time()
        for f in reply.get("faces", []):
            x, y, w, h = f["box"]
            self._motion.append((now, f["person_id"], x + w / 2, y + h / 2, w, h))

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

    def clip_features(self) -> dict | None:
        """Features of the clip so far: reaction dynamics plus head/face movement, relative to the resting baseline."""
        lik = [(t, self._cal(v)) for t, v, _ in list(self._samples) if t >= self._clip_start]
        return clip_features.clip_features(lik, list(self._probs), list(self._motion),
                                           self._clip_start, time.time(), self.rest)

    def begin_song(self, song) -> None:
        """Called when a song starts (lets a taste-aware mock model know what is playing)."""
        self._round += 1
        self._clip_start = time.time()
        if hasattr(self.model, "set_song"):
            self.model.set_song(song, present=self._round not in self.empty_rounds)

    def get_group_reaction(self, song_id: str, phone=None):
        """The group's current reaction (smoothed over the last fraction of a second)."""
        live = self._pcv.get_group_reaction(song_id)
        self.last_calibration = None
        if not live.people or (self.baseline is None and self.read_window_seconds <= 0):
            return live  # nobody in view (confidence 0 -> the bandit skips it), or nothing to adjust
        raw = live.group_liking
        if self.read_window_seconds > 0:
            recent = [lk for t, lk, _ in list(self._samples) if t >= time.time() - self.read_window_seconds]
            if recent:
                raw = sum(recent) / len(recent)
        cal = self._cal(raw)
        people = [dataclasses.replace(p, liking=self._cal(p.liking)) for p in live.people]
        self.last_calibration = {"raw": raw, "baseline": self.baseline, "calibrated": cal,
                                 "window": self.read_window_seconds, "trace": self.clip_trace()}
        return dataclasses.replace(live, group_liking=cal, people=people)

    def stop(self) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=2)
        if self._feeder is not None:
            self._feeder.join(timeout=5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.server is not None:
            self.server.should_exit = True
