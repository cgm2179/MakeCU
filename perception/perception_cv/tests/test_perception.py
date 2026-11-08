import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from perception_cv.capture.phone_server import create_app
from perception_cv.capture import local_camera
from perception_cv.config import PerceptionConfig
from perception_cv.emotion.base import Face
from perception_cv.emotion.mock_model import MockModel
from perception_cv.emotion import tuning
from perception_cv.group.scoring import aggregate
from perception_cv.pipeline import PerceptionService
from perception_cv.types import PersonReaction


def jpeg(w=320, h=240) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    return buf.tobytes()


class FixedModel:
    """One face with a fixed raw emotion distribution."""
    def __init__(self, **probs):
        base = {k: 0.0 for k in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")}
        base.update(probs)
        self.probs = base

    def predict(self, frame):
        h, w = frame.shape[:2]
        return [Face((w // 3, h // 3, w // 4, w // 4), dict(self.probs), 0.9)]


def run(model, frames=8):
    svc = PerceptionService(model, PerceptionConfig())
    for _ in range(frames):
        svc.process_frame("p", jpeg())
    return svc.get_group_reaction("s")


def test_happy_is_positive():
    r = run(FixedModel(happy=1.0))
    assert r.group_liking > 0.9 and r.people[0].emotion == "happy"
    assert abs(sum(r.group_emotion.values()) - 1) < 1e-6


def test_angry_is_negative():
    r = run(FixedModel(angry=0.7, disgust=0.3))
    assert r.group_liking < -0.9 and r.people[0].emotion == "angry"


def test_emotion_tuning_disables_classes_and_renormalizes(monkeypatch):
    monkeypatch.setattr(tuning, "ENABLED_RAW_EMOTIONS", {"happy", "sad", "neutral"})
    monkeypatch.setattr(tuning, "MIN_PROBABILITY", {"happy": 0.5, "sad": 0.0, "neutral": 0.0})

    probs = tuning.tune_probabilities({
        "happy": 0.4,
        "sad": 0.3,
        "neutral": 0.3,
        "angry": 1.0,
    })

    assert probs == {"sad": 0.5, "neutral": 0.5}


def test_emotion_tuning_omits_face_when_no_emotion_passes(monkeypatch):
    monkeypatch.setattr(tuning, "ENABLED_RAW_EMOTIONS", {"happy", "sad", "neutral"})
    monkeypatch.setattr(tuning, "MIN_PROBABILITY", {"happy": 0.8, "sad": 0.8, "neutral": 0.8})

    assert tuning.tune_probabilities({"happy": 0.4, "sad": 0.3, "neutral": 0.3}) == {}


def test_confidence_ramps_with_track_age():
    svc = PerceptionService(FixedModel(happy=1.0))
    svc.process_frame("p", jpeg())
    c1 = svc.get_group_reaction("s").people[0].confidence
    for _ in range(6):
        svc.process_frame("p", jpeg())
    assert svc.get_group_reaction("s").people[0].confidence > c1


def test_stale_phone_ignored():
    cfg = PerceptionConfig(STALE_SECONDS=0.05)
    svc = PerceptionService(FixedModel(happy=1.0), cfg)
    svc.process_frame("p", jpeg())
    time.sleep(0.1)
    assert svc.get_group_reaction("s").people == []


def test_two_phones_merge_and_cap():
    svc = PerceptionService(MockModel(n_faces=3, seed=1), PerceptionConfig(MAX_PEOPLE=4))
    svc.process_frame("a", jpeg())
    svc.process_frame("b", jpeg())
    r = svc.get_group_reaction("s")
    assert len(r.people) == 4
    assert len({p.person_id for p in r.people}) == 4


def test_empty_group():
    r = aggregate("s", [], time.time(), 6)
    assert r.group_liking == 0 and r.group_emotion["neutral"] == 1.0


def test_aggregate_mean_liking():
    ps = [PersonReaction(1, "happy", 1.0, 1.0), PersonReaction(2, "angry", -1.0, 1.0)]
    r = aggregate("s", ps, time.time(), 6)
    assert r.group_liking == 0 and r.group_emotion["happy"] == 0.5


def test_websocket_roundtrip():
    svc = PerceptionService(MockModel(n_faces=2, seed=0))
    client = TestClient(create_app(svc))
    with client.websocket_connect("/ws?phone=a") as ws:
        ws.send_bytes(jpeg())
        msg = ws.receive_json()
        assert len(msg["faces"]) == 2
        assert svc.phone_count() == 1
    assert svc.phone_count() == 0


def test_local_camera_processes_frames_and_releases_device(monkeypatch):
    class FakeCamera:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((240, 320, 3), np.uint8)

        def release(self):
            self.released = True

    camera = FakeCamera()
    monkeypatch.setattr(local_camera.cv2, "VideoCapture", lambda index, backend: camera)
    monkeypatch.setattr(local_camera.cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(local_camera.cv2, "waitKey", lambda _: ord("q"))
    monkeypatch.setattr(local_camera.cv2, "destroyAllWindows", lambda: None)
    svc = PerceptionService(MockModel(n_faces=1, seed=0))

    local_camera.run_local_camera(svc, 2)

    assert camera.released
    assert svc.phone_count() == 0


def test_local_camera_reports_unavailable_device(monkeypatch):
    class ClosedCamera:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    camera = ClosedCamera()
    monkeypatch.setattr(local_camera.cv2, "VideoCapture", lambda index, backend: camera)

    with pytest.raises(RuntimeError, match="Could not open camera index 3"):
        local_camera.run_local_camera(PerceptionService(MockModel()), 3)
    assert camera.released


def test_local_camera_explains_missing_opencv_gui(monkeypatch):
    class FakeCamera:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((240, 320, 3), np.uint8)

        def release(self):
            self.released = True

    camera = FakeCamera()
    service = PerceptionService(MockModel(n_faces=1, seed=0))
    monkeypatch.setattr(local_camera.cv2, "VideoCapture", lambda index, backend: camera)

    def fail_to_show(*_):
        raise local_camera.cv2.error("The function is not implemented")

    monkeypatch.setattr(local_camera.cv2, "imshow", fail_to_show)
    destroy_called = False

    def destroy_windows():
        nonlocal destroy_called
        destroy_called = True

    monkeypatch.setattr(local_camera.cv2, "destroyAllWindows", destroy_windows)

    with pytest.raises(RuntimeError, match="installed without GUI support"):
        local_camera.run_local_camera(service, 0)

    assert camera.released
    assert service.phone_count() == 0
    assert not destroy_called
