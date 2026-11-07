from __future__ import annotations
import itertools, threading, time
from dataclasses import dataclass, field
import cv2
import numpy as np

from .config import PerceptionConfig
from .emotion.base import EmotionModel
from .group.mapping import to_base_probs
from .group.tracker import CentroidTracker, Detection
from .group.scoring import person_reaction, aggregate
from .types import GroupReaction, PersonReaction


@dataclass
class PhoneState:
    index: int
    tracker: CentroidTracker
    people: list[PersonReaction] = field(default_factory=list)
    last_update: float = 0.0


class PerceptionService:
    """Frames in from any number of phones, GroupReaction out."""

    def __init__(self, model: EmotionModel, cfg: PerceptionConfig | None = None):
        self.model = model
        self.cfg = cfg or PerceptionConfig()
        self._lock = threading.Lock()
        self._phones: dict[str, PhoneState] = {}
        self._phone_counter = itertools.count(1)

    # ---- phone side -------------------------------------------------
    def process_frame(self, phone_id: str, jpeg: bytes) -> dict:
        t0 = time.time()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"faces": [], "error": "bad frame"}
        h, w = frame.shape[:2]
        faces = self.model.predict(frame)   # slow part, outside the lock
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        faces = faces[: self.cfg.MAX_PEOPLE]
        now = time.time()
        dets = [Detection(((f.box[0] + f.box[2] / 2) / w, (f.box[1] + f.box[3] / 2) / h),
                          to_base_probs(f.probs), f.face_confidence) for f in faces]
        reply_faces = []
        with self._lock:
            ph = self._phones.get(phone_id)
            if ph is None:
                ph = PhoneState(next(self._phone_counter), CentroidTracker(self.cfg))
                self._phones[phone_id] = ph
            tracks = ph.tracker.update(dets, now)
            people = []
            for f, t in zip(faces, tracks):
                pr = person_reaction(t, ph.index * 1000 + t.id, now, self.cfg)
                people.append(pr)
                reply_faces.append({
                    "box": [f.box[0] / w, f.box[1] / h, f.box[2] / w, f.box[3] / h],
                    "person_id": pr.person_id, "emotion": pr.emotion,
                    "liking": round(pr.liking, 3), "confidence": round(pr.confidence, 3),
                })
            ph.people = people
            ph.last_update = now
        return {"faces": reply_faces, "ms": int((time.time() - t0) * 1000)}

    def drop_phone(self, phone_id: str) -> None:
        with self._lock:
            self._phones.pop(phone_id, None)

    def phone_count(self) -> int:
        with self._lock:
            return len(self._phones)

    # ---- consumer side (AGENTS.md entry point) ----------------------
    def get_group_reaction(self, song_id: str) -> GroupReaction:
        now = time.time()
        with self._lock:
            people = [p for ph in self._phones.values()
                      if now - ph.last_update <= self.cfg.STALE_SECONDS for p in ph.people]
        return aggregate(song_id, people, now, self.cfg.MAX_PEOPLE)

    def start_publishing(self, get_song_id, period_s: float = 1.0, publish=None) -> threading.Thread:
        """Push GroupReaction on 'perception.group_reaction' every period_s.
        get_song_id: callable returning the current song_id (wire it to the bus once it exists).
        publish: callable(topic, dict); defaults to shared.bus.publish if importable, else prints."""
        if publish is None:
            try:
                from shared.bus import publish  # type: ignore
            except Exception:
                publish = lambda topic, msg: print(topic, msg)

        def loop():
            while True:
                publish("perception.group_reaction", self.get_group_reaction(get_song_id()).to_dict())
                time.sleep(period_s)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        return th
