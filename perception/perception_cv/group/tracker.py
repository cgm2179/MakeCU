"""Tiny centroid tracker per phone: stable identities let us smooth emotions and measure stillness."""
from __future__ import annotations
import math
from dataclasses import dataclass
from ..config import PerceptionConfig


@dataclass
class Detection:
    center: tuple[float, float]      # normalized 0..1
    probs: dict[str, float]          # BASE_EMOTIONS
    face_conf: float


class Track:
    def __init__(self, track_id: int, det: Detection, now: float):
        self.id = track_id
        self.center = det.center
        self.probs = dict(det.probs)
        self.face_conf = det.face_conf
        self.frames = 1
        self.last_seen = now
        self.still_since = now

    def update(self, det: Detection, now: float, cfg: PerceptionConfig):
        a = cfg.EMA_ALPHA
        for k in self.probs:
            self.probs[k] = (1 - a) * self.probs[k] + a * det.probs[k]
        self.face_conf = (1 - a) * self.face_conf + a * det.face_conf
        if math.dist(self.center, det.center) > cfg.STILL_EPS:
            self.still_since = now
        self.center = det.center
        self.frames += 1
        self.last_seen = now


class CentroidTracker:
    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, dets: list[Detection], now: float) -> list[Track]:
        """Returns one Track per detection, in the same order."""
        c = self.cfg
        self.tracks = {i: t for i, t in self.tracks.items() if now - t.last_seen <= c.TRACK_MAX_AGE_S}
        pairs = sorted((math.dist(d.center, t.center), di, tid)
                       for di, d in enumerate(dets) for tid, t in self.tracks.items())
        assigned: dict[int, int] = {}
        used: set[int] = set()
        for dist, di, tid in pairs:
            if dist > c.TRACK_MATCH_DIST:
                break
            if di in assigned or tid in used:
                continue
            assigned[di] = tid
            used.add(tid)
        out = []
        for di, d in enumerate(dets):
            if di in assigned:
                t = self.tracks[assigned[di]]
                t.update(d, now, c)
            else:
                t = Track(self._next, d, now)
                self.tracks[self._next] = t
                self._next += 1
            out.append(t)
        return out
