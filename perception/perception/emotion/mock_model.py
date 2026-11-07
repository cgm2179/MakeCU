from __future__ import annotations
import math, time
import numpy as np
from .base import Face, RAW_EMOTIONS


class MockModel:
    """Fake faces with random emotions. Lets you test the phone page and pipeline with no ML installed."""

    def __init__(self, n_faces: int = 2, seed: int | None = None):
        self.n = n_faces
        self.rng = np.random.default_rng(seed)

    def predict(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        t = time.time()
        faces = []
        for i in range(self.n):
            cx = w * (0.3 + 0.4 * i / max(self.n - 1, 1)) + 10 * math.sin(t + i)
            fw = w * 0.18
            p = self.rng.dirichlet(np.ones(len(RAW_EMOTIONS)) * 0.5)
            faces.append(Face((int(cx - fw / 2), int(h * 0.4), int(fw), int(fw)),
                              dict(zip(RAW_EMOTIONS, p.tolist())), 0.9))
        return faces
