from __future__ import annotations
import numpy as np
from .base import Face


class DeepFaceModel:
    """Multi-face emotion detection with DeepFace.

    detector_backend: "opencv" (fast, weak at odd angles), "ssd", "mediapipe", "yunet", "retinaface" (slow, best).
    Try "yunet" or "mediapipe" if faces are missed.
    """

    def __init__(self, detector_backend: str = "opencv", min_face_px: int = 40):
        from deepface import DeepFace  # heavy import, keep lazy
        self._df = DeepFace
        self.detector_backend = detector_backend
        self.min_face_px = min_face_px
        try:  # preload so the first frame is not slow; fine if it fails, it loads lazily
            try:
                DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            except TypeError:
                DeepFace.build_model("Emotion")
        except Exception:
            pass

    def predict(self, frame_bgr: np.ndarray) -> list[Face]:
        try:
            results = self._df.analyze(
                img_path=frame_bgr,
                actions=["emotion"],
                detector_backend=self.detector_backend,
                enforce_detection=True,   # raises ValueError when no face is found
                silent=True,
            )
        except ValueError:
            return []
        if isinstance(results, dict):
            results = [results]
        faces: list[Face] = []
        for r in results:
            reg = r["region"]
            if min(reg["w"], reg["h"]) < self.min_face_px:
                continue
            probs = {k: float(v) for k, v in r["emotion"].items()}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            fc = r.get("face_confidence") or 0.8   # some detector backends report 0
            faces.append(Face((reg["x"], reg["y"], reg["w"], reg["h"]), probs, min(float(fc), 1.0)))
        return faces
