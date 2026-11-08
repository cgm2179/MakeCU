"""Turn a live camera GroupReaction into the message the RL agent already learns from.

The camera pipeline (perception.perception_cv) and the bandit (shared.schemas) each
have their own GroupReaction type. This is the only place that converts one into
the other, so the agent can discourage a song and its genre from a real face.
"""
from __future__ import annotations

import threading
import time

from shared.schemas.messages import GroupReaction, PersonReaction


def to_shared(live, song_id: str | None = None) -> GroupReaction:
    people = tuple(
        PersonReaction(int(p.person_id), p.emotion, float(p.liking), float(p.confidence))
        for p in live.people
    )
    return GroupReaction(
        timestamp=float(live.timestamp),
        song_id=song_id or live.song_id,
        people=people,
        group_liking=float(live.group_liking),
        group_emotion=dict(live.group_emotion),
    )


class LiveCamera:
    """Read whatever faces the camera has seen over the last sample_s seconds."""

    def __init__(self, service, sample_s: float = 2.0):
        self.service = service
        self.sample_s = sample_s

    def __call__(self, song_id: str) -> GroupReaction:
        if self.sample_s > 0:
            time.sleep(self.sample_s)
        return to_shared(self.service.get_group_reaction(song_id), song_id)


def start_camera(camera_index: int = 2, detector: str = "yunet"):
    """Open the local camera (DroidCam) and keep feeding the perception service.

    Returns the service. The preview window runs on a background thread; press q
    in that window to stop it. DeepFace is imported only here, not at test time.
    """
    from perception.perception_cv.api import build_service
    from perception.perception_cv.capture.local_camera import run_local_camera

    service = build_service(detector=detector)
    threading.Thread(
        target=run_local_camera, args=(service, camera_index), daemon=True,
    ).start()
    return service
