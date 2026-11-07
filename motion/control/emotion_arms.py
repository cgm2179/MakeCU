"""Apply the listener's emotion to the arms. Does not import music or perception."""
from __future__ import annotations

from motion.control.arm_params import ArmProfile, motion_for_emotion
from motion.drivers.arm_link import ArmLink


class EmotionArms:
    def __init__(self, link: ArmLink | None = None):
        self.link = link or ArmLink()
        self.history: list[tuple[str, ArmProfile]] = []

    def apply(self, emotion: str) -> ArmProfile:
        profile = motion_for_emotion(emotion)
        self.link.send(profile, emotion)
        self.history.append((emotion, profile))
        return profile
