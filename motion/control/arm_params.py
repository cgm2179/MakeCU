"""Arm oscillation, chosen from the listener's emotion.

Both arms follow a sine around the center pose:

    left  = amplitude * sin(2 * pi * t / period_s)
    right = amplitude * sin(2 * pi * t / period_s + radians(phase_offset_deg))

phase_offset_deg
    0    both arms move together ("at the same time")
    180  arms move opposite each other (out of phase)

amplitude is a fraction of full travel, from 0.0 to 1.0.
period_s is seconds for one full cycle. Smaller means faster.

Edit the numbers in this file. Callers only look up a profile by emotion name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmProfile:
    period_s: float
    amplitude: float
    phase_offset_deg: float


# Happy: out of phase, fairly fast, large swing.
HAPPY = ArmProfile(period_s=0.55, amplitude=0.85, phase_offset_deg=180.0)

# Sad: in phase, much slower, small swing.
SAD = ArmProfile(period_s=2.80, amplitude=0.22, phase_offset_deg=0.0)

# Used when the camera sees nobody, or an emotion with no profile of its own.
NEUTRAL = ArmProfile(period_s=1.40, amplitude=0.40, phase_offset_deg=0.0)

# Starting points for the other emotions. Happy and sad are the ones specified.
SURPRISED = ArmProfile(period_s=0.70, amplitude=0.70, phase_offset_deg=180.0)
ANGRY = ArmProfile(period_s=0.90, amplitude=0.50, phase_offset_deg=180.0)
BORED = ArmProfile(period_s=3.20, amplitude=0.15, phase_offset_deg=0.0)

PROFILES: dict[str, ArmProfile] = {
    "happy": HAPPY,
    "surprised": SURPRISED,
    "sad": SAD,
    "angry": ANGRY,
    "bored": BORED,
    "neutral": NEUTRAL,
}


def motion_for_emotion(emotion: str) -> ArmProfile:
    """Profile for this emotion. Unknown names fall back to NEUTRAL."""
    return PROFILES.get(emotion, NEUTRAL)
