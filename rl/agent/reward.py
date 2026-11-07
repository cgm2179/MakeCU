"""One reward from every signal available: thumbs, behaviour (skips / manual changes) and the camera.

    reward = sum(weight_i * signal_i) / max(sum(weight_i), 1)        signals in [-1, +1]

  thumb      weight 1.0   explicit up/down from the person (most reliable)
  behaviour  weight 0.7   they moved to another song (-1) or picked this one themselves (+1)
  camera     weight w     w = measured reliability of THIS person's camera (rl.agent.camera_model), 0 to 1

Dividing by max(total, 1) means a weak signal on its own gives a small reward instead of a full-size one:
a camera with weight 0.2 and nothing else yields at most +/-0.2, while a thumb alone yields +/-1.
"""
from __future__ import annotations

from shared.schemas.messages import GroupReaction, PersonReaction

THUMB_WEIGHT = 1.0
BEHAVIOUR_WEIGHT = 0.7
NEGATIVE_BEHAVIOUR = {"replaced", "skipped"}
POSITIVE_BEHAVIOUR = {"chosen", "replayed"}


def _sign(v):
    if v in (True, 1):
        return 1.0
    if v in (False, 0, -1):
        return -1.0
    raise ValueError(f"thumb must be 1/0/True/False/-1, got {v!r}")


def combine(thumb=None, behaviour=None, camera=None, camera_weight=0.0):
    """Returns (reward, parts) where parts lists the (name, weight, signal) that were used.
    reward is None when no signal at all was available."""
    parts = []
    if thumb is not None:
        parts.append(("thumb", THUMB_WEIGHT, _sign(thumb)))
    if behaviour in NEGATIVE_BEHAVIOUR:
        parts.append(("behaviour", BEHAVIOUR_WEIGHT, -1.0))
    elif behaviour in POSITIVE_BEHAVIOUR:
        parts.append(("behaviour", BEHAVIOUR_WEIGHT, 1.0))
    if camera is not None and camera_weight > 0:
        parts.append(("camera", camera_weight, max(-1.0, min(1.0, camera))))
    if not parts:
        return None, parts
    total = sum(w for _, w, _ in parts)
    return sum(w * s for _, w, s in parts) / max(total, 1.0), parts


class CameraRewardModel:
    """Plug into run(..., reward_model=...). Replaces the raw camera reading by the combined reward.

    fit(rows) learns this person's camera model from earlier thumbs and measures its reliability; until
    then (or with too little data) the camera has weight 0 and only thumbs / behaviour teach the agent."""

    def __init__(self, model=None, weight=0.0):
        self.model, self.weight = model, weight
        self.last = None   # (reward, parts) of the latest call, for printing

    def fit(self, rows, **kw):
        from rl.agent.camera_model import CameraReliability, reliability
        self.weight, self.auc, self.n = reliability(rows, **kw)
        self.model = CameraReliability(**kw).fit(rows) if self.weight > 0 else None
        return self

    def __call__(self, song, reaction, feedback=None, thumb=None, behaviour=None):
        cam = None
        if self.model is not None and feedback is not None and hasattr(feedback, "clip_features"):
            feats = feedback.clip_features()
            cam = self.model.soft_reward(feats) if feats else None
        reward, parts = combine(thumb, behaviour, cam, self.weight)
        self.last = (reward, parts)
        if reward is None:   # nothing usable: confidence 0 makes the bandit skip this round instead of learning 0
            return GroupReaction(reaction.timestamp, reaction.song_id, (), 0.0, reaction.group_emotion)
        emotion = "happy" if reward > 0 else "sad"
        return GroupReaction(reaction.timestamp, reaction.song_id,
                             (PersonReaction(0, emotion, reward, 1.0),), reward, reaction.group_emotion)
