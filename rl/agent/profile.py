"""Save / load a user's learned profile (checkpoint P0.4).

A profile is the bandit's weights plus the bookkeeping that affects its choices, written to
data/profiles/<user>.json (git-ignored: it describes a person's taste). It records which base
model it sits on; loading against a different base is refused, because the weights only make
sense on top of the model they were learned with. Today there is no base model, so the id is
BASE_NONE (the agent starts from zero).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from rl.agent.bandit import BanditAgent

PROFILE_VERSION = 1
BASE_NONE = "none-v0"
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "profiles"
_USER_OK = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ProfileError(Exception):
    pass


def profile_path(user: str, directory=None) -> Path:
    if not _USER_OK.match(user or ""):
        raise ProfileError(f"bad user name {user!r}: use 1-40 letters, digits, '_' or '-'")
    return Path(directory or DEFAULT_DIR) / f"{user}.json"


def save_profile(agent: BanditAgent, user: str, base_model_id: str = BASE_NONE, directory=None) -> Path:
    path = profile_path(user, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": PROFILE_VERSION,
        "user": user,
        "base_model_id": base_model_id,
        "saved_at": time.time(),
        "params": {"alpha": agent.alpha, "epsilon": agent.epsilon, "epsilon_min": agent.epsilon_min,
                   "decay": agent.decay, "novelty_bonus": agent.novelty_bonus,
                   "mainstream_bias": agent.mainstream_bias},
        "n_updates": agent.n_updates,
        "play_counts": dict(agent.play_counts),
        "weights": [[scope, feat, w] for (scope, feat), w in sorted(agent.weights.items())],
    }
    tmp = path.with_suffix(".json.tmp")  # write then rename, so a crash never leaves half a profile
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)
    return path


def load_profile(user: str, base_model_id: str = BASE_NONE, directory=None, seed=None) -> BanditAgent:
    path = profile_path(user, directory)
    if not path.exists():
        raise ProfileError(f"no saved profile for '{user}' at {path}")
    try:
        data = json.loads(path.read_text())
        version, saved_base = data["version"], data["base_model_id"]
    except (ValueError, KeyError, TypeError) as e:
        raise ProfileError(f"profile {path} is unreadable ({e!r}); delete it to start fresh") from e
    if version != PROFILE_VERSION:
        raise ProfileError(f"profile {path} has version {version}, this code reads version {PROFILE_VERSION}")
    if saved_base != base_model_id:
        raise ProfileError(f"profile for '{user}' was learned on base model '{saved_base}', "
                           f"but the current base model is '{base_model_id}'. Refusing to load it.")
    try:
        p = data["params"]
        agent = BanditAgent(epsilon=p["epsilon"], epsilon_min=p["epsilon_min"], decay=p["decay"],
                            alpha=p["alpha"], novelty_bonus=p["novelty_bonus"],
                            mainstream_bias=p["mainstream_bias"], seed=seed)
        agent.weights = {(scope, feat): float(w) for scope, feat, w in data["weights"]}
        agent.play_counts.update({k: int(v) for k, v in data["play_counts"].items()})
        agent.n_updates = int(data["n_updates"])
    except (KeyError, TypeError, ValueError) as e:
        raise ProfileError(f"profile {path} is damaged ({e!r}); delete it to start fresh") from e
    return agent


def reset_profile(user: str, directory=None) -> bool:
    """Forget a user: delete the saved profile. Returns True if one existed."""
    path = profile_path(user, directory)
    if path.exists():
        path.unlink()
        return True
    return False
