"""Mood / activity -> target audio-feature vector, used by the first filter.

Valence and energy are the main axes. Each profile entry is (target, weight).
Tempo targets are in BPM; everything else is on the 0-1 scale.
"""
from __future__ import annotations

from shared.schemas.messages import Song

PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "sad":   {"valence": (0.15, 1.0), "energy": (0.25, 1.0)},
    "happy": {"valence": (0.85, 1.0), "energy": (0.70, 1.0)},
    "gym":   {"energy": (0.90, 1.0), "tempo": (150, 1.0), "valence": (0.60, 0.3)},
    "study": {"energy": (0.20, 1.0), "instrumentalness": (0.85, 1.0), "valence": (0.40, 0.3)},
    "party": {"danceability": (0.85, 1.0), "energy": (0.85, 1.0), "valence": (0.75, 0.7)},
    "chill": {"energy": (0.30, 1.0), "valence": (0.50, 0.7), "acousticness": (0.60, 0.5)},
}
ALIASES = {
    "studying": "study", "focus": "study", "workout": "gym", "work out": "gym",
    "exercise": "gym", "running": "gym", "relax": "chill", "relaxing": "chill",
    "calm": "chill", "christmas": "happy",
}
TEMPO_MIN, TEMPO_RANGE = 60.0, 140.0  # tempo is rescaled to 0-1 over 60-200 BPM


def profile_for(name: str):
    """Return the profile for a mood/activity name, or None if unknown."""
    key = name.strip().lower()
    return PROFILES.get(ALIASES.get(key, key))


def _scaled(feature: str, value: float) -> float:
    if feature == "tempo":
        return min(1.0, max(0.0, (value - TEMPO_MIN) / TEMPO_RANGE))
    return value


def distance(song: Song, profile: dict) -> float:
    """Weighted RMS distance from a song to the target vector (0 = exact match)."""
    total = weight_sum = 0.0
    for feat, (target, w) in profile.items():
        diff = _scaled(feat, song.audio_features.get(feat, 0.5)) - _scaled(feat, target)
        total += w * diff * diff
        weight_sum += w
    return (total / weight_sum) ** 0.5


POOL_POPULARITY_WEIGHT = 0.4  # distance units a fully popular song (100) is worth, at mainstream_bias=1


def _popularity(song: Song) -> float:
    return song.audio_features.get("popularity", 0.0)


def select_pool(songs: list[Song], profile: dict, size: int = 500,
                min_popularity: float = 30.0, mainstream_bias: float = 0.0) -> list[Song]:
    """The `size` best songs for the target, best first.

    Songs are ranked by distance to the target minus a popularity bonus, so with a bias the
    well-known songs near the target come first (mainstream_bias 0 = pure distance, as before).
    Songs below `min_popularity` are used only as a fallback to top the pool up to `size`,
    so a niche mood (e.g. instrumental study music) never ends up with a short or empty pool."""
    weight = POOL_POPULARITY_WEIGHT * max(0.0, mainstream_bias)
    key = lambda s: distance(s, profile) - weight * _popularity(s) / 100
    known = sorted((s for s in songs if _popularity(s) >= min_popularity), key=key)
    if len(known) >= size:
        return known[:size]
    rest = sorted((s for s in songs if _popularity(s) < min_popularity), key=key)
    return known + rest[: size - len(known)]
