"""Map raw model classes onto the schema emotions in AGENTS.md."""
BASE_EMOTIONS = ("happy", "sad", "neutral", "angry", "surprised")  # "bored" is derived later

RAW_TO_SCHEMA = {
    "happy": "happy", "sad": "sad", "neutral": "neutral", "angry": "angry",
    "disgust": "angry",      # disgust at the song reads as dislike
    "fear": "surprised",
    "surprise": "surprised",
}


def to_base_probs(raw: dict[str, float]) -> dict[str, float]:
    out = {k: 0.0 for k in BASE_EMOTIONS}
    for k, v in raw.items():
        tgt = RAW_TO_SCHEMA.get(k)
        if tgt:
            out[tgt] += v
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}
