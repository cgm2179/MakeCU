"""Contextual bandit over song features (Person B can swap in their own agent).

Context : the first filter (e.g. 'event:party') + the user's current emotion.
Action  : pick a song.
Reward  : the like/dislike signal (group_liking * mean confidence), in [-1, 1].

A song is described by features: genre, artist, energy bucket and mood. The agent
learns a weight per feature, at three levels so it can both generalise and specialise:

    global              : what this user likes in general
    filter:<filter>     : what they like for this first filter
    ctx:<filter>|<emotion> : what they like for this filter AND this emotion

A song's predicted reward is the sum of its feature weights over all three levels.
Because the weights belong to features, not songs, the agent can score songs it has
never played (e.g. a new jazz song after liking other jazz songs).
"""
from __future__ import annotations

import random
from collections import Counter

from rl.agent.context import Context
from shared.schemas.messages import GroupReaction, Song

# A phone override nudges a song's predicted score by this much (weaker than an
# explicit like/dislike, which pulls the score toward +/-1).
OVERRIDE_CHOSEN_NUDGE = +0.3  # user picked this song
OVERRIDE_REPLACED_NUDGE = -0.15  # user moved away from this one

# Below this mean confidence a reaction is treated as "nobody was really seen" and the
# learning step is skipped. (A reward of 0 is NOT neutral to the learner: it would drag the
# song's predicted score toward 0, wiping out what was learned.)
MIN_CONFIDENCE = 0.1


def mean_confidence(reaction: GroupReaction) -> float:
    conf = [p.confidence for p in reaction.people]
    return sum(conf) / len(conf) if conf else 0.0


# Audio features are bucketed into labelled ranges so the bandit can learn from them.
# (name, (cut1, cut2), (label1, label2, label3)); a feature is used only if the song has it.
# Kept to four on purpose: tested with up to ten buckets, the extra ones (acousticness,
# instrumentalness) mostly dilute the genre/artist signal and slowed learning.
BUCKETS = [
    ("energy", (0.4, 0.7), ("low", "mid", "high")),
    ("valence", (0.4, 0.7), ("low", "mid", "high")),
    ("danceability", (0.4, 0.7), ("low", "mid", "high")),
    ("tempo", (100.0, 125.0), ("slow", "mid", "fast")),
]


def energy_bucket(energy: float) -> str:
    return "low" if energy < 0.4 else "mid" if energy < 0.7 else "high"


def features(song: Song) -> list[str]:
    out = [f"genre:{song.genre}", f"artist:{song.artist}", f"mood:{song.mood or 'unknown'}"]
    for name, (c1, c2), (l1, l2, l3) in BUCKETS:
        if name in song.audio_features:
            v = song.audio_features[name]
            out.append(f"{name}:{l1 if v < c1 else l2 if v < c2 else l3}")
    return out


class BanditAgent:
    def __init__(self, epsilon=0.3, epsilon_min=0.05, decay=0.95, alpha=0.3,
                 novelty_bonus=0.3, seed=None):
        self.epsilon, self.epsilon_min, self.decay = epsilon, epsilon_min, decay
        self.alpha = alpha  # fraction of the prediction error closed per update
        self.novelty_bonus = novelty_bonus  # small push toward songs not yet heard
        self.weights: dict[tuple[str, str], float] = {}
        self.play_counts: Counter = Counter()  # songs that have received a reaction
        self.rng = random.Random(seed)

    # ---- scoring ----
    @staticmethod
    def scopes(ctx: Context) -> list[str]:
        return ["global", f"filter:{ctx.filter_key}", f"ctx:{ctx.filter_key}|{ctx.emotion}"]

    def _terms(self, song: Song, ctx: Context) -> list[tuple[str, str]]:
        return [(sc, f) for sc in self.scopes(ctx) for f in features(song)]

    def predict(self, song: Song, context: Context | None = None) -> float:
        ctx = context or Context()
        return sum(self.weights.get(t, 0.0) for t in self._terms(song, ctx))

    def score(self, song: Song, context: Context | None = None) -> float:
        """Prediction plus a small bonus for songs never played."""
        bonus = self.novelty_bonus if self.play_counts[song.song_id] == 0 else 0.0
        return self.predict(song, context) + bonus

    def recommend(self, pool: list[Song], context: Context | None = None, k: int = 3,
                  unplayed_only: bool = False) -> list[tuple[Song, float]]:
        """Best songs by predicted reward. unplayed_only=True shows what it would
        recommend among songs it has never played."""
        songs = [s for s in pool if not unplayed_only or self.play_counts[s.song_id] == 0]
        ranked = sorted(((s, self.predict(s, context)) for s in songs), key=lambda x: -x[1])
        return ranked[:k]

    # ---- policy ----
    def choose_next_song(self, candidate_pool: list[Song], last_song_id=None,
                         context: Context | None = None,
                         recent_ids=()) -> tuple[Song, str]:
        """Epsilon-greedy. Returns (song, 'explore' | 'exploit').

        Songs in recent_ids (and the last song) are skipped so it does not loop on
        the same two tracks; unplayed songs get a small novelty bonus."""
        skip = set(recent_ids) | {last_song_id}
        pool = [s for s in candidate_pool if s.song_id not in skip] or candidate_pool
        eps = self.epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)
        if self.rng.random() < eps:
            return self.rng.choice(pool), "explore"
        best = max(self.score(s, context) for s in pool)
        return self.rng.choice([s for s in pool if self.score(s, context) == best]), "exploit"

    # ---- learning ----
    @staticmethod
    def reward(reaction: GroupReaction) -> float:
        conf = [p.confidence for p in reaction.people] or [0.0]
        return reaction.group_liking * (sum(conf) / len(conf))

    def update(self, song: Song, reaction: GroupReaction,
               context: Context | None = None) -> dict:
        """Learn from an explicit like/dislike. Returns details for printing."""
        ctx = context or Context()
        r = self.reward(reaction)
        terms = self._terms(song, ctx)
        predicted = sum(self.weights.get(t, 0.0) for t in terms)
        confidence = mean_confidence(reaction)
        if confidence < MIN_CONFIDENCE:  # nobody in view: learn nothing, count nothing
            return {"reward": 0.0, "predicted": predicted, "error": 0.0, "skipped": True,
                    "confidence": confidence}
        error = r - predicted
        for t in terms:
            self.weights[t] = self.weights.get(t, 0.0) + self.alpha * error / len(terms)
        self.play_counts[song.song_id] += 1
        return {"reward": r, "predicted": predicted, "error": error, "skipped": False,
                "confidence": confidence}

    def record_manual_override(self, chosen: Song, replaced: Song | None,
                               context: Context | None = None) -> dict:
        """The user changed the song from their phone. Record it as a (weaker) signal:
        the song they picked is liked a bit, the one they moved away from disliked a bit."""
        ctx = context or Context()
        before = self.predict(chosen, ctx)
        self._nudge(chosen, ctx, OVERRIDE_CHOSEN_NUDGE)
        if replaced is not None and replaced.song_id != chosen.song_id:
            self._nudge(replaced, ctx, OVERRIDE_REPLACED_NUDGE)
        return {"chosen_before": before, "chosen_after": self.predict(chosen, ctx)}

    def _nudge(self, song: Song, ctx: Context, amount: float) -> None:
        terms = self._terms(song, ctx)
        for t in terms:
            self.weights[t] = self.weights.get(t, 0.0) + amount / len(terms)

    # ---- inspection ----
    def top_features(self, scope: str, k: int = 3) -> tuple[list, list]:
        items = sorted(((f, w) for (sc, f), w in self.weights.items() if sc == scope),
                       key=lambda x: -x[1])
        best = [x for x in items if x[1] > 0.005][:k]
        worst = [x for x in reversed(items) if x[1] < -0.005][:k]
        return best, worst
