"""New-user check (checkpoint L3.3, docs/THREE_LAYER_CHECKPOINTS.md): does the frozen base model
(trained only on Last.fm listeners, never on this person) predict a brand-new person's taste better
than just playing popular songs?

Reuses the Step 3.2 live-session machinery (integration/orchestrator/run_live_session.py, rl/live_test.py)
but the "personal" condition is driven by the frozen base model's preference head instead of a
bandit trained on the person's own ratings, since there are none yet. No rating session or saved
profile is needed first. Works for any new user, not just the person who first ran this.

    python3 -m integration.orchestrator.run_new_user_check --user <your_name>

Each song starts playing at once and asks for a thumb straight away; answering stops it and starts
the next one. u = liked, d = disliked, s = skip this song, q = quit (progress is saved; rerun to resume).
Saved to data/live/<user>.csv, the same file and format run_live_session.py uses. Once there are
MIN_THUMBS_FOR_ANALYSIS (50) explicit thumbs with some liked and some disliked, this script runs the
Step 7 fine-tuned/base-only/popularity/from-scratch comparison automatically (same as running
`python3 -m rl.personal_layer --user <your_name>` by hand).
"""
from __future__ import annotations

import argparse

from integration.orchestrator.run_live_session import build_pool, run_session
from rl import live_test as lt
from rl.base_lastfm import MODEL_DIR, load_base

MIN_THUMBS_FOR_ANALYSIS = 50   # matches the minimum rl/base_model.py requires before training on thumbs


class BaseModelAgent:
    """Wraps the frozen base model so run_session's "personal" slot ranks songs by it, not a bandit."""

    def __init__(self, base):
        self.base = base

    def predict(self, song, context=None):
        return self.base.preference_logit(song)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your name; used as the label for data/live/<user>.csv")
    ap.add_argument("--base", default=str(MODEL_DIR / "base_model_v2.json"))
    ap.add_argument("--per-condition", type=int, default=15,
                     help="songs per condition this run (default 15; the pre-registered gate wants 40+, rerun to add more)")
    ap.add_argument("--moods", default="gym,party,chill,happy,sad,study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)

    from music.catalog.spotify_catalog import dataset_available
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")
    base = load_base(a.base)
    agent = BaseModelAgent(base)
    pool = build_pool([m.strip() for m in a.moods.split(",") if m.strip()], rated_ids=set())
    print(f"{len(pool)} candidate songs. Conditions are hidden ('personal' here = the frozen base model's prediction).")
    player = None
    if not a.no_audio:
        from music.playback.audio import AudioPlayer
        player = AudioPlayer(clip_seconds=30)
    path = lt.live_path(a.user)
    try:
        n = run_session(pool, agent, a.user, path, player, per_condition=a.per_condition, seed=a.seed)
    finally:
        if player:
            player.close()
    rows = lt.load_rows(path, a.user)
    print(f"\nRated {n} songs this run; {len(rows)} saved in {path}\n")
    if not rows:
        return 0
    print(lt.format_text(lt.analyse(rows)))

    from music.catalog.spotify_catalog import load_spotify
    from rl.base_model import load_thumb_items
    by_id = {s.song_id: s for s in load_spotify()}
    items = load_thumb_items(a.user, by_id)
    if len(items) >= MIN_THUMBS_FOR_ANALYSIS and any(y for _, y in items) and any(not y for _, y in items):
        print(f"\n{len(items)} explicit thumbs for '{a.user}' — running the Step 7 fine-tuning comparison:\n")
        from rl.personal_layer import main as personal_layer_main
        personal_layer_main(["--user", a.user, "--base", a.base])
    else:
        print(f"\nOnce there are {MIN_THUMBS_FOR_ANALYSIS}+ thumbs with some liked and some disliked "
              f"(currently {len(items)}):\n  python3 -m rl.personal_layer --user {a.user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
