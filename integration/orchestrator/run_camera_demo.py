"""Music loop fed by the camera package's reactions instead of a fake listener.

Run from the repo root with a Python that has the camera deps (opencv-python, fastapi, uvicorn):

    python3 -m integration.orchestrator.run_camera_demo --mood gym --songs 4              # plain mock
    python3 -m integration.orchestrator.run_camera_demo --mood gym --songs 24 --taste     # adaptation
    python3 -m integration.orchestrator.run_camera_demo --mood gym --songs 24 --taste --empty 4,9
    python3 -m integration.orchestrator.run_camera_demo --songs 3 --audio                 # + real previews

--real           : your actual webcam and the DeepFace model (needs camera permission, internet on first run).
--mock (default): perception_cv.build_service(mock=True), random faces that do not depend on the song.
--taste         : mock faces that react to the song (hidden taste: likes happy-sounding songs,
                  valence >= 0.7), still going through the real tracker + scoring pipeline.
"""
from __future__ import annotations

import argparse
import statistics as st

from integration.orchestrator.run_loop import run
from music.catalog.songs import get_candidates
from music.catalog.spotify_catalog import dataset_available
from perception.emotion.camera_reaction import CameraFeedback
from shared.schemas.messages import Filter, FilterSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mood", default="gym")
    ap.add_argument("--songs", type=int, default=4, help="rounds to run")
    ap.add_argument("--listen-seconds", type=float, default=1.0, help="how long the camera watches per song (no audio)")
    ap.add_argument("--taste", action="store_true", help="faces react to the song (to see the bandit adapt)")
    ap.add_argument("--empty", default="", help="comma-separated rounds with nobody in view, e.g. 4,9 (needs --taste)")
    ap.add_argument("--real", action="store_true", help="REAL webcam + DeepFace instead of mock (no fallback to mock)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index to try first with --real (then 1, 2, 3)")
    ap.add_argument("--detector", default="opencv", help="DeepFace detector with --real (their default is opencv; yunet is more reliable)")
    ap.add_argument("--audio", action="store_true", help="play iTunes previews; the reaction is read when the clip ends")
    ap.add_argument("--clip-seconds", type=float, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if not dataset_available():
        raise SystemExit("Dataset missing. Run: python3 -m music.catalog.download_spotify")

    pool = get_candidates(Filter((FilterSource("event", a.mood),)))
    empty = {int(x) for x in a.empty.split(",") if x}
    if empty and not a.taste:
        raise SystemExit("--empty needs --taste (the plain mock model cannot remove faces)")
    liked = lambda song: song.audio_features["valence"] >= 0.7
    if a.real:
        feedback = CameraFeedback(mock=False, camera_index=a.camera, detector=a.detector)
        print(f"REAL camera (index {a.camera} first) + DeepFace, detector '{a.detector}'. No mock fallback.")
    elif a.taste:
        from perception.emotion.taste_mock_model import TasteMockModel

        feedback = CameraFeedback(model=TasteMockModel(liked, seed=a.seed), empty_rounds=empty)
        base = sum(map(liked, pool)) / len(pool)
        print(f"Mock cameras: 2 people whose faces react to the song (hidden taste: valence >= 0.7; "
              f"{base:.0%} of the {a.mood} pool qualifies)")
    else:
        feedback = CameraFeedback(mock=True)
        print("Mock cameras: perception_cv.build_service(mock=True) (random faces, not tied to the song)")

    player = None
    if a.audio:
        from music.playback.audio import AudioPlayer

        player = AudioPlayer(clip_seconds=a.clip_seconds)
    try:
        out = run(a.songs, event=a.mood, seed=a.seed, pool=pool, feedback=feedback, player=player,
                  wait_for_playback=bool(player), listen_seconds=a.listen_seconds, round_delay=0,
                  overrides={}, verbose=not a.quiet)
    finally:
        if player:
            player.close()

    log = out["log"]
    print(f"\nRounds: {len(log)}, learned from {len(out['rewards'])}, skipped (nobody in view): {out['skipped_rounds']}")
    if a.taste and len(log) >= 12:
        hit = lambda es: st.mean(liked(e["song"]) for e in es if not e["skipped"])
        print(f"Songs the faces liked: first 6 rounds {hit(log[:6]):.0%} -> last 6 rounds {hit(log[-6:]):.0%} "
              f"(random would be about {sum(map(liked, pool)) / len(pool):.0%})")
    if feedback.feeder_error:
        print("WARNING: frame feeder error:", feedback.feeder_error)


if __name__ == "__main__":
    main()
