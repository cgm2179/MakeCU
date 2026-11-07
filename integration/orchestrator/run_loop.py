"""Minimal end-to-end loop with fake inputs.

Run from the repo root:
    python3 -m integration.orchestrator.run_loop            # keyboard feedback
    python3 -m integration.orchestrator.run_loop --simulate 20   # no keyboard, 20 rounds
"""
from __future__ import annotations
import argparse

from music.catalog.songs import get_candidates
from perception.emotion.fake_reaction import get_group_reaction
from rl.agent.bandit import BanditAgent
from rl.env.simulated_listener import SimulatedListener


def print_weights(agent: BanditAgent) -> None:
    print("  Learned preferences:")
    for name, w in sorted(agent.weights.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(abs(w) * 10)
        print(f"    {name:<24}{w:+.2f} {'+' if w >= 0 else '-'}{bar}")


def run(simulate_rounds: int | None = None) -> None:
    agent = BanditAgent(epsilon=0.2, alpha=0.5)
    listener = SimulatedListener(("jazz",)) if simulate_rounds else None
    pool = get_candidates()
    last_id, step = None, 0

    while simulate_rounds is None or step < simulate_rounds:
        step += 1
        song, mode = agent.choose_next_song(pool, last_id)
        print(f"\n[{step}] Now playing: {song.title} by {song.artist} ({song.genre}) "
              f"[{mode}, predicted {agent.predict(song):+.2f}]")

        reaction = listener.react(song) if listener else get_group_reaction(song.song_id)
        if reaction is None:
            print("Bye!")
            break
        if listener:
            print(f"  Simulated reaction: {'like' if reaction.group_liking > 0 else 'dislike'}")

        info = agent.update(song, reaction)
        action = "keep similar" if info["reward"] > 0 else "switch"
        print(f"  reward={info['reward']:+.1f}  error={info['error']:+.2f}  -> {action}")
        print_weights(agent)
        last_id = song.song_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, metavar="N", help="run N rounds with a simulated listener")
    run(parser.parse_args().simulate)
