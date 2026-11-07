"""End-to-end loop with fake inputs: contextual bandit + playlist control + manual override.

Run from the repo root:
    python3 -m integration.orchestrator.run_loop                      # you give l / d feedback
    python3 -m integration.orchestrator.run_loop --simulate 25        # simulated listener + phone
    python3 -m integration.orchestrator.run_loop --event study --simulate 25
"""
from __future__ import annotations

import argparse
import time

from app.api.fake_phone import FakePhone
from motion.control.stub_motion import MotionStub
from music.catalog.songs import get_candidates
from music.playback.playlist import PlaylistController
from perception.emotion.fake_reaction import get_group_reaction
from rl.agent.bandit import BanditAgent
from rl.agent.context import Context, dominant_emotion, filter_key
from rl.env.simulated_listener import EMOTIONS, SimulatedListener
from shared.bus.bus import Bus
from shared.schemas.messages import Filter, FilterSource

# In --simulate mode the fake phone changes the song mid-way at these steps.
DEFAULT_OVERRIDES = {6: "s6", 14: "s11"}


def _fmt(items):
    return ", ".join(f"{f} {w:+.2f}" for f, w in items) or "-"


def print_learning(agent: BanditAgent, ctx: Context) -> None:
    best, worst = agent.top_features("global")
    print(f"  learned (overall)     likes: {_fmt(best)}   dislikes: {_fmt(worst)}")
    best, worst = agent.top_features(agent.scopes(ctx)[2])
    print(f"  learned (next context) likes: {_fmt(best)}   dislikes: {_fmt(worst)}")


def run(simulate_rounds=None, event="party", seed=None, overrides=None, verbose=True,
        round_delay=0.02):
    """Run the loop. Returns a dict of what happened (used by the tests)."""
    say = print if verbose else (lambda *a, **k: None)
    bus = Bus()
    flt = Filter(sources=(FilterSource("event", event),))
    pool = get_candidates(flt)
    agent = BanditAgent(seed=seed)
    controller = PlaylistController(pool, bus)
    phone = FakePhone(bus)
    listener = SimulatedListener(seed=seed or 0) if simulate_rounds else None
    if overrides is None:
        overrides = DEFAULT_OVERRIDES if simulate_rounds else {}
    motion = MotionStub()
    motion.start()  # robot movement runs on its own; music events never touch it

    emotion, step, last_id, forced = "neutral", 0, None, None
    log, rewards = [], []
    while simulate_rounds is None or step < simulate_rounds:
        step += 1
        ctx = Context(filter_key(flt), emotion)
        if forced:  # the user asked for this song from the phone
            song, mode, forced = forced, "phone", None
        else:
            song, mode = agent.choose_next_song(pool, last_id, ctx, recent_ids=controller.history[-4:])
        controller.play(song)
        say(f"\n[{step}] Now playing: {song.title} by {song.artist} "
            f"({song.genre}, {song.mood}, energy {song.audio_features['energy']}) "
            f"[{mode}, predicted {agent.predict(song, ctx):+.2f}]")
        say(f"  context: filter={ctx.filter_key}, emotion={ctx.emotion}")

        if listener and step in overrides:  # user taps another song on their phone, mid-song
            phone.change_song(overrides[step])
            say(f"  phone: user requested {overrides[step]} (current song keeps playing)")

        if listener:
            time.sleep(round_delay)  # a real song would take much longer
        reaction = listener.react(song) if listener else get_group_reaction(song.song_id, phone)
        if reaction is None:
            say("Bye!")
            break
        if listener:
            say(f"  simulated reaction: {'like' if reaction.group_liking > 0 else 'dislike'} "
                f"(listener emotion: {dominant_emotion(reaction)})")

        info = agent.update(song, reaction, ctx)
        rewards.append(info["reward"])
        say(f"  reward={info['reward']:+.1f}  error={info['error']:+.2f}  -> "
            f"{'keep similar' if info['reward'] > 0 else 'switch'}")

        emotion = dominant_emotion(reaction)
        manual = controller.take_manual_change()  # song boundary: apply phone change
        if manual:
            agent.record_manual_override(manual, song, ctx)
            forced = manual
            say(f"  manual override recorded: +signal for {manual.title}, "
                f"-signal for {song.title}; {manual.title} plays next")
        log.append({"step": step, "song_id": song.song_id, "mode": mode, "emotion": ctx.emotion})
        if verbose:
            print_learning(agent, Context(ctx.filter_key, emotion))
        last_id = song.song_id

    motion.stop()
    say("\n=== Top unplayed songs the agent would recommend ===")
    for emo in EMOTIONS:
        recs = agent.recommend(pool, Context(filter_key(flt), emo), k=3, unplayed_only=True)
        say(f"  when {emo:<7}: " + "; ".join(f"{s.title} ({s.genre}, {p:+.2f})" for s, p in recs))
    say(f"\nMotion stub: {len(motion.tick_times)} ticks, {motion.restarts} restarts, "
        f"longest gap {motion.longest_gap() * 1000:.0f} ms (music events did not interrupt it)")
    return {"agent": agent, "controller": controller, "motion": motion, "log": log,
            "rewards": rewards, "pool": pool, "ctx_filter": filter_key(flt)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, metavar="N", help="run N rounds with a simulated listener")
    parser.add_argument("--event", default="party", help="first filter, e.g. party, study")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.simulate, args.event, args.seed)
