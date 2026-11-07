"""End-to-end loop with fake inputs: contextual bandit + playlist control + manual override.

Run from the repo root:
    python3 -m integration.orchestrator.run_loop                      # you give l / d feedback
    python3 -m integration.orchestrator.run_loop --simulate 25        # simulated listener + phone
    python3 -m integration.orchestrator.run_loop --event study --simulate 25
    python3 -m integration.orchestrator.run_loop --live --camera 2 --detector yunet
"""
from __future__ import annotations

import argparse
import random
import time

from app.api.fake_phone import FakePhone
from motion.control.emotion_arms import EmotionArms
from motion.control.stub_motion import MotionStub
from motion.drivers.arm_link import ArmLink
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
MAX_SKIPS = 15  # give up after this many unplayable songs in a row
# group_liking at or below this is unhappy: the update lowers the song and its genre.
UNHAPPY_LIKING = -0.2


def describe(song) -> str:
    """One-line description using whichever audio features the song has."""
    af = song.audio_features
    bits = [f"{k} {af[k]:.2f}" if k != "tempo" else f"tempo {af[k]:.0f}"
            for k in ("energy", "valence", "tempo") if k in af]
    return f"{song.title} by {song.artist} ({song.genre}, {song.mood}; {', '.join(bits)})"


def _fmt(items):
    return ", ".join(f"{f} {w:+.2f}" for f, w in items) or "-"


def print_learning(agent: BanditAgent, ctx: Context) -> None:
    best, worst = agent.top_features("global")
    print(f"  learned (overall)     likes: {_fmt(best)}   dislikes: {_fmt(worst)}")
    best, worst = agent.top_features(agent.scopes(ctx)[2])
    print(f"  learned (next context) likes: {_fmt(best)}   dislikes: {_fmt(worst)}")


def run(simulate_rounds=None, event="party", seed=None, overrides=None, verbose=True,
        round_delay=0.02, pool=None, listener=None, taste_shift=None, agent=None,
        player=None, wait_for_playback=False, react=None, arms=None):
    """Run the loop. Returns a dict of what happened (used by the tests).

    pool         : candidate songs; default = get_candidates(first filter)
    listener     : fake listener with .react(song); default = SimulatedListener
    taste_shift  : {step: genres} changes a TasteListener's taste at that step
    player       : audio output (AudioPlayer); songs with no playable audio are skipped
    wait_for_playback : wait for each clip to finish before taking the reaction
    react        : callable(song_id) -> GroupReaction from the live camera
    arms         : EmotionArms; default sends arm lines in memory (no serial)
    """
    say = print if verbose else (lambda *a, **k: None)
    bus = Bus()
    flt = Filter(sources=(FilterSource("event", event),))
    pool = pool if pool is not None else get_candidates(flt)
    agent = agent or BanditAgent(seed=seed)
    controller = PlaylistController(pool, bus, player)
    phone = FakePhone(bus)
    if listener is None and simulate_rounds:
        listener = SimulatedListener(seed=seed or 0)
    if overrides is None:
        if not simulate_rounds:
            overrides = {}
        elif all(i in controller.catalog for i in DEFAULT_OVERRIDES.values()):
            overrides = DEFAULT_OVERRIDES
        else:  # dataset pool: have the phone ask for two songs from the pool
            rng = random.Random(seed)
            overrides = {6: rng.choice(pool).song_id, 14: rng.choice(pool).song_id}
    motion = MotionStub()
    motion.start()  # robot movement runs on its own; music events never touch it
    if arms is None:
        arms = EmotionArms()

    emotion, step, last_id, forced = "neutral", 0, None, None
    unavailable: set = set()  # songs the audio output could not play
    log, rewards = [], []
    while simulate_rounds is None or step < simulate_rounds:
        step += 1
        ctx = Context(filter_key(flt), emotion)
        if forced:  # the user asked for this song from the phone
            song, mode, forced = forced, "phone", None
        else:
            playable = [x for x in pool if x.song_id not in unavailable] or pool
            song, mode = agent.choose_next_song(playable, last_id, ctx, recent_ids=controller.history[-4:])
        skipped_now = 0
        while not controller.play(song):  # no playable audio: skip to the next pick
            unavailable.add(song.song_id)
            skipped_now += 1
            say(f"  no preview for {song.title} by {song.artist} -> skipping to next pick")
            candidates = [s for s in pool if s.song_id not in unavailable]
            if skipped_now >= MAX_SKIPS or not candidates:
                say("Too many songs without audio; stopping.")
                song = None
                break
            song, mode = agent.choose_next_song(candidates, last_id, ctx,
                                                recent_ids=controller.history[-4:])
        if song is None:
            break
        if taste_shift and step in taste_shift:
            listener.set_liked(taste_shift[step])
            say(f"\n*** listener's taste changes: now likes {sorted(taste_shift[step])} ***")
        say(f"\n[{step}] Now playing: {describe(song)} "
            f"[{mode}, predicted {agent.predict(song, ctx):+.2f}]")
        say(f"  context: filter={ctx.filter_key}, emotion={ctx.emotion}")
        if player is not None and getattr(player, "last_url", None):
            say(f"  playing iTunes preview: {player.last_url.rsplit('/', 1)[-1]}")

        if listener and step in overrides:  # user taps another song on their phone, mid-song
            phone.change_song(overrides[step])
            say(f"  phone: user requested {overrides[step]} (current song keeps playing)")

        if wait_for_playback and player is not None:
            player.wait()  # let the clip play out before reading the reaction
        elif listener and react is None:
            time.sleep(round_delay)  # a real song would take much longer
        if react is not None:
            reaction = react(song.song_id)
        elif listener:
            reaction = listener.react(song)
        else:
            reaction = get_group_reaction(song.song_id, phone)
        if reaction is None:
            say("Bye!")
            break
        if listener and react is None:
            say(f"  simulated reaction: {'like' if reaction.group_liking > 0 else 'dislike'} "
                f"(listener emotion: {dominant_emotion(reaction)})")

        info = agent.update(song, reaction, ctx)
        rewards.append(info["reward"])
        emotion = dominant_emotion(reaction)
        profile = arms.apply(emotion)
        if reaction.group_liking <= UNHAPPY_LIKING:
            say(f"  unhappy (liking {reaction.group_liking:+.2f}): discouraging "
                f"\"{song.title}\" and genre {song.genre}")
        say(f"  reward={info['reward']:+.1f}  error={info['error']:+.2f}  -> "
            f"{'keep similar' if info['reward'] > 0 else 'switch'}")
        say(f"  arms ({emotion}): period {profile.period_s:.2f}s  "
            f"amp {profile.amplitude:.2f}  phase {profile.phase_offset_deg:.0f} deg")
        manual = controller.take_manual_change()  # song boundary: apply phone change
        if manual:
            agent.record_manual_override(manual, song, ctx)
            forced = manual
            say(f"  manual override recorded: +signal for {manual.title}, "
                f"-signal for {song.title}; {manual.title} plays next")
        log.append({"step": step, "song_id": song.song_id, "mode": mode, "emotion": ctx.emotion,
                    "song": song, "reward": info["reward"]})
        if verbose:
            print_learning(agent, Context(ctx.filter_key, emotion))
        last_id = song.song_id

    controller.stop()
    motion.stop()
    say("\n=== Top unplayed songs the agent would recommend ===")
    for emo in EMOTIONS:
        recs = agent.recommend(pool, Context(filter_key(flt), emo), k=3, unplayed_only=True)
        say(f"  when {emo:<7}: " + "; ".join(f"{s.title} ({s.genre}, {p:+.2f})" for s, p in recs))
    say(f"\nMotion stub: {len(motion.tick_times)} ticks, {motion.restarts} restarts, "
        f"longest gap {motion.longest_gap() * 1000:.0f} ms (music events did not interrupt it)")
    return {"agent": agent, "controller": controller, "motion": motion, "arms": arms,
            "log": log, "rewards": rewards, "pool": pool, "ctx_filter": filter_key(flt)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, metavar="N", help="run N rounds with a simulated listener")
    parser.add_argument("--event", default="party", help="first filter, e.g. party, study")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--live", action="store_true",
                        help="read faces from the camera instead of the keyboard")
    parser.add_argument("--camera", type=int, default=2, help="DroidCam index (default 2)")
    parser.add_argument("--detector", default="yunet", help="DeepFace detector, e.g. yunet")
    parser.add_argument("--sample", type=float, default=2.0,
                        help="seconds of camera time per song before scoring the reaction")
    parser.add_argument("--arm-port", default=None,
                        help="Arduino serial port, e.g. COM3. Omit to only print arm lines.")
    args = parser.parse_args()
    react = None
    if args.live:
        from perception.bridge import LiveCamera, start_camera
        react = LiveCamera(start_camera(args.camera, args.detector), sample_s=args.sample)
    run(args.simulate, args.event, args.seed, react=react,
        arms=EmotionArms(ArmLink(args.arm_port)))
