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
from rl.agent.bandit import BanditAgent, mean_confidence
from rl.agent.context import Context, dominant_emotion, filter_key
from rl.env.simulated_listener import EMOTIONS, SimulatedListener
from shared.bus.bus import Bus
from shared.config.defaults import DEMO_SONG_SECONDS
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
        player=None, wait_for_playback=False, react=None, feedback=None,
        listen_seconds=2.0, mainstream_bias=None, arms=None, reward_model=None, thumb_fn=None, user=None, profile_dir=None):
    """Run the loop. Returns a dict of what happened (used by the tests).

    pool         : candidate songs; default = get_candidates(first filter)
    listener     : fake listener with .react(song); default = SimulatedListener
    taste_shift  : {step: genres} changes a TasteListener's taste at that step
    player       : audio output (AudioPlayer); songs with no playable audio are skipped
    mainstream_bias : 0..1 strength of the early lean toward well-known songs; None keeps the
                   agent's own setting (default shared.config.defaults.MAINSTREAM_BIAS). Applies
                   only when the pool came from the database path (pool.source == "database").
    reward_model : optional rl.agent.reward.CameraRewardModel. Turns the raw camera reading into one reward
                   from every signal available: thumb_fn(song) (+1 / -1 / None), a phone song change during
                   the clip (behaviour) and the camera weighted by how reliable it has proved for this person.
    wait_for_playback : wait for each clip to finish before taking the reaction
    react        : callable(song_id) -> GroupReaction from the live camera
    feedback     : real reactions instead of a fake listener, e.g. CameraFeedback (starts its
                   perception service here). With audio the reaction is read at the end of
                   the clip; without audio, after listen_seconds.
    arms         : EmotionArms; default sends arm lines in memory (no serial)
    """
    say = print if verbose else (lambda *a, **k: None)
    bus = Bus()
    flt = Filter(sources=(FilterSource("event", event),))
    pool = pool if pool is not None else get_candidates(flt)
    if agent is None and user:
        from rl.agent import profile as _profile
        try:
            agent = _profile.load_profile(user, directory=profile_dir, seed=seed)
            say(f"Loaded saved profile for '{user}' ({agent.n_updates} earlier updates).")
        except _profile.ProfileError as e:
            if "no saved profile" not in str(e):
                raise                       # a damaged or mismatched profile is never silently replaced
            say(f"No saved profile for '{user}' yet: starting fresh.")
    agent = agent or BanditAgent(seed=seed)
    controller = PlaylistController(pool, bus, player)
    if mainstream_bias is not None:
        agent.mainstream_bias = mainstream_bias
    agent.mainstream_active = getattr(pool, "source", None) == "database"  # never for a user's playlist
    if agent.mainstream_active and agent.mainstream_bias > 0:
        say(f"Mainstream bias {agent.mainstream_bias:.2f}: leaning toward well-known songs early on (database pool)")
    phone = FakePhone(bus)
    if feedback is not None:
        listener = None
        if hasattr(feedback, "start"):
            feedback.start()  # perception service runs in the background from here on
        if getattr(feedback, "calibrate_seconds", 0) > 0:
            feedback.calibrate(say=say)  # measure the resting baseline before any music
    elif react is None and listener is None and simulate_rounds:
        listener = SimulatedListener(seed=seed or 0)
    if overrides is None:
        if not simulate_rounds or feedback is not None or react is not None:
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

    emotion, step, last_id, forced, forced_from_phone = "neutral", 0, None, None, False
    unavailable: set = set()  # songs the audio output could not play
    log, rewards, skipped_rounds = [], [], 0
    while simulate_rounds is None or step < simulate_rounds:
        step += 1
        ctx = Context(filter_key(flt), emotion)
        if forced is not None:  # pre-queued (demo next-up) or phone manual override
            song, mode, forced = forced, ("phone" if forced_from_phone else "queued"), None
            forced_from_phone = False
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

        if feedback is not None and hasattr(feedback, "begin_song"):
            feedback.begin_song(song)

        if (listener or feedback is not None or react is not None) and step in overrides:  # user taps another song mid-song
            phone.change_song(overrides[step])
            say(f"  phone: user requested {overrides[step]} (current song keeps playing)")

        # Demo-friendly fixed duration: every song runs for DEMO_SONG_SECONDS then we advance.
        # No mid-song emotion skips — the DJ does not cut a track early.
        play_secs = DEMO_SONG_SECONDS
        if wait_for_playback and player is not None:
            # iTunes previews are ~30 s; player may already be clipped to play_secs
            player.wait()
        elif feedback is not None:
            time.sleep(listen_seconds if listen_seconds is not None else play_secs)
        elif listener and react is None:
            # simulate mode stays fast unless the caller asked for a real delay
            time.sleep(round_delay if round_delay > 0 else 0)
        else:
            time.sleep(play_secs)
        if feedback is not None:
            reaction = feedback.get_group_reaction(song.song_id)  # read at the end of the clip
        elif react is not None:
            reaction = react(song.song_id)
        elif listener:
            reaction = listener.react(song)
        else:
            reaction = get_group_reaction(song.song_id, phone)
        if reaction is None:
            say("Bye!")
            break
        if feedback is not None or react is not None:
            shares = {k: round(v, 2) for k, v in sorted(reaction.group_emotion.items(), key=lambda kv: -kv[1]) if v}
            say(f"  camera reaction: people={len(reaction.people)}, group_liking={reaction.group_liking:+.2f}, "
                f"mean confidence={mean_confidence(reaction):.2f}, emotions={shares}")
            cal = getattr(feedback, "last_calibration", None) if feedback is not None else None
            if cal:
                base = f"{cal['baseline']:+.2f}" if cal["baseline"] is not None else "none"
                win = f" (avg of last {cal['window']:.0f} s)" if cal["window"] else ""
                say(f"  liking: raw {cal['raw']:+.2f}{win} | baseline {base} | calibrated {cal['calibrated']:+.2f}  <- this is what the bandit gets")
                tr = cal["trace"]
                if tr:
                    mix = ", ".join(f"{e} {v:.0%}" for e, v in list(tr["emotions"].items())[:4])
                    say(f"  across the whole clip (calibrated): peak {tr['peak']:+.2f}, low {tr['low']:+.2f}, avg {tr['mean']:+.2f}; emotions: {mix}")
            for pr in reaction.people:
                say(f"    person {pr.person_id}: emotion={pr.emotion}, liking={pr.liking:+.2f}, "
                    f"confidence={pr.confidence:.2f}")
            if feedback is not None and hasattr(feedback, "consume_stats"):
                frames, avg_ms, faces = feedback.consume_stats()
                say(f"  camera feed this clip: {frames} frames analysed (avg {avg_ms:.0f} ms each), "
                    f"faces in last frame: {faces}")
        elif listener:
            say(f"  simulated reaction: {'like' if reaction.group_liking > 0 else 'dislike'} "
                f"(listener emotion: {dominant_emotion(reaction)})")

        if reward_model is not None:
            thumb = thumb_fn(song) if thumb_fn else None
            reaction = reward_model(song, reaction, feedback, thumb, "replaced" if step in overrides else None)
            reward, parts = reward_model.last
            say("  combined reward: " + ("no signal" if reward is None else
                f"{reward:+.2f} from " + ", ".join(f"{n} {s:+.1f} x{w:.2f}" for n, w, s in parts)))
        info = agent.update(song, reaction, ctx)
        if info.get("skipped"):
            skipped_rounds += 1
            say("  nobody in view (confidence ~0) -> skipping the learning step for this round")
        else:
            rewards.append(info["reward"])
            say(f"  reward={info['reward']:+.2f}  error={info['error']:+.2f}  -> "
                f"{'keep similar' if info['reward'] > 0 else 'switch'}")

        emotion = dominant_emotion(reaction)
        profile = arms.apply(emotion)
        if reaction.group_liking <= UNHAPPY_LIKING:
            say(f"  unhappy (liking {reaction.group_liking:+.2f}): discouraging "
                f"\"{song.title}\" and genre {song.genre}")
        say(f"  arms ({emotion}): period {profile.period_s:.2f}s  "
            f"amp {profile.amplitude:.2f}  phase {profile.phase_offset_deg:.0f} deg")

        # --- Demo messaging: what the agent will recommend more of, and the next track ---
        # Songs always finish their DEMO_SONG_SECONDS window; we only change what comes *next*.
        next_ctx = Context(ctx.filter_key, emotion)
        playable = [x for x in pool if x.song_id not in unavailable]
        next_song, next_mode = agent.choose_next_song(
            playable, last_song_id=song.song_id, context=next_ctx,
            recent_ids=controller.history[-4:])
        cur_genre = song.genre or "this"
        next_genre = next_song.genre or "similar"
        if reaction.group_liking <= UNHAPPY_LIKING or emotion in ("sad", "bored", "angry"):
            if next_genre != cur_genre:
                say(f"  Don't like {cur_genre}? Recommending more {next_genre} instead!")
            else:
                say(f"  Not feeling {cur_genre} right now — trying a different {next_genre} track next.")
        elif emotion == "happy" or reaction.group_liking > 0.25:
            say(f"  Loving the {cur_genre} energy! Queuing more {next_genre}.")
        else:
            say(f"  Neutral on {cur_genre}. Exploring {next_genre} next.")
        say(f"  >>> NEXT UP ({next_mode}): {describe(next_song)}")

        manual = controller.take_manual_change()  # song boundary: apply phone change
        if manual:
            agent.record_manual_override(manual, song, ctx)
            forced, forced_from_phone = manual, True    # Song is frozen: keep the flag beside it, not on it
            say(f"  manual override recorded: +signal for {manual.title}, "
                f"-signal for {song.title}; {manual.title} plays next")
        else:
            # Make the announced next song actually the one that plays (no surprise re-roll).
            forced, forced_from_phone = next_song, False
        log.append({"step": step, "song_id": song.song_id, "mode": mode, "emotion": ctx.emotion,
                    "song": song, "reward": info["reward"], "skipped": bool(info.get("skipped")),
                    "reaction": reaction})
        if verbose:
            print_learning(agent, Context(ctx.filter_key, emotion))
        last_id = song.song_id

    controller.stop()
    motion.stop()
    if feedback is not None and hasattr(feedback, "stop"):
        feedback.stop()
    say("\n=== Top unplayed songs the agent would recommend ===")
    for emo in EMOTIONS:
        recs = agent.recommend(pool, Context(filter_key(flt), emo), k=3, unplayed_only=True)
        say(f"  when {emo:<7}: " + "; ".join(f"{s.title} ({s.genre}, {p:+.2f})" for s, p in recs))
    say(f"\nMotion stub: {len(motion.tick_times)} ticks, {motion.restarts} restarts, "
        f"longest gap {motion.longest_gap() * 1000:.0f} ms (music events did not interrupt it)")
    saved = None
    if user:
        from rl.agent import profile as _profile
        saved = _profile.save_profile(agent, user, directory=profile_dir)
        say(f"Saved profile for '{user}' to {saved}")
    return {"agent": agent, "controller": controller, "motion": motion, "arms": arms, "profile_path": saved,
            "log": log, "rewards": rewards, "pool": pool, "ctx_filter": filter_key(flt),
            "skipped_rounds": skipped_rounds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, metavar="N", help="run N rounds with a simulated listener")
    parser.add_argument("--event", default="party", help="first filter, e.g. party, study")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--user", default=None,
                        help="load this user's saved taste profile at the start and save it at the end")
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
        # sample a few seconds of faces, but the song itself still runs for DEMO_SONG_SECONDS
        react = LiveCamera(start_camera(args.camera, args.detector), sample_s=args.sample)
        print(f"Demo mode: each song plays ~{DEMO_SONG_SECONDS}s then we announce the next track.")
        print(f"  (change DEMO_SONG_SECONDS in shared/config/defaults.py to speed up or slow down)")
    run(args.simulate, args.event, args.seed, react=react,
        arms=EmotionArms(ArmLink(args.arm_port)),
        listen_seconds=DEMO_SONG_SECONDS, user=args.user)
