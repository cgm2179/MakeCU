# AGENTS.md: Shared Contract for Humans and AI Agents

Read this before writing code. It is the single source of truth for names, variables, and boundaries. If you add or change a shared name, update this file in the same commit and tell the team.

## Rules

1. Work only inside your own area's folders (see ownership). Do not edit another area's files.
2. `shared/` and this file change only with team agreement.
3. Use the exact names below. Do not invent synonyms (`song_id`, never `track_id` or `songId`).
4. Python code uses `snake_case`; JSON/message fields use the same `snake_case`.
5. Every module must run standalone with mocked inputs.
5a. **Music and motion are independent.** No imports between `music/` and `motion/`. They talk only through topics in `shared/bus/`. A manual song change must never interrupt or reset motion.
6. Branches: `<area>/<feature>`. Open PRs into `main`.

## Ownership

| Area | Folders | Branch prefix |
|---|---|---|
| Perception | `perception/` | `perception/` |
| RL | `rl/` | `rl/` |
| Music + App | `music/`, `app/` | `music-app/` |
| Motion + Hardware + Integration | `motion/`, `hardware/`, `integration/` | `motion-hardware/` |

## Shared Variables and Messages (canonical names)

Definitions live in `shared/schemas/`. Units and ranges are fixed here.

| Name | Type | Meaning |
|---|---|---|
| `song_id` | str | Unique ID from the music database |
| `source_type` | str | `"event"`, `"random"`, `"playlist"`, `"artists"`, or `"themes"` |
| `source_value` | str | e.g. `"party"`, an Apple Music playlist ID, `"Drake,SZA"`, `"heartbreak"` |
| `person_id` | int | Per-frame ID of a tracked person |
| `emotion` | str | `"happy"`, `"sad"`, `"neutral"`, `"angry"`, `"bored"`, `"surprised"` |
| `group_liking` | float | Mean `liking` across people, in [-1.0, 1.0] |
| `group_emotion` | dict[str, float] | Share of people per `emotion` (sums to 1) |
| `candidate_pool` | list[Song] | Songs left after applying the filter |
| `liking` | float | Reaction score in [-1.0, 1.0]; -1 dislike, 0 neutral, 1 like |
| `confidence` | float | Perception confidence in [0.0, 1.0] |
| `timestamp` | float | Unix seconds (UTC) |
| `reward` | float | RL reward; defined as `group_liking * mean(confidence)` |
| `action` | str | `"skip"` or `"keep"` for the current song |
| `next_song_id` | str | Song the RL agent selects next |

### Messages

- `Song { song_id, title, artist, genre, audio_features: dict[str, float] (incl. "energy" 0-1), mood: str }`
- `FilterSource { source_type, source_value }`
- `Filter { sources: list[FilterSource] }`  (one or more signals combined)
- `UserInput { timestamp, text, source }`  (`source` = `"voice"` or `"typed"`)
- `PersonReaction { person_id, emotion, liking, confidence }`
- `GroupReaction { timestamp, song_id, people: list[PersonReaction], group_liking, group_emotion }`  (Perception -> RL)
- `PersonPose { person_id, keypoints: list[[x, y]], velocity: [vx, vy] }`
- `UserMotion { timestamp, people: list[PersonPose] }`  (Perception -> Motion)
- `NextSong { next_song_id }`  (RL -> Music)
- `ManualSongChange { song_id, timestamp }`  (App -> Music only, via bus topic `music.manual_change`)
- `PlaybackEvent { timestamp, song_id, action }`  (Music -> RL)
- `MotionCommand { timestamp, target, speed }`  (Motion -> drivers)

## Bus Topics (`shared/bus/`)

Music side: `music.filter`, `music.next_song`, `music.manual_change`, `music.playback_event`.
Motion side: `motion.user_motion`, `motion.command`.
Shared inputs: `perception.group_reaction`, `perception.user_motion`.
Music topics are never consumed by motion, and the reverse.

## RL Policy and Manual Override (implemented in the skeleton)

- Bandit context = `Context { filter_key, emotion }` (`rl/agent/context.py`). `filter_key` looks like `"event:party"`; `emotion` is the dominant emotion of the last `GroupReaction`.
- Reward = `group_liking * mean(confidence)`. Features per song: `genre`, `artist`, `energy` bucket (low/mid/high), `mood`.
- `agent.choose_next_song(pool, last_song_id, context, recent_ids)`, `agent.update(song, reaction, context)`, `agent.record_manual_override(chosen, replaced, context)`, `agent.recommend(pool, context, k, unplayed_only)`.
- A phone song change is posted to the bus and applied at the next song boundary (`PlaylistController.take_manual_change()`). It is recorded as a weaker signal (+0.3 chosen, -0.15 replaced). It never touches motion.
- Tests: `python3 -m unittest rl.tests.test_bandit music.tests.test_playlist integration.tests.test_core_loop -v`

## Song Data and First Filter (implemented)

- Dataset columns map to `Song`: `track_id`->`song_id`, `track_name`->`title`, first of `artists`->`artist`, `track_genre`->`genre`; `valence, energy, danceability, tempo (BPM), acousticness, instrumentalness, popularity` -> `audio_features`; `mood` is derived from the valence/energy quadrant (`upbeat`, `aggressive`, `calm`, `melancholy`).
- `get_candidates(filter, pool_size=500)`: an `event` source whose `source_value` is a mood name (see `music/filter/mood_targets.py`) returns the songs nearest that target; `random` returns a random sample. `playlist`, `artists`, `themes` sources are not implemented yet.
- The bandit buckets `energy`, `valence`, `danceability`, `tempo` (plus genre, artist, mood). Do not add more buckets without re-testing: acousticness/instrumentalness buckets slowed learning.
- Tests for the dataset need the CSV and are skipped without it.

## Audio Playback (implemented)

- `PlaylistController(catalog, bus, player=None).play(song) -> bool`. With an `AudioPlayer` it plays the song's iTunes preview; it returns `False` (and records nothing) when no preview exists, and the loop then skips to another pick and never retries that song.
- `AudioPlayer.play(song)` is non-blocking; `wait()`, `stop()`, `close()` control it. Swap in your own audio output by passing any object with `play(song) -> bool`, `wait()`, `stop()`.
- Matching requires the same artist and a similar title; a weak match is treated as "no preview" rather than playing the wrong song.

## Camera Feedback in the Loop (implemented, mock mode)

- `run(..., feedback=CameraFeedback(mock=True))` replaces the fake listener. It starts `perception_cv.serve(...)` in the background, calls `feedback.begin_song(song)` when a song starts, and `feedback.get_group_reaction(song_id)` at the end of the clip (or after `listen_seconds` without audio).
- Reward is still `group_liking * mean(confidence)`. If mean confidence is below `MIN_CONFIDENCE` (0.1) the bandit skips learning for that round (`info["skipped"]`), because a reward of 0 would pull the song's score toward 0.
- In mock mode a thread feeds synthetic frames to the service; real phones or a webcam replace that later.

## Gemini Feature Lookup (implemented, tested offline with a fake transport)

- `SongLookup(gemini=GeminiClient(), ...).lookup(title, artist) -> LookupResult(song, source, reason)`; `source` is `dataset` | `cache` | `gemini` | `unidentified` | `error` | `gemini-disabled`. `song` is `None` unless a Song was found; callers skip those. `extend_pool(pool, songs)` adds looked-up songs to a candidate pool.
- Looked-up songs have `song_id` starting `gem:`, genre from the dataset's 114 tags, `mood` in {upbeat, aggressive, calm, melancholy}, and `audio_features` with `valence`, `energy` (and `danceability`, `tempo` when given) but no `popularity`.
- API failures (network, rate limit, bad key) are returned as `source="error"` and are not cached; "unidentified" is cached. The key is read from `GEMINI_API_KEY`/`GOOGLE_API_KEY` or a git-ignored `.env`, sent in a header only.

## Mainstream Bias (implemented)

- `get_candidates(filter, pool_size=500, mainstream_bias=None)` returns a `CandidatePool` (a list with `.source` = `"database"` | `"hardcoded"`; a future playlist path should return `source="playlist"`). Default strength is `shared.config.defaults.MAINSTREAM_BIAS` (0.5; 0 = off, 1 = strong).
- Database path only: `select_pool` ranks by distance minus a popularity bonus (popularity floor is soft: lower-popularity songs top the pool up, so it is never short); `BanditAgent.mainstream_prior` adds a fading popularity prior while choosing. `run()` switches the prior on only when `pool.source == "database"`, so a user's own playlist is never biased.
- Tradeoff measured on 16 seeds: strength 0.5 lifts early-pick popularity from ~47 to ~70-82 and costs a taste for unpopular genres ~8 points of learning (91% to 83%); strength 1.0 costs ~20.

## Camera Signals, Reliability and Combined Reward (implemented, opt-in)

- `CameraFeedback.clip_features()` -> per-clip dict (`perception/emotion/clip_features.py`): reaction dynamics (`mean/peak/low/range/build/last3_liking`), `smile`, and movement (`motion`, `vert_motion`, `bob_share`, `bob_amp`), measured against the resting baseline taken in `calibrate()`.
- `rl/agent/camera_model.py`: `CameraReliability` (per-person logistic model of P(thumb up | features)); `reliability(rows)` returns a camera weight 0..1 = 2 x (leave-one-out AUC - 0.5); 0 below 30 songs with 8 of each thumb. `soft_reward` is relative to the person's usual rate and averages 0.
- `rl/agent/reward.py`: `combine(thumb, behaviour, camera, camera_weight)` = sum(w*s) / max(sum(w), 1) with thumb 1.0, behaviour 0.7 (song replaced / chosen), camera = its measured weight. `CameraRewardModel` plugs into `run(..., reward_model=..., thumb_fn=...)`. Off by default.
- Measured on synthetic worlds (`reports/step03c_signal_weighting_simulation.md`): weighting did not beat using the camera at full strength; reliability is poorly estimated from under ~150 songs with 12 features. Judge on real data (Step 3c) before turning it on.
- Tried and removed: a per-person leniency offset in the bandit (no gain; see `rl/agent/bandit.py`).

## Two-Question Ratings (implemented)

- `python3 -m integration.orchestrator.run_rating_session --user adi` plays 30-second iTunes previews and asks QUALITY (1-5, "ignoring my taste, how well-made?") and PREFERENCE (1-5, "how much do I want to hear it?"). Songs are spread over mood x popularity band, one per artist, at most 6 per genre. Every 5th rated song is held out (30 of 150). `--retest` re-rates 20 songs at least 12 h later (Step 5).
- `--hunt` (liked-song hunt): rates `--hunt-songs` (40) more songs, preference only, half picked by the bandit as likely-liked. Saved as `round` 3, `split` dev; not part of the 150, the gates or the held-out set. `python3 -m rl.ratings --user adi --export FILE --include-hunt` adds them to the labels.
- File: `data/ratings/<user>.csv` (git-ignored): `user, song_id, quality, preference, split, round, time`. `rl/ratings.py` holds the checks (disagreement >= 15% by 2+ points, re-rating within 1 point on >= 80%) and `export_labels` (preference 4-5 = liked, 1-2 = disliked, 3 left out) for `rl.eval`.

## Saved Profiles and Personalization (implemented)

- `run(..., user="adi")` (or `run_loop --user adi`) loads `data/profiles/<user>.json` if it exists and saves it at the end. A missing profile starts fresh; a damaged or wrong-base one raises `ProfileError` and is never silently replaced.
- `python3 -m rl.personalize --user adi [--include-heldout]` trains a profile from the ratings file (preference 4-5 liked, 1-2 disliked, hunt songs included) and saves it. Base model is `none-v0`: layer 1 fell back to popularity-as-objective because quality and preference ratings did not separate (reports/step04_ratings.md).

## Live Session Test (implemented)

- `python3 -m integration.orchestrator.run_live_session --user adi` plays songs from three hidden conditions in random blocks of three (random / one of the 10 most popular / one of the 10 the saved profile likes best), asks for a thumb at once, and stops the song the moment you answer. Never plays a song the profile was trained on. Saved to `data/live/<user>.csv` (git-ignored); resumes after `q`.
- `python3 -m rl.live_test --user adi` summarises and writes `reports/step07_live_session.md`. Gate: personal >= popularity + 15 points and >= random + 25 points, 40+ songs per condition (about +/-15 points of noise at that size).

## Base Model v1 (implemented)

- `python3 -m rl.base_model --user adi` trains a logistic-regression taste model on the user's thumbs (agreement + live), saves checkpoints every 40 epochs to `models/checkpoints/`, the best one to `models/base_model_v1.json` (self-contained, SHA-256 of the weights, `load_model` refuses a changed file), plus `models/training_history.csv` and `models/training_curve.svg`. `models/` is git-ignored. Split 70/15/15; the test split is read once. It does not beat popularity-only (reports/step06b_base_model.md).

## Base Model v2 on Last.fm (implemented, Step 6)

- `python3 -m rl.base_lastfm` trains the base on other people's listening (Last.fm 1K in `data/external/lastfm1k/`, git-ignored, non-commercial licence). Definitions and pass marks are in `reports/step06_preregistration.md`; do not change them after seeing results. Result: beats popularity by +0.054 AUC on 126 held-out users (first 600-epoch run +0.025; test users read twice), match rate 4.9% (mark 30%) and quality-vs-preference split fail: L1 gate FAIL (reports/step06_base_model.md).
- Two heads: preference (logistic) and objective q (ridge on log listeners). `models/base_model_v2.json` is read-only with a SHA-256 (`load_base()` refuses a changed file); `models/base_model_v2.bin` / `.h` are the 1,477-byte ESP32 versions (RAM limit: hard cap 520 KB, budget 32 KB). `models/` is git-ignored.

## Module Interfaces (function names)

| Module | Public entry point | Returns |
|---|---|---|
| perception | `get_group_reaction(song_id) -> GroupReaction` | all people seen by both cameras |
| perception (camera package) | `perception_cv.serve(perception_cv.build_service(), background=True)`, then `perception_cv.get_group_reaction(song_id)` | lives in `perception/perception_cv/`; import name `perception_cv` (run from inside `perception/`); wired into the loop in mock mode via `perception/emotion/camera_reaction.py` (`CameraFeedback`); `run(..., feedback=CameraFeedback(...))` |
| perception | `get_user_motion() -> UserMotion` | current poses for motion |
| rl | `choose_next_song(candidate_pool) -> NextSong` | agent's pick |
| rl | `update(reaction: GroupReaction)` | learns from reward |
| music | `build_filter(user_input: UserInput) -> Filter` | filter from speech/text |
| music | `get_candidates(filter: Filter) -> list[Song]` | filtered pool |
| music | `play(song_id)` / `stop()` | None |
| motion | `run_motion_loop()` | follows `UserMotion`, independent of music |
| integration | `start_all()` | launches music and motion as separate processes |

## Config Keys (`shared/config/`)

`CAMERA_LEFT_INDEX`, `CAMERA_RIGHT_INDEX`, `SKIP_THRESHOLD` (`group_liking` below this skips, default -0.2), `MAX_PEOPLE` (default 6), `MOTION_MAX_SPEED`, `MIN_PLAY_SECONDS` (default 15), `ROBOT_LINK_PORT`.

## Open Questions (fill in as decided)

- Compute board: TBD
- Music database / API: TBD
- Phone link (BLE vs Wi-Fi): TBD
