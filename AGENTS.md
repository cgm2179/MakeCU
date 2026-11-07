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

- `Song { song_id, title, artist, genre, audio_features: dict[str, float] }`
- `FilterSource { source_type, source_value }`
- `Filter { sources: list[FilterSource] }`  (one or more signals combined)
- `UserInput { timestamp, text, source }`  (`source` = `"voice"` or `"typed"`)
- `PersonReaction { person_id, emotion, liking, confidence }`
- `GroupReaction { timestamp, song_id, people: list[PersonReaction], group_liking, group_emotion }`  (Perception -> RL)
- `PersonPose { person_id, keypoints: list[[x, y]], velocity: [vx, vy] }`
- `UserMotion { timestamp, people: list[PersonPose] }`  (Perception -> Motion)
- `NextSong { next_song_id }`  (RL -> Music)
- `ManualSongChange { song_id }`  (App -> Music only)
- `PlaybackEvent { timestamp, song_id, action }`  (Music -> RL)
- `MotionCommand { timestamp, target, speed }`  (Motion -> drivers)

## Bus Topics (`shared/bus/`)

Music side: `music.filter`, `music.next_song`, `music.manual_change`, `music.playback_event`.
Motion side: `motion.user_motion`, `motion.command`.
Shared inputs: `perception.group_reaction`, `perception.user_motion`.
Music topics are never consumed by motion, and the reverse.

## Module Interfaces (function names)

| Module | Public entry point | Returns |
|---|---|---|
| perception | `get_group_reaction(song_id) -> GroupReaction` | all people seen by both cameras |
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
