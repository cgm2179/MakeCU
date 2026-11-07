# AGENTS.md: Shared Contract for Humans and AI Agents

Read this before writing code. It is the single source of truth for names, variables, and boundaries. If you add or change a shared name, update this file in the same commit and tell the team.

## Rules

1. Work only inside your own area's folders (see ownership). Do not edit another area's files.
2. `shared/` and this file change only with team agreement.
3. Use the exact names below. Do not invent synonyms (`song_id`, never `track_id` or `songId`).
4. Python code uses `snake_case`; JSON/message fields use the same `snake_case`.
5. Every module must run standalone with mocked inputs.
6. Branches: `<area>/<feature>`. Open PRs into `main`.

## Ownership

| Area | Folders | Branch prefix |
|---|---|---|
| Perception | `perception/` | `perception/` |
| RL | `rl/` | `rl/` |
| Music + App | `music/`, `app/` | `music-app/` |
| Hardware + Integration | `hardware/`, `integration/` | `hardware-integration/` |

## Shared Variables and Messages (canonical names)

Definitions live in `shared/schemas/`. Units and ranges are fixed here.

| Name | Type | Meaning |
|---|---|---|
| `song_id` | str | Unique ID from the music database |
| `filter_type` | str | `"genre"` or `"event"` |
| `filter_value` | str | e.g. `"jazz"`, `"party"`, `"christmas"` |
| `candidate_pool` | list[Song] | Songs left after applying the filter |
| `liking` | float | Reaction score in [-1.0, 1.0]; -1 dislike, 0 neutral, 1 like |
| `confidence` | float | Perception confidence in [0.0, 1.0] |
| `timestamp` | float | Unix seconds (UTC) |
| `reward` | float | RL reward; defined as `liking * confidence` |
| `action` | str | `"skip"` or `"keep"` for the current song |
| `next_song_id` | str | Song the RL agent selects next |

### Messages

- `Song { song_id, title, artist, genre, audio_features: dict[str, float] }`
- `Filter { filter_type, filter_value }`
- `ReactionScore { timestamp, song_id, liking, confidence }`  (Perception -> RL)
- `NextSong { next_song_id }`  (RL -> Music)
- `PlaybackEvent { timestamp, song_id, action }`  (Music/App -> RL)

## Module Interfaces (function names)

| Module | Public entry point | Returns |
|---|---|---|
| perception | `get_reaction(song_id) -> ReactionScore` | latest score from both cameras |
| rl | `choose_next_song(candidate_pool) -> NextSong` | agent's pick |
| rl | `update(reaction: ReactionScore)` | learns from reward |
| music | `get_candidates(filter: Filter) -> list[Song]` | filtered pool |
| music | `play(song_id)` / `stop()` | None |
| integration | `run_loop()` | main orchestrator loop |

## Config Keys (`shared/config/`)

`CAMERA_LEFT_INDEX`, `CAMERA_RIGHT_INDEX`, `SKIP_THRESHOLD` (liking below this skips, default -0.2), `MIN_PLAY_SECONDS` (default 15), `ROBOT_LINK_PORT`.

## Open Questions (fill in as decided)

- Compute board: TBD
- Music database / API: TBD
- Phone link (BLE vs Wi-Fi): TBD
