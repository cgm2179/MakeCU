# MakeCU: A Speaker That Learns Your Taste

Built for **MakeCU**, Columbia University Robotics Club's 24-hour hardware hackathon.

A robot that learns what music you like by watching how you react to it. It plays a song, reads your face and body language with two cameras, and uses reinforcement learning to get better at picking the next song.

## How It Works

1. **Set a filter.** From your phone, you give a starting input such as an event or genre ("party music", "Christmas music", "jazz"). This narrows the public music database to a candidate pool.
2. **Play a song.** The robot picks a song from the pool and plays it through its speaker. The phone controls the playlist.
3. **Read the reaction.** Two cameras act as the robot's eyes. A perception model estimates your physical and emotional response (head bobbing, smiling, tapping, bored or annoyed expressions) and turns it into a single "liking" score.
4. **Learn.**
   - Low score: skip to a different song.
   - High score: reinforce that choice.
   - Over time the RL agent learns your taste *within the chosen filter* and picks songs it predicts you'll like.

```
 Phone (filter, playlist UI)
        │  BLE / Wi-Fi
        ▼
┌──────────────────────────── Robot ────────────────────────────┐
│  Music catalog ─► RL agent ─► Playback ─► Speaker             │
│                      ▲                                        │
│                      │ reward                                 │
│  Camera 1 ┐                                                   │
│           ├─► Perception (reaction → liking score)            │
│  Camera 2 ┘                                                   │
└───────────────────────────────────────────────────────────────┘
```

## Hardware

- Speaker
- Two cameras
- Compute board for the robot (to be decided) and a mechanical chassis/mount

## Team Ownership

Each person owns one top-level area. Cross-area contracts live in `shared/`, so changes there need a heads-up to the team.

| Owner | Area | Responsibility |
|---|---|---|
| Person A | `perception/` | Camera capture, face/body reaction model, outputs a liking score |
| Person B | `rl/` | RL agent, state/action/reward design, learning loop, simulated environment for testing without hardware |
| Person C | `music/` and `app/` | Music database access, filtering, playback, and the phone interface/API |
| Person D | `hardware/` and `integration/` | Mechanical, electrical, firmware, and the orchestrator that wires everything together |

## Repository Layout

```
MakeCU/
├── perception/        # Person A: cameras → liking score
│   ├── capture/       #   camera drivers and frame sync
│   ├── models/        #   emotion/engagement models
│   └── tests/
├── rl/                # Person B: taste learning
│   ├── agent/         #   policy, reward, training
│   ├── env/           #   simulated listener for offline testing
│   └── tests/
├── music/             # Person C: song source and playback
│   ├── catalog/       #   public music DB client, genre/event filtering
│   ├── playback/      #   audio output and queue
│   └── tests/
├── app/               # Person C: phone side
│   ├── mobile/        #   phone UI (filter input, playlist control)
│   └── api/           #   robot ↔ phone link
├── hardware/          # Person D: physical build
│   ├── mechanical/    #   CAD, mounts, enclosure
│   ├── electrical/    #   wiring, schematics, BOM
│   └── firmware/      #   microcontroller code
├── integration/       # Person D: system glue
│   ├── orchestrator/  #   main loop connecting all modules
│   └── tests/         #   end-to-end tests
├── shared/            # Contracts everyone depends on
│   ├── schemas/       #   message formats (e.g. Song, ReactionScore)
│   └── config/        #   shared settings
├── docs/              # Notes, diagrams, demo script
└── data/              # Local datasets and logs (large files not committed)
```

## Interfaces Between Modules

Agree on these early (put them in `shared/schemas/`) so everyone can build in parallel against mocks:

- **Perception → RL:** `ReactionScore { timestamp, song_id, liking: float in [-1, 1], confidence }`
- **RL → Music:** `NextSong { song_id }`, plus `Skip` and `Reinforce` events
- **Music → RL:** `Song { id, title, artist, genre, audio_features }` for the candidate pool
- **App → System:** `Filter { type: "genre" | "event", value }`, plus playlist controls

## Working Without Colliding

- Stay inside your own folder; change `shared/` only after telling the team.
- Use branches named `<area>/<feature>` and open PRs into `main`.
- Each module should run on its own with mocked inputs (for example, the RL agent trains against the simulated listener in `rl/env/`).
- Keep large files (datasets, models, CAD exports) out of git unless they're small.

## Status

Scaffolding only. No component code yet.
