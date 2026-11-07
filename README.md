# MakeCU: A Speaker That Learns Your Taste

Built for **MakeCU**, Columbia University Robotics Club's 24-hour hardware hackathon.

A robot that learns what music you (or your group) like by watching how people react, and that physically moves with them while it plays. It picks a song, reads reactions with two cameras, and uses reinforcement learning to choose better songs over time.

## How It Works

1. **Set the first filter.** The user speaks or types an input on their phone. That input builds the first filter, which narrows the public music database to a candidate pool. The filter can be built from one or more of these signals:
   - **Event:** the occasion, such as "party", "study", "Christmas".
   - **Random:** a random selection.
   - **Existing playlist:** a playlist the user already has, such as an Apple Music playlist.
   - **Preferred artists:** artists the user names.
   - **Words or themes:** lyrical words or themes in songs the user tends to like.
2. **Play a song.** The robot plays a song from the pool through its speaker. The phone controls the playlist.
3. **Read the reaction.** Two cameras act as the robot's eyes. They can analyze **more than one person at once**, so the robot reads a group's reaction, not just a single user. For each person it estimates an emotional state (happy, sad, bored, and so on) and a liking score, then combines them into a group signal.
4. **Learn.** The emotional states are part of the feedback the RL agent uses to decide whether to keep or switch the song.
   - Negative group reaction: switch to a different song.
   - Positive group reaction: reinforce that choice.
   - Over time the agent learns the taste of the user or group *within the chosen filter*.
5. **Move.** The robot also **moves in response to the user's movement** (for example mirroring or following how people dance or sway) while the music plays.

### Two independent systems

Music control and physical movement **run independently** as separate processes that talk only through messages. If the user manually changes the song from their phone, the robot's current movement is not interrupted or reset, and a motion change never stops or skips the music.

```
 Phone (spoken/typed input, playlist UI)
        │  BLE / Wi-Fi
        ▼
┌────────────────────────────── Robot ──────────────────────────────┐
│                                                                    │
│  MUSIC SYSTEM (independent)          MOTION SYSTEM (independent)   │
│  filter ─► catalog ─► RL agent       tracking ─► control ─► drivers│
│                 │         ▲                ▲                  │    │
│                 ▼         │ reward         │ user pose        ▼    │
│             playback   ┌──┴────────────────┴───┐         motors    │
│                 │      │  PERCEPTION           │                   │
│                 ▼      │  Camera 1 + Camera 2  │                   │
│             speaker    │  pose · emotion · group│                  │
│                        └───────────────────────┘                   │
│                                                                    │
│            message bus (shared/bus): the only link between systems │
└────────────────────────────────────────────────────────────────────┘
```

Perception feeds both systems: group reactions go to the RL agent, and user pose goes to the motion system. Neither system calls the other directly.

## Hardware

- Speaker
- Two cameras
- Motors/actuators and a chassis for movement
- Compute board (to be decided)

## Team Ownership

Each person owns separate top-level folders. Cross-area contracts live in `shared/`, so changes there need a heads-up to the team. Exact variable names are in [AGENTS.md](AGENTS.md).

| Owner | Area | Responsibility |
|---|---|---|
| Person A | `perception/` | Camera capture, per-person pose and emotion, group aggregation |
| Person B | `rl/` | RL agent, reward from emotion/liking, learning loop, simulated listeners |
| Person C | `music/` and `app/` | Filter building, music database, playback, phone app, voice/text input |
| Person D | `motion/`, `hardware/`, `integration/` | Movement control, mechanical, electrical, firmware, process orchestration |

## Repository Layout

```
MakeCU/
├── perception/        # Person A: cameras → pose, emotion, group signal
│   ├── capture/       #   dual-camera drivers and frame sync
│   ├── pose/          #   per-person body tracking (feeds motion)
│   ├── emotion/       #   per-person emotion + liking models (feeds RL)
│   ├── group/         #   multi-person aggregation into one group signal
│   └── tests/
├── rl/                # Person B: taste learning
│   ├── agent/         #   policy, reward, training
│   ├── env/           #   simulated listeners/groups for offline testing
│   └── tests/
├── music/             # Person C: MUSIC SYSTEM (no motion code here)
│   ├── filter/        #   builds the first filter from user input
│   │   └── sources/   #   event, random, playlist (Apple Music), artists, themes
│   ├── catalog/       #   public music DB client, applies the filter
│   ├── playback/      #   audio output and queue, handles manual song changes
│   └── tests/
├── app/               # Person C: phone side
│   ├── mobile/        #   phone UI (playlist control)
│   ├── voice/         #   spoken/typed input → text
│   └── api/           #   robot ↔ phone link
├── motion/            # Person D: MOTION SYSTEM (no music code here)
│   ├── tracking/      #   turns user pose into movement targets
│   ├── control/       #   movement planning and control loop
│   ├── drivers/       #   motor/actuator interfaces
│   ├── sim/           #   motion simulator for testing without hardware
│   └── tests/
├── hardware/          # Person D: physical build
│   ├── mechanical/    #   CAD, mounts, enclosure
│   ├── electrical/    #   wiring, schematics, BOM
│   └── firmware/      #   microcontroller code
├── integration/       # Person D: starts and supervises the independent processes
│   ├── orchestrator/
│   └── tests/         #   end-to-end tests, including independence tests
├── shared/            # Contracts everyone depends on
│   ├── schemas/       #   message formats
│   ├── bus/           #   message topics between the music and motion systems
│   └── config/
├── docs/
└── data/              # Local datasets and logs (large files not committed)
```

## Interfaces Between Modules

Put these in `shared/schemas/` and `shared/bus/` early so everyone can build against mocks:

- **App → Music:** `UserInput { timestamp, text, source: "voice" | "typed" }`
- **Music (filter):** `Filter { sources: list[FilterSource] }`, where `FilterSource { source_type: "event" | "random" | "playlist" | "artists" | "themes", value }`
- **Perception → RL:** `GroupReaction { timestamp, song_id, people: list[PersonReaction], group_liking, group_emotion }`
- **Perception → Motion:** `UserMotion { timestamp, people: list[PersonPose] }`
- **RL → Music:** `NextSong { next_song_id }`
- **App → Music:** `ManualSongChange { song_id }`. This never reaches the motion system.

## Independence Rules

- Music and motion are separate processes with no imports between `music/` and `motion/`.
- They communicate only over topics in `shared/bus/`. Music topics and motion topics are separate.
- A `ManualSongChange` only affects the music system. Motion keeps running.
- `integration/tests/` must include a test that changes songs mid-movement and checks the motion is not interrupted.

## Working Without Colliding

- Stay inside your own folders; change `shared/` and `AGENTS.md` only after telling the team.
- Use branches named `<area>/<feature>` and open PRs into `main`.
- Each module should run alone with mocked inputs.
- Keep large files (datasets, models, CAD exports) out of git.

## Try It (fake inputs, no hardware)

```
python3 -m integration.orchestrator.run_loop --simulate 25   # simulated listener + phone
python3 -m integration.orchestrator.run_loop                 # you type l / d, or "m s6" for a phone song change
python3 -m unittest rl.tests.test_bandit music.tests.test_playlist integration.tests.test_core_loop -v
```

## Status

Core loop runs end to end with fake inputs: contextual-bandit RL, playlist control with manual override, and a motion stub. Cameras, audio, motors, and the real database are not connected yet.
