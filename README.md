# DJ Bot: A Speaker That Learns Your Taste

*DJ Bot: music that listens back.* (The repository is named MakeCU, after the hackathon.)

Built for **MakeCU**, Columbia University Robotics Club's 24-hour hardware hackathon.

A robot that learns what music you (or your group) like by watching how people react, and that physically moves with them while it plays. It picks a song, reads reactions with two cameras, and uses reinforcement learning to choose better songs over time.

## Try It in 30 Seconds (no robot needed)

You only need Python 3.9 or newer; the simulated session uses no extra packages and no camera.

```
git clone https://github.com/cgm2179/MakeCU.git
cd MakeCU
python3 -m integration.orchestrator.run_loop --simulate 25
```

You will see DJ Bot pick a song, read a simulated listener's reaction, learn from it, choose the next song, and print the arm movements it would make for the emotion it saw. A phone changing the song mid-session is simulated too, and it never interrupts the movement. Without the song dataset it uses a built-in list of well-known songs. To run with the real camera, audio and arms, see the camera demo sections below.

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

## Results and Evaluation

How well does DJ Bot work? Every pass mark was written down before the data was collected, and the results include both what worked and what did not yet. Start with the [one-page summary](reports/step10_writeup.md); the [full write-up](reports/step10_writeup_full.md) has every number and the pre-registrations are in `reports/`.

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

## Real Song Data (Spotify Tracks Dataset)

The song list comes from the Hugging Face dataset `maharshipandya/spotify-tracks-dataset` (about 114k rows, 80k unique tracks after cleaning). It is not stored in git. Download it once, then everything runs offline:

```
python3 -m music.catalog.download_spotify     # saves data/spotify/dataset.csv (20 MB)
```

Without the CSV the code falls back to a small hardcoded song list. The first filter maps a mood or activity to a target feature vector (valence and energy are the main axes) and keeps the 500 nearest songs. Moods: `sad`, `happy`, `gym`, `study`, `party`, `chill` (plus aliases like `workout`, `studying`). See `music/filter/mood_targets.py`.

```
python3 -m integration.orchestrator.run_mood_demo --mood gym --rounds 60 --likes punk-rock,emo
python3 -m integration.orchestrator.run_mood_demo --mood gym --likes punk-rock,emo --sweep 24   # many seeds
```

## Songs Not in the Dataset (Gemini lookup)

For a song given by title and artist, `music/catalog/song_lookup.py` first looks in the Spotify dataset (fast path, handles "- Remastered", "feat." and similar variants). Only if it is not there does it ask Google Gemini for the song's genre, mood, energy and valence (plus danceability and tempo), which are mapped into our song schema (genre restricted to the dataset's 114 tags) so the bandit and mood filter use it like any other song. Results are cached in `data/gemini_cache.json` (git-ignored), including "Gemini doesn't know this song", so a song is looked up once. A song Gemini cannot identify, or is not confident about, is skipped cleanly. These are Gemini's rough estimates, not measured audio features, and looked-up songs have no popularity score (so the mainstream bias leaves them alone).

Setup: put `GEMINI_API_KEY=...` in a `.env` file in the repo root (git-ignored), or export it. Only the title and artist are sent. Override the model with `GEMINI_MODEL`. Default model `gemini-3.5-flash`; Google retires models quickly (2.5 already returns 404 for new users), so if you see a 404 pick a current one from the model list (`GEMINI_MODEL=...`). The free tier has a small per-day request quota per model; the cache means each song costs one call, and a quota error is reported with its retry time and is not cached. Gemini's estimates vary slightly between calls (e.g. valence 0.63 to 0.84 for one song), so the first answer is cached and kept.

```
python3 -m integration.orchestrator.run_lookup_demo      # a few well-known songs not in the dataset
python3 -m integration.orchestrator.run_lookup_demo --songs "Billie Jean|Michael Jackson"
```

## Mainstream Bias (well-known songs first)

On the database (Spotify dataset) path, the first filter and the early picks lean toward well-known songs using the dataset's popularity score, so a mood like gym opens with recognizable tracks. Strength runs from 0 (off) to 1 (strong); default 0.5, set in `shared/config/defaults.py` or per run with `--mainstream-bias`. It has two parts: the mood pool is ranked by closeness to the target *plus* popularity, and the bandit adds a popularity prior that fades over its first ~10 rounds as it learns your taste. Exploration is kept, so less-known songs still appear and the bandit can find a niche taste. The pool is always topped up to full size, even for niche moods. It never applies to a user's own playlist: pools are tagged with their source (`database`, `hardcoded`, later `playlist`) and only `database` pools turn it on.

## Real Audio Playback (iTunes previews)

`music/playback/audio.py` looks each chosen song up on the free iTunes Search API (no key; only the title and artist are sent), downloads its 30-second preview and plays it through the computer speaker (`afplay` on macOS, or `ffplay`/`mpv`). Playback is non-blocking, so the motion system is not held up. Songs with no matching preview are skipped and the bandit picks again; about 7 in 10 songs in the gym pool have one.

```
python3 -m integration.orchestrator.run_playback_demo --mood gym --songs 3 --clip-seconds 10
```

## Camera Feedback (mock mode)

The loop can take its reactions from the camera package (`perception/perception_cv`) instead of the fake listener. The bridge in `perception/emotion/camera_reaction.py` lets the loop import it from the repo root, starts the perception service in a background thread, and reads `get_group_reaction(song_id)` at the end of each clip. If nobody is in view (confidence near 0), that round's learning step is skipped. Use a Python that has the camera deps (see `perception/perception_cv/README.md`):

```
python3 -m integration.orchestrator.run_camera_demo --mood gym --songs 4            # random mock faces
python3 -m integration.orchestrator.run_camera_demo --mood gym --songs 24 --taste   # faces react to the song
python3 -m integration.orchestrator.run_camera_demo --songs 4 --taste --audio --empty 3   # real previews, nobody in view in round 3
```

Without `feedback=...` the loop uses the fake listener as before, so the simulation tests are unchanged.

### Live camera, music, and Arduino emotion arms

Flash `hardware/firmware/robot_combined/robot_combined.ino` to the ESP32 and close Arduino IDE's Serial Monitor. With the camera dependencies installed in the active Python environment, run from the repository root:

```powershell
python -m integration.orchestrator.run_camera_demo --real --camera 0 --detector yunet --mood gym --songs 4 --audio --clip-seconds 10 --arm-port COM5
```

Replace `COM5` with the ESP32 port shown in Device Manager and `0` with the webcam index if needed. The live preview overlays each face's detected emotion and liking, plus a rolling average labeled as enjoying, not enjoying, or neutral toward the current song genre. Press `q` or Esc in the preview to close its window; the demo continues. Omit `--arm-port` to run without an Arduino. Audio previews need internet and play through the computer's default output; songs without a matching preview are skipped.

## Live Camera → RL

Start the live camera-to-song-picker loop from the repository root:

```bash
python -m integration.orchestrator.run_loop --live --camera 2 --detector yunet
```

The camera preview opens. The loop picks a song, samples the camera reaction, then prints the RL reward and arm profile for the detected emotion. If group liking is at or below **-0.2**, it reports discouraging the song and its genre. To run the ESP32 steppers and obstacle-drive motors at the same time, flash and use the combined firmware in the section below.

### Run camera/RL, emotion steppers, and obstacle drive together

The combined ESP32 firmware is `hardware/firmware/robot_combined/robot_combined.ino`. Flash this **single sketch** to the ESP32 in Arduino IDE; do not flash the standalone obstacle-drive or stepper-test sketches for this combined setup. It drives the motor-shield outputs forward when the ultrasonic sensor detects an object closer than 30 cm and stops otherwise, while accepting emotion-based stepper-arm commands over USB serial. Keep wheels raised or the robot secured during the first test.

After uploading, close Arduino IDE's Serial Monitor so Python can open the ESP32 COM port. From the repository root in PowerShell, run:

```powershell
python -m integration.orchestrator.run_loop --live --camera 0 --detector yunet --arm-port COM5
```

Replace `COM5` with the ESP32 port shown in Windows Device Manager under **Ports**, and use the correct camera index if the computer camera is not `0`. Install `pyserial` in the active Python environment if needed. Emotion commands control the steppers; the wheel drive remains controlled by the distance threshold and is not affected by emotion.

Power the stepper and drive motors from suitable external supplies and connect their grounds to ESP32 GND. An HC-SR04-style ECHO may output 5 V; ESP32 GPIOs are not 5 V tolerant, so use a level shifter or resistor divider before GPIO 18.

## Try It (fake inputs, no hardware)

```
python3 -m integration.orchestrator.run_loop --simulate 25   # simulated listener + phone
python3 -m integration.orchestrator.run_loop                 # you type l / d, or "m s6" for a phone song change
python3 -m unittest rl.tests.test_bandit music.tests.test_playlist music.tests.test_dataset_pipeline music.tests.test_audio integration.tests.test_core_loop -v
```

## Status

The fake-input core loop supports contextual-bandit RL, playlist control with manual override, and a motion stub. Applying `cv-rl-arms.patch` adds the live camera-to-RL flow. The combined ESP32 firmware adds emotion-driven stepper arms and ultrasonic-triggered drive; audio playback and camera-based pose tracking are separate integrations.
