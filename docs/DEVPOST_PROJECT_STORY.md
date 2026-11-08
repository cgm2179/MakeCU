## Inspiration

Every speaker plays what you ask for. A great DJ does something harder: they read the room. They see who lights up, who drifts, and who is nodding along, and they change the music before anyone has to say a word. We wanted to build that instinct into a robot. At MakeCU, Columbia's 24-hour hardware hackathon, we set out to make **DJ Bot**: a speaker that watches how you react and learns what you love.

## What it does

DJ Bot plays a song, watches the listeners with a camera, and turns what it sees into better music choices.

- **Reads the room.** A camera estimates each person's emotion and how much they seem to like the song, then combines them into one group signal, so it can read a whole group and not just one person.
- **Picks the next song.** A learning system uses that reaction to choose what plays next, starting from a mood or event like "gym" or "party".
- **Performs.** Its stepper-motor arms move with the emotion it detects, so the music comes with a performer.
- **Stays in control.** Music and movement run as separate processes, so changing a song never interrupts the robot's motion.

## How we built it

- **Perception.** Python, OpenCV and a YuNet face detector feed a DeepFace emotion model through a FastAPI service. A calibration step gives the system a resting baseline, and group aggregation turns many faces into one reaction.
- **Learning.** The reward for a song is the group's liking weighted by how confident the camera was: $r = \bar{\ell}\cdot\bar{c}$. For brand-new listeners we also built a learner that starts from a model trained on **19 million plays from 992 Last.fm listeners**, then adapts from feedback: $\text{score}(s) = \text{prior}(s) + x_s^{\top}\delta + \beta\,\sigma(s)$, where the correction $\delta$ is learned online and $\sigma$ is the model's uncertainty.
- **Music.** Song features come from the Spotify Tracks dataset (about 80,000 songs), 30-second previews come from the iTunes Search API, and a Gemini lookup estimates features for songs outside the catalog.
- **Hardware.** An ESP32 runs the stepper arms and an obstacle-aware drive base. The song-picking model is **under 2 KB**, small enough to fit a microcontroller's memory.
- **Engineering.** More than 250 automated tests, a replay simulator for offline policy testing, and models that are frozen and hashed so results are reproducible.

## Challenges we ran into

- **Faces at a speaker are subtle.** People rarely frown at a song they don't like; they just go still. Calibration also drifted between sessions. We responded by measuring each person against a short relaxed pre-roll before every song, saving raw samples so features can be re-analysed, and building features for head movement locked to the song's beat.
- **A cold start.** A new listener has no history. Training a base model on other people's listening gave DJ Bot something sensible to start from.
- **Honest evaluation on small data.** We wrote our pass marks down before looking at results and tested with real listeners on songs they had never heard, so we could trust what we measured.
- **24 hours of integration.** Cameras, learning, music and motors had to work together. We switched from servos to steppers and kept music and motion independent so one could never break the other.

## Accomplishments that we're proud of

- A working end-to-end robot: camera, learning, music and moving arms from a single command.
- A base model that predicted held-out strangers' likes better than a popularity ranking, and fits in under 2 KB.
- In simulation, the new online learner improves within a single session.
- Live testing with real listeners on more than 300 songs, and over 250 automated tests.

## What we learned

- **Measurement matters as much as the model.** A stable, well-calibrated camera signal is worth more than a cleverer learner.
- **Popularity is a strong baseline,** so the right design is a smart starting point that adapts, not learning from scratch.
- **Body language carries information that facial expression alone doesn't:** movement, stillness and nodding on the beat.
- **Writing the success criteria first** keeps you honest, and makes the results worth trusting.

## What's next for DJ Bot

- **Your own playlists,** so DJ Bot can reorder and skip within songs you already love.
- **Body-language learning.** Validating the movement-and-beat features across sessions and across people, then letting them guide the music.
- **Running the song model on the ESP32 itself,** and a phone app for control.
