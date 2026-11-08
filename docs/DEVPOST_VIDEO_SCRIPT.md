# DJ Bot: Devpost video script (target 2:45)

The voiceover is about 280 words, roughly 1:55 of speech at a confident pace. The rest of the 2:45 is the live demo and short music beats, where nobody is talking, so let the robot and the music breathe. Read it out loud with a timer before recording; Devpost asks for 2 to 3 minutes. If it runs short, hold longer on the demo.

| Time | On screen | Voiceover |
|---|---|---|
| **0:00 - 0:12** Hook | Close-up of the robot facing the camera, music playing, arms moving. Title card: **DJ Bot: music that listens back.** | "Every speaker plays what you ask for. DJ Bot plays what you actually enjoy, because it watches your face, learns from your reaction, and moves with the music." |
| **0:12 - 0:40** What it is | Wide shot: robot, speaker, camera, arms. Quick cuts of each part. Text: **Built in 24 hours at MakeCU.** | "DJ Bot is a robot speaker, built in twenty-four hours at MakeCU. A camera reads how you react. Reinforcement learning turns that reaction into better song picks. And its arms move with the emotion it sees, so your music comes with a performer. The whole thing starts from a single command." |
| **0:40 - 1:15** Live demo | Screen recording of the terminal beside the robot: the "now playing" line, the arms line, the "queuing more..." line. Person reacting on camera. | "Watch. The robot picks a song from a gym mood, reads my reaction, and moves its arms to match. Then it chooses what to play next from what it just saw. Every reaction teaches it something. And it can read a whole group at once, not just one person." |
| **1:15 - 1:55** A smart start | Simple diagram: Last.fm listeners -> base model -> small personal adjustment -> next song. Numbers on screen: **992 listeners · 19 million plays**. Then a rising line labelled "gets better within a session". | "The hardest moment is the first song, when it knows nothing about you. So we gave it a head start: a model trained on nineteen million plays from nearly a thousand Last.fm listeners. On people it had never seen, it predicted what they'd like better than a popularity ranking. And from there it keeps learning. In simulation, it gets noticeably better within a single session." |
| **1:55 - 2:20** Built to last | Quick montage: code, a test run passing, a tiny file. On screen: **under 2 KB · 259 automated tests · pass marks written down in advance.** | "It's tiny. The whole song-picking model fits in under two kilobytes, small enough for a microcontroller. We tested it with real listeners on songs they'd never heard, wrote our success criteria down before we looked at any results, and backed it with more than two hundred and fifty automated tests." |
| **2:20 - 2:38** What's next | Short clip of someone nodding along to a song. Text: **Body language, not just expressions.** | "And we're taking it further. We built tools that measure whether you're nodding along to the beat, so DJ Bot can learn from your body language, not just your face." |
| **2:38 - 2:50** Close | Robot playing, person smiling. End card: team names, GitHub link. | "DJ Bot. Music that listens back. Thanks for watching." |

## Recording notes

- **Record the demo with the real command**, once Sabaa's PR #24 is merged or on her branch: `python -m integration.orchestrator.run_camera_demo --real --camera 0 --detector yunet --mood gym --songs 4 --audio --clip-seconds 10 --arm-port COM3`. Do a dry run and confirm that the arms and the terminal lines you show are really happening.
- **Show what you say.** The script claims: the camera reads the face, the arms move with the emotion, the next song is chosen from that reaction, it can read several people at once, the base model starts from Last.fm data, it is under 2 KB, and there are 259 tests. Each of these is true and checkable in the repo.
- **Numbers to put on screen:** 992 listeners and 19 million plays (Last.fm 1K); under 2 KB (1,914 bytes with the personal layer, 1,477 for the base); 259 automated tests.
- **Music rights:** the audio is 30-second iTunes previews. Keep the volume low under your voice, or caption the song titles instead of letting full audio run, in case the video host flags it.
- **Faces:** only film people who agreed to be on video; the camera data stays on your computer.
- **Credits card:** team names and roles, plus the GitHub repo.

## If you need a 2:00 version

Drop the "What's next" row and shorten "Built to last" to one sentence. That trims about 35 seconds.
