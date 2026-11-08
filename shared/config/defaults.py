"""Shared tunable defaults (see AGENTS.md)."""

# Mainstream bias for the database (Spotify dataset) path, from 0 (off) to 1 (strong).
# It makes the first filter and the early picks lean toward well-known songs (popularity score).
# It never applies to a user's own playlist.
MAINSTREAM_BIAS = 0.5

# Demo / live show: how long each song stays on before the loop advances to the next one.
# iTunes previews are ~30 s; keep this <= 30 so the clip finishes with the song.
# Change this number to make the demo faster or slower.
DEMO_SONG_SECONDS = 30
