"""Shared tunable defaults (see AGENTS.md)."""

# Mainstream bias for the database (Spotify dataset) path, from 0 (off) to 1 (strong).
# It makes the first filter and the early picks lean toward well-known songs (popularity score).
# It never applies to a user's own playlist.
MAINSTREAM_BIAS = 0.5
