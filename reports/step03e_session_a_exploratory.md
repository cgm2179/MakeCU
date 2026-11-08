# Step 3e, session A (adi_v2a): first look with the v2 measurement (exploratory; the pre-registered test needs a second session)

Date: 2026-11-08. 60 songs rated, 57 usable (20 liked, 37 disliked), `--preroll 4`, quiet room, songs never rated or thumbed before.
**This is not the pre-registered result.** The test in `step03e_preregistration.md` (leave-one-session-out, AUC >= 0.65 on every held-out session) needs session B on a different occasion.
Nothing here changes the camera's status: it stays off as a reward.

## Did the new measurement work?
Yes. All 60 clips have a saved raw file with a pre-roll (median 4.3 s) and a tempo. The pre-roll baseline itself moved during the session (head activity 0.072, 0.035, 0.052 across thirds, sd 0.057 over songs),
which is what a per-song baseline is for: a single baseline taken at the start would have been wrong for much of the session.

## Engagement features in this session (means, and AUC for liked vs disliked inside the session)
| feature | liked | disliked | AUC |
|---|---|---|---|
| activity (head movement above the pre-roll) | +0.038 | -0.009 | 0.762 |
| stillness (share of quiet 1 s windows) | 0.430 | 0.754 | 0.231 (i.e. 0.769 the other way: disliked songs are stiller) |
| beat_amp (movement amplitude at the song's beat) | 0.002 | 0.001 | 0.776 |
| beat_power | 1.585 | 1.526 | 0.526 |
| lean | -0.010 | -0.013 | 0.428 |
| smile_frac | 0.086 | 0.081 | 0.453 |
| smile_peak | 0.338 | 0.423 | 0.415 |

Liked songs show more head movement and less stillness, as expected. **Smile did not separate them here** (it is at chance or slightly inverted), and lean did not either.
Within-session cross-validated AUC: engagement features 0.738 (old reaction-shape features on the same songs 0.639).

## Is it just the songs?
Movement correlates only weakly with song properties (activity: tempo +0.04, energy +0.15, popularity +0.10, danceability +0.20; stillness and beat_amp similar, all |r| <= 0.20).
Within-session CV AUC: song model alone 0.778, engagement alone 0.744, both together **0.813** (+0.035; with only activity, stillness and beat_amp: 0.815).
On these songs the Step 7 model scores 0.827 and popularity 0.836, so the songs predict the thumbs well and the camera adds a small amount at most.

## How much to believe it
- 57 songs, 20 liked: an AUC is known to about +/-0.1, and the +0.035 gain is well inside that.
- The earlier camera result looked just as good inside its own session (CV AUC 0.795) and then did not replicate (0.556; within the next session 0.461). Only a second session on a different occasion can tell these cases apart.
- The three movement features were singled out after seeing this session, so they are a hypothesis for session B, not a result. The pre-registered feature set is all seven.

## Next
Session B on a different occasion (another day, or re-seated and re-lit), then `python3 -m rl.camera_validation --sessions adi_v2a adi_v2b`.
