"""Per-clip features from the camera: how the face reacted AND how it moved during a song.

Pure Python (no numpy/OpenCV) so it can be tested anywhere. Inputs are the raw samples that
CameraFeedback records while a clip plays:

    liking  : [(t, calibrated_liking)]                       group liking over time
    probs   : [(t, happy, surprised)]                        mean emotion probabilities of the tracked faces
    motion  : [(t, person_id, cx, cy, w, h)]                 each face's centre and size per analysed frame (0..1)

Why movement: a smile is rare when someone listens alone, but head bobbing, nodding on the beat and
leaning in change how the face moves even when the expression stays neutral. Features are measured
against the person's own resting baseline (taken during calibration) wherever possible.
"""
from __future__ import annotations

import math
import statistics

# Order matters: this is the feature vector the reliability model is fitted on.
LIKING_FEATURES = ["mean_liking", "peak_liking", "low_liking", "range_liking", "build_liking", "last3_liking"]
FACE_FEATURES = ["smile"]
MOTION_FEATURES = ["motion", "vert_motion", "bob_share", "bob_amp"]
FEATURES = LIKING_FEATURES + FACE_FEATURES + MOTION_FEATURES
DIAGNOSTICS = ["presence", "fps", "n_liking"]   # logged, not used by the model

GRID_HZ = 10.0
BOB_BAND = (1.0, 3.5)     # head-bob / nod rhythm in Hz (about 60-210 beats per minute)
TOTAL_BAND = (0.1, 5.0)


def _mean(xs, default=0.0):
    return sum(xs) / len(xs) if xs else default


def _per_person(motion, t0, t1):
    by = {}
    for t, pid, cx, cy, w, h in motion:
        if t0 <= t <= t1:
            by.setdefault(pid, []).append((t, cx, cy))
    return {pid: sorted(v) for pid, v in by.items() if len(v) >= 3}


def motion_speeds(motion, t0, t1):
    """(horizontal+vertical speed, vertical-only speed) of the face centre, averaged over people,
    in frame-widths per second."""
    speeds, vert = [], []
    for pts in _per_person(motion, t0, t1).values():
        dur = pts[-1][0] - pts[0][0]
        if dur <= 0:
            continue
        speeds.append(sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(pts, pts[1:])) / dur)
        vert.append(sum(abs(b[2] - a[2]) for a, b in zip(pts, pts[1:])) / dur)
    return _mean(speeds), _mean(vert)


def _resample(pts, t0, t1, hz=GRID_HZ):
    """Linear interpolation of (t, y) onto an even grid; None outside the span the face was seen."""
    n = int((t1 - t0) * hz)
    out, j = [], 0
    for i in range(n):
        t = t0 + i / hz
        if t < pts[0][0] or t > pts[-1][0]:
            out.append(None)
            continue
        while j + 1 < len(pts) - 1 and pts[j + 1][0] < t:
            j += 1
        (ta, ya), (tb, yb) = pts[j], pts[min(j + 1, len(pts) - 1)]
        out.append(ya if tb == ta else ya + (yb - ya) * (t - ta) / (tb - ta))
    return out


def _band_power(sig, lo, hi, hz=GRID_HZ):
    n = len(sig)
    mean = _mean(sig)
    x = [v - mean for v in sig]
    total = 0.0
    for k in range(1, n // 2 + 1):
        f = k * hz / n
        if not (lo <= f <= hi):
            continue
        re = sum(v * math.cos(2 * math.pi * k * i / n) for i, v in enumerate(x))
        im = sum(v * math.sin(2 * math.pi * k * i / n) for i, v in enumerate(x))
        total += (re * re + im * im) / (n * n)
    return total


def bobbing(motion, t0, t1):
    """(share of vertical head movement that is in the 1-3.5 Hz 'nodding' band, amplitude of that band),
    averaged over people. share is 0..1; amp is in frame-widths."""
    shares, amps = [], []
    for pts in _per_person(motion, t0, t1).values():
        grid = _resample([(t, cy) for t, _, cy in pts], t0, t1)
        run = [v for v in grid if v is not None]
        if len(run) < 20:
            continue
        # remove a straight-line drift (posture, leaning) then taper the ends (Hann window) so slow swaying
        # and the edges of the clip do not leak into the nodding band
        n = len(run)
        xs = list(range(n))
        mx, my = _mean(xs), _mean(run)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, run)) / max(1e-12, sum((x - mx) ** 2 for x in xs))
        detr = [(y - my - slope * (x - mx)) * (0.5 - 0.5 * math.cos(2 * math.pi * x / (n - 1))) for x, y in zip(xs, run)]
        total = _band_power(detr, *TOTAL_BAND)
        band = _band_power(detr, *BOB_BAND)
        shares.append(band / total if total > 1e-12 else 0.0)
        amps.append(math.sqrt(band))
    return _mean(shares), _mean(amps)


def clip_features(liking, probs, motion, t0, t1, baseline=None):
    """Features for one clip window [t0, t1]. Returns None if too little was seen to say anything."""
    baseline = baseline or {}
    lk = [(t, v) for t, v in liking if t0 <= t <= t1]
    if len(lk) < 5:
        return None
    vals = [v for _, v in lk]
    third = max(1, len(vals) // 3)
    last3 = [v for t, v in lk if t >= t1 - 3.0] or vals[-third:]
    sm = [h for t, h, _ in probs if t0 <= t <= t1]
    speed, vert = motion_speeds(motion, t0, t1)
    share, amp = bobbing(motion, t0, t1)
    frames = [m for m in motion if t0 <= m[0] <= t1]
    persons = {m[1] for m in frames}
    feats = {
        "mean_liking": _mean(vals), "peak_liking": max(vals), "low_liking": min(vals),
        "range_liking": max(vals) - min(vals),
        "build_liking": _mean(vals[-third:]) - _mean(vals[:third]),   # did the reaction grow through the clip
        "last3_liking": _mean(last3),                                   # what the camera check used before
        "smile": _mean(sm) - baseline.get("smile", 0.0),
        "motion": speed - baseline.get("motion", 0.0),
        "vert_motion": vert - baseline.get("vert_motion", 0.0),
        "bob_share": share, "bob_amp": amp,
        "presence": min(1.0, len(lk) / max(1.0, (t1 - t0) * 10 * 0.5)),
        "fps": len(frames) / max(1, len(persons)) / max(t1 - t0, 1e-6),
        "n_liking": float(len(lk)),
    }
    return feats


def resting_baseline(probs, motion, t0, t1):
    """Smile level and movement while the person holds a neutral face and sits still (calibration)."""
    sm = [h for t, h, _ in probs if t0 <= t <= t1]
    speed, vert = motion_speeds(motion, t0, t1)
    return {"smile": _mean(sm), "motion": speed, "vert_motion": vert}
