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
# v2: engagement. A person who likes a song smiles and moves (nods, sways, leans in, on the beat); one who does not mostly keeps still.
# All are measured against the same person's relaxed pre-roll just before the song, so session-to-session drift in the camera cancels.
ENGAGEMENT_FEATURES = ["activity", "stillness", "beat_power", "beat_amp", "lean", "smile_frac", "smile_peak"]
FEATURES_V2 = FEATURES + ENGAGEMENT_FEATURES
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


PREROLL_SKIP_S = 1.0      # the first second of a pre-roll is ignored (the person has just pressed a key)
BEAT_TOL_HZ = 0.2
STILL_WINDOW_S = 1.0


def _grid_xy(pts, t0, t1):
    """Face-centre x and y resampled to an even grid (None where the face was not seen)."""
    gx = _resample([(t, x) for t, x, _ in pts], t0, t1)
    gy = _resample([(t, y) for t, _, y in pts], t0, t1)
    return gx, gy


def window_speeds(motion, t0, t1, win=STILL_WINDOW_S):
    """RMS head speed (frame-widths per second) in consecutive windows of `win` seconds, over all people."""
    out = []
    per = int(win * GRID_HZ)
    for pts in _per_person(motion, t0, t1).values():
        gx, gy = _grid_xy(pts, t0, t1)
        sp = [None if None in (gx[i], gx[i + 1], gy[i], gy[i + 1]) else
              math.hypot(gx[i + 1] - gx[i], gy[i + 1] - gy[i]) * GRID_HZ for i in range(len(gx) - 1)]
        for a in range(0, len(sp) - per + 1, per):
            seg = [v for v in sp[a:a + per] if v is not None]
            if len(seg) >= per // 2:
                out.append(math.sqrt(sum(v * v for v in seg) / len(seg)))
    return out


def head_activity(motion, t0, t1):
    """Overall head movement: RMS speed of the face centre, averaged over people (0 if nobody was seen long enough)."""
    w = window_speeds(motion, t0, t1)
    return math.sqrt(sum(v * v for v in w) / len(w)) if w else 0.0


def _power_at(x, f, hz=GRID_HZ):
    n = len(x)
    re = sum(v * math.cos(2 * math.pi * f * i / hz) for i, v in enumerate(x))
    im = sum(v * math.sin(2 * math.pi * f * i / hz) for i, v in enumerate(x))
    return (re * re + im * im) / (n * n)


def _taper(run):
    n = len(run)
    xs = list(range(n))
    mx, my = _mean(xs), _mean(run)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, run)) / max(1e-12, sum((x - mx) ** 2 for x in xs))
    return [(y - my - slope * (x - mx)) * (0.5 - 0.5 * math.cos(2 * math.pi * x / (n - 1))) for x, y in zip(xs, run)]


def beat_candidates(tempo_bpm):
    """Beat frequency in Hz and its half- and double-time versions (tempo estimates are often off by a factor of two)."""
    if not tempo_bpm or tempo_bpm <= 0:
        return []
    f0 = tempo_bpm / 60.0
    return [f for f in (f0, f0 / 2, f0 * 2) if 0.5 <= f <= 4.5]


def beat_locked(motion, t0, t1, tempo_bpm):
    """(beat_power, beat_amp): how much stronger head movement is at the song's beat than across the usual 0.3-5 Hz range
    (log of the ratio), and the amplitude of the movement at the beat in frame-widths. Both average over people; 0, 0 if no tempo or too little data."""
    cands = beat_candidates(tempo_bpm)
    if not cands:
        return 0.0, 0.0
    ratios, amps = [], []
    for pts in _per_person(motion, t0, t1).values():
        gx, gy = _grid_xy(pts, t0, t1)
        sigs = []
        for g in (gx, gy):
            run = [v for v in g if v is not None]
            if len(run) >= 30:
                sigs.append(_taper(run))
        if not sigs:
            continue
        ref_f = [0.3 + 0.1 * k for k in range(48)]
        ref = _mean([sum(_power_at(x, f) for x in sigs) for f in ref_f])
        peak = 0.0
        for fc in cands:
            for k in range(-4, 5):
                peak = max(peak, sum(_power_at(x, fc + k * BEAT_TOL_HZ / 4) for x in sigs))
        if ref > 1e-15:
            ratios.append(math.log1p(peak / ref))
            amps.append(math.sqrt(peak))
    return _mean(ratios), _mean(amps)


def mean_width(motion, t0, t1):
    ws = [w for t, _, _, _, w, _ in motion if t0 <= t <= t1]
    return _mean(ws)


def engagement(liking_probs, motion, t0, t1, baseline, tempo_bpm):
    """The v2 engagement features for one clip window, relative to the person's pre-roll baseline."""
    sm = [h for t, h, _ in liking_probs if t0 <= t <= t1]
    base_smile = baseline.get("smile", 0.0)
    win = window_speeds(motion, t0, t1)
    thr = baseline.get("still_thr")
    bp, ba = beat_locked(motion, t0, t1, tempo_bpm)
    bw = baseline.get("width", 0.0)
    return {
        "activity": head_activity(motion, t0, t1) - baseline.get("activity", 0.0),
        "stillness": (sum(1 for v in win if v < thr) / len(win)) if (win and thr is not None) else 0.0,
        "beat_power": bp, "beat_amp": ba,
        "lean": (mean_width(motion, t0, t1) / bw - 1.0) if bw > 0 else 0.0,
        "smile_frac": (sum(1 for h in sm if h > base_smile + 0.15) / len(sm)) if sm else 0.0,
        "smile_peak": (max(sm) - base_smile) if sm else 0.0,
    }


def clip_features(liking, probs, motion, t0, t1, baseline=None, tempo_bpm=None):
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
        **engagement(probs, motion, t0, t1, baseline, tempo_bpm),
        "presence": min(1.0, len(lk) / max(1.0, (t1 - t0) * 10 * 0.5)),
        "fps": len(frames) / max(1, len(persons)) / max(t1 - t0, 1e-6),
        "n_liking": float(len(lk)),
    }
    return feats


def resting_baseline(probs, motion, t0, t1):
    """Smile level and movement while the person holds a neutral face and sits still (calibration)."""
    sm = [h for t, h, _ in probs if t0 <= t <= t1]
    speed, vert = motion_speeds(motion, t0, t1)
    win = sorted(window_speeds(motion, t0, t1))
    still_thr = win[min(len(win) - 1, int(0.75 * len(win)))] if win else None       # 75th percentile of the relaxed window speeds
    return {"smile": _mean(sm), "motion": speed, "vert_motion": vert, "activity": head_activity(motion, t0, t1),
            "width": mean_width(motion, t0, t1), "still_thr": still_thr}


def baseline_from_preroll(probs, motion, preroll, rest=None):
    """Baseline for one clip: its own pre-roll window (start, end) if given, filled in from the session calibration `rest`.
    One function used by the live camera and by offline recomputation, so the two cannot drift apart."""
    base = dict(rest or {})
    if preroll:
        pre = resting_baseline(probs, motion, preroll[0] + PREROLL_SKIP_S, preroll[1])
        if pre.get("activity") is not None and pre.get("width", 0) > 0:
            base.update({k: v for k, v in pre.items() if v is not None})
    return base


def features_from_raw(raw):
    """Recompute a clip's features from its saved raw samples (the JSON written by run_agreement_check --preroll)."""
    probs = [tuple(x) for x in raw["probs"]]
    motion = [tuple(x) for x in raw["motion"]]
    liking = [tuple(x) for x in raw["liking"]]
    base = baseline_from_preroll(probs, motion, raw.get("preroll"), raw.get("rest"))
    return clip_features(liking, probs, motion, raw["clip_start"], raw["t1"], base, tempo_bpm=raw.get("tempo"))
