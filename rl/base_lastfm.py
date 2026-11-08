"""Base model v2 (checkpoint L1.3): trained on other people's listening (Last.fm 1K), definitions fixed in
reports/step06_preregistration.md before the data was opened.

    python3 -m rl.base_lastfm            # reads data/external/lastfm1k/lastfm-dataset-1K.tar.gz

Steps: aggregate plays per (user, artist, track) -> match to the song catalog by cleaned title and artist -> "liked" = a user
played a matched song at least 3 times -> known set K (matched songs played by 5+ users) -> split users 70/15/15 ->
weighted logistic regression on song features (P that a random user likes the song) with checkpoints -> test users read once.
Also fits q(song): ridge regression of log(number of listeners) on the same features (the "objective" head).
Needs numpy and pandas (see requirements-camera.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from music.catalog.names import artist_keys, clean_title
from rl.base_model import Featurizer, training_graph_svg

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "lastfm1k"
MODEL_DIR = ROOT / "models"
TSV_SUFFIX = "userid-timestamp-artid-artname-traid-traname.tsv"
MIN_USERS_MATCH = 5       # pre-registered
LIKE_PLAYS = 3            # pre-registered
MIN_LIKED = 10            # pre-registered
MATCH_PASS = 0.30         # pre-registered
AUC_GAIN_PASS = 0.03      # pre-registered
Q_CORR_PASS = 0.30        # pre-registered
FORMAT = "makecu-base-model"
VERSION = 2


# ---------------------------------------------------------------- reading and matching

def aggregate_plays(tar_path, chunk=2_000_000, log=print):
    """DataFrame (user, artist, track, plays) from the Last.fm 1K tarball, streamed so memory stays small."""
    parts, rows, t0 = [], 0, time.time()
    with tarfile.open(tar_path, "r:gz") as tf:
        member = next(m for m in tf if m.name.endswith(TSV_SUFFIX))
        reader = pd.read_csv(tf.extractfile(member), sep="\t", header=None, usecols=[0, 3, 5],
                             names=["user", "artist", "track"], quoting=3, on_bad_lines="skip", dtype=str,
                             encoding_errors="replace", chunksize=chunk)
        for df in reader:
            df = df.dropna()
            rows += len(df)
            parts.append(df.groupby(["user", "artist", "track"], sort=False).size())
            if len(parts) >= 4:
                parts = [pd.concat(parts).groupby(level=[0, 1, 2]).sum()]
            if log:
                log(f"  read {rows:,} listening events ({time.time() - t0:.0f}s)")
    out = pd.concat(parts).groupby(level=[0, 1, 2]).sum().rename("plays").reset_index()
    return out


def build_catalog_index(songs):
    """(clean artist, clean title) -> song_id; if several songs share a key the most popular wins."""
    idx = {}
    for s in sorted(songs, key=lambda s: s.audio_features.get("popularity", 0.0)):
        t = clean_title(s.title)
        if not t:
            continue
        for a in artist_keys(s.artist):
            idx[(a, t)] = s.song_id
    return idx


def match_pairs(pairs, idx):
    """{(artist, track): song_id} for the Last.fm pairs that match a catalog song."""
    out = {}
    for a, t in pairs:
        ct = clean_title(t)
        if not ct:
            continue
        for ak in artist_keys(a):
            sid = idx.get((ak, ct))
            if sid:
                out[(a, t)] = sid
                break
    return out


def match_stats(plays, mapping, min_users=MIN_USERS_MATCH):
    """Match rate on distinct pairs played by >= min_users users (the pre-registered pass mark) and on all events."""
    pair_users = plays.groupby(["artist", "track"])["user"].nunique()
    eligible = pair_users[pair_users >= min_users]
    n_match = sum(1 for k in eligible.index if k in mapping)
    pair_plays = plays.groupby(["artist", "track"])["plays"].sum()
    ev_match = sum(v for k, v in pair_plays.items() if k in mapping)
    return {"pairs_eligible": int(len(eligible)), "pairs_matched": int(n_match),
            "match_rate": n_match / len(eligible) if len(eligible) else 0.0,
            "events_total": int(pair_plays.sum()), "events_matched": int(ev_match),
            "event_match_rate": ev_match / pair_plays.sum() if len(pair_plays) else 0.0,
            "pairs_all": int(len(pair_plays))}


def build_matrix(plays, mapping, min_users=MIN_USERS_MATCH, like_plays=LIKE_PLAYS, min_liked=MIN_LIKED):
    """Returns (users, song_ids, liked) where liked[u, s] is True if user u played song s (in the known set K) at least
    like_plays times. Users with fewer than min_liked liked songs are dropped. Also returns listeners[s] = distinct users."""
    map_df = pd.DataFrame([(a, t, sid) for (a, t), sid in mapping.items()], columns=["artist", "track", "song_id"])
    m = plays.merge(map_df, on=["artist", "track"]).groupby(["user", "song_id"])["plays"].sum().reset_index()
    listeners = m.groupby("song_id")["user"].nunique()
    known = listeners[listeners >= min_users].index
    m = m[m["song_id"].isin(known)]
    song_ids = sorted(known)
    sidx = {s: i for i, s in enumerate(song_ids)}
    liked_rows = m[m["plays"] >= like_plays]
    per_user = liked_rows.groupby("user").size()
    users = sorted(per_user[per_user >= min_liked].index)
    uidx = {u: i for i, u in enumerate(users)}
    liked = np.zeros((len(users), len(song_ids)), dtype=bool)
    for u, s in zip(liked_rows["user"], liked_rows["song_id"]):
        if u in uidx:
            liked[uidx[u], sidx[s]] = True
    return users, song_ids, liked, np.array([listeners[s] for s in song_ids], dtype=float)


def split_users(n_users, seed=0, fractions=(0.70, 0.15, 0.15)):
    idx = np.random.RandomState(seed).permutation(n_users)
    a = int(round(n_users * fractions[0]))
    b = a + int(round(n_users * fractions[1]))
    return idx[:a], idx[a:b], idx[b:]


# ---------------------------------------------------------------- metrics

def weighted_auc(scores, pos_w, neg_w):
    """P(a liked (song, user) pair outranks a not-liked one), ties count half, pooled over users."""
    order = np.argsort(scores, kind="mergesort")
    s, pw, nw = scores[order], pos_w[order], neg_w[order]
    _, start = np.unique(s, return_index=True)
    pg, ng = np.add.reduceat(pw, start), np.add.reduceat(nw, start)
    below = np.cumsum(ng) - ng
    den = pg.sum() * ng.sum()
    return float(((pg * below).sum() + 0.5 * (pg * ng).sum()) / den) if den > 0 else float("nan")


def user_auc(scores, liked_row):
    """AUC for one user: their liked songs against their other songs in K (ties count half)."""
    p = int(liked_row.sum())
    n = len(liked_row) - p
    if p == 0 or n == 0:
        return float("nan")
    r = pd.Series(scores).rank(method="average").to_numpy()
    return float((r[liked_row].sum() - p * (p + 1) / 2) / (p * n))


def mean_user_auc(scores, liked, rows):
    vals = [user_auc(scores, liked[i]) for i in rows]
    return np.array([v for v in vals if not math.isnan(v)])


def bootstrap_mean_diff(a, b, n=2000, seed=0):
    """95% range of mean(a - b) resampling users."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- training

def song_weights(liked, rows):
    pos = liked[rows].sum(axis=0).astype(float)
    return pos, len(rows) - pos


def fit_preference(X, tr_pos, tr_neg, va_pos, va_neg, epochs=600, lr=1.0, lam=1e-4, checkpoint_every=50,
                   on_checkpoint=None, log=print, log_every=100):
    """Weighted logistic regression by full-batch gradient descent. Returns (best, last, history, checkpoint_epochs)."""
    d = X.shape[1]
    w = np.zeros(d)
    b = math.log(tr_pos.sum() / tr_neg.sum())          # start at the overall like rate
    t_tot, v_tot = (tr_pos + tr_neg).sum(), (va_pos + va_neg).sum()
    tw, vw = tr_pos + tr_neg, va_pos + va_neg
    history, ckpts, best = [], [], (float("inf"), None, 0)
    t0 = time.time()

    def loss(p, pos, neg, tot):
        return float(-(pos * np.log(np.maximum(p, 1e-12)) + neg * np.log(np.maximum(1 - p, 1e-12))).sum() / tot)
    for ep in range(epochs + 1):
        z = X @ w + b
        p = sigmoid(z)
        pv = p  # same songs; only the weights differ between train and validation users
        row = {"epoch": ep, "train_loss": loss(p, tr_pos, tr_neg, t_tot), "val_loss": loss(pv, va_pos, va_neg, v_tot),
               "train_auc": weighted_auc(z, tr_pos, tr_neg), "val_auc": weighted_auc(z, va_pos, va_neg)}
        history.append(row)
        if row["val_loss"] < best[0]:
            best = (row["val_loss"], (w.copy(), b), ep)
        if ep % checkpoint_every == 0:
            ckpts.append(ep)
            if on_checkpoint:
                on_checkpoint(ep, w.copy(), b, row)
        if log and (ep % log_every == 0 or ep == epochs):
            log(f"  epoch {ep:>4}/{epochs}  train loss {row['train_loss']:.5f}  val loss {row['val_loss']:.5f}  "
                f"val AUC {row['val_auc']:.3f}  ({time.time() - t0:.1f}s)")
        if ep == epochs:
            break
        err = (tw * p - tr_pos) / t_tot
        w -= lr * (X.T @ err + lam * w)
        b -= lr * err.sum()
    return best, (w, b), history, ckpts


def fit_objective(X, listeners, lam=10.0):
    """Ridge regression of log(listeners) on the features: q(song). Returns (weights, intercept)."""
    y = np.log1p(listeners)
    ybar = y.mean()
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ (y - ybar))
    return w, float(ybar)


# ---------------------------------------------------------------- file format

def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def save_base(path, feat, w, b, q_w, q_b, training, checkpoints=None):
    """One self-contained JSON: preference head + objective head + scaling + a hash over both. Written read-only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"feature_names": feat.names, "means": feat.means, "sds": feat.sds, "genres": feat.genres,
               "preference": {"weights": [round(float(x), 8) for x in w], "bias": round(float(b), 8)},
               "objective_q": {"weights": [round(float(x), 8) for x in q_w], "intercept": round(float(q_b), 8)}}
    doc = {"format": FORMAT, "version": VERSION, "base_model_id": "base-v2-lastfm1k", "saved_at": time.time(),
           "model": payload, "weights_sha256": _hash(payload), "training": training, "checkpoints": checkpoints or []}
    if path.exists():
        os.chmod(path, 0o644)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    os.chmod(path, 0o444)           # the frozen base is read-only on disk
    return path


class FrozenBase:
    """A loaded base model. Arrays are read-only; to build on it, take a copy with `.preference_weights()`."""

    def __init__(self, doc):
        m = doc["model"]
        self.id, self.doc = doc["base_model_id"], doc
        self.feat = Featurizer(m["means"], m["sds"], m["genres"])
        self._w = np.array(m["preference"]["weights"])
        self._w.setflags(write=False)
        self.bias = m["preference"]["bias"]
        self._qw = np.array(m["objective_q"]["weights"])
        self._qw.setflags(write=False)
        self.q_intercept = m["objective_q"]["intercept"]
        self.sha256 = doc["weights_sha256"]

    def preference_weights(self):
        return self._w.copy()

    def _x(self, song):
        return np.array(self.feat.vector(song))

    def preference_logit(self, song):
        return float(self._x(song) @ self._w + self.bias)

    def prob(self, song):
        return float(sigmoid(self.preference_logit(song)))

    def quality(self, song):
        return float(self._x(song) @ self._qw + self.q_intercept)


def load_base(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT or doc.get("version") != VERSION:
        raise ValueError(f"{path} is not a {FORMAT} v{VERSION} file")
    if _hash(doc["model"]) != doc["weights_sha256"]:
        raise ValueError(f"{path}: weights do not match their hash (file changed or damaged)")
    return FrozenBase(doc)


# ---------------------------------------------------------------- ESP32 export

ESP32_RAM_BYTES = 520 * 1024      # total SRAM of an ESP32 (hard cap)
ESP32_BUDGET_BYTES = 32 * 1024    # working budget (pre-registered)
MAGIC = b"MCUB"


def device_blob(feat, w, b, q_w, q_b):
    """Compact binary for the device. Layout: magic(4) version(u16) n_numeric(u8) n_mood(u8) n_genre(u8) pad(u8),
    then little-endian float32: means[n_numeric], sds[n_numeric], pref_w[d], pref_b, q_w[d], q_b,
    then the genre names as NUL-terminated ASCII. d = n_numeric + n_mood + n_genre."""
    import struct
    from rl.base_model import MOODS, NUMERIC
    d = len(NUMERIC) + len(MOODS) + len(feat.genres)
    assert len(w) == d and len(q_w) == d
    floats = [feat.means[c] for c in NUMERIC] + [feat.sds[c] for c in NUMERIC] + list(map(float, w)) + [float(b)] \
        + list(map(float, q_w)) + [float(q_b)]
    blob = MAGIC + struct.pack("<HBBBB", VERSION, len(NUMERIC), len(MOODS), len(feat.genres), 0)
    blob += struct.pack(f"<{len(floats)}f", *floats)
    for g in feat.genres:
        blob += g.encode("ascii", "replace") + b"\0"
    return blob


def decode_device_blob(blob):
    """What the device does: read the blob back. Returns dict with means, sds, w, b, q_w, q_b, genres."""
    import struct
    assert blob[:4] == MAGIC, "bad magic"
    version, nn, nm, ng, _ = struct.unpack("<HBBBB", blob[4:10])
    d = nn + nm + ng
    n_f = 2 * nn + d + 1 + d + 1
    f = struct.unpack(f"<{n_f}f", blob[10:10 + 4 * n_f])
    names = blob[10 + 4 * n_f:].split(b"\0")[:ng]
    return {"version": version, "means": f[:nn], "sds": f[nn:2 * nn], "w": f[2 * nn:2 * nn + d], "b": f[2 * nn + d],
            "q_w": f[2 * nn + d + 1:2 * nn + 2 * d + 1], "q_b": f[-1], "genres": [x.decode() for x in names]}


def device_score(dec, song, which="pref"):
    """Score a song exactly as the device would from the decoded blob (float32 numbers)."""
    from rl.base_model import MOODS, NUMERIC
    x = [(song.audio_features.get(c, dec["means"][i]) - dec["means"][i]) / dec["sds"][i] for i, c in enumerate(NUMERIC)]
    x += [1.0 if song.mood == m else 0.0 for m in MOODS]
    x += [1.0 if song.genre == g else 0.0 for g in dec["genres"]]
    if which == "q":
        return sum(a * c for a, c in zip(dec["q_w"], x)) + dec["q_b"]
    z = sum(a * c for a, c in zip(dec["w"], x)) + dec["b"]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def c_header(blob_dec, name="BASE_V2"):
    def arr(label, vals):
        return f"static const float {name}_{label}[{len(vals)}] = {{" + ", ".join(f"{v:.8g}f" for v in vals) + "};"
    lines = [f"// Generated by rl/base_lastfm.py. Base model v2: logistic preference head + ridge objective head.",
             f"// {len(blob_dec['w'])} features, float32. The phone/host builds the feature vector; the device only does dot products.",
             "#pragma once",
             f"#define {name}_N_FEATURES {len(blob_dec['w'])}",
             arr("MEANS", blob_dec["means"]), arr("SDS", blob_dec["sds"]), arr("PREF_W", blob_dec["w"]),
             f"static const float {name}_PREF_B = {blob_dec['b']:.8g}f;", arr("Q_W", blob_dec["q_w"]),
             f"static const float {name}_Q_B = {blob_dec['q_b']:.8g}f;",
             f"static const char* const {name}_GENRES[{len(blob_dec['genres'])}] = {{" +
             ", ".join(json.dumps(g) for g in blob_dec["genres"]) + "};"]
    return "\n".join(lines) + "\n"


def footprint(blob):
    """Bytes the model needs in RAM on the device and how that compares with the ESP32's SRAM."""
    return {"bytes": len(blob), "share_of_ram": len(blob) / ESP32_RAM_BYTES, "budget_bytes": ESP32_BUDGET_BYTES,
            "ram_bytes": ESP32_RAM_BYTES, "within_budget": len(blob) <= ESP32_BUDGET_BYTES,
            "within_ram": len(blob) <= ESP32_RAM_BYTES}


# ---------------------------------------------------------------- the whole run

def run(tar_path, out_dir=MODEL_DIR, seed=0, epochs=600, lr=1.0, lam=1e-4, checkpoint_every=50, log=print,
        catalog=None, ratings_rows=None, plays=None, min_songs=200, min_users=30):
    """The full pipeline. Returns a dict with every number the report needs."""
    from music.catalog.spotify_catalog import load_spotify
    from rl import ratings as rt
    log = log or (lambda *a, **k: None)
    out_dir = Path(out_dir)
    songs = catalog if catalog is not None else load_spotify()
    by_id = {s.song_id: s for s in songs}
    if plays is None:
        cache = DATA_DIR / "plays.pkl"
        if cache.exists():
            log(f"Reading cached play counts {cache}")
            plays = pd.read_pickle(cache)
        else:
            log("Aggregating Last.fm plays (this reads about 19 million rows)...")
            plays = aggregate_plays(tar_path, log=log)
            plays.to_pickle(cache)
    res = {"n_users_raw": int(plays["user"].nunique()), "n_pairs": int(len(plays))}
    log(f"{res['n_users_raw']} users, {res['n_pairs']:,} (user, track) pairs")
    mapping = match_pairs(plays[["artist", "track"]].drop_duplicates().itertuples(index=False, name=None),
                          build_catalog_index(songs))
    res["match"] = match_stats(plays, mapping)
    log(f"Match rate (pairs played by {MIN_USERS_MATCH}+ users): {res['match']['match_rate']:.1%} "
        f"({res['match']['pairs_matched']:,} of {res['match']['pairs_eligible']:,}); events {res['match']['event_match_rate']:.1%}")
    res["l11_pass"] = res["match"]["match_rate"] >= MATCH_PASS
    users, song_ids, liked, listeners = build_matrix(plays, mapping)
    res.update({"n_users": len(users), "n_songs_K": len(song_ids), "n_liked_pairs": int(liked.sum())})
    log(f"{len(users)} users with >= {MIN_LIKED} liked songs; known set K = {len(song_ids):,} songs; {int(liked.sum()):,} liked pairs")
    if len(users) < min_users or len(song_ids) < min_songs:
        res["error"] = "too little matched data to train"
        return res
    ks = [by_id[s] for s in song_ids]
    feat = Featurizer.fit(ks)
    X = np.array([feat.vector(s) for s in ks])
    tr, va, te = split_users(len(users), seed)
    res["split"] = (len(tr), len(va), len(te))
    tr_pos, tr_neg = song_weights(liked, tr)
    va_pos, va_neg = song_weights(liked, va)
    ckpt_dir = out_dir / "checkpoints_v2"
    ckpt_paths = []

    def on_ckpt(ep, w, b, row):
        qw, qb = fit_objective(X, listeners)
        p = save_base(ckpt_dir / f"base_v2_epoch_{ep:04d}.json", feat, w, b, qw, qb,
                      {"epoch": ep, **{k: row[k] for k in ("train_loss", "val_loss", "train_auc", "val_auc")}})
        os.chmod(p, 0o644)
        ckpt_paths.append(str(p))
    best, last, history, ckpts = fit_preference(X, tr_pos, tr_neg, va_pos, va_neg, epochs, lr, lam, checkpoint_every,
                                                on_ckpt, log)
    (bw, bb), best_epoch = best[1], best[2]
    res.update({"history": history, "checkpoint_epochs": ckpts, "best_epoch": best_epoch})
    # ---- test users, read once
    z = X @ bw + bb
    pop = np.array([s.audio_features.get("popularity", 0.0) for s in ks])
    m_auc, p_auc = mean_user_auc(z, liked, te), mean_user_auc(pop, liked, te)
    lc_auc = mean_user_auc(tr_pos, liked, te)             # ceiling reference: train users' like counts per song
    n = min(len(m_auc), len(p_auc))
    lo, hi = bootstrap_mean_diff(m_auc[:n], p_auc[:n])
    res["test"] = {"n_users": int(n), "model_auc": float(m_auc.mean()), "popularity_auc": float(p_auc.mean()),
                   "like_count_auc": float(mean_user_auc(tr_pos, liked, te).mean()),
                   "gain": float(m_auc.mean() - p_auc.mean()), "gain_lo": lo, "gain_hi": hi,
                   "pooled_model_auc": weighted_auc(z, *song_weights(liked, te)),
                   "pooled_popularity_auc": weighted_auc(pop, *song_weights(liked, te)),
                   "val_popularity_auc": weighted_auc(pop, va_pos, va_neg),
                   "pass": bool(m_auc.mean() - p_auc.mean() >= AUC_GAIN_PASS)}
    log(f"TEST ({n} users): model AUC {m_auc.mean():.3f}, popularity {p_auc.mean():.3f}, gain {m_auc.mean() - p_auc.mean():+.3f} "
        f"(95% range {lo:+.3f} to {hi:+.3f}); like-count ceiling {lc_auc.mean():.3f}")
    # ---- objective head and separability on Adi's ratings
    qw, qb = fit_objective(X, listeners)
    res["q_fit_corr"] = float(np.corrcoef(X @ qw + qb, np.log1p(listeners))[0, 1])
    rows = rt.first_pass(ratings_rows) if ratings_rows is not None else []
    rows = [r for r in rows if r["song_id"] in by_id]
    if rows:
        qs = [float(np.array(feat.vector(by_id[r["song_id"]])) @ qw + qb) for r in rows]
        cq = rt.spearman(qs, [r["quality"] for r in rows])
        cp = rt.spearman(qs, [r["preference"] for r in rows])
        res["separable"] = {"n": len(rows), "corr_quality": cq, "corr_preference": cp,
                            "pass": bool(cq is not None and cp is not None and cq >= Q_CORR_PASS and cq > cp)}
        log(f"Adi ({len(rows)} songs): Spearman(q, quality) {cq}, Spearman(q, preference) {cp}")
    training = {"user_split": res["split"], "seed": seed, "epochs": epochs, "lr": lr, "lam": lam, "best_epoch": best_epoch,
                "n_songs_K": len(song_ids), "n_users": len(users), "match_rate": res["match"]["match_rate"],
                "test_gain_over_popularity": res["test"]["gain"], "data": "Last.fm 1K (Celma), Zenodo 6090214, non-commercial"}
    res["model_path"] = str(save_base(out_dir / "base_model_v2.json", feat, bw, bb, qw, qb, training, ckpt_paths))
    res["sha256"] = load_base(res["model_path"]).sha256
    blob = device_blob(feat, bw, bb, qw, qb)
    (out_dir / "base_model_v2.bin").write_bytes(blob)
    dec = decode_device_blob(blob)
    (out_dir / "base_model_v2.h").write_text(c_header(dec))
    res["device"] = footprint(blob)
    probe = ks[:200]
    res["device"]["max_score_diff"] = float(max(abs(device_score(dec, s) - float(sigmoid(np.array(feat.vector(s)) @ bw + bb)))
                                               for s in probe))
    log(f"ESP32 export: {len(blob)} bytes ({res['device']['share_of_ram']:.3%} of 520 KB RAM; budget 32 KB)")
    import csv
    with open(out_dir / "training_history_v2.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(history[0]))
        wr.writeheader()
        wr.writerows(history)
    svg = training_graph_svg(history, ckpts, best_epoch, res["test"]["val_popularity_auc"],
                             "Base model v2 (Last.fm 1K): training on other people's likes", log_x=epochs >= 2000)
    (out_dir / "training_curve_v2.svg").write_text(svg)
    res["top_features"] = sorted(zip(feat.names, bw), key=lambda x: -abs(x[1]))[:10]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", default=str(DATA_DIR / "lastfm-dataset-1K.tar.gz"))
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user", default="adi")
    a = ap.parse_args(argv)
    from rl import ratings as rt
    rows = rt.load_rows(rt.ratings_path(a.user), a.user)
    res = run(a.tar, seed=a.seed, epochs=a.epochs, lr=a.lr, lam=a.lam, ratings_rows=rows)
    (MODEL_DIR / "step06_results_v2.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k != "history"}, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
