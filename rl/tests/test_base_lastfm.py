"""Run from the repo root: python3 -m unittest rl.tests.test_base_lastfm -v"""
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rl import base_lastfm as bl
from rl.base_model import Featurizer
from rl.tests.test_eval import fake_catalog

CAT = fake_catalog(150)
LIKED_GENRES = {"jazz", "folk", "rock"}


def synthetic_plays(n_users=80, seed=0):
    """Users like songs in a few genres (3-6 plays) and sample others once; names equal the catalog's, so matching works."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(n_users):
        mine = {g for g in sorted(LIKED_GENRES) if rng.rand() < 0.7} or {"jazz"}
        for s in CAT:
            if s.genre in mine and rng.rand() < 0.6:
                rows.append((f"u{u}", s.artist, s.title, int(rng.randint(3, 7))))
            elif rng.rand() < 0.15:
                rows.append((f"u{u}", s.artist, s.title, 1))
    return pd.DataFrame(rows, columns=["user", "artist", "track", "plays"])


class TestReadAndMatch(unittest.TestCase):
    def test_aggregate_plays_counts_repeats_and_skips_broken_lines(self):
        lines = ["u1\t2009-01-01T00:00:00Z\ta1\tArtist A\tt1\tSong One",
                 "u1\t2009-01-02T00:00:00Z\ta1\tArtist A\tt1\tSong One",
                 "u2\t2009-01-02T00:00:00Z\ta1\tArtist A\tt1\tSong One",
                 "broken line without tabs",
                 "u1\t2009-01-03T00:00:00Z\ta2\tArtist B\tt2\tSong \"Two\""]
        with tempfile.TemporaryDirectory() as d:
            tsv = Path(d) / bl.TSV_SUFFIX
            tsv.write_text("\n".join(lines) + "\n")
            tar = Path(d) / "x.tar.gz"
            with tarfile.open(tar, "w:gz") as tf:
                tf.add(tsv, arcname="lastfm-dataset-1K/" + bl.TSV_SUFFIX)
            got = bl.aggregate_plays(tar, chunk=2, log=None)
        d = {(r.user, r.artist, r.track): r.plays for r in got.itertuples()}
        self.assertEqual(d[("u1", "Artist A", "Song One")], 2)
        self.assertEqual(d[("u2", "Artist A", "Song One")], 1)
        self.assertEqual(sum(d.values()), 5 - 1)                      # the broken line is dropped

    def test_matching_cleans_names_and_prefers_the_popular_duplicate(self):
        from dataclasses import replace
        a = replace(CAT[0], song_id="low", title="Hello (Live)", artist="The Band", audio_features={**CAT[0].audio_features, "popularity": 10.0})
        b = replace(CAT[1], song_id="high", title="Hello", artist="The Band", audio_features={**CAT[1].audio_features, "popularity": 90.0})
        idx = bl.build_catalog_index([a, b])
        got = bl.match_pairs([("The Band", "Hello - Remastered"), ("Nobody", "Nothing")], idx)
        self.assertEqual(got, {("The Band", "Hello - Remastered"): "high"})

    def test_match_rate_uses_pairs_with_enough_users_and_reports_events(self):
        rows = [(f"u{i}", "A", "popular", 1) for i in range(6)] + [(f"u{i}", "B", "missing", 1) for i in range(6)] + \
               [("u0", "C", "rare", 9)]
        plays = pd.DataFrame(rows, columns=["user", "artist", "track", "plays"])
        st = bl.match_stats(plays, {("A", "popular"): "s1"})
        self.assertEqual(st["pairs_eligible"], 2)                      # "rare" has only 1 user
        self.assertAlmostEqual(st["match_rate"], 0.5)
        self.assertAlmostEqual(st["event_match_rate"], 6 / 21)


class TestMatrix(unittest.TestCase):
    def test_known_set_liked_threshold_and_min_liked_users(self):
        plays = synthetic_plays()
        idx = bl.build_catalog_index(CAT)
        mapping = bl.match_pairs(plays[["artist", "track"]].drop_duplicates().itertuples(index=False, name=None), idx)
        users, songs, liked, listeners = bl.build_matrix(plays, mapping)
        self.assertEqual(liked.shape, (len(users), len(songs)))
        self.assertTrue((listeners >= bl.MIN_USERS_MATCH).all())
        self.assertTrue((liked.sum(axis=1) >= bl.MIN_LIKED).all())
        # a play count below 3 never makes a "like"
        one_play = plays[plays["plays"] == 1]
        u0, a0, t0 = one_play.iloc[0][["user", "artist", "track"]]
        if u0 in users and mapping.get((a0, t0)) in songs:
            self.assertFalse(liked[users.index(u0), songs.index(mapping[(a0, t0)])])

    def test_split_is_disjoint_complete_and_reproducible(self):
        a, b, c = bl.split_users(100, seed=2)
        self.assertEqual(sorted(list(a) + list(b) + list(c)), list(range(100)))
        self.assertEqual(list(a), list(bl.split_users(100, seed=2)[0]))
        self.assertAlmostEqual(len(a) / 100, 0.70, delta=0.02)


class TestMetrics(unittest.TestCase):
    def test_user_auc_matches_brute_force_including_ties(self):
        rng = np.random.RandomState(1)
        scores = rng.randint(0, 5, 40).astype(float)
        liked = rng.rand(40) < 0.3
        pos, neg = scores[liked], scores[~liked]
        brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(bl.user_auc(scores, liked), brute)

    def test_weighted_auc_matches_brute_force(self):
        rng = np.random.RandomState(2)
        scores = rng.randint(0, 6, 30).astype(float)
        pw, nw = rng.randint(0, 4, 30).astype(float), rng.randint(1, 5, 30).astype(float)
        num = sum(pw[i] * nw[j] * ((scores[i] > scores[j]) + 0.5 * (scores[i] == scores[j])) for i in range(30) for j in range(30))
        self.assertAlmostEqual(bl.weighted_auc(scores, pw, nw), num / (pw.sum() * nw.sum()))

    def test_bootstrap_range_contains_the_mean_difference(self):
        a, b = np.array([0.7, 0.8, 0.75, 0.9, 0.65]), np.array([0.6, 0.7, 0.7, 0.8, 0.6])
        lo, hi = bl.bootstrap_mean_diff(a, b, n=500)
        self.assertLessEqual(lo, (a - b).mean())
        self.assertGreaterEqual(hi, (a - b).mean())


class TestTraining(unittest.TestCase):
    def setUp(self):
        plays = synthetic_plays()
        idx = bl.build_catalog_index(CAT)
        mapping = bl.match_pairs(plays[["artist", "track"]].drop_duplicates().itertuples(index=False, name=None), idx)
        self.users, self.songs, self.liked, self.listeners = bl.build_matrix(plays, mapping)
        by = {s.song_id: s for s in CAT}
        self.ks = [by[i] for i in self.songs]
        self.feat = Featurizer.fit(self.ks)
        self.X = np.array([self.feat.vector(s) for s in self.ks])
        self.tr, self.va, self.te = bl.split_users(len(self.users), 0)

    def test_learns_planted_genre_taste_and_records_checkpoints(self):
        tp, tn = bl.song_weights(self.liked, self.tr)
        vp, vn = bl.song_weights(self.liked, self.va)
        seen = []
        best, last, hist, ckpts = bl.fit_preference(self.X, tp, tn, vp, vn, epochs=200, checkpoint_every=50,
                                                    on_checkpoint=lambda ep, w, b, row: seen.append(ep), log=None)
        self.assertEqual(ckpts, [0, 50, 100, 150, 200])
        self.assertEqual(seen, ckpts)
        self.assertLess(hist[-1]["train_loss"], hist[0]["train_loss"])
        self.assertGreater(hist[-1]["val_auc"], 0.8)
        (w, b), ep = best[1], best[2]
        z = self.X @ w + b
        auc = bl.mean_user_auc(z, self.liked, self.te)
        self.assertGreater(auc.mean(), 0.8)
        pop = np.array([s.audio_features.get("popularity", 0.0) for s in self.ks])
        self.assertGreater(auc.mean(), bl.mean_user_auc(pop, self.liked, self.te).mean())

    def test_objective_head_tracks_listener_counts(self):
        qw, qb = bl.fit_objective(self.X, self.listeners)
        corr = np.corrcoef(self.X @ qw + qb, np.log1p(self.listeners))[0, 1]
        self.assertGreater(corr, 0.0)


class TestExportAndFreeze(unittest.TestCase):
    def setUp(self):
        self.feat = Featurizer.fit(CAT)
        d = len(self.feat.names)
        rng = np.random.RandomState(0)
        self.w, self.qw = rng.randn(d) * 0.1, rng.randn(d) * 0.1
        self.b, self.qb = -2.0, 1.5

    def test_device_blob_is_tiny_decodes_and_reproduces_scores(self):
        blob = bl.device_blob(self.feat, self.w, self.b, self.qw, self.qb)
        fp = bl.footprint(blob)
        self.assertTrue(fp["within_budget"] and fp["within_ram"])
        self.assertLess(fp["bytes"], 4096)
        dec = bl.decode_device_blob(blob)
        self.assertEqual(dec["genres"], self.feat.genres)
        for s in CAT[:40]:
            x = np.array(self.feat.vector(s))
            self.assertAlmostEqual(bl.device_score(dec, s), float(bl.sigmoid(x @ self.w + self.b)), places=4)
            self.assertAlmostEqual(bl.device_score(dec, s, "q"), float(x @ self.qw + self.qb), places=4)
        with self.assertRaises(AssertionError):
            bl.decode_device_blob(b"XXXX" + blob[4:])
        header = bl.c_header(dec)
        self.assertIn("BASE_V2_PREF_W", header)
        self.assertIn(f"BASE_V2_N_FEATURES {len(self.w)}", header)

    def test_saved_base_is_hashed_read_only_and_unchanged_by_building_on_it(self):
        with tempfile.TemporaryDirectory() as d:
            path = bl.save_base(Path(d) / "b.json", self.feat, self.w, self.b, self.qw, self.qb, {"epoch": 1})
            self.assertEqual(os.stat(path).st_mode & 0o222, 0)            # no write bit
            base = bl.load_base(path)
            before = (base.sha256, base.preference_weights().tobytes(), Path(path).read_bytes())
            mine = base.preference_weights()
            mine += 5.0                                                    # "fine-tuning" a copy
            with self.assertRaises(ValueError):
                base._w[0] = 9.0                                           # the loaded base itself is read-only
            after = bl.load_base(path)
            self.assertEqual(before, (after.sha256, after.preference_weights().tobytes(), Path(path).read_bytes()))
            self.assertAlmostEqual(base.prob(CAT[0]), after.prob(CAT[0]))
            # tampering is detected
            os.chmod(path, 0o644)
            doc = json.loads(Path(path).read_text())
            doc["model"]["preference"]["bias"] += 1
            Path(path).write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                bl.load_base(path)
            # saving again over a read-only file works
            os.chmod(path, 0o444)
            bl.save_base(path, self.feat, self.w, self.b, self.qw, self.qb, {"epoch": 2})
            self.assertEqual(bl.load_base(path).doc["training"]["epoch"], 2)


class TestWholeRun(unittest.TestCase):
    def test_too_little_matched_data_stops_with_an_error_instead_of_training(self):
        with tempfile.TemporaryDirectory() as d:
            res = bl.run(None, out_dir=d, epochs=5, log=None, catalog=CAT, ratings_rows=[], plays=synthetic_plays())
        self.assertIn("error", res)
        self.assertNotIn("test", res)

    def test_pipeline_end_to_end_on_a_synthetic_world(self):
        rows = [{"user": "a", "song_id": s.song_id, "quality": 1 + i % 5, "preference": 5 if s.genre in LIKED_GENRES else 1,
                 "split": "dev", "round": 1, "time": 0.0} for i, s in enumerate(CAT[:60])]
        with tempfile.TemporaryDirectory() as d:
            res = bl.run(None, out_dir=d, epochs=120, checkpoint_every=40, log=None, catalog=CAT,
                         ratings_rows=rows, plays=synthetic_plays(), min_songs=100)
            self.assertTrue(res["l11_pass"])
            self.assertGreater(res["match"]["match_rate"], 0.9)
            self.assertTrue(res["test"]["pass"])
            self.assertGreater(res["test"]["model_auc"], res["test"]["popularity_auc"])
            self.assertEqual(res["checkpoint_epochs"], [0, 40, 80, 120])
            self.assertTrue((Path(d) / "base_model_v2.json").exists())
            self.assertTrue((Path(d) / "base_model_v2.bin").exists())
            self.assertTrue((Path(d) / "base_model_v2.h").exists())
            self.assertTrue((Path(d) / "training_curve_v2.svg").exists())
            self.assertTrue(res["device"]["within_budget"])
            self.assertLess(res["device"]["max_score_diff"], 1e-4)
            self.assertIn("separable", res)
            self.assertEqual(bl.load_base(res["model_path"]).sha256, res["sha256"])


if __name__ == "__main__":
    unittest.main()
