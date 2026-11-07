"""Real audio playback (Person C). Looks a song up on the free iTunes Search API
(no key needed), downloads its 30-second preview and plays it through the speaker.

Only the song's title and artist are sent to Apple. The player is non-blocking:
play() starts the clip and returns, so the robot loop and motion keep running.
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from shared.schemas.messages import Song

SEARCH_URL = "https://itunes.apple.com/search"
USER_AGENT = "MakeCU-robot/0.1"
ALLOWED_AUDIO_HOSTS = ("apple.com", "mzstatic.com")  # previews are served from here
MIN_TITLE_SIMILARITY = 0.75


# ---- matching a dataset song to an iTunes result ----
def _clean_title(title: str) -> str:
    title = re.sub(r"[\(\[].*?[\)\]]", " ", title.lower())  # (feat. X), [Remix]
    title = re.sub(r"\s+-\s+.*$", "", title)  # " - Radio Edit", " - Remastered 2011"
    return re.sub(r"[^a-z0-9]+", " ", title).strip()


def _clean_artist(artist: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", artist.lower()).strip()


def best_match(song: Song, results: list) -> dict | None:
    """The iTunes result that is really this song (same artist, similar title) and has
    a preview, or None. Refusing a weak match beats playing the wrong song."""
    want_title, want_artist = _clean_title(song.title), _clean_artist(song.artist)
    best, best_key = None, (0.0, 0)
    for r in results:
        if not r.get("previewUrl"):
            continue
        artist = _clean_artist(r.get("artistName", ""))
        if not (want_artist and artist and (want_artist in artist or artist in want_artist)):
            continue
        sim = difflib.SequenceMatcher(None, want_title, _clean_title(r.get("trackName", ""))).ratio()
        key = (sim, -len(r.get("trackName", "")))  # on a tie prefer the plainer title (the original, not a remix)
        if key > best_key:
            best, best_key = r, key
    return best if best_key[0] >= MIN_TITLE_SIMILARITY else None


class PreviewFinder:
    """Song -> preview URL via the iTunes Search API. Results are cached per song."""

    def __init__(self, min_interval: float = 1.0, timeout: float = 10.0, fetch=None):
        self.min_interval, self.timeout = min_interval, timeout
        self._fetch = fetch or self._http_get  # injectable for tests
        self._cache: dict[str, str | None] = {}
        self._last_request = 0.0

    def _http_get(self, url: str) -> bytes:
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:  # stay well under the API's rate limit (~20 requests/min)
            time.sleep(wait)
        self._last_request = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def find(self, song: Song) -> str | None:
        """Preview URL for the song, or None if iTunes has no matching preview.
        Network errors also return None (and are not cached, so a retry can work)."""
        if song.song_id in self._cache:
            return self._cache[song.song_id]
        query = urllib.parse.urlencode({"term": f"{song.title} {song.artist}",
                                        "media": "music", "entity": "song", "limit": 10})
        try:
            results = json.loads(self._fetch(f"{SEARCH_URL}?{query}")).get("results", [])
        except (urllib.error.URLError, OSError, ValueError) as err:
            print(f"  (iTunes lookup failed for '{song.title}': {err})", file=sys.stderr)
            return None
        match = best_match(song, results)
        self._cache[song.song_id] = match["previewUrl"] if match else None
        return self._cache[song.song_id]


# ---- playing it ----
def _find_player_command(seconds: float | None):
    """Returns a function path -> argv for whichever audio player this computer has."""
    if shutil.which("afplay"):  # macOS
        return lambda p: ["afplay", *(["-t", str(seconds)] if seconds else []), p]
    if shutil.which("ffplay"):
        return lambda p: ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                          *(["-t", str(seconds)] if seconds else []), p]
    if shutil.which("mpv"):
        return lambda p: ["mpv", "--no-video", "--really-quiet",
                          *([f"--length={seconds}"] if seconds else []), p]
    raise RuntimeError("No audio player found (need afplay, ffplay, or mpv)")


class AudioPlayer:
    """Plug into PlaylistController(player=...). play(song) returns False when the
    song has no preview, so the caller can skip to the next pick."""

    def __init__(self, finder: PreviewFinder | None = None, clip_seconds: float | None = None,
                 popen=subprocess.Popen, download=None):
        self.finder = finder or PreviewFinder()
        self._command = _find_player_command(clip_seconds)
        self._popen = popen
        self._download = download or self._http_download
        self._dir = tempfile.mkdtemp(prefix="makecu_audio_")  # fresh dir just for clips
        self._proc = None
        self.last_url: str | None = None

    def _http_download(self, url: str, dest: str) -> None:
        host = urllib.parse.urlparse(url).hostname or ""
        if not url.startswith("https://") or not host.endswith(ALLOWED_AUDIO_HOSTS):
            raise ValueError(f"refusing to download from {host or url}")
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)

    def play(self, song: Song) -> bool:
        self.stop()
        url = self.finder.find(song)
        self.last_url = url
        if not url:
            return False
        path = f"{self._dir}/{song.song_id}.m4a"
        try:
            self._download(url, path)
            self._proc = self._popen(self._command(path), stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        except (OSError, ValueError, urllib.error.URLError) as err:
            print(f"  (could not play '{song.title}': {err})", file=sys.stderr)
            return False
        return True

    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def wait(self) -> None:
        if self._proc is not None:
            self._proc.wait()

    def stop(self) -> None:
        if self.is_playing():
            self._proc.terminate()
            self._proc.wait()
        self._proc = None

    def close(self) -> None:
        self.stop()
        shutil.rmtree(self._dir, ignore_errors=True)
