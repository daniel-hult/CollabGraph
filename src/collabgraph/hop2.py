"""
hop2.py

Build a 2-hop collaboration network, while keeping scope tight:
- Hop 1: seed -> collaborators (weight = shared track count)
- Hop 2: collaborations among {seed + hop1} only (no new nodes),
         derived from tracks of hop1 artists.

Outputs (in outputs/):
- nodes.csv: seed + hop1 nodes
- edges.csv: undirected edges among allowed nodes with weights
- edge_tracks.csv: one row per (edge, track) association

Includes a minimal rate-limit backoff wrapper for Spotify API calls.
"""

from __future__ import annotations

import os
import time
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException


# ----------------------------
# Rate limit hardening knobs
# ----------------------------

# Turn on/off all pacing
THROTTLE_ENABLED = True

# Sustained request rate target (requests per second).
TARGET_RPS = 2.0  # 120 requests/minute on average (but with jitter, not perfectly uniform)

# Random jitter to avoid perfectly periodic request patterns
JITTER_SECONDS = 0.10  # up to +/-100ms

# Extra cooldown after a 429 to avoid immediate re-trigger
COOLDOWN_AFTER_429_SECONDS = 1.0

# If Spotify tells us to wait "forever" (hours), fail fast.
# We do NOT keep retrying and risk making the ban worse.
HUGE_RETRY_AFTER_SECONDS = 120

# Safety budget: stop Hop 2 scan early to avoid catastrophic bans.
# This does NOT reduce Hop 1; it only caps how deep Hop 2 scanning goes.
ENABLE_CALL_BUDGET = True
MAX_SPOTIFY_CALLS_PER_RUN = 8000  # adjust based on your needs

# --- Simple run stats (helpful for you to understand runtime) ---
SPOTIFY_CALL_COUNT = 0
SPOTIFY_SLEEP_SECONDS = 0.0
SPOTIFY_429_COUNT = 0


class CallBudgetExceeded(Exception):
    """Raised when we exceed the per-run Spotify API call budget."""

class HardRateLimit(Exception):
    """
    Raised when we hit a 429 but cannot reliably extract a safe Retry-After.
    In practice this often corresponds to the long lockouts (hours).
    """

class RateLimiter:
    """
    Very small sustained-rate limiter.

    We enforce an average rate (TARGET_RPS) across the entire run.
    This is more effective than micro-sleeps because it controls long-run request volume.
    """

    def __init__(self, target_rps: float, jitter_seconds: float = 0.0):
        self.target_rps = max(0.1, float(target_rps))
        self.min_interval = 1.0 / self.target_rps
        self.jitter_seconds = max(0.0, float(jitter_seconds))
        self._next_allowed = time.monotonic()

    def wait(self, extra: float = 0.0) -> float:
        """
        Sleep until the next request is allowed.
        Returns how long we slept (seconds).
        """
        if not THROTTLE_ENABLED:
            return 0.0

        now = time.monotonic()

        # Jitter: small randomization around the schedule
        jitter = random.uniform(-self.jitter_seconds, self.jitter_seconds)

        scheduled = self._next_allowed + extra + jitter
        sleep_for = max(0.0, scheduled - now)

        if sleep_for > 0:
            time.sleep(sleep_for)

        # Move the schedule forward by one slot, anchored to "now" so we don't drift weirdly
        now_after = time.monotonic()
        self._next_allowed = max(self._next_allowed, now_after) + self.min_interval

        return sleep_for
    
    def set_target_rps(self, target_rps: float) -> None:
        self.target_rps = max(0.1, float(target_rps))
        self.min_interval = 1.0 / self.target_rps


# Global limiter instance used by spotify_call()
RATE_LIMITER = RateLimiter(target_rps=TARGET_RPS, jitter_seconds=JITTER_SECONDS)

# ----------------------------
# Data shapes
# ----------------------------

@dataclass(frozen=True)
class ArtistInfo:
    artist_id: str
    name: str
    popularity: Optional[int]
    followers: Optional[int]
    hop: int  # 0 for seed, 1 for hop1


# ----------------------------
# Spotify client + safe call wrapper
# ----------------------------

def make_spotify_client() -> spotipy.Spotify:
    load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Spotify credentials in .env: "
            "SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET"
        )

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return spotipy.Spotify(
        auth_manager=auth_manager,
        retries=0,  # we handle retries ourselves
        status_retries=0,
        backoff_factor=0,
    )


def spotify_call(fn, *args, max_retries: int = 5, **kwargs):
    """
    Retry/backoff wrapper for Spotify API calls.
    Also applies a small throttle to reduce burstiness and tracks call stats.
    """

    global SPOTIFY_CALL_COUNT
    global SPOTIFY_429_COUNT

    delay_seconds = 1.0

    for attempt in range(1, max_retries + 1):
        # Count and throttle PER ACTUAL REQUEST ATTEMPT (important!)
        SPOTIFY_CALL_COUNT += 1

        # Global safety budget: prevent runaway call volume anywhere in the code.
        if ENABLE_CALL_BUDGET and SPOTIFY_CALL_COUNT >= MAX_SPOTIFY_CALLS_PER_RUN:
            raise CallBudgetExceeded(
                f"Spotify call budget exceeded ({SPOTIFY_CALL_COUNT} >= {MAX_SPOTIFY_CALLS_PER_RUN}). "
                "Stopping Hop 2 scan early to avoid triggering long rate-limit bans."
            )

        throttle_sleep()

        try:
            return fn(*args, **kwargs)

        except SpotifyException as e:
            if e.http_status == 429:
                SPOTIFY_429_COUNT += 1

                retry_after = None
                retry_after_source = "unknown"

                # 1) Best case: Retry-After header exists
                if hasattr(e, "headers") and e.headers:
                    header_val = e.headers.get("Retry-After")
                    if header_val is not None:
                        retry_after = header_val
                        retry_after_source = "header"

                # 2) Fallback: sometimes the wait time is embedded in the exception message
                # NOTE: In some cases Spotipy prints the "Retry will occur after" line separately
                # and the exception string won't contain it. That's why we treat "unknown" as dangerous.
                if retry_after is None:
                    msg = str(e)
                    m = re.search(r"retry\s*will\s*occur\s*after:\s*([0-9]+)\s*s", msg, flags=re.IGNORECASE)
                    if m:
                        retry_after = m.group(1)
                        retry_after_source = "message"

                # Normalize
                if retry_after is not None:
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = None
                        retry_after_source = "unknown"

                # If we can't read a sane Retry-After, treat it as a hard lockout and stop.
                # This prevents your code from poking the API repeatedly during a ban window.
                if retry_after is None:
                    raise HardRateLimit(
                        "Spotify 429 received but Retry-After could not be determined. "
                        "Stopping immediately to avoid making a long ban worse."
                    )

                # If Spotify tells us to wait "forever" (hours), abort immediately.
                if retry_after > HUGE_RETRY_AFTER_SECONDS:
                    raise HardRateLimit(
                        f"Spotify rate limit hit. Retry-After={retry_after:.0f}s (too large). "
                        "Stopping immediately to avoid making the ban worse. Try again later."
                    )

                # Otherwise: short backoff + retry
                sleep_for = min(retry_after, 60.0)

                print(
                    f"[rate-limit] 429 from Spotify. Sleeping {sleep_for:.1f}s "
                    f"(attempt {attempt}/{max_retries}, source={retry_after_source})"
                )
                sleep_and_track(sleep_for)

                delay_seconds = min(delay_seconds * 2, 30.0)

                # Extra cooldown before next attempt
                throttle_sleep(extra=COOLDOWN_AFTER_429_SECONDS)
                continue

            # Optional: retry some 5xx errors (rare but happens)
            if e.http_status and 500 <= e.http_status < 600 and attempt < max_retries:
                print(f"[spotify] {e.http_status} server error. Sleeping {delay_seconds:.1f}s (attempt {attempt}/{max_retries})")
                sleep_and_track(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 30)
                continue

            raise

    raise RuntimeError(
        f"Spotify call failed after {max_retries} retries: {getattr(fn, '__name__', str(fn))}"
    )


# ----------------------------
# Helpers: paging + bulk endpoints
# ----------------------------

def status(message: str) -> None:
    print(f"⏳ {message}")


def done(message: str) -> None:
    print(f"✅ {message}")


def throttle_sleep(extra: float = 0.0) -> None:
    """
    Apply sustained-rate limiting + jitter. Tracks time slept in stats.
    """
    global SPOTIFY_SLEEP_SECONDS

    slept = RATE_LIMITER.wait(extra=extra)
    SPOTIFY_SLEEP_SECONDS += slept


def sleep_and_track(seconds: float) -> None:
    """
    Sleep and include the time in SPOTIFY_SLEEP_SECONDS stats so your totals
    reflect BOTH throttling and retry/backoff sleeps.
    """
    global SPOTIFY_SLEEP_SECONDS
    time.sleep(seconds)
    SPOTIFY_SLEEP_SECONDS += seconds


def iter_paged(sp: spotipy.Spotify, first_page: dict) -> Iterable[dict]:
    """
    Generator over Spotify paged results.
    """
    page = first_page
    while page:
        for item in page.get("items", []):
            yield item
        if page.get("next"):
            page = spotify_call(sp.next, page)
        else:
            break


def get_tracks_details_bulk(sp: spotipy.Spotify, track_ids: List[str]) -> List[dict]:
    """
    Spotify tracks endpoint: up to 50 per call.
    """
    results: List[dict] = []
    chunk_size = 50
    for i in range(0, len(track_ids), chunk_size):
        chunk = track_ids[i : i + chunk_size]
        response = spotify_call(sp.tracks, chunk)
        results.extend(response.get("tracks", []))
    return results


def get_artists_details_bulk(sp: spotipy.Spotify, artist_ids: List[str]) -> List[dict]:
    """
    Spotify artists endpoint: up to 50 per call.
    """
    results: List[dict] = []
    chunk_size = 50
    for i in range(0, len(artist_ids), chunk_size):
        chunk = artist_ids[i : i + chunk_size]
        response = spotify_call(sp.artists, chunk)
        results.extend(response.get("artists", []))
    return results


# ----------------------------
# Collecting track IDs for an artist (album-based approach)
# ----------------------------

def get_artist_albums(sp: spotipy.Spotify, artist_id: str, max_albums: int, include_appears_on: bool) -> List[dict]:
    include_groups = "album,single,compilation"
    if include_appears_on:
        include_groups += ",appears_on"
    
    first_page = spotify_call(
        sp.artist_albums,
        artist_id,
        include_groups=include_groups,
        country="US",
        limit=50,
    )

    albums_by_id: Dict[str, dict] = {}
    for album in iter_paged(sp, first_page):
        albums_by_id[album["id"]] = album

    albums = list(albums_by_id.values())
    return albums[:max_albums]


def get_album_tracks_ids(sp: spotipy.Spotify, album_id: str) -> List[str]:
    first_page = spotify_call(sp.album_tracks, album_id, limit=50)
    track_ids: List[str] = []
    for t in iter_paged(sp, first_page):
        if t.get("id"):
            track_ids.append(t["id"])
    return track_ids


def get_artist_track_ids(
    sp: spotipy.Spotify,
    artist_id: str,
    max_albums: int,
    max_tracks: int,
    include_appears_on: bool,
) -> List[str]:
    """
    Returns a deduped list of track IDs for an artist via albums -> tracks.
    Controlled by caps to limit API usage.
    """
    albums = get_artist_albums(
        sp,
        artist_id,
        max_albums=max_albums,
        include_appears_on=include_appears_on,
    )

    track_ids: List[str] = []
    for album in albums:
        track_ids.extend(get_album_tracks_ids(sp, album["id"]))

        # Soft cap while collecting (prevents huge memory and extra paging)
        if len(track_ids) >= max_tracks:
            break

    # Deduplicate while preserving order
    track_ids = list(dict.fromkeys(track_ids))
    return track_ids[:max_tracks]


# ----------------------------
# Hop 1: seed -> collaborators
# ----------------------------

def build_hop1_from_seed_tracks(
    sp: spotipy.Spotify,
    seed_artist_id: str,
    max_seed_albums: int,
    max_seed_tracks: int,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """
    Returns:
    - collaborator_to_tracks: collaborator_id -> list of full track objects shared with seed
    - seed_tracks: list of full track objects that include seed (deduped)
    """
    seed_track_ids = get_artist_track_ids(
        sp,
        artist_id=seed_artist_id,
        max_albums=max_seed_albums,
        max_tracks=max_seed_tracks,
        include_appears_on=True,
    )

    seed_tracks = get_tracks_details_bulk(sp, seed_track_ids)

    collaborator_to_tracks: Dict[str, List[dict]] = defaultdict(list)

    for track in seed_tracks:
        if not track or not track.get("id"):
            continue

        artists = track.get("artists", [])
        artist_ids = [a["id"] for a in artists if a.get("id")]

        if seed_artist_id not in artist_ids:
            continue

        for a in artists:
            a_id = a.get("id")
            if not a_id or a_id == seed_artist_id:
                continue
            collaborator_to_tracks[a_id].append(track)

    # Dedup seed_tracks to only those that actually include seed (and unique by track_id)
    seed_tracks_including_seed = []
    seen_track_ids: Set[str] = set()
    for tr in seed_tracks:
        if not tr or not tr.get("id"):
            continue
        artists = tr.get("artists", [])
        ids = [a.get("id") for a in artists if a.get("id")]
        if seed_artist_id in ids and tr["id"] not in seen_track_ids:
            seed_tracks_including_seed.append(tr)
            seen_track_ids.add(tr["id"])

    return collaborator_to_tracks, seed_tracks_including_seed


# ----------------------------
# Hop 2: edges among allowed nodes only
# ----------------------------

def canonical_edge(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def build_hop2_network(
    sp: spotipy.Spotify,
    seed_artist_id: str,
    max_hop1: int = 200,
    max_seed_albums: int = 200,
    max_seed_tracks: int = 3000,
    max_albums_per_hop1_artist: int = 50,
    max_tracks_per_hop1_artist: int = 1500,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds a 2-hop network while keeping nodes limited to {seed + hop1}.
    """
    seed = spotify_call(sp.artist, seed_artist_id)
    
    done(f"Loaded seed artist: {seed.get('name')} ({seed_artist_id})")
    status("Building Hop 1 (seed collaborations)…")

    # Hop 1 mapping and seed tracks
    collaborator_to_tracks, seed_tracks = build_hop1_from_seed_tracks(
        sp,
        seed_artist_id=seed_artist_id,
        max_seed_albums=max_seed_albums,
        max_seed_tracks=max_seed_tracks,
    )

    done(f"Hop 1 built: {len(collaborator_to_tracks)} collaborators found from {len(seed_tracks)} seed tracks")

    # Choose hop1 artists: top by hop1 weight, capped at max_hop1
    hop1_sorted = sorted(
        collaborator_to_tracks.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    hop1_ids = [artist_id for artist_id, _tracks in hop1_sorted[:max_hop1]]

    done(f"Selected {len(hop1_ids)} Hop 1 artists (cap={max_hop1})")

    # Adaptive pacing: slow down automatically for large networks
    if len(hop1_ids) >= 150:
        RATE_LIMITER.set_target_rps(1.0)
        print("🧯 Large network detected (>=150 hop1). Setting TARGET_RPS=1.0 for safety.")
    elif len(hop1_ids) >= 100:
        RATE_LIMITER.set_target_rps(1.2)
        print("🧯 Medium-large network detected (>=100 hop1). Setting TARGET_RPS=1.2 for safety.")
    else:
        RATE_LIMITER.set_target_rps(TARGET_RPS)

    allowed_ids: Set[str] = set([seed_artist_id] + hop1_ids)

    status("Fetching Hop 1 artist metadata (name/popularity/followers)…")
    
    # Fetch metadata for nodes (seed + hop1)
    hop1_artist_objs = get_artists_details_bulk(sp, hop1_ids) if hop1_ids else []

    done(f"Fetched metadata for {len(hop1_artist_objs)} Hop 1 artists")

    nodes: List[ArtistInfo] = [
        ArtistInfo(
            artist_id=seed["id"],
            name=seed["name"],
            popularity=seed.get("popularity"),
            followers=(seed.get("followers") or {}).get("total"),
            hop=0,
        )
    ]
    for a in hop1_artist_objs:
        nodes.append(
            ArtistInfo(
                artist_id=a["id"],
                name=a["name"],
                popularity=a.get("popularity"),
                followers=(a.get("followers") or {}).get("total"),
                hop=1,
            )
        )

    nodes_df = pd.DataFrame([n.__dict__ for n in nodes]).sort_values(["hop", "name"])

    # Edge storage:
    # edge_to_track_ids ensures we never double-count the same track for an edge.
    edge_to_track_ids: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    edge_to_track_rows: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    def add_edge_track(u: str, v: str, track: dict):
        if not track or not track.get("id"):
            return
        e = canonical_edge(u, v)
        track_id = track["id"]
        if track_id in edge_to_track_ids[e]:
            return  # prevent duplicates across overlapping discographies
        edge_to_track_ids[e].add(track_id)

        album = track.get("album") or {}
        edge_to_track_rows[e].append(
            {
                "source_artist_id": e[0],
                "target_artist_id": e[1],
                "track_id": track_id,
                "track_name": track.get("name"),
                "album_name": album.get("name"),
                "release_date": album.get("release_date"),
            }
        )

    status("Building edges from seed tracks…")

    # 1) Add edges from seed tracks (covers seed-hop1 cleanly)
    for tr in seed_tracks:
        artists = tr.get("artists", [])
        present = [a.get("id") for a in artists if a.get("id") and a.get("id") in allowed_ids]
        # For seed tracks, add edges between seed and any hop1 artist on the track
        for other_id in present:
            if other_id != seed_artist_id:
                add_edge_track(seed_artist_id, other_id, tr)

    done("Seed-track edges added")
    status("Scanning Hop 1 artists for Hop 2 edges (within seed+hop1)…")

    # 2) Hop 2: scan tracks for each hop1 artist, add edges among allowed ids on those tracks
    total_hop1 = len(hop1_ids)
    hop2_scan_completed = True  # we'll flip this if we stop early
    hop2_stop_reason: Optional[str] = None

    for idx, hop1_id in enumerate(hop1_ids, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total_hop1:
            print(f"⏳ Hop 2 scan: {idx}/{total_hop1} hop1 artists…")

        try:
            # (Optional) Fast pre-check to stop before a new expensive chunk begins
            if ENABLE_CALL_BUDGET and SPOTIFY_CALL_COUNT > MAX_SPOTIFY_CALLS_PER_RUN:
                raise CallBudgetExceeded(
                    f"Budget reached at {SPOTIFY_CALL_COUNT} calls (cap={MAX_SPOTIFY_CALLS_PER_RUN})."
                )

            hop1_track_ids = get_artist_track_ids(
                sp,
                artist_id=hop1_id,
                max_albums=max_albums_per_hop1_artist,
                max_tracks=max_tracks_per_hop1_artist,
                include_appears_on=False,
            )

            hop1_tracks = get_tracks_details_bulk(sp, hop1_track_ids)

            for tr in hop1_tracks:
                if not tr or not tr.get("id"):
                    continue

                artists = tr.get("artists", [])
                present = [
                    a.get("id")
                    for a in artists
                    if a.get("id") and a.get("id") in allowed_ids
                ]

                unique_present = list(dict.fromkeys(present))
                for i in range(len(unique_present)):
                    for j in range(i + 1, len(unique_present)):
                        u = unique_present[i]
                        v = unique_present[j]
                        if u != v:
                            add_edge_track(u, v, tr)

        except (CallBudgetExceeded, HardRateLimit) as e:
            hop2_scan_completed = False
            hop2_stop_reason = e.__class__.__name__
            print(f"⚠️ {e}")
            print("⚠️ Stopping Hop 2 scan early and writing partial results.")
            break
    
    if hop2_scan_completed:
        done("Hop 2 scan completed for all hop1 artists")
    else:
        print(f"ℹ️ Hop 2 scan ended early ({hop2_stop_reason}). Partial Hop 2.")

    # Build edges_df from edge_to_track_ids (weight = # unique shared tracks)
    edges_rows = []
    for (u, v), track_ids in edge_to_track_ids.items():
        edges_rows.append(
            {
                "source_artist_id": u,
                "target_artist_id": v,
                "weight": len(track_ids),
            }
        )
    edges_df = pd.DataFrame(edges_rows).sort_values(["weight", "source_artist_id", "target_artist_id"], ascending=[False, True, True])

    # Build edge_tracks_df
    edge_tracks_rows: List[dict] = []
    for _edge, rows in edge_to_track_rows.items():
        edge_tracks_rows.extend(rows)
    edge_tracks_df = pd.DataFrame(edge_tracks_rows).sort_values(
        ["source_artist_id", "target_artist_id", "release_date", "track_name"],
        ascending=[True, True, False, True],
    )

    done(f"Hop 2 complete: nodes={len(nodes_df)}, edges={len(edges_df)}, edge_tracks={len(edge_tracks_df)}")
    print(
        f"\n📊 Spotify call stats: calls={SPOTIFY_CALL_COUNT}, "
        f"throttle_sleep={SPOTIFY_SLEEP_SECONDS:.1f}s, 429s={SPOTIFY_429_COUNT}"
    )

    return nodes_df, edges_df, edge_tracks_df


def slugify(text: str) -> str:
    """
    Turn a string into a filesystem-safe folder name.
    Example: "Kendrick Lamar" -> "kendrick_lamar"
    Example: "A$AP Rocky" -> "aap_rocky"
    Example: "Guns N' Roses" -> "guns_n_roses"
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)  # non-alnum -> underscore
    text = re.sub(r"_+", "_", text)          # collapse multiple underscores
    return text.strip("_")


def get_output_dir(seed_name: str, seed_artist_id: str) -> str:
    """
    Create a stable per-seed output folder.

    We include a short prefix of the artist_id to avoid collisions in rare cases
    (two artists with the same name).
    """
    seed_slug = slugify(seed_name)
    short_id = seed_artist_id[:8]
    return os.path.join("outputs", f"{seed_slug}_{short_id}")


def main():
    sp = make_spotify_client()

    # Kendrick Lamar (you can swap this any time)
    seed_artist_id = "2YZyLoL8N0Wb9xBt1NhZWg"

    nodes_df, edges_df, edge_tracks_df = build_hop2_network(
        sp,
        seed_artist_id=seed_artist_id,
        max_hop1=200,
        max_seed_albums=200,
        max_seed_tracks=3000,
        max_albums_per_hop1_artist=25,
        max_tracks_per_hop1_artist=1000,
    )

    seed_name = nodes_df[nodes_df["hop"] == 0].iloc[0]["name"]
    out_dir = get_output_dir(seed_name, seed_artist_id)
    os.makedirs(out_dir, exist_ok=True)

    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    edge_tracks_path = os.path.join(out_dir, "edge_tracks.csv")

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    edge_tracks_df.to_csv(edge_tracks_path, index=False)

    print("Hop 2 complete.")
    print(f"Wrote: {nodes_path} ({len(nodes_df)} rows)")
    print(f"Wrote: {edges_path} ({len(edges_df)} rows)")
    print(f"Wrote: {edge_tracks_path} ({len(edge_tracks_df)} rows)")

    # Human-friendly top edges
    if not edges_df.empty:
        nodes_lookup = {row["artist_id"]: row["name"] for _, row in nodes_df.iterrows()}
        print("\nTop edges (by # shared tracks):")
        for _, row in edges_df.head(15).iterrows():
            a = nodes_lookup.get(row["source_artist_id"], row["source_artist_id"])
            b = nodes_lookup.get(row["target_artist_id"], row["target_artist_id"])
            print(f"- {a}  <->  {b}   weight={row['weight']}")


if __name__ == "__main__":
    main()