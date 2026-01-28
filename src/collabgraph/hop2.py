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
    Minimal retry/backoff wrapper to handle rate limits (429) and transient errors.
    Keeps the logic readable and prevents the project from blowing up on Hop 2.

    - If 429: sleep and retry
    - Otherwise: re-raise
    """
    delay_seconds = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:
                retry_after = None
                if hasattr(e, "headers") and e.headers:
                    retry_after = e.headers.get("Retry-After")

                if retry_after is not None:
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = None

                # If Spotify says something insane (hours), don't hang the script.
                if retry_after and retry_after > 120:
                    raise RuntimeError(
                        f"Spotify rate limit hit with Retry-After={retry_after:.0f}s (very large). "
                        "Stop the run and try again later, or reduce caps."
                    )

                sleep_for = retry_after if retry_after is not None else delay_seconds
                sleep_for = min(sleep_for, 60)

                print(f"[rate-limit] 429 from Spotify. Sleeping {sleep_for:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(sleep_for)
                delay_seconds = min(delay_seconds * 2, 30)
                continue

            # Optional: retry some 5xx errors (rare but happens)
            if e.http_status and 500 <= e.http_status < 600 and attempt < max_retries:
                print(f"[spotify] {e.http_status} server error. Sleeping {delay_seconds:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 30)
                continue

            raise


# ----------------------------
# Helpers: paging + bulk endpoints
# ----------------------------

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

def get_artist_albums(sp: spotipy.Spotify, artist_id: str, max_albums: int) -> List[dict]:
    first_page = spotify_call(
        sp.artist_albums,
        artist_id,
        album_type="album,single,compilation",
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
) -> List[str]:
    """
    Returns a deduped list of track IDs for an artist via albums -> tracks.
    Controlled by caps to limit API usage.
    """
    albums = get_artist_albums(sp, artist_id, max_albums=max_albums)

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

    # Hop 1 mapping and seed tracks
    collaborator_to_tracks, seed_tracks = build_hop1_from_seed_tracks(
        sp,
        seed_artist_id=seed_artist_id,
        max_seed_albums=max_seed_albums,
        max_seed_tracks=max_seed_tracks,
    )

    # Choose hop1 artists: top by hop1 weight, capped at max_hop1
    hop1_sorted = sorted(
        collaborator_to_tracks.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    hop1_ids = [artist_id for artist_id, _tracks in hop1_sorted[:max_hop1]]

    allowed_ids: Set[str] = set([seed_artist_id] + hop1_ids)

    # Fetch metadata for nodes (seed + hop1)
    hop1_artist_objs = get_artists_details_bulk(sp, hop1_ids) if hop1_ids else []

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

    # 1) Add edges from seed tracks (covers seed-hop1 cleanly)
    for tr in seed_tracks:
        artists = tr.get("artists", [])
        present = [a.get("id") for a in artists if a.get("id") and a.get("id") in allowed_ids]
        # For seed tracks, add edges between seed and any hop1 artist on the track
        for other_id in present:
            if other_id != seed_artist_id:
                add_edge_track(seed_artist_id, other_id, tr)

    # 2) Hop 2: scan tracks for each hop1 artist, add edges among allowed ids on those tracks
    for hop1_id in hop1_ids:
        hop1_track_ids = get_artist_track_ids(
            sp,
            artist_id=hop1_id,
            max_albums=max_albums_per_hop1_artist,
            max_tracks=max_tracks_per_hop1_artist,
        )

        hop1_tracks = get_tracks_details_bulk(sp, hop1_track_ids)

        for tr in hop1_tracks:
            if not tr or not tr.get("id"):
                continue

            artists = tr.get("artists", [])
            present = [a.get("id") for a in artists if a.get("id") and a.get("id") in allowed_ids]

            # Create edges for every pair among present artists
            # (Typically 2–4 artists per track, so this is cheap.)
            unique_present = list(dict.fromkeys(present))
            for i in range(len(unique_present)):
                for j in range(i + 1, len(unique_present)):
                    u = unique_present[i]
                    v = unique_present[j]
                    if u != v:
                        add_edge_track(u, v, tr)

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