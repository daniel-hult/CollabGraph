"""
hop1.py

Build Hop 1 collaborations for a seed artist:
- Find tracks the seed appears on (via artist albums -> tracks)
- For each track, collect co-appearing artists
- Output nodes.csv, edges.csv, edge_tracks.csv into outputs/

Note: This is "good enough" for Hop 1. We'll improve coverage later.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# ----------------------------
# Data shapes (simple + explicit)
# ----------------------------

@dataclass(frozen=True)
class ArtistInfo:
    artist_id: str
    name: str
    popularity: Optional[int]
    followers: Optional[int]
    hop: int  # 0 for seed, 1 for hop1


# ----------------------------
# Spotify client
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
    return spotipy.Spotify(auth_manager=auth_manager)


# ----------------------------
# Helpers: paging through Spotify results
# ----------------------------

def iter_paged(sp: spotipy.Spotify, first_page: dict) -> Iterable[dict]:
    """
    Spotify returns many endpoints as paged results:
    { "items": [...], "next": "url" or None, ... }

    This generator yields every item across all pages.
    """
    page = first_page
    while page:
        for item in page.get("items", []):
            yield item
        if page.get("next"):
            page = sp.next(page)
        else:
            break


# ----------------------------
# Core: Hop 1 track collection
# ----------------------------

def get_seed_albums(sp: spotipy.Spotify, seed_artist_id: str) -> List[dict]:
    """
    Returns album objects for the seed artist.
    We include albums + singles/eps because collabs often show up there.
    """
    first_page = sp.artist_albums(
        seed_artist_id,
        album_type="album,single,compilation",
        country="US",
        limit=50,
    )

    # Deduplicate albums by Spotify album id (same album can appear multiple times)
    albums_by_id: Dict[str, dict] = {}
    for album in iter_paged(sp, first_page):
        albums_by_id[album["id"]] = album
    return list(albums_by_id.values())


def get_album_tracks(sp: spotipy.Spotify, album_id: str) -> List[dict]:
    first_page = sp.album_tracks(album_id, limit=50)
    return list(iter_paged(sp, first_page))


def get_track_details_bulk(sp: spotipy.Spotify, track_ids: List[str]) -> List[dict]:
    """
    Spotify tracks endpoint supports up to 50 ids per call.
    We'll fetch in chunks.
    """
    results: List[dict] = []
    chunk_size = 50
    for i in range(0, len(track_ids), chunk_size):
        chunk = track_ids[i : i + chunk_size]
        response = sp.tracks(chunk)
        results.extend(response.get("tracks", []))
    return results


def get_artists_details_bulk(sp: spotipy.Spotify, artist_ids: List[str]) -> List[dict]:
    """
    Spotify artists endpoint supports up to 50 ids per call.
    We'll fetch in chunks.
    """
    results: List[dict] = []
    chunk_size = 50
    for i in range(0, len(artist_ids), chunk_size):
        chunk = artist_ids[i : i + chunk_size]
        response = sp.artists(chunk)
        results.extend(response.get("artists", []))
    return results


def build_hop1(
    sp: spotipy.Spotify,
    seed_artist_id: str,
    max_albums: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds hop1 dataframes:
    - nodes_df: seed + collaborators
    - edges_df: seed->collaborator with weight = shared track count
    - edge_tracks_df: one row per shared track per collaborator
    """
    seed = sp.artist(seed_artist_id)

    seed_info = ArtistInfo(
        artist_id=seed["id"],
        name=seed["name"],
        popularity=seed.get("popularity"),
        followers=(seed.get("followers") or {}).get("total"),
        hop=0,
    )

    albums = get_seed_albums(sp, seed_artist_id)[:max_albums]

    # Collect all track ids from all albums
    track_ids: List[str] = []
    for album in albums:
        album_tracks = get_album_tracks(sp, album["id"])
        for t in album_tracks:
            if t.get("id"):
                track_ids.append(t["id"])

    # Deduplicate track ids
    track_ids = list(dict.fromkeys(track_ids))
    #max_tracks = 2000
    #track_ids = track_ids[:max_tracks]
    
    # Fetch full track details (includes album + release_date + full artists list)
    tracks = get_track_details_bulk(sp, track_ids)

    # For each collaborator: track list + weight count
    collaborator_to_tracks: Dict[str, List[dict]] = defaultdict(list)

    for track in tracks:
        if not track or not track.get("id"):
            continue

        # Track-level artists list (main listed artists; not "credits")
        artists = track.get("artists", [])
        artist_ids = [a["id"] for a in artists if a.get("id")]

        # Only count tracks where the seed is actually listed as an artist
        if seed_artist_id not in artist_ids:
            continue

        for a in artists:
            a_id = a.get("id")
            if not a_id or a_id == seed_artist_id:
                continue
            collaborator_to_tracks[a_id].append(track)

    # Build nodes: seed + unique collaborators
    collaborator_ids = sorted(collaborator_to_tracks.keys())
    collaborator_artists = get_artists_details_bulk(sp, collaborator_ids) if collaborator_ids else []

    nodes: List[ArtistInfo] = [seed_info]
    for a in collaborator_artists:
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

    # Build edges: seed -> collaborator with weight
    edges_rows = []
    for collab_id, shared_tracks in collaborator_to_tracks.items():
        edges_rows.append(
            {
                "source_artist_id": seed_artist_id,
                "target_artist_id": collab_id,
                "weight": len(shared_tracks),
            }
        )
    edges_df = pd.DataFrame(edges_rows).sort_values(["weight", "target_artist_id"], ascending=[False, True])

    # Build edge_tracks: one row per shared track per collaborator
    edge_tracks_rows = []
    for collab_id, shared_tracks in collaborator_to_tracks.items():
        for tr in shared_tracks:
            album = tr.get("album") or {}
            edge_tracks_rows.append(
                {
                    "source_artist_id": seed_artist_id,
                    "target_artist_id": collab_id,
                    "track_id": tr.get("id"),
                    "track_name": tr.get("name"),
                    "album_name": album.get("name"),
                    "release_date": album.get("release_date"),
                }
            )
    edge_tracks_df = pd.DataFrame(edge_tracks_rows).sort_values(
        ["target_artist_id", "release_date", "track_name"],
        ascending=[True, False, True],
    )

    return nodes_df, edges_df, edge_tracks_df


def main():
    sp = make_spotify_client()

    # Use Kendrick by default (you can override later)
    seed_artist_id = "2YZyLoL8N0Wb9xBt1NhZWg"

    nodes_df, edges_df, edge_tracks_df = build_hop1(sp, seed_artist_id=seed_artist_id)

    os.makedirs("outputs", exist_ok=True)
    nodes_path = os.path.join("outputs", "nodes.csv")
    edges_path = os.path.join("outputs", "edges.csv")
    edge_tracks_path = os.path.join("outputs", "edge_tracks.csv")

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    edge_tracks_df.to_csv(edge_tracks_path, index=False)

    print("Hop 1 complete.")
    print(f"Wrote: {nodes_path} ({len(nodes_df)} rows)")
    print(f"Wrote: {edges_path} ({len(edges_df)} rows)")
    print(f"Wrote: {edge_tracks_path} ({len(edge_tracks_df)} rows)")

    if not edges_df.empty:
        top = edges_df.head(10)
        nodes_lookup = {
            row["artist_id"]: row["name"]
            for _, row in nodes_df.iterrows()
        }

        print("\nTop collaborators (by # shared tracks):")
        for _, row in top.iterrows():
            name = nodes_lookup.get(row["target_artist_id"], "UNKNOWN")
            print(f"- {name}  weight={row['weight']}")


if __name__ == "__main__":
    main()