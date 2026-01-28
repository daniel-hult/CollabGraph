"""
test_auth.py

Minimal sanity check:
- Loads Spotify credentials from .env
- Authenticates with Spotify
- Fetches a single artist by ID
"""

import os

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def main():
    # Load environment variables from .env into os.environ
    load_dotenv()

    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Spotify credentials. "
            "Check that SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET "
            "are set in your .env file."
        )

    # Set up Spotify client using Client Credentials flow
    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)

    # Example seed artist (Taylor Swift)
    seed_artist_id = "2YZyLoL8N0Wb9xBt1NhZWg"

    artist = sp.artist(seed_artist_id)

    print("Spotify authentication successful!")
    print(f"Artist name: {artist['name']}")
    print(f"Popularity: {artist['popularity']}")
    print(f"Followers: {artist['followers']['total']}")


if __name__ == "__main__":
    main()