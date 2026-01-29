"""
run.py

One-command runner for CollabGraph (Option A: seed artist ID only).

Examples:
  python src/collabgraph/run.py --seed 2YZyLoL8N0Wb9xBt1NhZWg
  python src/collabgraph/run.py
"""

from __future__ import annotations

import argparse
import sys

from collabgraph.hop2 import make_spotify_client, build_hop2_network, get_output_dir
from collabgraph.visualize import (
    find_output_dir_by_seed_id,
    load_graph_data,
    build_networkx_graph,
    write_pyvis_html,
    write_static_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CollabGraph runner (Hop 2 + visualization)")
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Spotify seed artist ID (e.g., 2YZyLoL8N0Wb9xBt1NhZWg). If omitted, you will be prompted.",
    )

    # Keep these configurable but with safe defaults
    parser.add_argument("--max-hop1", type=int, default=200)
    parser.add_argument("--max-seed-albums", type=int, default=200)
    parser.add_argument("--max-seed-tracks", type=int, default=3000)
    parser.add_argument("--max-hop1-albums", type=int, default=25)
    parser.add_argument("--max-hop1-tracks", type=int, default=1000)

    return parser.parse_args()


def prompt_for_seed() -> str:
    seed = input("Paste Spotify seed artist ID: ").strip()
    if not seed:
        print("No seed provided. Exiting.")
        sys.exit(1)
    return seed


def main():
    args = parse_args()
    seed_artist_id = args.seed or prompt_for_seed()

    sp = make_spotify_client()

    # Run Hop 2 build
    nodes_df, edges_df, edge_tracks_df = build_hop2_network(
        sp,
        seed_artist_id=seed_artist_id,
        max_hop1=args.max_hop1,
        max_seed_albums=args.max_seed_albums,
        max_seed_tracks=args.max_seed_tracks,
        max_albums_per_hop1_artist=args.max_hop1_albums,
        max_tracks_per_hop1_artist=args.max_hop1_tracks,
    )

    # Determine per-seed output folder using the seed node (hop=0) in nodes_df
    seed_name = nodes_df[nodes_df["hop"] == 0].iloc[0]["name"]
    out_dir = get_output_dir(seed_name, seed_artist_id)

    import os
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Write CSVs (overwrite within this seed folder)
    nodes_path = f"{data_dir}/nodes.csv"
    edges_path = f"{data_dir}/edges.csv"
    edge_tracks_path = f"{data_dir}/edge_tracks.csv"

    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    edge_tracks_df.to_csv(edge_tracks_path, index=False)

    print("\nData outputs written:")
    print(f"- {nodes_path}")
    print(f"- {edges_path}")
    print(f"- {edge_tracks_path}")

    # Build visualizations from the written data
    # (We re-load to ensure the visualization step matches persisted outputs.)
    out_dir_verified = find_output_dir_by_seed_id(seed_artist_id)
    nodes_df2, edges_df2 = load_graph_data(out_dir_verified)
    G = build_networkx_graph(nodes_df2, edges_df2)

    html_path = write_pyvis_html(G, out_dir=out_dir_verified, filename="network.html")
    png_path = write_static_png(G, out_dir=out_dir_verified, filename="network.png")

    print("\nVisualizations written:")
    print(f"- {html_path}")
    print(f"- {png_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()