"""
analyze.py

Run network analysis on an artist collaboration graph
WITHOUT making any Spotify API calls.

Inputs (from outputs/<artist>/data/):
- nodes.csv
- edges.csv

Outputs:
- node_metrics.csv
- network_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import networkx as nx
import pandas as pd


# ----------------------------
# Graph construction
# ----------------------------

def load_graph(nodes_path: str, edges_path: str) -> nx.Graph:
    """
    Build a weighted undirected graph from CSV files.
    Edge weight = collaboration strength (# shared tracks).
    """
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        G.add_node(
            row["artist_id"],
            name=row["name"],
            hop=int(row["hop"]),
            popularity=row.get("popularity"),
            followers=row.get("followers"),
        )

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source_artist_id"],
            row["target_artist_id"],
            weight=float(row["weight"]),
        )

    return G


# ----------------------------
# Centrality computations
# ----------------------------

def compute_weighted_betweenness(G: nx.Graph) -> Dict[str, float]:
    """
    Weighted betweenness centrality.
    Distance = 1 / weight (stronger tie = closer).
    """
    # Create a temporary distance attribute
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / max(d["weight"], 1e-9)

    return nx.betweenness_centrality(
        G,
        weight="distance",
        normalized=True,
    )


def compute_weighted_eigenvector(G: nx.Graph) -> Dict[str, float]:
    """
    Weighted eigenvector centrality.
    Computed per connected component for stability.
    """
    scores: Dict[str, float] = {}

    for component in nx.connected_components(G):
        sub = G.subgraph(component)

        if len(sub) == 1:
            node = next(iter(sub.nodes))
            scores[node] = 0.0
            continue

        try:
            sub_scores = nx.eigenvector_centrality(
                sub,
                weight="weight",
                max_iter=1000,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback: numpy-based solver
            sub_scores = nx.eigenvector_centrality_numpy(
                sub,
                weight="weight",
            )

        scores.update(sub_scores)

    return scores


# ----------------------------
# Summary stats
# ----------------------------

def compute_summary_stats(G: nx.Graph) -> Dict:
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))

    components = sorted(
        (len(c) for c in nx.connected_components(G)),
        reverse=True,
    )

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "num_connected_components": len(components),
        "largest_component_size": components[0] if components else 0,
        "average_degree": sum(degrees.values()) / max(len(degrees), 1),
        "average_weighted_degree": sum(weighted_degrees.values()) / max(len(weighted_degrees), 1),
        "average_clustering": nx.average_clustering(G, weight="weight"),
    }


# ----------------------------
# Main entry point
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to outputs/<artist_folder>",
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.output_dir, "data")
    nodes_path = os.path.join(data_dir, "nodes.csv")
    edges_path = os.path.join(data_dir, "edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError("nodes.csv or edges.csv not found")

    print("⏳ Loading graph…")
    G = load_graph(nodes_path, edges_path)

    print("⏳ Computing centrality metrics…")
    betweenness = compute_weighted_betweenness(G)
    eigenvector = compute_weighted_eigenvector(G)

    print("⏳ Building node metrics table…")
    nodes_df = pd.read_csv(nodes_path)

    nodes_df["degree"] = nodes_df["artist_id"].map(dict(G.degree()))
    nodes_df["weighted_degree"] = nodes_df["artist_id"].map(dict(G.degree(weight="weight")))
    nodes_df["betweenness"] = nodes_df["artist_id"].map(betweenness)
    nodes_df["eigenvector"] = nodes_df["artist_id"].map(eigenvector)

    metrics_path = os.path.join(data_dir, "node_metrics.csv")
    nodes_df.to_csv(metrics_path, index=False)

    print("⏳ Computing network summary…")
    summary = compute_summary_stats(G)

    summary_path = os.path.join(data_dir, "network_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Analysis complete")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()