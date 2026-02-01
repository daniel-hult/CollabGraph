"""
visualize.py

Reads per-seed outputs and generates:
- Interactive HTML network using PyVis
- Static PNG using NetworkX + Matplotlib

Node size is based on Spotify popularity (0–100).

Usage:
  python src/collabgraph/visualize.py
"""

from __future__ import annotations

import os
import re
import argparse
from typing import Dict, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

THEME = {
    "bg": "#121212",
    "text": "#FFFFFF",
    "seed_node": "#1ED760",    # official Spotify green
    "hop1_node": "#3FA9E6",    # baby blue (not neon)
    "node_border": "#121212",  # official Spotify black
    "edge_rgb": (172, 173, 172), # light gray
}


def slugify(text: str) -> str:
    """
    Turn a string into a filesystem-safe folder name.
    Example: "Kendrick Lamar" -> "kendrick_lamar"
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def safe_matplotlib_label(text: str) -> str:
    """
    Matplotlib treats '$' as math-mode. Escape it so artist names render safely.
    """
    if text is None:
        return ""
    return str(text).replace("$", r"\$")


def get_output_dir(seed_name: str, seed_artist_id: str) -> str:
    seed_slug = slugify(seed_name)
    short_id = seed_artist_id[:8]
    return os.path.join("outputs", f"{seed_slug}_{short_id}")


def find_output_dir_by_seed_id(seed_artist_id: str) -> str:
    """
    Find the per-seed output folder by matching the short id prefix used in folder naming.
    Example folder: outputs/kendrick_lamar_2YZyLoL8
    """
    short_id = seed_artist_id[:8]
    base = "outputs"

    if not os.path.isdir(base):
        raise FileNotFoundError("outputs/ folder not found. Run hop2.py first.")

    candidates = []
    for name in os.listdir(base):
        if name.endswith(f"_{short_id}"):
            candidates.append(os.path.join(base, name))

    if not candidates:
        raise FileNotFoundError(
            f"No outputs folder found for seed id prefix {short_id}. "
            "Run hop2.py for that seed first."
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple output folders match seed id prefix {short_id}: {candidates}"
        )

    return candidates[0]


def popularity_to_node_size(popularity: float) -> float:
    """
    Map Spotify popularity (0-100) to a reasonable node size.
    We keep a minimum so small artists are still visible.
    """
    # Clamp defensively
    p = max(0.0, min(100.0, float(popularity)))
    return 10 + (p * 0.6)  # 10..70


def edge_style(weight: float) -> Dict[str, float]:
    """
    Map edge weight (# shared tracks) to styling.
    """
    w = float(weight)
    width = min(20.0, 1.6 + (w * 0.4))
    opacity = min(0.9, 0.2 + (w * 0.05))
    return {"width": width, "opacity": opacity}


def load_graph_data(out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = os.path.join(out_dir, "data", "nodes.csv")
    edges_path = os.path.join(out_dir, "data", "edges.csv")

    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        raise FileNotFoundError(
            f"Missing nodes.csv or edges.csv in {out_dir}. "
            "Run hop2.py first for this seed."
        )

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    return nodes_df, edges_df


def build_networkx_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected NetworkX graph with node attributes.
    """
    G = nx.Graph()

    for _, row in nodes_df.iterrows():
        artist_id = row["artist_id"]
        G.add_node(
            artist_id,
            name=row["name"],
            popularity=0 if pd.isna(row.get("popularity")) else row.get("popularity", 0),
            followers=0 if pd.isna(row.get("followers")) else row.get("followers", 0),
            hop=int(row["hop"]),
        )

    for _, row in edges_df.iterrows():
        G.add_edge(
            row["source_artist_id"],
            row["target_artist_id"],
            weight=float(row["weight"]),
        )

    return G


def write_pyvis_html(G: nx.Graph, out_dir: str, filename: str = "network.html") -> str:
    """
    Interactive HTML graph.
    """
    net = Network(height="800px", width="100%", bgcolor=THEME["bg"], font_color=THEME["text"])

    # Physics makes it readable; users can drag nodes around.
    net.force_atlas_2based()

    net.set_options("""
    var options = {
    "physics": {
        "forceAtlas2Based": {
        "gravitationalConstant": -40,
        "centralGravity": 0.01,
        "springLength": 140,
        "springConstant": 0.06,
        "avoidOverlap": 0.0
        },
        "minVelocity": 0.5,
        "solver": "forceAtlas2Based"
    }
    }
    """)

    for node_id, attrs in G.nodes(data=True):
        popularity = attrs.get("popularity", 0) or 0
        hop = attrs.get("hop", 1)

        size = popularity_to_node_size(popularity)

        # Simple coloring by hop for now (seed vs others)
        color = THEME["seed_node"] if hop == 0 else THEME["hop1_node"]

        title = (
            f"{attrs.get('name', node_id)}<br>"
            f"Popularity: {popularity}<br>"
            f"Followers: {attrs.get('followers', '')}<br>"
            f"Hop: {hop}"
        )

        net.add_node(
            node_id,
            label=attrs.get("name", node_id),
            title=title,
            size=size,
            color={
                "background": color,
                "border": THEME["node_border"],
                "highlight": {"background": color, "border": THEME["text"]},
                "hover": {"background": color, "border": THEME["text"]},
            },
            font={"color": THEME["text"], "size": 16, "face": "system-ui"},
        )

    for u, v, attrs in G.edges(data=True):
        weight = attrs.get("weight", 1)
        # Use weight to make edges thicker in the interactive view
        style = edge_style(weight)
        r, g, b = THEME["edge_rgb"]
        net.add_edge(
            u,
            v,
            value=weight,
            title=f"Shared tracks: {weight}",
            width=style["width"],
            color=f"rgba({r}, {g}, {b}, {style['opacity']})",
        )

    out_path = os.path.join(out_dir, filename)
    net.write_html(out_path)
    return out_path


def write_static_png(G: nx.Graph, out_dir: str, filename: str = "network.png") -> str:
    """
    Static PNG using matplotlib.
    We'll use a force-directed layout (spring_layout).
    """
    # Layout: deterministic-ish for the same graph
    pos = nx.spring_layout(G, seed=42, k=0.6)

    # Node sizes based on popularity
    sizes = []
    colors = []
    labels = {}

    for node_id, attrs in G.nodes(data=True):
        popularity = attrs.get("popularity", 0) or 0
        hop = attrs.get("hop", 1)
        sizes.append(popularity_to_node_size(popularity) * 20)  # scale for matplotlib
        colors.append(THEME["seed_node"] if hop == 0 else THEME["hop1_node"])
        labels[node_id] = safe_matplotlib_label(attrs.get("name", node_id))

    # Edge widths based on weight (cap so it doesn't get ridiculous)
    widths = []
    for _, _, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1))
        widths.append(min(8.0, 0.3 + (w * 0.15)))

    plt.figure(figsize=(18, 12), dpi=200)
    ax = plt.gca()
    ax.set_facecolor(THEME["bg"])
    plt.gcf().patch.set_facecolor(THEME["bg"])
    
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.28, edge_color=THEME["hop1_node"])
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        node_color=colors,
        alpha=0.92,
        linewidths=1.0,
        edgecolors=THEME["node_border"],
    )

    # Labels: only label the seed + top-degree nodes to keep it readable
    degree_by_node = dict(G.degree())
    top_nodes = set(sorted(degree_by_node, key=degree_by_node.get, reverse=True)[:15])

    label_subset = {n: labels[n] for n in G.nodes() if (G.nodes[n].get("hop") == 0 or n in top_nodes)}
    nx.draw_networkx_labels(G, pos, labels=label_subset, font_size=8, font_color=THEME["text"])

    plt.axis("off")
    out_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CollabGraph visualizations from saved CSV outputs.")
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Spotify seed artist ID. If provided, visualize outputs for that seed folder.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If provided, use the seed to locate the correct output folder automatically
    if args.seed:
        out_dir = find_output_dir_by_seed_id(args.seed)
    else:
        # Fallback: keep your existing default behavior (or pick the most recent folder later)
        seed_artist_id = "2YZyLoL8N0Wb9xBt1NhZWg"
        out_dir = find_output_dir_by_seed_id(seed_artist_id)

    nodes_df, edges_df = load_graph_data(out_dir)
    G = build_networkx_graph(nodes_df, edges_df)

    html_path = write_pyvis_html(G, out_dir=out_dir, filename="network.html")
    png_path = write_static_png(G, out_dir=out_dir, filename="network.png")

    print("Visualization complete.")
    print(f"Wrote: {html_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()