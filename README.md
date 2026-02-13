# 🎧 CollabGraph

### Interactive Artist Collaboration Networks (Powered by Spotify Data)

> Explore how artists connect through collaboration: visually,
> analytically, and interactively.

🔗 **Live Demo (Kendrick Lamar Network):**
https://daniel-hult.github.io/CollabGraph/network.html

------------------------------------------------------------------------

## 📸 Example Network: Kendrick Lamar

<img width="1135" height="731" alt="Screenshot 2026-02-14 at 00 03 30" src="https://github.com/user-attachments/assets/67cb0214-3ca2-4905-9059-2f8f514d5711" />

------------------------------------------------------------------------

## 📌 What Is This?

**CollabGraph** builds and visualizes 2-hop collaboration networks for
music artists using Spotify's API.

Given a seed artist, the project:

1.  Collects all collaborators (Hop 1)
2.  Identifies collaborations among those collaborators (Hop 2)
3.  Builds a weighted network graph
4.  Computes network science metrics
5.  Produces an interactive HTML visualization

The result is a clean, dynamic network where:

-   Node size = Spotify popularity
-   Edge thickness = number of shared tracks
-   Tooltip = detailed network role analysis
-   Sidebar = glossary + search functionality

------------------------------------------------------------------------

## 🧠 What Insights Does It Provide?

Each artist in the network includes:

-   🎵 **Popularity (0--100)**
-   👥 **Follower count**
-   🌉 **Bridge Score** (Betweenness Centrality)
-   ⭐ **Influence Score** (Eigenvector Centrality)
-   🧩 Plain-English interpretation of their network role

This allows you to identify:

-   Core hubs
-   Key connectors between sub-scenes
-   Peripheral collaborators
-   Tight local clusters

It's not just who worked together, but it's also who matters structurally.

------------------------------------------------------------------------

## 🔬 Network Methodology

### Graph Construction

-   Undirected weighted graph
-   Edge weight = number of shared tracks
-   Distance for shortest-path metrics = 1 / weight

### Centrality Metrics (Weighted)

-   **Betweenness Centrality** → Measures how often an artist acts as a
    bridge between others
-   **Eigenvector Centrality** → Measures influence based on connections
    to other influential artists

Centrality values are converted into percentile buckets:

-   Very Low
-   Low
-   Medium
-   High
-   Very High

Each combination maps to a human-readable interpretation.

------------------------------------------------------------------------

## 🖥 Features

### ✨ Interactive HTML Visualization

-   Hover tooltips with full analysis
-   Click to pin artist card
-   Spotify profile link
-   Search any artist in the network
-   Minimal collapsible info sidebar

### 📊 Static PNG Export

High-resolution version for sharing.

### 📁 Structured Outputs

For each seed artist:

    outputs/<artist>/
    ├── data/
    │   ├── nodes.csv
    │   ├── edges.csv
    │   ├── edge_tracks.csv
    │   ├── node_metrics.csv
    │   ├── node_tooltips.csv
    │   └── network_summary.json
    ├── network.html
    └── network.png

------------------------------------------------------------------------

## ⚙️ How It Works

### 1️⃣ Data Collection (`hop2.py`)

-   Pull albums and tracks via Spotify API
-   Build 2-hop collaboration graph
-   Save structured CSV outputs
-   Includes custom rate-limiting safeguards

### 2️⃣ Network Analysis (`analyze.py`)

-   Computes weighted centrality metrics using NetworkX
-   Generates percentile buckets
-   Produces tooltip-ready enrichment dataset

### 3️⃣ Visualization (`visualize.py`)

-   Generates interactive HTML using PyVis
-   Applies custom tooltip design
-   Builds collapsible info panel
-   Exports static PNG

------------------------------------------------------------------------

## 🚀 Running Locally

``` bash
git clone https://github.com/daniel-hult/CollabGraph.git
cd CollabGraph

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

    SPOTIPY_CLIENT_ID=your_client_id
    SPOTIPY_CLIENT_SECRET=your_client_secret

Run:

``` bash
PYTHONPATH=src python src/collabgraph/run.py --seed <spotify_artist_id>
PYTHONPATH=src python -m collabgraph.analyze --output-dir outputs/<artist_folder>
PYTHONPATH=src python -m collabgraph.visualize --seed <spotify_artist_id>
```

------------------------------------------------------------------------

## 📦 Tech Stack

-   Python
-   Spotipy
-   NetworkX
-   Pandas
-   PyVis
-   Matplotlib
-   Vanilla HTML/CSS/JS
-   GitHub Pages

------------------------------------------------------------------------

## 🎯 Design Philosophy

The goal was to create:

-   A visually compelling network
-   Accessible metrics (no raw math exposure)
-   A product-feeling interactive experience
-   Something shareable outside of GitHub

------------------------------------------------------------------------

## 👤 Author

Daniel Hult
Business Analyst & Data Enthusiast
Stockholm, Sweden
