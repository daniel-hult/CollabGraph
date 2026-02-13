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
import html
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


def load_tooltip_data(out_dir: str) -> Dict[str, Dict]:
    """
    Load tooltip-ready fields keyed by artist_id from node_tooltips.csv.
    Returns: { artist_id: {field: value, ...}, ... }
    """
    tooltips_path = os.path.join(out_dir, "data", "node_tooltips.csv")
    if not os.path.exists(tooltips_path):
        return {}

    df = pd.read_csv(tooltips_path, dtype={"artist_id": str})
    # Replace NaNs with empty strings so JSON serialization is clean
    df = df.where(pd.notna(df), "")

    tooltip_map: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        d = row.to_dict()
        artist_id = str(d.get("artist_id", "")).strip()
        if not artist_id:
            continue
        tooltip_map[artist_id] = d
    return tooltip_map


def load_node_tooltips(out_dir: str) -> Dict[str, dict]:
    """
    Load tooltip-ready fields produced by analyze.py.

    Expected path:
      outputs/<seed_folder>/data/node_tooltips.csv

    Returns:
      artist_id -> row dict (all values JSON-safe)
    """
    tooltips_path = os.path.join(out_dir, "data", "node_tooltips.csv")
    if not os.path.exists(tooltips_path):
        return {}

    df = pd.read_csv(tooltips_path)

    # Important: make values JSON/JS friendly (no NaNs)
    df = df.fillna("")

    # Ensure artist_id is treated as string key
    out: Dict[str, dict] = {}
    for _, row in df.iterrows():
        artist_id = str(row["artist_id"])
        out[artist_id] = row.to_dict()
    return out


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


def inject_sidebar_and_search(html_path: str) -> None:
    """
    Post-process the PyVis HTML to add:
    - a right-side slide-out info panel
    - a search box that focuses/selects a node by artist name
    """
    with open(html_path, "r", encoding="utf-8") as f:
        s = f.read()

    # ---------- 1) CSS ----------
    css = """
<style>
  :root{
    --cg-bg: rgba(18,18,18,0.92);
    --cg-panel: rgba(20,20,20,0.92);
    --cg-border: rgba(255,255,255,0.10);
    --cg-text: #ffffff;
    --cg-muted: rgba(255,255,255,0.65);
    --cg-accent: #1ED760;
  }

  /* Right-side toggle button */
  #cgInfoBtn{
    position: fixed;
    top: 16px;
    right: 16px;
    z-index: 9999;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    border: 1px solid var(--cg-border);
    background: rgba(0,0,0,0.35);
    color: var(--cg-text);
    display: grid;
    place-items: center;
    cursor: pointer;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    user-select: none;
  }
  #cgInfoBtn:hover{ background: rgba(0,0,0,0.48); }

  /* Slide-out panel */
  #cgSidePanel{
    position: fixed;
    top: 0;
    right: 0;
    height: 100vh;
    width: 360px;
    max-width: 92vw;
    z-index: 9998;
    transform: translateX(110%);
    transition: transform 180ms ease;
    background: var(--cg-panel);
    border-left: 1px solid var(--cg-border);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: -20px 0 50px rgba(0,0,0,0.35);
    padding: 18px 16px;
    overflow: auto;
    color: var(--cg-text);
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  #cgSidePanel.open{ transform: translateX(0); }

  #cgPanelHeader{
    display:flex;
    align-items:center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 12px;
  }
  #cgPanelTitle{
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 0;
  }
  #cgCloseBtn{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    border: 1px solid var(--cg-border);
    background: rgba(0,0,0,0.25);
    color: var(--cg-text);
    cursor: pointer;
  }
  #cgCloseBtn:hover{ background: rgba(0,0,0,0.40); }

  .cgSection{
    border-top: 1px solid var(--cg-border);
    padding-top: 14px;
    margin-top: 14px;
  }
  .cgLabel{
    color: var(--cg-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 11px;
    margin: 0 0 8px 0;
  }
  .cgBody{
    color: rgba(255,255,255,0.85);
    font-size: 13px;
    line-height: 1.45;
    margin: 0;
  }

  /* Search */
  #cgSearchInput{
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--cg-border);
    background: rgba(0,0,0,0.28);
    color: var(--cg-text);
    padding: 10px 12px;
    outline: none;
    font-size: 13px;
  }
  #cgSearchInput::placeholder{ color: rgba(255,255,255,0.45); }
  #cgSearchHint{
    margin-top: 8px;
    color: rgba(255,255,255,0.55);
    font-size: 12px;
  }
  #cgSearchResult{
    margin-top: 10px;
    font-size: 12px;
    color: rgba(255,255,255,0.70);
  }

  .cgTerm{
    margin: 10px 0 0 0;
    padding: 10px 10px;
    border: 1px solid var(--cg-border);
    border-radius: 14px;
    background: rgba(0,0,0,0.22);
  }
  .cgTerm b{
    color: var(--cg-text);
    font-size: 13px;
  }
  .cgTerm p{
    margin: 6px 0 0 0;
    color: rgba(255,255,255,0.78);
    font-size: 12px;
    line-height: 1.4;
  }
</style>
"""

    # ---------- 2) HTML ----------
    panel_html = """
<div id="cgInfoBtn" title="Info / Search" aria-label="Info / Search">
  <!-- simple info icon -->
  <span style="font-weight:800;font-size:18px;line-height:1;">i</span>
</div>

<div id="cgSidePanel" aria-label="Info panel">
  <div id="cgPanelHeader">
    <h2 id="cgPanelTitle">CollabGraph</h2>
    <button id="cgCloseBtn" title="Close" aria-label="Close">✕</button>
  </div>

  <div class="cgSection" style="border-top:none;margin-top:0;padding-top:0;">
    <div class="cgLabel">Search artist</div>
    <input id="cgSearchInput" type="text" placeholder="Type an artist name… (e.g., J. Cole)" />
    <div id="cgSearchHint">Press Enter to jump. Click a node to pin the tooltip.</div>
    <div id="cgSearchResult"></div>
  </div>

  <div class="cgSection">
    <div class="cgLabel">What am I looking at?</div>
    <p class="cgBody">
      This network shows artists connected by collaborations. Each line indicates at least one shared track.
      Thicker lines mean more shared tracks. Node size reflects Spotify popularity.
    </p>
  </div>

  <div class="cgSection">
    <div class="cgLabel">Tooltip glossary</div>

    <div class="cgTerm">
      <b>Popularity</b>
      <p>Spotify’s 0–100 popularity score (higher means more listened-to recently).</p>
    </div>

    <div class="cgTerm">
      <b>Followers</b>
      <p>Total Spotify followers (a rough proxy for long-term audience size).</p>
    </div>

    <div class="cgTerm">
      <b>Bridge score</b>
      <p>How strongly this artist connects different parts of the network (high = important connector).</p>
    </div>

    <div class="cgTerm">
      <b>Influence score</b>
      <p>How “central” an artist is based on being connected to other central artists (high = influential hub).</p>
    </div>
  </div>
</div>
"""

    # ---------- 3) JS (hooks into the existing vis network) ----------
    js = """
<script>
(function(){
  const btn = document.getElementById('cgInfoBtn');
  const panel = document.getElementById('cgSidePanel');
  const closeBtn = document.getElementById('cgCloseBtn');
  const input = document.getElementById('cgSearchInput');
  const result = document.getElementById('cgSearchResult');

  function openPanel(){ panel.classList.add('open'); }
  function closePanel(){ panel.classList.remove('open'); }

  btn && btn.addEventListener('click', () => {
    if(panel.classList.contains('open')) closePanel();
    else openPanel();
  });
  closeBtn && closeBtn.addEventListener('click', closePanel);

  // Close on ESC
  document.addEventListener('keydown', (e) => {
    if(e.key === 'Escape') closePanel();
  });

  function normalize(s){
    return (s || '').toString().trim().toLowerCase();
  }

  function findNodeIdByName(name){
    if(typeof nodes === 'undefined' || !nodes || !nodes.get) return null;
    const target = normalize(name);
    if(!target) return null;

    const all = nodes.get();
    // exact match first
    let hit = all.find(n => normalize(n.label) === target);
    if(hit) return hit.id;

    // contains match second
    hit = all.find(n => normalize(n.label).includes(target));
    if(hit) return hit.id;

    return null;
  }

  function focusNode(nodeId){
    if(typeof network === 'undefined' || !network) return;

    // Focus + select
    network.selectNodes([nodeId]);
    network.focus(nodeId, {
      scale: 1.35,
      animation: { duration: 450, easingFunction: "easeInOutQuad" }
    });

    // If your tooltip JS exposes showForNode(nodeId, pinnedBool), pin it.
    if(typeof showForNode === 'function'){
      try { showForNode(nodeId, true); } catch(e) {}
    }
  }

  input && input.addEventListener('keydown', (e) => {
    if(e.key !== 'Enter') return;
    const q = input.value;
    const nodeId = findNodeIdByName(q);
    if(!nodeId){
      result.textContent = "No match found. Try a shorter name.";
      return;
    }
    result.textContent = "";
    focusNode(nodeId);
  });
})();
</script>
"""

    # ---------- Inject CSS before </head> ----------
    if "</head>" in s:
        s = s.replace("</head>", css + "\n</head>", 1)

    # ---------- Inject panel HTML right after <body> ----------
    # Handle <body> with attributes too
    body_match = re.search(r"<body[^>]*>", s, flags=re.IGNORECASE)
    if body_match:
        insert_at = body_match.end()
        s = s[:insert_at] + "\n" + panel_html + "\n" + s[insert_at:]
    else:
        # fallback: append at start if body not found
        s = panel_html + "\n" + s

    # ---------- Inject JS near the end, before </body> ----------
    if "</body>" in s:
        s = s.replace("</body>", js + "\n</body>", 1)
    else:
        s += "\n" + js

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(s)


def write_pyvis_html(G: nx.Graph, out_dir: str, filename: str = "network.html") -> str:
    """
    Interactive HTML graph.
    """
    net = Network(height="800px", width="100%", bgcolor=THEME["bg"], font_color=THEME["text"])
    tooltip_map = load_tooltip_data(out_dir)

    # Physics makes it readable; users can drag nodes around.
    net.force_atlas_2based()

    net.set_options("""
    var options = {
      "interaction": {
        "hover": true,
        "tooltipDelay": 999999,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      },
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

    # Load tooltip-ready node fields (if analyze.py has been run)
    tooltips_by_id = load_node_tooltips(out_dir)

    for node_id, attrs in G.nodes(data=True):
        popularity = attrs.get("popularity", 0) or 0
        hop = attrs.get("hop", 1)

        size = popularity_to_node_size(popularity)

        # Simple coloring by hop for now (seed vs others)
        color = THEME["seed_node"] if hop == 0 else THEME["hop1_node"]

        # Tooltip payload (from analyze.py). If not found, empty dict.
        cg_payload = tooltips_by_id.get(str(node_id), {})

        cg = tooltip_map.get(str(node_id))

        net.add_node(
            node_id,
            label=attrs.get("name", node_id),
            title="",  # we'll render our own tooltip
            cg=cg,      # <-- IMPORTANT
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

    # Inject our custom tooltip UI + hover behavior
    inject_tooltip_code(out_path)

    # Post-process to add sidebar and search functionality
    inject_sidebar_and_search(out_path)

    return out_path

def inject_tooltip_code(html_path: str) -> None:
    """
    Injects a custom tooltip container + JS into the PyVis-generated HTML.
    This avoids relying on vis-network's default 'title' tooltip.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # If we've already injected once, replace the old block cleanly
    start_marker = "<!-- CG_TOOLTIP_START -->"
    end_marker = "<!-- CG_TOOLTIP_END -->"
    if start_marker in html and end_marker in html:
        pre = html.split(start_marker)[0]
        post = html.split(end_marker)[1]
        html = pre + post

    injected = r"""
    <!-- CG_TOOLTIP_START -->
    <style>
    #cg-tooltip {
        position: fixed;
        z-index: 9999;
        pointer-events: auto;
        display: none;
        width: 360px;
        max-width: calc(100vw - 24px);
        background: rgba(18,18,18,0.92);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.55);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 18px 18px 14px 18px;
        color: #fff;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }

    .cg-top { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .cg-name { font-size:40px; font-weight:800; line-height:1.0; letter-spacing:-0.02em; }
    .cg-sub  { margin-top:8px; color: rgba(255,255,255,0.62); font-size:18px; }

    .cg-spotify {
        width: 44px; height: 44px; border-radius: 999px;
        display:flex; align-items:center; justify-content:center;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        cursor: pointer;
        flex: 0 0 auto;
    }
    .cg-spotify:hover { background: rgba(255,255,255,0.10); }

    .cg-role { margin-top: 12px; color: rgba(255,255,255,0.82); font-size: 16px; line-height:1.35; }

    .cg-grid { margin-top: 14px; display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .cg-k { color: rgba(255,255,255,0.55); font-size: 14px; letter-spacing: 0.14em; }
    .cg-v { margin-top: 6px; font-size: 30px; font-weight: 700; }

    .cg-row { margin-top: 14px; display:flex; gap:12px; flex-wrap:wrap; }
    .cg-label { width: 100%; color: rgba(255,255,255,0.55); font-size: 14px; }

    .cg-pill {
        margin-top: 6px;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 18px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.06);
        width: fit-content;
    }

    .cg-line { margin-top: 14px; height: 1px; background: rgba(255,255,255,0.08); }

    /* Bridge colors */
    .cg-b-vh { background: rgba(120, 0, 0, 0.30); border-color: rgba(120, 0, 0, 0.55); }
    .cg-b-h  { background: rgba(190, 40, 40, 0.22); border-color: rgba(190, 40, 40, 0.48); }
    .cg-b-m  { background: rgba(224, 154, 60, 0.18); border-color: rgba(224, 154, 60, 0.42); }
    .cg-b-l  { background: rgba(76, 132, 224, 0.18); border-color: rgba(76, 132, 224, 0.42); }
    .cg-b-vl { background: rgba(120, 170, 255, 0.16); border-color: rgba(120, 170, 255, 0.38); }
    .cg-b-u  { background: rgba(255, 255, 255, 0.06); border-color: rgba(255, 255, 255, 0.14); }

    /* Influence colors (same palette; you can change later if you want different) */
    .cg-i-vh { background: rgba(120, 0, 0, 0.30); border-color: rgba(120, 0, 0, 0.55); }
    .cg-i-h  { background: rgba(190, 40, 40, 0.22); border-color: rgba(190, 40, 40, 0.48); }
    .cg-i-m  { background: rgba(224, 154, 60, 0.18); border-color: rgba(224, 154, 60, 0.42); }
    .cg-i-l  { background: rgba(76, 132, 224, 0.18); border-color: rgba(76, 132, 224, 0.42); }
    .cg-i-vl { background: rgba(120, 170, 255, 0.16); border-color: rgba(120, 170, 255, 0.38); }
    .cg-i-u  { background: rgba(255, 255, 255, 0.06); border-color: rgba(255, 255, 255, 0.14); }

    .cg-metrics {
        margin-top: 14px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    .cg-metric .cg-label { margin-bottom: 10px; }
    .cg-role {
        margin-top: 12px;
        color: rgba(255,255,255,0.82);
        font-size: 16px;
        line-height: 1.35;
    }
    </style>

    <div id="cg-tooltip" aria-hidden="true"></div>

    <script>
    (function() {
    function byId(id){ return document.getElementById(id); }

    function safeText(x){
        if (x === null || x === undefined) return "";
        return String(x);
    }

    function formatFollowers(n){
        var v = Number(n);
        if (!isFinite(v) || v <= 0) return "";
        if (v >= 1e9) return (v/1e9).toFixed(1).replace(/\.0$/,"") + "B";
        if (v >= 1e6) return (v/1e6).toFixed(1).replace(/\.0$/,"") + "M";
        if (v >= 1e3) return (v/1e3).toFixed(1).replace(/\.0$/,"") + "K";
        return String(Math.round(v));
    }

    function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

    function bucketClass(bucket){
        bucket = safeText(bucket || "Unknown");
        if (bucket === "Very High") return "vh";
        if (bucket === "High") return "h";
        if (bucket === "Medium") return "m";
        if (bucket === "Low") return "l";
        if (bucket === "Very Low") return "vl";
        return "u";
    }
    function bridgeCssClass(bucket){ return "cg-b-" + bucketClass(bucket); }
    function influenceCssClass(bucket){ return "cg-i-" + bucketClass(bucket); }

    function renderTooltip(cg){
        var el = byId("cg-tooltip");
        if (!el) return;

        var name = safeText(cg.name);
        var genres = safeText(cg.genres);
        var pop = Number(cg.popularity);
        if (!isFinite(pop)) pop = 0;
        pop = clamp(pop, 0, 100);

        var followersTxt = formatFollowers(cg.followers);
        var bridge = safeText(cg.bridge_category || "Unknown");
        var infl  = safeText(cg.influence_category || "Unknown");
        var interpretation = safeText(cg.interpretation);

        var spotifyUrl = safeText(cg.spotify_url);

        // Build without template literals to avoid "unterminated template" issues
        var html = "";
        html += '<div class="cg-top">';
        html +=   '<div>';
        html +=     '<div class="cg-name">' + name + '</div>';
        if (genres) html += '<div class="cg-sub">' + genres + '</div>';
        html +=   '</div>';
        if (spotifyUrl) {
        html += '<div class="cg-spotify" title="Open in Spotify" data-url="' + spotifyUrl + '">';
        html +=   '<svg width="20" height="20" viewBox="0 0 24 24" fill="white" aria-hidden="true"><path d="M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm4.586 14.424a.75.75 0 0 1-1.03.247c-2.82-1.722-6.37-2.112-10.55-1.16a.75.75 0 1 1-.333-1.462c4.57-1.04 8.49-.59 11.63 1.33.354.217.466.678.283 1.045zm1.47-3.27a.9.9 0 0 1-1.236.297c-3.23-1.988-8.15-2.565-11.96-1.405a.9.9 0 0 1-.525-1.722c4.35-1.32 9.76-.68 13.48 1.61.42.26.55.81.241 1.22zm.128-3.406C14.51 7.58 8.52 7.37 4.86 8.49a1.05 1.05 0 1 1-.615-2.01c4.2-1.28 11.02-1.03 15.34 1.54a1.05 1.05 0 0 1-1.077 1.77z"/></svg>';
        html += '</div>';
        }
        html += '</div>';
        html += '<div class="cg-grid">';
        html +=   '<div>';
        html +=     '<div class="cg-k">POPULARITY</div>';
        html +=     '<div class="cg-v">' + Math.round(pop) + ' / 100</div>';
        html +=   '</div>';
        html +=   '<div>';
        html +=     '<div class="cg-k">FOLLOWERS</div>';
        html +=     '<div class="cg-v">' + (followersTxt || "") + '</div>';
        html +=   '</div>';
        html += '</div>';

        html += '<div class="cg-line"></div>';

        html += '<div class="cg-metrics">';
        html +=   '<div class="cg-metric">';
        html +=     '<div class="cg-k">BRIDGE SCORE\n</div>';
        html +=     '<div class="cg-pill ' + bridgeCssClass(bridge) + '">' + bridge + '</div>'
        html +=   '</div>';
        html +=   '<div class="cg-metric">';
        html +=     '<div class="cg-k">INFLUENCE\n</div>';
        html +=     '<div class="cg-pill ' + influenceCssClass(infl) + '">' + infl + '</div>';
        html +=   '</div>';
        html += '</div>';

        if (interpretation) {
            html += '<div class="cg-line"></div>';
            html += '<div class="cg-role">' + interpretation + '</div>';
        }

        el.innerHTML = html;

        // Click handler for Spotify button
        var btn = el.querySelector(".cg-spotify");
        if (btn) {
        btn.addEventListener("click", function(ev){
            ev.stopPropagation();
            var url = btn.getAttribute("data-url");
            if (url) window.open(url, "_blank", "noopener,noreferrer");
        });
        }
    }

    function positionTooltip(evt){
        var el = byId("cg-tooltip");
        if (!el) return;

        // Use the actual pointer coordinates from vis event
        var x = evt && evt.pointer && evt.pointer.DOM ? evt.pointer.DOM.x : 24;
        var y = evt && evt.pointer && evt.pointer.DOM ? evt.pointer.DOM.y : 24;

        // Convert canvas coords to page coords by using the network container rect
        var container = byId("mynetwork");
        if (!container) return;
        var rect = container.getBoundingClientRect();

        // Start near cursor, then clamp so it never goes off-screen
        var left = rect.left + x + 16;
        var top  = rect.top + y + 16;

        // Clamp using element's current size
        el.style.display = "block";
        var w = el.offsetWidth;
        var h = el.offsetHeight;

        var maxLeft = window.innerWidth - w - 12;
        var maxTop  = window.innerHeight - h - 12;

        left = clamp(left, 12, maxLeft);
        top  = clamp(top, 12, maxTop);

        el.style.left = left + "px";
        el.style.top  = top + "px";
    }

    function hideTooltip(){
        var el = byId("cg-tooltip");
        if (el) el.style.display = "none";
    }

    function showForNode(nodeId, evt){
        if (!nodes || !nodes.get) return;
        var n = nodes.get(nodeId);
        if (!n || !n.cg) { hideTooltip(); return; }
        renderTooltip(n.cg);
        positionTooltip(evt);
    }

    // Wait until pyvis creates "network" and "nodes"
    function boot(){
        if (typeof network === "undefined" || typeof nodes === "undefined") {
        setTimeout(boot, 50);
        return;
        }

        var pinnedNode = null;

        // Hover shows tooltip (unless pinned)
        network.on("hoverNode", function(params){
        if (pinnedNode) return;
        showForNode(params.node, params.event);
        });

        // Moving mouse while hovering keeps tooltip positioned
        network.on("mousemove", function(params){
        if (!pinnedNode) return;
        // If pinned, follow mouse? (you can change this; I leave pinned static)
        });

        network.on("blurNode", function(){
        if (pinnedNode) return;
        hideTooltip();
        });

        // Click pins/unpins tooltip, but should NOT change drag mechanics
        network.on("click", function(params){
        if (params.nodes && params.nodes.length) {
            pinnedNode = params.nodes[0];
            showForNode(pinnedNode, params.event);
        } else {
            pinnedNode = null;
            hideTooltip();
        }
        });

        // Keep pinned tooltip visible while zoom/drag happens
        network.on("dragEnd", function(params){
        if (!pinnedNode) return;
        // Reposition near last known pointer if available
        if (params && params.event) positionTooltip(params.event);
        });
    }

    window.addEventListener("load", boot);
    })();
    </script>
    <!-- CG_TOOLTIP_END -->
    """

    # Inject right before </body>
    if "</body>" in html:
        html = html.replace("</body>", injected + "\n</body>")
    else:
        html = html + injected

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


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