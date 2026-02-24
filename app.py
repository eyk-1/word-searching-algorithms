import streamlit as st
import numpy as np
import heapq
import time
from collections import deque

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Word Ladder Search",
    page_icon="🔤",
    layout="centered"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0d0d0d;
        color: #f0f0f0;
    }
    .stApp {
        background-color: #0d0d0d;
    }
    h1 {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        letter-spacing: -1px;
        color: #f0f0f0;
    }
    .subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-top: -10px;
        margin-bottom: 30px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem !important;
    }
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
    }
    .stSlider > div {
        color: #f0f0f0;
    }
    .stButton > button {
        background-color: #e8ff47 !important;
        color: #0d0d0d !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85 !important;
    }
    .result-box {
        background-color: #141414;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 24px 28px;
        margin-top: 24px;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 18px;
    }
    .stat-card {
        background-color: #1c1c1c;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #666;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e8ff47;
        margin-top: 4px;
        font-family: 'Syne', sans-serif;
    }
    .path-display {
        background-color: #111;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #ccc;
        word-break: break-all;
        line-height: 1.8;
    }
    .path-word {
        color: #e8ff47;
        font-weight: 700;
    }
    .path-arrow {
        color: #444;
        margin: 0 4px;
    }
    .failed-box {
        background-color: #1a0d0d;
        border: 1px solid #5c1a1a;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 18px;
        color: #ff6b6b;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
    }
    .section-label {
        font-size: 0.72rem;
        color: #555;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .divider {
        border: none;
        border-top: 1px solid #222;
        margin: 28px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── CORE FUNCTIONS ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings(filepath):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

@st.cache_resource(show_spinner=False)
def precompute_neighbors(_embeddings, k):
    words = list(_embeddings.keys())
    vectors = np.array(list(_embeddings.values()), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    neighbors = {}
    for i, word in enumerate(words):
        sims = normalized @ normalized[i]
        sims[i] = -1
        top_k = np.argsort(sims)[-k:][::-1]
        neighbors[word] = [(float(sims[j]), words[j]) for j in top_k]
    return neighbors

def reconstruct_path(came_from, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def bfs(start, goal, neighbors_map):
    frontier = deque([start])
    came_from = {start: None}
    nodes_expanded = 0
    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1
        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded
        for sim, neighbor in neighbors_map[current]:
            if neighbor not in came_from:
                came_from[neighbor] = current
                frontier.append(neighbor)
    return None, nodes_expanded

def dfs(start, goal, neighbors_map, depth_limit=20):
    frontier = [(start, 0)]
    came_from = {start: None}
    explored = set()
    nodes_expanded = 0
    while frontier:
        current, depth = frontier.pop()
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded
        if depth >= depth_limit:
            continue
        for sim, neighbor in neighbors_map[current]:
            if neighbor not in explored:
                came_from[neighbor] = current
                frontier.append((neighbor, depth + 1))
    return None, nodes_expanded

def ucs(start, goal, neighbors_map, max_nodes=20000):
    frontier = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    explored = set()
    nodes_expanded = 0
    while frontier:
        current_cost, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            return None, nodes_expanded
        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded
        for sim, neighbor in neighbors_map[current]:
            edge_cost = 1 - sim
            new_cost = current_cost + edge_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))
    return None, nodes_expanded

def greedy_best_first(start, goal, embeddings, neighbors_map):
    goal_vec = embeddings[goal] / np.linalg.norm(embeddings[goal])
    def h(word):
        v = embeddings[word] / np.linalg.norm(embeddings[word])
        return 1 - float(np.dot(v, goal_vec))
    frontier = [(h(start), start)]
    came_from = {start: None}
    explored = set()
    nodes_expanded = 0
    while frontier:
        _, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded
        for sim, neighbor in neighbors_map[current]:
            if neighbor not in explored:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(frontier, (h(neighbor), neighbor))
    return None, nodes_expanded

def astar(start, goal, embeddings, neighbors_map):
    goal_vec = embeddings[goal] / np.linalg.norm(embeddings[goal])
    def h(word):
        v = embeddings[word] / np.linalg.norm(embeddings[word])
        return 1 - float(np.dot(v, goal_vec))
    frontier = [(h(start), start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    explored = set()
    nodes_expanded = 0
    while frontier:
        f, current = heapq.heappop(frontier)
        if current in explored:
            continue
        explored.add(current)
        nodes_expanded += 1
        if current == goal:
            return reconstruct_path(came_from, goal), nodes_expanded
        for sim, neighbor in neighbors_map[current]:
            edge_cost = 1 - sim
            new_g = cost_so_far[current] + edge_cost
            if neighbor not in cost_so_far or new_g < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_g + h(neighbor), neighbor))
    return None, nodes_expanded


# ─── UI ──────────────────────────────────────────────────────────────────────

st.markdown("<h1>Word Ladder Search</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">// semantic embedding space explorer</p>', unsafe_allow_html=True)

# Load embeddings
with st.spinner("Loading embeddings..."):
    try:
        embeddings = load_embeddings("glove.100d.20000.txt")
        st.success(f"Loaded {len(embeddings):,} words from GloVe vocabulary.")
    except FileNotFoundError:
        st.error("glove.100d.20000.txt not found. Make sure it's in the same folder as app.py.")
        st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Controls
col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="section-label">Start Word</p>', unsafe_allow_html=True)
    start_word = st.text_input("start", placeholder="e.g. doped", label_visibility="collapsed").strip().lower()

with col2:
    st.markdown('<p class="section-label">Goal Word</p>', unsafe_allow_html=True)
    goal_word = st.text_input("goal", placeholder="e.g. leather", label_visibility="collapsed").strip().lower()

st.markdown('<p class="section-label" style="margin-top:16px;">Algorithm</p>', unsafe_allow_html=True)
algorithm = st.selectbox(
    "algo",
    ["BFS", "DFS", "UCS", "Greedy Best-First Search", "A*"],
    label_visibility="collapsed"
)

st.markdown('<p class="section-label" style="margin-top:16px;">Number of Neighbors (k)</p>', unsafe_allow_html=True)
k = st.slider("k", min_value=5, max_value=50, value=10, label_visibility="collapsed")

st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
run = st.button("Run Search")

# ─── RUN ─────────────────────────────────────────────────────────────────────

if run:
    # Validate inputs
    if not start_word or not goal_word:
        st.error("Please enter both a start word and a goal word.")
        st.stop()

    if start_word not in embeddings:
        st.error(f'"{start_word}" is not in the GloVe vocabulary. Try a different word.')
        st.stop()

    if goal_word not in embeddings:
        st.error(f'"{goal_word}" is not in the GloVe vocabulary. Try a different word.')
        st.stop()

    if start_word == goal_word:
        st.warning("Start and goal words are the same.")
        st.stop()

    # Precompute neighbors
    with st.spinner(f"Precomputing top-{k} neighbors for all words..."):
        neighbors_map = precompute_neighbors(embeddings, k)

    # Run selected algorithm
    with st.spinner(f"Running {algorithm}..."):
        t0 = time.time()
        if algorithm == "BFS":
            path, nodes = bfs(start_word, goal_word, neighbors_map)
        elif algorithm == "DFS":
            path, nodes = dfs(start_word, goal_word, neighbors_map)
        elif algorithm == "UCS":
            path, nodes = ucs(start_word, goal_word, neighbors_map)
        elif algorithm == "Greedy Best-First Search":
            path, nodes = greedy_best_first(start_word, goal_word, embeddings, neighbors_map)
        elif algorithm == "A*":
            path, nodes = astar(start_word, goal_word, embeddings, neighbors_map)
        elapsed = time.time() - t0

    # ─── Results ─────────────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<p class="section-label">Results — {algorithm}</p>', unsafe_allow_html=True)

    if path:
        steps = len(path) - 1

        # Stat cards
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Steps</div>
                <div class="stat-value">{steps}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Nodes Expanded</div>
                <div class="stat-value">{nodes:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time</div>
                <div class="stat-value">{elapsed:.4f}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Path display
        path_html = ' <span class="path-arrow">→</span> '.join(
            f'<span class="path-word">{w}</span>' for w in path
        )
        st.markdown(f'<div class="path-display">{path_html}</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="failed-box">
            ✗ No path found from <strong>{start_word}</strong> to <strong>{goal_word}</strong>
            using {algorithm} with k={k}.<br><br>
            Try increasing k or choosing words that are more semantically connected.
        </div>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Steps</div>
                <div class="stat-value">—</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Nodes Expanded</div>
                <div class="stat-value">{nodes:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time</div>
                <div class="stat-value">{elapsed:.4f}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)