import numpy as np
import heapq
import time
from collections import deque

def load_embeddings(filepath):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            embeddings[word] = vector
    return embeddings

def precompute_neighbors(embeddings, k=10):
    print("Precomputing neighbors")
    neighbors = {}
    words = list(embeddings.keys())
    vectors = np.array(list(embeddings.values()))

    # Normalize all vectors once
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms

    for i, word in enumerate(words):
        sims = normalized @ normalized[i]
        sims[i] = -1  # exclude self
        top_k = np.argsort(sims)[-k:][::-1]
        neighbors[word] = [(sims[j], words[j]) for j in top_k]
    return neighbors

def reconstruct_path(came_from, start, goal):
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
            return reconstruct_path(came_from, start, goal), nodes_expanded

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
            return reconstruct_path(came_from, start, goal), nodes_expanded

        if depth >= depth_limit:
            continue

        for sim, neighbor in neighbors_map[current]:
            if neighbor not in explored:
                came_from[neighbor] = current
                frontier.append((neighbor, depth + 1))

    return None, nodes_expanded

def ucs(start, goal, neighbors_map, max_nodes=20000):
    frontier = [(0, start)]
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
            return reconstruct_path(came_from, start, goal), nodes_expanded

        for sim, neighbor in neighbors_map[current]:
            edge_cost = 1 - sim
            new_cost = current_cost + edge_cost

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(frontier, (new_cost, neighbor))

    return None, nodes_expanded

def greedy_best_first(start, goal, embeddings, neighbors_map):
    goal_vec = np.array(embeddings[goal])
    goal_vec = goal_vec / np.linalg.norm(goal_vec)

    def heuristic(word):
        v = np.array(embeddings[word])
        v = v / np.linalg.norm(v)
        return 1 - np.dot(v, goal_vec)

    frontier = [(heuristic(start), start)]
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
            return reconstruct_path(came_from, start, goal), nodes_expanded

        for sim, neighbor in neighbors_map[current]:
            if neighbor not in explored:
                if neighbor not in came_from:
                    came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor), neighbor))

    return None, nodes_expanded

def astar(start, goal, embeddings, neighbors_map):
    goal_vec = np.array(embeddings[goal])
    goal_vec = goal_vec / np.linalg.norm(goal_vec)

    def heuristic(word):
        v = np.array(embeddings[word])
        v = v / np.linalg.norm(v)
        return 1 - np.dot(v, goal_vec)

    frontier = [(heuristic(start), start)]
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
            return reconstruct_path(came_from, start, goal), nodes_expanded

        for sim, neighbor in neighbors_map[current]:
            edge_cost = 1 - sim
            new_g = cost_so_far[current] + edge_cost

            if neighbor not in cost_so_far or new_g < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_g
                came_from[neighbor] = current
                f_score = new_g + heuristic(neighbor)
                heapq.heappush(frontier, (f_score, neighbor))

    return None, nodes_expanded

def run_search(name, func, *args):
    print(f"{name} output:")
    start_time = time.time()
    path, nodes = func(*args)
    elapsed = time.time() - start_time
    if path:
        print(f"  Path: {' -> '.join(path)}")
        print(f"  Steps: {len(path) - 1}")
    else:
        print("  No path found")
    print(f"  Nodes expanded: {nodes}")
    print(f"  Time: {elapsed:.4f}s\n")

# ─── MAIN ────────────────────────────────────────────────────────────────────

embeddings = load_embeddings("glove.100d.20000.txt")
print(f"Loaded {len(embeddings)} words\n")

K = 10
neighbors_map = precompute_neighbors(embeddings, k=K)

# ── Word pairs to test ────────────────────────────────────────────────────────
pairs = [
    ("hate",   "harmony"),
    ("badger",  "shark"),
    ("rose",   "leather"),
    ("pleasure", "beauty"),
    ("heuristic", "storyboards"),  # semantically distant / likely failure case
]

for start, goal in pairs:
    print("=" * 60)
    print(f"  {start.upper()} --> {goal.upper()}")
    print("=" * 60)

    if start not in embeddings:
        print(f"'{start}' not in vocabulary.\n")
        continue
    if goal not in embeddings:
        print(f"'{goal}' not in vocabulary.\n")
        continue

    run_search("BFS",              bfs,               start, goal, neighbors_map)
    run_search("DFS",              dfs,               start, goal, neighbors_map)
    run_search("UCS",              ucs,               start, goal, neighbors_map)
    run_search("Greedy Best-First",greedy_best_first, start, goal, embeddings, neighbors_map)
    run_search("A*",               astar,             start, goal, embeddings, neighbors_map)