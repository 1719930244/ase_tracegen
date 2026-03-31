"""
Graph utilities for Stage 0 fault localization.

Provides BFS path finding on the code graph to build
localization chain skeletons without LLM calls.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import networkx as nx
from loguru import logger


def find_node_by_file_entity(
    graph: nx.DiGraph, file_path: str, entity_name: Optional[str] = None
) -> Optional[str]:
    """Find a graph node matching file_path:entity_name pattern."""
    if entity_name:
        candidate = f"{file_path}:{entity_name}"
        if candidate in graph:
            return candidate
        # Fuzzy: try suffix match
        for nid in graph.nodes():
            if ":" in nid:
                g_file, g_func = nid.split(":", 1)
                if g_file.endswith(file_path) and g_func.endswith(entity_name):
                    return nid
    # File-level match
    if file_path in graph:
        return file_path
    for nid in graph.nodes():
        if nid.endswith(file_path) and ":" not in nid:
            return nid
    return None


def bfs_path(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_depth: int = 6,
    edge_types: Optional[set[str]] = None,
) -> list[str]:
    """
    BFS shortest path from source to target on the code graph.

    Args:
        graph: Code dependency graph.
        source: Start node ID.
        target: End node ID.
        max_depth: Maximum search depth.
        edge_types: If set, only traverse edges of these types
                    (e.g. {"invokes", "contains"}).

    Returns:
        List of node IDs forming the path, or empty list if not found.
    """
    if source not in graph or target not in graph:
        return []
    if source == target:
        return [source]

    visited = {source}
    queue: deque[tuple[str, list[str]]] = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()
        if len(path) > max_depth:
            continue

        for neighbor in graph.successors(current):
            if edge_types:
                edges = graph.get_edge_data(current, neighbor)
                if edges is None:
                    continue
                # nx.DiGraph stores single edge; nx.MultiDiGraph stores dict of edges
                if isinstance(edges, dict) and "type" in edges:
                    if edges["type"] not in edge_types:
                        continue
                elif isinstance(edges, dict):
                    # MultiDiGraph: edges is {0: {type: ...}, 1: {type: ...}}
                    if not any(
                        e.get("type") in edge_types for e in edges.values()
                    ):
                        continue

            if neighbor == target:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []


def find_upstream_callers(
    graph: nx.DiGraph, target: str, max_depth: int = 4
) -> list[str]:
    """
    Find upstream callers of target node via reverse BFS.

    Returns list of node IDs ordered from furthest caller to target.
    """
    if target not in graph:
        return []

    visited = {target}
    queue: deque[tuple[str, int]] = deque([(target, 0)])
    callers: list[tuple[str, int]] = []

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for pred in graph.predecessors(current):
            if pred not in visited:
                visited.add(pred)
                callers.append((pred, depth + 1))
                queue.append((pred, depth + 1))

    # Sort by depth (furthest first) and return node IDs
    callers.sort(key=lambda x: -x[1])
    return [c[0] for c in callers]


def build_chain_skeleton(
    graph: nx.DiGraph,
    test_nodes: list[str],
    root_cause_nodes: list[str],
    max_depth: int = 6,
) -> list[str]:
    """
    Build a localization chain skeleton from test entry to root cause.

    Tries BFS from each test node to each root cause node,
    returns the shortest path found.
    """
    best_path: list[str] = []

    for test_node in test_nodes:
        for rc_node in root_cause_nodes:
            path = bfs_path(graph, test_node, rc_node, max_depth=max_depth)
            if path and (not best_path or len(path) < len(best_path)):
                best_path = path

    if best_path:
        logger.debug(
            f"Graph BFS chain: {' → '.join(best_path)} ({len(best_path)} nodes)"
        )

    return best_path
