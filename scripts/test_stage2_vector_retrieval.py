#!/usr/bin/env python3
"""
Offline evaluation: can Stage2 (vector retrieval) find usable synthesis candidates for each seed?

Constraints (per user request):
- Do NOT run any LLM-based extraction/analysis.
- Only use existing cached artifacts:
  - data/assets/extractions/*.json
  - data/assets/graphs/*.pkl
  - data/assets/embeddings/<repo>/vector_pool.npy + commits/<commit>.json

Outputs a CSV with per-seed diagnostics and a short console summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

# Add project root to sys.path so `src.*` imports work when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.structures import ExtractionResult  # noqa: E402
from src.modules.synthesis.matcher import SubgraphMatcher, is_test_file  # noqa: E402


@dataclass(frozen=True)
class SeedEvalRow:
    commit_short: str
    instance_id: str
    base_commit: str
    seed_node_id: str
    seed_node_resolved: str
    seed_resolution_mode: str
    seed_depth: int
    embed_nodes: int
    vector_used: bool
    seed_pool_idx: int
    scanned: int
    skipped_original: int
    skipped_test_file: int
    skipped_depth_mismatch: int
    top_indices: int
    compat_nonempty: int
    candidates: int
    best_anchor_node_id: str
    best_vector_score: float
    best_topology_score: float
    best_final_score: float
    failure_reason: str
    elapsed_s: float


def _default_outputs_dir() -> Path:
    # scripts/ -> project_root -> parent/tracegen-outputs
    project_root = Path(__file__).resolve().parent.parent
    return project_root.parent / "tracegen-outputs" / "full_run"


def _load_pickle_silent(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _parse_commit_short_from_filename(path: Path) -> str:
    # e.g. django_django__004b4620.json
    stem = path.stem
    if "__" not in stem:
        return stem
    return stem.split("__", 1)[1]


def _resolve_seed_node_id(seed_node_id: str, node_to_hash: dict[str, str]) -> tuple[str, str]:
    """
    Mirror SubgraphMatcher._find_node_idx behavior for V2 commit mapping:
    - Prefer exact match.
    - If seed_node_id is a file path, map to the first child node with `file_path:` prefix.
    """
    if seed_node_id in node_to_hash:
        return seed_node_id, "exact"

    prefix = f"{seed_node_id}:"
    for mapped_id in node_to_hash.keys():
        if mapped_id.startswith(prefix):
            return mapped_id, "file_prefix"

    return "", "missing"


def _seed_chain_depth(seed_chains: list[Any]) -> int:
    if not seed_chains:
        return 0
    first = seed_chains[0]
    nodes = first.get("nodes", []) if isinstance(first, dict) else getattr(first, "nodes", [])
    return int(len(nodes) or 0)


def _seed_root_cause_node_id(seed_chains: list[Any]) -> str:
    if not seed_chains:
        return ""
    first = seed_chains[0]
    nodes = first.get("nodes", []) if isinstance(first, dict) else getattr(first, "nodes", [])
    if not nodes:
        return ""
    rc = nodes[-1]
    if isinstance(rc, dict):
        return str(rc.get("node_id") or "").strip()
    return str(getattr(rc, "node_id", "") or "").strip()


def _seed_subgraph(seed_chains: list[Any]) -> dict[str, Any]:
    if not seed_chains:
        return {}
    first = seed_chains[0]
    if isinstance(first, dict):
        return dict(first.get("extraction_metadata", {}).get("subgraph", {}) or {})
    return dict(getattr(first, "extraction_metadata", {}).get("subgraph", {}) or {})


def _collect_original_node_ids(seed_chains: list[Any]) -> set[str]:
    original: set[str] = set()
    for chain in seed_chains or []:
        nodes = chain.get("nodes", []) if isinstance(chain, dict) else getattr(chain, "nodes", [])
        for n in nodes or []:
            if isinstance(n, dict):
                nid = n.get("node_id")
            else:
                nid = getattr(n, "node_id", None)
            if nid:
                original.add(str(nid))
    return original


def _build_commit_node_index(
    *,
    node_to_hash: dict[str, str],
    hash_to_idx: dict[str, int],
) -> tuple[list[str], np.ndarray]:
    node_ids: list[str] = []
    pool_idxs: list[int] = []
    for node_id, h in node_to_hash.items():
        idx = hash_to_idx.get(h)
        if idx is None:
            continue
        node_ids.append(node_id)
        pool_idxs.append(int(idx))
    return node_ids, np.asarray(pool_idxs, dtype=np.int64)


def _cosine_similarities_chunked(
    *,
    vector_pool: np.ndarray,
    norms_pool: np.ndarray,
    pool_idxs: np.ndarray,
    seed_pool_idx: int,
    chunk_size: int,
) -> np.ndarray:
    seed_norm = float(norms_pool[seed_pool_idx])
    if seed_norm <= 0.0:
        return np.empty((0,), dtype=np.float32)
    seed_unit = np.asarray(vector_pool[seed_pool_idx], dtype=np.float64) / (seed_norm + 1e-9)

    sims = np.empty((len(pool_idxs),), dtype=np.float32)
    norms = norms_pool[pool_idxs]

    for start in range(0, len(pool_idxs), chunk_size):
        end = min(start + chunk_size, len(pool_idxs))
        idxs = pool_idxs[start:end]
        vecs = np.asarray(vector_pool[idxs], dtype=np.float64)
        dots = vecs @ seed_unit
        denom = np.asarray(norms[start:end], dtype=np.float64) + 1e-9
        sims[start:end] = (dots / denom).astype(np.float32, copy=False)

    return sims


def evaluate_one_seed(
    *,
    extraction_path: Path,
    graphs_dir: Path,
    embeddings_repo_dir: Path,
    vector_pool: np.ndarray,
    norms_pool: np.ndarray,
    hash_to_idx: dict[str, int],
    matcher: SubgraphMatcher,
    top_k_vector: int,
    top_k_final: int,
    chunk_size: int,
) -> SeedEvalRow:
    t0 = time.time()
    commit_short = _parse_commit_short_from_filename(extraction_path)

    # Defaults for failure rows
    instance_id = ""
    base_commit = ""
    seed_node_id = ""
    seed_node_resolved = ""
    seed_resolution_mode = "missing"
    seed_depth = 0
    embed_nodes = 0
    vector_used = False
    seed_pool_idx = -1
    scanned = 0
    skipped_original = 0
    skipped_test_file = 0
    skipped_depth_mismatch = 0
    top_indices_count = 0
    compat_nonempty = 0
    candidates_count = 0
    best_anchor_node_id = ""
    best_vector_score = 0.0
    best_topology_score = 0.0
    best_final_score = 0.0
    failure_reason = ""

    try:
        data = json.loads(extraction_path.read_text(encoding="utf-8"))
        extraction = ExtractionResult(**data)
        instance_id = extraction.instance_id
        base_commit = str(extraction.seed_metadata.get("base_commit", "") or "")
        seed_chains = extraction.mined_data.get("extracted_chains", []) or []

        if not seed_chains:
            failure_reason = "no_extracted_chains"
            raise RuntimeError(failure_reason)

        seed_depth = _seed_chain_depth(seed_chains)
        seed_node_id = _seed_root_cause_node_id(seed_chains)
        if not seed_node_id:
            failure_reason = "missing_seed_root_cause"
            raise RuntimeError(failure_reason)

        original_node_ids = _collect_original_node_ids(seed_chains)
        seed_subgraph = _seed_subgraph(seed_chains)

        graph_path = graphs_dir / f"django_django__{commit_short}_v1.pkl"
        if not graph_path.exists():
            failure_reason = "missing_graph"
            raise RuntimeError(failure_reason)
        graph = _load_pickle_silent(graph_path)

        commit_map_path = embeddings_repo_dir / "commits" / f"{commit_short}.json"
        if not commit_map_path.exists():
            failure_reason = "missing_commit_embedding_map"
            raise RuntimeError(failure_reason)
        node_to_hash = json.loads(commit_map_path.read_text(encoding="utf-8"))
        if not isinstance(node_to_hash, dict) or not node_to_hash:
            failure_reason = "empty_commit_embedding_map"
            raise RuntimeError(failure_reason)

        seed_node_resolved, seed_resolution_mode = _resolve_seed_node_id(seed_node_id, node_to_hash)
        if not seed_node_resolved:
            failure_reason = "seed_anchor_not_in_commit_map"
            raise RuntimeError(failure_reason)

        # Build commit-local node list + pool indices (mirrors matcher V2 behavior without materializing embeddings)
        node_ids, pool_idxs = _build_commit_node_index(node_to_hash=node_to_hash, hash_to_idx=hash_to_idx)
        embed_nodes = len(node_ids)
        if embed_nodes == 0:
            failure_reason = "no_nodes_mapped_to_vector_pool"
            raise RuntimeError(failure_reason)

        seed_hash = node_to_hash.get(seed_node_resolved)
        if not seed_hash:
            failure_reason = "seed_hash_missing"
            raise RuntimeError(failure_reason)
        seed_pool_idx = int(hash_to_idx.get(seed_hash, -1))
        if seed_pool_idx < 0:
            failure_reason = "seed_hash_not_in_hash_to_idx"
            raise RuntimeError(failure_reason)
        vector_used = True

        sims = _cosine_similarities_chunked(
            vector_pool=vector_pool,
            norms_pool=norms_pool,
            pool_idxs=pool_idxs,
            seed_pool_idx=seed_pool_idx,
            chunk_size=chunk_size,
        )
        if sims.size == 0:
            failure_reason = "seed_vector_norm_zero"
            raise RuntimeError(failure_reason)

        sorted_local_indices = np.argsort(sims)[::-1]

        # Vector-stage filtering (exclude original nodes, tests, and depth-mismatched nodes)
        top_local_indices: list[int] = []
        for local_idx in sorted_local_indices:
            scanned += 1
            candidate_id = node_ids[int(local_idx)]
            if candidate_id in original_node_ids:
                skipped_original += 1
                continue
            if is_test_file(candidate_id):
                skipped_test_file += 1
                continue
            candidate_depth = matcher._calculate_chain_depth(candidate_id, graph)
            if abs(int(candidate_depth) - int(seed_depth)) > 1:
                skipped_depth_mismatch += 1
                continue
            top_local_indices.append(int(local_idx))
            if len(top_local_indices) >= int(top_k_vector):
                break

        top_indices_count = len(top_local_indices)
        if top_indices_count == 0:
            failure_reason = "no_candidates_after_vector_filters"
            raise RuntimeError(failure_reason)

        # Full Stage2 candidate building (topology + intent compatibility)
        candidates: list[dict[str, Any]] = []
        for local_idx in top_local_indices:
            anchor_id = node_ids[local_idx]
            vec_score = float(sims[local_idx])
            cand_sub = matcher._get_node_subgraph(anchor_id, graph)
            topo_score = float(matcher._calculate_subgraph_similarity(seed_subgraph, cand_sub))
            compatible = matcher._filter_compatible_intents(anchor_id, extraction.fix_intents, graph)
            if compatible:
                compat_nonempty += 1
                candidates.append(
                    {
                        "anchor_node_id": anchor_id,
                        "vector_score": vec_score,
                        "topology_score": topo_score,
                        "final_score": 0.4 * vec_score + 0.6 * topo_score,
                    }
                )

        candidates.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        candidates = candidates[: int(top_k_final)]
        candidates_count = len(candidates)
        if candidates_count == 0:
            failure_reason = "no_candidates_after_intent_compatibility"
            raise RuntimeError(failure_reason)

        best = candidates[0]
        best_anchor_node_id = str(best.get("anchor_node_id", "") or "")
        best_vector_score = float(best.get("vector_score", 0.0) or 0.0)
        best_topology_score = float(best.get("topology_score", 0.0) or 0.0)
        best_final_score = float(best.get("final_score", 0.0) or 0.0)

        failure_reason = ""

    except Exception as e:
        if not failure_reason:
            failure_reason = f"error:{type(e).__name__}"

    elapsed = time.time() - t0
    return SeedEvalRow(
        commit_short=commit_short,
        instance_id=instance_id,
        base_commit=base_commit,
        seed_node_id=seed_node_id,
        seed_node_resolved=seed_node_resolved,
        seed_resolution_mode=seed_resolution_mode,
        seed_depth=int(seed_depth or 0),
        embed_nodes=int(embed_nodes or 0),
        vector_used=bool(vector_used),
        seed_pool_idx=int(seed_pool_idx),
        scanned=int(scanned),
        skipped_original=int(skipped_original),
        skipped_test_file=int(skipped_test_file),
        skipped_depth_mismatch=int(skipped_depth_mismatch),
        top_indices=int(top_indices_count),
        compat_nonempty=int(compat_nonempty),
        candidates=int(candidates_count),
        best_anchor_node_id=best_anchor_node_id,
        best_vector_score=float(best_vector_score),
        best_topology_score=float(best_topology_score),
        best_final_score=float(best_final_score),
        failure_reason=failure_reason,
        elapsed_s=float(elapsed),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "assets" / "extractions",
        help="Directory with cached extraction results (*.json).",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "assets" / "graphs",
        help="Directory with cached graphs (*.pkl).",
    )
    parser.add_argument(
        "--embeddings-repo-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "assets" / "embeddings" / "django_django",
        help="Repo-level embeddings dir containing vector_pool.npy/hash_to_idx.json/commits/.",
    )
    parser.add_argument("--top-k-vector", type=int, default=50)
    parser.add_argument("--top-k-final", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_default_outputs_dir() / "stage2_vector_retrieval_report.csv",
        help="Where to write the CSV report.",
    )
    args = parser.parse_args()

    # Keep logs quiet (this script can otherwise produce tens of thousands of lines).
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    extractions_dir: Path = args.extractions_dir
    graphs_dir: Path = args.graphs_dir
    embeddings_repo_dir: Path = args.embeddings_repo_dir
    out_csv: Path = args.out_csv

    vector_pool_path = embeddings_repo_dir / "vector_pool.npy"
    hash_to_idx_path = embeddings_repo_dir / "hash_to_idx.json"
    if not vector_pool_path.exists() or not hash_to_idx_path.exists():
        print(f"Missing embeddings files: {vector_pool_path} / {hash_to_idx_path}", file=sys.stderr)
        return 2

    vector_pool = np.load(vector_pool_path, mmap_mode="r")
    hash_to_idx = json.loads(hash_to_idx_path.read_text(encoding="utf-8"))
    if not isinstance(hash_to_idx, dict) or not hash_to_idx:
        print(f"Invalid hash_to_idx: {hash_to_idx_path}", file=sys.stderr)
        return 2

    # Precompute norms once for the whole pool (small array; speeds up per-commit similarities).
    t_norm = time.time()
    norms_pool = np.linalg.norm(vector_pool, axis=1)
    norms_s = time.time() - t_norm

    # A SubgraphMatcher instance is used only for helper methods (depth/subgraph/intents).
    matcher = SubgraphMatcher(
        {
            "embedding_dir": str(embeddings_repo_dir.parent),
            "top_k_vector": int(args.top_k_vector),
            "top_k_final": int(args.top_k_final),
            "auto_generate": False,
        }
    )

    extraction_files = sorted(extractions_dir.glob("*.json"))
    if args.limit and int(args.limit) > 0:
        extraction_files = extraction_files[: int(args.limit)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[SeedEvalRow] = []
    t_all = time.time()
    for p in extraction_files:
        rows.append(
            evaluate_one_seed(
                extraction_path=p,
                graphs_dir=graphs_dir,
                embeddings_repo_dir=embeddings_repo_dir,
                vector_pool=vector_pool,
                norms_pool=norms_pool,
                hash_to_idx=hash_to_idx,
                matcher=matcher,
                top_k_vector=int(args.top_k_vector),
                top_k_final=int(args.top_k_final),
                chunk_size=int(args.chunk_size),
            )
        )
    elapsed_all = time.time() - t_all

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "commit_short",
                "instance_id",
                "base_commit",
                "seed_node_id",
                "seed_node_resolved",
                "seed_resolution_mode",
                "seed_depth",
                "embed_nodes",
                "vector_used",
                "seed_pool_idx",
                "scanned",
                "skipped_original",
                "skipped_test_file",
                "skipped_depth_mismatch",
                "top_indices",
                "compat_nonempty",
                "candidates",
                "best_anchor_node_id",
                "best_vector_score",
                "best_topology_score",
                "best_final_score",
                "failure_reason",
                "elapsed_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.commit_short,
                    r.instance_id,
                    r.base_commit,
                    r.seed_node_id,
                    r.seed_node_resolved,
                    r.seed_resolution_mode,
                    r.seed_depth,
                    r.embed_nodes,
                    int(bool(r.vector_used)),
                    r.seed_pool_idx,
                    r.scanned,
                    r.skipped_original,
                    r.skipped_test_file,
                    r.skipped_depth_mismatch,
                    r.top_indices,
                    r.compat_nonempty,
                    r.candidates,
                    r.best_anchor_node_id,
                    f"{r.best_vector_score:.6f}",
                    f"{r.best_topology_score:.6f}",
                    f"{r.best_final_score:.6f}",
                    r.failure_reason,
                    f"{r.elapsed_s:.4f}",
                ]
            )

    # Console summary
    total = len(rows)
    ok = sum(1 for r in rows if r.failure_reason == "" and r.candidates > 0 and r.vector_used)
    vector_missing = sum(1 for r in rows if not r.vector_used)
    no_candidates = sum(1 for r in rows if r.vector_used and r.candidates == 0)
    failures = total - ok

    reason_counts: dict[str, int] = {}
    for r in rows:
        reason = r.failure_reason or ""
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Stage2 vector retrieval eval: {ok}/{total} seeds have >=1 candidate (vector-used).")
    print(f"- vector_used=0: {vector_missing}")
    print(f"- vector_used=1 but candidates=0: {no_candidates}")
    print(f"- total failures: {failures}")
    print(f"- pool norms time: {norms_s:.2f}s | total eval time: {elapsed_all:.2f}s")
    print(f"- CSV: {out_csv}")
    if top_reasons:
        print("Top failure reasons:")
        for reason, cnt in top_reasons:
            print(f"  - {reason}: {cnt}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

