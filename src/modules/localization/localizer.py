"""
Stage 0 Fault Localizer for TraceGen.

Generates ``raw_output_loc`` (and populates ``found_files / found_modules /
found_entities``) for instances that lack LocAgent data, enabling multi-repo
support beyond Django.

Three-layer strategy (each layer enriches the next):
  Layer 1 – Static patch analysis  (zero cost)
  Layer 2 – Graph-guided BFS path  (zero LLM calls)
  Layer 3 – LLM semantic localization (single call)
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from loguru import logger

from .graph_utils import (
    build_chain_skeleton,
    find_node_by_file_entity,
    find_upstream_callers,
)
from .prompts import LOCALIZATION_PROMPT
from ...modules.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LocalizationResult:
    """Output of Stage 0 for a single instance."""

    instance_id: str
    raw_output_loc: List[str] = field(default_factory=list)
    found_files: List[str] = field(default_factory=list)
    found_modules: List[str] = field(default_factory=list)
    found_entities: List[str] = field(default_factory=list)

    # Quality metrics (populated by validation step)
    quality: Dict[str, Any] = field(default_factory=dict)

    # Internal: graph node IDs resolved during Layer 2
    graph_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "raw_output_loc": self.raw_output_loc,
            "found_files": self.found_files,
            "found_modules": self.found_modules,
            "found_entities": self.found_entities,
            "quality": self.quality,
            "graph_node_ids": self.graph_node_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalizationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Patch parsing helpers (adapted from prepare_multi_repo.py)
# ---------------------------------------------------------------------------

def _parse_patch_files(patch: str) -> List[str]:
    """Extract modified file paths from a unified diff."""
    files: List[str] = []
    for line in patch.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fp = line[6:].strip()
            if fp and fp != "/dev/null" and fp not in files:
                files.append(fp)
    return files


def _parse_patch_entities(patch: str) -> List[str]:
    """Extract modified function/class names from @@ hunk headers."""
    entities: List[str] = []
    current_file: Optional[str] = None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("@@") and current_file:
            m = re.search(r"@@.*@@\s+(?:def|class)\s+(\w+)", line)
            if m:
                entity = f"{current_file}:{m.group(1)}"
                if entity not in entities:
                    entities.append(entity)
    return entities


def _parse_patch_modules(patch: str, files: List[str]) -> List[str]:
    """Extract module-level entities from patch."""
    modules: List[str] = []
    for fp in files:
        file_entities: List[str] = []
        current_file: Optional[str] = None
        for line in patch.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
            elif line.startswith("@@") and current_file == fp:
                m = re.search(r"@@.*@@\s+(?:def|class)\s+(\w+)", line)
                if m:
                    entity = f"{fp}:{m.group(1)}"
                    if entity not in file_entities:
                        file_entities.append(entity)
        if file_entities:
            modules.extend(file_entities)
        else:
            modules.append(fp)
    return modules


# ---------------------------------------------------------------------------
# FaultLocalizer
# ---------------------------------------------------------------------------

class FaultLocalizer:
    """
    Stage 0 fault localizer.

    Produces a ``LocalizationResult`` that is wire-compatible with the
    ``raw_output_loc / found_files / found_modules / found_entities`` fields
    on ``SWEBenchInstance``, so Stage 1 can consume it transparently.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm_client
        self.config = config or {}
        self.max_depth = self.config.get("max_depth", 6)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def localize(
        self,
        instance_id: str,
        repo: str,
        problem_statement: str,
        patch: str,
        test_patch: str,
        graph: Optional[nx.DiGraph] = None,
    ) -> LocalizationResult:
        """Run the three-layer localization pipeline."""
        result = LocalizationResult(instance_id=instance_id)

        # Layer 1: static patch analysis
        self._layer1_static(result, patch, test_patch)

        # Layer 2: graph-guided BFS
        if graph is not None:
            self._layer2_graph(result, graph, test_patch)

        # Layer 3: LLM semantic localization
        if self.llm is not None:
            self._layer3_llm(
                result,
                repo=repo,
                problem_statement=problem_statement,
                patch=patch,
                test_patch=test_patch,
            )

        # Ensure raw_output_loc is never empty: fallback from Layer 1/2
        if not result.raw_output_loc:
            result.raw_output_loc = self._build_fallback_loc(result)

        # Validation
        self._validate(result, patch)

        return result

    # -----------------------------------------------------------------------
    # Layer 1: Static patch analysis (zero cost)
    # -----------------------------------------------------------------------

    def _layer1_static(
        self, result: LocalizationResult, patch: str, test_patch: str
    ) -> None:
        result.found_files = _parse_patch_files(patch)
        result.found_entities = _parse_patch_entities(patch)
        result.found_modules = _parse_patch_modules(patch, result.found_files)

        # Also extract test entry points from test_patch
        test_files = _parse_patch_files(test_patch) if test_patch else []
        result.quality["test_files"] = test_files
        result.quality["layer1_files"] = len(result.found_files)
        result.quality["layer1_entities"] = len(result.found_entities)

        logger.debug(
            f"[Stage 0 L1] {result.instance_id}: "
            f"{len(result.found_files)} files, "
            f"{len(result.found_entities)} entities, "
            f"{len(test_files)} test files"
        )

    # -----------------------------------------------------------------------
    # Layer 2: Graph-guided BFS (zero LLM calls)
    # -----------------------------------------------------------------------

    def _layer2_graph(
        self, result: LocalizationResult, graph: nx.DiGraph, test_patch: str
    ) -> None:
        test_files = result.quality.get("test_files", [])

        # Resolve test entry nodes
        test_nodes: List[str] = []
        for tf in test_files:
            for entity in _parse_patch_entities(test_patch):
                nid = find_node_by_file_entity(
                    graph, entity.split(":")[0], entity.split(":")[-1]
                )
                if nid:
                    test_nodes.append(nid)
            if not test_nodes:
                nid = find_node_by_file_entity(graph, tf)
                if nid:
                    test_nodes.append(nid)

        # Resolve root-cause nodes from patch entities
        rc_nodes: List[str] = []
        for entity in result.found_entities:
            parts = entity.split(":", 1)
            nid = find_node_by_file_entity(
                graph, parts[0], parts[1] if len(parts) > 1 else None
            )
            if nid:
                rc_nodes.append(nid)

        # Fallback: use found_files as root-cause nodes
        if not rc_nodes:
            for fp in result.found_files:
                nid = find_node_by_file_entity(graph, fp)
                if nid:
                    rc_nodes.append(nid)

        # BFS: test → root cause
        skeleton = build_chain_skeleton(
            graph, test_nodes, rc_nodes, max_depth=self.max_depth
        )

        # Also collect upstream callers for each root-cause node
        upstream: List[str] = []
        for rc in rc_nodes:
            upstream.extend(find_upstream_callers(graph, rc, max_depth=4))

        result.graph_node_ids = list(dict.fromkeys(skeleton or upstream))
        result.quality["layer2_skeleton_len"] = len(skeleton)
        result.quality["layer2_test_nodes"] = len(test_nodes)
        result.quality["layer2_rc_nodes"] = len(rc_nodes)

        logger.debug(
            f"[Stage 0 L2] {result.instance_id}: "
            f"skeleton={len(skeleton)} nodes, "
            f"test_nodes={len(test_nodes)}, rc_nodes={len(rc_nodes)}"
        )

    # -----------------------------------------------------------------------
    # Layer 3: LLM semantic localization (single call)
    # -----------------------------------------------------------------------

    def _layer3_llm(
        self,
        result: LocalizationResult,
        *,
        repo: str,
        problem_statement: str,
        patch: str,
        test_patch: str,
    ) -> None:
        # Build graph path context string
        if result.graph_node_ids:
            graph_path_context = (
                "- Graph BFS path (test → root cause): "
                + " → ".join(result.graph_node_ids)
            )
        else:
            graph_path_context = ""

        test_files = result.quality.get("test_files", [])

        prompt = LOCALIZATION_PROMPT.format(
            repo=repo,
            problem_statement=problem_statement[:3000],
            patch=patch[:4000],
            found_files=", ".join(result.found_files),
            found_entities=", ".join(result.found_entities),
            test_files=", ".join(test_files),
            graph_path_context=graph_path_context,
        )

        try:
            response = self.llm.complete(prompt)
            result.raw_output_loc = [response]
            result.quality["layer3_response_len"] = len(response)
            logger.debug(
                f"[Stage 0 L3] {result.instance_id}: "
                f"LLM response {len(response)} chars"
            )
        except Exception as e:
            logger.warning(f"[Stage 0 L3] LLM call failed for {result.instance_id}: {e}")
            # Fallback: build raw_output_loc from static analysis
            result.raw_output_loc = self._build_fallback_loc(result)

    def _build_fallback_loc(self, result: LocalizationResult) -> List[str]:
        """Build raw_output_loc from Layer 1/2 results when LLM is unavailable."""
        lines: List[str] = []

        # If Layer 2 produced a graph skeleton, use it as the chain
        if result.graph_node_ids:
            for i, nid in enumerate(result.graph_node_ids):
                parts = nid.split(":", 1)
                fp = parts[0]
                func = parts[1] if len(parts) > 1 else ""
                lines.append(fp)
                if func:
                    lines.append(f"function: {func}")
                if i == 0:
                    lines.append("Description: test entry point / symptom")
                elif i == len(result.graph_node_ids) - 1:
                    lines.append("Description: root cause (modified in patch)")
                else:
                    lines.append("Description: intermediate caller")
                lines.append("")
        else:
            # Fallback to patch entities only
            for entity in result.found_entities:
                parts = entity.split(":", 1)
                fp = parts[0]
                func = parts[1] if len(parts) > 1 else ""
                lines.append(fp)
                if func:
                    lines.append(f"function: {func}")
                lines.append("Description: modified in patch")
                lines.append("")
            if not lines:
                for fp in result.found_files:
                    lines.append(fp)
                    lines.append("Description: modified file")
                    lines.append("")

        return ["\n".join(lines)] if lines else []

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate(self, result: LocalizationResult, patch: str) -> None:
        """
        Four-step validation of localization results.

        1. File existence check (deferred – needs repo checkout)
        2. Graph node match rate
        3. Patch coverage
        4. raw_output_loc format check
        """
        patch_files = _parse_patch_files(patch)
        patch_entities = _parse_patch_entities(patch)

        # V1: file existence – we can only check found_files ⊆ patch_files
        # (full repo check requires checkout, done externally)

        # V2: graph node match rate
        if result.found_entities:
            matched = sum(1 for e in result.found_entities if e in (result.graph_node_ids or []))
            result.quality["graph_match_rate"] = (
                matched / len(result.found_entities) if result.found_entities else 0.0
            )
        else:
            result.quality["graph_match_rate"] = 0.0

        # V3: patch coverage – found_files should cover patch_files
        if patch_files:
            covered = sum(1 for pf in patch_files if pf in result.found_files)
            result.quality["patch_coverage"] = covered / len(patch_files)
        else:
            result.quality["patch_coverage"] = 0.0

        # V4: raw_output_loc format check
        has_structured = False
        for loc in result.raw_output_loc:
            if re.search(r"(?:function|class|line):", loc):
                has_structured = True
                break
        result.quality["format_valid"] = has_structured

        # Composite quality score
        pc = result.quality.get("patch_coverage", 0.0)
        fv = 1.0 if has_structured else 0.5
        sk = min(1.0, result.quality.get("layer2_skeleton_len", 0) / 3.0)
        result.quality["quality_score"] = round(0.4 * pc + 0.3 * fv + 0.3 * sk, 3)

        logger.info(
            f"[Stage 0] {result.instance_id}: "
            f"quality={result.quality['quality_score']:.2f} "
            f"(patch_cov={pc:.1%}, format={'OK' if has_structured else 'WEAK'}, "
            f"skeleton={result.quality.get('layer2_skeleton_len', 0)})"
        )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_localization_cache(
    result: LocalizationResult, cache_dir: Path
) -> Path:
    """Persist a LocalizationResult to JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{result.instance_id}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    return path


def load_localization_cache(
    instance_id: str, cache_dir: Path
) -> Optional[LocalizationResult]:
    """Load a cached LocalizationResult, or None."""
    path = cache_dir / f"{instance_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return LocalizationResult.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load localization cache {path}: {e}")
        return None
