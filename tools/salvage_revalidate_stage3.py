#!/usr/bin/env python3
"""
Salvage script: re-run Stage3 validation for an existing TraceGen Stage2 output.

Why this exists:
- The original 4repo run (pytest + sklearn + sympy) can be massively mis-scored when
  validation uses the wrong repo profile / test command (e.g., Django's ./tests/runtests.py).
- The pipeline's Stage3 assumes a single repo_profile per run, which is incompatible with
  multi-repo synthetic datasets.

This tool validates per instance, routing by `result.repo`:
- loads `2_synthesis/details/*_rank*.json`
- builds a RepoProfile for each repo
- generates a repo-aware test command
- runs the Docker-based validator
- writes `3_validation/*_validation.json` and logs (same structure as pipeline)
- writes a merged summary JSON as the final artifact

Run with the project's Python (3.11), e.g.:
  python tools/salvage_revalidate_stage3.py --run-dir ../tracegen-outputs/4repo_run/stage2_full
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

# Ensure repo root on sys.path (so `import src.*` works when executing from tools/)
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.repo_profiles import get_repo_profile
from src.core.structures import SynthesisResult
from src.core.utils import ensure_dir
from src.modules.validation.adapter import ValidationAdapter
from src.modules.validation.constants import ValidationConfig, ValidationStatus
from src.modules.validation.profiles.python import PythonProfile, UnittestProfile
from src.modules.validation.validator import Validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_yaml_best_effort(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_validation_cfg(run_dir: Path) -> dict[str, Any]:
    """
    Extract validation config from `config_resolved.yaml` or `run_manifest.json`.

    `config_resolved.yaml` is preferred because it's the resolved Hydra config, but
    some runs may not include all keys; `run_manifest.json` is a robust fallback.
    """
    cfg = {}
    resolved = run_dir / "config_resolved.yaml"
    if resolved.exists():
        cfg = _load_yaml_best_effort(resolved)

    # Common layouts:
    # - {"validation": {...}}
    # - {"tracegen": {"validation": {...}}}
    for keypath in (("validation",), ("tracegen", "validation")):
        cur: Any = cfg
        ok = True
        for k in keypath:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, dict) and cur:
            return dict(cur)

    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        try:
            m = _read_json(manifest)
            v = ((m or {}).get("tracegen", {}) or {}).get("validation", {}) or {}
            if isinstance(v, dict) and v:
                return dict(v)
        except Exception:
            pass

    return {}


_RANK_RE = re.compile(r"^(?P<iid>.+)_rank(?P<rank>\d+)\.json$")


def _iter_best_details(details_dir: Path) -> Iterable[Path]:
    """
    Yield one details json per `instance_id`.

    Pipeline writes `<instance_id>_rankK.json`. If multiple ranks exist for the same
    instance_id, we keep the lowest rank number (rank1).
    """
    best_by_iid: dict[str, tuple[int, Path]] = {}
    for p in details_dir.glob("*.json"):
        m = _RANK_RE.match(p.name)
        if not m:
            # Unknown naming - treat whole stem as iid and pick by mtime
            iid = p.stem
            rank = 999_999
        else:
            iid = m.group("iid")
            rank = int(m.group("rank"))

        prev = best_by_iid.get(iid)
        if prev is None or rank < prev[0]:
            best_by_iid[iid] = (rank, p)

    # stable order: by mtime of chosen file
    chosen = [p for _rank, p in best_by_iid.values()]
    yield from sorted(chosen, key=lambda x: x.stat().st_mtime)


def _should_skip_existing(path: Path, *, policy: str) -> bool:
    if not path.exists():
        return False
    policy = (policy or "done").strip().lower()
    if policy == "all":
        return True
    try:
        data = _read_json(path) or {}
        status = str(data.get("status", "") or "")
        return status in {
            ValidationStatus.VALID.value,
            ValidationStatus.INVALID.value,
            ValidationStatus.MISSING_IMAGE.value,
        }
    except Exception:
        return False


def _is_valid_bug_from_json(d: dict[str, Any]) -> bool:
    """
    Mirror ValidationResult.is_valid_bug() without importing dataclass instances.
    """
    if str(d.get("status", "")) != ValidationStatus.VALID.value:
        return False
    p2f = d.get("PASS_TO_FAIL") or []
    p2p = d.get("PASS_TO_PASS") or []
    f2p = d.get("FAIL_TO_PASS") or []
    f2f = d.get("FAIL_TO_FAIL") or []
    return (len(p2f) > 0) and (len(p2p) > 0) and (len(f2p) == 0) and (len(f2f) == 0)


def _summarize_validation_dir(validation_dir: Path) -> dict[str, Any]:
    """
    Summarize all `*_validation.json` under `validation_dir` (run-resumable).

    This is the source of truth for the "final result" because a salvage run may
    be resumed multiple times with `--force` off (skipping already validated instances).
    """
    status_counts: Counter[str] = Counter()
    per_repo_status: dict[str, Counter[str]] = defaultdict(Counter)
    per_repo_valid_bug: Counter[str] = Counter()
    valid_bug_total = 0
    total = 0

    for p in sorted(validation_dir.glob("*_validation.json")):
        try:
            d = _read_json(p) or {}
        except Exception:
            continue

        status = str(d.get("status", "") or "")
        repo = str(d.get("repo", "") or "")
        status_counts[status] += 1
        per_repo_status[repo][status] += 1
        total += 1

        if _is_valid_bug_from_json(d):
            valid_bug_total += 1
            per_repo_valid_bug[repo] += 1

    return {
        "total": int(total),
        "status_counts": dict(status_counts),
        "per_repo_status_counts": {k: dict(v) for k, v in sorted(per_repo_status.items())},
        "per_repo_valid_bug_counts": dict(per_repo_valid_bug),
        "valid_bug_total": int(valid_bug_total),
    }


def _validate_one(
    *,
    details_path: Path,
    out_validation_json: Path,
    val_config: ValidationConfig,
    validation_cfg: dict[str, Any],
    force: bool,
    disable_auto_pull: bool,
    skip_existing_policy: str,
) -> dict[str, Any]:
    if (not force) and _should_skip_existing(out_validation_json, policy=skip_existing_policy):
        return {"skipped": True, "path": str(out_validation_json)}

    payload = _read_json(details_path)
    result = SynthesisResult.model_validate(payload)
    repo = (result.repo or "").strip()

    repo_profile = get_repo_profile(repo)

    adapter_cfg = dict(validation_cfg or {})
    adapter_cfg["enabled"] = True
    if disable_auto_pull:
        adapter_cfg["auto_pull"] = False

    adapter = ValidationAdapter(adapter_cfg, repo_profile=repo_profile)

    # Validator profile is still "python"; instance["test_cmd"] determines the command.
    default_image = str(adapter_cfg.get("default_image", "python:3.10-slim") or "python:3.10-slim")
    profile = UnittestProfile(image_name=default_image) if repo_profile.is_django else PythonProfile(image_name=default_image)
    validator = Validator(profile=profile, config=val_config, repo_profile=repo_profile)

    instance_dict = adapter.adapt(result)
    val_result = validator.validate(instance_dict)

    out = val_result.to_dict()
    # Add extra fields for post-analysis; pipeline ignores unknown keys.
    out["repo"] = repo
    out["image_name"] = instance_dict.get("image_name")
    out["test_cmd"] = instance_dict.get("test_cmd")
    _atomic_write_json(out_validation_json, out)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Existing TraceGen run dir (contains 2_synthesis/details)")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory for salvage validation results (default: sibling dir next to run-dir)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent validations")
    parser.add_argument("--max-instances", type=int, default=0, help="Limit instances (0 = no limit)")
    parser.add_argument("--repos", default="", help="Comma-separated repo allowlist, e.g. 'pytest-dev/pytest,sympy/sympy'")
    parser.add_argument("--force", action="store_true", help="Re-run even if *_validation.json already exists")
    parser.add_argument(
        "--skip-existing",
        default="done",
        choices=["done", "all"],
        help="Skip policy for existing *_validation.json. "
        "'done' skips {valid, invalid, missing_image}; 'all' skips any existing file.",
    )
    parser.add_argument(
        "--disable-auto-pull",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Docker image auto-pull to avoid concurrent pulls (default: true)",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Override validation timeout seconds (0 = use run cfg)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    details_dir = run_dir / "2_synthesis" / "details"
    if not details_dir.exists():
        raise SystemExit(f"details dir not found: {details_dir}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (run_dir.parent / f"{run_dir.name}_salvage_validation")
    validation_dir = out_dir / "3_validation"
    ensure_dir(validation_dir)

    validation_cfg = _extract_validation_cfg(run_dir)
    logger.info(f"Loaded validation cfg keys: {sorted(validation_cfg.keys())}")

    timeout = int(args.timeout or validation_cfg.get("timeout", 300) or 300)
    memory_limit = str(validation_cfg.get("memory_limit", "4g") or "4g")
    clean_containers = bool(validation_cfg.get("clean_containers", True))
    verbose = bool(validation_cfg.get("verbose", False))
    enforce_chain_coverage = bool(validation_cfg.get("enforce_chain_coverage", False))
    min_chain_coverage = float(validation_cfg.get("min_chain_coverage", 0.34) or 0.34)
    require_target_node_in_traceback = bool(validation_cfg.get("require_target_node_in_traceback", False))

    val_config = ValidationConfig(
        mode="injection",
        timeout=timeout,
        memory_limit=memory_limit,
        clean_containers=clean_containers,
        verbose=verbose,
        enforce_chain_coverage=enforce_chain_coverage,
        min_chain_coverage=min_chain_coverage,
        require_target_node_in_traceback=require_target_node_in_traceback,
        log_dir=validation_dir / "logs",
    )

    allow_repos = {r.strip().lower() for r in (args.repos.split(",") if args.repos else []) if r.strip()}

    details_paths = []
    for p in _iter_best_details(details_dir):
        if args.max_instances and len(details_paths) >= int(args.max_instances):
            break
        # Filter by repo (without parsing pydantic unless needed)
        if allow_repos:
            try:
                payload = _read_json(p) or {}
                repo = str(payload.get("repo", "") or "").strip().lower()
                if repo and repo not in allow_repos:
                    continue
            except Exception:
                continue
        details_paths.append(p)

    logger.info(f"run_dir={run_dir}")
    logger.info(f"out_dir={out_dir}")
    logger.info(f"details={len(details_paths)} (workers={args.workers}, force={args.force})")

    start = time.time()
    status_counter: Counter[str] = Counter()
    repo_status: dict[str, Counter[str]] = defaultdict(Counter)
    repo_valid_bug: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    skipped = 0
    validated = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = {}
        for details_path in details_paths:
            try:
                payload = _read_json(details_path) or {}
                instance_id = str(payload.get("instance_id", "") or "").strip()
                repo = str(payload.get("repo", "") or "").strip()
            except Exception:
                continue
            if not instance_id:
                continue
            out_validation_json = validation_dir / f"{instance_id}_validation.json"
            fut = ex.submit(
                _validate_one,
                details_path=details_path,
                out_validation_json=out_validation_json,
                val_config=val_config,
                validation_cfg=validation_cfg,
                force=bool(args.force),
                disable_auto_pull=bool(args.disable_auto_pull),
                skip_existing_policy=str(args.skip_existing),
            )
            futs[fut] = {"instance_id": instance_id, "repo": repo, "details_path": str(details_path)}

        for fut in as_completed(list(futs.keys())):
            meta = futs[fut]
            try:
                out = fut.result()
                if out.get("skipped"):
                    skipped += 1
                    continue
                validated += 1
                repo = str(out.get("repo", meta.get("repo", "")) or "")
                status = str(out.get("status", "") or "")
                status_counter[status] += 1
                repo_status[repo][status] += 1
                if _is_valid_bug_from_json(out):
                    repo_valid_bug[repo] += 1
            except Exception as e:
                validated += 1
                status_counter["error"] += 1
                repo = meta.get("repo", "")
                repo_status[repo]["error"] += 1
                errors.append({**meta, "error": str(e)})

    elapsed = time.time() - start

    final_summary = _summarize_validation_dir(validation_dir)

    summary: dict[str, Any] = {
        "timestamp_utc": _utc_now_iso(),
        "source_run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "config": {
            "workers": int(args.workers),
            "force": bool(args.force),
            "disable_auto_pull": bool(args.disable_auto_pull),
            "timeout": timeout,
            "memory_limit": memory_limit,
            "repos_filter": sorted(allow_repos),
            "max_instances": int(args.max_instances),
        },
        "counts": {
            "details_considered": len(details_paths),
            "validated": validated,
            "skipped": skipped,
            "elapsed_seconds": round(elapsed, 3),
        },
        # These counters reflect only what happened in this invocation.
        "invocation_status_counts": dict(status_counter),
        "invocation_errors_sample": errors[:30],
        # These counters reflect the "final state" across all existing outputs.
        "final_status_counts": final_summary["status_counts"],
        "final_per_repo_status_counts": final_summary["per_repo_status_counts"],
        "final_per_repo_valid_bug_counts": final_summary["per_repo_valid_bug_counts"],
        "final_valid_bug_total": final_summary["valid_bug_total"],
        "final_total_instances": final_summary["total"],
        "errors_sample": errors[:30],
    }

    # Write a merged summary as the "final result" artifact.
    _atomic_write_json(out_dir / "merged_summary.json", summary)

    # Also write a short report.json for convenience (mirrors pipeline's convention).
    report = {
        "timestamp": summary["timestamp_utc"],
        "validation": {
            "total": int(final_summary["total"]),
            "skipped": int(skipped),
            "status_counts": summary["final_status_counts"],
            "valid_bug_total": summary["final_valid_bug_total"],
            "per_repo_valid_bug_counts": summary["final_per_repo_valid_bug_counts"],
        },
        "output_structure": {
            "validation_dir": "3_validation",
            "validation_logs": "3_validation/logs",
            "merged_summary": "merged_summary.json",
        },
    }
    _atomic_write_json(out_dir / "report.json", report)

    logger.info(f"Wrote {out_dir / 'merged_summary.json'}")
    logger.info(f"Wrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
