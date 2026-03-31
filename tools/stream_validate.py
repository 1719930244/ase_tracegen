#!/usr/bin/env python3
"""
Stream/async Stage3 validation for an in-progress TraceGen run.

Motivation:
- TraceGen's built-in pipeline runs Stage3 validation only after Stage2 synthesis finishes.
- This tool overlaps Stage3 with Stage2 by continuously watching `2_synthesis/details/*.json`
  and writing validation results to `3_validation/*_validation.json` as soon as each
  synthetic instance appears.

Notes:
- Does NOT modify the running pipeline process.
- Validation results are written in the same format as the pipeline so that the later
  Stage3 phase can skip already-completed instances (status in {valid, invalid, missing_image}).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from loguru import logger

# Ensure repo root is on sys.path so `import src.*` works even when executing from `tools/`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.structures import SynthesisResult
from src.core.utils import ensure_dir
from src.core.repo_profiles import get_repo_profile
from src.modules.synthesis.chain_alignment import calculate_chain_alignment
from src.modules.validation.adapter import ValidationAdapter
from src.modules.validation.constants import ValidationConfig, ValidationStatus
from src.modules.validation.profiles.python import PythonProfile, UnittestProfile
from src.modules.validation.validator import Validator


_STAGE3_MARKERS = (
    "阶段三: 缺陷验证",
    "开启验证阶段",
    "validation phase",
)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except Exception:
        return {}

    try:
        cfg = OmegaConf.load(str(path))
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]
    except Exception:
        return {}


def _read_json_best_effort(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _atomic_write_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _is_django_run(run_cfg: Dict[str, Any], details_dir: Path) -> bool:
    # Prefer data-based detection over path substring matching.
    # Rationale: filenames like "target_instances_non_django.json" contain "django" as a substring
    # but are explicitly non-Django runs.
    try:
        for p in sorted(details_dir.glob("*.json"))[:3]:
            d = _read_json_best_effort(p) or {}
            seed_id = str(d.get("seed_id", "") or "")
            repo = str(d.get("repo", "") or "")
            if repo.strip().lower() == "django/django":
                return True
            if seed_id.strip().lower().startswith("django__django-"):
                return True
    except (OSError, ValueError) as e:
        logger.debug(f"Fallback django detection failed: {e}")
    return False


def _extract_seed_nodes(seed_extraction_chains: Any) -> list[dict[str, Any]]:
    # Runner uses the first seed chain.
    if not seed_extraction_chains:
        return []

    first = seed_extraction_chains[0]
    if isinstance(first, dict):
        nodes = first.get("nodes", []) or []
    else:
        nodes = getattr(first, "nodes", []) or []
    if not isinstance(nodes, list):
        return []

    normalized: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        if isinstance(node, dict):
            node_id = str(node.get("node_id", "") or "")
            file_path = str(node.get("file_path", "") or "")
        else:
            node_id = str(getattr(node, "node_id", "") or "")
            file_path = str(getattr(node, "file_path", "") or "")

        if idx == 0:
            node_type = "symptom"
        elif idx == len(nodes) - 1:
            node_type = "root_cause"
        else:
            node_type = "intermediate"

        normalized.append({"node_id": node_id, "node_type": node_type, "file_path": file_path})
    return normalized


def _extract_synth_chain(metadata: Dict[str, Any]) -> list[dict[str, Any]]:
    chain = (metadata or {}).get("proposed_chain", []) or []
    if isinstance(chain, list):
        return [c for c in chain if isinstance(c, dict)]
    return []


def _stage3_started(run_dir: Path) -> bool:
    log_path = run_dir / "pipeline.log"
    if not log_path.exists():
        return False
    try:
        # Read only the tail for speed.
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256_000), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore")
        return any(m in tail for m in _STAGE3_MARKERS)
    except Exception:
        return False


def _should_skip_existing(validation_json: Path) -> bool:
    if not validation_json.exists():
        return False
    try:
        data = _read_json_best_effort(validation_json) or {}
        status = str(data.get("status", "") or "")
        return status in {
            ValidationStatus.VALID.value,
            ValidationStatus.INVALID.value,
            ValidationStatus.MISSING_IMAGE.value,
        }
    except Exception:
        return False


def _iter_details(details_dir: Path) -> Iterable[Path]:
    # Details are written as `<instance_id>_rankK.json` by the pipeline.
    yield from sorted(details_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)


def _build_validator(
    *,
    run_cfg: Dict[str, Any],
    run_dir: Path,
    details_dir: Path,
    validation_dir: Path,
) -> dict[str, Any]:
    validation_cfg = (run_cfg or {}).get("validation", {}) or {}
    val_config = ValidationConfig(
        mode="injection",
        timeout=int(validation_cfg.get("timeout", 300) or 300),
        memory_limit=str(validation_cfg.get("memory_limit", "4g") or "4g"),
        clean_containers=bool(validation_cfg.get("clean_containers", True)),
        verbose=bool(validation_cfg.get("verbose", False)),
        enforce_chain_coverage=bool(validation_cfg.get("enforce_chain_coverage", False)),
        min_chain_coverage=float(validation_cfg.get("min_chain_coverage", 0.34) or 0.34),
        require_target_node_in_traceback=bool(validation_cfg.get("require_target_node_in_traceback", False)),
        log_dir=validation_dir / "logs",
    )

    default_image = str(validation_cfg.get("default_image", "python:3.10-slim") or "python:3.10-slim")
    return {
        "validation_cfg": dict(validation_cfg),
        "val_config": val_config,
        "default_image": default_image,
        "is_django_run": _is_django_run(run_cfg, details_dir),
    }


def _validate_one(
    *,
    details_path: Path,
    validation_path: Path,
    ctx: dict[str, Any],
) -> Dict[str, Any]:
    payload = _read_json_best_effort(details_path)
    if not payload:
        raise RuntimeError(f"Failed to read details json: {details_path}")

    result = SynthesisResult.model_validate(payload)

    repo_profile = get_repo_profile(result.repo or "")
    adapter = ValidationAdapter(dict(ctx.get("validation_cfg", {}) or {}), repo_profile=repo_profile)

    default_image = str(ctx.get("default_image", "python:3.10-slim") or "python:3.10-slim")
    profile = UnittestProfile(image_name=default_image) if repo_profile.is_django else PythonProfile(image_name=default_image)
    validator = Validator(profile=profile, config=ctx["val_config"], repo_profile=repo_profile)

    instance_dict = adapter.adapt(result)
    val_result = validator.validate(instance_dict)

    # Add structural alignment as a sub-field (don't overwrite traceback-based score).
    try:
        seed_nodes = _extract_seed_nodes(result.seed_extraction_chains or [])
        synth_chain = _extract_synth_chain(result.metadata or {})
        if seed_nodes and synth_chain:
            alignment = calculate_chain_alignment(seed_nodes, synth_chain)
            if not val_result.chain_alignment_score:
                val_result.chain_alignment_score = {}
            val_result.chain_alignment_score["structural_alignment"] = alignment.to_dict()
    except Exception as e:
        logger.warning(f"{result.instance_id}: chain alignment failed: {e}")

    out = val_result.to_dict()
    _atomic_write_json(out, validation_path)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="TraceGen run dir (or a symlink like .../latest)")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent validations")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval for new details")
    parser.add_argument("--once", action="store_true", help="Run one scan then exit")
    parser.add_argument(
        "--stop-on-stage3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop scheduling once pipeline enters Stage3 (default: true)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    details_dir = run_dir / "2_synthesis" / "details"
    validation_dir = run_dir / "3_validation"
    ensure_dir(validation_dir)

    if not details_dir.exists():
        print(f"ERROR: details dir not found: {details_dir}", file=sys.stderr)
        return 2

    run_cfg_path = run_dir / "config_resolved.yaml"
    run_cfg = _load_yaml_config(run_cfg_path) if run_cfg_path.exists() else {}

    ctx = _build_validator(
        run_cfg=run_cfg,
        run_dir=run_dir,
        details_dir=details_dir,
        validation_dir=validation_dir,
    )

    print(f"[stream_validate] run_dir={run_dir}")
    print(f"[stream_validate] details_dir={details_dir}")
    print(f"[stream_validate] validation_dir={validation_dir}")
    print(f"[stream_validate] workers={args.workers} poll_seconds={args.poll_seconds} once={args.once}")
    sys.stdout.flush()

    last_report_ts = 0.0
    validated_ok = 0
    validated_err = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        in_flight: dict[Any, tuple[Path, Path]] = {}

        def submit_pending() -> int:
            submitted = 0
            for details_path in _iter_details(details_dir):
                payload = _read_json_best_effort(details_path)
                if not payload:
                    continue
                instance_id = str(payload.get("instance_id", "") or "").strip()
                if not instance_id:
                    continue
                validation_path = validation_dir / f"{instance_id}_validation.json"

                if _should_skip_existing(validation_path):
                    continue
                if any(vp == validation_path for _dp, vp in in_flight.values()):
                    continue

                fut = executor.submit(
                    _validate_one,
                    details_path=details_path,
                    validation_path=validation_path,
                    ctx=ctx,
                )
                in_flight[fut] = (details_path, validation_path)
                submitted += 1
            return submitted

        while True:
            if args.stop_on_stage3 and _stage3_started(run_dir):
                print("[stream_validate] Detected Stage3 start in pipeline.log; stop scheduling new validations.")
                sys.stdout.flush()
                break

            submitted = submit_pending()
            now = time.time()
            if submitted and now - last_report_ts > 5:
                print(f"[stream_validate] submitted={submitted} inflight={len(in_flight)}")
                sys.stdout.flush()
                last_report_ts = now

            if args.once:
                break

            # Drain completed futures without blocking.
            done = [f for f in in_flight.keys() if f.done()]
            for fut in done:
                details_path, validation_path = in_flight.pop(fut)
                try:
                    out = fut.result()
                    status = str(out.get("status", "") or "")
                    if status in {ValidationStatus.VALID.value, ValidationStatus.INVALID.value, ValidationStatus.MISSING_IMAGE.value}:
                        validated_ok += 1
                    else:
                        validated_err += 1
                    print(f"[stream_validate] done {validation_path.name} status={status}")
                except Exception as e:
                    validated_err += 1
                    print(f"[stream_validate] failed {details_path.name}: {e}")
                sys.stdout.flush()

            time.sleep(max(1.0, float(args.poll_seconds)))

        # Finish remaining tasks (best-effort) when stopping.
        if in_flight:
            print(f"[stream_validate] waiting inflight={len(in_flight)} ...")
            sys.stdout.flush()
            for fut in as_completed(list(in_flight.keys())):
                details_path, validation_path = in_flight[fut]
                try:
                    out = fut.result()
                    status = str(out.get("status", "") or "")
                    print(f"[stream_validate] done {validation_path.name} status={status}")
                except Exception as e:
                    print(f"[stream_validate] failed {details_path.name}: {e}")
                sys.stdout.flush()

    print(f"[stream_validate] summary ok={validated_ok} other={validated_err}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
