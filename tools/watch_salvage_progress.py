#!/usr/bin/env python3
"""
Cron-friendly watcher for TraceGen salvage validation progress.

This script is intended to be run periodically (e.g. every 5 minutes) to:
- count existing `*_validation.json` in `OUT_DIR/3_validation/`
- compare against expected totals inferred from `RUN_DIR/2_synthesis/details/`
- write lightweight progress artifacts under OUT_DIR
- when complete, refresh `report.json`/`merged_summary.json` and drop a DONE marker

It does NOT start validations. It only monitors and finalizes summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _infer_expected_counts(run_dir: Path) -> dict[str, int]:
    """
    Build expected per-repo totals from `RUN_DIR/2_synthesis/details/*.json`.
    We count unique instance_ids, and pick the best-rank details file per iid.
    """
    details_dir = run_dir / "2_synthesis" / "details"
    if not details_dir.exists():
        raise FileNotFoundError(f"details dir not found: {details_dir}")

    # Resolve "best details per iid": choose the smallest rank number when multiple exist.
    import re

    pat = re.compile(r"^(?P<iid>.+?)_rank(?P<rank>\d+)\.json$")
    best: dict[str, tuple[int, Path]] = {}
    for p in details_dir.glob("*.json"):
        m = pat.match(p.name)
        if not m:
            # allow older naming conventions: treat as rank=inf
            iid = p.stem
            rank = 999_999
        else:
            iid = m.group("iid")
            rank = int(m.group("rank"))
        prev = best.get(iid)
        if prev is None or rank < prev[0]:
            best[iid] = (rank, p)

    expected_by_repo: Counter[str] = Counter()
    for _rank, p in best.values():
        try:
            payload = _read_json(p) or {}
        except Exception:
            continue
        repo = str(payload.get("repo", "") or "").strip()
        if not repo:
            continue
        expected_by_repo[repo] += 1
    return dict(expected_by_repo)


def _count_validation(out_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    validation_dir = out_dir / "3_validation"
    by_repo: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    if not validation_dir.exists():
        return {}, {}

    for p in validation_dir.glob("*_validation.json"):
        try:
            d = _read_json(p) or {}
        except Exception:
            continue
        repo = str(d.get("repo", "") or "").strip()
        status = str(d.get("status", "") or "").strip()
        by_repo[repo] += 1
        by_status[status] += 1
    return dict(by_repo), dict(by_status)


def _find_runner_pid(out_dir: Path) -> int | None:
    """
    Best-effort: find a salvage runner whose command line mentions this out_dir.
    """
    needle_abs = str(out_dir.resolve())
    needle_base = out_dir.name

    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], universal_newlines=True)
    except Exception:
        return None

    matches: list[int] = []
    runners: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if "tools/salvage_revalidate_stage3.py" not in line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        pid_s = parts[0]
        if not pid_s.isdigit():
            continue
        pid = int(pid_s)
        runners.append(pid)
        if (needle_abs in line) or (needle_base in line):
            matches.append(pid)

    if matches:
        return matches[0]
    return runners[0] if len(runners) == 1 else None


def _refresh_summaries(py: Path, run_dir: Path, out_dir: Path) -> None:
    """
    Re-run salvage tool in summary-only mode (1 instance, skip all existing)
    to rewrite `report.json` and `merged_summary.json` based on current outputs.
    """
    salvage = Path(__file__).resolve().parents[1] / "tools" / "salvage_revalidate_stage3.py"
    # Use max_instances=1 + skip-existing all to avoid any real validation work.
    subprocess.check_call(
        [
            str(py),
            str(salvage),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
            "--disable-auto-pull",
            "--skip-existing",
            "all",
            "--max-instances",
            "1",
        ]
    )


@dataclass(frozen=True)
class Progress:
    expected_by_repo: dict[str, int]
    actual_by_repo: dict[str, int]
    actual_by_status: dict[str, int]

    @property
    def expected_total(self) -> int:
        return int(sum(self.expected_by_repo.values()))

    @property
    def actual_total(self) -> int:
        return int(sum(self.actual_by_repo.values()))

    def is_complete(self) -> bool:
        if self.expected_total <= 0:
            return False
        if self.actual_total != self.expected_total:
            return False
        for repo, exp in self.expected_by_repo.items():
            if int(self.actual_by_repo.get(repo, 0)) != int(exp):
                return False
        return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--python",
        default="python",
        help="Python used to refresh summaries when complete",
    )
    ap.add_argument("--cache-expected-seconds", type=int, default=3600)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    py = Path(args.python).expanduser().resolve()

    expected_cache = out_dir / "expected_counts.json"
    expected: dict[str, int] = {}
    now = time.time()
    if expected_cache.exists():
        try:
            age = now - expected_cache.stat().st_mtime
            if age < int(args.cache_expected_seconds):
                expected = _read_json(expected_cache) or {}
        except Exception:
            expected = {}
    if not expected:
        expected = _infer_expected_counts(run_dir)
        _atomic_write_json(expected_cache, expected)

    actual_by_repo, actual_by_status = _count_validation(out_dir)

    progress = Progress(
        expected_by_repo={k: int(v) for k, v in expected.items()},
        actual_by_repo={k: int(v) for k, v in actual_by_repo.items()},
        actual_by_status={k: int(v) for k, v in actual_by_status.items()},
    )

    runner_pid = _find_runner_pid(out_dir)
    meta = {
        "timestamp_utc": _utc_now_iso(),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "runner_pid": runner_pid,
        "expected_total": progress.expected_total,
        "actual_total": progress.actual_total,
        "expected_by_repo": progress.expected_by_repo,
        "actual_by_repo": progress.actual_by_repo,
        "actual_by_status": progress.actual_by_status,
        "complete": progress.is_complete(),
    }

    _atomic_write_json(out_dir / "progress_latest.json", meta)

    # Human-friendly one-liner for tail -f
    line = (
        f"[{meta['timestamp_utc']}] total={progress.actual_total}/{progress.expected_total} "
        f"sklearn={progress.actual_by_repo.get('scikit-learn/scikit-learn',0)}/{progress.expected_by_repo.get('scikit-learn/scikit-learn',0)} "
        f"sympy={progress.actual_by_repo.get('sympy/sympy',0)}/{progress.expected_by_repo.get('sympy/sympy',0)} "
        f"pytest={progress.actual_by_repo.get('pytest-dev/pytest',0)}/{progress.expected_by_repo.get('pytest-dev/pytest',0)} "
        f"pid={runner_pid or '-'} complete={meta['complete']}"
    )
    with (out_dir / "progress_watch.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    done_marker = out_dir / "DONE"
    if meta["complete"] and not done_marker.exists():
        # Refresh summaries based on final outputs.
        _refresh_summaries(py=py, run_dir=run_dir, out_dir=out_dir)

        _atomic_write(done_marker, meta["timestamp_utc"] + "\n")
        _atomic_write(
            out_dir / "FINAL_READY.txt",
            "Salvage validation is complete.\n"
            f"- out_dir: {out_dir}\n"
            "- See: report.json and merged_summary.json\n",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
