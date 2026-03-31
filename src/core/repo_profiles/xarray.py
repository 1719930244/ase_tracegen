"""
XarrayProfile — pydata/xarray 仓库的 RepoProfile 实现。

xarray 的目录结构：xarray/core/... → xarray/tests/test_<mod>.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class XarrayProfile(RepoProfile):
    """pydata/xarray 仓库 Profile。"""

    repo: str = "pydata/xarray"
    test_framework: str = "pytest"
    test_root: str = "xarray/tests"
    test_cmd_template: str = "pytest {targets} -v --tb=short"
    source_to_test_mapping: dict[str, list[str]] = field(default_factory=lambda: {
        "xarray/core/alignment.py": ["xarray/tests/test_dataarray.py", "xarray/tests/test_dataset.py"],
    })

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        xarray 的约定：
        xarray/core/computation.py → xarray/tests/test_computation.py
        xarray/conventions.py → xarray/tests/test_conventions.py
        xarray/core/duck_array_ops.py → xarray/tests/test_duck_array_ops.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        if "/tests/" in source_file:
            return [source_file]

        candidates = [
            f"xarray/tests/test_{stem}.py",
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # Search xarray/tests/ for matching files
            tests_dir = repo_root / "xarray" / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob(f"test_*{stem}*.py")
                ]
                if found:
                    return found[:5]

        return candidates[:1]
