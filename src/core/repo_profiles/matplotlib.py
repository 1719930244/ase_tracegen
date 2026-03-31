"""
MatplotlibProfile — matplotlib/matplotlib 仓库的 RepoProfile 实现。

matplotlib 的目录结构：lib/matplotlib/<pkg>/... → lib/matplotlib/tests/test_<mod>.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class MatplotlibProfile(RepoProfile):
    """matplotlib/matplotlib 仓库 Profile。"""

    repo: str = "matplotlib/matplotlib"
    test_framework: str = "pytest"
    test_root: str = "lib/matplotlib/tests"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        matplotlib 的约定：
        lib/matplotlib/artist.py → lib/matplotlib/tests/test_artist.py
        lib/matplotlib/colors.py → lib/matplotlib/tests/test_colors.py
        lib/mpl_toolkits/... → lib/mpl_toolkits/tests/test_*.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        if "/tests/" in source_file:
            return [source_file]

        # lib/matplotlib/*.py → lib/matplotlib/tests/test_*.py
        if source_file.startswith("lib/matplotlib/"):
            candidates = [f"lib/matplotlib/tests/test_{stem}.py"]
        elif source_file.startswith("lib/mpl_toolkits/"):
            parent = str(p.parent)
            candidates = [f"{parent}/tests/test_{stem}.py"]
        else:
            candidates = [f"lib/matplotlib/tests/test_{stem}.py"]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing
            # Broader search in test dir
            test_dir = repo_root / "lib/matplotlib/tests"
            if test_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in test_dir.glob(f"test_*{stem}*.py")
                ]
                if found:
                    return found[:5]

        return candidates[:1]
