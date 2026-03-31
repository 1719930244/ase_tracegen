"""
SphinxProfile — sphinx-doc/sphinx 仓库的 RepoProfile 实现。

sphinx 的目录结构：sphinx/<pkg>/... → tests/test_<domain>.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class SphinxProfile(RepoProfile):
    """sphinx-doc/sphinx 仓库 Profile。"""

    repo: str = "sphinx-doc/sphinx"
    test_framework: str = "pytest"
    test_root: str = "tests"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        sphinx 的约定：
        sphinx/ext/autodoc/mock.py → tests/test_ext_autodoc.py, tests/test_ext_autodoc_mock.py
        sphinx/pycode/__init__.py → tests/test_pycode.py
        sphinx/builders/latex/... → tests/test_build_latex.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        if source_file.startswith("tests/"):
            return [source_file]

        # Build search patterns from path components
        # sphinx/ext/napoleon/docstring.py → test_ext_napoleon_docstring, test_ext_napoleon
        parts = Path(source_file).parts
        search_stems = []
        if len(parts) >= 2 and parts[0] == "sphinx":
            # sphinx/ext/autodoc/mock.py → ext_autodoc_mock, ext_autodoc
            path_parts = list(parts[1:])
            if path_parts[-1].endswith(".py"):
                path_parts[-1] = Path(path_parts[-1]).stem
            if path_parts[-1] == "__init__":
                path_parts = path_parts[:-1]
            # Build progressively shorter stems
            for i in range(len(path_parts), 0, -1):
                search_stems.append("_".join(path_parts[:i]))

        candidates = [f"tests/test_{s}.py" for s in search_stems]
        # Also try with "build" prefix for builders
        if "builders" in source_file:
            builder_type = stem if stem != "__init__" else parts[-2] if len(parts) > 2 else ""
            if builder_type:
                candidates.insert(0, f"tests/test_build_{builder_type}.py")

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # Fallback: search tests/ for files containing the stem
            tests_dir = repo_root / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob(f"test_*{stem}*.py")
                ]
                if found:
                    return found[:5]

        return candidates[:1] if candidates else [f"tests/test_{stem}.py"]
