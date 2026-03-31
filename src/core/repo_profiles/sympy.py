"""
SympyProfile — sympy/sympy 仓库的 RepoProfile 实现。

sympy 有非常规律的目录结构：sympy/<pkg>/tests/test_<mod>.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class SympyProfile(RepoProfile):
    """sympy/sympy 仓库 Profile。"""

    repo: str = "sympy/sympy"
    test_framework: str = "sympy_bin_test"
    test_root: str = "sympy"
    test_cmd_template: str = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {targets}"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """sympy 的约定：sympy/core/add.py → sympy/core/tests/test_add.py"""
        p = Path(source_file)
        stem = p.stem.lstrip("_")
        parent = str(p.parent)

        # 主要约定：同包下的 tests/ 子目录
        candidates = [
            f"{parent}/tests/test_{stem}.py",
        ]

        # 如果源文件在 tests/ 下，直接返回
        if "/tests/" in source_file:
            return [source_file]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # 尝试在同包 tests/ 下搜索包含 stem 的测试文件
            tests_dir = repo_root / parent / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob("test_*.py")
                    if stem in f.stem
                ]
                if found:
                    return found[:5]

        return candidates[:1]

    def get_test_file_path(self, source_file: str) -> str:
        """合成测试文件放在对应包的 tests/ 下。"""
        p = Path(source_file)
        parent = str(p.parent)
        return f"{parent}/tests/test_synthetic.py"

    def infer_seed_test_cmd(self, seed_info: dict, max_total_labels: int = 6) -> str:
        """从 seed 元数据推断 sympy bin/test 测试命令。"""
        if not isinstance(seed_info, dict):
            return ""

        f2p = seed_info.get("FAIL_TO_PASS", [])
        p2p = seed_info.get("PASS_TO_PASS", [])
        if isinstance(f2p, str):
            try:
                import json
                f2p = json.loads(f2p)
            except Exception:
                f2p = [f2p] if f2p.strip() else []
        if isinstance(p2p, str):
            try:
                import json
                p2p = json.loads(p2p)
            except Exception:
                p2p = [p2p] if p2p.strip() else []

        # 提取唯一的测试文件路径
        files: list[str] = []
        for label in (f2p + p2p):
            if not isinstance(label, str):
                continue
            # 可能是 file.py 或 file.py::class::method 格式
            parts = label.split("::")
            if parts and parts[0] not in files:
                files.append(parts[0])

        files = files[:max_total_labels]
        if not files:
            return ""
        return f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(files)}"

    def build_validation_test_cmd(
        self,
        fail_to_pass: list[str] | None,
        pass_to_pass: list[str] | None,
        test_selection_mode: str = "module_suite",
        max_fail_labels: int = 1,
        max_pass_labels: int = 1,
        max_total_labels: int = 6,
        modified_files: list[str] | None = None,
        target_node: str = "",
        planned_test_cmd: str = "",
        planned_test_modules: list[str] | None = None,
    ) -> str:
        """
        构建sympy验证测试命令。

        sympy特殊处理：测试标签可能缺少文件路径，需要推断。
        """
        from pathlib import Path
        from loguru import logger

        # 策略1: 使用planned_test_cmd (但必须是 bin/test 命令，拒绝 pytest)
        if planned_test_cmd and planned_test_cmd.strip():
            cmd = planned_test_cmd.strip()
            if "bin/test" in cmd:
                return cmd
            # planned_test_cmd 可能是合成阶段遗留的错误 pytest 命令，
            # 提取其中的文件路径，转换为 bin/test 命令
            if cmd.startswith("pytest "):
                logger.warning(f"sympy: planned_test_cmd 使用了 pytest (容器中不可用), 转换为 bin/test: {cmd[:80]}")
                parts = cmd.split()
                test_files = [p for p in parts[1:] if p.endswith('.py') and '/' in p]
                if test_files:
                    return f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(test_files)}"

        # 策略2: 使用planned_test_modules
        if planned_test_modules and isinstance(planned_test_modules, list):
            modules = [m for m in planned_test_modules if m and isinstance(m, str)]
            if modules:
                modules = modules[:max_total_labels]
                return f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(modules)}"

        # 策略3: 从FAIL_TO_PASS和PASS_TO_PASS提取文件
        files: list[str] = []
        test_names_without_path: list[str] = []

        for label_list in [fail_to_pass, pass_to_pass]:
            if not label_list:
                continue
            for label in label_list:
                if not isinstance(label, str) or not label.strip():
                    continue
                label = label.strip()

                # 检查是否有文件路径
                if "::" in label:
                    # 标准pytest格式: file.py::class::method
                    parts = label.split("::")
                    if parts[0] and parts[0] not in files:
                        files.append(parts[0])
                elif "/" in label or label.endswith(".py"):
                    # 文件路径
                    if label not in files:
                        files.append(label)
                else:
                    # 只有测试名，需要推断路径
                    test_names_without_path.append(label)

        # 策略4: 推断缺少路径的测试名
        if test_names_without_path and target_node:
            # 从target_node推断测试文件
            if ":" in target_node:
                source_file = target_node.split(":")[0]
                inferred_tests = self.infer_test_modules(source_file)
                for test_file in inferred_tests:
                    if test_file not in files:
                        files.append(test_file)
                        logger.info(f"sympy: 从target_node推断测试文件: {test_file}")

        # 策略5: 从modified_files推断
        if not files and modified_files:
            files = self.map_files_to_test_modules(modified_files)

        # 限制文件数量
        files = files[:max_total_labels]

        if files:
            return f"PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose {' '.join(files)}"

        # 最终回退
        logger.warning("sympy: 无法确定测试文件，使用通用回退")
        return "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose sympy/"
