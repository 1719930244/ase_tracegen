"""
DjangoProfile — django/django 仓库的 RepoProfile 实现。

所有 Django 特有逻辑从 agent.py / adapter.py / validator.py / stage4.py 中提取到此处。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.repo_profile import RepoProfile


# Django 源码目录 → 测试模块的静态映射
# 保持与原 agent.py:435-452 和 adapter.py:661-680 一致
DJANGO_SOURCE_TO_TEST_MAPPING = {
    "django/utils/decorators.py": ["decorators", "utils_tests.test_decorators"],
    "django/utils/text.py": ["utils_tests.test_text"],
    "django/core/handlers/": ["handlers"],
    "django/contrib/sites/": ["sites_tests", "sites_framework"],
    "django/views/decorators/": ["decorators"],
    "django/contrib/auth/": ["auth_tests"],
    "django/contrib/admin/": [
        "admin_views", "admin_checks", "admin_utils", "modeladmin",
    ],
    "django/forms/": ["forms_tests", "model_forms", "model_formsets"],
    "django/db/models/": ["queries", "model_fields", "expressions"],
    "django/db/": ["db_functions", "db_utils", "backends"],
    "django/template/": ["template_tests", "templates"],
    "django/utils/": ["utils_tests"],
    "django/core/": ["cache", "mail", "validators"],
    "django/views/": ["view_tests", "generic_views"],
    "django/http/": ["httpwrappers", "requests", "responses"],
    "django/middleware/": ["middleware"],
    "django/urls/": ["urlpatterns", "urlpatterns_reverse"],
}

# Django 特有的崩溃指示器
DJANGO_CRASH_INDICATORS = [
    ("django.setup()", "django_setup_crash"),
    ("apps.populate(", "apps_populate_crash"),
    ("SystemCheckError:", "system_check_error"),
    ("System check identified", "system_check_error"),
]

# 已知在 SWE-bench Django 镜像中会 abort 的测试前缀
DJANGO_DISALLOWED_PREFIXES = ["gis_tests"]


@dataclass
class DjangoProfile(RepoProfile):
    """django/django 仓库的完整 Profile。"""

    repo: str = "django/django"
    test_framework: str = "unittest"
    source_to_test_mapping: dict[str, list[str]] = field(
        default_factory=lambda: dict(DJANGO_SOURCE_TO_TEST_MAPPING)
    )
    test_root: str = "tests"
    test_cmd_template: str = "./tests/runtests.py {targets} --verbosity=2"
    disallowed_test_prefixes: list[str] = field(
        default_factory=lambda: list(DJANGO_DISALLOWED_PREFIXES)
    )
    extra_crash_indicators: list[tuple[str, str]] = field(
        default_factory=lambda: list(DJANGO_CRASH_INDICATORS)
    )
    test_base_class: str = "TestCase"
    test_imports: list[str] = field(
        default_factory=lambda: ["from django.test import TestCase"]
    )

    # ── 覆盖方法 ──

    def source_path_to_test_label(self, test_file: str) -> str:
        """tests/utils_tests/test_text.py → utils_tests.test_text"""
        rel = test_file
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("tests/"):
            rel = rel[len("tests/"):]
        # 去掉 .py 后缀，替换 / 为 .
        label = rel.replace(".py", "").replace("/", ".")
        # 去掉尾部的 . (如果有)
        return label.rstrip(".")

    def get_test_file_path(self, source_file: str) -> str:
        """根据 Django 源文件路径确定合成测试文件位置。"""
        if "utils" in source_file:
            return "tests/utils_tests/test_synthetic.py"
        elif "decorators" in source_file:
            return "tests/decorators/test_synthetic.py"
        elif "views" in source_file:
            return "tests/view_tests/test_synthetic.py"
        elif "db" in source_file:
            return "tests/db_tests/test_synthetic.py"
        return "tests/test_synthetic.py"

    def generate_test_class(
        self,
        test_class_name: str,
        test_method_name: str,
        imports: list[str],
        test_body: str,
        target_node: str,
        expected_failure_reason: str,
        helper_code: str = "",
    ) -> str:
        """生成 Django TestCase 风格的测试类。"""
        # 确保 TestCase 导入
        all_imports = list(imports)
        if not any("from django.test import TestCase" in i for i in all_imports):
            all_imports.append("from django.test import TestCase")
        imports_str = "\n".join(sorted(set(all_imports)))

        # 缩进测试体（8 空格 = class 内 method 内）
        indented_lines = []
        for line in test_body.split("\n"):
            if line.strip():
                stripped = line.lstrip()
                orig_indent = len(line) - len(stripped)
                indented_lines.append("        " + " " * orig_indent + stripped)
            else:
                indented_lines.append("")
        indented_body = "\n".join(indented_lines)

        if not indented_body.strip() or indented_body.strip() == "pass":
            func_name = target_node.split(":")[-1] if ":" in target_node else "target"
            indented_body = (
                f'        # Synthetic test for {func_name}\n'
                f'        self.assertTrue(True, "Placeholder assertion")'
            )

        helper_section = ""
        if helper_code and helper_code.strip():
            helper_section = f"\n\n# Helper classes and functions for testing\n{helper_code}\n"

        return f'''"""
Synthetic test case generated by TraceGen.
Target: {target_node}
Expected failure: {expected_failure_reason}
"""
{imports_str}{helper_section}


class {test_class_name}(TestCase):
    """
    Synthetic test to verify bug injection.
    This test should FAIL when the bug is present.
    """

    def {test_method_name}(self):
        """
        Test that verifies the synthetic bug.
        Expected: {expected_failure_reason or "Test should fail due to injected bug"}
        """
{indented_body}
'''

    def get_validation_profile(self, image_name: str):
        """Django 使用 UnittestProfile。"""
        from src.modules.validation.profiles.python import UnittestProfile
        return UnittestProfile(image_name=image_name)

    def infer_seed_test_cmd(self, seed_info: dict, max_total_labels: int = 6) -> str:
        """从 Django seed 元数据推断测试命令（unittest label 格式）。"""
        if not isinstance(seed_info, dict):
            return ""

        def _parse_json_list(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(x) for x in value if x is not None]
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return []
                try:
                    import json
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed if x is not None]
                except Exception:
                    return [s]
                return []
            return [str(value)]

        f2p = _parse_json_list(seed_info.get("FAIL_TO_PASS"))
        p2p = _parse_json_list(seed_info.get("PASS_TO_PASS"))
        labels = [x.strip() for x in (f2p + p2p) if isinstance(x, str) and x.strip()]

        # 从 unittest label 中提取模块：
        # "test_name (pkg.module.Class)" → "pkg.module"
        modules: list[str] = []
        for s in labels:
            m = re.search(r"\(([^)]+)\)", s)
            if not m:
                continue
            inside = m.group(1).strip()
            if "." not in inside:
                continue
            mod = inside.rsplit(".", 1)[0].strip()
            if mod.startswith("tests.") and len(mod) > 6:
                mod = mod[6:]
            if mod and mod not in modules:
                modules.append(mod)

        # 过滤黑名单
        modules = [m for m in modules if not self.is_disallowed_label(m)]
        modules = modules[:max_total_labels]

        if not modules:
            return ""
        return f"./tests/runtests.py {' '.join(modules)} --verbosity=2"

    def plan_test_suite(self, file_path: str, repo_root: Path) -> dict[str, Any]:
        """Django 特化的测试套件规划。"""
        planned: dict[str, Any] = {
            "is_django": True,
            "test_modules": [],
            "test_files": [],
            "test_cmd": "",
        }

        # 静态映射
        modules: list[str] = []
        for pattern, mods in self.source_to_test_mapping.items():
            if pattern in file_path:
                modules = list(mods)
                break

        # 动态推断：在 tests/ 下搜索引用了目标文件的测试
        inferred_test_files = self._infer_related_test_files(file_path, repo_root)

        if inferred_test_files:
            inferred_labels = [self.source_path_to_test_label(f) for f in inferred_test_files]
            merged: list[str] = []
            for m in (modules or []):
                if m and m not in merged:
                    merged.append(m)
            for m in inferred_labels:
                if len(merged) >= 6:
                    break
                if m and m not in merged:
                    merged.append(m)
            planned["test_modules"] = [m for m in merged if not self.is_disallowed_label(m)][:6]
        elif modules:
            planned["test_modules"] = [m for m in modules if not self.is_disallowed_label(m)]

        if planned["test_modules"]:
            planned["test_cmd"] = self.build_test_cmd(planned["test_modules"])

        # 展开为具体文件
        test_files: set[str] = set()
        if inferred_test_files:
            test_files.update(inferred_test_files)
        for m in planned.get("test_modules", []):
            test_files.update(self._expand_module_to_files(m, repo_root))
        planned["test_files"] = sorted(test_files)

        return planned

    def _infer_related_test_files(
        self, file_path: str, repo_root: Path, max_files: int = 10
    ) -> list[str]:
        """在 tests/ 下搜索引用了目标源文件的测试文件。"""
        tests_dir = repo_root / "tests"
        if not tests_dir.exists():
            return []

        p = Path(file_path)
        stem = p.stem
        mod = str(p.with_suffix("")).replace("/", ".")

        scored: list[tuple[float, str]] = []
        for tf in tests_dir.rglob("test_*.py"):
            try:
                content = tf.read_text(errors="ignore")[:20000]
            except Exception:
                continue
            rel = str(tf.relative_to(repo_root))
            score = 0.0
            parent = str(p.parent).replace("/", ".")
            if parent and stem and f"from {parent} import {stem}" in content:
                score += 6.0
            if f"from {mod} import" in content or f"import {mod}" in content:
                score += 4.0
            if stem and re.search(rf"\b{re.escape(stem)}\b", content):
                score += 1.5
            if score > 0:
                scored.append((score, rel))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rel for _s, rel in scored[:max_files]]

    def _expand_module_to_files(self, module: str, repo_root: Path) -> set[str]:
        """将 Django 测试模块名展开为具体文件路径。"""
        files: set[str] = set()
        if not module:
            return files

        if "." in module:
            rel = f"tests/{module.replace('.', '/')}.py"
            if (repo_root / rel).exists():
                files.add(rel)
            return files

        dir_path = repo_root / "tests" / module
        if dir_path.exists() and dir_path.is_dir():
            for tf in list(dir_path.glob("*.py"))[:40]:
                files.add(str(tf.relative_to(repo_root)))
            return files

        rel = f"tests/{module}.py"
        if (repo_root / rel).exists():
            files.add(rel)
        return files

    # ── adapter.py 中提取的 Django label 处理方法 ──

    def looks_like_django_label(self, label: str) -> bool:
        """检查字符串是否像 Django 测试 label（非文件路径）。"""
        if not label or "/" in label or label.endswith(".py"):
            return False
        return bool(re.match(r"^[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*$", label))

    def collapse_label_to_module(self, label: str) -> str:
        """将 Django 测试 label 折叠到模块级别。
        auth_tests.test_models.UserModelTest.test_create → auth_tests
        """
        parts = label.split(".")
        return parts[0] if parts else label

    def convert_path_to_module(self, path: str) -> str:
        """将文件路径转换为 Django 测试模块名。
        tests/auth_tests/test_models.py → auth_tests.test_models
        """
        rel = path
        if rel.startswith("tests/"):
            rel = rel[len("tests/"):]
        return rel.replace(".py", "").replace("/", ".")

    def prune_modules(self, modules: list[str], max_total: int = 6) -> list[str]:
        """裁剪 Django 测试模块列表，去重并限制数量。"""
        seen: set[str] = set()
        pruned: list[str] = []
        for m in modules:
            collapsed = self.collapse_label_to_module(m)
            if collapsed not in seen and not self.is_disallowed_label(collapsed):
                seen.add(collapsed)
                pruned.append(collapsed)
        return pruned[:max_total]

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
        构建Django验证测试命令。

        Django特殊处理：
        1. 优先使用planned_test_cmd/planned_test_modules
        2. 从FAIL_TO_PASS/PASS_TO_PASS提取测试标签
        3. 转换pytest格式到Django runtests.py格式
        4. 折叠到模块级（module_suite模式）
        5. 过滤不可运行的测试（Base*类、GIS测试）
        """
        from loguru import logger

        # 策略1: 使用planned_test_cmd（仅接受 runtests.py 格式）
        if planned_test_cmd and planned_test_cmd.strip():
            cmd = planned_test_cmd.strip()
            if "runtests.py" in cmd:
                return cmd
            # pytest 格式（含 python -m pytest 等变体）: 提取目标并转换为 Django label
            if "pytest" in cmd:
                logger.debug(f"Django: 将 pytest planned_test_cmd 转换为 runtests.py: {cmd[:80]}")
                parts = cmd.split()
                converted = []
                for p in parts:
                    if p.startswith("-") or p in ("pytest", "python", "-m"):
                        continue
                    # 处理 nodeid: tests/xxx.py::Class::method → 取 :: 前的路径
                    file_part = p.split("::")[0] if "::" in p else p
                    if "/" in file_part or file_part.endswith(".py"):
                        label = file_part.replace("/", ".").replace(".py", "")
                        if label.startswith("tests."):
                            label = label[6:]
                        module = self.collapse_label_to_module(label)
                        if module and not self.is_disallowed_label(module) and module not in converted:
                            converted.append(module)
                if converted:
                    return f"./tests/runtests.py {' '.join(converted[:max_total_labels])} --verbosity=2"
            # 其他格式: 忽略，继续走后续策略
            logger.debug(f"Django: 忽略非 runtests.py 的 planned_test_cmd: {cmd[:80]}")

        # 策略2: 使用planned_test_modules（自动转换文件路径为 Django label）
        if planned_test_modules and isinstance(planned_test_modules, list):
            modules = [m for m in planned_test_modules if m and isinstance(m, str)]
            converted_modules = []
            for m in modules:
                was_converted = False
                if "/" in m or m.endswith(".py"):
                    # 文件路径格式，转换为 Django dotted label
                    label = m.replace("/", ".").replace(".py", "")
                    if label.startswith("tests."):
                        label = label[6:]
                    mod = self.collapse_label_to_module(label)
                    if mod:
                        m = mod
                        was_converted = True
                # 转换过的 label 已经过 collapse 验证；未转换的需要合法性检查
                if " " in m or "\t" in m or "\n" in m:
                    continue  # 拒绝含空白的非法 label
                if not self.is_disallowed_label(m) and m not in converted_modules:
                    converted_modules.append(m)
            if converted_modules:
                converted_modules = converted_modules[:max_total_labels]
                return f"./tests/runtests.py {' '.join(converted_modules)} --verbosity=2"

        # 策略3: 从FAIL_TO_PASS和PASS_TO_PASS提取
        django_modules: list[str] = []

        if fail_to_pass:
            for label in fail_to_pass:
                if not isinstance(label, str) or not label.strip():
                    continue
                # 转换pytest格式到Django格式
                django_label = self._convert_pytest_to_django_label(label)
                if django_label and self._looks_like_django_label(django_label):
                    # 折叠到模块级
                    module = self.collapse_label_to_module(django_label)
                    if module and module not in django_modules:
                        django_modules.append(module)

        if pass_to_pass and test_selection_mode == "module_suite":
            for label in pass_to_pass:
                if not isinstance(label, str) or not label.strip():
                    continue
                django_label = self._convert_pytest_to_django_label(label)
                if django_label and self._looks_like_django_label(django_label):
                    module = self.collapse_label_to_module(django_label)
                    if module and module not in django_modules:
                        django_modules.append(module)

        # 策略4: 从modified_files推断
        if not django_modules and modified_files:
            django_modules = self.map_files_to_test_modules(modified_files)

        # 策略5: 从target_node推断
        if not django_modules and target_node:
            if ":" in target_node:
                file_path = target_node.split(":")[0]
                django_modules = self.map_files_to_test_modules([file_path])

        # 过滤和限制
        django_modules = self.prune_modules(django_modules, max_total_labels)

        if django_modules:
            return f"./tests/runtests.py {' '.join(django_modules)} --verbosity=2"

        # 最终回退
        logger.warning("Django: 无法确定测试模块，使用默认回退")
        return "./tests/runtests.py decorators utils_tests --verbosity=2"

    def _convert_pytest_to_django_label(self, label: str) -> str:
        """
        转换pytest格式的测试标签到Django runtests.py格式。

        Examples:
            "tests/decorators/tests.py::DecoratorsTest::test_attributes"
            → "decorators.tests.DecoratorsTest.test_attributes"
        """
        if not label or not isinstance(label, str):
            return ""

        label = label.strip()

        # 如果已经是Django格式（点号分隔），直接返回
        if "::" not in label and "." in label and "/" not in label:
            return label

        # 转换pytest格式
        if "::" in label:
            # 替换 :: → .
            label = label.replace("::", ".")

        # 替换 / → .
        label = label.replace("/", ".")

        # 移除 tests/ 前缀
        if label.startswith("tests."):
            label = label[6:]

        # 移除 .py 后缀
        if ".py." in label:
            label = label.replace(".py.", ".")

        return label.strip(".")

    def _looks_like_django_label(self, label: str) -> bool:
        """检查是否看起来像Django测试标签。"""
        if not label or not isinstance(label, str):
            return False
        s = label.strip()
        if not s or " " in s or "\t" in s or "\n" in s:
            return False
        if "/" in s:
            return False
        if "." not in s:
            return False
        # 避免明显的非标签
        if s.startswith("Traceback") or s.startswith("ERROR") or s.startswith("FAIL"):
            return False
        return True
