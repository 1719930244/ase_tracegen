"""
Adapter module to bridge TraceGen SynthesisResult with the Validation module.

Supports automatic Docker image pulling from DockerHub (swebench organization).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from src.core.structures import SynthesisResult
from src.modules.validation.constants import KEY_INSTANCE_ID, KEY_PATCH, KEY_SEED_FIX_PATCH, KEY_INJECTION_PATCH, ValidationStatus
from src.modules.validation.docker_utils import image_exists, pull_image, get_container_client


# SWE-bench DockerHub organization
SWEBENCH_DOCKERHUB_ORG = "swebench"

# Image name templates
SWEBENCH_IMAGE_TEMPLATE = "{org}/sweb.eval.x86_64.{instance_id}"
SWESMITH_IMAGE_TEMPLATE = "{org}/swesmith.x86_64.{repo_name}.{commit}"


class ValidationAdapter:
    """
    Adapter to convert TraceGen SynthesisResult to validation instance format.

    Supports automatic image pulling from DockerHub.
    """

    def __init__(self, config: Dict[str, Any], repo_profile=None):
        """
        Initialize adapter with validation configuration.

        Args:
            config: Validation configuration dictionary (from config.yaml)
            repo_profile: RepoProfile instance (optional, for repo-aware test command generation)
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.default_image = config.get("default_image", "python:3.10-slim")
        self.timeout = config.get("timeout", 300)
        self.auto_pull = config.get("auto_pull", True)  # 自动拉取镜像
        self.dockerhub_org = config.get("dockerhub_org", SWEBENCH_DOCKERHUB_ORG)
        self.repo_profile = repo_profile

        # 缓存已检查的镜像
        self._image_cache: Dict[str, bool] = {}

    def adapt(self, result: SynthesisResult) -> Dict[str, Any]:
        """
        Convert a SynthesisResult to validation instance format.

        验证流程（SWE-smith 风格）：
        1. pre-gold: 在 fixed 基线上跑测试（先应用 seed_fix_patch）
        2. post-gold: 应用 injection_patch（fixed→buggy），再跑测试
        3. 期望看到 PASS→FAIL（测试从通过变为失败）

        Args:
            result: The synthesis result to convert

        Returns:
            Instance dictionary for Validator
        """
        # 确定并准备镜像（不再回退到默认镜像；缺失则标记跳过）
        image_name, image_available = self._determine_and_prepare_image(result)

        # 获取 seed_fix_patch（从 seed_metadata 中获取）
        # seed_fix_patch 是 SWE-bench 的 gold patch，把 buggy base_commit 变成 fixed
        seed_fix_patch = ""
        seed_metadata = getattr(result, "seed_metadata", None)
        if not seed_metadata and getattr(result, "metadata", None):
            # 兼容 scripts/validate_only.py 的 SimpleSynthesisResult（seed_metadata 存在于 metadata 内）
            seed_metadata = result.metadata.get("seed_metadata", {})  # type: ignore[attr-defined]
        if isinstance(seed_metadata, dict):
            seed_fix_patch = seed_metadata.get("patch", "") or ""
        if seed_fix_patch:
            logger.debug(f"获取 seed_fix_patch: {len(seed_fix_patch)} bytes")

        # 获取 injection_patch（从 metadata 中获取）
        # injection_patch 是从 fixed 状态变成 buggy 状态的 patch
        injection_patch = result.metadata.get("injection_patch", "")
        if not injection_patch:
            # 回退到 result.patch 如果没有 injection_patch
            logger.warning(f"实例 {result.instance_id} 没有 injection_patch，使用 result.patch")
            injection_patch = result.patch

        # 获取 test_patch（如果有的话）
        # test_patch 用于添加合成的测试文件
        test_patch = getattr(result, 'test_patch', '') or ''

        # pre-gold 需要在 fixed 基线上跑“同一套测试”，因此 test_patch（若存在）应在 pre/post 都生效
        # 我们将 test_patch 合并到 seed_fix_patch（pre-gold 阶段使用），post-gold 则再额外叠加 injection_patch
        combined_seed_fix_patch = seed_fix_patch
        if test_patch:
            combined_seed_fix_patch = self._combine_patches(seed_fix_patch, test_patch)
            logger.info(
                f"合并 seed_fix_patch 和 test_patch: {len(seed_fix_patch)} + {len(test_patch)} = {len(combined_seed_fix_patch)} bytes"
            )

        instance_data = {
            KEY_INSTANCE_ID: result.instance_id,
            "repo": result.repo,
            # Loc-chain context for post-validation scoring (optional, no effect on execution).
            "target_node": result.metadata.get("target_node", ""),
            "proposed_chain": result.metadata.get("proposed_chain", []),
            "seed_loc_chain": getattr(result, "seed_extraction_chains", []) or [],
            # seed_fix_patch: 用于 pre-gold 阶段，把镜像从 buggy 变成 fixed
            KEY_SEED_FIX_PATCH: combined_seed_fix_patch,
            # injection_patch: 用于 post-gold 阶段，把 fixed 变成 buggy
            KEY_INJECTION_PATCH: injection_patch,
            # patch: 保持兼容性，指向 injection_patch
            KEY_PATCH: injection_patch,
            "image_name": image_name,
            "image_available": image_available,
            "validation_skip_reason": "missing_image" if not image_available else "",
            "test_cmd": self._get_test_command(result),
            # Include expected test results for validation comparison
            "FAIL_TO_PASS": getattr(result, 'FAIL_TO_PASS', []) or [],
            "PASS_TO_PASS": getattr(result, 'PASS_TO_PASS', []) or [],
        }

        return instance_data

    def _combine_patches(self, patch1: str, patch2: str) -> str:
        """
        合并两个 patch。

        Args:
            patch1: 第一个 patch (通常是 injection_patch)
            patch2: 第二个 patch (通常是 test_patch)

        Returns:
            合并后的 patch
        """
        if not patch1:
            return patch2
        if not patch2:
            return patch1

        # 确保两个 patch 都以换行结尾
        if not patch1.endswith('\n'):
            patch1 += '\n'
        if not patch2.endswith('\n'):
            patch2 += '\n'

        return patch1 + patch2

    def _determine_and_prepare_image(self, result: SynthesisResult) -> Tuple[Optional[str], bool]:
        """
        Determine the Docker image and ensure it's available.

        Will attempt to pull from DockerHub if not available locally.

        Args:
            result: The synthesis result

        Returns:
            (Image name to use, image_available)
        """
        # 1. 检查 metadata 中是否指定了镜像
        if result.metadata and "docker_image" in result.metadata:
            image = result.metadata["docker_image"]
            available = self._ensure_image_available(image)
            return image, available

        # 2. 尝试 SWE-bench 镜像格式 (基于 seed_id)
        seed_id = result.seed_id
        if seed_id:
            swebench_image = self._build_swebench_image_name(seed_id)
            if self._ensure_image_available(swebench_image):
                return swebench_image, True

        # 3. 尝试 SWE-smith 镜像格式 (基于 repo 和 commit)
        if result.repo:
            swesmith_image = self._build_swesmith_image_name(result)
            if swesmith_image and self._ensure_image_available(swesmith_image):
                return swesmith_image, True

        # 4. 不再回退到默认镜像：镜像缺失则跳过该实例（镜像已预构建）
        missing_candidate: Optional[str] = None
        if seed_id:
            missing_candidate = self._build_swebench_image_name(seed_id)
        elif result.repo:
            missing_candidate = self._build_swesmith_image_name(result)

        if missing_candidate:
            logger.warning(f"缺少专用镜像，跳过实例: {missing_candidate}")
            return missing_candidate, False

        logger.warning("无法确定实例需要的专用镜像，跳过实例")
        return None, False

    def _build_swebench_image_name(self, instance_id: str) -> str:
        """
        Build SWE-bench image name from instance_id.

        Format: swebench/sweb.eval.x86_64.<owner>_1776_<repo>-<issue>
        Example: swebench/sweb.eval.x86_64.django_1776_django-11099

        Note: Docker Hub uses '_1776_' instead of '__' due to Docker naming restrictions.
        """
        # 转换 instance_id 格式: "django__django-11099" -> "django_1776_django-11099"
        formatted_id = instance_id.replace("__", "_1776_")
        return SWEBENCH_IMAGE_TEMPLATE.format(
            org=self.dockerhub_org,
            instance_id=formatted_id
        )

    def _build_swesmith_image_name(self, result: SynthesisResult) -> Optional[str]:
        """
        Build SWE-smith image name from repo info.

        Format: swebench/swesmith.x86_64.<owner>_1776_<repo>.<commit[:8]>
        Example: swebench/swesmith.x86_64.django_1776_django.e50f6f40

        Note: SWE-smith uses '_1776_' instead of '__' due to Docker naming restrictions.
        """
        repo = result.repo
        if not repo or "/" not in repo:
            return None

        # 获取 commit (优先使用 base_commit 字段)
        commit = getattr(result, 'base_commit', None)
        if not commit and result.metadata:
            commit = result.metadata.get("base_commit", "")

        if not commit:
            return None

        commit = commit[:8]

        # 转换 repo 格式: "owner/repo" -> "owner_1776_repo"
        owner, repo_name = repo.split("/", 1)
        repo_formatted = f"{owner}_1776_{repo_name}"

        return SWESMITH_IMAGE_TEMPLATE.format(
            org=self.dockerhub_org,
            repo_name=repo_formatted,
            commit=commit
        )

    def _ensure_image_available(self, image_name: str) -> bool:
        """
        Ensure the Docker image is available locally.

        Will attempt to pull from DockerHub if auto_pull is enabled.

        Args:
            imaname: Name of the Docker image

        Returns:
            True if image is available, False otherwise
        """
        # 检查缓存
        if image_name in self._image_cache:
            return self._image_cache[image_name]

        # 检查本地是否存在
        if image_exists(image_name):
            logger.debug(f"镜像已存在: {image_name}")
            self._image_cache[image_name] = True
            return True

        # 尝试拉取
        if self.auto_pull:
            logger.info(f"正在拉取镜像: {image_name}")
            if pull_image(image_name):
                logger.info(f"镜像拉取成功: {image_name}")
                self._image_cache[image_name] = True
                return True
            else:
                logger.warning(f"镜像拉取失败: {image_name}")
                self._image_cache[image_name] = False
                return False

        logger.warning(f"镜像不存在且未启用自动拉取: {image_name}")
        self._image_cache[image_name] = False
        return False

    def _get_test_command(self, result) -> str:
        """
        Construct test command based on patch analysis.

        Strategy (SWE-smith style - prioritize existing tests):
        1. Use custom test_cmd from metadata if provided
        2. Use RepoProfile.build_validation_test_cmd if available (NEW)
        3. Use agent-selected target tests if provided (preferred)
        4. Parse injection_patch to find modified files -> map to test modules
        5. Infer from target_node
        6. Fallback to minimal relevant tests

        NOTE: Synthetic test file (test_patch) is now used as last resort,
        since existing tests are more reliable for validation.
        """
        # 检查 metadata 中是否有自定义测试命令
        if result.metadata and "test_cmd" in result.metadata:
            return result.metadata["test_cmd"]

        test_selection_mode = (self.config.get("test_selection_mode", "") or "").strip().lower()
        minimize_test_suite = bool(self.config.get("minimize_test_suite", False))
        # Backwards compatibility:
        # - minimize_test_suite=true => fast mode
        # - otherwise => module_suite mode (tight, module-related suite)
        if not test_selection_mode:
            test_selection_mode = "fast" if minimize_test_suite else "module_suite"
        if test_selection_mode in {"min", "minimal"}:
            test_selection_mode = "fast"
        if test_selection_mode in {"suite", "module"}:
            test_selection_mode = "module_suite"
        max_fail_labels = int(self.config.get("max_fail_labels", 1) or 1)
        max_pass_labels = int(self.config.get("max_pass_labels", 1) or 1)
        max_total_labels = int(self.config.get("max_total_labels", 6) or 6)

        # 动态选择 RepoProfile: multi-repo 场景下按实例的 repo 字段切换
        active_profile = self.repo_profile
        instance_repo = getattr(result, 'repo', '') or ''
        if instance_repo:
            try:
                from src.core.repo_profiles import get_repo_profile
                active_profile = get_repo_profile(instance_repo)
            except Exception:
                pass

        # NEW: 尝试使用RepoProfile统一接口
        if active_profile is not None:
            try:
                fail_to_pass = getattr(result, 'FAIL_TO_PASS', None)
                pass_to_pass = getattr(result, 'PASS_TO_PASS', None)
                injection_patch = result.metadata.get("injection_patch", "") or result.patch
                modified_files = self._parse_patch_for_files(injection_patch)
                target_node = result.metadata.get("target_node", "")
                planned_test_cmd = (result.metadata.get("planned_test_cmd", "") or "").strip()
                planned_test_modules = result.metadata.get("planned_test_modules", []) or []

                test_cmd = active_profile.build_validation_test_cmd(
                    fail_to_pass=fail_to_pass,
                    pass_to_pass=pass_to_pass,
                    test_selection_mode=test_selection_mode,
                    max_fail_labels=max_fail_labels,
                    max_pass_labels=max_pass_labels,
                    max_total_labels=max_total_labels,
                    modified_files=modified_files,
                    target_node=target_node,
                    planned_test_cmd=planned_test_cmd,
                    planned_test_modules=planned_test_modules,
                )
                if test_cmd:
                    logger.debug(f"使用RepoProfile构建测试命令: {test_cmd[:100]}")
                    return test_cmd
            except Exception as e:
                logger.warning(f"RepoProfile.build_validation_test_cmd失败: {e}，回退到传统逻辑")

        # 检测是否是 Django 项目（优先使用 active_profile）
        is_django = False
        if active_profile is not None:
            is_django = active_profile.is_django
        else:
            seed_id = getattr(result, 'seed_id', '') or result.metadata.get('seed_id', '')
            if result.repo and "django" in result.repo.lower():
                is_django = True
            if seed_id and "django" in seed_id.lower():
                is_django = True

        # Prefer the synthesis-time planned suite when running in module_suite mode.
        # This prevents "generic inference" from drifting to unrelated tests.
        if test_selection_mode == "module_suite" and result.metadata:
            planned_cmd = (result.metadata.get("planned_test_cmd", "") or "").strip()
            planned_modules = result.metadata.get("planned_test_modules", []) or []
            if planned_cmd:
                # Guard: reject runtests.py commands for non-Django repos (synthesis agent may hallucinate).
                if not is_django and "runtests.py" in planned_cmd:
                    logger.warning(f"planned_test_cmd contains runtests.py for non-Django repo ({result.repo}); ignoring")
                # Guard: reject pytest commands for repos that don't have pytest (e.g., sympy uses bin/test).
                elif active_profile is not None and active_profile.test_framework not in ("pytest",) and "pytest" in planned_cmd:
                    logger.warning(
                        f"planned_test_cmd uses pytest but repo {result.repo} uses {active_profile.test_framework}; "
                        f"delegating to profile"
                    )
                else:
                    return planned_cmd
            if planned_modules and isinstance(planned_modules, list):
                planned_modules = [m for m in planned_modules if m and isinstance(m, str)]
                planned_modules = planned_modules[:max_total_labels]
                if planned_modules:
                    if is_django:
                        # Avoid known-aborting suites in SWE-bench images (e.g., GIS backends).
                        disallowed_prefixes = ("gis_tests",)
                        planned_modules = [
                            m for m in planned_modules
                            if not any(m == p or m.startswith(p + ".") for p in disallowed_prefixes)
                        ]
                        if not planned_modules:
                            logger.warning("planned_test_modules were all filtered (disallowed suites); falling back to inference")
                        else:
                            return f"./tests/runtests.py {' '.join(planned_modules)} --verbosity=2"
                    elif active_profile is not None and active_profile.test_framework not in ("pytest",):
                        # Non-pytest repo (e.g., sympy): use profile's build_test_cmd
                        return active_profile.build_test_cmd(planned_modules)
                    else:
                        return f"pytest {' '.join(planned_modules)} -v --tb=short"

        # Extract modified files early so Strategy 1 (agent-selected tests) can
        # be unioned with file-based inference for better coverage.
        injection_patch = result.metadata.get("injection_patch", "") or result.patch
        modified_files = self._parse_patch_for_files(injection_patch)

        # 策略 1: 使用 Agent 指定的 expected_tests_to_fail（如果有有效的非占位符测试）
        target_tests = getattr(result, 'FAIL_TO_PASS', None)
        pass_to_pass_tests = getattr(result, 'PASS_TO_PASS', None) or []
        placeholder_tests = {'test_reproduce_bug', 'test_synthetic', 'test_bug', 'test_method', 'test_view'}

        def _looks_like_django_label(label: str) -> bool:
            if not label or not isinstance(label, str):
                return False
            s = label.strip()
            if not s:
                return False
            if " " in s or "\t" in s or "\n" in s:
                return False
            if "/" in s:
                return False
            # runtests.py labels are dotted modules/classes/tests.
            if "." not in s:
                return False
            # Avoid obvious non-labels.
            if s.startswith("Traceback") or s.startswith("ERROR") or s.startswith("FAIL"):
                return False
            return True

        def _is_bad_django_test_label(label: str) -> bool:
            """
            Filter out labels that are frequently not directly runnable via unittest loader.
            Example: cache.tests.BaseCacheTests.test_x (Base* classes often aren't real TestCase subclasses).
            """
            if not label or not isinstance(label, str):
                return True
            s = label.strip()
            if not s:
                return True
            parts = s.split(".")
            # Heuristic: if it contains a class component starting with Base, disallow in fast mode.
            for part in parts:
                if part.startswith("Base") and (part.endswith("Test") or part.endswith("Tests") or "Test" in part):
                    return True
            return False

        def _prune_django_modules(mods: list[str]) -> list[str]:
            """Prefer more specific labels: drop directory-level prefixes like `utils_tests` if `utils_tests.test_text` exists."""
            cleaned = []
            for m in mods:
                if not m:
                    continue
                t = m.strip()
                if t:
                    cleaned.append(t)
            cleaned = list(dict.fromkeys(cleaned))
            if not cleaned:
                return []
            has_dotted = any("." in m for m in cleaned)
            if has_dotted:
                cleaned = [m for m in cleaned if "." in m]
            # Drop prefixes that are strict prefixes of other labels.
            pruned: list[str] = []
            for m in cleaned:
                is_prefix = False
                for n in cleaned:
                    if n != m and n.startswith(m + "."):
                        is_prefix = True
                        break
                if not is_prefix:
                    pruned.append(m)
            return pruned

        if target_tests:
            valid_tests = []
            for t in target_tests:
                if not t or not isinstance(t, str):
                    continue
                t = t.strip()
                if not t or t in placeholder_tests:
                    continue
                valid_tests.append(t)
            if valid_tests:
                logger.info(f"策略 1: 使用 Agent 指定的测试: {valid_tests}")
                if is_django:
                    django_fail_labels = [t for t in self._convert_to_django_test_format(valid_tests) if _looks_like_django_label(t)]
                    # Optional: include a tiny set of PASS_TO_PASS labels to keep validation fast,
                    # while still satisfying PASS->PASS requirement.
                    django_pass_labels: list[str] = []
                    if pass_to_pass_tests:
                        p2p_valid = []
                        for t in pass_to_pass_tests:
                            if not t or not isinstance(t, str):
                                continue
                            t = t.strip()
                            if not t or t in placeholder_tests:
                                continue
                            p2p_valid.append(t)
                        django_pass_labels = [t for t in self._convert_to_django_test_format(p2p_valid) if _looks_like_django_label(t)]

                    if test_selection_mode == "fast" and django_fail_labels and django_pass_labels:
                        django_fail_labels = [t for t in django_fail_labels if not _is_bad_django_test_label(t)]
                        django_pass_labels = [t for t in django_pass_labels if not _is_bad_django_test_label(t)]
                        if not django_fail_labels:
                            logger.warning("fast 模式下 FAIL labels 被过滤为空（Base* 等不可运行），回退到 module_suite")
                            test_selection_mode = "module_suite"
                        else:
                            selected: list[str] = []
                            for t in django_fail_labels[:max_fail_labels]:
                                if t not in selected:
                                    selected.append(t)
                            for t in django_pass_labels:
                                if len(selected) >= max_total_labels:
                                    break
                                if t not in selected:
                                    selected.append(t)
                                if len([x for x in selected if x in django_pass_labels]) >= max_pass_labels:
                                    break
                            if len(selected) < max_total_labels:
                                # top-up with more fail labels if any left
                                for t in django_fail_labels:
                                    if len(selected) >= max_total_labels:
                                        break
                                    if t not in selected:
                                        selected.append(t)
                            logger.info(f"最小化 Django 测试套件: {selected}")
                            return f"./tests/runtests.py {' '.join(selected)} --verbosity=2"

                    django_tests = django_fail_labels
                    # Prefer running at the module level to keep at least some passing tests
                    # for robustness (avoid all-fail / over-targeting).
                    django_modules = sorted({self._collapse_django_label_to_module(t) for t in django_tests if t})
                    # Django runtests.py does NOT accept file paths (e.g., tests/foo/test_bar.py).
                    # Normalize any leftover path-style labels into dotted module names.
                    normalized_modules: list[str] = []
                    for m in django_modules:
                        if not m:
                            continue
                        m = m.strip()
                        if not m:
                            continue
                        if m.startswith("tests/") or m.startswith("./tests/"):
                            m = self._convert_path_to_django_module(m.lstrip("./"))
                        elif "/" in m and m.endswith(".py"):
                            m = self._convert_path_to_django_module(m)
                        elif m.startswith("tests.") and len(m) > 6:
                            m = m[6:]
                        if m.endswith(".py"):
                            m = m[:-3]
                        if "/" in m:
                            m = m.replace("/", ".")
                        normalized_modules.append(m)
                    django_modules = sorted({m for m in normalized_modules if m and "/" not in m and not m.endswith(".py")})
                    # Union with file-based inference so we don't accidentally select
                    # a too-narrow test subset that misses the relevant failures.
                    inferred_modules: list[str] = []
                    if modified_files:
                        inferred_modules = self._map_files_to_test_modules(modified_files, True)
                    else:
                        inferred_modules = self._infer_test_from_target_node(result.metadata.get("target_node", ""), True)

                    # If PASS_TO_PASS is provided, treat it as "suite to keep passing" and run at module granularity.
                    p2p_modules: list[str] = []
                    if pass_to_pass_tests:
                        p2p_valid = []
                        for t in pass_to_pass_tests:
                            if not t or not isinstance(t, str):
                                continue
                            t = t.strip()
                            if not t or t in placeholder_tests:
                                continue
                            p2p_valid.append(t)
                        p2p_django = [t for t in self._convert_to_django_test_format(p2p_valid) if _looks_like_django_label(t)]
                        p2p_modules = [self._collapse_django_label_to_module(t) for t in p2p_django if t]

                    merged_modules = _prune_django_modules(sorted({m for m in (django_modules + p2p_modules + inferred_modules) if m}))
                    if merged_modules:
                        return f"./tests/runtests.py {' '.join(merged_modules)} --verbosity=2"
                    logger.warning("Django 测试标签归一化后为空，将回退到其他策略推断 test_cmd")
                else:
                    if test_selection_mode == "fast" and pass_to_pass_tests:
                        p2p = []
                        for t in pass_to_pass_tests:
                            if not t or not isinstance(t, str):
                                continue
                            t = t.strip()
                            if not t or t in placeholder_tests:
                                continue
                            p2p.append(t)
                        selected = []
                        for t in valid_tests[:max_fail_labels]:
                            if t not in selected:
                                selected.append(t)
                        for t in p2p:
                            if len(selected) >= max_total_labels:
                                break
                            if t not in selected:
                                selected.append(t)
                            if len([x for x in selected if x in p2p]) >= max_pass_labels:
                                break
                        if selected:
                            logger.info(f"最小化 pytest 测试套件: {selected}")
                            return f"pytest {' '.join(selected)} -v --tb=short"
                    return f"pytest {' '.join(valid_tests)} -v --tb=short"

        # 策略 2: 从 injection_patch 解析修改的文件 -> 推断测试模块
        if modified_files:
            test_modules = self._map_files_to_test_modules(modified_files, is_django)
            if test_modules:
                logger.info(f"策略 2: 从 patch 推断测试模块: {test_modules}")
                if is_django:
                    return f"./tests/runtests.py {' '.join(test_modules)} --verbosity=2"
                else:
                    return f"pytest {' '.join(test_modules)} -v"

        # 策略 3: 从 target_node 推断
        target_node = result.metadata.get("target_node", "")
        if target_node:
            test_modules = self._infer_test_from_target_node(target_node, is_django)
            if test_modules:
                logger.info(f"策略 3: 从 target_node 推断测试模块: {test_modules}")
                if is_django:
                    return f"./tests/runtests.py {' '.join(test_modules)} --verbosity=2"
                else:
                    return f"pytest {' '.join(test_modules)} -v"

        # 策略 4: 最小回退 - 运行装饰器相关测试（因为大部分合成是 Type_Cast_Fix）
        logger.warning("无法确定相关测试，使用最小回退测试")
        if is_django:
            # 基于 seed_id 推断测试模块
            if "14787" in seed_id:
                return "./tests/runtests.py decorators --verbosity=2"
            elif "11099" in seed_id:
                return "./tests/runtests.py auth_tests.test_validators --verbosity=2"
            # 通用回退：运行装饰器测试
            return "./tests/runtests.py decorators utils_tests.test_decorators --verbosity=2"
        return "pytest tests/ -v --tb=short -x"

    def _collapse_django_label_to_module(self, label: str) -> str:
        """
        Collapse a Django test label down to a module label.

        Examples:
            utils_tests.test_decorators.DecoratorTests.test_x -> utils_tests.test_decorators
            decorators.tests.DecoratorTests.test_x -> decorators.tests
            utils_tests.test_decorators -> utils_tests.test_decorators
        """
        if not label:
            return ""
        parts = label.split(".")
        # Only strip method/class components; keep module names like `foo.test_bar`.
        if len(parts) >= 3 and parts[-1].startswith("test"):
            parts = parts[:-1]
        if len(parts) >= 3 and (parts[-1].startswith("Test") or parts[-1][:1].isupper()):
            parts = parts[:-1]
        return ".".join(parts)

    def _convert_path_to_django_module(self, path: str) -> str:
        """
        Convert a test file path to Django runtests.py module format.

        Examples:
            tests/utils_tests/test_synthetic.py -> utils_tests.test_synthetic
            tests/decorators/test_synthetic.py -> decorators.test_synthetic
        """
        if not path:
            return ""

        # 移除 tests/ 前缀和 .py 后缀
        module = path
        if module.startswith("tests/"):
            module = module[6:]
        if module.endswith(".py"):
            module = module[:-3]

        # 将路径分隔符转换为点
        module = module.replace("/", ".")

        return module

    def _convert_to_django_test_format(self, tests: List[str]) -> List[str]:
        """Convert pytest-style test names to Django runtests.py format."""
        django_tests = []
        for t in tests:
            if "::" in t:
                # tests/decorators/tests.py::TestClass::test_method -> decorators.tests.TestClass.test_method
                parts = t.replace("::", ".").replace("/", ".")
                if parts.startswith("tests."):
                    parts = parts[6:]
                if ".py." in parts:
                    parts = parts.replace(".py.", ".")
                django_tests.append(parts)
            else:
                # Handle file-path style labels: tests/utils_tests/test_decorators.py -> utils_tests.test_decorators
                if (t.startswith("tests/") or t.startswith("./tests/")) and t.endswith(".py"):
                    django_tests.append(self._convert_path_to_django_module(t.lstrip("./")))
                elif "/" in t and t.endswith(".py"):
                    django_tests.append(self._convert_path_to_django_module(t))
                else:
                    django_tests.append(t)
        return django_tests

    def _parse_patch_for_files(self, patch: str) -> List[str]:
        """Parse a unified diff patch to extract modified file paths."""
        modified_files = []
        for line in patch.split('\n'):
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                file_path = line[6:].strip()
                if file_path and file_path != '/dev/null':
                    modified_files.append(file_path)
        return list(set(modified_files))

    def _map_files_to_test_modules(self, files: List[str], is_django: bool) -> List[str]:
        """Map source files to their corresponding test modules."""
        # 优先使用 repo_profile
        if self.repo_profile is not None:
            return self.repo_profile.map_files_to_test_modules(files)

        test_modules = set()

        # Legacy: Django 源文件到测试模块的映射
        django_mapping = {
            'django/utils/text.py': ['utils_tests.test_text'],
            'django/utils/decorators.py': ['decorators', 'utils_tests.test_decorators'],
            'django/core/handlers/': ['handlers'],
            'django/contrib/sites/': ['sites_tests', 'sites_framework'],
            'django/views/decorators/': ['decorators'],
            'django/contrib/auth/': ['auth_tests'],
            'django/contrib/admin/': ['admin_views', 'admin_checks', 'admin_utils', 'modeladmin'],
            'django/forms/': ['forms_tests', 'model_forms', 'model_formsets'],
            'django/db/models/': ['queries', 'model_fields', 'expressions'],
            'django/db/': ['db_functions', 'db_utils', 'backends'],
            'django/template/': ['template_tests', 'templates'],
            'django/utils/': ['utils_tests'],
            'django/core/': ['cache', 'mail', 'validators'],
            'django/views/': ['view_tests', 'generic_views'],
            'django/http/': ['httpwrappers', 'requests', 'responses'],
            'django/middleware/': ['middleware'],
            'django/urls/': ['urlpatterns', 'urlpatterns_reverse'],
        }

        for file_path in files:
            if is_django:
                for pattern, modules in django_mapping.items():
                    if pattern in file_path:
                        test_modules.update(modules)
                        break
            else:
                if 'test' not in file_path:
                    base_name = file_path.split('/')[-1].replace('.py', '')
                    test_modules.add(f"tests/test_{base_name}.py")
                else:
                    test_modules.add(file_path)

        return list(test_modules)

    def _infer_test_from_target_node(self, target_node: str, is_django: bool) -> List[str]:
        """Infer test module from target node path."""
        if ':' in target_node:
            file_path = target_node.split(':')[0]
            return self._map_files_to_test_modules([file_path], is_django)
        return []

    def get_available_images(self) -> List[str]:
        """
        List all locally available Docker images.

        Returns:
            List of image names
        """
        from src.modules.validation.docker_utils import list_images
        return list_images()

    def check_image_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Check the status of images for a given instance.

        Args:
            instance_id: The SWE-bench instance ID

        Returns:
            Dictionary with image availability status
        """
        swebench_image = self._build_swebench_image_name(instance_id)

        return {
            "instance_id": instance_id,
            "swebench_image": swebench_image,
            "swebench_available": image_exists(swebench_image),
            "default_image": self.default_image,
            "default_available": image_exists(self.default_image),
        }
