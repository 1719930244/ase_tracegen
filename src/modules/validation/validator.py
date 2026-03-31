"""
Core Validator class for running Docker-based bug validation.

This module provides the main Validator class that orchestrates:
1. Docker container creation and management
2. Patch application
3. Test execution (pre and post patch)
4. Result grading and report generation
"""

import json
import os
import re
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.modules.validation.constants import (
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    FAIL_TO_FAIL,
    PASS_TO_FAIL,
    KEY_INSTANCE_ID,
    KEY_PATCH,
    KEY_SEED_FIX_PATCH,
    KEY_INJECTION_PATCH,
    KEY_TIMED_OUT,
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    LOG_TEST_OUTPUT_PRE_GOLD,
    LOG_INSTANCE,
    LOG_PATCH,
    LOG_EVAL_SH,
)
from src.modules.validation.docker_utils import (
    create_container,
    start_container,
    cleanup_container,
    exec_command,
    run_test_in_container,
    apply_patch,
    ExecResult,
)
from src.modules.validation.grading import (
    get_valid_report,
    read_test_output,
)


# =============================================================================
# Main Validator Class
# =============================================================================

class Validator:
    """
    Main validator for running Docker-based bug validation.

    Example:
        profile = PythonProfile(
            owner="mewwts",
            repo="addict",
            commit="75284f95",
            image_name="my-image",
            test_cmd="pytest tests/"
        )
        validator = Validator(profile=profile)
        result = validator.validate(instance)
    """

    def __init__(
        self,
        profile: Any = None,
        config: ValidationConfig = None,
        repo_profile: Any = None,
    ):
        """
        Initialize the validator.

        Args:
            profile: Repository profile (test config, parser, etc.)
            config: Validation configuration
            repo_profile: RepoProfile instance (optional, for repo-specific crash indicators)
        """
        self.profile = profile
        self.config = config or ValidationConfig()
        self.repo_profile = repo_profile
        self._log_dir = Path(self.config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self,
        instance: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a single bug instance.

        验证流程（SWE-smith 风格）：
        1. Create Docker container
        2. Apply seed_fix_patch to get fixed baseline
        3. Run tests on fixed baseline (pre-gold) - expect PASS
        4. Apply injection_patch (fixed→buggy)
        5. Run tests on buggy state (post-gold) - expect FAIL
        6. Compare results: look for PASS→FAIL

        Args:
            instance: Instance dict with patch and metadata

        Returns:
            ValidationResult with test comparison results
        """
        instance_id = instance.get(KEY_INSTANCE_ID, "unknown")
        repo = instance.get("repo", instance_id.split(".")[0])

        # Set up logging directory
        log_dir = self._log_dir / repo / instance_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize result
        result = ValidationResult(
            instance_id=instance_id,
            status=ValidationStatus.VALID,
        )

        # Set up logger (file-based)
        logger = self._setup_logger(log_dir / LOG_INSTANCE)

        try:
            # 镜像缺失：直接跳过（镜像已预构建，不再回退到默认镜像）
            image_name = instance.get("image_name")
            image_available = instance.get("image_available", True)
            if not image_name:
                logger.info("Missing image_name; skipping instance")
                result.status = ValidationStatus.MISSING_IMAGE
                result.error_message = "image_name is empty"
                with open(log_dir / LOG_REPORT, "w") as f:
                    json.dump(
                        {"status": result.status.value, "reason": "missing_image", "image_name": image_name},
                        f,
                        indent=2,
                    )
                return result

            if not image_available:
                logger.info(f"Image not available; skipping instance: {image_name}")
                result.status = ValidationStatus.MISSING_IMAGE
                result.error_message = str(image_name)
                with open(log_dir / LOG_REPORT, "w") as f:
                    json.dump(
                        {"status": result.status.value, "reason": "missing_image", "image_name": image_name},
                        f,
                        indent=2,
                    )
                return result

            # 获取 seed_fix_patch 和 injection_patch
            seed_fix_patch = instance.get(KEY_SEED_FIX_PATCH, "")
            injection_patch = instance.get(KEY_INJECTION_PATCH, "") or instance.get(KEY_PATCH, "")

            logger.info(f"seed_fix_patch: {len(seed_fix_patch)} bytes")
            logger.info(f"injection_patch: {len(injection_patch)} bytes")

            # Step 1: Pre-gold test run (on FIXED baseline)
            # 先应用 seed_fix_patch 把镜像从 buggy 变成 fixed，然后跑测试
            logger.info(f"Starting pre-gold test run for {instance_id}")
            logger.info("Pre-gold: Applying seed_fix_patch to get fixed baseline")
            pre_gold_output, pre_timed_out = self._run_test(
                instance=instance,
                patch=seed_fix_patch if seed_fix_patch else None,  # 应用 seed_fix_patch
                log_dir=log_dir,
                logger=logger,
            )

            if pre_timed_out:
                logger.info(f"Pre-gold test timed out for {instance_id}")
                result.status = ValidationStatus.TIMEOUT
                result.timed_out = True
                result.timeout_value = self.config.timeout
                return result

            # Save pre-gold output
            with open(log_dir / LOG_TEST_OUTPUT_PRE_GOLD, "w") as f:
                f.write(pre_gold_output)

            # Step 2: Post-gold test run (apply injection_patch on top of fixed baseline)
            # 在 fixed 基线上应用 injection_patch，把代码变成 buggy，然后跑测试
            logger.info(f"Starting post-gold test run for {instance_id}")
            logger.info("Post-gold: Applying seed_fix_patch then injection_patch")

            # NOTE:
            # - We intentionally apply patches sequentially (seed then injection).
            # - For TraceGen-generated patches (synthesized from a fixed baseline),
            #   injection_patch is expected to apply cleanly. If it doesn't, treat it
            #   as an invalid sample (do NOT silently accept partial rejects).
            post_gold_patches: list[tuple[str, bool]] = []
            if seed_fix_patch:
                post_gold_patches.append((seed_fix_patch, False))
            if injection_patch:
                post_gold_patches.append((injection_patch, False))

            post_gold_output, post_timed_out = self._run_test(
                instance=instance,
                patch=None,
                patches=post_gold_patches,
                log_dir=log_dir,
                logger=logger,
            )

            if post_timed_out:
                logger.info(f"Post-gold test timed out for {instance_id}")
                result.status = ValidationStatus.TIMEOUT
                result.timed_out = True
                result.timeout_value = self.config.timeout
                return result

            # Save post-gold output
            with open(log_dir / LOG_TEST_OUTPUT, "w") as f:
                f.write(post_gold_output)

            # Check for environment crash (setup failure)
            crash_status = self._detect_crash(pre_gold_output, post_gold_output)
            if crash_status:
                logger.info(f"Crash detected for {instance_id}: {crash_status['message']}")
                result.status = ValidationStatus.ERROR
                result.error_message = crash_status['message']
                # 保存崩溃信息到报告
                with open(log_dir / LOG_REPORT, "w") as f:
                    json.dump({
                        "status": "crash",
                        "crash_type": crash_status['type'],
                        "message": crash_status['message'],
                    }, f, indent=2)
                return result

            # Step 3: Generate report
            logger.info(f"Generating validation report for {instance_id}")

            # 动态选择 log_parser: multi-repo 场景下按实例的 repo 字段切换
            # 不依赖全局 self.profile（multi-repo 时可能不匹配当前实例）
            instance_repo = instance.get("repo", "")
            if instance_repo and "sympy" in instance_repo.lower():
                from src.modules.validation.grading import parse_sympy_log
                log_parser_fn = parse_sympy_log
            elif instance_repo and "django" in instance_repo.lower():
                from src.modules.validation.profiles.python import UnittestProfile
                log_parser_fn = UnittestProfile().log_parser
            else:
                # 默认 pytest 解析器（覆盖全局 profile 可能不匹配的情况）
                from src.modules.validation.profiles.python import PythonProfile
                log_parser_fn = PythonProfile().log_parser

            report = get_valid_report(
                pre_gold_output,
                post_gold_output,
                log_parser_fn,
            )

            # Update result with report data
            result.FAIL_TO_PASS = report.get(FAIL_TO_PASS, [])
            result.PASS_TO_PASS = report.get(PASS_TO_PASS, [])
            result.FAIL_TO_FAIL = report.get(FAIL_TO_FAIL, [])
            result.PASS_TO_FAIL = report.get(PASS_TO_FAIL, [])

            # Determine validation status
            result.status = self._determine_status(result)

            # P2: 提取失败测试的调用栈信息
            failed_tests = result.PASS_TO_FAIL + result.FAIL_TO_FAIL
            if failed_tests:
                result.traceback_info = self._extract_traceback(post_gold_output, failed_tests)
                if result.traceback_info:
                    logger.info(f"提取到 {len(result.traceback_info)} 个测试的调用栈信息")
            
            # P3: 链路对齐评分（基于失败 traceback 是否覆盖合成 loc-chain）
            try:
                result.chain_alignment_score = self._score_chain_alignment(
                    instance=instance,
                    traceback_info=result.traceback_info,
                )
                if (
                    self.config.mode == "injection"
                    and result.status == ValidationStatus.VALID
                ):
                    coverage = float(result.chain_alignment_score.get("trace_coverage", 0.0) or 0.0)
                    target_hit = bool(result.chain_alignment_score.get("target_node_hit", False))
                    would_reject = False
                    reject_reason = ""
                    if coverage < float(self.config.min_chain_coverage):
                        would_reject = True
                        reject_reason = f"Chain coverage too low: {coverage:.2f} < {self.config.min_chain_coverage:.2f}"
                    elif self.config.require_target_node_in_traceback and not target_hit:
                        would_reject = True
                        reject_reason = "Target node not observed in failing traceback"

                    if would_reject:
                        if self.config.enforce_chain_coverage:
                            result.status = ValidationStatus.INVALID
                            result.error_message = reject_reason
                        else:
                            # Shadow mode: log but don't reject
                            logger.warning(
                                f"[SHADOW] Would reject {instance.get('instance_id', '?')}: {reject_reason} "
                                f"(coverage={coverage:.2f}, target_hit={target_hit})"
                            )
            except Exception as e:
                logger.warning(f"Failed to compute chain alignment score: {e}")

            # Save report
            with open(log_dir / LOG_REPORT, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Validation complete: {result.summary()}")

        except Exception as e:
            logger.info(f"Error during validation: {e}")
            logger.info(traceback.format_exc())
            result.status = ValidationStatus.ERROR
            result.error_message = str(e)

        return result

    def validate_batch(
        self,
        instances: list[dict[str, Any]],
        workers: int = 1,
    ) -> dict[str, ValidationResult]:
        """
        Validate multiple instances in parallel.

        Args:
            instances: List of instance dicts
            workers: Number of parallel workers

        Returns:
            Dict mapping instance_id to ValidationResult
        """
        results = {}

        if workers <= 0:
            # Sequential execution
            for instance in instances:
                instance_id = instance[KEY_INSTANCE_ID]
                results[instance_id] = self.validate(instance)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.validate, instance): instance
                    for instance in instances
                }

                for future in as_completed(futures):
                    instance = futures[future]
                    try:
                        result = future.result()
                        results[result.instance_id] = result
                    except Exception as e:
                        instance_id = instance[KEY_INSTANCE_ID]
                        results[instance_id] = ValidationResult(
                            instance_id=instance_id,
                            status=ValidationStatus.ERROR,
                            error_message=str(e),
                        )

        return results

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _setup_logger(self, log_file: Path):
        """Set up a file-based logger."""
        import logging

        logger = logging.getLogger(f"validator.{id(self)}")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Avoid duplicate handlers, and prevent propagation to root (console formatting is handled by Loguru).
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False

        return logger

    def _run_test(
        self,
        instance: dict[str, Any],
        patch: Optional[str],
        log_dir: Path,
        logger: Any,
        patches: Optional[list[tuple[str, bool]]] = None,
    ) -> tuple[str, bool]:
        """
        Run tests in a container.

        Args:
            instance: Instance dict
            patch: Patch to apply (None for pre-gold)
            log_dir: Directory for logs
            logger: Logger instance

        Returns:
            tuple: (test_output, timed_out)
        """
        container = None
        try:
            # Get test command
            if self.profile:
                test_cmd, _ = self.profile.get_test_cmd(instance)
            else:
                # Use default test command
                test_cmd = instance.get("test_cmd", "pytest -v")

            # Create container
            image_name = instance.get("image_name")
            if not image_name:
                raise ValueError("Missing required instance.image_name")
            container = create_container(
                image_name=image_name,
                instance_id=instance[KEY_INSTANCE_ID],
                platform=self.config.platform,
                memory_limit=self.config.memory_limit,
            )
            start_container(container)
            logger.info(f"Container created and started: {container.name}")

            # Apply patches (supports either a single patch or multiple patches)
            patches_to_apply: list[tuple[str, bool]] = []
            if patches:
                patches_to_apply = patches
            elif patch:
                patches_to_apply = [(patch, False)]

            if patches_to_apply:
                # Write a combined patch to file for debugging (even if applied sequentially)
                combined_for_log = ""
                for content, _allow_rejects in patches_to_apply:
                    if not content:
                        continue
                    if combined_for_log and not combined_for_log.endswith("\n"):
                        combined_for_log += "\n"
                    combined_for_log += content
                    if not combined_for_log.endswith("\n"):
                        combined_for_log += "\n"

                patch_file = log_dir / LOG_PATCH
                with open(patch_file, "w") as f:
                    f.write(combined_for_log)

                for idx, (content, allow_rejects) in enumerate(patches_to_apply):
                    if not content or not content.strip():
                        continue
                    logger.info(f"Applying patch {idx+1}/{len(patches_to_apply)} (allow_rejects={allow_rejects})...")
                    apply_patch(container, content, allow_rejects=allow_rejects)

            # Run tests
            logger.info(f"Running test command: {test_cmd}")
            result = run_test_in_container(
                container=container,
                test_command=test_cmd,
                timeout=self.config.timeout,
            )

            logger.info(f"Test completed. Exit code: {result.exit_code}")
            if self.config.verbose:
                logger.info(f"Output preview: {result.output[:500]}...")

            return result.output, result.timed_out

        finally:
            # Clean up container
            if self.config.clean_containers:
                cleanup_container(container)

    def _detect_crash(self, pre_gold_output: str, post_gold_output: str) -> dict | None:
        """
        检测环境崩溃（区分逻辑错误和破坏性 bug）。

        如果 pre-gold 正常但 post-gold 崩溃，这是一个"破坏性 bug"，不算有效。

        Args:
            pre_gold_output: 未修改代码的测试输出
            post_gold_output: 注入 bug 后的测试输出

        Returns:
            如果检测到崩溃，返回包含类型和消息的字典；否则返回 None
        """
        # 通用崩溃指示器（适用于所有 Python 项目）
        crash_indicators = [
            ("ImportError:", "import_error"),
            ("ModuleNotFoundError:", "module_not_found"),
            ("SyntaxError:", "syntax_error"),
            ("IndentationError:", "indentation_error"),
            ("AttributeError: module", "module_attribute_error"),
            ("NameError:", "name_error"),
            ("TypeError: 'NoneType'", "none_type_error"),
        ]

        # 追加仓库特有的崩溃指示器（如 Django 的 django.setup() 等）
        if self.repo_profile is not None and hasattr(self.repo_profile, 'extra_crash_indicators'):
            crash_indicators.extend(self.repo_profile.extra_crash_indicators)

        # 检查 pre-gold 是否正常（不应该有崩溃）
        pre_has_crash = False
        for indicator, _ in crash_indicators:
            if indicator in pre_gold_output:
                pre_has_crash = True
                break

        # 检查 post-gold 是否崩溃
        post_crash_type = None
        for indicator, crash_type in crash_indicators:
            if indicator in post_gold_output:
                post_crash_type = crash_type
                break

        # 如果 pre-gold 正常但 post-gold 崩溃，这是破坏性 bug
        if not pre_has_crash and post_crash_type:
            return {
                "type": post_crash_type,
                "message": f"Bug causes environment crash ({post_crash_type}), too destructive",
            }

        return None

    def _determine_status(self, result: ValidationResult) -> ValidationStatus:
        """
        Determine validation status based on test results.

        对于 Bug 注入场景：
        - 有效 Bug：
          - 至少一个测试从 PASSED 变为 FAILED (PASS_TO_FAIL > 0)
          - 且至少一个测试仍然保持 PASSED (PASS_TO_PASS > 0)
        - FAIL_TO_FAIL（pre-gold baseline 残余失败）不阻止 valid 判定，
          因为 seed gold patch 不保证修复所有测试，这些是 baseline noise。

        注意：FAIL_TO_PASS 用于验证修复补丁场景（测试从 FAIL 变为 PASS）

        Args:
            result: ValidationResult with test results

        Returns:
            ValidationStatus enum value
        """
        has_pass_to_fail = len(result.PASS_TO_FAIL) > 0
        has_pass_to_pass = len(result.PASS_TO_PASS) > 0

        # Bug 注入场景：只要注入确实翻转了测试 (P2F>0) 且没有全部破坏 (P2P>0)
        if has_pass_to_fail and has_pass_to_pass:
            return ValidationStatus.VALID
        else:
            return ValidationStatus.INVALID

    def _extract_traceback(self, test_output: str, failed_tests: list[str]) -> dict[str, list[dict]]:
        """
        从测试输出中提取指定失败测试的 traceback。

        P2: 解析 Python unittest/pytest 风格的 traceback，提取调用栈信息。

        Args:
            test_output: 完整的测试输出日志
            failed_tests: 失败测试的名称列表

        Returns:
            调用栈字典，每个失败测试名称映射到帧列表
            每个帧包含 file_path, line, function, code
        """
        traceback_info = {}

        # 正则匹配 traceback 帧
        # 格式: File "xxx.py", line 123, in function_name
        #         code_line
        frame_pattern = r'File "([^"]+)", line (\d+), in (\w+)\n\s+(.+)'

        for failed_test in failed_tests:
            frames = []

            # Django unittest 风格: ======...====\nFAIL: test_name (...)\n------...------\nTraceback...
            # 构建正则匹配测试失败块
            escaped_test = re.escape(failed_test) if failed_test else ""

            # 尝试多种格式匹配
            patterns = [
                # Django unittest 格式
                rf"={60,}\n(?:FAIL|ERROR): .*?{escaped_test}.*?\n-{60,}\n(.*?)(?=={60,}|\Z)",
                # pytest 格式
                rf"_{10,} {escaped_test} _{10,}\n(.*?)(?=_{10,}|\Z)",
                # 简化格式 - 查找任何包含测试名的 traceback 块
                rf"Traceback \(most recent call last\):\n(.*?)(?:AssertionError|Exception|Error).*",
            ]

            for pattern in patterns:
                match = re.search(pattern, test_output, re.DOTALL | re.IGNORECASE)
                if match:
                    traceback_section = match.group(1)

                    for frame_match in re.finditer(frame_pattern, traceback_section):
                        frames.append({
                            "file_path": frame_match.group(1),
                            "line": int(frame_match.group(2)),
                            "function": frame_match.group(3),
                            "code": frame_match.group(4).strip(),
                        })

                    if frames:
                        break

            # 如果没有找到特定测试的 traceback，尝试全局搜索
            if not frames and test_output:
                # 通用 traceback 搜索
                for frame_match in re.finditer(frame_pattern, test_output):
                    file_path = frame_match.group(1)
                    # 过滤掉 Python 标准库和测试框架的帧
                    if not any(skip in file_path for skip in ['/usr/lib/', '/site-packages/', 'unittest/', 'pytest/']):
                        frames.append({
                            "file_path": file_path,
                            "line": int(frame_match.group(2)),
                            "function": frame_match.group(3),
                            "code": frame_match.group(4).strip(),
                        })

            if frames:
                traceback_info[failed_test] = frames

        return traceback_info

    def _score_chain_alignment(
        self,
        instance: dict[str, Any],
        traceback_info: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        Heuristic chain alignment scoring for controllable loc-chain synthesis.

        Uses failing traceback frames as a proxy for "was the intended call path exercised?".
        """

        def _normalize_chain(chain_val: Any) -> list[str]:
            if chain_val is None:
                return []
            if isinstance(chain_val, str):
                s = chain_val.strip()
                if not s:
                    return []
                try:
                    return _normalize_chain(json.loads(s))
                except Exception:
                    return [s]
            if isinstance(chain_val, list):
                out: list[str] = []
                for item in chain_val:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        t = item.strip()
                        if t:
                            out.append(t)
                        continue
                    if isinstance(item, dict):
                        nid = item.get("node_id") or item.get("id") or item.get("name")
                        if nid:
                            out.append(str(nid).strip())
                        continue
                    out.append(str(item).strip())
                return [x for x in out if x]
            if isinstance(chain_val, dict):
                if "nodes" in chain_val and isinstance(chain_val["nodes"], list):
                    return _normalize_chain(chain_val["nodes"])
                return []
            return [str(chain_val).strip()]

        proposed = _normalize_chain(instance.get("proposed_chain"))
        target_node = (instance.get("target_node") or "").strip()
        if target_node and target_node not in proposed:
            proposed = proposed + [target_node]

        if not proposed or not traceback_info:
            return {
                "trace_coverage": 0.0,
                "matched_nodes": [],
                "total_nodes": len(proposed),
                "target_node_hit": False,
                "causal_ordering": 0.0,
                "overall_score": 0.0,
            }

        def _node_hints(node_id: str) -> tuple[str, str]:
            node_id = (node_id or "").strip()
            if not node_id:
                return "", ""
            file_hint = ""
            func_hint = ""
            if ":" in node_id:
                left, right = node_id.split(":", 1)
                if left.endswith(".py") or "/" in left:
                    file_hint = left
                func_hint = right.split(":")[-1]
            elif "/" in node_id and node_id.endswith(".py"):
                file_hint = node_id
            else:
                parts = node_id.split(".")
                if parts:
                    func_hint = parts[-1]
            return file_hint, func_hint

        def _frame_text(frame: dict[str, Any]) -> str:
            return " ".join(
                [
                    str(frame.get("file_path", "") or ""),
                    str(frame.get("function", "") or ""),
                    str(frame.get("code", "") or ""),
                ]
            )

        def _match_nodes_in_frames(frames: list[dict]) -> tuple[set[str], dict[str, int]]:
            """Match proposed chain nodes against a single traceback's frames."""
            m: set[str] = set()
            idx_map: dict[str, int] = {}
            for node in proposed:
                file_hint, func_hint = _node_hints(node)
                for fi, fr in enumerate(frames):
                    text = _frame_text(fr)
                    if file_hint and file_hint in text:
                        m.add(node)
                        idx_map[node] = fi
                        break
                    if func_hint and func_hint in text:
                        m.add(node)
                        idx_map[node] = fi
                        break
            return m, idx_map

        def _compute_ordering(idx_map: dict[str, int]) -> float:
            """Compute causal ordering score from frame index map.
            Returns 0.0 when evidence is insufficient (< 2 matched nodes)."""
            if len(idx_map) < 2:
                return 0.0  # Not enough evidence to judge ordering
            ordered = [idx_map[n] for n in proposed if n in idx_map]
            inversions = 0
            total_pairs = 0
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    total_pairs += 1
                    if ordered[i] > ordered[j]:
                        inversions += 1
            return 1.0 - (inversions / total_pairs) if total_pairs > 0 else 0.0

        # Score per-traceback (each failing test separately), then take the best.
        # This avoids cross-test frame mixing that destroys causal ordering semantics.
        best_coverage = 0.0
        best_target_hit = False
        best_ordering = 0.0
        all_matched: set[str] = set()

        for _test_name, frames in (traceback_info or {}).items():
            if not isinstance(frames, list):
                continue
            valid_frames = [f for f in frames if isinstance(f, dict)]
            if not valid_frames:
                continue
            m, idx_map = _match_nodes_in_frames(valid_frames)
            all_matched |= m
            cov = len(m) / len(proposed)
            t_hit = bool(target_node and target_node in m)
            ordering = _compute_ordering(idx_map)
            if cov > best_coverage or (cov == best_coverage and ordering > best_ordering):
                best_coverage = cov
                best_target_hit = t_hit
                best_ordering = ordering

        # Fallback: also check target_hit across all tracebacks
        if not best_target_hit and target_node:
            best_target_hit = target_node in all_matched

        # Weighted overall: coverage 0.5 + target_hit 0.25 + ordering 0.25
        # ordering weight reduced to avoid inflation when evidence is sparse
        overall = (
            0.5 * best_coverage
            + 0.25 * (1.0 if best_target_hit else 0.0)
            + 0.25 * best_ordering
        )

        return {
            "trace_coverage": round(best_coverage, 4),
            "matched_nodes": sorted(all_matched),
            "total_nodes": len(proposed),
            "target_node_hit": best_target_hit,
            "causal_ordering": round(best_ordering, 4),
            "overall_score": round(overall, 4),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_single(
    instance: dict[str, Any],
    profile: Any,
    config: Optional[ValidationConfig] = None,
) -> ValidationResult:
    """
    Validate a single instance with a given profile.

    Convenience function that creates a Validator and runs validation.

    Args:
        instance: Instance dict
        profile: Repository profile
        config: Optional validation config

    Returns:
        ValidationResult
    """
    validator = Validator(profile=profile, config=config)
    return validator.validate(instance)


def validate_from_files(
    patch_file: Path,
    profile: Any,
    instance_id: str = "manual_test",
    config: Optional[ValidationConfig] = None,
) -> ValidationResult:
    """
    Validate a patch from a file.

    Args:
        patch_file: Path to patch file
        profile: Repository profile
        instance_id: Unique identifier for this test
        config: Optional validation config

    Returns:
        ValidationResult
    """
    with open(patch_file) as f:
        patch_content = f.read()

    instance = {
        KEY_INSTANCE_ID: instance_id,
        KEY_PATCH: patch_content,
        "image_name": profile.image_name,
    }

    return validate_single(instance, profile, config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "Validator",
    "validate_single",
    "validate_from_files",
]
