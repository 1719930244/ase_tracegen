"""
PromptsMixin - Prompt building and context formatting methods for SynthesisAgent.

Extracted from agent.py to reduce file size. Contains:
- Seed context formatting
- Candidate context building
- Injection point constraint building
- Fix intent extraction
- Action history formatting
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import re

import networkx as nx
from loguru import logger

from .prompts.synthesis_agent_prompts import (
    INJECTION_POINT_CONSTRAINT,
    INTENT_EXAMPLES,
    DEFAULT_INTENT_EXAMPLES,
    MODULE_SAFETY_CONSTRAINT,
)
from .pattern_matcher import (
    SeedPattern,
    InjectionPoint,
    format_injection_points_for_prompt,
)
from .tools.context_tools import ReadCodeTool
from ...core.structures import ExtractionResult


class PromptsMixin:
    """
    Mixin providing prompt building and context formatting methods for SynthesisAgent.

    These methods format seed context, build candidate context, and construct
    prompt constraints for the LLM conversation loop.
    """

    def _build_injection_point_constraint(
        self,
        seed_pattern: SeedPattern,
        injection_points: List[InjectionPoint],
        fix_intent: str
    ) -> str:
        """
        构建注入点约束 Prompt 片段

        Args:
            seed_pattern: 从 Seed 提取的模式信息
            injection_points: 候选代码中找到的注入点列表
            fix_intent: Fix Intent 类型

        Returns:
            格式化的约束 Prompt
        """
        # 获取 intent-specific examples
        intent_examples = INTENT_EXAMPLES.get(fix_intent, DEFAULT_INTENT_EXAMPLES)

        # 格式化注入点列表
        injection_points_formatted = format_injection_points_for_prompt(
            injection_points, max_points=5
        )

        # 如果没有注入点，返回简化的约束
        if not injection_points:
            return f"""
## WARNING: No Specific Injection Points Identified

The system could not identify specific injection points matching the seed's pattern type '{seed_pattern.pattern_type}'.
You should still try to apply the Fix Intent '{fix_intent}' by:
1. Looking for similar patterns in the target code
2. Following the seed's transformation logic as closely as possible

**Seed Transformation Reference**:
- BEFORE (Buggy): `{seed_pattern.before_pattern}`
- AFTER (Fixed): `{seed_pattern.after_pattern}`
"""

        # 构建完整的约束
        return INJECTION_POINT_CONSTRAINT.format(
            intent_type=seed_pattern.intent_type,
            pattern_type=seed_pattern.pattern_type,
            semantic_category=seed_pattern.semantic_category,
            before_pattern=seed_pattern.before_pattern,
            after_pattern=seed_pattern.after_pattern,
            key_change_from=seed_pattern.key_change.get("from", ""),
            key_change_to=seed_pattern.key_change.get("to", ""),
            injection_points_formatted=injection_points_formatted,
            intent_specific_examples=intent_examples,
        )

    def _get_fix_intent_details(self, extraction_result: ExtractionResult) -> str:
        """
        获取完整的fix intent详情，包括变换逻辑
        """
        fix_intents = extraction_result.mined_data.get("fix_intents", [])
        if not fix_intents:
            return "Fix intent details not available."

        first_intent = fix_intents[0]

        # 兼容字典和对象访问
        itype = first_intent.get("type") if isinstance(first_intent, dict) else getattr(first_intent, "type", "Unknown")
        summary = first_intent.get("summary") if isinstance(first_intent, dict) else getattr(first_intent, "summary", "")

        # 获取变换逻辑
        code_trans = first_intent.get("code_transformation", {}) if isinstance(first_intent, dict) else getattr(first_intent, "code_transformation", {})
        before_code = code_trans.get('before', '') if isinstance(code_trans, dict) else getattr(code_trans, 'before', '')
        after_code = code_trans.get('after', '') if isinstance(code_trans, dict) else getattr(code_trans, 'after', '')

        details = f"Type: {itype}\n"
        details += f"Summary: {summary}\n"
        if before_code and after_code:
            details += f"Seed Transformation:\n"
            details += f"  BEFORE (Buggy): {before_code}\n"
            details += f"  AFTER (Fixed): {after_code}\n"
            details += (
                "Your Task: Mimic the seed's bug pattern on the target code. "
                "Use the BEFORE/AFTER as a reference for what kind of mistake to introduce, "
                "but avoid defaulting to generic logic/boolean reversal unless the seed pattern indicates it."
            )

        return details

    def _extract_fix_intent(self, extraction_result: ExtractionResult) -> str:
        """
        Extract fix intent from extraction result.
        从提取结果中提取修复意图
        """
        fix_intents = extraction_result.mined_data.get("fix_intents", [])
        if fix_intents:
            first_intent = fix_intents[0]
            # 兼容字典和对象访问
            itype = first_intent.get("type") if isinstance(first_intent, dict) else getattr(first_intent, "type", "Condition_Refinement")
            return itype or "Condition_Refinement"

        # Fallback to chains if fix_intents is empty
        chains = extraction_result.mined_data.get("extracted_chains", [])
        if not chains:
            return "Condition_Refinement"

        chain = chains[0]
        repair_chain = chain.get("extraction_metadata", {}).get("repair_chain", {})
        repair_op = repair_chain.get("repair_chain", {})

        return repair_op.get("type", "Condition_Refinement")

    def _build_module_safety_constraint(self, file_path: str) -> str:
        """
        Build module-specific mutation rules as a prompt-only constraint.

        Goal: reduce invalid generations caused by import-time crashes or broad breakage,
        especially for fragile utility/decorator modules.
        """
        fp = (file_path or "").strip()
        if not fp:
            return MODULE_SAFETY_CONSTRAINT

        # Django-specific tightening: decorators/utils modules are fragile and easy to break globally.
        if fp == "django/utils/decorators.py" or fp.startswith("django/utils/"):
            extra = """
### Django utils hardening (additional blacklist)
- Do NOT change function signatures anywhere in this file/module.
- Do NOT change decorator factory return types or wrapper construction patterns.
- Do NOT change `wraps()/update_wrapper()` usage, wrapper metadata propagation, or attribute copying.
- Prefer: tiny condition/constant/boundary changes inside existing control flow.
"""
            return MODULE_SAFETY_CONSTRAINT + "\n" + extra

        return MODULE_SAFETY_CONSTRAINT

    def _get_seed_chain_meta(self) -> Dict[str, Any]:
        """获取Seed链路的元信息"""
        if not self.current_seed:
            return {}

        chains = self.current_seed.mined_data.get("extracted_chains", [])
        if not chains:
            return {"depth": 3, "types": [], "node_ids": []}

        chain = chains[0] if isinstance(chains[0], dict) else chains[0]
        nodes = chain.get("nodes", [])

        return {
            "depth": len(nodes),
            "types": [n.get("node_type", "unknown") for n in nodes],
            "node_ids": [n.get("node_id", "") for n in nodes if isinstance(n, dict)],
        }

    def _format_action_history(self) -> str:
        """
        Format conversation history for prompt.
        为提示词格式化对话历史
        """
        if not self.conversation_history:
            return "No previous actions."

        history_lines = []
        # Show more history to avoid looping
        # 显示更多历史记录以避免循环
        for turn in self.conversation_history[-10:]:  # Last 10 turns
            history_lines.append(f"Turn {turn.turn_number}:")
            history_lines.append(f"  Thought: {turn.thought}")
            history_lines.append(f"  Action: {turn.action}")
            # Keep observation but truncate only if extremely long
            # 保留观察结果，但仅在极长时截断
            obs = turn.observation
            if len(obs) > 5000:
                obs = obs[:5000] + "... (truncated)"
            history_lines.append(f"  Observation: {obs}")
            history_lines.append("")

        return "\n".join(history_lines)

    def _format_recent_history(self, max_turns: int = 3) -> str:
        """P1 优化: 只格式化最近 N 轮历史，减少 token 用量"""
        if not self.conversation_history:
            return "No previous actions."
        recent = self.conversation_history[-max_turns:]
        lines = []
        for turn in recent:
            obs = turn.observation[:1000] + "..." if len(turn.observation) > 1000 else turn.observation
            lines.append(f"Turn {turn.turn_number}: {turn.action} → {obs}")
        return "\n".join(lines)

    def _format_seed_context(self, extraction_result: ExtractionResult, skip_fix_intents: bool = False) -> str:
        """
        Format the seed context into a readable string for the LLM.
        """
        repo = extraction_result.seed_metadata.get("repo", "unknown")
        problem_statement = extraction_result.seed_metadata.get("problem_statement", "No problem statement")
        output = [f"Seed Instance ID: {extraction_result.instance_id}"]
        output.append(f"Repo: {repo}")
        output.append(f"\n### SEED PROBLEM STATEMENT (Reference for style)")
        output.append(f"{problem_statement}")

        # Support for multiple fix intents mined from the seed
        # [Ablation] disable_fix_intent: 跳过 fix intent 信息
        fix_intents = extraction_result.mined_data.get("fix_intents", [])
        if fix_intents and not skip_fix_intents:
            output.append("\n## Available Fix Intents from Seed")
            for i, intent_data in enumerate(fix_intents):
                intent = intent_data.get("type", "Unknown")
                summary = intent_data.get("summary", "No summary")
                output.append(f"### Intent {i+1}: {intent}")
                output.append(f"Summary: {summary}")

                code_trans = intent_data.get("code_transformation", {})
                if code_trans:
                    before_code = code_trans.get('before', '')
                    after_code = code_trans.get('after', '')
                    output.append("Transformation Pattern:")
                    output.append(f"  - BEFORE (Buggy): ```python\n{before_code}\n```")
                    output.append(f"  - AFTER (Fixed): ```python\n{after_code}\n```")

        chains = extraction_result.mined_data.get("extracted_chains", [])
        if chains:
            for i, chain in enumerate(chains):
                output.append(f"\n--- Seed Localization Chain {i+1} ---")
                nodes = chain.get("nodes", [])
                output.append(f"Chain depth: {len(nodes)} nodes")
                for step in nodes:
                    if isinstance(step, dict):
                        output.append(f"  - {step.get('node_type', 'unknown')}: {step.get('node_id', 'unknown')}")
                    else:
                        output.append(f"  - {step.node_type}: {step.node_id}")

                # Include difficulty features if available
                meta = chain.get("extraction_metadata", {}) if isinstance(chain, dict) else getattr(chain, "extraction_metadata", {})
                difficulty = meta.get("difficulty_features", {})
                if difficulty:
                    output.append(f"  Difficulty: depth={difficulty.get('depth', '?')}, "
                                  f"in_degree={difficulty.get('in_degree', '?')}, "
                                  f"centrality={difficulty.get('betweenness_centrality', '?')}")
                node_pattern = meta.get("node_type_pattern", [])
                if node_pattern:
                    output.append(f"  Node type pattern: {' → '.join(node_pattern)}")

        return "\n".join(output)

    def _build_candidate_context(self, candidate: Dict[str, Any], graph: nx.DiGraph, repo_path: str) -> Dict[str, Any]:
        """
        预先读取所有必要信息，避免 Agent 调用工具。
        仿照 SWE-Smith 设计：系统掌控精确的代码位置信息。
        """
        # 1. 读取目标节点源码和精确位置信息 (Target Code with precise location)
        target_node_id = candidate.get('anchor_node_id', '')
        file_path = target_node_id.split(":")[0] if ":" in target_node_id else target_node_id

        # 从图中获取精确的行号范围
        line_start = 0
        line_end = 0
        indent_level = 0
        if graph and graph.has_node(target_node_id):
            node_data = graph.nodes[target_node_id]
            line_start = node_data.get("start_line", 0) or node_data.get("line_range", [0, 0])[0]
            line_end = node_data.get("end_line", 0) or node_data.get("line_range", [0, 0])[1]
            indent_level = node_data.get("indent_level", 0)

        # 直接从文件读取精确的源代码（确保与文件内容完全一致）
        target_code = ""
        full_path = Path(repo_path) / file_path
        if full_path.exists() and line_start > 0 and line_end > 0:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                    target_code = "".join(all_lines[line_start - 1 : line_end])
            except Exception as e:
                logger.warning(f"Failed to read target code from file: {e}")
                target_code = self._read_node_code(target_node_id)
        else:
            # 回退到原来的方式
            target_code = self._read_node_code(target_node_id)

        # 2. 挖掘子图/前驱 (Upstream Context)
        predecessors = []
        if graph and graph.has_node(target_node_id):
            preds = list(graph.predecessors(target_node_id))
            logger.debug(f"Found {len(preds)} predecessors for {target_node_id}")

            for p in preds[:5]:  # 限制数量，防止 Prompt 爆炸
                try:
                    p_data = graph.nodes[p]
                    p_file_path = p_data.get("file_path", "")
                    if not p_file_path and ":" in p:
                        p_file_path = p.split(":")[0]

                    predecessors.append({
                        "node_id": p,
                        "type": p_data.get("type", "unknown"),
                        "file": p_file_path
                    })
                    logger.debug(f"Added predecessor: {p} (file: {p_file_path})")
                except KeyError:
                    logger.warning(f"Predecessor {p} not found in graph nodes")
                    p_file_path = p.split(":")[0] if ":" in p else ""
                    predecessors.append({
                        "node_id": p,
                        "type": "unknown",
                        "file": p_file_path
                    })

        planned_suite = self._plan_validation_test_suite(file_path)
        nearby_test_files = self._get_nearby_test_files(file_path)

        related_tests = self._collect_related_test_details(
            file_path,
            target_node_id,
            planned_test_files=planned_suite.get("test_files", []),
        )
        planned_suite_tests = self._collect_test_cases_from_files(planned_suite.get("test_files", []), limit=60)
        planned_files_set = set(planned_suite.get("test_files", []) or [])

        # FAIL selection should be tightly related to the target (to avoid NO_FAIL).
        related_case_ids: list[str] = []
        strict_case_ids: list[str] = []
        for t in related_tests:
            if not isinstance(t, dict):
                continue
            tf = (t.get("test_file") or "").strip()
            fp = (t.get("full_path") or "").strip()
            if tf and fp and tf in planned_files_set:
                related_case_ids.append(fp)
                if t.get("target_hit"):
                    strict_case_ids.append(fp)

        allowed_fail_test_case_ids: list[str] = strict_case_ids or related_case_ids or planned_suite_tests
        allowed_pass_test_case_ids: list[str] = planned_suite_tests

        return {
            "target_node": target_node_id,
            "target_code": target_code,
            "file_path": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "indent_level": indent_level,
            "upstream_callers": predecessors,
            "planned_test_cmd": planned_suite.get("test_cmd", ""),
            "planned_test_modules": planned_suite.get("test_modules", []),
            "planned_test_files": planned_suite.get("test_files", []),
            "planned_suite_tests": planned_suite_tests,
            # Separate allow-lists: FAIL must be target-related; PASS can be any module-suite regression test.
            "allowed_fail_test_case_ids": allowed_fail_test_case_ids,
            "allowed_pass_test_case_ids": allowed_pass_test_case_ids,
            # Back-compat: keep the old key as the PASS allow-list.
            "allowed_test_case_ids": allowed_pass_test_case_ids,
            "nearby_test_files": nearby_test_files,
            "related_tests": related_tests,
            "seed_chain_meta": self._get_seed_chain_meta()
        }

    def _format_context_dump(self, ctx: Dict[str, Any]) -> str:
        """格式化上下文信息为字符串，包含精确的代码位置信息和测试断言"""
        dump = []

        # 显示精确的代码位置信息（SWE-Smith 风格）
        dump.append(f"### TARGET CODE: {ctx['target_node']}")
        dump.append(f"**File**: `{ctx.get('file_path', 'unknown')}`")
        dump.append(f"**Lines**: {ctx.get('line_start', 0)}-{ctx.get('line_end', 0)}")
        dump.append("")
        dump.append("**IMPORTANT**: The code below is the EXACT content from the file.")
        dump.append("You do NOT need to provide `code_before` - the system already has it.")
        dump.append("You ONLY need to provide `code_after` (the modified buggy version).")
        dump.append("")
        dump.append("```python")
        dump.append(ctx['target_code'] or "Code not available")
        dump.append("```")

        dump.append("\n### KNOWN UPSTREAM CALLERS (Potential Chain Entry Points)")
        for i, p in enumerate(ctx['upstream_callers']):
            dump.append(f"{i+1}. {p['node_id']} (File: {p['file']})")

        # Show which test suite the validator is expected to run.
        dump.append("\n### VALIDATION TEST PLAN (What WILL be executed during validation)")
        planned_cmd = (ctx.get("planned_test_cmd") or "").strip()
        planned_modules = ctx.get("planned_test_modules") or []
        planned_files = ctx.get("planned_test_files") or []
        if planned_cmd:
            dump.append(f"**Planned test command**: `{planned_cmd}`")
        if planned_modules:
            dump.append(f"**Planned test modules/labels**: {', '.join(planned_modules[:20])}")
        if planned_files:
            dump.append("**Planned test files to inspect** (read these to pick expected_tests_to_fail):")
            for i, f in enumerate(planned_files[:20]):
                dump.append(f"  {i+1}. {f}")
        if not (planned_cmd or planned_modules or planned_files):
            dump.append("No explicit test plan was inferred. Validation will fall back to generic test selection.")
        dump.append("**IMPORTANT**: Your injected bug MUST fail under the planned validation suite above.")

        seed_meta = ctx.get("seed_chain_meta", {}) or {}
        seed_node_ids = seed_meta.get("node_ids", []) if isinstance(seed_meta, dict) else []
        if seed_node_ids:
            dump.append("\n### SEED LOC CHAIN (Ground-truth structure to mimic)")
            for i, nid in enumerate(seed_node_ids[:15]):
                dump.append(f"{i+1}. {nid}")
            dump.append("**IMPORTANT**: Your proposed_chain should mirror this structure (same depth) and be plausible callers->target.")

        planned_suite_tests = ctx.get("planned_suite_tests") or []
        if planned_suite_tests:
            dump.append("\n### PLANNED SUITE TEST CASES (Choose FAIL/PASS labels from these)")
            for i, t in enumerate(planned_suite_tests[:30]):
                dump.append(f"{i+1}. {t}")
            dump.append("**IMPORTANT**: Pick `expected_tests_to_fail` ONLY from the list above (or from RELATED EXISTING TESTS below).")
            dump.append("Also pick `pass_to_pass` labels ONLY from the list above that should remain passing.")

        dump.append("\n### AVAILABLE TEST FILES (You can read these using read_code)")
        if ctx.get('nearby_test_files'):
            for i, f in enumerate(ctx['nearby_test_files']):
                dump.append(f"{i+1}. {f}")
        else:
            dump.append("No nearby test files found automatically.")

        # P0: 显示相关的现有测试用例（用于 expected_tests_to_fail），包括断言信息
        dump.append("\n### RELATED EXISTING TESTS (Use these for expected_tests_to_fail)")
        related_tests = ctx.get('related_tests', [])
        if related_tests:
            for test in related_tests[:8]:  # 多展示一些，便于 Agent 选择可触发失败的断言
                dump.append(f"\n**Test**: `{test['full_path']}`")
                # 显示断言信息
                if test.get('assertions'):
                    dump.append("**Assertions** (break these to make the test fail):")
                    for assertion in test['assertions'][:5]:  # 每个测试最多5个断言
                        # 截断过长的断言
                        if len(assertion) > 150:
                            assertion = assertion[:150] + "..."
                        dump.append(f"  - `{assertion}`")
                # 显示源码预览
                if test.get('source_preview'):
                    preview = test['source_preview']
                    if len(preview) > 400:
                        preview = preview[:400] + "..."
                    dump.append(f"**Preview**: ```python\n{preview}\n```")

            dump.append("\n**IMPORTANT**: Use the exact full_path format above for expected_tests_to_fail")
            dump.append("Your bug should make at least ONE of the assertions above fail!")
            dump.append("Do NOT make up test names like 'test_reproduce_bug' or 'test_synthetic'")
        else:
            dump.append("No related tests found. The validation will auto-discover failing tests.")

        return "\n".join(dump)
