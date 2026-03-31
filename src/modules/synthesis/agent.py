"""
SynthesisAgent - Multi-turn conversation agent for bug synthesis.
合成 Agent - 基于多轮对话的 Bug 合成 Agent

This agent uses a ReAct-style loop to:
1. Understand seed defect patterns
2. Search for similar code locations
3. Generate new bug injections with controlled complexity

核心设计理念 (Based on Paper Design):
1. 从 Seed 的 Loc Chain 和 Fix Intent 出发
2. 搜索结构相似的目标节点（相似的图深度、入度、出度）
3. 基于 Fix Intent 约束，执行符合语义的代码变换
4. 生成标准格式的 Patch

Refactored: core orchestration logic only.
Validation methods -> agent_validation.py (ValidationMixin)
Prompt building methods -> agent_prompts.py (PromptsMixin)
Tool/action handling methods -> agent_tools.py (ToolsMixin)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import re
import time
import traceback
import difflib
import networkx as nx
from loguru import logger

from ..llm_client import LLMClient
from .tools.base import ToolRegistry, BaseTool
from .tools.context_tools import ReadCodeTool
from .tools.graph_tools import QueryGraphTool, SearchSimilarNodesTool
from .tools.metrics_tools import ComputeChainMetricsTool
from .heuristics.intent_rules import FixIntentTransformer, INTENT_INJECTION_RULES
from .prompts.synthesis_agent_prompts import (
    SYNTHESIS_AGENT_SYSTEM_PROMPT,
    TURN_TEMPLATE,
    ERROR_RECOVERY_PROMPT,
    REGRESSION_TEST_CONSTRAINT,
    FAILURE_DRIVEN_CONSTRAINT,
    PS_LEVEL_INSTRUCTIONS,
    DEFAULT_PS_LEVEL,
)
from .pattern_matcher import (
    FixIntentPatternMatcher,
    check_intent_compatibility,
)
from ...core.structures import DefectChain, ExtractionResult, SynthesisResult

# Mixin imports
from .agent_validation import ValidationMixin
from .agent_prompts import PromptsMixin
from .agent_tools import ToolsMixin


@dataclass
class AgentTurn:
    """
    Represents a single turn in the agent conversation.
    表示 Agent 对话中的单个轮次
    """
    turn_number: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


from ...core.interfaces import Synthesizer

class SynthesisAgent(ValidationMixin, PromptsMixin, ToolsMixin, Synthesizer):
    """
    Multi-turn conversation agent for bug synthesis.
    用于 Bug 合成的多轮对话 Agent

    Uses a ReAct (Reasoning + Acting) loop to:
    1. Understand seed patterns
    2. Explore the repository graph
    3. Find suitable injection targets
    4. Generate realistic bugs

    Methods are split across mixin classes:
    - ValidationMixin: patch validation, sanitization, intent alignment
    - PromptsMixin: prompt building, context formatting, seed/candidate context
    - ToolsMixin: response parsing, patch generation, test discovery/generation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Dict[str, Any],
        graph: Optional[nx.DiGraph] = None,
        repo_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
        repo_profile=None,
    ):
        """
        Initialize SynthesisAgent.
        初始化合成 Agent

        Args:
            llm_client: LLM client for conversation
            config: Agent configuration
            graph: Repository graph (can be set later per-instance)
            repo_path: Path to cloned repository
            output_dir: Directory for saving outputs
        """
        self.llm = llm_client
        self.config = config
        self.graph = graph
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.repo_profile = repo_profile  # RepoProfile instance (injected by pipeline)

        # Configuration
        # 配置
        self.max_turns = config.get("max_turns", 75)
        self.early_stop_threshold = config.get("early_stop_threshold", 3)  # P0: 连续相同失败N次则早停
        self.temperature = config.get("temperature", 0.7)
        # 是否生成合成测试文件（默认关闭，使用仓库现有测试）
        self.generate_synthetic_test = config.get("generate_synthetic_test", False)
        # PS level control: minimal / standard / detailed / mixed
        self._ps_level = config.get("ps_level", DEFAULT_PS_LEVEL)

        # Ablation switches
        _abl = config.get("ablation", {})
        self._disable_fix_intent = bool(_abl.get("disable_fix_intent", False))
        self._disable_agent_exploration = bool(_abl.get("disable_agent_exploration", False))
        self._disable_quality_controls = bool(_abl.get("disable_quality_controls", False))

        # Initialize components
        # 初始化组件
        self.tool_registry = ToolRegistry()
        self.intent_transformer = FixIntentTransformer()

        # Conversation state
        # 对话状态
        self.conversation_history: List[AgentTurn] = []
        self.current_seed: Optional[ExtractionResult] = None
        self.system_prompt: str = ""  # 存储当前的 system prompt

        # LLM call tracking
        # LLM 调用追踪
        self.llm_calls: List[Dict[str, Any]] = []

        logger.info("SynthesisAgent initialized")

    def _setup_tools(self, context: Dict[str, Any]) -> None:
        """
        Setup tools with shared context.
        使用共享上下文设置工具
        """
        tools = [
            ReadCodeTool(context),
            QueryGraphTool(context),
            SearchSimilarNodesTool(context),
            ComputeChainMetricsTool(context),
        ]

        self.tool_registry = ToolRegistry()
        self.tool_registry.register_all(tools)

    def synthesize(
        self,
        extraction_result: ExtractionResult,
        graph: nx.DiGraph,
        repo_path: str,
        candidate: Dict[str, Any],
        rank: int = 1,
        validation_feedback: str = "",
    ) -> Optional[SynthesisResult]:
        """
        针对特定的候选点运行聚焦合成过程。
        """
        logger.info(f"Starting focused synthesis for candidate {rank}: {candidate['anchor_node_id']}")

        # Reset state
        self.conversation_history = []
        self.current_seed = extraction_result
        self.graph = graph
        self.repo_path = repo_path

        # Setup context for tools
        repo = extraction_result.seed_metadata.get("repo", "unknown")
        context = {
            "graph": graph,
            "repo_path": repo_path,
            "extraction_result": extraction_result,
            "target_candidate": candidate,
            "seed_data": {
                "instance_id": extraction_result.instance_id,
                "repo": repo,
            }
        }
        self._setup_tools(context)

        # Get fix intent from seed
        # [Ablation] disable_fix_intent: 不传递 fix_intent 给 agent
        if self._disable_fix_intent:
            logger.info("[Ablation] disable_fix_intent: 使用通用 intent 替代 seed fix intent")
            fix_intent = "Generic_Bug_Injection"
            fix_intent_details = "Inject a realistic bug that causes at least one test to fail."
            injection_strategies = "Apply any plausible code mutation that introduces a behavioral defect."
        else:
            fix_intent = self._extract_fix_intent(extraction_result)
            fix_intent_details = self._get_fix_intent_details(extraction_result)
            injection_strategies = self.intent_transformer.to_prompt_format(fix_intent)

        # Format seed context as string
        seed_context = self._format_seed_context(extraction_result, skip_fix_intents=self._disable_fix_intent)

        # 1. 提取 Seed 链路深度和难度特征
        seed_chains = extraction_result.mined_data.get("extracted_chains", [])
        target_depth = 3
        seed_difficulty = {}
        seed_node_type_pattern = []
        if seed_chains:
            first_chain = seed_chains[0] if isinstance(seed_chains[0], dict) else seed_chains[0]
            nodes = first_chain.get("nodes", []) if isinstance(first_chain, dict) else first_chain.nodes
            target_depth = len(nodes)
            # Extract difficulty features from extraction metadata
            meta = first_chain.get("extraction_metadata", {}) if isinstance(first_chain, dict) else getattr(first_chain, "extraction_metadata", {})
            seed_difficulty = meta.get("difficulty_features", {})
            seed_node_type_pattern = meta.get("node_type_pattern", [])

        # 2. 构建候选者上下文（全知视角）
        candidate_context = self._build_candidate_context(candidate, graph, repo_path)
        context_dump = self._format_context_dump(candidate_context)

        # Fix Intent 语义兼容性检查和注入点提取
        fix_intent_data = extraction_result.mined_data.get("fix_intents", [{}])
        if fix_intent_data:
            fix_intent_data = fix_intent_data[0]
        else:
            fix_intent_data = {}

        # 从 candidate 中获取已预计算的注入点（如果有）
        injection_points = candidate.get("_injection_points", [])

        # 如果没有预计算，则现场计算
        if not injection_points:
            target_code = candidate_context.get("target_code", "")
            if target_code and fix_intent_data:
                is_compatible, injection_points, reason = check_intent_compatibility(
                    target_code, fix_intent_data, min_match_score=0.5
                )
                if not is_compatible:
                    logger.warning(f"候选点 {rank} 与 Fix Intent 不兼容: {reason}")

        # 提取 Seed Pattern 信息
        pattern_matcher = FixIntentPatternMatcher()
        seed_pattern = pattern_matcher.extract_seed_pattern(fix_intent_data)

        # 保存到实例变量供后验校验使用
        self._current_injection_points = injection_points
        self._current_seed_pattern = seed_pattern
        self._current_fix_intent = fix_intent
        # P3: 保存候选分数供对话循环判断是否走单轮直出
        self._current_candidate_score = float(candidate.get("vector_score", 0.0) or 0.0)

        # 构建注入点约束 Prompt
        # [Ablation] disable_fix_intent: 跳过注入点约束
        if self._disable_fix_intent:
            injection_point_constraint = ""
        else:
            injection_point_constraint = self._build_injection_point_constraint(
                seed_pattern, injection_points, fix_intent
            )

        # 3. 运行对话循环（全知模式）
        result = self._run_conversation_loop(
            seed_id=extraction_result.instance_id,
            fix_intent=fix_intent,
            fix_intent_details=fix_intent_details,
            injection_strategies=injection_strategies,
            seed_context=seed_context,
            candidate_context_dump=context_dump,
            candidate_context=candidate_context,
            seed_depth=target_depth,
            target_node_id=candidate.get('anchor_node_id', ''),
            injection_point_constraint=injection_point_constraint,
            validation_feedback=validation_feedback,
        )

        return result

    def _run_conversation_loop(
        self,
        seed_id: str,
        fix_intent: str,
        fix_intent_details: str,
        injection_strategies: str,
        seed_context: str,
        candidate_context_dump: str,
        candidate_context: Dict[str, Any],
        seed_depth: int,
        target_node_id: str,
        injection_point_constraint: str = "",
        validation_feedback: str = "",
    ) -> Optional[SynthesisResult]:
        """
        Main conversation loop.
        主对话循环
        """
        action_history = "No previous actions."

        # 保存 candidate_context 供 _handle_generate_bug 使用
        self._current_candidate_context = candidate_context

        # 全知视角模式：System Prompt已包含所有必要信息
        module_safety_constraint = self._build_module_safety_constraint(
            file_path=(candidate_context.get("file_path", "") or "")
        )
        # Resolve PS level instruction
        ps_level = self._ps_level
        if ps_level == "mixed":
            import random
            ps_level = random.choice(["minimal", "standard", "detailed"])
            logger.info(f"PS level 'mixed' resolved to '{ps_level}' for this instance")
        ps_level_instruction = PS_LEVEL_INSTRUCTIONS.get(ps_level, PS_LEVEL_INSTRUCTIONS[DEFAULT_PS_LEVEL])

        # P2: 动态约束裁剪 - 按 fix_intent 类型选择相关约束，减少 token
        # 简单局部修改类型不需要 MODULE_SAFETY_CONSTRAINT（使用项目 canonical taxonomy）
        _simple_intents = {"Constant_Update", "Condition_Refinement", "Variable_Replacement",
                           "Argument_Update", "Exception_Fix"}
        if fix_intent in _simple_intents:
            effective_module_constraint = ""  # 简单修改不需要模块安全约束
        else:
            effective_module_constraint = module_safety_constraint

        self.system_prompt = SYNTHESIS_AGENT_SYSTEM_PROMPT.format(
            fix_intent=fix_intent,
            fix_intent_details=fix_intent_details,
            seed_depth=seed_depth,
            seed_depth_plus1=seed_depth + 1,
            seed_context=seed_context,
            target_node_id=target_node_id,
            candidate_context_dump=candidate_context_dump,
            injection_point_constraint=injection_point_constraint,
            module_safety_constraint=effective_module_constraint,
            regression_test_constraint=REGRESSION_TEST_CONSTRAINT,
            failure_driven_constraint=FAILURE_DRIVEN_CONSTRAINT,
            ps_level_instruction=ps_level_instruction,
        )

        # 获取目标文件（以第一个 candidate 为参考）
        recommended_tests = []
        if self.current_seed.mined_data.get("extracted_chains"):
            first_node = self.current_seed.mined_data["extracted_chains"][0]["nodes"][-1]
            f_path = first_node.get("file_path") if isinstance(first_node, dict) else first_node.file_path
            recommended_tests = self._get_recommended_tests(f_path)

        test_hint = f"\n## Recommended Regression Tests (PASS_TO_PASS)\nBased on the target location, consider these existing tests: {', '.join(recommended_tests)}" if recommended_tests else ""
        if validation_feedback:
            test_hint += f"\n\n## Previous Validation Feedback\n{validation_feedback.strip()}\n"

        # [Ablation] disable_agent_exploration: 单次 LLM 调用替代多轮 ReAct
        if self._disable_agent_exploration:
            logger.info("[Ablation] disable_agent_exploration: 单次 LLM 调用模式")
            single_shot_prompt = (
                "You have ALL the context you need in the system prompt.\n"
                "Generate the bug injection NOW in a single response.\n"
                "Do NOT use any tools (read_code, query_graph, etc.).\n\n"
                "Respond EXACTLY in this format:\n"
                "Thought: [Your analysis]\n"
                "Action: generate_bug\n"
                "Action Input: {JSON with code_after, bug_description, proposed_chain, "
                "FAIL_TO_PASS, PASS_TO_PASS}\n"
            ) + test_hint

            start_time = time.time()
            response = self.llm.complete(
                prompt=single_shot_prompt,
                system_message=self.system_prompt
            )
            duration = time.time() - start_time
            self._record_llm_call(1, single_shot_prompt, response, duration)

            thought, action, action_input = self._parse_response(response)
            if action and action.lower() == "generate_bug" and action_input:
                result = self._handle_generate_bug(action_input, seed_id, fix_intent)
                if result:
                    self.conversation_history.append(AgentTurn(
                        turn_number=1, thought=thought or "", action=action,
                        action_input=action_input, observation="Bug generated (single-shot)"
                    ))
                    self._save_agent_trace()
                    return result

            logger.warning("[Ablation] 单次调用未生成有效 bug")
            self._save_agent_trace()
            return None

        # P0: 早停机制 - 跟踪连续相同失败类型
        consecutive_fail_type = ""
        consecutive_fail_count = 0
        early_stop_limit = self.early_stop_threshold

        # P3: 高置信候选使用单轮直出（vector_score > 0.7 且有 injection points）
        candidate_score = getattr(self, '_current_candidate_score', 0.0)
        has_injection_points = bool(getattr(self, '_current_injection_points', []))
        if candidate_score > 0.7 and has_injection_points:
            logger.info(f"P3: 高置信候选 (score={candidate_score:.2f})，尝试单轮直出")
            single_shot_prompt = (
                "You have ALL the context you need in the system prompt.\n"
                "Generate the bug injection NOW in a single response.\n"
                "Do NOT use any tools.\n\n"
                "Thought: [Your analysis]\n"
                "Action: generate_bug\n"
                "Action Input: {JSON with code_after, bug_description, proposed_chain, "
                "FAIL_TO_PASS, PASS_TO_PASS}\n"
            ) + test_hint

            start_time = time.time()
            response = self.llm.complete(prompt=single_shot_prompt, system_message=self.system_prompt)
            duration = time.time() - start_time
            self._record_llm_call(0, single_shot_prompt, response, duration)

            thought, action, action_input = self._parse_response(response)
            if action and action.lower() == "generate_bug" and action_input:
                result = self._handle_generate_bug(action_input, seed_id, fix_intent)
                if result:
                    self.conversation_history.append(AgentTurn(
                        turn_number=0, thought=thought or "", action=action,
                        action_input=action_input, observation="Bug generated (high-confidence single-shot)"
                    ))
                    self._save_agent_trace()
                    return result
            logger.info("P3: 单轮直出失败，回退到多轮模式")
            # 修复: fallback 前重算 action_history，让多轮循环看到单轮失败记录
            action_history = self._format_action_history() if self.conversation_history else action_history

        for turn_num in range(1, self.max_turns + 1):
            logger.info(f"Turn {turn_num}/{self.max_turns}")

            # P1 修正: llm.complete() 是无状态调用，必须每轮发完整系统提示
            # 改为只精简 action_history（减少用户 prompt 侧 token）
            current_system = self.system_prompt

            # P1: 后续轮精简 action_history，只保留最近 3 轮（减少 prompt token）
            if turn_num > 3 and len(self.conversation_history) > 3:
                recent_history = self._format_recent_history(max_turns=3)
                turn_action_history = f"[Turns 1-{turn_num-4} omitted, {turn_num-4} prior attempts failed]\n\n" + recent_history
            else:
                turn_action_history = action_history

            turn_prompt = TURN_TEMPLATE.format(
                turn_number=turn_num,
                max_turns=self.max_turns,
                action_history=turn_action_history,
                fix_intent=fix_intent
            ) + test_hint

            start_time = time.time()
            response = self.llm.complete(
                prompt=turn_prompt,
                system_message=current_system
            )
            duration = time.time() - start_time
            self._record_llm_call(turn_num, turn_prompt, response, duration)

            thought, action, action_input = self._parse_response(response)

            # 强制拦截：全知模式下禁止调用图查询工具
            forbidden_tools = ["query_graph", "search_similar_nodes", "compute_chain_metrics"]
            if action and action.lower() in forbidden_tools:
                error_msg = (
                    "CRITICAL ERROR: You are in Omniscient Mode with FULL CONTEXT provided.\n"
                    "You MUST NOT call graph exploration tools.\n"
                    "Call 'generate_bug' directly with your bug injection proposal.\n"
                )
                logger.warning(f"Turn {turn_num}: Agent attempted forbidden tool call: {action}")
                self.conversation_history.append(AgentTurn(
                    turn_number=turn_num,
                    thought="I attempted to call an exploration tool, but I should directly generate the bug.",
                    action="forbidden_tool_call", action_input={}, observation=error_msg
                ))
                action_history = self._format_action_history()
                # P0: 早停检查
                fail_type = "forbidden_tool"
                if fail_type == consecutive_fail_type:
                    consecutive_fail_count += 1
                else:
                    consecutive_fail_type, consecutive_fail_count = fail_type, 1
                if consecutive_fail_count >= early_stop_limit:
                    logger.warning(f"P0 早停: 连续 {consecutive_fail_count} 次 {fail_type}，放弃该候选")
                    self._save_agent_trace()
                    return None
                continue

            if not action or (not action_input and action and action.lower() == "generate_bug"):
                missing = []
                if not thought: missing.append("Thought:")
                if not action: missing.append("Action:")
                if not action_input and action and action.lower() == "generate_bug": missing.append("Action Input (valid JSON)")

                error_msg = (
                    f"Format Error: Missing tags: {', '.join(missing)}.\n"
                    f"You must respond with:\nThought: [reasoning]\nAction: generate_bug\n"
                    f"Action Input: [JSON]\n\n--- YOUR PREVIOUS RESPONSE ---\n{response[:500]}..."
                )
                logger.warning(f"Turn {turn_num}: format error")
                self.conversation_history.append(AgentTurn(
                    turn_number=turn_num, thought="Format error.",
                    action="format_error", action_input={}, observation=error_msg
                ))
                action_history = self._format_action_history()
                # P0: 早停检查
                fail_type = "format_error"
                if fail_type == consecutive_fail_type:
                    consecutive_fail_count += 1
                else:
                    consecutive_fail_type, consecutive_fail_count = fail_type, 1
                if consecutive_fail_count >= early_stop_limit:
                    logger.warning(f"P0 早停: 连续 {consecutive_fail_count} 次 {fail_type}，放弃该候选")
                    self._save_agent_trace()
                    return None
                continue

            if action.lower() == "generate_bug":
                result = self._handle_generate_bug(action_input, seed_id, fix_intent)
                if result:
                    self.conversation_history.append(AgentTurn(
                        turn_number=turn_num, thought=thought, action=action,
                        action_input=action_input, observation="Bug generated successfully"
                    ))
                    self._save_agent_trace()
                    return result
                else:
                    # P0: 早停 - 统一归类为 generate_bug_fail
                    # (不区分子类型，连续 N 次 generate_bug 返回 None 即早停)
                    fail_type = "generate_bug_fail"
                    if fail_type == consecutive_fail_type:
                        consecutive_fail_count += 1
                    else:
                        consecutive_fail_type, consecutive_fail_count = fail_type, 1
                    if consecutive_fail_count >= early_stop_limit:
                        logger.warning(f"P0 早停: 连续 {consecutive_fail_count} 次 {fail_type}，放弃该候选")
                        self._save_agent_trace()
                        return None
                    action_history = self._format_action_history()
                    continue

            # Execute tool
            observation = self.tool_registry.call(action, action_input)
            turn = AgentTurn(
                turn_number=turn_num, thought=thought, action=action,
                action_input=action_input, observation=observation
            )
            self.conversation_history.append(turn)
            action_history = self._format_action_history()
            # 工具调用成功，重置早停计数
            consecutive_fail_type, consecutive_fail_count = "", 0

        logger.warning("Max turns reached without generating bug")
        self._save_agent_trace()
        return None

    def _handle_generate_bug(
        self,
        action_input: Dict[str, Any],
        seed_id: str,
        fix_intent: str
    ) -> Optional[SynthesisResult]:
        """
        处理 Bug 生成请求。
        SWE-Smith 风格：系统控制代码位置，Agent 只需提供 code_after。
        """
        try:
            # SWE-Smith 风格：从系统预读取的上下文获取代码位置信息
            ctx = getattr(self, '_current_candidate_context', None)
            if not ctx:
                error_msg = "System error: No candidate context available."
                logger.error(error_msg)
                return None

            # 从系统上下文获取精确的代码位置
            target_node = ctx.get('target_node', '')
            file_path = ctx.get('file_path', '')
            line_start = ctx.get('line_start', 0)
            line_end = ctx.get('line_end', 0)
            code_before = ctx.get('target_code', '')

            # 从 Agent 输出获取 code_after 和其他信息
            code_after = (
                action_input.get("code_after") or
                action_input.get("after") or
                action_input.get("code_transformation", {}).get("after")
            )
            injection_strategy = (
                action_input.get("injection_strategy") or
                action_input.get("injection_type") or
                fix_intent or
                "unspecified"
            )
            bug_description = (
                action_input.get("bug_description") or
                action_input.get("problem_statement") or
                "Synthetic defect"
            )

            # Problem Statement 泄露检测与清洗
            bug_description = self._sanitize_problem_statement(bug_description, code_before, code_after)

            # PS quality gate: validate against target level
            ps_level = getattr(self, "_ps_level", "standard")
            if ps_level == "mixed":
                ps_level = "standard"  # validate against standard for mixed mode
            ps_ok, ps_reason = self._validate_ps_quality(bug_description, ps_level)
            if not ps_ok:
                logger.warning(f"PS quality check failed: {ps_reason}. PS: '{bug_description[:80]}...'")
                # Try to salvage: if PS became fallback after sanitization, keep it but log
                # Don't reject the entire bug for a bad PS; the bug itself may be valuable
                if bug_description in ("Synthetic defect", ""):
                    logger.info("PS degraded to fallback after sanitization; proceeding with fallback PS")
                else:
                    logger.info(f"PS quality below target ({ps_level}), but proceeding with available PS")

            # 验证必要参数
            if not code_after:
                error_msg = "Missing required parameter: code_after. You MUST provide the modified buggy code."
                logger.warning(error_msg)
                self.conversation_history.append(AgentTurn(
                    turn_number=len(self.conversation_history) + 1,
                    thought="I missed the code_after parameter.",
                    action="generate_bug_failed",
                    action_input=action_input,
                    observation=error_msg
                ))
                return None

            if not code_before:
                error_msg = "System error: target_code not available in context."
                logger.warning(error_msg)
                return None

            # Capture model's raw output before any sanitization (for audit)
            raw_code_after = code_after

            # Best-effort sanitize: strip newly introduced comments from code_after.
            try:
                sanitized, removed = self._strip_new_comments(code_before=code_before, code_after=code_after)
                if removed > 0:
                    logger.warning(f"Auto-stripped {removed} newly introduced comment(s) from code_after")
                    code_after = sanitized
            except Exception as e:
                logger.debug(f"Failed to sanitize code_after comments: {e}")

            # Auto-preserve docstrings: replace model-drifted docstrings with
            # exact originals from code_before (source-span replacement, no ast.unparse).
            try:
                preserved, n_restored = self._preserve_original_docstrings(code_before, code_after)
                if n_restored > 0:
                    logger.info(f"Auto-restored {n_restored} docstring(s) from original source")
                    code_after = preserved
            except Exception as e:
                logger.debug(f"Failed to preserve docstrings: {e}")

            # 全知视角：Agent必须提供完整的proposed_chain
            proposed_chain_raw = action_input.get("proposed_chain")
            if not proposed_chain_raw:
                error_msg = "Agent did not provide proposed_chain. This is required to build the defect chain."
                logger.warning(error_msg)
                self.conversation_history.append(AgentTurn(
                    turn_number=len(self.conversation_history) + 1,
                    thought="I forgot to provide proposed_chain.",
                    action="generate_bug_failed",
                    action_input=action_input,
                    observation=error_msg
                ))
                return None

            # 验证是否为原始 Bug 位置
            seed_chains = self.current_seed.mined_data.get("extracted_chains", []) if self.current_seed else []

            # P1: 获取 Seed 链路深度
            seed_depth = 3
            if seed_chains:
                first_chain_for_depth = seed_chains[0]
                if isinstance(first_chain_for_depth, dict):
                    depth_nodes = first_chain_for_depth.get("nodes", [])
                else:
                    depth_nodes = first_chain_for_depth.nodes if hasattr(first_chain_for_depth, "nodes") else []
                if depth_nodes:
                    seed_depth = len(depth_nodes)

            # P1: 将 proposed_chain 转换为结构化格式
            proposed_chain = self._parse_structured_chain(proposed_chain_raw, seed_depth)
            logger.info(f"结构化 proposed_chain: {len(proposed_chain)} 节点 (seed_depth: {seed_depth})")

            original_node_ids = set()
            original_locations = []

            for chain in seed_chains:
                chain_nodes = chain.nodes if not isinstance(chain, dict) else chain.get("nodes", [])
                for node in chain_nodes:
                    nid = node.node_id if not isinstance(node, dict) else node.get("node_id")
                    if nid:
                        original_node_ids.add(nid)
                        if self.graph and nid in self.graph:
                            n_data = self.graph.nodes[nid]
                            f_path = n_data.get("file_path") or (nid.split(":")[0] if ":" in nid else "")
                            l_range = n_data.get("line_range")
                            if f_path and l_range and l_range[0] > 0:
                                original_locations.append((f_path, l_range[0], l_range[1]))

            if target_node in original_node_ids:
                error_msg = f"Injection rejected: '{target_node}' is the original seed bug location."
                logger.warning(error_msg)
                self.conversation_history.append(AgentTurn(
                    turn_number=len(self.conversation_history) + 1,
                    thought="I tried to inject into the original bug location.",
                    action="generate_bug_failed",
                    action_input=action_input,
                    observation=error_msg
                ))
                return None

            # Fix Intent 对齐后验校验
            seed_pattern = getattr(self, '_current_seed_pattern', None)
            expected_intent = getattr(self, '_current_fix_intent', fix_intent)
            injection_points = getattr(self, '_current_injection_points', [])

            # 检查 Agent 是否选择了正确的注入点
            selected_line = action_input.get("selected_injection_line")
            if selected_line and injection_points:
                valid_lines = [p.line_start for p in injection_points]
                if selected_line not in valid_lines:
                    warning_msg = (
                        f"WARNING: Agent selected line {selected_line} which is not in the "
                        f"identified injection points {valid_lines}."
                    )
                    logger.warning(warning_msg)

            # 验证生成的修改是否与预期的 Fix Intent 类型一致
            if seed_pattern and code_before and code_after:
                alignment_score = self._validate_intent_alignment(
                    code_before, code_after, seed_pattern, expected_intent
                )
                if alignment_score < 0.5:
                    logger.warning(f"Intent alignment score: {alignment_score:.0%} for {target_node}")
                else:
                    logger.info(f"Intent alignment verified: {alignment_score:.0%} for {target_node}")

            # SWE-Smith 风格：使用系统的 code_before 生成 patch
            logger.info(f"SWE-Smith 风格 patch 生成: {file_path}:{line_start}-{line_end}")
            injection_patch = self._build_patch_system(file_path, code_before, code_after, line_start)

            # P0: 验证 patch 语义
            is_valid, validation_error = self._validate_patch_semantics(injection_patch, code_before, code_after)

            if not is_valid:
                error_msg = f"PATCH VALIDATION FAILED: {validation_error}\n\nPlease provide code_after with ACTUAL CODE CHANGES."
                logger.warning(f"Patch validation failed: {validation_error}")
                self.conversation_history.append(AgentTurn(
                    turn_number=len(self.conversation_history) + 1,
                    thought="My patch was rejected - only contained comments/trivial changes.",
                    action="generate_bug_failed",
                    action_input=action_input,
                    observation=error_msg
                ))
                return None

            logger.info("Patch validation passed - contains meaningful code changes")

            # fix_patch: 从 buggy 状态变成 fixed 状态
            fix_patch = self._build_patch_difflib(file_path, code_after, code_before, line_start)

            if not injection_patch.strip():
                error_msg = (
                    f"CRITICAL ERROR: The generated patch is empty for {file_path}.\n"
                    "Please provide a `code_after` that actually introduces a bug."
                )
                logger.warning(error_msg)
                self.conversation_history.append(AgentTurn(
                    turn_number=len(self.conversation_history) + 1,
                    thought="The patch was empty. I need to rethink my injection logic.",
                    action="generate_bug_failed",
                    action_input=action_input,
                    observation=error_msg
                ))
                return None

            # --- 确定 repo 信息 ---
            repo = self.current_seed.seed_metadata.get("repo", "unknown")
            base_commit = self.current_seed.seed_metadata.get("base_commit", "unknown")

            # 获取 Seed 链路目标深度
            seed_chains_for_depth = self.current_seed.mined_data.get("extracted_chains", [])
            target_depth = 3
            if seed_chains_for_depth:
                nodes = seed_chains_for_depth[0].get("nodes", []) if isinstance(seed_chains_for_depth[0], dict) else seed_chains_for_depth[0].nodes
                target_depth = len(nodes)

            # 解析测试用例
            test_code = (
                action_input.get("test_case_code") or
                action_input.get("fail_to_pass_code") or
                action_input.get("failing_test_case") or
                action_input.get("test_code") or
                ""
            )

            fail_to_pass = action_input.get("FAIL_TO_PASS") or action_input.get("fail_to_pass")
            pass_to_pass = action_input.get("PASS_TO_PASS") or action_input.get("pass_to_pass") or []

            # 生成 test_patch 和正确的 FAIL_TO_PASS
            test_patch = ""
            test_file_path = ""
            test_class_name = ""
            test_method_name = ""

            # 从 Agent 获取预期失败的现有测试
            expected_tests_to_fail = (
                action_input.get("expected_tests_to_fail") or
                action_input.get("FAIL_TO_PASS") or
                action_input.get("fail_to_pass") or
                []
            )
            expected_failure_behavior = action_input.get("expected_failure_behavior", "")

            def _normalize_test_list(value) -> list[str]:
                if value is None:
                    return []
                if isinstance(value, list):
                    out: list[str] = []
                    for x in value:
                        if x is None:
                            continue
                        s = str(x).strip()
                        if s:
                            out.append(s)
                    return out
                if isinstance(value, str):
                    s = value.strip()
                    if not s:
                        return []
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return [str(x).strip() for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                    return [s]
                return []

            # Normalize early
            expected_tests_to_fail = _normalize_test_list(expected_tests_to_fail)
            fail_to_pass = _normalize_test_list(fail_to_pass)
            pass_to_pass = _normalize_test_list(pass_to_pass)

            # Auto-fill expected_failure_behavior
            if (
                (not (expected_failure_behavior or "").strip())
                and (not self.generate_synthetic_test)
                and (ctx.get("related_tests") or ctx.get("planned_suite_tests"))
            ):
                related = ctx.get("related_tests", []) or []
                if related:
                    t0 = related[0] or {}
                    fp = (t0.get("full_path", "") or "").strip()
                    a0 = ""
                    assertions = t0.get("assertions", []) or []
                    if assertions and isinstance(assertions, list):
                        a0 = (assertions[0] or "").strip()
                    if fp and a0:
                        expected_failure_behavior = f"Break assertion in `{fp}`: `{a0}`"
                    elif fp:
                        expected_failure_behavior = f"Break behavior checked by `{fp}`"
                    else:
                        expected_failure_behavior = "Cause at least one planned suite test to fail."
                else:
                    expected_failure_behavior = "Cause at least one planned suite test to fail."
                logger.info(f"Auto-filled expected_failure_behavior: {expected_failure_behavior}")

            # 仅在配置启用时生成合成测试
            if self.generate_synthetic_test and test_code:
                test_patch, test_file_path, test_class_name, test_method_name = self._generate_test_patch(
                    test_code=test_code,
                    target_node=target_node,
                    file_path=file_path,
                    instance_id=f"synthetic_{repo.replace('/', '_')[:20]}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    expected_failure_reason=action_input.get("expected_failure_reason", ""),
                    bug_description=bug_description,
                )
                logger.info(f"生成合成测试 patch: {test_file_path}::{test_class_name}::{test_method_name}")
            else:
                logger.info("跳过合成测试生成，将使用仓库现有测试进行验证")

            # P1: 验证 expected_tests_to_fail
            if expected_tests_to_fail and isinstance(expected_tests_to_fail, list):
                valid_tests, invalid_tests = self._validate_expected_tests(expected_tests_to_fail)
                if invalid_tests:
                    logger.warning(f"Filtered invalid/placeholder tests: {invalid_tests}")
                if not valid_tests:
                    related = ctx.get('related_tests', [])
                    if related:
                        valid_tests = [t['full_path'] for t in related[:3]]
                        logger.info(f"Using auto-discovered related tests: {valid_tests}")
                allowed_suite = ctx.get("allowed_fail_test_case_ids", []) or ctx.get("allowed_test_case_ids", []) or []
                constrained = self._filter_tests_to_allowed_suite(valid_tests, allowed_suite)
                if constrained != valid_tests:
                    dropped = [t for t in valid_tests if t not in constrained]
                    if dropped:
                        logger.warning(f"Dropped tests outside planned suite: {dropped}")
                expected_tests_to_fail = constrained

            # Auto-select expected_tests_to_fail if empty
            if (
                (not expected_tests_to_fail)
                and (not self.generate_synthetic_test)
                and (ctx.get("planned_suite_tests") or ctx.get("related_tests"))
            ):
                allowed_suite = ctx.get("allowed_fail_test_case_ids", []) or ctx.get("allowed_test_case_ids", []) or []
                related = ctx.get("related_tests", []) or []
                auto = [t.get("full_path", "") for t in related[:3] if t.get("full_path")] if related else []
                auto = self._filter_tests_to_allowed_suite(auto, allowed_suite)
                if not auto and allowed_suite:
                    auto = [t for t in allowed_suite[:3] if isinstance(t, str) and t.strip()]
                expected_tests_to_fail = auto
                logger.info(f"Auto-selected expected_tests_to_fail: {expected_tests_to_fail}")

            # 构建 FAIL_TO_PASS
            if expected_tests_to_fail and isinstance(expected_tests_to_fail, list):
                fail_to_pass = expected_tests_to_fail
            elif test_file_path and test_class_name and test_method_name:
                fail_to_pass = [f"{test_file_path}::{test_class_name}::{test_method_name}"]
            elif not fail_to_pass:
                fail_to_pass = []
                logger.info("FAIL_TO_PASS 留空，将从验证日志中自动发现失败测试")

            if isinstance(fail_to_pass, str):
                try: fail_to_pass = json.loads(fail_to_pass)
                except (json.JSONDecodeError, ValueError): fail_to_pass = [fail_to_pass]

            # Constrain PASS_TO_PASS to planned suite
            if pass_to_pass and isinstance(pass_to_pass, list):
                allowed_suite = ctx.get("allowed_pass_test_case_ids", []) or ctx.get("allowed_test_case_ids", []) or []
                constrained_p2p = self._filter_tests_to_allowed_suite(pass_to_pass, allowed_suite)
                if constrained_p2p != pass_to_pass:
                    dropped = [t for t in pass_to_pass if t not in constrained_p2p]
                    if dropped:
                        logger.warning(f"Dropped PASS_TO_PASS outside planned suite: {dropped}")
                pass_to_pass = constrained_p2p

            # F2P/P2P 不相交约束: 同一测试不可能既是 expected fail 又是 expected pass
            if pass_to_pass and fail_to_pass:
                f2p_set = set(fail_to_pass)
                overlap = [t for t in pass_to_pass if t in f2p_set]
                if overlap:
                    logger.warning(f"Removed {len(overlap)} tests from PASS_TO_PASS that overlap with FAIL_TO_PASS: {overlap}")
                    pass_to_pass = [t for t in pass_to_pass if t not in f2p_set]
            if (not pass_to_pass) and (ctx.get("planned_suite_tests")):
                candidates = [t for t in (ctx.get("planned_suite_tests") or []) if t and t not in (fail_to_pass or [])]
                if candidates:
                    pass_to_pass = candidates[:5]
                    logger.info(f"Auto-selected PASS_TO_PASS from planned suite: {pass_to_pass}")

            result = SynthesisResult(
                instance_id=f"synthetic_{repo.replace('/', '_')[:20]}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                repo=repo,
                base_commit=base_commit,
                problem_statement=bug_description,
                patch=fix_patch,
                test_patch=test_patch,
                FAIL_TO_PASS=fail_to_pass,
                PASS_TO_PASS=pass_to_pass if isinstance(pass_to_pass, list) else [pass_to_pass],
                seed_id=seed_id,
                fix_intent=fix_intent,
                injection_strategy=injection_strategy,
                seed_metadata=self.current_seed.seed_metadata if self.current_seed else {},
                seed_extraction_chains=self.current_seed.mined_data.get("extracted_chains", []) if self.current_seed else [],
                metadata={
                    "proposed_chain": proposed_chain,
                    "target_node": target_node,
                    "injection_patch": injection_patch,
                    "code_transformation": {"before": code_before, "after": code_after, "raw_after": raw_code_after},
                    "test_case_code": test_code,
                    "test_file_path": test_file_path,
                    "planned_test_cmd": (ctx.get("planned_test_cmd", "") if isinstance(ctx, dict) else ""),
                    "planned_test_modules": (ctx.get("planned_test_modules", []) if isinstance(ctx, dict) else []),
                    "planned_test_files": (ctx.get("planned_test_files", []) if isinstance(ctx, dict) else []),
                    "expected_failure_reason": action_input.get("expected_failure_reason", ""),
                    "expected_failure_behavior": expected_failure_behavior,
                    "expected_tests_to_fail": expected_tests_to_fail if isinstance(expected_tests_to_fail, list) else [],
                    "start_line": line_start,
                    "end_line": line_end,
                    "ps_level": getattr(self, "_ps_level", "standard"),
                    "ps_quality_ok": ps_ok,
                    "ps_quality_reason": ps_reason,
                    "timestamp": datetime.now().isoformat()
                }
            )

            logger.info(f"成功生成合成结果: {file_path}:{line_start}-{line_end}")
            return result

        except Exception as e:
            logger.error(f"Bug 生成失败: {e}")
            logger.error(traceback.format_exc())
            return None

    def _record_llm_call(
        self,
        turn_num: int,
        prompt: str,
        response: str,
        duration: float
    ) -> None:
        """
        Record LLM call for tracking and debugging.
        """
        call_record = {
            "turn": turn_num,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 3),
            "model": getattr(self.llm, "model", "unknown")
        }
        self.llm_calls.append(call_record)

    def _save_agent_trace(self) -> None:
        """
        Save the agent's conversation trace to a per-instance JSON file.

        Called once at the end of the conversation loop (success, failure,
        or max-turns). Writes directly without reading existing data.
        """
        if not self.output_dir:
            return

        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        instance_id = self.current_seed.instance_id if self.current_seed else "unknown"

        try:
            simplified_trace = []
            for turn in self.conversation_history:
                t_num = getattr(turn, 'turn_number', turn.get('turn_number') if isinstance(turn, dict) else 'unknown')
                t_action = getattr(turn, 'action', turn.get('action') if isinstance(turn, dict) else 'unknown')
                t_obs = getattr(turn, 'observation', turn.get('observation') if isinstance(turn, dict) else '')

                simplified_trace.append({
                    "turn": t_num,
                    "tool": t_action,
                    "reply": t_obs[:1000] + "..." if len(t_obs) > 1000 else t_obs
                })

            full_trace = []
            for turn in self.conversation_history:
                if isinstance(turn, dict):
                    full_trace.append(turn)
                else:
                    full_trace.append({
                        "turn": turn.turn_number,
                        "thought": turn.thought,
                        "action": turn.action,
                        "action_input": turn.action_input,
                        "observation": turn.observation
                    })

            trace_content = {
                "instance_id": instance_id,
                "system_prompt": self.system_prompt,
                "simplified_trace": simplified_trace,
                "full_trace": full_trace
            }

            # Per-instance file: no read-merge-write overhead
            safe_id = instance_id.replace("/", "_").replace("\\", "_")
            output_file = logs_dir / f"agent_trace_{safe_id}.json"
            with open(output_file, "w") as f:
                json.dump(trace_content, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save agent trace: {e}")


