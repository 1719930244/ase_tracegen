"""
Fix Intent Pattern Matcher - 基于语义模式匹配候选注入点

核心思想：
- 从 Seed 的 code_transformation 提取代码模式
- 在候选代码中寻找相同模式的位置
- 返回精确的注入点建议

This module ensures that the generated bug is semantically equivalent to the seed's fix intent.
"""

import re
import difflib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class InjectionPoint:
    """表示一个可注入的代码位置"""
    line_start: int
    line_end: int
    code_snippet: str
    pattern_type: str  # e.g., "regex_literal", "condition_expr", "function_call"
    match_score: float  # 与 seed 模式的匹配度 (0-1)
    suggested_change: str  # 建议的修改描述
    semantic_category: str = ""  # 语义类别，如 "regex_terminator"


@dataclass
class SeedPattern:
    """从 Seed 提取的模式信息"""
    intent_type: str
    pattern_type: str
    before_pattern: str
    after_pattern: str
    key_change: Dict[str, str]
    semantic_category: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_type": self.intent_type,
            "pattern_type": self.pattern_type,
            "before_pattern": self.before_pattern,
            "after_pattern": self.after_pattern,
            "key_change": self.key_change,
            "semantic_category": self.semantic_category
        }


class FixIntentPatternMatcher:
    """
    基于 Fix Intent 的语义模式匹配器

    功能：
    1. 从 Seed 的 code_transformation 提取代码模式
    2. 在候选代码中寻找相同模式
    3. 返回精确的注入点和建议修改
    """

    # 模式类型定义 - 每种 intent 对应的代码模式
    PATTERN_TYPES = {
        "Constant_Update": [
            ("regex_literal", r"r['\"].*?['\"]"),  # Python raw string (regex)
            ("string_literal", r"['\"][^'\"]+['\"]"),  # 普通字符串
            ("numeric_literal", r"\b\d+\.?\d*\b"),  # 数字 (包括浮点数)
            # 新增：方法调用链
            ("method_chain", r"\.\w+\([^)]*\)\.\w+\("),
            ("strip_call", r"\.strip\s*\([^)]*\)"),
            ("sub_call", r"re\.sub\s*\("),
        ],
        "Condition_Refinement": [
            ("if_condition", r"if\s+(.+?)\s*:"),
            ("elif_condition", r"elif\s+(.+?)\s*:"),
            ("while_condition", r"while\s+(.+?)\s*:"),
            ("ternary", r".+\s+if\s+.+\s+else\s+.+"),
        ],
        "Guard_Clause_Addition": [
            ("none_check", r"if\s+\w+\s+is\s+(not\s+)?None"),
            ("empty_check", r"if\s+(not\s+)?\w+\s*:"),
            ("isinstance_check", r"isinstance\s*\([^)]+\)"),
            ("hasattr_check", r"hasattr\s*\([^)]+\)"),
        ],
        "Argument_Update": [
            ("function_call_args", r"\w+\s*\([^)]+\)"),
            ("keyword_arg", r"\w+\s*=\s*[^,)]+"),
            ("method_call_args", r"\.\w+\s*\([^)]+\)"),
        ],
        "API_Replacement": [
            ("method_call", r"\.\w+\s*\("),
            ("module_function", r"\b\w+\.\w+\s*\("),
            ("builtin_function", r"\b(open|print|len|range|str|int|float|list|dict|set)\s*\("),
        ],
        "Exception_Fix": [
            ("try_block", r"try\s*:"),
            ("except_clause", r"except\s+(\w+(\s*,\s*\w+)*)?(\s+as\s+\w+)?\s*:"),
            ("raise_stmt", r"raise\s+\w+"),
        ],
        "Variable_Replacement": [
            ("variable_assignment", r"\b([a-z_][a-z0-9_]*)\s*="),
            ("variable_usage", r"\b([a-z_][a-z0-9_]*)\b"),
            ("attribute_access", r"self\.\w+"),
        ],
        "Type_Cast_Fix": [
            # 原有模式：基础类型转换
            ("type_cast", r"\b(int|str|float|bool|list|dict|tuple|set)\s*\("),
            ("type_annotation", r":\s*(int|str|float|bool|List|Dict|Tuple|Set)"),
            # 新增：函数式包装模式 (wraps, partial 等)
            ("wrapper_func", r"\b(wraps|partial|functools\.wraps|functools\.partial)\s*\("),
            ("decorator_wrapper", r"@functools\.\w+"),
            ("method_get", r"\.__get__\s*\("),
            # 通用函数调用包装 func1(func2(...))
            ("func_wrapper", r"\w+\s*\(\s*\w+\s*\("),
        ],
        "Data_Initialization": [
            # 原有模式
            ("assignment_none", r"\w+\s*=\s*None"),
            ("assignment_empty", r"\w+\s*=\s*(\[\]|\{\}|\(\)|'')"),
            ("self_init", r"self\.\w+\s*="),
            ("default_param", r"def\s+\w+\s*\([^)]*\w+\s*=\s*[^,)]+"),
            # 新增：字典字面量模式
            ("dict_literal", r"\{[^}]*:[^}]*\}"),
            ("dict_key_value", r"'\w+':\s*\w+"),
            ("dict_update", r"\.update\s*\("),
        ],
        "Statement_Insertion": [
            ("return_stmt", r"return\s+"),
            ("break_continue", r"\b(break|continue)\b"),
            ("assignment", r"\w+\s*=\s*.+"),
        ],
        "Complex_Logic_Rewrite": [
            ("multi_line_block", r".*"),  # 通用匹配
        ],
    }

    # 语义类别关键词
    SEMANTIC_KEYWORDS = {
        "regex_terminator_end": [r"\$", r"\\Z", r"\\z"],
        "regex_terminator_start": [r"\^", r"\\A"],
        "regex_pattern": [r"\\w", r"\\d", r"\\s", r"\+", r"\*", r"\?"],
        "null_safety": ["None", "is None", "is not None"],
        "type_check": ["isinstance", "type("],
        "boundary_check": [">", "<", ">=", "<=", "==", "!="],
    }

    def __init__(self):
        pass

    def extract_seed_pattern(self, fix_intent: Dict[str, Any]) -> SeedPattern:
        """
        从 Seed 的 fix_intent 提取代码模式

        Args:
            fix_intent: 包含 type, code_transformation 等字段的字典

        Returns:
            SeedPattern 对象
        """
        intent_type = fix_intent.get("type", "Unknown")
        code_trans = fix_intent.get("code_transformation", {})
        before_code = code_trans.get("before", "")
        after_code = code_trans.get("after", "")

        if not before_code or not after_code:
            return SeedPattern(
                intent_type=intent_type,
                pattern_type="unknown",
                before_pattern="",
                after_pattern="",
                key_change={},
                semantic_category=""
            )

        # 识别模式类型
        pattern_type = self._identify_pattern_type(intent_type, before_code)

        # 提取核心变化点
        key_change = self._extract_key_change(before_code, after_code)

        # 识别语义类别
        semantic_category = self._identify_semantic_category(
            intent_type, before_code, after_code, key_change
        )

        return SeedPattern(
            intent_type=intent_type,
            pattern_type=pattern_type,
            before_pattern=before_code,
            after_pattern=after_code,
            key_change=key_change,
            semantic_category=semantic_category
        )

    def _identify_pattern_type(self, intent_type: str, code: str) -> str:
        """识别代码的模式类型"""
        patterns = self.PATTERN_TYPES.get(intent_type, [])
        for pattern_name, regex in patterns:
            if re.search(regex, code):
                return pattern_name
        return "unknown"

    def _extract_key_change(self, before: str, after: str) -> Dict[str, str]:
        """提取 before 和 after 之间的核心差异"""
        matcher = difflib.SequenceMatcher(None, before, after)
        changes = {"from": "", "to": ""}

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "replace":
                changes["from"] = before[i1:i2]
                changes["to"] = after[j1:j2]
                break
            elif op == "delete":
                changes["from"] = before[i1:i2]
                changes["to"] = ""
                break
            elif op == "insert":
                changes["from"] = ""
                changes["to"] = after[j1:j2]
                break

        return changes

    def _identify_semantic_category(
        self,
        intent_type: str,
        before: str,
        after: str,
        key_change: Dict[str, str]
    ) -> str:
        """识别语义类别，用于更精确的匹配"""

        if intent_type == "Constant_Update":
            # 识别 regex 相关的修改
            if re.search(r"r['\"].*['\"]", before):
                change_from = key_change.get("from", "")
                change_to = key_change.get("to", "")

                # 检查是否是 regex 终止符相关
                if "$" in change_from or r"\Z" in change_to:
                    return "regex_terminator_end"
                elif "^" in change_from or r"\A" in change_to:
                    return "regex_terminator_start"
                elif any(kw in before for kw in [r"\w", r"\d", r"\s"]):
                    return "regex_pattern"
                else:
                    return "regex_other"
            elif re.search(r"\d+", before):
                return "numeric_constant"
            else:
                return "string_constant"

        elif intent_type == "Condition_Refinement":
            if ">=" in before or "<=" in before or ">" in before or "<" in before:
                return "comparison_operator"
            elif "and" in before or "or" in before:
                return "logical_operator"
            elif "not" in before:
                return "negation"
            else:
                return "condition_general"

        elif intent_type == "Guard_Clause_Addition":
            if "None" in before or "None" in after:
                return "null_check"
            elif "isinstance" in before or "isinstance" in after:
                return "type_check"
            else:
                return "validation_check"

        return intent_type.lower()

    def find_injection_points(
        self,
        candidate_code: str,
        seed_pattern: SeedPattern,
        file_path: str = ""
    ) -> List[InjectionPoint]:
        """
        在候选代码中寻找与 Seed 模式匹配的注入点

        Args:
            candidate_code: 候选节点的源代码
            seed_pattern: 从 Seed 提取的模式信息
            file_path: 文件路径（用于日志）

        Returns:
            List[InjectionPoint]: 按匹配度排序的注入点列表
        """
        intent_type = seed_pattern.intent_type
        pattern_type = seed_pattern.pattern_type
        semantic_category = seed_pattern.semantic_category
        key_change = seed_pattern.key_change

        injection_points = []
        lines = candidate_code.split("\n")

        # 根据 intent_type 选择匹配策略
        if intent_type == "Constant_Update":
            injection_points = self._find_constant_injection_points(
                lines, pattern_type, semantic_category, key_change
            )
        elif intent_type == "Condition_Refinement":
            injection_points = self._find_condition_injection_points(
                lines, semantic_category
            )
        elif intent_type == "Guard_Clause_Addition":
            injection_points = self._find_guard_injection_points(
                lines, semantic_category
            )
        elif intent_type == "Argument_Update":
            injection_points = self._find_argument_injection_points(lines)
        elif intent_type == "API_Replacement":
            injection_points = self._find_api_injection_points(lines)
        elif intent_type == "Exception_Fix":
            injection_points = self._find_exception_injection_points(lines)
        elif intent_type == "Variable_Replacement":
            injection_points = self._find_variable_injection_points(lines)
        elif intent_type == "Type_Cast_Fix":
            injection_points = self._find_type_cast_injection_points(lines)
        elif intent_type == "Data_Initialization":
            injection_points = self._find_initialization_injection_points(lines)
        else:
            # 通用匹配
            injection_points = self._find_generic_injection_points(
                lines, intent_type
            )

        # 按匹配度排序
        injection_points.sort(key=lambda x: x.match_score, reverse=True)

        return injection_points

    def _find_constant_injection_points(
        self,
        lines: List[str],
        pattern_type: str,
        semantic_category: str,
        key_change: Dict[str, str]
    ) -> List[InjectionPoint]:
        """寻找常量更新类型的注入点"""
        points = []

        for i, line in enumerate(lines):
            # 检查是否包含与 seed 相同类型的常量
            if pattern_type == "regex_literal":
                # 寻找 regex 模式 (r'...' 或 r"...")
                match = re.search(r"(r['\"][^'\"]*['\"])", line)
                if match:
                    regex_str = match.group(1)
                    score = 0.5  # 基础分：找到了 regex
                    suggested = "Modify regex pattern"

                    # 检查语义类别是否匹配
                    if semantic_category == "regex_terminator_end":
                        # Seed 是修改 regex 结尾的 $ -> \Z
                        if r"\Z" in regex_str:
                            score = 0.95  # 完美匹配：可以改回 $
                            suggested = "Change \\Z back to $ in regex terminator"
                        elif "$" in regex_str:
                            score = 0.85  # 高匹配：已经有 $ 但可以做类似修改
                            suggested = "Modify regex terminator pattern"
                        elif regex_str.endswith("'") or regex_str.endswith('"'):
                            score = 0.6  # 中等：有 regex 但没有终止符
                            suggested = "Consider adding/modifying regex terminator"

                    elif semantic_category == "regex_terminator_start":
                        if r"\A" in regex_str:
                            score = 0.95
                            suggested = "Change \\A back to ^ in regex"
                        elif "^" in regex_str:
                            score = 0.85
                            suggested = "Modify regex start anchor"

                    elif semantic_category == "regex_pattern":
                        score = 0.7
                        suggested = "Modify regex character class or quantifier"

                    points.append(InjectionPoint(
                        line_start=i + 1,
                        line_end=i + 1,
                        code_snippet=line.strip(),
                        pattern_type=pattern_type,
                        match_score=score,
                        suggested_change=suggested,
                        semantic_category=semantic_category
                    ))

            elif pattern_type == "string_literal":
                # 寻找普通字符串（排除 docstring）
                if not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    match = re.search(r"(['\"][^'\"]+['\"])", line)
                    if match and "=" in line:  # 只匹配赋值语句中的字符串
                        points.append(InjectionPoint(
                            line_start=i + 1,
                            line_end=i + 1,
                            code_snippet=line.strip(),
                            pattern_type=pattern_type,
                            match_score=0.6,
                            suggested_change="Modify string literal value",
                            semantic_category=semantic_category
                        ))

            elif pattern_type == "numeric_literal":
                # 寻找数字常量
                match = re.search(r"=\s*(\d+\.?\d*)", line)
                if match:
                    points.append(InjectionPoint(
                        line_start=i + 1,
                        line_end=i + 1,
                        code_snippet=line.strip(),
                        pattern_type=pattern_type,
                        match_score=0.6,
                        suggested_change="Modify numeric constant (e.g., off-by-one)",
                        semantic_category=semantic_category
                    ))

        return points

    def _find_condition_injection_points(
        self,
        lines: List[str],
        semantic_category: str
    ) -> List[InjectionPoint]:
        """寻找条件语句注入点"""
        points = []
        for i, line in enumerate(lines):
            # if/elif/while 语句
            if re.search(r"\b(if|elif|while)\s+.+:", line):
                score = 0.7
                suggested = "Modify condition logic"

                # 根据语义类别调整分数
                if semantic_category == "comparison_operator":
                    if re.search(r"[<>=!]=?", line):
                        score = 0.9
                        suggested = "Change comparison operator (e.g., > to >=)"
                elif semantic_category == "logical_operator":
                    if " and " in line or " or " in line:
                        score = 0.9
                        suggested = "Modify logical operator (and/or)"
                elif semantic_category == "negation":
                    if " not " in line:
                        score = 0.9
                        suggested = "Remove or add negation"

                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="condition",
                    match_score=score,
                    suggested_change=suggested,
                    semantic_category=semantic_category
                ))
        return points

    def _find_guard_injection_points(
        self,
        lines: List[str],
        semantic_category: str
    ) -> List[InjectionPoint]:
        """寻找卫语句注入点"""
        points = []
        for i, line in enumerate(lines):
            score = 0.6
            suggested = "Remove or bypass guard clause"
            pattern_type = "guard_clause"

            if re.search(r"if\s+\w+\s+is\s+(not\s+)?None", line):
                if semantic_category == "null_check":
                    score = 0.9
                suggested = "Remove None check guard"
                pattern_type = "none_check"
            elif re.search(r"isinstance\s*\(", line):
                if semantic_category == "type_check":
                    score = 0.9
                suggested = "Remove isinstance check"
                pattern_type = "isinstance_check"
            elif re.search(r"if\s+not\s+\w+", line):
                score = 0.7
                suggested = "Remove empty/falsy check"
                pattern_type = "empty_check"
            elif re.search(r"hasattr\s*\(", line):
                score = 0.7
                suggested = "Remove hasattr check"
                pattern_type = "hasattr_check"
            else:
                continue  # 不匹配任何卫语句模式

            points.append(InjectionPoint(
                line_start=i + 1,
                line_end=i + 1,
                code_snippet=line.strip(),
                pattern_type=pattern_type,
                match_score=score,
                suggested_change=suggested,
                semantic_category=semantic_category
            ))
        return points

    def _find_argument_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找函数参数注入点"""
        points = []
        for i, line in enumerate(lines):
            # 带参数的函数调用
            if re.search(r"\w+\s*\([^)]+\)", line):
                # 排除函数定义
                if not line.strip().startswith("def "):
                    points.append(InjectionPoint(
                        line_start=i + 1,
                        line_end=i + 1,
                        code_snippet=line.strip(),
                        pattern_type="function_call",
                        match_score=0.7,
                        suggested_change="Modify function arguments",
                        semantic_category="argument"
                    ))
        return points

    def _find_api_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找 API 调用注入点"""
        points = []
        for i, line in enumerate(lines):
            # 方法调用 (object.method())
            if re.search(r"\.\w+\s*\(", line):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="method_call",
                    match_score=0.75,
                    suggested_change="Replace with incorrect/deprecated API",
                    semantic_category="api"
                ))
            # 模块函数调用 (module.function())
            elif re.search(r"\b\w+\.\w+\s*\(", line):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="module_function",
                    match_score=0.7,
                    suggested_change="Use incorrect module function",
                    semantic_category="api"
                ))
        return points

    def _find_exception_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找异常处理注入点"""
        points = []
        in_try_block = False
        try_start = 0
        try_lines = []

        for i, line in enumerate(lines):
            if "try:" in line:
                in_try_block = True
                try_start = i
                try_lines = [line]
            elif in_try_block:
                try_lines.append(line)
                if "except" in line:
                    points.append(InjectionPoint(
                        line_start=try_start + 1,
                        line_end=i + 1,
                        code_snippet=line.strip(),
                        pattern_type="exception_handler",
                        match_score=0.8,
                        suggested_change="Modify exception type or remove handler",
                        semantic_category="exception"
                    ))
                    in_try_block = False
        return points

    def _find_variable_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找变量替换注入点"""
        points = []
        for i, line in enumerate(lines):
            # 变量赋值
            match = re.search(r"\b([a-z_][a-z0-9_]*)\s*=\s*", line)
            if match and not line.strip().startswith("def "):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="variable_assignment",
                    match_score=0.6,
                    suggested_change="Use wrong variable name",
                    semantic_category="variable"
                ))
        return points

    def _find_type_cast_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找类型转换注入点"""
        points = []
        for i, line in enumerate(lines):
            if re.search(r"\b(int|str|float|bool|list|dict)\s*\(", line):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="type_cast",
                    match_score=0.8,
                    suggested_change="Remove type cast or use wrong type",
                    semantic_category="type_cast"
                ))
        return points

    def _find_initialization_injection_points(self, lines: List[str]) -> List[InjectionPoint]:
        """寻找初始化注入点"""
        points = []
        for i, line in enumerate(lines):
            # self.xxx = 初始化
            if re.search(r"self\.\w+\s*=", line):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="self_init",
                    match_score=0.75,
                    suggested_change="Remove or corrupt initialization",
                    semantic_category="initialization"
                ))
            # xxx = None/[]/{}
            elif re.search(r"\w+\s*=\s*(None|\[\]|\{\}|''|\"\")", line):
                points.append(InjectionPoint(
                    line_start=i + 1,
                    line_end=i + 1,
                    code_snippet=line.strip(),
                    pattern_type="empty_init",
                    match_score=0.7,
                    suggested_change="Change initialization value",
                    semantic_category="initialization"
                ))
        return points

    def _find_generic_injection_points(
        self,
        lines: List[str],
        intent_type: str
    ) -> List[InjectionPoint]:
        """通用注入点查找 - 作为后备"""
        points = []
        # 返回整个代码块作为一个低匹配度注入点
        if lines:
            points.append(InjectionPoint(
                line_start=1,
                line_end=len(lines),
                code_snippet=lines[0].strip() if lines else "",
                pattern_type="generic",
                match_score=0.3,
                suggested_change=f"Apply {intent_type} transformation (generic)",
                semantic_category="generic"
            ))
        return points


def check_intent_compatibility(
    candidate_code: str,
    seed_fix_intent: Dict[str, Any],
    min_match_score: float = 0.4
) -> Tuple[bool, List[InjectionPoint], str]:
    """
    三级 Fallback 兼容性检查

    Level 1: 精确模式匹配 (正则)
    Level 2: 语义关键词匹配 (从 code_transformation 提取)
    Level 3: 通用 Fallback (如果代码非空)

    Args:
        candidate_code: 候选节点的源代码
        seed_fix_intent: Seed 的 fix intent 字典
        min_match_score: 最低匹配分数阈值

    Returns:
        (is_compatible, injection_points, reason)
    """
    matcher = FixIntentPatternMatcher()

    # 1. 提取 Seed 模式
    seed_pattern = matcher.extract_seed_pattern(seed_fix_intent)

    # === Level 1: 精确模式匹配 ===
    if seed_pattern.pattern_type != "unknown":
        injection_points = matcher.find_injection_points(candidate_code, seed_pattern)
        valid_points = [p for p in injection_points if p.match_score >= min_match_score]

        if valid_points:
            return True, valid_points, (
                f"[Level 1] 精确匹配: {len(valid_points)} 个注入点, "
                f"模式 '{seed_pattern.pattern_type}'"
            )

    # === Level 2: 语义关键词匹配 ===
    keywords = _extract_keywords_from_transformation(seed_fix_intent)
    if keywords:
        keyword_points = _find_keyword_injection_points(
            candidate_code, keywords, seed_pattern.intent_type
        )
        valid_keyword_points = [p for p in keyword_points if p.match_score >= min_match_score]

        if valid_keyword_points:
            return True, valid_keyword_points, (
                f"[Level 2] 关键词匹配: {len(valid_keyword_points)} 个注入点, "
                f"关键词 {keywords[:3]}"
            )

    # === Level 3: 通用 Fallback ===
    # 如果候选代码非空且足够长，返回一个低置信度的通用注入点
    lines = candidate_code.split('\n')
    if len(lines) >= 3 and len(candidate_code) >= 50:
        generic_point = InjectionPoint(
            line_start=1,
            line_end=len(lines),
            code_snippet=lines[0].strip()[:80] if lines else "",
            pattern_type="fallback",
            match_score=0.35,
            suggested_change=f"Apply {seed_pattern.intent_type} transformation (fallback)",
            semantic_category="fallback"
        )
        return True, [generic_point], (
            f"[Level 3] Fallback 匹配: 代码 {len(lines)} 行, "
            f"Intent '{seed_pattern.intent_type}'"
        )

    return False, [], (
        f"三级匹配均失败: pattern='{seed_pattern.pattern_type}', "
        f"keywords={keywords[:3] if keywords else '[]'}"
    )


def _extract_keywords_from_transformation(fix_intent: Dict[str, Any]) -> List[str]:
    """从 code_transformation 提取关键词"""
    ct = fix_intent.get("code_transformation", {})
    before = ct.get("before", "")
    after = ct.get("after", "")

    # 提取标识符
    identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', before + after))

    # 过滤常见关键字和短标识符
    python_keywords = {
        'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'import',
        'from', 'as', 'try', 'except', 'with', 'and', 'or', 'not', 'in',
        'is', 'None', 'True', 'False', 'self', 'lambda', 'yield', 'pass',
        'break', 'continue', 'raise', 'finally', 'global', 'nonlocal', 'assert'
    }

    keywords = [
        kw for kw in identifiers
        if kw.lower() not in python_keywords and len(kw) > 2
    ]

    # 按长度排序，优先匹配较长的标识符
    return sorted(keywords, key=len, reverse=True)


def _find_keyword_injection_points(
    code: str,
    keywords: List[str],
    intent_type: str
) -> List[InjectionPoint]:
    """基于关键词寻找注入点"""
    points = []
    lines = code.split('\n')

    for i, line in enumerate(lines):
        # 跳过注释行和空行
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        matched_keywords = [kw for kw in keywords if kw in line]
        if matched_keywords:
            # 根据匹配关键词数量计算分数
            score = min(0.75, 0.45 + 0.1 * len(matched_keywords))
            points.append(InjectionPoint(
                line_start=i + 1,
                line_end=i + 1,
                code_snippet=stripped[:100],
                pattern_type="keyword_match",
                match_score=score,
                suggested_change=f"Apply {intent_type} using: {matched_keywords[:3]}",
                semantic_category="keyword"
            ))

    # 按分数排序，返回前 5 个
    return sorted(points, key=lambda x: x.match_score, reverse=True)[:5]


def format_injection_points_for_prompt(
    points: List[InjectionPoint],
    max_points: int = 5
) -> str:
    """
    格式化注入点列表供 Prompt 使用

    Args:
        points: 注入点列表
        max_points: 最多显示的点数

    Returns:
        格式化的字符串
    """
    if not points:
        return "No injection points identified."

    lines = []
    for i, p in enumerate(points[:max_points], 1):
        lines.append(f"**Injection Point {i}** (Match Score: {p.match_score:.0%})")
        lines.append(f"  - Line: {p.line_start}")
        lines.append(f"  - Code: `{p.code_snippet[:80]}{'...' if len(p.code_snippet) > 80 else ''}`")
        lines.append(f"  - Pattern Type: {p.pattern_type}")
        lines.append(f"  - Suggested Change: {p.suggested_change}")
        lines.append("")

    if len(points) > max_points:
        lines.append(f"... and {len(points) - max_points} more points")

    return "\n".join(lines)


# Module exports
__all__ = [
    "FixIntentPatternMatcher",
    "InjectionPoint",
    "SeedPattern",
    "check_intent_compatibility",
    "format_injection_points_for_prompt",
]
