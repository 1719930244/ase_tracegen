"""
ValidationMixin - Validation and sanitization methods for SynthesisAgent.

Extracted from agent.py to reduce file size. Contains:
- Patch semantic validation
- Intent alignment validation
- Problem statement leak detection & sanitization
- Comment stripping
- Test filtering
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re
import difflib

from loguru import logger

from .pattern_matcher import SeedPattern


def _python_comment_counter(code: str):
    """Count Python comment tokens in code. Returns a Counter."""
    from collections import Counter
    if not code:
        return Counter()
    import io
    import tokenize
    import textwrap
    dedented = textwrap.dedent(code)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(dedented).readline)
        comments = [t.string.strip() for t in tokens if t.type == tokenize.COMMENT]
        return Counter(comments)
    except Exception:
        c = Counter()
        for ln in code.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                c[s] += 1
        return c


class ValidationMixin:
    """
    Mixin providing validation and sanitization methods for SynthesisAgent.

    These methods validate generated patches, sanitize problem statements,
    and filter test selections.
    """

    # ------------------------------------------------------------------
    # Problem Statement leak detection & sanitization
    # ------------------------------------------------------------------

    _FIX_DIRECTIVE_PATTERNS: list[re.Pattern] = [
        re.compile(p, re.IGNORECASE) for p in [
            r"\bthe fix is\b",
            r"\bfix(?:ed)? by\b",
            r"\bshould (?:be )?(?:changed|replaced|updated|converted|switched) to\b",
            r"\breplace\s+\S+\s+with\b",
            r"\buse\s+\S+\s+instead of\b",
            r"\bshould use\b",
            r"\bneeds to be\b",
            r"\bchange\s+\S+\s+to\b",
            r"\bswitch(?:ing)?\s+from\s+\S+\s+to\b",
        ]
    ]

    # Layer 4: Implicit leak patterns (subtler than fix directives)
    _IMPLICIT_LEAK_PATTERNS: list[re.Pattern] = [
        re.compile(p, re.IGNORECASE) for p in [
            # Points to specific code location (file + line/function)
            r"\bin\s+(?:line|method|function)\s+\w+\s+of\s+\w+\.py\b",
            # Describes what was removed/deleted (reveals the fix action)
            r"\b(?:missing|removed|deleted|dropped)\s+(?:the\s+)?(?:call|check|guard|wrapper|import|conversion|cast)\b",
            # Directly states the bug cause (not just symptoms)
            r"\b(?:the bug is|the issue is caused by|the root cause is|this is because)\b",
            # References specific variable/parameter + modification verb
            r"\b(?:parameter|argument|variable)\s+`?\w+`?\s+(?:should be|was|is)\s+(?:removed|added|changed)\b",
            # Explicitly names the buggy vs fixed code pattern
            r"\b(?:was changed from|changed from|was replaced by|instead of the correct)\b",
        ]
    ]

    def _sanitize_problem_statement(
        self,
        description: str,
        code_before: str,
        code_after: str,
    ) -> str:
        """
        Detect and remove fix-leaking content from bug_description.

        Three layers of detection (inspired by SWE-Smith):
        1. Fix directive phrases ("the fix is to", "should use X instead of Y", etc.)
        2. Verbatim patch lines (exact code from the diff appearing in the description)
        3. Explicit test references ("pytest", "test_xxx failed")

        Returns the sanitized description. Logs warnings for each detected leak.
        """
        import re

        warnings: list[str] = []
        layer1_hit = False
        layer2_hit = False

        # --- Layer 1: Fix directive phrases ---
        for pat in self._FIX_DIRECTIVE_PATTERNS:
            if pat.search(description):
                warnings.append(f"fix_directive: '{pat.pattern}'")
                layer1_hit = True

        # --- Layer 2: Verbatim patch lines ---
        # Extract meaningful changed lines from code_before vs code_after
        if code_before and code_after:
            before_lines = set(l.strip() for l in code_before.splitlines() if len(l.strip()) > 15)
            after_lines = set(l.strip() for l in code_after.splitlines() if len(l.strip()) > 15)
            diff_lines = (after_lines - before_lines) | (before_lines - after_lines)
            for dl in diff_lines:
                # Skip common boilerplate
                if dl.startswith(("def ", "class ", "import ", "from ", "return")):
                    continue
                if dl in description:
                    warnings.append(f"verbatim_code: '{dl[:60]}...'")
                    layer2_hit = True

        # --- Layer 3: Test references ---
        test_patterns = [
            re.compile(r"\bpytest\b", re.IGNORECASE),
            re.compile(r"\btest_\w+\s+fail", re.IGNORECASE),
            re.compile(r"\bexisting tests?\s+fail", re.IGNORECASE),
            re.compile(r"\brunning\s+(?:the\s+)?tests?\b", re.IGNORECASE),
        ]
        for pat in test_patterns:
            if pat.search(description):
                warnings.append(f"test_reference: '{pat.pattern}'")

        # --- Layer 4: Implicit leak patterns ---
        for pat in self._IMPLICIT_LEAK_PATTERNS:
            if pat.search(description):
                warnings.append(f"implicit_leak: '{pat.pattern}'")

        if not warnings:
            return description

        logger.warning(
            f"Problem statement leak detected ({len(warnings)} issues): "
            + "; ".join(warnings)
        )

        # --- Actual sanitization ---
        # Severe leak: Layer 1 (fix directives) + Layer 2 (verbatim code) both triggered.
        # The PS is fundamentally compromised; sentence-level removal is insufficient.
        # Fall back to a symptom-only description.
        if layer1_hit and layer2_hit:
            logger.warning(
                "Severe PS leak (fix directive + verbatim code). "
                "Replacing entire PS with symptom-only fallback."
            )
            # Extract just the first sentence (usually the symptom/title) if possible
            first_sentence_match = re.match(r"^([^.!?\n]{15,}[.!?])", description)
            if first_sentence_match:
                sanitized = first_sentence_match.group(1).strip()
            else:
                sanitized = "Synthetic defect"
            # Verify the fallback doesn't still leak
            still_leaks = any(p.search(sanitized) for p in self._FIX_DIRECTIVE_PATTERNS)
            if still_leaks:
                sanitized = "Synthetic defect"
            logger.info(f"Severe leak fallback PS: {len(sanitized)} chars")
            return sanitized

        sanitized = description

        # Remove sentences containing fix directive phrases
        for pat in self._FIX_DIRECTIVE_PATTERNS:
            # Remove the entire sentence containing the directive
            sanitized = re.sub(
                r"[^.!?\n]*" + pat.pattern + r"[^.!?\n]*[.!?]?\s*",
                "",
                sanitized,
                flags=re.IGNORECASE,
            )

        # Remove verbatim code lines that leaked into the description
        if code_before and code_after:
            before_lines = set(l.strip() for l in code_before.splitlines() if len(l.strip()) > 15)
            after_lines = set(l.strip() for l in code_after.splitlines() if len(l.strip()) > 15)
            diff_lines = (after_lines - before_lines) | (before_lines - after_lines)
            for dl in diff_lines:
                if dl.startswith(("def ", "class ", "import ", "from ", "return")):
                    continue
                sanitized = sanitized.replace(dl, "[CODE]")

        # Remove test references
        test_patterns_to_strip = [
            re.compile(r"[^.!?\n]*\bpytest\b[^.!?\n]*[.!?]?\s*", re.IGNORECASE),
            re.compile(r"[^.!?\n]*\btest_\w+\s+fail[^.!?\n]*[.!?]?\s*", re.IGNORECASE),
            re.compile(r"[^.!?\n]*\bexisting tests?\s+fail[^.!?\n]*[.!?]?\s*", re.IGNORECASE),
            re.compile(r"[^.!?\n]*\brunning\s+(?:the\s+)?tests?\b[^.!?\n]*[.!?]?\s*", re.IGNORECASE),
        ]
        for pat in test_patterns_to_strip:
            sanitized = pat.sub("", sanitized)

        # Remove sentences containing implicit leak patterns (Layer 4)
        for pat in self._IMPLICIT_LEAK_PATTERNS:
            sanitized = re.sub(
                r"[^.!?\n]*" + pat.pattern + r"[^.!?\n]*[.!?]?\s*",
                "",
                sanitized,
                flags=re.IGNORECASE,
            )

        sanitized = sanitized.strip()
        if not sanitized:
            sanitized = "Synthetic defect"

        if sanitized != description:
            logger.info(f"Problem statement sanitized: {len(description)} -> {len(sanitized)} chars")

        return sanitized

    def _preserve_original_docstrings(self, code_before: str, code_after: str) -> tuple[str, int]:
        """
        Replace docstrings in *code_after* with their exact originals from *code_before*.

        Uses AST **only** to locate the ``ast.Expr`` nodes that wrap docstrings;
        the actual replacement is a line-level splice on the original source text,
        so formatting, indentation, and comments are never disturbed.

        Returns:
            (normalized_code_after, number_of_restored_docstrings)
        """
        import ast
        import textwrap

        if not code_before or not code_after:
            return code_after, 0

        # --- helpers -----------------------------------------------------------
        def _docstring_expr(node):
            """Return the ast.Expr node wrapping the docstring, or None."""
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                return node.body[0]
            return None

        def _collect(tree):
            """Qualified-key -> ast.Expr for every docstring in *tree*."""
            result: dict[str, ast.Expr] = {}

            def _visit(node, path=""):
                if isinstance(node, ast.Module):
                    ds = _docstring_expr(node)
                    if ds:
                        result["<module>"] = ds
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        key = (
                            f"{path}/{type(child).__name__}:{child.name}"
                            if path
                            else f"{type(child).__name__}:{child.name}"
                        )
                        ds = _docstring_expr(child)
                        if ds:
                            result[key] = ds
                        _visit(child, key)

            _visit(tree)
            return result

        # --- parse (dedent for AST; line numbers stay 1-1 with original) -------
        try:
            tree_before = ast.parse(textwrap.dedent(code_before))
            tree_after = ast.parse(textwrap.dedent(code_after))
        except SyntaxError:
            return code_after, 0

        before_info = _collect(tree_before)
        after_info = _collect(tree_after)

        before_lines = code_before.splitlines(keepends=True)
        after_lines = code_after.splitlines(keepends=True)

        # --- build replacement list -------------------------------------------
        replacements: list[tuple[int, int, list[str]]] = []
        for key, before_expr in before_info.items():
            if key not in after_info:
                continue  # model removed the node entirely – don't inject
            after_expr = after_info[key]

            b_start = before_expr.lineno - 1          # 0-based inclusive
            b_end   = before_expr.end_lineno           # 0-based exclusive
            a_start = after_expr.lineno - 1
            a_end   = after_expr.end_lineno

            b_raw = before_lines[b_start:b_end]
            a_raw = after_lines[a_start:a_end]

            if b_raw != a_raw:
                replacements.append((a_start, a_end, b_raw))

        if not replacements:
            return code_after, 0

        # Apply bottom-up so earlier indices stay valid
        replacements.sort(key=lambda x: x[0], reverse=True)
        for a_start, a_end, b_raw in replacements:
            after_lines[a_start:a_end] = b_raw

        return "".join(after_lines), len(replacements)

    def _strip_new_comments(self, *, code_before: str, code_after: str) -> tuple[str, int]:
        """
        Remove newly introduced Python comments from `code_after` (best-effort).

        This keeps final patches clean by stripping model-added explanations like:
        - full-line comments: `# Inject bug ...`
        - inline comments: `x = y  # explain`

        Existing comments that already appear in `code_before` are preserved (by comment-string counts).

        Returns:
            (sanitized_code_after, removed_comment_tokens)
        """
        if not code_after or not code_after.strip():
            return code_after, 0

        try:
            from collections import Counter
            import io
            import tokenize
            import textwrap

            before_comments = _python_comment_counter(code_before)
            after_comments = _python_comment_counter(code_after)
            to_remove: Counter = after_comments - before_comments
            if sum(to_remove.values()) <= 0:
                return code_after, 0

            # Tokenize dedented `code_after` to find concrete occurrences (line/col) of comments to remove.
            original_lines = code_after.splitlines(keepends=True)
            dedented = textwrap.dedent(code_after)
            dedented_lines = dedented.splitlines(keepends=True)

            def _leading_ws_len(s: str) -> int:
                i = 0
                while i < len(s) and s[i] in (" ", "\t"):
                    i += 1
                return i

            indent_removed_by_line: list[int] = []
            max_len = max(len(original_lines), len(dedented_lines))
            for i in range(max_len):
                o = original_lines[i] if i < len(original_lines) else ""
                d = dedented_lines[i] if i < len(dedented_lines) else ""
                indent_removed_by_line.append(max(0, _leading_ws_len(o) - _leading_ws_len(d)))

            removed = 0
            # We'll mutate a copy of original_lines in-place.
            out_lines = list(original_lines)

            try:
                tokens = list(tokenize.generate_tokens(io.StringIO(dedented).readline))
            except Exception:
                tokens = []

            # Remove marked comments by token occurrences.
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                comment = tok.string.strip()
                if to_remove.get(comment, 0) <= 0:
                    continue

                line_no, col = tok.start  # 1-based line number in dedented content
                idx = int(line_no) - 1
                if idx < 0 or idx >= len(out_lines):
                    continue

                line = out_lines[idx]
                # Map dedented column back to original column.
                col0 = int(col) + (indent_removed_by_line[idx] if idx < len(indent_removed_by_line) else 0)

                # Split newline suffix.
                nl = "\n" if line.endswith("\n") else ""
                raw = line[:-1] if nl else line

                # Full-line comment (only whitespace before '#'): drop the entire line.
                if raw[:col0].strip() == "":
                    out_lines[idx] = ""
                else:
                    # Inline comment: remove from the comment start to end-of-line.
                    out_lines[idx] = raw[:col0].rstrip() + nl

                to_remove[comment] -= 1
                removed += 1
                if sum(to_remove.values()) <= 0:
                    break

            # If tokenization fails, fallback: drop full-line comments whose stripped form is newly introduced.
            if removed == 0:
                out_lines = []
                to_remove = after_comments - before_comments
                for ln in original_lines:
                    s = ln.strip()
                    if s.startswith("#") and to_remove.get(s, 0) > 0:
                        to_remove[s] -= 1
                        removed += 1
                        continue
                    out_lines.append(ln)

            sanitized = "".join(out_lines)
            return sanitized, removed
        except Exception:
            return code_after, 0

    def _validate_patch_semantics(self, patch: str, code_before: str, code_after: str) -> tuple[bool, str]:
        """验证 patch 是否包含有意义的代码修改。"""
        if not patch or not patch.strip():
            return False, "Patch is empty"

        # Syntax gate: report real syntax errors before docstring comparison,
        # avoiding misleading "docstring changed" when AST parse simply fails.
        if code_after and not self._validate_python_syntax(code_after):
            return False, "code_after has invalid Python syntax"

        # Reject whitespace-only changes (no semantic token changes).
        try:
            if re.sub(r"\s+", "", code_before or "") == re.sub(r"\s+", "", code_after or ""):
                return False, "Patch changes only whitespace/text formatting (forbidden)"
        except (TypeError, re.error) as e:
            logger.debug(f"Whitespace check failed: {e}")
        added_lines = [l[1:].strip() for l in patch.split('\n') if l.startswith('+') and not l.startswith('+++')]
        removed_lines = [l[1:].strip() for l in patch.split('\n') if l.startswith('-') and not l.startswith('---')]

        # Strict: forbid introducing new comments in code_after (including inline comments).
        # Explanations must live in the LLM "Thought", not inside the patched code.
        # [Ablation] disable_quality_controls: 跳过注释和 docstring 检查
        _skip_quality = getattr(self, "_disable_quality_controls", False)
        if not _skip_quality:
            try:
                before_comments = _python_comment_counter(code_before)
                after_comments = _python_comment_counter(code_after)
                newly_introduced = after_comments - before_comments
                if sum(newly_introduced.values()) > 0:
                    example = next(iter(newly_introduced.keys()))
                    return False, (
                        "Patch introduces new comments in code_after (forbidden). "
                        f"Example: {example}. Remove ALL newly added '#' comments from code_after."
                    )
            except Exception:
                # If comment detection fails, continue with other validations.
                pass

            # Strict: forbid docstring changes (pure-text noise) in code_after.
            try:
                import ast
                import textwrap

                def _collect_docstrings(code: str) -> dict[str, str]:
                    """Collect docstrings with qualified-path keys (consistent with _preserve_original_docstrings)."""
                    if not code:
                        return {}
                    try:
                        tree = ast.parse(textwrap.dedent(code))
                    except Exception:
                        return {}

                    docs: dict[str, str] = {}

                    mod_doc = ast.get_docstring(tree, clean=True)
                    if mod_doc is not None:
                        docs["<module>"] = mod_doc

                    def _visit(node, path=""):
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                key = (
                                    f"{path}/{type(child).__name__}:{child.name}"
                                    if path
                                    else f"{type(child).__name__}:{child.name}"
                                )
                                doc = ast.get_docstring(child, clean=True)
                                if doc is not None:
                                    docs[key] = doc
                                _visit(child, key)

                    _visit(tree)
                    return docs

                before_docs = _collect_docstrings(code_before)
                after_docs = _collect_docstrings(code_after)
                if before_docs != after_docs:
                    changed_keys = sorted(set(before_docs.keys()) | set(after_docs.keys()))
                    example_key = changed_keys[0] if changed_keys else "<unknown>"
                    return False, f"Patch modifies docstrings (forbidden). Docstring key changed: {example_key}"
            except (SyntaxError, TypeError, IndexError) as e:
                logger.debug(f"Docstring comparison failed: {e}")
        def is_comment_or_empty(line):
            stripped = line.strip()
            return not stripped or stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")

        meaningful_added = [l for l in added_lines if not is_comment_or_empty(l)]
        meaningful_removed = [l for l in removed_lines if not is_comment_or_empty(l)]

        if not meaningful_added and not meaningful_removed:
            return False, "Patch contains ONLY comment changes"

        # 检查 trivial 修改 (仅 print/logging)
        trivial_patterns = [r'^print\s*\(', r'^logging\.', r'^logger\.', r'^pass$']
        non_trivial = [l for l in meaningful_added if not any(re.match(p, l.strip()) for p in trivial_patterns)]

        if not non_trivial and not meaningful_removed:
            return False, "Patch contains ONLY trivial changes (print/logging/pass)"

        return True, ""

    def _validate_intent_alignment(
        self,
        code_before: str,
        code_after: str,
        seed_pattern: SeedPattern,
        expected_intent: str
    ) -> float:
        """
        验证生成的代码修改是否与 Seed 的 Fix Intent 对齐。

        Args:
            code_before: 原始代码
            code_after: 修改后的代码
            seed_pattern: Seed 的模式信息
            expected_intent: 期望的 Intent 类型

        Returns:
            对齐分数 (0.0-1.0)
        """
        if not code_before or not code_after:
            return 0.0

        # 计算差异
        before_lines = code_before.splitlines()
        after_lines = code_after.splitlines()

        # 使用 difflib 找到具体改变的行
        diff = list(difflib.unified_diff(before_lines, after_lines, lineterm=''))

        added_lines = [l[1:] for l in diff if l.startswith('+') and not l.startswith('+++')]
        removed_lines = [l[1:] for l in diff if l.startswith('-') and not l.startswith('---')]

        if not added_lines and not removed_lines:
            # 没有实际改变
            return 0.0

        # 根据不同的 Intent 类型检查修改是否匹配
        score = 0.0
        pattern_type = seed_pattern.pattern_type
        semantic_category = seed_pattern.semantic_category

        # 检查改变的类型是否匹配预期
        all_changes = ' '.join(added_lines + removed_lines)

        if expected_intent == "Constant_Update":
            if semantic_category.startswith("regex"):
                if re.search(r'[r]?["\'][^"\']*[\$\\][^"\']*["\']', all_changes):
                    score = 0.9
                elif re.search(r'[r]?["\'][^"\']+["\']', all_changes):
                    score = 0.7
            elif semantic_category.startswith("numeric"):
                if re.search(r'\d+', all_changes):
                    score = 0.8
            elif semantic_category.startswith("string"):
                if re.search(r'["\'][^"\']+["\']', all_changes):
                    score = 0.8
            else:
                if re.search(r'["\'][^"\']*["\']|\d+', all_changes):
                    score = 0.6

        elif expected_intent == "Condition_Refinement":
            if re.search(r'(if|while|elif|and|or|not|>=|<=|>|<|==|!=|is\s+not|is\s+None)', all_changes):
                score = 0.9
            else:
                score = 0.3

        elif expected_intent == "Guard_Clause_Addition":
            if re.search(r'(if\s+.*is\s+(None|not)|isinstance|assert|raise|return\s+(None|$))', all_changes):
                score = 0.9
            else:
                score = 0.3

        elif expected_intent == "Argument_Update":
            if re.search(r'\([^)]*[,=][^)]*\)', all_changes):
                score = 0.8
            else:
                score = 0.3

        elif expected_intent == "Exception_Fix":
            if re.search(r'(except|raise|try|finally|Exception|Error)', all_changes):
                score = 0.9
            else:
                score = 0.3

        elif expected_intent == "API_Replacement":
            if re.search(r'\w+\.\w+\(', all_changes):
                score = 0.7
            else:
                score = 0.4

        elif expected_intent == "Variable_Replacement":
            if len(removed_lines) > 0 and len(added_lines) > 0:
                score = 0.6
            else:
                score = 0.3

        else:
            # 其他类型，给予中等分数
            score = 0.5

        # 额外奖励：如果修改行数合理（不是大规模重写）
        total_changes = len(added_lines) + len(removed_lines)
        if 1 <= total_changes <= 10:
            score = min(1.0, score + 0.1)
        elif total_changes > 20:
            score = max(0.0, score - 0.2)

        return score

    def _validate_expected_tests(self, expected_tests: List[str]) -> tuple[List[str], List[str]]:
        """验证 Agent 指定的测试是否存在。"""
        placeholders = {'test_reproduce_bug', 'test_synthetic', 'test_bug', 'test_method', 'TestClass', 'test_example'}
        valid, invalid = [], []

        for test in expected_tests:
            if not test or not isinstance(test, str):
                continue

            # 过滤占位符
            if any(p in test for p in placeholders):
                invalid.append(test)
                continue

            # 检查文件是否存在
            test_file = test.split("::")[0] if "::" in test else test
            if self.repo_path and (Path(self.repo_path) / test_file).exists():
                valid.append(test)
            else:
                invalid.append(test)

        return valid, invalid

    def _filter_tests_to_allowed_suite(self, tests: List[str], allowed: List[str]) -> List[str]:
        """Restrict test ids to the allowed suite list (module-related)."""
        if not tests:
            return []
        allowed_set = {t.strip() for t in (allowed or []) if isinstance(t, str) and t.strip()}
        if not allowed_set:
            return tests
        # File-level relaxation:
        # The agent may propose a concrete test case from a planned test *file* that is not
        # explicitly enumerated in `allowed_set` (e.g., because we only collected a subset of tests
        # from that file). To avoid dropping valid signals, allow any test whose file part matches
        # an allowed test file.
        allowed_files: set[str] = set()
        for a in allowed_set:
            if "::" in a:
                fp = a.split("::", 1)[0].strip()
                if fp.endswith(".py"):
                    allowed_files.add(fp)
        filtered: list[str] = []
        for t in tests:
            if not t or not isinstance(t, str):
                continue
            s = t.strip()
            if not s:
                continue
            if s in allowed_set:
                filtered.append(s)
                continue
            if "::" in s and allowed_files:
                fp = s.split("::", 1)[0].strip()
                if fp in allowed_files:
                    filtered.append(s)
        return filtered

    # ------------------------------------------------------------------
    # Problem Statement quality gate
    # ------------------------------------------------------------------

    # Fallback PS patterns that indicate low quality
    _PS_FALLBACK_PATTERNS = [
        "Synthetic defect",
        "synthetic defect",
        "No description",
        "Bug injection",
    ]

    # Behavioral description keywords
    _PS_BEHAVIORAL_PATTERNS: list[re.Pattern] = [
        re.compile(r"\b(when|instead of|expected|returns|produces|should|fails to)\b", re.IGNORECASE),
    ]

    def _validate_ps_quality(self, ps: str, target_level: str = "standard") -> tuple[bool, str]:
        """
        Validate problem statement quality against target level requirements.

        Returns (is_acceptable, reason).
        """
        if not ps or not ps.strip():
            return False, "empty_ps"

        tokens = ps.split()
        token_count = len(tokens)

        # Check for fallback content
        if any(fp in ps for fp in self._PS_FALLBACK_PATTERNS):
            return False, "fallback_detected"

        # Too short to be useful
        if token_count < 8:
            return False, f"too_short ({token_count} tokens)"

        # Check for remaining leaks (post-sanitization)
        leak_count = sum(1 for p in self._FIX_DIRECTIVE_PATTERNS if p.search(ps))
        if leak_count > 0:
            return False, f"fix_leak_detected ({leak_count} patterns)"

        # Level-specific checks
        if target_level == "minimal":
            if token_count > 60:
                return False, f"too_verbose_for_minimal ({token_count} tokens)"
        elif target_level == "standard":
            if token_count < 15:
                return False, f"too_short_for_standard ({token_count} tokens)"
            has_behavioral = any(p.search(ps) for p in self._PS_BEHAVIORAL_PATTERNS)
            if not has_behavioral:
                return False, "missing_behavioral_desc_for_standard"
        elif target_level == "detailed":
            if token_count < 30:
                return False, f"too_short_for_detailed ({token_count} tokens)"

        return True, ""

    def _validate_python_syntax(self, code: str) -> bool:
        """验证 Python 代码语法是否有效。"""
        try:
            import ast
            import textwrap
            ast.parse(textwrap.dedent(code))
            return True
        except SyntaxError as e:
            logger.debug(f"语法错误: {e}")
            return False

