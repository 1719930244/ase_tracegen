"""
Hybrid Problem Statement Generator.

Combines sanitized LLM-generated PS with test-output-derived information
to produce higher-quality, leak-free problem statements at configurable
information levels.

Modes:
- llm_only:      Use the original LLM PS as-is (current default behavior).
- test_only:     Generate PS entirely from test_output.txt (zero leak risk).
- hybrid:        LLM behavioral description + test output assertion details.
- test_enhanced: Start from test output, add LLM context if quality passes.

Integration:
- Post-validation step in runner.py (when test_output.txt is available)
- Standalone via scripts/quality/generate_ps.py (batch reprocessing)
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Test output parsing (self-contained, mirrors generate_ps.py)
# ---------------------------------------------------------------------------

_FAIL_HEADER_RE = re.compile(
    r"^(FAIL|ERROR): (\S+) \(([^)]+)\)(?:\s*\n(.+))?", re.MULTILINE
)

_TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\):.*?^([A-Z]\w*(?:Error|Exception)[^\n]*))",
    re.MULTILINE | re.DOTALL,
)

_ASSERTION_MSG_RE = re.compile(
    r"^(Assertion(?:Error)?:?\s*.+?)$", re.MULTILINE
)

_SOURCE_FILE_RE = re.compile(
    r'File "(/testbed/(?!tests/)[^"]+)", line (\d+), in (\w+)'
)

_RAN_RE = re.compile(r"Ran (\d+) tests? in ([\d.]+)s")
_RESULT_RE = re.compile(r"FAILED \(([^)]+)\)")


def parse_test_output(content: str) -> Dict:
    """Parse test_output.txt into structured failure information."""
    result = {
        "failing_tests": [],
        "tracebacks": [],
        "assertion_messages": [],
        "exception_types": [],
        "test_modules": set(),
        "source_files": set(),
        "total_tests": 0,
        "total_failures": 0,
        "total_errors": 0,
    }

    for m in _FAIL_HEADER_RE.finditer(content):
        module = m.group(3).rsplit(".", 1)[0] if "." in m.group(3) else m.group(3)
        result["failing_tests"].append({
            "type": m.group(1),
            "method": m.group(2),
            "class": m.group(3),
            "module": module,
            "description": (m.group(4) or "").strip(),
        })
        result["test_modules"].add(module)

    for m in _TRACEBACK_RE.finditer(content):
        exc_line = m.group(2).strip()
        result["tracebacks"].append({
            "full": m.group(1).strip(),
            "exception": exc_line,
        })
        exc_type = exc_line.split(":")[0].split("(")[0].strip()
        if exc_type not in result["exception_types"]:
            result["exception_types"].append(exc_type)

    for m in _ASSERTION_MSG_RE.finditer(content):
        msg = m.group(1).strip()
        if len(msg) > 500:
            msg = msg[:500] + "..."
        result["assertion_messages"].append(msg)

    for m in _SOURCE_FILE_RE.finditer(content):
        result["source_files"].add(m.group(1).replace("/testbed/", ""))

    ran_match = _RAN_RE.search(content)
    if ran_match:
        result["total_tests"] = int(ran_match.group(1))

    result_match = _RESULT_RE.search(content)
    if result_match:
        for part in result_match.group(1).split(","):
            part = part.strip()
            if part.startswith("failures="):
                result["total_failures"] = int(part.split("=")[1])
            elif part.startswith("errors="):
                result["total_errors"] = int(part.split("=")[1])

    result["test_modules"] = sorted(result["test_modules"])
    result["source_files"] = sorted(result["source_files"])
    return result


# ---------------------------------------------------------------------------
# Leak-free component extraction from test output
# ---------------------------------------------------------------------------

def _extract_symptom_from_assertion(assertion_msg: str) -> str:
    """Extract a symptom description from an assertion message.

    Removes internal details (file paths, variable names) and keeps
    the observable behavioral difference (expected vs actual).
    """
    # Strip "AssertionError: " prefix
    msg = re.sub(r"^Assertion(?:Error)?:?\s*", "", assertion_msg).strip()

    # Common patterns: "X != Y", "X not in Y", "False is not true"
    # These are inherently leak-free (they describe symptoms, not fixes)
    if not msg or len(msg) < 5:
        return ""

    # Truncate overly long assertion messages
    if len(msg) > 200:
        msg = msg[:200] + "..."

    return msg


def _extract_exception_symptom(tracebacks: List[Dict]) -> str:
    """Build a symptom description from exception type and message."""
    if not tracebacks:
        return ""

    exc = tracebacks[0]["exception"]
    # "TypeError: foo() takes 2 positional arguments but 3 were given"
    # This is a symptom, not a fix hint.
    # Only keep the exception type + first line of message
    parts = exc.split("\n")
    return parts[0].strip()


# ---------------------------------------------------------------------------
# Hybrid PS generation
# ---------------------------------------------------------------------------

_LEAK_PATTERNS_FOR_HYBRID = [
    # Fix directive verbs (even with intervening words)
    re.compile(r"\b(fix|replace|change|switch)\b.*\b(to|from|with)\b", re.IGNORECASE),
    re.compile(r"\bshould\s+(?:use|be|have|call|return|raise)\b", re.IGNORECASE),
    re.compile(r"\b(needs to be|must be|has to be)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+(bug|issue|root cause|problem)\s+is\b", re.IGNORECASE),
    re.compile(r"\binstead of\s+(?:using|calling)\b", re.IGNORECASE),
    re.compile(r"\b(?:missing|removed|deleted|added)\s+(?:the\s+)?(?:call|check|import)\b", re.IGNORECASE),
]


def _is_sentence_leak_free(sentence: str) -> bool:
    """Quick check if a sentence is free of fix-leaking content."""
    return not any(p.search(sentence) for p in _LEAK_PATTERNS_FOR_HYBRID)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    # Split on period/exclamation/question followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class HybridPSGenerator:
    """Generate problem statements by combining LLM output with test results.

    The key insight: test output components (assertion messages, exception
    types) are inherently leak-free because they describe observable symptoms,
    not implementation details or fixes.
    """

    # Modes
    LLM_ONLY = "llm_only"
    TEST_ONLY = "test_only"
    HYBRID = "hybrid"
    TEST_ENHANCED = "test_enhanced"

    VALID_MODES = {LLM_ONLY, TEST_ONLY, HYBRID, TEST_ENHANCED}

    def generate(
        self,
        llm_ps: str,
        test_output: str,
        target_level: str = "standard",
        mode: str = "hybrid",
        repo: str = "",
    ) -> Dict:
        """Generate a hybrid problem statement.

        Args:
            llm_ps: Original LLM-generated problem statement (already sanitized).
            test_output: Raw content of test_output.txt.
            target_level: Target PS level (minimal / standard / detailed).
            mode: Generation mode (llm_only / test_only / hybrid / test_enhanced).
            repo: Repository name (for minimal-level PS).

        Returns:
            Dict with keys: problem_statement, mode, level, components, quality.
        """
        if mode not in self.VALID_MODES:
            logger.warning(f"Unknown mode '{mode}', falling back to 'hybrid'")
            mode = self.HYBRID

        parsed = parse_test_output(test_output) if test_output else {}

        if mode == self.LLM_ONLY:
            return self._wrap(llm_ps, mode, target_level, {"source": "llm"})

        if mode == self.TEST_ONLY:
            ps = self._from_test_output(parsed, target_level, repo)
            return self._wrap(ps, mode, target_level, {"source": "test_output"})

        if mode == self.TEST_ENHANCED:
            ps = self._test_enhanced(llm_ps, parsed, target_level, repo)
            return self._wrap(ps, mode, target_level, {"source": "test_output+llm_context"})

        # Default: hybrid
        ps = self._hybrid(llm_ps, parsed, target_level, repo)
        components = {
            "llm_sentences": self._count_llm_contribution(llm_ps, ps),
            "test_sentences": self._count_test_contribution(parsed, ps),
        }
        return self._wrap(ps, mode, target_level, components)

    # ------------------------------------------------------------------
    # Core generation strategies
    # ------------------------------------------------------------------

    def _hybrid(
        self, llm_ps: str, parsed: Dict, level: str, repo: str
    ) -> str:
        """Combine LLM behavioral description with test output assertions.

        Strategy:
        1. Extract leak-free sentences from LLM PS (behavioral description).
        2. Add assertion/exception details from test output.
        3. Trim/expand to match target level.
        """
        parts = []

        # Step 1: Filter LLM PS for leak-free behavioral sentences
        if llm_ps and llm_ps not in ("Synthetic defect", ""):
            for sentence in _split_sentences(llm_ps):
                if _is_sentence_leak_free(sentence):
                    parts.append(sentence)

        # Step 2: Add test-output-derived components
        if parsed:
            # Assertion messages (most informative, zero leak risk)
            for msg in parsed.get("assertion_messages", [])[:2]:
                symptom = _extract_symptom_from_assertion(msg)
                if symptom and not self._already_covered(symptom, parts):
                    parts.append(f"Observed: {symptom}")

            # Exception type (if not AssertionError, adds diagnostic value)
            exc_symptom = _extract_exception_symptom(parsed.get("tracebacks", []))
            if exc_symptom and "AssertionError" not in exc_symptom:
                if not self._already_covered(exc_symptom, parts):
                    parts.append(f"The operation raises: {exc_symptom}")

        # Step 3: Level-specific trimming
        ps = self._trim_to_level(parts, level, parsed, repo)
        return ps

    def _from_test_output(
        self, parsed: Dict, level: str, repo: str
    ) -> str:
        """Generate PS entirely from test output (zero LLM content)."""
        parts = []

        if level == "minimal":
            modules = parsed.get("test_modules", [])
            n_fail = parsed.get("total_failures", 0) + parsed.get("total_errors", 0)
            if modules:
                return f"{n_fail} test(s) in {modules[0]} are failing."
            return f"Some tests in {repo or 'the project'} are failing."

        # Standard / detailed: build from assertions and exceptions
        for msg in parsed.get("assertion_messages", [])[:3]:
            symptom = _extract_symptom_from_assertion(msg)
            if symptom:
                parts.append(f"Observed: {symptom}")

        exc_symptom = _extract_exception_symptom(parsed.get("tracebacks", []))
        if exc_symptom:
            parts.append(f"The operation raises: {exc_symptom}")

        if not parts:
            # Fallback: describe the failing tests
            for ft in parsed.get("failing_tests", [])[:3]:
                desc = ft.get("description", "")
                if desc:
                    parts.append(f"Test {ft['method']} ({ft['class']}): {desc}")
                else:
                    parts.append(f"Test {ft['method']} in {ft['class']} fails.")

        if level == "detailed" and parsed.get("tracebacks"):
            # Add first traceback for detailed level
            tb = parsed["tracebacks"][0]["full"]
            # Truncate to ~500 chars
            if len(tb) > 500:
                tb = tb[:500] + "\n..."
            parts.append(f"```\n{tb}\n```")

        if not parts:
            return f"Tests in {repo or 'the project'} are failing unexpectedly."

        return " ".join(parts) if level != "detailed" else "\n\n".join(parts)

    def _test_enhanced(
        self, llm_ps: str, parsed: Dict, level: str, repo: str
    ) -> str:
        """Start from test output, add LLM context if quality is sufficient."""
        test_ps = self._from_test_output(parsed, level, repo)

        if not llm_ps or llm_ps in ("Synthetic defect", ""):
            return test_ps

        # Add leak-free LLM context
        llm_context = []
        for sentence in _split_sentences(llm_ps):
            if _is_sentence_leak_free(sentence) and not self._already_covered(sentence, [test_ps]):
                llm_context.append(sentence)

        if not llm_context:
            return test_ps

        # Prepend LLM context (behavioral overview) before test details
        combined = " ".join(llm_context[:2]) + " " + test_ps
        return combined.strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trim_to_level(
        self, parts: List[str], level: str, parsed: Dict, repo: str
    ) -> str:
        """Trim or expand parts to match the target information level."""
        if not parts:
            return self._from_test_output(parsed, level, repo)

        if level == "minimal":
            # Keep only the first sentence, trim to ~30 words
            first = parts[0]
            words = first.split()
            if len(words) > 30:
                first = " ".join(words[:30]) + "."
            return first

        if level == "standard":
            # 50-100 words target
            combined = " ".join(parts)
            words = combined.split()
            if len(words) > 120:
                combined = " ".join(words[:100]) + "."
            return combined

        # detailed: keep everything
        return "\n\n".join(parts)

    def _already_covered(self, new_text: str, existing: List[str]) -> bool:
        """Check if the content is already covered by existing parts."""
        new_lower = new_text.lower()[:60]
        for part in existing:
            if new_lower in part.lower():
                return True
        return False

    def _count_llm_contribution(self, llm_ps: str, final_ps: str) -> int:
        """Count how many LLM-originated sentences survive in final PS."""
        if not llm_ps:
            return 0
        count = 0
        for sentence in _split_sentences(llm_ps):
            # Check if the sentence (or substantial overlap) appears in final
            if sentence[:40].lower() in final_ps.lower():
                count += 1
        return count

    def _count_test_contribution(self, parsed: Dict, final_ps: str) -> int:
        """Count how many test-output-derived elements are in final PS."""
        count = 0
        for msg in parsed.get("assertion_messages", []):
            if msg[:30].lower() in final_ps.lower():
                count += 1
        for tb in parsed.get("tracebacks", []):
            if tb["exception"][:30].lower() in final_ps.lower():
                count += 1
        return count

    @staticmethod
    def _wrap(ps: str, mode: str, level: str, components: Dict) -> Dict:
        """Wrap result in a standard output format."""
        tokens = len(ps.split())
        return {
            "problem_statement": ps,
            "mode": mode,
            "level": level,
            "token_count": tokens,
            "components": components,
        }


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------

def enhance_ps_post_validation(
    original_ps: str,
    test_output_path: Path,
    target_level: str = "standard",
    mode: str = "hybrid",
    repo: str = "",
) -> Optional[Dict]:
    """Enhance a problem statement using test output from validation.

    Called after Stage 3 validation produces test_output.txt.
    Returns None if test output is unavailable or parsing fails.
    """
    if not test_output_path or not test_output_path.exists():
        logger.debug(f"No test output at {test_output_path}, skipping PS enhancement")
        return None

    try:
        content = test_output_path.read_text(errors="replace")
        if not content or len(content) < 50:
            return None
    except Exception as e:
        logger.warning(f"Failed to read test output: {e}")
        return None

    generator = HybridPSGenerator()
    result = generator.generate(
        llm_ps=original_ps,
        test_output=content,
        target_level=target_level,
        mode=mode,
        repo=repo,
    )

    # Only return if the hybrid PS is meaningfully different and non-empty
    if result["problem_statement"] and result["problem_statement"] != original_ps:
        logger.info(
            f"PS enhanced: {len(original_ps.split())} → {result['token_count']} tokens "
            f"(mode={mode}, level={target_level})"
        )
        return result

    return None


# ---------------------------------------------------------------------------
# Continuous Information Level Controller
# ---------------------------------------------------------------------------

# Information components ranked by informativeness (low → high).
# Each component is a tuple: (name, weight, extraction_fn_name)
# The controller progressively includes components as info_level increases.
_INFO_COMPONENTS = [
    # Level 0.0 - 0.2: Bare minimum (module/area affected)
    ("affected_area", 0.0),
    # Level 0.2 - 0.4: Symptom category (assertion, error, crash)
    ("symptom_category", 0.2),
    # Level 0.4 - 0.6: Behavioral description (expected vs actual)
    ("behavioral_desc", 0.4),
    # Level 0.6 - 0.7: Concrete example (assertion message)
    ("concrete_example", 0.6),
    # Level 0.7 - 0.8: Exception type and message
    ("exception_info", 0.7),
    # Level 0.8 - 0.9: Affected source files (not test files)
    ("source_hint", 0.8),
    # Level 0.9 - 1.0: Traceback (truncated)
    ("traceback", 0.9),
]


class ContinuousInfoController:
    """Generate problem statements with a continuous information level ∈ [0, 1].

    Instead of discrete levels (minimal/standard/detailed), this controller
    takes a float value and progressively includes more information components.

    Mapping:
        0.0: "Tests in module X fail."          (~7 tokens, minimal)
        0.3: + symptom category                  (~15 tokens)
        0.5: + behavioral description            (~50 tokens, ≈ standard)
        0.7: + concrete assertion + exception    (~80 tokens)
        1.0: + traceback + source files          (~200+ tokens, ≈ detailed)

    Usage:
        controller = ContinuousInfoController()
        ps = controller.generate(info_level=0.5, llm_ps=..., test_output=..., repo=...)
    """

    def generate(
        self,
        info_level: float,
        llm_ps: str = "",
        test_output: str = "",
        repo: str = "",
    ) -> Dict:
        """Generate PS at a specific information level.

        Args:
            info_level: Float in [0.0, 1.0] controlling information density.
            llm_ps: Original LLM-generated PS (sanitized).
            test_output: Raw test_output.txt content.
            repo: Repository name.

        Returns:
            Dict with problem_statement, info_level, components_included, token_count.
        """
        info_level = max(0.0, min(1.0, info_level))

        parsed = parse_test_output(test_output) if test_output else {}
        parts = []
        included = []

        # Determine which components to include
        for comp_name, threshold in _INFO_COMPONENTS:
            if info_level >= threshold:
                text = self._extract_component(
                    comp_name, llm_ps, parsed, repo
                )
                if text:
                    parts.append(text)
                    included.append(comp_name)

        if not parts:
            ps = f"Tests in {repo or 'the project'} are failing."
            included = ["affected_area"]
        else:
            ps = " ".join(parts)

        # Trim to approximate target token count based on info_level
        target_tokens = int(10 + info_level * 200)  # 10 at 0.0, 210 at 1.0
        words = ps.split()
        if len(words) > target_tokens * 1.5:
            ps = " ".join(words[:target_tokens]) + "."

        return {
            "problem_statement": ps,
            "info_level": round(info_level, 2),
            "components_included": included,
            "token_count": len(ps.split()),
            "target_tokens": target_tokens,
        }

    def _extract_component(
        self,
        component: str,
        llm_ps: str,
        parsed: Dict,
        repo: str,
    ) -> str:
        """Extract a specific information component."""
        if component == "affected_area":
            modules = parsed.get("test_modules", [])
            if modules:
                n = parsed.get("total_failures", 0) + parsed.get("total_errors", 0)
                return f"{n} test(s) in {modules[0]} are failing."
            return f"Tests in {repo or 'the project'} are failing."

        if component == "symptom_category":
            exc_types = parsed.get("exception_types", [])
            if exc_types:
                primary = exc_types[0]
                if "Assertion" in primary:
                    return "The tests report incorrect output values."
                if "Type" in primary:
                    return "The operation encounters a type mismatch."
                if "Attribute" in primary:
                    return "An attribute access fails unexpectedly."
                if "Value" in primary:
                    return "The operation receives an invalid value."
                return f"The operation raises a {primary}."
            return ""

        if component == "behavioral_desc":
            # Extract behavioral sentences from LLM PS
            if llm_ps and llm_ps not in ("Synthetic defect", ""):
                behavioral = []
                for sentence in _split_sentences(llm_ps):
                    if _is_sentence_leak_free(sentence):
                        # Check if it describes behavior (expected vs actual)
                        if any(kw in sentence.lower() for kw in
                               ["instead", "expected", "returns", "produces",
                                "should", "fails to", "incorrect", "wrong"]):
                            behavioral.append(sentence)
                if behavioral:
                    return " ".join(behavioral[:2])
            return ""

        if component == "concrete_example":
            msgs = parsed.get("assertion_messages", [])
            if msgs:
                symptom = _extract_symptom_from_assertion(msgs[0])
                if symptom:
                    return f"Observed: {symptom}"
            return ""

        if component == "exception_info":
            tbs = parsed.get("tracebacks", [])
            if tbs:
                exc = tbs[0]["exception"]
                # Only the exception line, not full traceback
                return f"Error: {exc.split(chr(10))[0].strip()}"
            return ""

        if component == "source_hint":
            files = parsed.get("source_files", [])
            if files:
                names = [Path(f).name for f in files[:3]]
                return f"Affected files: {', '.join(names)}."
            return ""

        if component == "traceback":
            tbs = parsed.get("tracebacks", [])
            if tbs:
                tb = tbs[0]["full"]
                if len(tb) > 400:
                    tb = tb[:400] + "\n..."
                return f"Traceback:\n{tb}"
            return ""

        return ""


def generate_ps_at_level(
    info_level: float,
    llm_ps: str = "",
    test_output_path: Optional[Path] = None,
    test_output: str = "",
    repo: str = "",
) -> Dict:
    """Convenience function: generate PS at a continuous info level.

    Either provide test_output (str) directly or test_output_path (Path).
    """
    if not test_output and test_output_path and test_output_path.exists():
        test_output = test_output_path.read_text(errors="replace")

    controller = ContinuousInfoController()
    return controller.generate(
        info_level=info_level,
        llm_ps=llm_ps,
        test_output=test_output,
        repo=repo,
    )
