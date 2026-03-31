"""
Adversarial Problem Statement Verification.

Uses an LLM as a "solver" to check whether a problem statement leaks
enough information to directly identify the fix. If the LLM can accurately
guess the fix location and strategy from PS alone, the PS is too leaky.

This is a post-hoc quality check, not part of the main synthesis pipeline.
It requires LLM API calls, so it should be run as a batch analysis step.

Usage (standalone):
    python -m src.modules.synthesis.ps_adversarial \
        --input quality_metrics.json \
        --output ps_adversarial_results.json \
        --provider openai --model qwen3-coder-plus

Usage (from code):
    checker = AdversarialPSChecker(llm_client)
    result = await checker.check_single(problem_statement, repo, file_path)
    # result.leak_score: 0.0 (no leak) to 1.0 (full leak)
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adversarial prompt templates
# ---------------------------------------------------------------------------

_ADVERSARIAL_SYSTEM = """You are a code debugging expert. Given a bug report (problem statement) \
for a Python project, your task is to guess:
1. Which file(s) are likely buggy.
2. What type of code change is needed (e.g., add a condition, fix a return value, change an import).
3. A specific fix strategy (1-2 sentences).

Be as specific as possible. If the problem statement is vague and you cannot \
narrow it down, say "CANNOT_DETERMINE" for each field.

Respond in JSON format:
{
  "predicted_file": "<file path or CANNOT_DETERMINE>",
  "predicted_change_type": "<change type or CANNOT_DETERMINE>",
  "predicted_fix_strategy": "<strategy or CANNOT_DETERMINE>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}"""

_ADVERSARIAL_USER = """Project: {repo}

Bug Report:
{problem_statement}

Based ONLY on the information in this bug report, predict:
1. Which file is most likely buggy?
2. What type of code change is needed?
3. What is the specific fix strategy?

If you cannot determine any of these from the bug report alone, say CANNOT_DETERMINE."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdversarialResult:
    """Result of adversarial PS verification for a single instance."""
    instance_id: str
    problem_statement: str
    predicted_file: str = ""
    predicted_change_type: str = ""
    predicted_fix_strategy: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    # Ground truth (for scoring)
    actual_file: str = ""
    actual_fix_intent: str = ""
    # Computed scores
    file_match: bool = False
    change_type_match: bool = False
    leak_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "predicted_file": self.predicted_file,
            "predicted_change_type": self.predicted_change_type,
            "predicted_fix_strategy": self.predicted_fix_strategy,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "actual_file": self.actual_file,
            "actual_fix_intent": self.actual_fix_intent,
            "file_match": self.file_match,
            "change_type_match": self.change_type_match,
            "leak_score": self.leak_score,
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_CANNOT_DETERMINE = "cannot_determine"


def _normalize(s: str) -> str:
    """Normalize a string for comparison."""
    return s.strip().lower().replace("_", " ").replace("-", " ")


def _file_matches(predicted: str, actual: str) -> bool:
    """Check if predicted file matches actual (fuzzy)."""
    if not predicted or not actual:
        return False
    if _CANNOT_DETERMINE in predicted.lower():
        return False
    # Exact match
    pred_norm = predicted.strip().strip("/")
    actual_norm = actual.strip().strip("/")
    if pred_norm == actual_norm:
        return True
    # Basename match
    if Path(pred_norm).name == Path(actual_norm).name:
        return True
    # Partial path match (last 2 components)
    pred_parts = Path(pred_norm).parts
    actual_parts = Path(actual_norm).parts
    if len(pred_parts) >= 2 and len(actual_parts) >= 2:
        if pred_parts[-2:] == actual_parts[-2:]:
            return True
    return False


def _change_type_matches(predicted: str, actual_intent: str) -> bool:
    """Check if predicted change type matches actual fix intent."""
    if not predicted or not actual_intent:
        return False
    if _CANNOT_DETERMINE in predicted.lower():
        return False

    pred = _normalize(predicted)
    actual = _normalize(actual_intent)

    # Direct substring match
    if actual in pred or pred in actual:
        return True

    # Semantic mapping (fix_intent → common descriptions)
    intent_synonyms = {
        "condition refinement": ["condition", "if statement", "guard", "check", "boolean"],
        "argument update": ["argument", "parameter", "function call", "passing"],
        "statement insertion": ["add", "insert", "missing", "new line"],
        "constant update": ["constant", "value", "literal", "number", "string"],
        "type cast": ["type", "cast", "convert", "coerce"],
        "guard clause": ["guard", "check", "validation", "early return"],
        "var replace": ["variable", "rename", "replace", "swap"],
        "return value": ["return", "output", "result"],
    }

    for intent_key, synonyms in intent_synonyms.items():
        if intent_key in actual:
            if any(s in pred for s in synonyms):
                return True

    return False


def compute_leak_score(result: AdversarialResult) -> float:
    """Compute a leak score from adversarial verification result.

    Scoring:
    - 0.0: LLM cannot determine anything (good PS, no leak)
    - 0.25: LLM guesses file correctly but not change type
    - 0.50: LLM guesses change type but not file
    - 0.75: LLM guesses both file and change type
    - 1.0: LLM guesses both with high confidence (>0.7)

    The score is adjusted by LLM confidence.
    """
    base = 0.0
    if result.file_match:
        base += 0.35
    if result.change_type_match:
        base += 0.35

    # Confidence adjustment
    conf = result.confidence
    if conf > 0.7:
        base += 0.30
    elif conf > 0.4:
        base += 0.15

    return min(1.0, base)


# ---------------------------------------------------------------------------
# Main checker class
# ---------------------------------------------------------------------------

class AdversarialPSChecker:
    """Check problem statements for information leakage using adversarial LLM queries."""

    def __init__(self, llm_client: Any):
        """
        Args:
            llm_client: An LLM client with a `chat()` or `generate()` method.
                        Expected interface: client.chat(messages, temperature=...) -> str
        """
        self.llm_client = llm_client

    def check_single(
        self,
        problem_statement: str,
        repo: str,
        actual_file: str = "",
        actual_fix_intent: str = "",
        instance_id: str = "",
    ) -> AdversarialResult:
        """Run adversarial check on a single problem statement.

        Returns AdversarialResult with leak_score.
        """
        result = AdversarialResult(
            instance_id=instance_id,
            problem_statement=problem_statement,
            actual_file=actual_file,
            actual_fix_intent=actual_fix_intent,
        )

        if not problem_statement or problem_statement in ("Synthetic defect", ""):
            result.reasoning = "Empty or fallback PS, skipped"
            return result

        try:
            response = self._query_llm(problem_statement, repo)
            parsed = self._parse_response(response)

            result.predicted_file = parsed.get("predicted_file", "")
            result.predicted_change_type = parsed.get("predicted_change_type", "")
            result.predicted_fix_strategy = parsed.get("predicted_fix_strategy", "")
            result.confidence = float(parsed.get("confidence", 0.0))
            result.reasoning = parsed.get("reasoning", "")

            # Score against ground truth
            result.file_match = _file_matches(result.predicted_file, actual_file)
            result.change_type_match = _change_type_matches(
                result.predicted_change_type, actual_fix_intent
            )
            result.leak_score = compute_leak_score(result)

        except Exception as e:
            logger.warning(f"Adversarial check failed for {instance_id}: {e}")
            result.reasoning = f"Error: {e}"

        return result

    def check_batch(
        self,
        instances: List[Dict],
        repo_key: str = "repo",
        ps_key: str = "problem_statement",
        file_key: str = "file_path",
        intent_key: str = "fix_intent",
        id_key: str = "instance_id",
    ) -> List[AdversarialResult]:
        """Run adversarial check on a batch of instances."""
        results = []
        for i, inst in enumerate(instances):
            logger.info(f"Checking {i+1}/{len(instances)}: {inst.get(id_key, 'unknown')}")
            r = self.check_single(
                problem_statement=inst.get(ps_key, ""),
                repo=inst.get(repo_key, ""),
                actual_file=inst.get(file_key, ""),
                actual_fix_intent=inst.get(intent_key, ""),
                instance_id=inst.get(id_key, ""),
            )
            results.append(r)
        return results

    def _query_llm(self, problem_statement: str, repo: str) -> str:
        """Send adversarial prompt to LLM."""
        user_msg = _ADVERSARIAL_USER.format(
            repo=repo, problem_statement=problem_statement
        )
        messages = [
            {"role": "system", "content": _ADVERSARIAL_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        # Try chat interface first, then generate
        if hasattr(self.llm_client, "chat"):
            return self.llm_client.chat(messages, temperature=0.1)
        elif hasattr(self.llm_client, "generate"):
            prompt = f"{_ADVERSARIAL_SYSTEM}\n\n{user_msg}"
            return self.llm_client.generate(prompt, temperature=0.1)
        else:
            raise ValueError("LLM client must have a chat() or generate() method")

    @staticmethod
    def _parse_response(response: str) -> Dict:
        """Parse JSON response from LLM."""
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in response
        brace_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse LLM response as JSON: {response[:200]}")
        return {
            "predicted_file": "CANNOT_DETERMINE",
            "predicted_change_type": "CANNOT_DETERMINE",
            "predicted_fix_strategy": "CANNOT_DETERMINE",
            "confidence": 0.0,
            "reasoning": "Failed to parse response",
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Standalone CLI for batch adversarial PS verification."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Adversarial PS verification (batch)"
    )
    parser.add_argument("--input", required=True, help="Input JSON (quality_metrics or final_dataset)")
    parser.add_argument("--output", required=True, help="Output JSON for results")
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="qwen3-coder-plus", help="LLM model")
    parser.add_argument("--limit", type=int, default=0, help="Max instances to check (0=all)")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # quality_metrics format: {instance_id: {...}}
        instances = [{"instance_id": k, **v} for k, v in data.items()]
    else:
        instances = data

    if args.limit > 0:
        instances = instances[:args.limit]

    print(f"Loaded {len(instances)} instances")

    # Create LLM client
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.modules.llm_client import create_llm_client

    llm_config = {
        "provider": args.provider,
        "model": args.model,
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    llm_client = create_llm_client(llm_config)

    # Run checks
    checker = AdversarialPSChecker(llm_client)
    results = checker.check_batch(instances)

    # Save results
    output_data = [r.to_dict() for r in results]
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    leak_scores = [r.leak_score for r in results]
    file_matches = sum(1 for r in results if r.file_match)
    type_matches = sum(1 for r in results if r.change_type_match)
    n = len(results)

    print(f"\n{'='*50}")
    print(f"Adversarial PS Verification Summary")
    print(f"{'='*50}")
    print(f"Total instances: {n}")
    print(f"File match rate: {file_matches}/{n} ({file_matches/n*100:.1f}%)")
    print(f"Change type match: {type_matches}/{n} ({type_matches/n*100:.1f}%)")
    print(f"Avg leak score: {sum(leak_scores)/n:.3f}")
    print(f"High leak (>0.5): {sum(1 for s in leak_scores if s > 0.5)}/{n}")
    print(f"No leak (0.0): {sum(1 for s in leak_scores if s == 0.0)}/{n}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
