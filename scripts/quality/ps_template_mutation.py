#!/usr/bin/env python3
"""
Saving SWE-Bench Template Mutation Baseline for TraceGen.

Reproduces the mutation methodology from:
  Garg et al., "Saving SWE-Bench: A Benchmark Mutation Approach for
  Realistic Agent Evaluation", NeurIPS 2025 / CAIN 2026.

Official repo: https://github.com/microsoft/SWE-Bench-Mutated-CAIN26

Adapts their 14 communication templates + LLM mutation prompt to work
with TraceGen's synthetic bug data (summary.json format) and DashScope
qwen3-coder-plus API.

Usage:
    PYTHON=python
    $PYTHON scripts/quality/ps_template_mutation.py \
        --output ../tracegen-outputs/template_mutated_ps.json \
        [--num-workers 3] [--dry-run]
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from random import Random
from typing import Any

import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
SYNTH_SUMMARY = Path(
    "../tracegen-outputs/4repo_run/2026-02-25/19-52-01/2_synthesis/summary.json"
)
QUALITY_METRICS = Path("../tracegen-outputs/quality_metrics.json")

# ── LLM Config ───────────────────────────────────────────────────────────────
# Load API key from .env
_env_path = Path(".env")
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = "qwen3-coder-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
TEMPERATURE = 0.7
MAX_TOKENS = 4096
SEED = 42

# ── 14 Templates (verbatim from official repo) ──────────────────────────────
# Source: microsoft/SWE-Bench-Mutated-CAIN26/src/swebench_mutate/prompt_customization.py
DEFAULT_TEMPLATES = """### 1. **Paste Error Message or Stack Trace Without (Much) Explanation**
**Description:**
The user copies the error message, stack trace, or standard error output from their tool, test framework, or compiler, often with minimal or no context or explanation.

**Examples:**
5. `./app/page.tsx ... Error: ...`
7. `Analyze why I would get this error? ... fatal error: 'include/type.h' file not found`
26. `I am getting this error ... error C2039: 'getTenant': is not a member of '`global namespace'' ...`
114. `What does this error mean? OSError: [Errno 5] Unable to synchronously open file ... Input/output error ...`
188. `In file included from driver.cu:42: ... error: use of undeclared identifier 'bar' ...`

---

### 2. **Ask to Fix/Investigate a Specific Line, Function, or File**
**Description:**
The user pinpoints a function, code cell, file, or line (often by number) and asks for a fix or analysis.

**Examples:**
0. `fix issue on line 24`
17. `#file:data_access.py Fix this data setup cell given new local file architecture. ...`
19. `I made some changes to the Prompt for Finance, fix the CurrencyExchange_SucceedsAsync to reflect the changes`
41. `What would be the correct syntax line 30 to fix this error message? ...`
148. `fix the deleteme.py script`

---

### 3. **Describe Expected vs Actual Behavior**
**Description:**
The user describes what they expected versus what actually happened, sometimes mentioning environments or specific steps.

**Examples:**
6. `when mcp server is enabled in MCP server configuration, the button did not change.`
44. `When I run my app in desktop (macos) and try to scan the barcode, It gets recoginzed ... however on mobile (safary) it's not recognized ...`
64. `Actual Result: Upon invoking close button from the error dialog one tab focus lost ... Expected Result: ... it should immediately land on the previous interactive element ...`
140. `values from settings page is getting stored in seession. but i am only getting null values for them in the generate project`
183. `when I change type via FieldPicker the selectFieldValues ... update ... This means we're losing data. I'd like changing type to not remove any of this data. ... do you understand?`

---

### 4. **Paste Minimal Reproducible Example/Test Case**
**Description:**
The user provides a code snippet, test command, or small script, describing how to reproduce the bug.

**Examples:**
3. `When I run the following command: ... I get the following error: ...`
12. `I am running the below tape. It should provide output ... but it is only providing output from ... Please advise what is happening ...`
30. `when i run this test with TEST_ALL_FEATURES=1 bin/rails test ... it fails with the below message. ...`
67. `How to fix the error caused by this test? 'npm test -- tests/users.test.ts'`
184. `python image_generator.py ... ImportError: cannot import name ...`

---

### 5. **Direct "Fix This" or "Resolve the Error" Requests**
**Description:**
Concise directives asking for a fix or resolution, sometimes with pasted error/code, sometimes minimal.

**Examples:**
25. `fix lint C:\\platform\\apps\\me\\src\\components\\common\\domain\\DataSourceSection.tsx(96,6): error ...`
29. `fix this it is resulting in 2020-01-01 10:02:03.456 [error] Unexpected end of JSON input`
35. `fix sys:1: RuntimeWarning: Cannot safely stop actor at [McpApp.__del__]: loop is closed or not running ...`
84. `fix the issue`
121. `Fix issues`

---

### 6. **Copy/Paste Build/Test Output with Request to Analyze**
**Description:**
The user pastes a lengthy output from a build or test run (may include logs, diffs, or CI output) and asks what is wrong or how to fix.

**Examples:**
11. `build step: cxx "./obj/chrome/browser/app/client.obj" ... error ...`
20. `Fix these errors: x run @ui/workbench:test --runInBand ... test output ...`
24. `PS ...> python -m uvicorn API.app:app --reload --port 8000 --app-dir src ... [WinError 10013] An attempt was made to access a socket in a way forbidden ...`
60. `ERROR in ./node_modules/@modulename/test-runner/utils/config-file-creator.ts ... webpack ... errors ...`
190. `Analyze failure [671/672] Linking CXX executable app/functions/lib/table FAILED ...`

---

### 7. **Describe a System/Integration/Workflow Failure**
**Description:**
The user describes an error at the level of a pipeline, workflow, cloud deployment, multi-agent system, or connected services.

**Examples:**
36. `Starting multi-agent workflow... ... Multi-agent workflow failed: 'NoneType' object is not subscriptable`
59. `I'm debugging export-environment and it's not working as expected, I think I should be shown available environments before ...`
65. `Why we are getting 500 internal server error when trying to start server ... Connection state: Error 500 status sending message to ...`
111. `I'm not convinced the unpacking is working as expected, if I check the folder in the temp directories there are no files ...`
116. `looks like there's a memory leak in the browser when display a flow run in the canvas`

---

### 8. **Ask for Root Cause/Why/Diagnosis (May or May Not Provide Logs)**
**Description:**
The user asks "why" a failure, error, exception, or strange behavior is happening, sometimes with clues, sometimes not.

**Examples:**
14. `What could be the issue here?`
16. `Can you help me understand why the batched loss is so far from the single input loss?`
23. `why is this error thrown on production server? 2020-01-01 10:02:03.456 ... ModuleNotFoundError: No module named 'module'`
33. `can you investigate why the ProfilieCard is rendering the cursor as a text cursor and not a pointer`
62. `For some reason, isFooTokenRequired returns an Bar token, the same as isBarTokenRequired. ... Investigate why this is happening ...`

---

### 9. **Refer to an External or Linked Issue for Context**
**Description:**
The user asks to fix or analyze a bug based on a previously filed or documented issue, often referencing it by number or URL.

**Examples:**
4. `Fix the security vulnerability described in issue01.md`
49. `https://github.com/org/repo/issues/12345 Analyse the issue and give me a fix`
51. `Follow instructions in [investigate-violations.prompt.md](file:///C%3A/code/UX/.github/prompts/investigate-violations.prompt.md). ...`

---

### 10. **Request Debugging or Diagnosis Without Providing Code**
**Description:**
The user vaguely asks for help, diagnoses, or explanations for a problem, without directly providing the code or output.

**Examples:**
27. `see the error in output. how to fix it?`
28. `@lastCommand What is the error i'm getting?`
66. `why is my device login failing?`
110. `can you help me with the error`
171. `what is this error ?`

---

### 11. **Combine Question with Code Snippet**
**Description:**
The user posts code (partial, not always a full repro) and asks a direct question or seeks a fix/explanation about it.

**Examples:**
13. `How would do modify these lines so they work. I am not sure how to create the Credential in the return []`
98. `export async function submitResponses(gridControl: ClientApi.Controls.GridControl): Promise<void> {...} why is this async`
143. `const byteCharacters = atob(videoData); ... video is not playing`
196. `Scanner.cs calls the external library ... what is wrong in my code which is causing this. can it be a setting/flight value that i am passing?`

---

### 12. **Report Test or CI/CD Failure With Output**
**Description:**
The user includes output from continuous integration or a test suite and seeks a solution.

**Examples:**
20. `Fix these errors: x run @ui/kit:test --runInBand ...`
34. `I am seeing errors while running this pytest. Resolve the error.`
104. `getting error npm ERR! code ELIFECYCLE ...`
137. `the InteractiveAppTitle.test.tsx is failing. Can you look into it? It is supposed to do well`
175. `/Users/me/Repos/iOSApp/ios/AppKit/Tests/StoreTests.swift:143 testMakePurchase_onSuccessfulPurchase(): AssertEqual failed: ...`

---

### 13. **Describe Unintended Behavior or Bug Without Errors**
**Description:**
User describes a bug/issue that does not manifest as an error or stack trace, but as wrong logic, UI, or state.

**Examples:**
38. `I have a strange behavior with my prompt. Most of the time it correctly replaces ... instead of creating a new topic list, it actually repeats a reply from a previous conversation turn. ...`
52. `the chart doesn't display anything`
80. `There is no call being made on tapping the indicator view , then how the scroll to item happening?`
140. `values from settings page is getting stored in seession. but i am only getting null values for them in the generate project`

---

### 14. **Reference or Request to Analyze Log Files**
**Description:**
User asks for analysis of logs or error files to determine what went wrong.

**Examples:**
109. `I am executing below command to run a simulator: ... this error.log is what I am encountering. I want your help to explore why ...`
125. `In the current directory, there are results for running a LLM on a benchmark called multiswebench ... can you go through the logs and see why the instances are failed. ...`
161. `Find the cause of this error. What file or service triggers this. Trace it up to the offending code please: ... Error ... buildItems at line 12:34 ...`
162. `what this could be about "g 12 34:56:78 app appio_dns[597]: [INFO] ... [ERROR] plugin/errors: ...`

---
"""


# ── Mutation Prompt (verbatim from official repo _format_mutation_prompt) ────

def format_mutation_prompt(problem_statement: str, patch: str) -> str:
    """Construct the mutation prompt exactly as in the official implementation."""
    return '''You are given:
1. A set of transformation templates showing how people informally describe bugs to an interactive coding assistant.
2. A software bug description from the SWE-bench-Verified dataset (in GitHub issue style).
3. The corresponding code patch that fixes the bug.

Your task:
- Apply as many transformation templates as make sense for this example.
- For each applicable template, rewrite the bug description in the style of that template.
- Use the patch details (e.g., file paths, function names, and line numbers) where relevant to make the request realistic.
- Make the query realistic to how users may query a chat-based agent. Users tend to write short, incomplete descriptions, often with typos. This is very IMPROTANT!!
- Skip templates that clearly do not apply.
- Output each transformed description as a separate bullet point, exactly following the below format (with **Transform Kind N** header and separating variants with hyphens):

```
**Transform Kind 1**
TEXT

-----------
**Transform Kind 2**
TEXT

-----------
**Transform Kind 3**
TEXT

-----------
```

Transformation templates observed in telemetry data of bug fixing queries issued by users in the past:
Here are the identified templates/patterns for bug reporting in the dataset, each with a description and five representative examples (with their indices):

---

{templates}

These templates cover the primary patterns in the dataset. Many queries combine multiple patterns (for example, a pasted stack trace plus a question or a code snippet plus expected/actual output), but the above organizing scheme captures the dominant approaches to bug communication to an AI assistant.

Bug description:
"""{problem}"""

Code patch:
"""{patch}"""

Now generate all applicable transformed descriptions.
'''.format(
        templates=DEFAULT_TEMPLATES,
        problem=problem_statement,
        patch=patch,
    )


# ── Response Parser (adapted from official repo _parse_response) ────────────

_VARIANT_PATTERN = re.compile(
    r"""
    \s*                           # Leading whitespace
    (?:```)?                      # Optional opening code fence
    \s*                           # Whitespace after code fence
    (?:-\s*)?                     # Optional bullet prefix (- )
    \*\*Transform[ ]Kind[ ](\d+)\*\*  # Header with variant number (captured)
    \s*                           # Whitespace after header
    (.+?)                         # Variant content (captured, non-greedy)
    \s*                           # Trailing whitespace before code fence
    (?:```)?                      # Optional closing code fence
    \s*$                          # Trailing whitespace to end
    """,
    re.DOTALL | re.VERBOSE,
)


def parse_response(response_text: str) -> dict[str, str]:
    """Parse LLM response into {variant_num: text} dict."""
    if "-----------" not in response_text:
        logger.warning("Response missing separator, trying fallback parse")
        # Try splitting by **Transform Kind N** headers
        parts = re.split(r'\n(?=\*\*Transform Kind \d+\*\*)', response_text)
    else:
        parts = [p for p in response_text.split("-----------") if p.strip()]

    variants = {}
    for part in parts:
        m = _VARIANT_PATTERN.match(part)
        if m:
            num = m.group(1)
            text = m.group(2).strip()
            if text and num not in variants:
                variants[num] = text
        else:
            # Fallback: try simpler pattern
            m2 = re.search(r'\*\*Transform Kind (\d+)\*\*\s*\n(.+)', part, re.DOTALL)
            if m2:
                num = m2.group(1)
                text = m2.group(2).strip()
                if text and num not in variants:
                    variants[num] = text

    return variants


# ── LLM Caller ──────────────────────────────────────────────────────────────

def call_llm(prompt: str, max_retries: int = 3) -> str:
    """Call DashScope qwen3-coder-plus via OpenAI-compatible API."""
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"enable_thinking": False},
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    return ""


# ── Data Loading ────────────────────────────────────────────────────────────

def load_valid_django_bugs() -> list[dict[str, Any]]:
    """Load 76 valid Django synthetic bugs with PS and injection_patch."""
    # Load quality metrics to get valid instance IDs
    qm_data = json.loads(QUALITY_METRICS.read_text())
    valid_ids = {
        m["instance_id"]
        for m in qm_data
        if "django" in m.get("repo", "")
    }
    logger.info(f"Found {len(valid_ids)} valid Django instances in quality_metrics")

    # Load synthesis summary
    synth_data = json.loads(SYNTH_SUMMARY.read_text())
    synth_map = {s["synthetic_instance_id"]: s for s in synth_data}

    results = []
    for iid in sorted(valid_ids):
        synth = synth_map.get(iid)
        if not synth:
            logger.warning(f"No synthesis data for {iid}")
            continue
        results.append({
            "instance_id": iid,
            "problem_statement": synth.get("synthetic_problem_statement", ""),
            "patch": synth.get("synthetic_injection_patch", ""),
            "seed_id": synth.get("seed_instance_id", ""),
            "fix_intent": synth.get("synthetic_injection_strategy", ""),
        })

    logger.info(f"Loaded {len(results)} bugs with PS and patch")
    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Saving SWE-Bench Template Mutation Baseline")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt for first instance only")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances")
    args = parser.parse_args()

    rng = Random(SEED)
    bugs = load_valid_django_bugs()

    if args.limit:
        bugs = bugs[:args.limit]

    if args.dry_run:
        bug = bugs[0]
        prompt = format_mutation_prompt(bug["problem_statement"], bug["patch"])
        print(f"=== Prompt for {bug['instance_id']} ===")
        print(f"Prompt length: {len(prompt)} chars, ~{len(prompt.split())} words")
        print(f"\n{prompt[:2000]}...")
        return

    results = []
    total_variants = 0
    total_tokens_in = 0
    start_time = time.time()

    # Check for existing partial results (resume support)
    done_ids = set()
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        done_ids = {r["instance_id"] for r in existing}
        results = existing
        logger.info(f"Resuming: {len(done_ids)} already done")

    for i, bug in enumerate(bugs):
        if bug["instance_id"] in done_ids:
            continue

        logger.info(f"[{i+1}/{len(bugs)}] Mutating {bug['instance_id'][-30:]}")

        prompt = format_mutation_prompt(bug["problem_statement"], bug["patch"])
        prompt_words = len(prompt.split())

        try:
            response_text = call_llm(prompt)
            variants = parse_response(response_text)

            if variants:
                chosen_idx, chosen_text = rng.choice(list(variants.items()))
            else:
                chosen_idx, chosen_text = "-1", bug["problem_statement"]
                logger.warning(f"  No variants extracted, using original PS")

            result = {
                "instance_id": bug["instance_id"],
                "seed_id": bug["seed_id"],
                "fix_intent": bug["fix_intent"],
                "original_ps": bug["problem_statement"],
                "original_ps_tokens": len(bug["problem_statement"].split()),
                "mutated_ps": chosen_text,
                "mutated_ps_tokens": len(chosen_text.split()),
                "chosen_variant": int(chosen_idx),
                "num_variants": len(variants),
                "all_variants": variants,
            }
            results.append(result)
            total_variants += len(variants)

            status = f"v={len(variants)} chosen=T{chosen_idx} " \
                     f"({result['original_ps_tokens']}→{result['mutated_ps_tokens']} words)"
            logger.info(f"  {status}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                "instance_id": bug["instance_id"],
                "seed_id": bug["seed_id"],
                "fix_intent": bug["fix_intent"],
                "original_ps": bug["problem_statement"],
                "original_ps_tokens": len(bug["problem_statement"].split()),
                "mutated_ps": bug["problem_statement"],
                "mutated_ps_tokens": len(bug["problem_statement"].split()),
                "chosen_variant": -1,
                "num_variants": 0,
                "all_variants": {},
                "error": str(e),
            })

        # Save after each instance (crash resilience)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    elapsed = time.time() - start_time
    n = len(results)
    success = sum(1 for r in results if r["num_variants"] > 0)

    # Compute info-theory metrics for mutated PS
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.modules.synthesis.ps_chain_guided import compute_ps_metrics

    for r in results:
        bug = next((b for b in bugs if b["instance_id"] == r["instance_id"]), None)
        if bug:
            metrics = compute_ps_metrics(r["mutated_ps"], bug["patch"])
            r["mutated_metrics"] = metrics
            orig_metrics = compute_ps_metrics(r["original_ps"], bug["patch"])
            r["original_metrics"] = orig_metrics

    # Final save with metrics
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"TEMPLATE MUTATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"Total: {n}, Success: {success}/{n} ({100*success/max(n,1):.0f}%)")
    print(f"Avg variants per instance: {total_variants/max(success,1):.1f}")

    # Aggregate metrics
    orig_tokens = [r["original_ps_tokens"] for r in results]
    mut_tokens = [r["mutated_ps_tokens"] for r in results if r["num_variants"] > 0]
    orig_id = [r["original_metrics"]["identifier_density"] for r in results if "original_metrics" in r]
    mut_id = [r["mutated_metrics"]["identifier_density"] for r in results if "mutated_metrics" in r]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\nOriginal PS:  {avg(orig_tokens):.1f} tokens, ID density={avg(orig_id):.3f}")
    print(f"Mutated PS:   {avg(mut_tokens):.1f} tokens, ID density={avg(mut_id):.3f}")

    # Variant distribution
    from collections import Counter
    variant_counts = Counter(r["chosen_variant"] for r in results if r["num_variants"] > 0)
    print(f"\nChosen variant distribution: {dict(sorted(variant_counts.items()))}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
