"""
Prompt templates for Stage 0 fault localization.

Output format is compatible with LocAgent's raw_output_loc format,
so Stage 1's LOC_EXTRACTION_PROMPT can consume it directly.
"""

LOCALIZATION_PROMPT = """You are an expert fault localizer for the {repo} project.

## Task
Given a bug report and the developer's fix patch, analyze the causal chain from the observable symptom to the root cause.

## Bug Report
{problem_statement}

## Developer's Fix Patch
{patch}

## Static Analysis Context
- Modified files: {found_files}
- Modified entities: {found_entities}
- Test entry points: {test_files}
{graph_path_context}

## Instructions
1. Identify the symptom: what error/behavior does the user observe?
2. Trace the execution path from the test/entry point to the buggy code.
3. Identify intermediate functions/methods in the call chain.
4. Pinpoint the root cause: which specific function/line contains the defect?

## Output Format
List the key locations in order of the call chain (from symptom to root cause):

```
file_path_1.py
line: <start_line>
function: <qualified_name>
Description: <role in the chain, e.g. "test entry point", "intermediate caller", "root cause">

file_path_2.py
line: <start_line>
class: <class_name>
function: <method_name>
Description: <role in the chain>
```

Use EXACT file paths from the repository. Include about 3-5 locations.
Return just the locations.
"""
