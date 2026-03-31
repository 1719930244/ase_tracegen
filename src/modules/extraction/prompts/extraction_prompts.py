"""
链路提取阶段的所有 Prompt 模板
"""

# -----------------------------------------------------------------------------
# 1. 定位链路提取 (Localization Chain)
# -----------------------------------------------------------------------------
LOC_EXTRACTION_PROMPT = """You are an expert in code defect localization and chain extraction.

## Task
Extract a defect chain from the localization log. The chain should follow the format:
- symptom_node: entry point where error occurs
- chain: list of (source_node, target_node, relation_type) tuples showing the call path
- root_cause_node: the final node where the root cause is located

## Instance Information
- Instance ID: {instance_id}
- Repository: {repo}
- Problem: {problem_statement}

## Localization Log
{loc_text}

## Instructions
1. Analyze the localization log to identify key nodes
2. Find the symptom node (entry point where the error is triggered)
3. Trace through the code to identify intermediate nodes in the call chain
4. Identify the root cause node (the deepest/final node causing the issue)
5. Return JSON with all nodes using the EXACT format specified below

## CRITICAL: Node ID Format
**DO NOT include line numbers in node IDs.** Use this exact format:
  `path/to/file.py:function_name` or `path/to/file.py:ClassName.method_name`

Examples of CORRECT format:
  - `django/urls/resolvers.py:ResolverMatch.__repr__`
  - `astropy/io/ascii/qdp.py:_line_type`
  - `django/utils/decorators.py:method_decorator`

Examples of WRONG format (DO NOT USE):
  - `django/urls/resolvers.py:61-65:ResolverMatch.__repr__` (has line numbers)
  - `django/urls/resolvers.py:61:ResolverMatch.__repr__` (has line number)

{graph_nodes_sample}

## Response Format
Return ONLY valid JSON:
{{
  "instance_id": "{instance_id}",
  "chain_type": "loc_chain",
  "reasoning_trace": [
    {{ "step": 1, "action": "Analyze_Logs", "thought": "..." }},
    {{ "step": 2, "action": "Trace_Chain", "thought": "..." }}
  ],
  "loc_chain": [
    ["path/file.py:func_or_Class.method", "path/file.py:func_or_Class.method", "invokes"],
    ["path/file.py:func_or_Class.method", "path/file.py:func_or_Class.method", "invokes"]
  ]
}}
"""

# -----------------------------------------------------------------------------
# 2. 修复链路提取 (Repair Chain)
# -----------------------------------------------------------------------------
REPAIR_EXTRACTION_PROMPT = """# Role
You are a Senior Automated Program Repair (APR) Specialist.
Your task is to analyze a bug fix and extract a **Structured, Semantic Repair Chain** based on a strict taxonomy.

# Task Definition
We need to understand the **LOGIC** of the fix (the "What"), decoupled from the editing tools (the "How").
You must categorize the fix into specific AST-level operators.

# Inputs

## 1. Issue Description
{problem_statement}

## 2. Localization Context
{localization_summary}
**Target Node:** `{root_cause_node_id}`

## 3. Buggy Code
```python
{buggy_code_snippet}
```

## 4. Ground Truth Patch (Diff)
```diff
{patch_diff}
```

# Taxonomy Reference (Classification Standards)
You MUST classify the ast_operation.type into EXACTLY ONE of the following categories based on the Root Intent of the fix.

A. LOGIC Dimension (Control Flow)
- Condition_Refinement: Modifying the boolean logic in if, while, or for loops (e.g., changing > to >=, adding and/or conditions).
- Guard_Clause_Addition: Inserting a NEW check at the beginning of a block to handle None, empty lists, or invalid states to prevent crashes (e.g., if x is None: return).
- Exception_Fix: Adding try/except blocks, changing caught exception types, or modifying the exception handling logic.

B. INTERFACE Dimension (Function Calls)
- Argument_Update: Changing the values, keywords, or number of arguments passed to a function call (e.g., func(x) -> func(x, timeout=10)).
- API_Replacement: Swapping the called function name itself (e.g., json.load -> json.loads).

C. DATA Dimension (Values, Types & Init)
- Variable_Replacement: Replacing one variable identifier with another compatible variable in scope (e.g., typo fix widht -> width or logic fix x -> y).
- Constant_Update: Modifying hardcoded literals, including numbers, strings, and Regex patterns.
- Type_Cast_Fix (Python Specific): Adding explicit type conversion or casting to fix Runtime Type Errors (e.g., x -> int(x), str(x)).
- Data_Initialization (Python Specific): Adding missing attribute definition or variable initialization (e.g., self.cache = {{}}) to prevent AttributeError.

D. STRUCTURE Dimension
- Statement_Insertion: Adding necessary non-control logic (e.g., logging, config setting, resource cleanup) that doesn't fit above.
- Complex_Logic_Rewrite: (Fallback) Multi-line changes that mix multiple dimensions above and cannot be described by a single atomic operator.

# Instructions
1. Analyze the Diff: Compare the Buggy Code and the Patch. Ignore indentation changes.
2. Abstract the Change: Create a "Logical Snapshot" of the change.
3. Select Category: Check Python-specific first, then Logic/Interface, use Complex_Logic_Rewrite as last resort.

# Output Format (JSON Only)
Return ONLY a valid JSON object:
{{
  "instance_id": "{instance_id}",
  "chain_type": "repair_chain",
  "reasoning_trace": [
    {{ "step": 1, "action": "Analyze_Diff", "thought": "..." }},
    {{ "step": 2, "action": "Classify", "thought": "..." }}
  ],
  "repair_chain": {{
    "type": "Type_Cast_Fix",
    "target_node": "{root_cause_node_id}",
    "target_obj": "variable_name",
    "code_transformation": {{
      "before": "code_snapshot_before",
      "after": "code_snapshot_after"
    }},
    "summary": "Short description of intent.",
    "metrics": {{
      "operator_category": "Data",
      "complexity_level": "Low",
      "python_specific": true
    }}
  }}
}}
"""
