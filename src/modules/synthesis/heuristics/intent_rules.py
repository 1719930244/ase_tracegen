"""
Heuristic rules for Fix Intent transformation in bug synthesis.
Bug 合成中修复意图转换的启发式规则

Based on:
- IEEE 1044 Software Anomaly Classification
- APR (Automated Program Repair) Taxonomy
- Learning-based Patch Generation Research

This module defines which fix intents can be transformed into others
while maintaining semantic coherence.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class FixIntentCategory(str, Enum):
    """
    Top-level fix intent categories based on IEEE 1044.
    基于 IEEE 1044 的顶层修复意图分类
    """
    LOGIC = "logic"           # Control flow related
    INTERFACE = "interface"   # Function calls / API related
    DATA = "data"            # Values, types, initialization
    STRUCTURE = "structure"  # Statement-level changes


# -----------------------------------------------------------------------------
# Fix Intent Injection Rules
# 修复意图注入规则
# 
# Maps source fix intent -> list of compatible target intents for bug injection
# Each target intent can create a bug that would require the source intent to fix
# -----------------------------------------------------------------------------

INTENT_INJECTION_RULES: Dict[str, Dict[str, Any]] = {
    # LOGIC Category
    # 逻辑类
    "Condition_Refinement": {
        "category": FixIntentCategory.LOGIC,
        "description": "Modify boolean logic in control flow (if, while, for)",
        "compatible_injections": [
            "Condition_Refinement",   # Same type: invert or modify condition
            "Guard_Clause_Addition",  # Upgrade: add missing guard
        ],
        "injection_strategies": [
            "invert_condition",       # if x > 0 -> if x <= 0
            "remove_condition_part",  # if x > 0 and y < 10 -> if x > 0
            "change_operator",        # if x >= 0 -> if x > 0
        ],
        "typical_patterns": [
            "Change '>' to '>=' or vice versa",
            "Add/remove 'and'/'or' clauses",
            "Invert boolean expression with 'not'"
        ]
    },
    
    "Guard_Clause_Addition": {
        "category": FixIntentCategory.LOGIC,
        "description": "Insert check for None or invalid states to prevent crashes",
        "compatible_injections": [
            "Guard_Clause_Addition",
            "Condition_Refinement",
        ],
        "injection_strategies": [
            "remove_null_check",      # Remove 'if x is None: return'
            "remove_empty_check",     # Remove 'if not items: return []'
            "remove_type_guard",      # Remove 'if not isinstance(x, str)'
        ],
        "typical_patterns": [
            "Remove 'if x is None: return' guard",
            "Remove 'if not items:' empty check",
            "Remove isinstance() type guard"
        ]
    },
    
    "Exception_Fix": {
        "category": FixIntentCategory.LOGIC,
        "description": "Modify try/except blocks, exception types, or handling logic",
        "compatible_injections": [
            "Exception_Fix",
            "Guard_Clause_Addition",
        ],
        "injection_strategies": [
            "wrong_exception_type",   # except ValueError -> except TypeError
            "missing_except",         # Remove except block
            "too_broad_except",       # except Exception -> except ValueError
        ],
        "typical_patterns": [
            "Catch wrong exception type",
            "Remove necessary try/except",
            "Make exception handling too broad or narrow"
        ]
    },
    
    # INTERFACE Category
    # 接口类
    "Argument_Update": {
        "category": FixIntentCategory.INTERFACE,
        "description": "Modify function call arguments (values, keywords, count)",
        "compatible_injections": [
            "Argument_Update",
            "Constant_Update",        # Closely related: literal argument
            "Variable_Replacement",   # Related: wrong variable as argument
        ],
        "injection_strategies": [
            "remove_argument",        # func(a, b) -> func(a)
            "wrong_argument_order",   # func(a, b) -> func(b, a)
            "missing_keyword",        # func(x=1) -> func(1)
            "wrong_default",          # func(timeout=30) -> func(timeout=5)
        ],
        "typical_patterns": [
            "Remove required argument",
            "Swap argument order",
            "Use wrong keyword argument",
            "Pass incorrect default value"
        ]
    },
    
    "API_Replacement": {
        "category": FixIntentCategory.INTERFACE,
        "description": "Replace function/method name (e.g., load -> loads)",
        "compatible_injections": [
            "API_Replacement",
            "Argument_Update",        # API change often needs arg changes
        ],
        "injection_strategies": [
            "similar_api_confusion",  # json.load -> json.loads
            "deprecated_api",         # Use old API instead of new
            "wrong_module_func",      # os.path.join -> os.join
        ],
        "typical_patterns": [
            "Use json.load instead of json.loads",
            "Use deprecated API method",
            "Call method from wrong module"
        ]
    },
    
    # DATA Category
    # 数据类
    "Variable_Replacement": {
        "category": FixIntentCategory.DATA,
        "description": "Replace variable identifier with another compatible variable",
        "compatible_injections": [
            "Variable_Replacement",
            "Constant_Update",
        ],
        "injection_strategies": [
            "typo_variable",          # width -> widht
            "similar_name",           # items -> item
            "scope_confusion",        # self.x -> x
        ],
        "typical_patterns": [
            "Typo in variable name",
            "Use singular instead of plural (or vice versa)",
            "Confuse instance vs local variable"
        ]
    },
    
    "Constant_Update": {
        "category": FixIntentCategory.DATA,
        "description": "Modify hardcoded literals (numbers, strings, regex)",
        "compatible_injections": [
            "Constant_Update",
            "Variable_Replacement",
        ],
        "injection_strategies": [
            "wrong_number",           # 100 -> 10
            "wrong_string",           # "GET" -> "POST"
            "wrong_regex",            # r'\d+' -> r'\d*'
            "boundary_error",         # range(10) -> range(9)
        ],
        "typical_patterns": [
            "Off-by-one in numeric constant",
            "Wrong string literal",
            "Incorrect regex pattern"
        ]
    },
    
    "Type_Cast_Fix": {
        "category": FixIntentCategory.DATA,
        "description": "Add explicit type conversion (int(), str(), etc.)",
        "compatible_injections": [
            "Type_Cast_Fix",
            "Variable_Replacement",
        ],
        "injection_strategies": [
            "remove_type_cast",       # int(x) -> x
            "wrong_type_cast",        # int(x) -> str(x)
        ],
        "typical_patterns": [
            "Remove necessary int()/str()/float() conversion",
            "Use wrong type conversion function"
        ]
    },
    
    "Data_Initialization": {
        "category": FixIntentCategory.DATA,
        "description": "Add missing attribute or variable initialization",
        "compatible_injections": [
            "Data_Initialization",
            "Guard_Clause_Addition",
        ],
        "injection_strategies": [
            "remove_init",            # Remove self.x = None in __init__
            "wrong_init_value",       # self.cache = {} -> self.cache = None
        ],
        "typical_patterns": [
            "Remove variable initialization",
            "Initialize with wrong default value"
        ]
    },
    
    # STRUCTURE Category (Fallback)
    # 结构类（兜底）
    "Statement_Insertion": {
        "category": FixIntentCategory.STRUCTURE,
        "description": "Insert non-control statements (logging, config, cleanup)",
        "compatible_injections": [
            "Statement_Insertion",
        ],
        "injection_strategies": [
            "remove_statement",       # Remove necessary statement
            "reorder_statements",     # Change statement order
        ],
        "typical_patterns": [
            "Remove necessary configuration statement",
            "Remove cleanup/resource release statement"
        ]
    },
    
    "Complex_Logic_Rewrite": {
        "category": FixIntentCategory.STRUCTURE,
        "description": "Multi-line changes spanning multiple dimensions (fallback)",
        "compatible_injections": [
            "Complex_Logic_Rewrite",
            "Condition_Refinement",
            "Statement_Insertion",
        ],
        "injection_strategies": [
            "simplify_logic",         # Oversimplify complex logic
            "partial_implementation", # Incomplete implementation
        ],
        "typical_patterns": [
            "Oversimplify complex conditional logic",
            "Remove part of multi-step process"
        ]
    }
}


@dataclass
class InjectionSuggestion:
    """
    A suggestion for bug injection.
    Bug 注入建议
    """
    source_intent: str           # Original fix intent from seed
    target_intent: str           # Suggested injection type
    strategy: str                # Specific injection strategy
    description: str             # Human-readable description
    confidence: float            # Confidence score (0-1)
    example_pattern: str         # Example of the pattern


class FixIntentTransformer:
    """
    Transforms fix intents to generate bug injection suggestions.
    将修复意图转换为 Bug 注入建议
    
    Given a seed's fix intent, this class provides:
    - Compatible injection types
    - Specific injection strategies
    - Confidence scores for each suggestion
    """
    
    def __init__(self, rules: Dict[str, Dict[str, Any]] = None):
        """
        Initialize transformer with injection rules.
        使用注入规则初始化转换器
        
        Args:
            rules: Custom rules dict, defaults to INTENT_INJECTION_RULES
        """
        self.rules = rules or INTENT_INJECTION_RULES
    
    def get_injection_suggestions(
        self,
        source_intent: str,
        max_suggestions: int = 5
    ) -> List[InjectionSuggestion]:
        """
        Get bug injection suggestions for a given fix intent.
        为给定的修复意图获取 Bug 注入建议
        
        Args:
            source_intent: The fix intent type from the seed (e.g., "Condition_Refinement")
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of InjectionSuggestion objects
        """
        if source_intent not in self.rules:
            logger.warning(f"Unknown fix intent: {source_intent}")
            # Return generic suggestions
            # 返回通用建议
            return [
                InjectionSuggestion(
                    source_intent=source_intent,
                    target_intent="Condition_Refinement",
                    strategy="generic_logic_change",
                    description="Apply generic logic modification",
                    confidence=0.3,
                    example_pattern="Modify conditional logic"
                )
            ]
        
        rule = self.rules[source_intent]
        suggestions = []
        
        # Generate suggestions from compatible injections
        # 从兼容注入生成建议
        compatible = rule.get("compatible_injections", [])
        strategies = rule.get("injection_strategies", [])
        patterns = rule.get("typical_patterns", [])
        
        for i, target_intent in enumerate(compatible):
            # Same intent type gets highest confidence
            # 相同意图类型获得最高置信度
            if target_intent == source_intent:
                confidence = 0.9
            else:
                confidence = 0.7 - (i * 0.1)
            
            # Get strategy and pattern
            # 获取策略和模式
            strategy = strategies[min(i, len(strategies) - 1)] if strategies else "default"
            pattern = patterns[min(i, len(patterns) - 1)] if patterns else "Generic pattern"
            
            target_rule = self.rules.get(target_intent, {})
            
            suggestions.append(InjectionSuggestion(
                source_intent=source_intent,
                target_intent=target_intent,
                strategy=strategy,
                description=target_rule.get("description", f"Apply {target_intent}"),
                confidence=max(0.1, confidence),
                example_pattern=pattern
            ))
        
        # Sort by confidence and limit
        # 按置信度排序并限制数量
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:max_suggestions]
    
    def get_all_strategies(self, intent: str) -> List[str]:
        """
        Get all injection strategies for an intent.
        获取意图的所有注入策略
        
        Args:
            intent: Fix intent type
            
        Returns:
            List of strategy names
        """
        rule = self.rules.get(intent, {})
        return rule.get("injection_strategies", [])
    
    def get_category(self, intent: str) -> Optional[FixIntentCategory]:
        """
        Get the category of a fix intent.
        获取修复意图的类别
        
        Args:
            intent: Fix intent type
            
        Returns:
            Category enum or None
        """
        rule = self.rules.get(intent, {})
        return rule.get("category")
    
    def list_all_intents(self) -> List[str]:
        """
        List all available fix intent types.
        列出所有可用的修复意图类型
        
        Returns:
            List of intent names
        """
        return list(self.rules.keys())
    
    def to_prompt_format(self, source_intent: str) -> str:
        """
        Generate a formatted string for LLM prompts.
        生成用于 LLM 提示词的格式化字符串
        
        Args:
            source_intent: The source fix intent
            
        Returns:
            Formatted string describing injection options
        """
        suggestions = self.get_injection_suggestions(source_intent)
        
        lines = [f"## Bug Injection Options for '{source_intent}'", ""]
        
        for i, sugg in enumerate(suggestions, 1):
            lines.append(f"### Option {i}: {sugg.target_intent}")
            lines.append(f"- Strategy: {sugg.strategy}")
            lines.append(f"- Description: {sugg.description}")
            lines.append(f"- Example: {sugg.example_pattern}")
            lines.append(f"- Confidence: {sugg.confidence:.1%}")
            lines.append("")
        
        return "\n".join(lines)
