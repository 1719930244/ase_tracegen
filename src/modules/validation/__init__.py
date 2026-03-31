"""
Bug validation module for TraceGen.

This module provides Docker-based validation of synthesized bugs,
ensuring they cause the expected test failures.
"""

from src.modules.validation.constants import (
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.modules.validation.validator import Validator
from src.modules.validation.adapter import ValidationAdapter

__all__ = [
    "Validator",
    "ValidationAdapter",
    "ValidationConfig",
    "ValidationResult",
    "ValidationStatus",
]
