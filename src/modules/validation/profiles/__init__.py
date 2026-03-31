"""
Repository profiles for different programming languages.

A profile defines how to:
1. Run tests for a specific repository
2. Parse test output
3. Configure the Docker environment
"""

from src.modules.validation.profiles.base import BaseProfile, Registry
from src.modules.validation.profiles.python import PythonProfile

# Global profile registry
registry = Registry()

# Register built-in profiles
registry.register(PythonProfile)

__all__ = [
    "BaseProfile",
    "Registry",
    "PythonProfile",
    "registry",
]
