"""
Custom exceptions for TraceGen
"""

class TraceGenError(Exception):
    """Base class for all TraceGen exceptions."""
    pass

class GraphConstructionError(TraceGenError):
    """Raised when code graph construction fails."""
    pass

class LLMResponseError(TraceGenError):
    """Raised when LLM response is invalid or cannot be parsed."""
    pass

class ExtractionError(TraceGenError):
    """Raised when defect chain extraction fails."""
    pass

class SynthesisError(TraceGenError):
    """Raised when reproduction test synthesis fails."""
    pass

class GitOperationError(TraceGenError):
    """Raised when git operations (clone, checkout, diff) fail."""
    pass
