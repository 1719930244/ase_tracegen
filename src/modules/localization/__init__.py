"""
Stage 0: Fault Localization module for TraceGen.

Generates `raw_output_loc` for instances that lack LocAgent data,
enabling multi-repo support beyond Django.
"""

from src.modules.localization.localizer import FaultLocalizer

__all__ = ["FaultLocalizer"]
