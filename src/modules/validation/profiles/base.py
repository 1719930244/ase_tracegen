"""
Base profile class and registry for repository configurations.

A profile encapsulates all repository-specific configuration needed
for validation: test commands, log parsing, Docker settings, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# Simple Registry (no Singleton complexity)
# =============================================================================

class Registry:
    """
    Simple registry for mapping repo identifiers to profile classes.

    Unlike SWE-smith's complex registry, this is a straightforward dict.
    """
    def __init__(self):
        self._profiles = {}

    def register(self, profile_class: type) -> None:
        """Register a profile class."""
        if profile_class.__name__ == "BaseProfile":
            return
        profile = profile_class()
        self._profiles[profile.name] = profile_class
        if hasattr(profile, 'repo_name'):
            self._profiles[profile.repo_name] = profile_class

    def get(self, name: str) -> 'BaseProfile':
        """Get a profile instance by name."""
        if name not in self._profiles:
            raise KeyError(f"No profile registered for: {name}")
        return self._profiles[name]()

    def register_instance(self, profile: 'BaseProfile') -> None:
        """Register a profile instance directly."""
        self._profiles[profile.name] = profile.__class__
        if hasattr(profile, 'repo_name'):
            self._profiles[profile.repo_name] = profile.__class__

    def list_profiles(self) -> list[str]:
        """List all registered profile names."""
        return list(self._profiles.keys())


# =============================================================================
# Base Profile Class
# =============================================================================

@dataclass
class BaseProfile(ABC):
    """
    Abstract base class for repository profiles.

    A profile defines:
    - How to run tests (test_cmd)
    - How to parse test output (log_parser)
    - Docker image configuration
    - Repository metadata

    Subclasses must implement:
    - log_parser: Parse test output into status map
    - test_cmd: Command to run tests

    Optional overrides:
    - get_test_cmd: Get test command with optional file filtering
    - get_test_files: Extract test file paths from instance
    """

    # Required: repository identification
    name: str = ""               # Profile name (e.g., "python", "go")
    owner: str = ""              # GitHub owner
    repo: str = ""               # GitHub repository name
    commit: str = ""             # Git commit hash

    # Docker configuration
    image_name: str = ""         # Docker image name
    platform: str = "linux/x86_64"
    arch: str = "x86_64"

    # Test configuration
    test_cmd: str = ""           # Base test command
    timeout: int = 90            # Test timeout (seconds)
    timeout_ref: int = 900       # Full test suite timeout

    # Language-specific settings
    language: str = "unknown"
    file_extensions: list[str] = field(default_factory=list)

    # Optional settings
    min_testing: bool = False    # Run only relevant tests
    min_pregold: bool = False    # Run pre-gold per-instance

    def __post_init__(self):
        """Generate repo_name after initialization."""
        if self.owner and self.repo and self.commit:
            self.repo_name = f"{self.owner}__{self.repo}.{self.commit[:8]}"

    @abstractmethod
    def log_parser(self, log: str) -> dict[str, str]:
        """
        Parse test output into a status map.

        Args:
            log: Raw test output string

        Returns:
            Dictionary mapping test names to status strings:
            {"test_file.py::TestClass::test_name": "PASSED", ...}
        """
        pass

    def get_test_cmd(
        self,
        instance: dict,
        f2p_only: bool = False,
    ) -> tuple[str, list[str]]:
        """
        Get the test command for an instance.

        Args:
            instance: Instance dict with potential test file info
            f2p_only: If True, only run FAIL_TO_PASS tests

        Returns:
            tuple: (test_command, test_files_list)
        """
        from src.modules.validation.constants import FAIL_TO_PASS, PASS_TO_PASS

        test_files = []

        # 优先使用 instance 中已设置的 test_cmd（由 adapter 生成）
        if "test_cmd" in instance and instance["test_cmd"]:
            return instance["test_cmd"], []

        if f2p_only and FAIL_TO_PASS in instance:
            # Extract test file paths from test names
            test_files = list(set([
                t.split("::")[0] for t in instance[FAIL_TO_PASS]
            ]))
            if test_files:
                return f"{self.test_cmd} {' '.join(test_files)}", test_files

        if self.min_testing and FAIL_TO_PASS in instance:
            f2p_files = [t.split("::")[0] for t in instance.get(FAIL_TO_PASS, [])]
            p2p_files = [t.split("::")[0] for t in instance.get(PASS_TO_PASS, [])]
            test_files = list(set(f2p_files + p2p_files))
            if test_files:
                return f"{self.test_cmd} {' '.join(test_files)}", test_files

        return self.test_cmd, []

    def get_test_files(self, instance: dict) -> tuple[list[str], list[str]]:
        """
        Extract test file paths from an instance.

        Args:
            instance: Instance dict with FAIL_TO_PASS and PASS_TO_PASS

        Returns:
            tuple: (f2p_files, p2p_files) - unique test file paths
        """
        from src.modules.validation.constants import FAIL_TO_PASS, PASS_TO_PASS

        def extract(tests: list[str]) -> list[str]:
            return sorted(list(set([
                t.split("::")[0] for t in tests
            ])))

        return (
            extract(instance.get(FAIL_TO_PASS, [])),
            extract(instance.get(PASS_TO_PASS, [])),
        )

    def get_test_cmd_from_file(self, test_file: str) -> str:
        """Get test command for a specific file."""
        return f"{self.test_cmd} {test_file}"

    def validate_image(self) -> bool:
        """Check if the Docker image exists."""
        from src.modules.validation.docker_utils import image_exists
        return image_exists(self.image_name)

    def pull_image(self) -> bool:
        """Pull the Docker image."""
        from src.modules.validation.docker_utils import pull_image
        return pull_image(self.image_name)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BaseProfile",
    "Registry",
]
