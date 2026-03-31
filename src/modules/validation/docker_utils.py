"""
Docker/Podman container utilities for running tests in isolated environments.

This module provides functions to:
1. Create and manage Docker/Podman containers
2. Apply patches to code in containers
3. Run test commands and capture output
4. Clean up resources

Supports both Docker and Podman (via socket compatibility).
"""

import os
import docker
import traceback
from dataclasses import dataclass
from datetime import datetime
from docker.models.containers import Container
from loguru import logger
from pathlib import Path
from typing import Callable, Optional

from src.modules.validation.constants import (
    DOCKER_USER,
    DOCKER_WORKDIR,
    DOCKER_PATCH,
    GIT_APPLY_CMDS,
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
    TESTS_TIMEOUT,
    UTF8,
    KEY_INSTANCE_ID,
)


# =============================================================================
# Client Management (Docker/Podman compatibility)
# =============================================================================

_client = None

def get_container_client():
    """
    Get a Docker/Podman client with automatic socket detection.

    Checks for Podman socket first, then falls back to Docker.
    Supports both rootful and rootless Podman.

    Returns:
        docker.DockerClient instance
    """
    global _client
    if _client is not None:
        return _client

    # Podman socket paths to check (in order of preference)
    podman_sockets = [
        "/run/podman/podman.sock",  # Rootful Podman
        f"/run/user/{os.getuid()}/podman/podman.sock",  # Rootless Podman
    ]

    # Check for Podman sockets
    for socket_path in podman_sockets:
        if os.path.exists(socket_path):
            try:
                _client = docker.DockerClient(base_url=f"unix://{socket_path}")
                _client.ping()  # Test connection
                return _client
            except Exception:
                continue

    # Fall back to default Docker
    try:
        _client = docker.from_env()
        _client.ping()
        return _client
    except Exception as e:
        raise ContainerError(f"无法连接到 Docker/Podman: {e}")


# =============================================================================
# Exceptions
# =============================================================================

class ContainerError(Exception):
    """Base exception for container-related errors."""
    pass


class PatchApplicationError(ContainerError):
    """Raised when patch application fails."""
    pass


class TestExecutionError(ContainerError):
    """Raised when test execution fails."""
    pass


class ContainerTimeoutError(ContainerError):
    """Raised when container execution times out."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecResult:
    """Result of a container exec command."""
    exit_code: int
    output: str
    error: str = ""
    timed_out: bool = False
    runtime: float = 0


# =============================================================================
# Container Management
# =============================================================================

def create_container(
    image_name: str,
    instance_id: str,
    platform: str = "linux/x86_64",
    memory_limit: str = "10g",
) -> Container:
    """
    Create a Docker container for running tests.

    Args:
        image_name: Docker image to use
        instance_id: Unique identifier for the instance
        platform: Platform specification (e.g., linux/x86_64)
        memory_limit: Memory limit for container

    Returns:
        Created Docker container object
    """
    client = get_container_client()

    # Generate unique container name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    container_name = f"validator_{instance_id}_{timestamp}"

    # Determine user based on image type
    # Only use DOCKER_USER (swesmith) for SWE-smith images (swesmith.x86_64)
    # SWE-bench eval images (sweb.eval.x86_64) use root by default
    user = None
    if "swesmith.x86_64" in image_name.lower():
        user = DOCKER_USER

    container = client.containers.create(
        image=image_name,
        name=container_name,
        user=user,
        detach=True,
        command="tail -f /dev/null",
        platform=platform,
        mem_limit=memory_limit,
    )

    return container


def start_container(container: Container) -> None:
    """Start a Docker container."""
    container.start()


def stop_container(container: Container) -> None:
    """Stop a Docker container."""
    try:
        container.stop(timeout=5)
    except docker.errors.APIError as e:
        logger.debug(f"Container stop failed (may already be stopped): {e}")


def remove_container(container: Container) -> None:
    """Remove a Docker container."""
    try:
        container.remove(force=True)
    except docker.errors.APIError as e:
        logger.debug(f"Container remove failed (may already be removed): {e}")


def cleanup_container(container: Optional[Container]) -> None:
    """
    Clean up a container by stopping and removing it.

    Args:
        container: Container to clean up (can be None)
    """
    if container is None:
        return

    try:
        stop_container(container)
    except Exception as e:
        logger.warning(f"Failed to stop container: {e}")

    try:
        remove_container(container)
    except Exception as e:
        logger.warning(f"Failed to remove container: {e}")


# =============================================================================
# Command Execution
# =============================================================================

def exec_command(
    container: Container,
    command,
    workdir: str = None,
    user: str = None,
) -> ExecResult:
    """
    Execute a command in a container.

    Args:
        container: Docker container
        command: Command to execute
        workdir: Working directory for command (None = container default)
        user: User to run command as (None = container default)

    Returns:
        ExecResult with exit code and output
    """
    import time
    start_time = time.time()

    # Build exec_run kwargs, only include non-None values
    kwargs = {}
    if workdir:
        kwargs['workdir'] = workdir
    if user:
        kwargs['user'] = user

    exit_code, output = container.exec_run(command, **kwargs)

    runtime = time.time() - start_time

    # Decode output
    decoded_output = output.decode(UTF8) if output else ""

    return ExecResult(
        exit_code=exit_code,
        output=decoded_output,
        runtime=runtime,
    )


def exec_with_timeout(
    container: Container,
    command,
    timeout: int,
    workdir: str = None,
    user: str = None,
) -> ExecResult:
    """
    Execute a command with a timeout.

    Args:
        container: Docker container
        command: Command to execute
        timeout: Timeout in seconds
        workdir: Working directory for command (None = container default)
        user: User to run command as (None = container default)

    Returns:
        ExecResult with timeout status
    """
    """
    IMPORTANT:
    The previous implementation used a background thread and `join(timeout=...)`.
    That marks the command as timed out, but **does not stop** the underlying
    `container.exec_run(...)` call, so the process keeps running in the container.
    Over time this leaks long-lived containers/processes and can stall the whole run.

    We instead enforce timeout *inside* the container using coreutils `timeout`,
    which actually terminates the command process tree.
    """
    if timeout is None:
        timeout = 0
    timeout = int(timeout or 0)
    if timeout <= 0:
        return exec_command(container, command, workdir, user)

    # Build a safe "timeout ..." wrapper:
    # - If `command` is a list/tuple (exec form), we can directly prefix it.
    # - If `command` is a string, run it through `bash -lc` so shell features work.
    #
    # `timeout` exit code:
    # - 124: timed out
    # - 137: killed (SIGKILL) in some cases
    if isinstance(command, (list, tuple)):
        wrapped = ["timeout", "-k", "5", str(timeout), *list(command)]
    else:
        wrapped = ["timeout", "-k", "5", str(timeout), "/bin/bash", "-lc", str(command)]

    res = exec_command(container, wrapped, workdir, user)
    if res.exit_code in {124, 137}:
        res.timed_out = True
        if not res.error:
            res.error = f"Command timed out after {timeout} seconds"
    return res


# =============================================================================
# Patch Application
# =============================================================================

def apply_patch(
    container: Container,
    patch_content: str,
    reverse: bool = False,
    allow_rejects: bool = False,
) -> None:
    """
    Apply a patch to the code in a container.

    Args:
        container: Docker container
        patch_content: Patch diff content
        reverse: If True, apply patch in reverse (for reverting changes)
        allow_rejects: If True, allow partial application via `git apply --reject`
            when at least some hunks apply (useful when patches contain redundant
            hunks that are already applied).

    Raises:
        PatchApplicationError: If patch application fails
    """
    if not patch_content or not patch_content.strip():
        return  # No patch to apply

    # Write patch content to container using exec + echo (Podman compatible)
    # Escape special characters for shell
    import base64
    patch_b64 = base64.b64encode(patch_content.encode(UTF8)).decode('ascii')

    # Use base64 decode to write file (avoids shell escaping issues)
    # IMPORTANT: Must use /bin/bash -c to execute pipe commands in container
    write_cmd = f"/bin/bash -c \"echo '{patch_b64}' | base64 -d > {DOCKER_PATCH}\""
    result = exec_command(container, write_cmd)

    if result.exit_code != 0:
        raise PatchApplicationError(f"Failed to write patch file: {result.error}")

    def _run_in_repo(cmd: str) -> ExecResult:
        # IMPORTANT: Must use /bin/bash -c and cd to DOCKER_WORKDIR
        full_cmd = f"/bin/bash -c 'cd {DOCKER_WORKDIR} && {cmd}'"
        return exec_command(container, full_cmd)

    def _format_cmd(cmd_template: str) -> str:
        if "{patch}" in cmd_template:
            cmd = cmd_template.format(patch=DOCKER_PATCH)
        else:
            cmd = f"{cmd_template} {DOCKER_PATCH}"
        if reverse:
            cmd = f"{cmd} --reverse"
        return cmd

    if not allow_rejects:
        last_result: ExecResult | None = None
        for cmd_template in GIT_APPLY_CMDS:
            cmd = _format_cmd(cmd_template)
            last_result = _run_in_repo(cmd)
            if last_result.exit_code == 0:
                return  # Success
        details = ""
        if last_result is not None:
            snippet = (last_result.output or last_result.error or "").strip()
            if snippet:
                details = f" Last output: {snippet[:400]}"
        raise PatchApplicationError(
            f"Failed to apply patch. Tried commands: {GIT_APPLY_CMDS}.{details}"
        )

    # allow_rejects=True: prefer git-apply based strategies; avoid `patch` tool
    # auto-reverse behavior and rely on diffstat to detect partial success.
    before_stat = _run_in_repo("git diff --numstat").output

    # First try strict apply.
    strict = _run_in_repo(_format_cmd("git apply --verbose --recount"))
    if strict.exit_code == 0:
        return

    # Then try reject mode. It may return non-zero even if some hunks applied.
    reject = _run_in_repo(_format_cmd("git apply --verbose --recount --reject"))
    after_stat = _run_in_repo("git diff --numstat").output

    if after_stat != before_stat:
        # Clean up any reject/orig artifacts so they won't affect tests.
        _run_in_repo("find . \\( -name '*.rej' -o -name '*.orig' \\) -exec rm -f {} +")
        return

    snippet = (reject.output or reject.error or strict.output or strict.error or "").strip()
    details = f" Last output: {snippet[:400]}" if snippet else ""
    raise PatchApplicationError(
        f"Failed to apply patch (allow_rejects=True).{details}"
    )


def copy_file_to_container(container: Container, local_path: Path, container_path: str) -> None:
    """
    Copy a file from host to container.

    Args:
        container: Docker container
        local_path: Path to local file
        container_path: Destination path in container
    """
    import base64

    # Read file content
    with open(local_path, 'rb') as f:
        content = f.read()

    # Use base64 to transfer file (Podman compatible)
    content_b64 = base64.b64encode(content).decode('ascii')

    # Ensure parent directory exists
    parent_dir = str(Path(container_path).parent)
    exec_command(container, f"/bin/bash -c \"mkdir -p {parent_dir}\"")

    # Write file using base64 decode
    # IMPORTANT: Must use /bin/bash -c to execute pipe commands in container
    write_cmd = f"/bin/bash -c \"echo '{content_b64}' | base64 -d > {container_path}\""
    result = exec_command(container, write_cmd)

    if result.exit_code != 0:
        raise ContainerError(f"Failed to copy file to container: {result.error}")


# =============================================================================
# Test Execution
# =============================================================================

def create_test_script(test_command: str) -> str:
    """
    Create a shell script for running tests.

    Args:
        test_command: Test command to execute

    Returns:
        Shell script content
    """
    return "\n".join([
        "#!/bin/bash",
        "set -uxo pipefail",
        # 激活 miniconda testbed 环境（SWE-bench 容器的标准配置）
        "export PATH=/opt/miniconda3/bin:$PATH",
        "source /opt/miniconda3/bin/activate testbed 2>/dev/null || source /opt/miniconda3/bin/activate 2>/dev/null || true",
        # 确保 testbed 环境的 bin 优先于 base conda（修复 #!/usr/bin/env python shebang 解析问题）
        "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH",
        # Limit native thread pools to avoid massive oversubscription when running many
        # validations concurrently (common for NumPy/SciPy/OpenBLAS/MKL/numexpr).
        # Keep user-provided values if already set.
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}",
        "export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}",
        "export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}",
        "export BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}",
        f"cd {DOCKER_WORKDIR}",
        f": '{TEST_OUTPUT_START}'",
        test_command,
        f": '{TEST_OUTPUT_END}'",
    ])


def run_test_in_container(
    container: Container,
    test_command: str,
    timeout: int,
) -> ExecResult:
    """
    Run tests in a container.

    Args:
        container: Docker container
        test_command: Test command to execute
        timeout: Timeout in seconds

    Returns:
        ExecResult with test output
    """
    import base64

    script_content = create_test_script(test_command)

    # Write script to container using base64 (Podman compatible)
    # IMPORTANT: Use /bin/bash -c to execute pipe commands
    script_b64 = base64.b64encode(script_content.encode(UTF8)).decode('ascii')
    write_cmd = f"/bin/bash -c \"echo '{script_b64}' | base64 -d > /tmp/eval.sh\""
    exec_command(container, write_cmd)

    # Make executable and run
    exec_command(container, "chmod +x /tmp/eval.sh")

    result = exec_with_timeout(
        container,
        "/bin/bash /tmp/eval.sh",
        timeout=timeout,
    )

    # Add timeout marker if needed
    if result.timed_out:
        result.output += f"\n\n{TESTS_TIMEOUT}: {timeout} seconds exceeded"

    return result


# =============================================================================
# Image Utilities
# =============================================================================

def image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        client = get_container_client()
        client.images.get(image_name)
        return True
    except Exception:
        return False


def pull_image(image_name: str, timeout: int = 300) -> bool:
    """
    Pull a Docker image from registry using subprocess.

    Uses subprocess to call podman/docker directly, which allows
    proper proxy configuration through environment variables.

    Args:
        image_name: Name of image to pull
        timeout: Timeout in seconds for pull operation

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    import os

    # 确定使用 podman 还是 docker
    container_cmd = "podman" if os.path.exists("/usr/bin/podman") else "docker"

    try:
        # 使用 subprocess 调用，继承环境变量（包括代理设置）
        result = subprocess.run(
            [container_cmd, "pull", image_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy()  # 继承当前环境变量
        )

        if result.returncode == 0:
            return True
        else:
            logger.warning(f"Failed to pull image {image_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Image pull timed out after {timeout}s: {image_name}")
        return False
    except Exception as e:
        logger.warning(f"Failed to pull image {image_name}: {e}")
        return False


def list_images() -> list[str]:
    """List available Docker images."""
    try:
        client = get_container_client()
        images = client.images.list()
        return [img.tags[0] if img.tags else img.id for img in images]
    except Exception:
        return []


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ContainerError",
    "PatchApplicationError",
    "TestExecutionError",
    "ContainerTimeoutError",
    "ExecResult",
    "create_container",
    "start_container",
    "stop_container",
    "remove_container",
    "cleanup_container",
    "exec_command",
    "exec_with_timeout",
    "apply_patch",
    "run_test_in_container",
    "image_exists",
    "pull_image",
    "list_images",
]
