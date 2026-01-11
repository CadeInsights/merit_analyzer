"""Environment capture utilities for test runs."""

import os
import subprocess

from merit.testing.models import RunEnvironment


def _get_git_info() -> tuple[str | None, str | None, bool | None]:
    """Capture git metadata if available."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            timeout=1,
        )

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        ).stdout.strip()
        dirty = bool(status)

        return commit, branch, dirty

    except (subprocess.SubprocessError, FileNotFoundError):
        return None, None, None


def _filter_env_vars() -> dict[str, str]:
    """Capture and mask relevant environment variables."""
    allowlist = {
        "MODEL_VENDOR",
        "INFERENCE_VENDOR",
        "CLOUD_ML_REGION",
        "GOOGLE_CLOUD_PROJECT",
        "AWS_REGION",
    }

    captured = {}
    for key, value in os.environ.items():
        if key in allowlist:
            captured[key] = value

    sensitive_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    for key in sensitive_keys:
        if key in os.environ:
            val = os.environ[key]
            if len(val) > 4:
                captured[key] = f"***{val[-4:]}"
            else:
                captured[key] = "***"

    return captured


def capture_environment() -> RunEnvironment:
    """Capture current environment metadata."""
    commit, branch, dirty = _get_git_info()

    return RunEnvironment(
        commit_hash=commit,
        branch=branch,
        dirty=dirty,
        env_vars=_filter_env_vars(),
    )
