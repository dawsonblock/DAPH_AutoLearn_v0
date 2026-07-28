from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import json
import hashlib
from typing import Sequence

from .concrete import ExecutionSummary


class SandboxExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DockerSandboxConfig:
    image: str
    cpus: float = 1.0
    memory: str = "4g"
    timeout_seconds: int = 30
    pids_limit: int = 256
    network: str = "none"
    read_only_root: bool = True
    workspace_mode: str = "tmpfs_copy"  # safer compatibility than a writable host bind

    def __post_init__(self) -> None:
        if not self.image.strip():
            raise ValueError("image must be non-empty")
        if self.cpus <= 0 or self.timeout_seconds <= 0 or self.pids_limit <= 0:
            raise ValueError("cpus, timeout_seconds, and pids_limit must be > 0")
        if self.workspace_mode not in {"tmpfs_copy", "read_only"}:
            raise ValueError("workspace_mode must be 'tmpfs_copy' or 'read_only'")


_PYTEST_COUNT = re.compile(r"(?P<n>\d+)\s+(?P<kind>passed|failed|error|errors|skipped|xfailed|xpassed)")


def parse_pytest_summary(output: str) -> ExecutionSummary:
    counts: dict[str, int] = {}
    for match in _PYTEST_COUNT.finditer(output):
        kind = "errors" if match.group("kind") == "error" else match.group("kind")
        counts[kind] = counts.get(kind, 0) + int(match.group("n"))
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0) + counts.get("errors", 0)
    total = passed + failed
    if total == 0:
        raise SandboxExecutionError("could not parse pytest pass/fail counts from sandbox output")
    return ExecutionSummary(passed=passed, total=total)


class DockerPytestRunner:
    """Run tests inside a no-network, unprivileged, resource-limited container.

    The default mounts the host repository read-only at /repo, copies it into an
    ephemeral tmpfs /workspace, and runs there. This preserves host immutability
    while allowing test suites that legitimately create temporary relative files.
    """

    def __init__(self, repo_path: str | Path, cfg: DockerSandboxConfig, *, pytest_args: Sequence[str] = ("-q",)) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.cfg = cfg
        self.pytest_args = tuple(pytest_args)
        if not self.repo_path.is_dir():
            raise ValueError("repo_path must be an existing directory")

    def __call__(self, candidate_adapter: object) -> ExecutionSummary:
        adapter_path: Path | None = None
        if candidate_adapter is not None:
            if not isinstance(candidate_adapter, (str, Path)):
                raise TypeError("sandbox candidate must be an adapter path or None")
            adapter_path = Path(candidate_adapter).resolve()
            if not adapter_path.is_file():
                raise ValueError("adapter path must be an existing file")

        cmd = [
            "docker", "run", "--rm", "--pull=never",
            "--network", self.cfg.network,
            "--cpus", str(self.cfg.cpus),
            "--memory", self.cfg.memory,
            "--memory-swap", self.cfg.memory,
            "--pids-limit", str(self.cfg.pids_limit),
            "--ulimit", "nofile=1024:1024",
            "--security-opt", "no-new-privileges:true",
            "--cap-drop", "ALL",
            "--user", "65534:65534",
            "--ipc", "none",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=512m",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", "HOME=/tmp",
        ]
        if self.cfg.read_only_root:
            cmd.append("--read-only")

        if self.cfg.workspace_mode == "tmpfs_copy":
            cmd += ["--tmpfs", "/workspace:rw,nosuid,nodev,size=2g", "-v", f"{self.repo_path}:/repo:ro"]
            shell = "cp -a /repo/. /workspace/ && cd /workspace && python -m pytest -p no:cacheprovider " + " ".join(_shell_quote(x) for x in self.pytest_args)
        else:
            cmd += ["-v", f"{self.repo_path}:/workspace:ro", "-w", "/workspace"]
            shell = "python -m pytest -p no:cacheprovider " + " ".join(_shell_quote(x) for x in self.pytest_args)

        if adapter_path is not None:
            cmd += ["-v", f"{adapter_path}:/adapter/adapter.safetensors:ro", "-e", "DAPH_ADAPTER_PATH=/adapter/adapter.safetensors"]
        cmd += [self.cfg.image, "/bin/sh", "-lc", shell]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.cfg.timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError(f"sandbox timed out after {self.cfg.timeout_seconds}s") from exc
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode not in (0, 1):
            raise SandboxExecutionError(f"sandbox pytest exited with code {proc.returncode}: {combined[-2000:]}")
        return parse_pytest_summary(combined)


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def sandbox_environment_fingerprint(image: str) -> str:
    """Return immutable Docker image ID/digest for provenance."""
    proc = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SandboxExecutionError(f"cannot inspect sandbox image {image!r}")
    return proc.stdout.strip()


def adversarial_sandbox_cases() -> tuple[str, ...]:
    """Containment behaviors that a production sandbox qualification must test."""
    return (
        "network_connect", "write_readonly_repo", "write_outside_workspace",
        "read_host_path", "fork_bomb_guard", "memory_limit", "pid_limit",
        "privileged_syscall", "mount_attempt", "timeout_enforcement",
    )
