"""Process-level sandbox for shell command execution.

Wraps shell commands with platform-specific isolation:
- Linux: bubblewrap (bwrap) — filesystem + network namespace isolation
- macOS: sandbox-exec (Seatbelt) — filesystem restriction profiles

Falls back to plain execution when sandbox tools are unavailable.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


def is_sandbox_available() -> bool:
    """Check if any sandbox mechanism is available on the current platform."""
    system = platform.system()
    if system == "Linux":
        return shutil.which("bwrap") is not None
    if system == "Darwin":
        return shutil.which("sandbox-exec") is not None
    return False


def sandbox_backend() -> str:
    """Return the name of the available sandbox backend, or 'none'."""
    system = platform.system()
    if system == "Linux" and shutil.which("bwrap"):
        return "bubblewrap"
    if system == "Darwin" and shutil.which("sandbox-exec"):
        return "seatbelt"
    return "none"


def _generate_seatbelt_profile(workspace: str) -> str:
    """Generate a macOS Seatbelt profile: allow all reads, restrict writes to workspace + tmp.

    Uses ``(deny file-write* (require-not ...))`` so only whitelisted paths are
    writable.  The workspace path is resolved through symlinks so that
    ``/tmp`` → ``/private/tmp`` works correctly on macOS.
    """
    resolved = str(Path(workspace).resolve())
    return (
        "(version 1)\n"
        "(allow default)\n"
        "(deny file-write*\n"
        "    (require-not\n"
        "        (require-any\n"
        f'            (subpath "{resolved}")\n'
        '            (subpath "/private/tmp")\n'
        '            (subpath "/private/var/folders")\n'
        '            (subpath "/dev")\n'
        "        )\n"
        "    )\n"
        ")\n"
    )


def build_sandboxed_command(
    command: str,
    cwd: str,
    *,
    allow_network: bool = True,
    extra_writable_paths: list[str] | None = None,
) -> list[str]:
    """Wrap a shell command for sandboxed execution.

    Returns a command list suitable for subprocess.Popen(args=..., shell=False).
    Falls back to ['bash', '-c', command] when no sandbox is available.
    """
    system = platform.system()
    writable = [cwd] + (extra_writable_paths or [])

    if system == "Linux" and shutil.which("bwrap"):
        args = ["bwrap"]
        # Read-only bind system directories
        for sys_dir in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt"):
            d = Path(sys_dir)
            if d.exists():
                args.extend(["--ro-bind", sys_dir, sys_dir])
        # Bind /proc and /dev for basic functionality
        args.extend(["--proc", "/proc", "--dev", "/dev"])
        # Writable workspace and tmp
        for wp in writable:
            p = Path(wp)
            if p.exists():
                args.extend(["--bind", str(p), str(p)])
        args.extend(["--tmpfs", "/tmp"])
        # Network isolation (optional)
        if not allow_network:
            args.append("--unshare-net")
        args.extend(["--die-with-parent", "--", "bash", "-c", command])
        return args

    if system == "Darwin" and shutil.which("sandbox-exec"):
        profile = _generate_seatbelt_profile(cwd)
        # Write profile to temp file — `-f` is more reliable than inline `-p`
        profile_fd, profile_path = tempfile.mkstemp(suffix=".sb", prefix="ict_sandbox_")
        import os as _os
        _os.write(profile_fd, profile.encode("utf-8"))
        _os.close(profile_fd)
        return ["sandbox-exec", "-f", profile_path, "bash", "-c", command]

    # Fallback: no sandbox
    return ["bash", "-c", command]


def run_sandboxed(
    command: str,
    cwd: str,
    env: dict | None = None,
    timeout_sec: int = 120,
    allow_network: bool = True,
) -> tuple[int, str, str]:
    """Execute a command inside a sandbox.

    Returns (exit_code, stdout, stderr).
    """
    args = build_sandboxed_command(command, cwd, allow_network=allow_network)
    use_shell = len(args) == 3 and args[0] == "bash"  # fallback mode

    try:
        proc = subprocess.run(
            args if not use_shell else command,
            shell=use_shell,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout_sec}s"
    except Exception as exc:
        return -1, "", f"Sandbox execution error: {exc}"
