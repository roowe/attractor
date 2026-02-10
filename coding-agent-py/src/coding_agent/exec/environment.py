# coding-agent-py/src/coding_agent/exec/environment.py
import asyncio
import os
import signal
import platform as platform_module
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

from coding_agent.exec.grep_options import GrepOptions


class ExecResult:
    """Result of executing a command."""

    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        timed_out: bool,
        duration_ms: int,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.duration_ms = duration_ms


class DirEntry:
    """A directory entry."""

    def __init__(self, name: str, is_dir: bool, size: Optional[int] = None):
        self.name = name
        self.is_dir = is_dir
        self.size = size


class ExecutionEnvironment(ABC):
    """Abstract interface for tool execution environments."""

    @abstractmethod
    async def read_file(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Read a file's content."""
        pass

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file."""
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass

    @abstractmethod
    async def list_directory(self, path: str, depth: int = 1) -> List[DirEntry]:
        """List a directory's contents."""
        pass

    @abstractmethod
    async def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> ExecResult:
        """Execute a command."""
        pass

    @abstractmethod
    async def grep(self, options: GrepOptions) -> str:
        """Search for patterns in files."""
        pass

    @abstractmethod
    async def glob(self, pattern: str, path: str = ".") -> List[str]:
        """Find files matching a pattern."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the environment."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def working_directory(self) -> str:
        """Get the current working directory."""
        pass

    @abstractmethod
    def platform(self) -> str:
        """Get the platform identifier."""
        pass

    @abstractmethod
    def os_version(self) -> str:
        """Get the OS version."""
        pass


class LocalExecutionEnvironment(ExecutionEnvironment):
    """Local execution environment that runs commands on the host machine."""

    def __init__(self, working_dir: Optional[str] = None):
        self._working_dir = working_dir or os.getcwd()
        self._platform = platform_module.system().lower()
        if self._platform == "darwin":
            self._platform = "darwin"

    async def initialize(self) -> None:
        """No initialization needed for local environment."""
        pass

    async def cleanup(self) -> None:
        """No cleanup needed for local environment."""
        pass

    def working_directory(self) -> str:
        return self._working_dir

    def platform(self) -> str:
        return self._platform

    def os_version(self) -> str:
        return platform_module.release()

    async def read_file(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        full_path = Path(self._working_dir) / path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = full_path.read_text()

        # Apply offset and limit
        if offset is not None or limit is not None:
            lines = content.split("\n")
            start = (offset - 1) if offset else 0
            end = start + limit if limit else len(lines)
            content = "\n".join(lines[start:end])

        return content

    async def write_file(self, path: str, content: str) -> None:
        full_path = Path(self._working_dir) / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    async def file_exists(self, path: str) -> bool:
        full_path = Path(self._working_dir) / path
        return full_path.exists()

    async def list_directory(self, path: str, depth: int = 1) -> List[DirEntry]:
        full_path = Path(self._working_dir) / path
        entries = []

        if depth == 1:
            for item in full_path.iterdir():
                stat = item.stat()
                entries.append(DirEntry(
                    name=item.name,
                    is_dir=item.is_dir(),
                    size=stat.st_size if not item.is_dir() else None
                ))
        else:
            # Recursively collect
            for item in full_path.rglob("*"):
                if item.is_file() or depth > 1:
                    stat = item.stat()
                    entries.append(DirEntry(
                        name=str(item.relative_to(full_path)),
                        is_dir=item.is_dir(),
                        size=stat.st_size if not item.is_dir() else None
                    ))

        return entries

    async def exec_command(
        self,
        command: str,
        timeout_ms: int,
        working_dir: Optional[str] = None,
        env_vars: Optional[dict] = None,
    ) -> ExecResult:
        start_time = datetime.now()
        work_dir = working_dir or self._working_dir

        # Filter sensitive environment variables
        filtered_env = self._filter_environment(env_vars)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=filtered_env,
                start_new_session=True,  # Create new process group
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_ms / 1000.0,
                )
                timed_out = False
            except asyncio.TimeoutError:
                # Kill the process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    await asyncio.sleep(2)
                    if process.returncode is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

                stdout, stderr = await process.communicate()
                timed_out = True
                stderr_decoded = stderr.decode("utf-8", errors="replace") + "\n[Command timed out]"

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr_decoded if timed_out else stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return ExecResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                timed_out=False,
                duration_ms=duration_ms,
            )

    def _filter_environment(self, custom_vars: Optional[dict] = None) -> dict:
        """Filter out sensitive environment variables."""
        import os as os_module

        env = os_module.environ.copy()

        # Remove sensitive variables
        sensitive_patterns = ["_API_KEY", "_SECRET", "_TOKEN", "_PASSWORD", "_CREDENTIAL"]
        for key in list(env.keys()):
            if any(pattern in key.upper() for pattern in sensitive_patterns):
                del env[key]

        # Always include core variables
        core_vars = ["PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "TMPDIR"]
        for var in core_vars:
            if var in os_module.environ:
                env.setdefault(var, os_module.environ[var])

        # Add custom variables
        if custom_vars:
            env.update(custom_vars)

        return env

    async def grep(self, options: GrepOptions) -> str:
        """Use ripgrep if available, otherwise fall back to Python implementation."""
        try:
            # Try to use ripgrep
            cmd_parts = ["rg", options.pattern]

            if options.case_insensitive:
                cmd_parts.append("-i")

            if options.glob_filter:
                cmd_parts.extend(["-g", options.glob_filter])

            if options.context_lines > 0:
                cmd_parts.extend(["-C", str(options.context_lines)])

            cmd_parts.append(options.path)

            result = await self.exec_command(
                " ".join(cmd_parts),
                timeout_ms=5000,
            )

            if result.exit_code == 0:
                return result.stdout

        except FileNotFoundError:
            pass  # ripgrep not found, fall back to Python implementation

        # Python implementation as fallback
        return await self._grep_python(options)

    async def _grep_python(self, options: GrepOptions) -> str:
        """Python-based grep implementation."""
        import re

        results = []
        pattern = re.compile(
            options.pattern,
            re.IGNORECASE if options.case_insensitive else 0,
        )
        base_path = Path(self._working_dir) / options.path

        for file_path in base_path.rglob("*"):
            if not file_path.is_file():
                continue

            if options.glob_filter:
                if not file_path.match(options.glob_filter):
                    continue

            try:
                content = file_path.read_text()
                for i, line in enumerate(content.split("\n"), 1):
                    if pattern.search(line):
                        rel_path = file_path.relative_to(self._working_dir)
                        results.append(f"{rel_path}:{i}:{line}")
                        if len(results) >= options.max_results:
                            break
            except Exception:
                pass

            if len(results) >= options.max_results:
                break

        return "\n".join(results)

    async def glob(self, pattern: str, path: str = ".") -> List[str]:
        """Find files matching a glob pattern."""
        base_path = Path(self._working_dir) / path
        matches = sorted(
            base_path.rglob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return [str(m.relative_to(self._working_dir)) for m in matches]
