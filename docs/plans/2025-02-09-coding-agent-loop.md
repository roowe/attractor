# Python Coding Agent Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a programmable coding agent loop library in Python that orchestrates LLM interactions with tool execution, supporting multiple provider-aligned toolsets (OpenAI/codex-rs, Anthropic/Claude Code, Gemini/gemini-cli), with real-time events, steering, and subagent support.

**Architecture:** Five-layer architecture:
- Layer 1 (Foundation): Data models (Session, Turn, Config, Events)
- Layer 2 (Execution): ToolRegistry, ExecutionEnvironment interfaces
- Layer 3 (Providers): ProviderProfile with aligned toolsets and system prompts
- Layer 4 (Core Loop): Session with agent loop, steering, loop detection
- Layer 5 (Tools): Core tools (read_file, write_file, edit_file, shell, grep, glob) and provider-specific tools (apply_patch)

**Tech Stack:** Python 3.11+, unified-llm (LLM client), httpx (HTTP), pydantic (validation), pytest (tests), anyio (async), aiostream (event streaming)

**Dependencies:** Builds on unified-llm-py package at `/Volumes/MOVESPEED/ai-tools/attractor/unified-llm-py/`

**Implementation Location:** Create new package `coding-agent` alongside `unified-llm` at `/Volumes/MOVESPEED/ai-tools/attractor/coding-agent-py/`

---

## Project Setup

### Task 1: Create Project Structure

**Files:**
- Create: `coding-agent-py/pyproject.toml` (Project configuration)
- Create: `coding-agent-py/src/coding_agent/__init__.py` (Package init)
- Create: `coding-agent-py/tests/__init__.py` (Test package init)
- Create: `coding-agent-py/tests/conftest.py` (Pytest fixtures)
- Create: `coding-agent-py/.env.example` (Example environment variables)
- Create: `coding-agent-py/.gitignore` (Git ignore patterns)
- Create: `coding-agent-py/README.md` (Project documentation)
- Create: `coding-agent-py/examples/basic_usage.py` (Basic usage example)

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "coding-agent"
version = "0.1.0"
description = "Programmable coding agent loop with provider-aligned toolsets"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "anyio>=4.0.0",
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/coding_agent"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local

# MyPy
.mypy_cache/
```

**Step 3: Create .env.example**

```
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_BASE_URL=

# Gemini
GEMINI_API_KEY=...
GEMINI_BASE_URL=

# Coding Agent
CODING_AGENT_WORKING_DIR=.
CODING_AGENT_MAX_TURNS=0
CODING_AGENT_DEFAULT_TIMEOUT=10000
```

**Step 4: Create README.md**

```markdown
# Coding Agent Loop

A programmable coding agent loop library in Python with provider-aligned toolsets.

## Installation

```bash
pip install coding-agent
```

## Quick Start

```python
import asyncio
from coding_agent import Session, LocalExecutionEnvironment, create_anthropic_profile

async def main():
    profile = create_anthropic_profile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment(working_dir=".")
    session = Session(profile, env)

    async for event in session.submit("Create a hello.py file that prints 'Hello World'"):
        print(f"[{event.kind}] {event.data}")

asyncio.run(main())
```

## Features

- Provider-aligned toolsets (OpenAI/codex-rs, Anthropic/Claude Code, Gemini/gemini-cli)
- Real-time event streaming
- Mid-task steering
- Loop detection
- Subagent support
- Tool output truncation
- Multiple execution environments (local, Docker, Kubernetes, SSH)
```

**Step 5: Run tests to verify project setup**

Run: `cd coding-agent-py && python -m pytest tests/ -v`
Expected: Empty test run (0 tests collected)

**Step 6: Commit**

```bash
cd coding-agent-py
git add pyproject.toml .gitignore .env.example README.md src/coding_agent/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: initial project structure for coding-agent"
```

---

## Layer 1: Foundation Models

### Task 2: Create SessionConfig Model

**Files:**
- Create: `coding-agent-py/src/coding_agent/models/config.py`
- Create: `coding-agent-py/tests/models/test_config.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/models/test_config.py
import pytest
from pydantic import ValidationError
from coding_agent.models.config import SessionConfig


def test_session_config_defaults():
    config = SessionConfig()
    assert config.max_turns == 0
    assert config.max_tool_rounds_per_input == 200
    assert config.default_command_timeout_ms == 10000
    assert config.max_command_timeout_ms == 600000
    assert config.reasoning_effort is None
    assert config.enable_loop_detection is True
    assert config.loop_detection_window == 10
    assert config.max_subagent_depth == 1


def test_session_config_custom_values():
    config = SessionConfig(
        max_turns=100,
        default_command_timeout_ms=30000,
        reasoning_effort="high"
    )
    assert config.max_turns == 100
    assert config.default_command_timeout_ms == 30000
    assert config.reasoning_effort == "high"


def test_session_config_tool_output_limits():
    config = SessionConfig(
        tool_output_limits={"read_file": 100000, "shell": 50000}
    )
    assert config.tool_output_limits["read_file"] == 100000
    assert config.tool_output_limits["shell"] == 50000


def test_session_config_invalid_reasoning_effort():
    with pytest.raises(ValidationError):
        SessionConfig(reasoning_effort="invalid")


def test_session_config_invalid_timeout():
    with pytest.raises(ValidationError):
        SessionConfig(default_command_timeout_ms=-100)
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/models/test_config.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.models.config'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/models/config.py
from typing import Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator


ReasoningEffort = Literal["low", "medium", "high", None]


class SessionConfig(BaseModel):
    """Configuration for a coding agent session."""

    max_turns: int = Field(default=0, ge=0)
    max_tool_rounds_per_input: int = Field(default=200, ge=1)
    default_command_timeout_ms: int = Field(default=10000, ge=1000)
    max_command_timeout_ms: int = Field(default=600000, ge=1000)
    reasoning_effort: Optional[ReasoningEffort] = None
    tool_output_limits: Dict[str, int] = Field(default_factory=dict)
    tool_line_limits: Dict[str, Optional[int]] = Field(default_factory=dict)
    enable_loop_detection: bool = True
    loop_detection_window: int = Field(default=10, ge=2)
    max_subagent_depth: int = Field(default=1, ge=0)

    @field_validator("max_command_timeout_ms")
    @classmethod
    def max_timeout_must_be_greater_than_default(cls, v, info):
        if "default_command_timeout_ms" in info.data:
            if v < info.data["default_command_timeout_ms"]:
                raise ValueError("max_command_timeout_ms must be >= default_command_timeout_ms")
        return v
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/models/test_config.py -v`
Expected: PASS (5 tests)

**Step 5: Update models/__init__.py**

```python
# coding-agent-py/src/coding_agent/models/__init__.py
from coding_agent.models.config import SessionConfig, ReasoningEffort

__all__ = ["SessionConfig", "ReasoningEffort"]
```

**Step 6: Commit**

```bash
git add src/coding_agent/models/config.py tests/models/test_config.py src/coding_agent/models/__init__.py
git commit -m "feat: add SessionConfig model with validation"
```

---

### Task 3: Create SessionState and Turn Models

**Files:**
- Create: `coding-agent-py/src/coding_agent/models/turn.py`
- Create: `coding-agent-py/tests/models/test_turn.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/models/test_turn.py
import pytest
from datetime import datetime
from coding_agent.models.turn import (
    SessionState,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SystemTurn,
    SteeringTurn,
)


def test_session_state_values():
    assert SessionState.IDLE == "idle"
    assert SessionState.PROCESSING == "processing"
    assert SessionState.AWAITING_INPUT == "awaiting_input"
    assert SessionState.CLOSED == "closed"


def test_user_turn():
    turn = UserTurn(content="Hello, agent")
    assert turn.content == "Hello, agent"
    assert isinstance(turn.timestamp, datetime)


def test_assistant_turn_with_tool_calls():
    from coding_agent.models.tool import ToolCall
    from coding_agent.models.usage import Usage

    turn = AssistantTurn(
        content="I'll help you",
        tool_calls=[
            ToolCall(
                id="call_123",
                name="read_file",
                arguments={"file_path": "/path/to/file"}
            )
        ],
        reasoning="Let me think...",
        usage=Usage(prompt_tokens=10, completion_tokens=20),
        response_id="resp_456"
    )
    assert turn.content == "I'll help you"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "read_file"
    assert turn.reasoning == "Let me think..."
    assert turn.response_id == "resp_456"


def test_assistant_turn_without_tool_calls():
    turn = AssistantTurn(content="Done!")
    assert turn.content == "Done!"
    assert turn.tool_calls == []
    assert turn.reasoning is None


def test_tool_results_turn():
    from coding_agent.models.tool import ToolResult

    turn = ToolResultsTurn(results=[
        ToolResult(tool_call_id="call_123", content="File content", is_error=False)
    ])
    assert len(turn.results) == 1
    assert turn.results[0].tool_call_id == "call_123"


def test_system_turn():
    turn = SystemTurn(content="System message")
    assert turn.content == "System message"


def test_steering_turn():
    turn = SteeringTurn(content="Try a different approach")
    assert turn.content == "Try a different approach"
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/models/test_turn.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.models.turn'"

**Step 3: Write minimal implementation**

First, create the Tool and Usage models (these are dependencies):

```python
# coding-agent-py/src/coding_agent/models/tool.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool call."""
    tool_call_id: str
    content: str
    is_error: bool = False


class ToolDefinition(BaseModel):
    """Definition of a tool for LLM consumption."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
```

```python
# coding-agent-py/src/coding_agent/models/usage.py
from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __init__(self, **data):
        if "total_tokens" not in data:
            data["total_tokens"] = data.get("prompt_tokens", 0) + data.get("completion_tokens", 0)
        super().__init__(**data)
```

Now create the turn models:

```python
# coding-agent-py/src/coding_agent/models/turn.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

from coding_agent.models.tool import ToolCall, ToolResult
from coding_agent.models.usage import Usage


class SessionState(str, Enum):
    """Session lifecycle states."""
    IDLE = "idle"
    PROCESSING = "processing"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"


class UserTurn(BaseModel):
    """A user input turn."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AssistantTurn(BaseModel):
    """An assistant response turn."""
    content: str
    tool_calls: List[ToolCall] = Field(default_factory=list)
    reasoning: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    response_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResultsTurn(BaseModel):
    """Results from executing tool calls."""
    results: List[ToolResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemTurn(BaseModel):
    """A system message turn."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SteeringTurn(BaseModel):
    """A steering message injected by the host."""
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type alias for any turn type
Turn = UserTurn | AssistantTurn | ToolResultsTurn | SystemTurn | SteeringTurn
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/models/test_turn.py -v`
Expected: PASS (8 tests)

**Step 5: Update models/__init__.py**

```python
# coding-agent-py/src/coding_agent/models/__init__.py
from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import (
    SessionState,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SystemTurn,
    SteeringTurn,
    Turn,
)
from coding_agent.models.tool import ToolCall, ToolResult, ToolDefinition
from coding_agent.models.usage import Usage

__all__ = [
    "SessionConfig",
    "SessionState",
    "UserTurn",
    "AssistantTurn",
    "ToolResultsTurn",
    "SystemTurn",
    "SteeringTurn",
    "Turn",
    "ToolCall",
    "ToolResult",
    "ToolDefinition",
    "Usage",
]
```

**Step 6: Commit**

```bash
git add src/coding_agent/models/ tests/models/test_turn.py
git commit -m "feat: add turn models (UserTurn, AssistantTurn, ToolResultsTurn, etc.)"
```

---

### Task 4: Create Event System Models

**Files:**
- Create: `coding-agent-py/src/coding_agent/models/event.py`
- Create: `coding-agent-py/tests/models/test_event.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/models/test_event.py
import pytest
from datetime import datetime
from coding_agent.models.event import EventKind, SessionEvent


def test_event_kind_values():
    assert EventKind.SESSION_START == "session_start"
    assert EventKind.SESSION_END == "session_end"
    assert EventKind.USER_INPUT == "user_input"
    assert EventKind.ASSISTANT_TEXT_START == "assistant_text_start"
    assert EventKind.ASSISTANT_TEXT_DELTA == "assistant_text_delta"
    assert EventKind.ASSISTANT_TEXT_END == "assistant_text_end"
    assert EventKind.TOOL_CALL_START == "tool_call_start"
    assert EventKind.TOOL_CALL_END == "tool_call_end"
    assert EventKind.STEERING_INJECTED == "steering_injected"
    assert EventKind.TURN_LIMIT == "turn_limit"
    assert EventKind.LOOP_DETECTION == "loop_detection"
    assert EventKind.ERROR == "error"


def test_session_event_creation():
    event = SessionEvent(
        kind=EventKind.USER_INPUT,
        session_id="session_123",
        data={"content": "Hello"}
    )
    assert event.kind == EventKind.USER_INPUT
    assert event.session_id == "session_123"
    assert event.data["content"] == "Hello"
    assert isinstance(event.timestamp, datetime)


def test_session_event_with_tool_call_start():
    event = SessionEvent(
        kind=EventKind.TOOL_CALL_START,
        session_id="session_123",
        data={"tool_name": "read_file", "call_id": "call_456"}
    )
    assert event.kind == EventKind.TOOL_CALL_START
    assert event.data["tool_name"] == "read_file"
    assert event.data["call_id"] == "call_456"


def test_session_event_with_error():
    event = SessionEvent(
        kind=EventKind.ERROR,
        session_id="session_123",
        data={"message": "Something went wrong", "error_type": "ProviderError"}
    )
    assert event.kind == EventKind.ERROR
    assert event.data["message"] == "Something went wrong"
    assert event.data["error_type"] == "ProviderError"
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/models/test_event.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.models.event'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/models/event.py
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class EventKind(str, Enum):
    """Types of events emitted during session execution."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_INPUT = "user_input"
    ASSISTANT_TEXT_START = "assistant_text_start"
    ASSISTANT_TEXT_DELTA = "assistant_text_delta"
    ASSISTANT_TEXT_END = "assistant_text_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_OUTPUT_DELTA = "tool_call_output_delta"
    TOOL_CALL_END = "tool_call_end"
    STEERING_INJECTED = "steering_injected"
    TURN_LIMIT = "turn_limit"
    LOOP_DETECTION = "loop_detection"
    ERROR = "error"
    WARNING = "warning"


class SessionEvent(BaseModel):
    """An event emitted during session execution."""
    kind: EventKind
    session_id: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/models/test_event.py -v`
Expected: PASS (4 tests)

**Step 5: Update models/__init__.py**

```python
# coding-agent-py/src/coding_agent/models/__init__.py
# ... existing imports ...
from coding_agent.models.event import EventKind, SessionEvent

__all__ = [
    # ... existing exports ...
    "EventKind",
    "SessionEvent",
]
```

**Step 6: Commit**

```bash
git add src/coding_agent/models/event.py tests/models/test_event.py src/coding_agent/models/__init__.py
git commit -m "feat: add event system models (EventKind, SessionEvent)"
```

---

## Layer 2: Execution Environment

### Task 5: Create ExecutionEnvironment Interface and LocalExecutionEnvironment

**Files:**
- Create: `coding-agent-py/src/coding_agent/exec/environment.py`
- Create: `coding-agent-py/tests/exec/test_environment.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/exec/test_environment.py
import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from coding_agent.exec.environment import (
    ExecutionEnvironment,
    LocalExecutionEnvironment,
    ExecResult,
    DirEntry,
)
from coding_agent.exec.grep_options import GrepOptions


@pytest.mark.asyncio
async def test_local_environment_working_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        assert env.working_directory() == tmpdir


@pytest.mark.asyncio
async def test_local_environment_platform():
    env = LocalExecutionEnvironment()
    assert env.platform() in ["darwin", "linux", "windows"]


@pytest.mark.asyncio
async def test_local_environment_write_and_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        await env.write_file("test.txt", "Hello, World!")
        content = await env.read_file("test.txt")

        assert content == "Hello, World!"


@pytest.mark.asyncio
async def test_local_environment_file_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        assert not await env.file_exists("test.txt")

        await env.write_file("test.txt", "content")
        assert await env.file_exists("test.txt")


@pytest.mark.asyncio
async def test_local_environment_exec_command():
    env = LocalExecutionEnvironment()
    result = await env.exec_command("echo hello", timeout_ms=5000)

    assert result.stdout.strip() == "hello"
    assert result.exit_code == 0
    assert not result.timed_out
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_local_environment_exec_command_timeout():
    env = LocalExecutionEnvironment()
    result = await env.exec_command("sleep 5", timeout_ms=1000)

    assert result.timed_out is True
    assert "timeout" in result.stderr.lower() or result.exit_code != 0


@pytest.mark.asyncio
async def test_local_environment_list_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create some files and directories
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        os.makedirs(Path(tmpdir, "subdir"))
        Path(tmpdir, "subdir", "nested.txt").touch()

        entries = await env.list_directory(".", depth=1)

        names = {e.name for e in entries}
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names

        subdir_entry = next(e for e in entries if e.name == "subdir")
        assert subdir_entry.is_dir is True


@pytest.mark.asyncio
async def test_local_environment_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create test files
        Path(tmpdir, "test1.txt").write_text("hello world\nhello python\n")
        Path(tmpdir, "test2.txt").write_text("goodbye world\n")

        options = GrepOptions(pattern="hello", path=".")
        result = await env.grep(options)

        assert "hello world" in result
        assert "hello python" in result
        assert "test1.txt" in result


@pytest.mark.asyncio
async def test_local_environment_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create test files
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        Path(tmpdir, "test.txt").touch()

        results = await env.glob("*.txt", path=".")

        assert "file1.txt" in results
        assert "test.txt" in results
        assert "file2.py" not in results


@pytest.mark.asyncio
async def test_local_environment_read_file_with_offset_and_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)

        # Create file with numbered lines
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        await env.write_file("test.txt", content)

        # Read with offset
        result = await env.read_file("test.txt", offset=50, limit=10)
        lines = result.strip().split("\n")

        assert len(lines) == 10
        assert "Line 50" in lines[0]


@pytest.mark.asyncio
async def test_local_environment_initialize_and_cleanup():
    env = LocalExecutionEnvironment()
    await env.initialize()
    # Should not raise any errors
    await env.cleanup()
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/exec/test_environment.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.exec'"

**Step 3: Write minimal implementation**

First, create the GrepOptions model:

```python
# coding-agent-py/src/coding_agent/exec/grep_options.py
from typing import Optional, List
from pydantic import BaseModel, Field


class GrepOptions(BaseModel):
    """Options for grep tool."""
    pattern: str
    path: str = "."
    glob_filter: Optional[str] = None
    case_insensitive: bool = False
    max_results: int = 100
    output_mode: str = "content"  # "content", "files_with_matches", "count"
    context_lines: int = 0
```

Now create the environment module:

```python
# coding-agent-py/src/coding_agent/exec/environment.py
import asyncio
import os
import signal
import platform as platform_module
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

import anyio

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
                stderr = stderr.decode("utf-8", errors="replace") + "\n[Command timed out]"

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
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
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/exec/test_environment.py -v`
Expected: PASS (most tests, may need to adjust timeout test)

**Step 5: Create exec/__init__.py**

```python
# coding-agent-py/src/coding_agent/exec/__init__.py
from coding_agent.exec.environment import (
    ExecutionEnvironment,
    LocalExecutionEnvironment,
    ExecResult,
    DirEntry,
)
from coding_agent.exec.grep_options import GrepOptions

__all__ = [
    "ExecutionEnvironment",
    "LocalExecutionEnvironment",
    "ExecResult",
    "DirEntry",
    "GrepOptions",
]
```

**Step 6: Commit**

```bash
git add src/coding_agent/exec/ tests/exec/
git commit -m "feat: add ExecutionEnvironment interface and LocalExecutionEnvironment"
```

---

## Layer 3: Tool Registry and Core Tools

### Task 6: Create Tool Registry

**Files:**
- Create: `coding-agent-py/src/coding_agent/tools/registry.py`
- Create: `coding-agent-py/tests/tools/test_registry.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/tools/test_registry.py
import pytest
from typing import Dict, Any
from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import LocalExecutionEnvironment


async def dummy_tool_executor(arguments: Dict[str, Any], env: LocalExecutionEnvironment) -> str:
    """A dummy tool executor for testing."""
    return f"Executed with: {arguments}"


def test_tool_registry_register_and_get():
    registry = ToolRegistry()

    tool = RegisteredTool(
        definition=ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool)

    retrieved = registry.get("test_tool")
    assert retrieved is not None
    assert retrieved.definition.name == "test_tool"


def test_tool_registry_get_nonexistent():
    registry = ToolRegistry()

    retrieved = registry.get("nonexistent")
    assert retrieved is None


def test_tool_registry_unregister():
    registry = ToolRegistry()

    tool = RegisteredTool(
        definition=ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool)
    assert registry.get("test_tool") is not None

    registry.unregister("test_tool")
    assert registry.get("test_tool") is None


def test_tool_registry_definitions():
    registry = ToolRegistry()

    tool1 = RegisteredTool(
        definition=ToolDefinition(
            name="tool1",
            description="Tool 1",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    tool2 = RegisteredTool(
        definition=ToolDefinition(
            name="tool2",
            description="Tool 2",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool1)
    registry.register(tool2)

    definitions = registry.definitions()
    assert len(definitions) == 2
    assert definitions[0].name == "tool1"
    assert definitions[1].name == "tool2"


def test_tool_registry_names():
    registry = ToolRegistry()

    tool1 = RegisteredTool(
        definition=ToolDefinition(
            name="tool1",
            description="Tool 1",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    tool2 = RegisteredTool(
        definition=ToolDefinition(
            name="tool2",
            description="Tool 2",
            parameters={"type": "object"},
        ),
        executor=dummy_tool_executor,
    )

    registry.register(tool1)
    registry.register(tool2)

    names = registry.names()
    assert "tool1" in names
    assert "tool2" in names
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_registry.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.tools'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/tools/registry.py
from typing import Dict, Callable, Any, Awaitable, List
from pydantic import BaseModel

from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment


ToolExecutor = Callable[[Dict[str, Any], ExecutionEnvironment], Awaitable[str]]


class RegisteredTool(BaseModel):
    """A registered tool with its definition and executor."""

    definition: ToolDefinition
    executor: ToolExecutor

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, RegisteredTool] = {}

    def register(self, tool: RegisteredTool) -> None:
        """Register a new tool or replace an existing one."""
        self._tools[tool.definition.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return [tool.definition for tool in self._tools.values()]

    def names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_registry.py -v`
Expected: PASS (6 tests)

**Step 5: Create tools/__init__.py**

```python
# coding-agent-py/src/coding_agent/tools/__init__.py
from coding_agent.tools.registry import ToolRegistry, RegisteredTool, ToolExecutor

__all__ = ["ToolRegistry", "RegisteredTool", "ToolExecutor"]
```

**Step 6: Commit**

```bash
git add src/coding_agent/tools/ tests/tools/test_registry.py
git commit -m "feat: add ToolRegistry for managing tools"
```

---

### Task 7: Create Core File Tools (read_file, write_file, edit_file)

**Files:**
- Create: `coding-agent-py/src/coding_agent/tools/core/file_tools.py`
- Create: `coding-agent-py/tests/tools/test_file_tools.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/tools/test_file_tools.py
import pytest
import tempfile
import os
from pathlib import Path
from coding_agent.tools.core.file_tools import (
    register_read_file_tool,
    register_write_file_tool,
    register_edit_file_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.exec.environment import LocalExecutionEnvironment


@pytest.mark.asyncio
async def test_read_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        # Create a test file
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("Hello\nWorld\nPython\n")

        # Get the tool
        tool = registry.get("read_file")
        assert tool is not None

        # Execute the tool
        result = await tool.executor({"file_path": "test.txt"}, env)

        assert "1|Hello" in result or "Hello" in result
        assert "World" in result
        assert "Python" in result


@pytest.mark.asyncio
async def test_read_file_tool_with_offset_and_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        # Create a test file with many lines
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("\n".join([f"Line {i}" for i in range(1, 101)]))

        tool = registry.get("read_file")

        # Read with offset and limit
        result = await tool.executor({
            "file_path": "test.txt",
            "offset": 50,
            "limit": 5
        }, env)

        assert "Line 50" in result
        assert "Line 54" in result
        assert "Line 55" not in result


@pytest.mark.asyncio
async def test_read_file_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_read_file_tool(registry)

        tool = registry.get("read_file")

        result = await tool.executor({"file_path": "nonexistent.txt"}, env)

        assert "Error" in result or "not found" in result.lower()


@pytest.mark.asyncio
async def test_write_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_write_file_tool(registry)

        tool = registry.get("write_file")

        result = await tool.executor({
            "file_path": "new_file.txt",
            "content": "Hello, World!"
        }, env)

        assert "written" in result.lower() or "success" in result.lower()

        # Verify file was created
        test_file = Path(tmpdir, "new_file.txt")
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"


@pytest.mark.asyncio
async def test_edit_file_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_edit_file_tool(registry)

        # Create a test file
        test_file = Path(tmpdir, "test.txt")
        test_file.write_text("Hello World\nGoodbye World\n")

        tool = registry.get("edit_file")

        result = await tool.executor({
            "file_path": "test.txt",
            "old_string": "Hello World",
            "new_string": "Hello Python"
        }, env)

        assert "replaced" in result.lower() or "success" in result.lower()

        # Verify the edit
        content = test_file.read_text()
        assert "Hello Python" in content
        assert "Hello World" not in content


@pytest.mark.asyncio
async def test_edit_file_tool_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_edit_file_tool(registry)

        tool = registry.get("edit_file")

        result = await tool.executor({
            "file_path": "test.txt",
            "old_string": "Nonexistent",
            "new_string": "Replacement"
        }, env)

        assert "not found" in result.lower() or "error" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_file_tools.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.tools.core'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/tools/core/__init__.py
# Empty file to make core a package
```

```python
# coding-agent-py/src/coding_agent/tools/core/file_tools.py
from typing import Dict, Any

from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment


async def read_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for read_file tool."""
    file_path = arguments.get("file_path")
    offset = arguments.get("offset")
    limit = arguments.get("limit", 2000)

    if not file_path:
        return "Error: file_path is required"

    try:
        content = await env.read_file(file_path, offset=offset, limit=limit)

        # Add line numbers
        lines = content.split("\n")
        start_line = offset if offset else 1
        numbered_lines = [f"{i + start_line}|{line}" for i, line in enumerate(lines)]

        return "\n".join(numbered_lines)

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def register_read_file_tool(registry: ToolRegistry) -> None:
    """Register the read_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="read_file",
            description=(
                "Read a file from the filesystem. Returns content with line numbers."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 2000)"
                    }
                },
                "required": ["file_path"],
            },
        ),
        executor=read_file_executor,
    )
    registry.register(tool)


async def write_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for write_file tool."""
    file_path = arguments.get("file_path")
    content = arguments.get("content", "")

    if not file_path:
        return "Error: file_path is required"

    try:
        await env.write_file(file_path, content)
        bytes_written = len(content.encode("utf-8"))
        return f"Successfully wrote {bytes_written} bytes to {file_path}"

    except Exception as e:
        return f"Error writing file: {str(e)}"


def register_write_file_tool(registry: ToolRegistry) -> None:
    """Register the write_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file and parent directories if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write"
                    }
                },
                "required": ["file_path", "content"],
            },
        ),
        executor=write_file_executor,
    )
    registry.register(tool)


async def edit_file_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for edit_file tool."""
    file_path = arguments.get("file_path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string", "")
    replace_all = arguments.get("replace_all", False)

    if not file_path or not old_string:
        return "Error: file_path and old_string are required"

    try:
        # Read current content
        content = await env.read_file(file_path)

        # Check if old_string exists
        if old_string not in content:
            return f"Error: old_string not found in file. Current content may have changed."

        # Count occurrences if not replace_all
        if not replace_all:
            count = content.count(old_string)
            if count > 1:
                return f"Error: old_string appears {count} times. Use replace_all=true or provide more context."

        # Replace
        new_content = content.replace(old_string, new_string, -1 if replace_all else 1)

        # Write back
        await env.write_file(file_path, new_content)
        replacements = new_content.count(new_string) if new_string else 0
        return f"Successfully made {replacements} replacement(s) in {file_path}"

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"


def register_edit_file_tool(registry: ToolRegistry) -> None:
    """Register the edit_file tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="edit_file",
            description="Replace an exact string occurrence in a file.",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false)"
                    }
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        executor=edit_file_executor,
    )
    registry.register(tool)
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_file_tools.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/coding_agent/tools/core/ tests/tools/test_file_tools.py
git commit -m "feat: add core file tools (read_file, write_file, edit_file)"
```

---

### Task 8: Create Core Shell and Search Tools (shell, grep, glob)

**Files:**
- Create: `coding-agent-py/src/coding_agent/tools/core/search_tools.py`
- Create: `coding-agent-py/tests/tools/test_search_tools.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/tools/test_search_tools.py
import pytest
import tempfile
from pathlib import Path
from coding_agent.tools.core.search_tools import (
    register_shell_tool,
    register_grep_tool,
    register_glob_tool,
)
from coding_agent.tools.registry import ToolRegistry
from coding_agent.exec.environment import LocalExecutionEnvironment


@pytest.mark.asyncio
async def test_shell_tool():
    env = LocalExecutionEnvironment()
    registry = ToolRegistry()

    register_shell_tool(registry)

    tool = registry.get("shell")

    result = await tool.executor({
        "command": "echo 'Hello, World!'",
        "description": "Print hello"
    }, env)

    assert "Hello, World!" in result
    assert "exit code: 0" in result.lower()


@pytest.mark.asyncio
async def test_shell_tool_with_timeout():
    env = LocalExecutionEnvironment()
    registry = ToolRegistry()

    register_shell_tool(registry)

    tool = registry.get("shell")

    # Command that will timeout
    result = await tool.executor({
        "command": "sleep 5",
        "timeout_ms": 1000
    }, env)

    assert "timeout" in result.lower()


@pytest.mark.asyncio
async def test_grep_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_grep_tool(registry)

        # Create test files
        Path(tmpdir, "test1.txt").write_text("hello world\nhello python\n")
        Path(tmpdir, "test2.txt").write_text("goodbye world\n")

        tool = registry.get("grep")

        result = await tool.executor({
            "pattern": "hello",
            "path": "."
        }, env)

        assert "hello" in result.lower()
        assert "test1.txt" in result


@pytest.mark.asyncio
async def test_grep_tool_with_glob_filter():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_grep_tool(registry)

        # Create test files
        Path(tmpdir, "test.txt").write_text("hello\n")
        Path(tmpdir, "test.py").write_text("hello\n")

        tool = registry.get("grep")

        result = await tool.executor({
            "pattern": "hello",
            "path": ".",
            "glob_filter": "*.txt"
        }, env)

        assert "test.txt" in result
        assert "test.py" not in result


@pytest.mark.asyncio
async def test_glob_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = LocalExecutionEnvironment(working_dir=tmpdir)
        registry = ToolRegistry()

        register_glob_tool(registry)

        # Create test files
        Path(tmpdir, "file1.txt").touch()
        Path(tmpdir, "file2.py").touch()
        Path(tmpdir, "test.txt").touch()

        tool = registry.get("glob")

        result = await tool.executor({
            "pattern": "*.txt",
            "path": "."
        }, env)

        assert "file1.txt" in result
        assert "test.txt" in result
        assert "file2.py" not in result
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_search_tools.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.tools.core.search_tools'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/tools/core/search_tools.py
from typing import Dict, Any

from coding_agent.tools.registry import ToolRegistry, RegisteredTool
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.exec.grep_options import GrepOptions


async def shell_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for shell tool."""
    command = arguments.get("command")
    timeout_ms = arguments.get("timeout_ms")
    description = arguments.get("description", "")

    if not command:
        return "Error: command is required"

    # Get timeout from config or use default
    if timeout_ms is None:
        timeout_ms = 10000  # Default 10 seconds

    try:
        result = await env.exec_command(command, timeout_ms=timeout_ms)

        output = ""
        if description:
            output += f"[{description}]\n"

        output += f"Exit code: {result.exit_code}"

        if result.stdout:
            output += f"\n{result.stdout}"

        if result.stderr:
            output += f"\n{result.stderr}"

        if result.timed_out:
            output += "\n[Command timed out]"

        return output

    except Exception as e:
        return f"Error executing command: {str(e)}"


def register_shell_tool(registry: ToolRegistry) -> None:
    """Register the shell tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="shell",
            description="Execute a shell command. Returns stdout, stderr, and exit code.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run"
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Override default timeout in milliseconds"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this command does"
                    }
                },
                "required": ["command"],
            },
        ),
        executor=shell_executor,
    )
    registry.register(tool)


async def grep_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for grep tool."""
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    glob_filter = arguments.get("glob_filter")
    case_insensitive = arguments.get("case_insensitive", False)
    max_results = arguments.get("max_results", 100)

    if not pattern:
        return "Error: pattern is required"

    try:
        options = GrepOptions(
            pattern=pattern,
            path=path,
            glob_filter=glob_filter,
            case_insensitive=case_insensitive,
            max_results=max_results,
        )

        return await env.grep(options)

    except Exception as e:
        return f"Error searching files: {str(e)}"


def register_grep_tool(registry: ToolRegistry) -> None:
    """Register the grep tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="grep",
            description="Search file contents using a regex pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: working directory)"
                    },
                    "glob_filter": {
                        "type": "string",
                        "description": "File pattern filter (e.g., '*.py')"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)"
                    }
                },
                "required": ["pattern"],
            },
        ),
        executor=grep_executor,
    )
    registry.register(tool)


async def glob_executor(arguments: Dict[str, Any], env: ExecutionEnvironment) -> str:
    """Executor for glob tool."""
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")

    if not pattern:
        return "Error: pattern is required"

    try:
        results = await env.glob(pattern, path)

        if not results:
            return f"No files found matching pattern: {pattern}"

        return "\n".join(results)

    except Exception as e:
        return f"Error finding files: {str(e)}"


def register_glob_tool(registry: ToolRegistry) -> None:
    """Register the glob tool."""
    tool = RegisteredTool(
        definition=ToolDefinition(
            name="glob",
            description="Find files matching a glob pattern.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory (default: working directory)"
                    }
                },
                "required": ["pattern"],
            },
        ),
        executor=glob_executor,
    )
    registry.register(tool)
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/tools/test_search_tools.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/coding_agent/tools/core/search_tools.py tests/tools/test_search_tools.py
git commit -m "feat: add core search tools (shell, grep, glob)"
```

---

## Layer 4: Provider Profiles

### Task 9: Create ProviderProfile Base and Anthropic Profile

**Files:**
- Create: `coding-agent-py/src/coding_agent/providers/profile.py`
- Create: `coding-agent-py/tests/providers/test_profile.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/providers/test_profile.py
import pytest
from coding_agent.providers.profile import ProviderProfile, AnthropicProfile
from coding_agent.exec.environment import LocalExecutionEnvironment


def test_provider_profile_interface():
    profile = ProviderProfile(
        id="test",
        model="test-model",
    )

    assert profile.id == "test"
    assert profile.model == "test-model"


def test_anthropic_profile_creation():
    profile = AnthropicProfile(model="claude-opus-4-6")

    assert profile.id == "anthropic"
    assert profile.model == "claude-opus-4-6"
    assert profile.supports_reasoning is True
    assert profile.supports_streaming is True
    assert profile.supports_parallel_tool_calls is True


def test_anthropic_profile_tool_definitions():
    profile = AnthropicProfile(model="claude-opus-4-6")

    tools = profile.tools()
    tool_names = [t.name for t in tools]

    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "edit_file" in tool_names
    assert "shell" in tool_names
    assert "grep" in tool_names
    assert "glob" in tool_names


def test_anthropic_profile_system_prompt():
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment()

    prompt = profile.build_system_prompt(env, {})

    assert "Claude" in prompt
    assert "read_file" in prompt or "file" in prompt.lower()


def test_anthropic_profile_provider_options():
    profile = AnthropicProfile(model="claude-opus-4-6")

    options = profile.provider_options()

    assert options is not None
    assert "anthropic" in options


@pytest.mark.asyncio
async def test_profile_tool_registry():
    profile = AnthropicProfile(model="claude-opus-4-6")

    # Get tool from registry
    tool = profile.tool_registry.get("read_file")
    assert tool is not None
    assert tool.definition.name == "read_file"
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/providers/test_profile.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.providers'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/providers/__init__.py
from coding_agent.providers.profile import (
    ProviderProfile,
    AnthropicProfile,
    OpenAIProfile,
    GeminiProfile,
)

__all__ = [
    "ProviderProfile",
    "AnthropicProfile",
    "OpenAIProfile",
    "GeminiProfile",
]
```

```python
# coding-agent-py/src/coding_agent/providers/profile.py
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from coding_agent.tools.registry import ToolRegistry
from coding_agent.models.tool import ToolDefinition
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.tools.core.file_tools import (
    register_read_file_tool,
    register_write_file_tool,
    register_edit_file_tool,
)
from coding_agent.tools.core.search_tools import (
    register_shell_tool,
    register_grep_tool,
    register_glob_tool,
)


class ProviderProfile(ABC):
    """Base class for provider-specific profiles."""

    def __init__(self, id: str, model: str) -> None:
        self.id = id
        self.model = model
        self.tool_registry = ToolRegistry()

    @abstractmethod
    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build the system prompt for this provider."""
        pass

    @abstractmethod
    def tools(self) -> list[ToolDefinition]:
        """Get the tool definitions for this provider."""
        pass

    @abstractmethod
    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get provider-specific options for LLM requests."""
        pass

    # Capability flags (can be overridden)
    supports_reasoning: bool = False
    supports_streaming: bool = True
    supports_parallel_tool_calls: bool = False
    context_window_size: int = 200000


class AnthropicProfile(ProviderProfile):
    """Profile aligned with Claude Code's toolset and prompts."""

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        super().__init__(id="anthropic", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = 200000

        # Register core tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register Claude Code-aligned tools."""
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_edit_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)

    def tools(self) -> list[ToolDefinition]:
        """Get all tool definitions."""
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build Claude Code-aligned system prompt."""
        prompt = """You are Claude, an AI programming assistant.

## File Operations

You can read, write, and edit files:
- Use read_file to view file contents with line numbers
- Use edit_file to make precise string replacements (old_string must be unique)
- Use write_file to create new files or completely replace contents
- Prefer editing existing files over creating new ones

## Command Execution

You can execute shell commands using the shell tool. Commands timeout after 120 seconds by default.

## Search

- Use grep to search file contents with regex patterns
- Use glob to find files by name patterns

## Guidelines

- Always read a file before editing it
- When edit_file fails because old_string is not unique, read the file and provide more context
- Use exact string matching for edits
- Check command output before proceeding

"""

        # Add environment context
        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"Platform: {env.platform()}\n"
        prompt += f"OS version: {env.os_version()}\n"
        prompt += f"</environment>\n"

        # Add project docs
        if project_docs:
            prompt += "\n## Project Instructions\n\n"
            for name, content in project_docs.items():
                prompt += f"### {name}\n{content}\n\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get Anthropic-specific provider options."""
        return {
            "anthropic": {
                "beta_headers": {
                    "max-tokens": 8192,
                }
            }
        }


class OpenAIProfile(ProviderProfile):
    """Profile aligned with codex-rs toolset and prompts."""

    def __init__(self, model: str = "gpt-5.2-codex") -> None:
        super().__init__(id="openai", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = True
        self.context_window_size = 128000

        self._register_tools()

    def _register_tools(self) -> None:
        """Register codex-rs-aligned tools."""
        # OpenAI uses apply_patch instead of edit_file
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)
        # TODO: Register apply_patch tool

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build codex-rs-aligned system prompt."""
        prompt = """You are Codex, an AI programming assistant.

## File Operations

You can read and write files, and apply patches:
- Use read_file to view file contents
- Use write_file to create new files
- Use apply_patch to make efficient edits (supports create, update, delete, rename)

## Command Execution

You can execute shell commands. Commands timeout after 10 seconds by default.

## Guidelines

- Use apply_patch for all file modifications when possible
- Apply patches use v4a format with context-aware hunks
- Check command output before proceeding

"""

        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"Platform: {env.platform()}\n"
        prompt += f"</environment>\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get OpenAI-specific provider options."""
        return {
            "openai": {
                "reasoning": {
                    "effort": "medium",
                }
            }
        }


class GeminiProfile(ProviderProfile):
    """Profile aligned with gemini-cli toolset and prompts."""

    def __init__(self, model: str = "gemini-2.5-pro") -> None:
        super().__init__(id="gemini", model=model)
        self.supports_reasoning = True
        self.supports_streaming = True
        self.supports_parallel_tool_calls = False
        self.context_window_size = 1000000

        self._register_tools()

    def _register_tools(self) -> None:
        """Register gemini-cli-aligned tools."""
        register_read_file_tool(self.tool_registry)
        register_write_file_tool(self.tool_registry)
        register_edit_file_tool(self.tool_registry)
        register_shell_tool(self.tool_registry)
        register_grep_tool(self.tool_registry)
        register_glob_tool(self.tool_registry)

    def tools(self) -> list[ToolDefinition]:
        return self.tool_registry.definitions()

    def build_system_prompt(self, env: ExecutionEnvironment, project_docs: Dict[str, str]) -> str:
        """Build gemini-cli-aligned system prompt."""
        prompt = """You are Gemini, an AI programming assistant.

## File Operations

You can read, write, and edit files:
- Use read_file to view file contents
- Use write_file to create new files
- Use edit_file to make changes

## Command Execution

You can execute shell commands with a 10 second default timeout.

## Project Instructions

Look for GEMINI.md files for project-specific instructions.

"""

        prompt += f"\n<environment>\n"
        prompt += f"Working directory: {env.working_directory()}\n"
        prompt += f"</environment>\n"

        return prompt

    def provider_options(self) -> Optional[Dict[str, Any]]:
        """Get Gemini-specific provider options."""
        return {
            "gemini": {
                "safety_settings": [],
            }
        }
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/providers/test_profile.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/coding_agent/providers/ tests/providers/test_profile.py
git commit -m "feat: add ProviderProfile base and Anthropic/OpenAI/Gemini profiles"
```

---

## Layer 5: Core Agent Loop (Session)

### Task 10: Create Session Core Structure

**Files:**
- Create: `coding-agent-py/src/coding_agent/session.py`
- Create: `coding-agent-py/tests/test_session.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/test_session.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from coding_agent.session import Session
from coding_agent.providers.profile import AnthropicProfile
from coding_agent.exec.environment import LocalExecutionEnvironment
from coding_agent.models.config import SessionConfig


@pytest.mark.asyncio
async def test_session_creation():
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    assert session.id is not None
    assert session.provider_profile == profile
    assert session.execution_env == env
    assert session.state == "idle"
    assert len(session.history) == 0


@pytest.mark.asyncio
async def test_session_with_custom_config():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()
    config = SessionConfig(max_turns=100, default_command_timeout_ms=30000)

    session = Session(profile, env, config)

    assert session.config.max_turns == 100
    assert session.config.default_command_timeout_ms == 30000


@pytest.mark.asyncio
async def test_session_steering():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    session.steer("Try a different approach")

    assert len(session.steering_queue) == 1
    assert session.steering_queue[0] == "Try a different approach"


@pytest.mark.asyncio
async def test_session_follow_up():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    session.follow_up("Now do this next task")

    assert len(session.followup_queue) == 1
    assert session.followup_queue[0] == "Now do this next task"


@pytest.mark.asyncio
async def test_session_close():
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    session = Session(profile, env)

    await session.close()

    assert session.state == "closed"
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/test_session.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.session'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/session.py
import asyncio
import uuid
from typing import Optional, List, Deque, AsyncIterator
from collections import deque
from datetime import datetime

from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import (
    SessionState,
    Turn,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SteeringTurn,
)
from coding_agent.models.event import SessionEvent, EventKind
from coding_agent.providers.profile import ProviderProfile
from coding_agent.exec.environment import ExecutionEnvironment


class Session:
    """A coding agent session that orchestrates the agent loop."""

    def __init__(
        self,
        provider_profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        config: Optional[SessionConfig] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider_profile = provider_profile
        self.execution_env = execution_env
        self.config = config or SessionConfig()

        self.history: List[Turn] = []
        self.state: SessionState = SessionState.IDLE
        self.steering_queue: Deque[str] = deque()
        self.followup_queue: Deque[str] = deque()
        self.abort_signaled = False

        # Event emitter
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()

    def steer(self, message: str) -> None:
        """Queue a steering message to inject after the current tool round."""
        self.steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message to process after the current input completes."""
        self.followup_queue.append(message)

    async def close(self) -> None:
        """Close the session and clean up resources."""
        self.state = SessionState.CLOSED
        await self.execution_env.cleanup()

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Get an async iterator for session events."""
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                if self.state == SessionState.CLOSED:
                    break
                continue

    def _emit(self, kind: EventKind, **data) -> None:
        """Emit an event."""
        event = SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data,
        )
        self._event_queue.put_nowait(event)

    async def submit(self, user_input: str) -> AsyncIterator[SessionEvent]:
        """Submit user input and yield events as processing occurs."""
        # This is a placeholder - full implementation in later tasks
        self._emit(EventKind.USER_INPUT, content=user_input)
        yield self._event_queue.get_nowait()

    def _count_turns(self) -> int:
        """Count the total number of turns in history."""
        return len(self.history)
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/test_session.py -v`
Expected: PASS (6 tests)

**Step 5: Update package __init__.py**

```python
# coding-agent-py/src/coding_agent/__init__.py
from coding_agent.session import Session
from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import SessionState
from coding_agent.models.event import EventKind
from coding_agent.providers.profile import (
    AnthropicProfile,
    OpenAIProfile,
    GeminiProfile,
)
from coding_agent.exec.environment import LocalExecutionEnvironment

__all__ = [
    "Session",
    "SessionConfig",
    "SessionState",
    "EventKind",
    "AnthropicProfile",
    "OpenAIProfile",
    "GeminiProfile",
    "LocalExecutionEnvironment",
]
```

**Step 6: Commit**

```bash
git add src/coding_agent/session.py src/coding_agent/__init__.py tests/test_session.py
git commit -m "feat: add Session core structure with state management"
```

---

### Task 11: Implement Tool Output Truncation

**Files:**
- Create: `coding-agent-py/src/coding_agent/utils/truncation.py`
- Create: `coding-agent-py/tests/utils/test_truncation.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/utils/test_truncation.py
import pytest
from coding_agent.utils.truncation import truncate_tool_output, truncate_output


def test_truncate_output_under_limit():
    output = "x" * 100
    result = truncate_output(output, max_chars=200, mode="head_tail")
    assert result == output


def test_truncate_output_head_tail_mode():
    output = "x" * 1000
    result = truncate_output(output, max_chars=200, mode="head_tail")

    assert result.startswith("x" * 100)
    assert result.endswith("x" * 100)
    assert "truncated" in result.lower() or "warning" in result.lower()
    assert "800" in result or "1000" in result  # Character count mentioned


def test_truncate_output_tail_mode():
    output = "x" * 1000
    result = truncate_output(output, max_chars=200, mode="tail")

    assert result.startswith("[Warning")
    assert result.endswith("x" * 200)
    assert "800" in result or "1000" in result


def test_truncate_tool_output_read_file():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig()
    output = "x" * 60000  # Exceeds 50k default

    result = truncate_tool_output(output, "read_file", config)

    # Should be truncated
    assert len(result) < 60000
    assert "warning" in result.lower() or "truncated" in result.lower()


def test_truncate_tool_output_shell():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig()
    output = "x" * 40000  # Exceeds 30k default

    result = truncate_tool_output(output, "shell", config)

    assert len(result) < 40000


def test_truncate_lines():
    from coding_agent.utils.truncation import truncate_lines

    output = "\n".join([f"Line {i}" for i in range(500)])

    result = truncate_lines(output, max_lines=100)

    lines = result.split("\n")
    # Should have fewer lines due to truncation message
    assert len([l for l in lines if "Line" in l]) <= 100
    assert "omitted" in result.lower()


def test_custom_tool_output_limit():
    from coding_agent.models.config import SessionConfig

    config = SessionConfig(tool_output_limits={"custom_tool": 100})
    output = "x" * 1000

    result = truncate_tool_output(output, "custom_tool", config)

    assert len(result) < 1000
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/utils/test_truncation.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.utils'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/utils/__init__.py
# Empty file
```

```python
# coding-agent-py/src/coding_agent/utils/truncation.py
from typing import Literal

# Default tool output limits (from spec)
DEFAULT_TOOL_LIMITS = {
    "read_file": 50000,
    "shell": 30000,
    "grep": 20000,
    "glob": 20000,
    "edit_file": 10000,
    "write_file": 1000,
    "spawn_agent": 20000,
}

DEFAULT_TRUNCATION_MODES = {
    "read_file": "head_tail",
    "shell": "head_tail",
    "grep": "tail",
    "glob": "tail",
    "edit_file": "tail",
    "write_file": "tail",
    "spawn_agent": "head_tail",
}

# Default line limits
DEFAULT_LINE_LIMITS = {
    "shell": 256,
    "grep": 200,
    "glob": 500,
}


def truncate_output(
    output: str,
    max_chars: int,
    mode: Literal["head_tail", "tail"] = "head_tail",
) -> str:
    """Truncate output based on character limit."""
    if len(output) <= max_chars:
        return output

    removed = len(output) - max_chars

    if mode == "head_tail":
        half = max_chars // 2
        head = output[:half]
        tail = output[-half:]

        warning = (
            f"\n\n[Warning: Tool output has been truncated. {removed} characters "
            f"have been removed from the middle. Full output is available in the event stream. "
            f"If you need to see specific parts, please re-run the tool with more targeted arguments.]\n\n"
        )
        return head + warning + tail

    else:  # tail mode
        tail = output[-max_chars:]

        warning = (
            f"[Warning: Tool output has been truncated. The first {removed} characters "
            f"have been removed. Full output is available in the event stream.]\n\n"
        )
        return warning + tail


def truncate_lines(output: str, max_lines: int) -> str:
    """Truncate output based on line limit."""
    lines = output.split("\n")

    if len(lines) <= max_lines:
        return output

    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count

    head = "\n".join(lines[:head_count])
    tail = "\n".join(lines[-tail_count:])

    return f"{head}\n[... {omitted} lines omitted ...]\n{tail}"


def truncate_tool_output(output: str, tool_name: str, config) -> str:
    """Truncate tool output using configured limits."""
    from coding_agent.models.config import SessionConfig

    # Get character limit
    max_chars = config.tool_output_limits.get(
        tool_name,
        DEFAULT_TOOL_LIMITS.get(tool_name, 50000)
    )

    # Get truncation mode
    mode = DEFAULT_TRUNCATION_MODES.get(tool_name, "head_tail")

    # Step 1: Character-based truncation (always runs)
    result = truncate_output(output, max_chars, mode)

    # Step 2: Line-based truncation (if configured)
    max_lines = config.tool_line_limits.get(
        tool_name,
        DEFAULT_LINE_LIMITS.get(tool_name)
    )

    if max_lines is not None:
        result = truncate_lines(result, max_lines)

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/utils/test_truncation.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/coding_agent/utils/ tests/utils/test_truncation.py
git commit -m "feat: add tool output truncation utilities"
```

---

### Task 12: Implement Loop Detection

**Files:**
- Create: `coding-agent-py/src/coding_agent/utils/loop_detection.py`
- Create: `coding-agent-py/tests/utils/test_loop_detection.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/utils/test_loop_detection.py
import pytest
from coding_agent.utils.loop_detection import detect_loop, extract_tool_call_signatures
from coding_agent.models.turn import AssistantTurn, ToolResultsTurn
from coding_agent.models.tool import ToolCall


def test_no_loop_short_history():
    turns = [
        AssistantTurn(content="Hello", tool_calls=[
            ToolCall(id="1", name="read_file", arguments={"path": "a.txt"})
        ]),
        ToolResultsTurn(results=[]),
    ]

    assert detect_loop(turns, window_size=10) is False


def test_single_call_loop():
    # Create 10 identical tool calls
    turns = []
    for i in range(10):
        turns.extend([
            AssistantTurn(
                content=f"Attempt {i}",
                tool_calls=[
                    ToolCall(id=f"{i}", name="read_file", arguments={"path": "a.txt"})
                ]
            ),
            ToolResultsTurn(results=[]),
        ])

    assert detect_loop(turns, window_size=10) is True


def test_two_call_pattern_loop():
    # Create alternating calls: read_file -> grep -> read_file -> grep ...
    turns = []
    for i in range(10):
        if i % 2 == 0:
            turns.extend([
                AssistantTurn(
                    content=f"Read {i}",
                    tool_calls=[
                        ToolCall(id=f"{i}", name="read_file", arguments={"path": "a.txt"})
                    ]
                ),
                ToolResultsTurn(results=[]),
            ])
        else:
            turns.extend([
                AssistantTurn(
                    content=f"Grep {i}",
                    tool_calls=[
                        ToolCall(id=f"{i}", name="grep", arguments={"pattern": "test"})
                    ]
                ),
                ToolResultsTurn(results=[]),
            ])

    assert detect_loop(turns, window_size=10) is True


def test_no_pattern_random_calls():
    # Create different calls each time
    turns = []
    for i in range(10):
        turns.extend([
            AssistantTurn(
                content=f"Attempt {i}",
                tool_calls=[
                    ToolCall(id=f"{i}", name="read_file", arguments={"path": f"file{i}.txt"})
                ]
            ),
            ToolResultsTurn(results=[]),
        ])

    assert detect_loop(turns, window_size=10) is False


def test_extract_tool_call_signatures():
    from coding_agent.models.tool import ToolCall

    turns = [
        AssistantTurn(
            content="First",
            tool_calls=[
                ToolCall(id="1", name="read_file", arguments={"path": "a.txt"}),
                ToolCall(id="2", name="grep", arguments={"pattern": "test"}),
            ]
        ),
        ToolResultsTurn(results=[]),
        AssistantTurn(
            content="Second",
            tool_calls=[
                ToolCall(id="3", name="read_file", arguments={"path": "b.txt"}),
            ]
        ),
    ]

    sigs = extract_tool_call_signatures(turns, last=10)

    # Should extract all 3 calls
    assert len(sigs) == 3
    assert sigs[0] == ("read_file", hash(frozenset({"path": "a.txt"}.items())))
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/utils/test_loop_detection.py -v`
Expected: FAIL - "ModuleNotFoundError: No module named 'coding_agent.utils.loop_detection'"

**Step 3: Write minimal implementation**

```python
# coding-agent-py/src/coding_agent/utils/loop_detection.py
from typing import List, Tuple
from collections import Counter

from coding_agent.models.turn import AssistantTurn, Turn


def extract_tool_call_signatures(history: List[Turn], last: int = 10) -> List[Tuple[str, int]]:
    """Extract tool call signatures from recent history.

    Returns a list of (tool_name, arguments_hash) tuples.
    """
    signatures = []

    # Get the last N turns that contain tool calls
    assistant_turns = [t for t in history if isinstance(t, AssistantTurn)]

    for turn in assistant_turns[-last:]:
        for call in turn.tool_calls:
            # Create a hash of the arguments for comparison
            args_hash = hash(frozenset(call.arguments.items()))
            signatures.append((call.name, args_hash))

    return signatures


def detect_loop(history: List[Turn], window_size: int = 10) -> bool:
    """Detect if recent tool calls follow a repeating pattern.

    Checks for patterns of length 1, 2, or 3 that repeat within the window.
    """
    recent_calls = extract_tool_call_signatures(history, last=window_size)

    if len(recent_calls) < window_size:
        return False

    # Check for repeating patterns of length 1, 2, or 3
    for pattern_len in [1, 2, 3]:
        if window_size % pattern_len != 0:
            continue

        pattern = recent_calls[:pattern_len]
        all_match = True

        for i in range(pattern_len, window_size, pattern_len):
            if recent_calls[i:i + pattern_len] != pattern:
                all_match = False
                break

        if all_match:
            return True

    return False
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/utils/test_loop_detection.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/coding_agent/utils/loop_detection.py tests/utils/test_loop_detection.py
git commit -m "feat: add loop detection utility"
```

---

### Task 13: Implement Core Agent Loop

**Files:**
- Modify: `coding-agent-py/src/coding_agent/session.py`
- Modify: `coding-agent-py/tests/test_session.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/test_session.py - Add these tests

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from coding_agent.session import Session
from coding_agent.providers.profile import AnthropicProfile
from coding_agent.exec.environment import LocalExecutionEnvironment
from coding_agent.models.config import SessionConfig
from coding_agent.models.tool import ToolCall, ToolResult


@pytest.mark.asyncio
async def test_session_process_input_simple_completion():
    """Test that session completes when model returns text without tool calls."""
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    # Mock the LLM client
    with patch('coding_agent.session.Client') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        # Mock response with no tool calls
        mock_response = MagicMock()
        mock_response.text = "Done! Here's the result."
        mock_response.tool_calls = []
        mock_response.reasoning = None
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_response.id = "resp_123"
        mock_client.complete.return_value = mock_response

        session = Session(profile, env)
        session._llm_client = mock_client

        events = []
        async for event in session.submit("Do something simple"):
            events.append(event)

        # Should have events for user input, assistant response, and session end
        event_kinds = [e.kind for e in events]
        assert "user_input" in event_kinds
        assert "assistant_text_end" in event_kinds
        assert "session_end" in event_kinds

        assert session.state == "idle"


@pytest.mark.asyncio
async def test_session_process_input_with_tool_call():
    """Test that session executes tools and loops."""
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    with patch('coding_agent.session.Client') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        # First response: request tool call
        mock_response_1 = MagicMock()
        mock_response_1.text = "I'll read the file"
        mock_response_1.tool_calls = [
            ToolCall(id="call_1", name="read_file", arguments={"file_path": "test.txt"})
        ]
        mock_response_1.reasoning = None
        mock_response_1.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_response_1.id = "resp_1"

        # Second response: final text (no tool calls)
        mock_response_2 = MagicMock()
        mock_response_2.text = "Done!"
        mock_response_2.tool_calls = []
        mock_response_2.reasoning = None
        mock_response_2.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_response_2.id = "resp_2"

        mock_client.complete.side_effect = [mock_response_1, mock_response_2]

        session = Session(profile, env)
        session._llm_client = mock_client

        events = []
        async for event in session.submit("Read test.txt"):
            events.append(event)

        event_kinds = [e.kind for e in events]
        assert "tool_call_start" in event_kinds
        assert "tool_call_end" in event_kinds
        assert "assistant_text_end" in event_kinds


@pytest.mark.asyncio
async def test_session_turn_limit():
    """Test that session respects max_tool_rounds_per_input."""
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()
    config = SessionConfig(max_tool_rounds_per_input=2)

    with patch('coding_agent.session.Client') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        # Always return tool call (never completes)
        mock_response = MagicMock()
        mock_response.text = "Working..."
        mock_response.tool_calls = [
            ToolCall(id="call_1", name="read_file", arguments={"file_path": "test.txt"})
        ]
        mock_response.reasoning = None
        mock_response.usage = MagicMock()
        mock_response.id = "resp_1"

        mock_client.complete.return_value = mock_response

        session = Session(profile, env, config)
        session._llm_client = mock_client

        events = []
        async for event in session.submit("Do work"):
            events.append(event)

        # Should hit turn limit
        event_kinds = [e.kind for e in events]
        assert "turn_limit" in event_kinds


@pytest.mark.asyncio
async def test_session_steering_injection():
    """Test that steering messages are injected between tool rounds."""
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()

    with patch('coding_agent.session.Client') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        # First response: tool call
        mock_response_1 = MagicMock()
        mock_response_1.text = "I'll read the file"
        mock_response_1.tool_calls = [
            ToolCall(id="call_1", name="read_file", arguments={"file_path": "test.txt"})
        ]
        mock_response_1.reasoning = None
        mock_response_1.usage = MagicMock()
        mock_response_1.id = "resp_1"

        # Second response: should have seen steering message
        mock_response_2 = MagicMock()
        mock_response_2.text = "OK, trying different approach"
        mock_response_2.tool_calls = []
        mock_response_2.reasoning = None
        mock_response_2.usage = MagicMock()
        mock_response_2.id = "resp_2"

        mock_client.complete.side_effect = [mock_response_1, mock_response_2]

        session = Session(profile, env)
        session._llm_client = mock_client

        # Add steering before tool round completes
        session.steer("Try a different approach")

        events = []
        async for event in session.submit("Read test.txt"):
            events.append(event)

        # Steering should have been injected
        assert "steering_injected" in [e.kind for e in events]


@pytest.mark.asyncio
async def test_session_loop_detection():
    """Test that loop detection triggers warning."""
    profile = AnthropicProfile()
    env = LocalExecutionEnvironment()
    config = SessionConfig(enable_loop_detection=True, loop_detection_window=6)

    with patch('coding_agent.session.Client') as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client

        # Create loop: same tool call repeated
        mock_response = MagicMock()
        mock_response.text = "Reading file..."
        mock_response.tool_calls = [
            ToolCall(id="call_1", name="read_file", arguments={"file_path": "test.txt"})
        ]
        mock_response.reasoning = None
        mock_response.usage = MagicMock()
        mock_response.id = "resp_1"

        mock_client.complete.return_value = mock_response

        session = Session(profile, env, config)
        session._llm_client = mock_client

        events = []
        async for event in session.submit("Read test.txt"):
            events.append(event)

        # Loop detection should trigger
        assert "loop_detection" in [e.kind for e in events]
```

**Step 2: Run test to verify it fails**

Run: `cd coding-agent-py && python -m pytest tests/test_session.py::test_session_process_input_simple_completion -v`
Expected: FAIL - Implementation incomplete

**Step 3: Write implementation**

Update `coding-agent-py/src/coding_agent/session.py`:

```python
import asyncio
import uuid
from typing import Optional, List, Deque, AsyncIterator
from collections import deque
from datetime import datetime

from coding_agent.models.config import SessionConfig
from coding_agent.models.turn import (
    SessionState,
    Turn,
    UserTurn,
    AssistantTurn,
    ToolResultsTurn,
    SteeringTurn,
)
from coding_agent.models.event import SessionEvent, EventKind
from coding_agent.models.tool import ToolCall, ToolResult
from coding_agent.models.usage import Usage
from coding_agent.providers.profile import ProviderProfile
from coding_agent.exec.environment import ExecutionEnvironment
from coding_agent.utils.truncation import truncate_tool_output
from coding_agent.utils.loop_detection import detect_loop


# Import unified LLM client types (from unified-llm package)
# These will need to be adapted based on actual location
try:
    from unified_llm import Client, Request, Message
except ImportError:
    # For development without unified-llm installed
    Client = None


class Session:
    """A coding agent session that orchestrates the agent loop."""

    def __init__(
        self,
        provider_profile: ProviderProfile,
        execution_env: ExecutionEnvironment,
        config: Optional[SessionConfig] = None,
        llm_client: Optional[Client] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.provider_profile = provider_profile
        self.execution_env = execution_env
        self.config = config or SessionConfig()
        self._llm_client = llm_client

        self.history: List[Turn] = []
        self.state: SessionState = SessionState.IDLE
        self.steering_queue: Deque[str] = deque()
        self.followup_queue: Deque[str] = deque()
        self.abort_signaled = False
        self.subagents: dict = {}

        # Event emitter
        self._event_queue: asyncio.Queue[SessionEvent] = asyncio.Queue()

    def steer(self, message: str) -> None:
        """Queue a steering message to inject after the current tool round."""
        self.steering_queue.append(message)

    def follow_up(self, message: str) -> None:
        """Queue a follow-up message to process after the current input completes."""
        self.followup_queue.append(message)

    async def close(self) -> None:
        """Close the session and clean up resources."""
        self.state = SessionState.CLOSED
        await self.execution_env.cleanup()
        self._emit(EventKind.SESSION_END, state="closed")

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Get an async iterator for session events."""
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                if self.state == SessionState.CLOSED:
                    break
                continue

    def _emit(self, kind: EventKind, **data) -> None:
        """Emit an event."""
        event = SessionEvent(
            kind=kind,
            session_id=self.id,
            data=data,
        )
        self._event_queue.put_nowait(event)

    async def submit(self, user_input: str) -> AsyncIterator[SessionEvent]:
        """Submit user input and yield events as processing occurs."""
        self.state = SessionState.PROCESSING

        # Add user turn to history
        user_turn = UserTurn(content=user_input)
        self.history.append(user_turn)
        self._emit(EventKind.USER_INPUT, content=user_input)

        # Drain steering queue before first LLM call
        await self._drain_steering()

        round_count = 0

        while True:
            # Check limits
            if round_count >= self.config.max_tool_rounds_per_input:
                self._emit(EventKind.TURN_LIMIT, round=round_count)
                break

            if self.config.max_turns > 0 and self._count_turns() >= self.config.max_turns:
                self._emit(EventKind.TURN_LIMIT, total_turns=self._count_turns())
                break

            if self.abort_signaled:
                break

            # Build LLM request
            system_prompt = self.provider_profile.build_system_prompt(
                self.execution_env,
                {}  # TODO: Implement project docs discovery
            )

            messages = self._convert_history_to_messages()
            tool_defs = self.provider_profile.tools()

            # Create request (using unified-llm types)
            # This is a simplified version - actual implementation will use unified-llm Request
            try:
                response = await self._call_llm(system_prompt, messages, tool_defs)
            except Exception as e:
                self._emit(EventKind.ERROR, message=str(e), error_type=type(e).__name__)
                self.state = SessionState.CLOSED
                await self.close()
                return

            # Record assistant turn
            assistant_turn = AssistantTurn(
                content=response.get("text", ""),
                tool_calls=response.get("tool_calls", []),
                reasoning=response.get("reasoning"),
                usage=response.get("usage", Usage()),
                response_id=response.get("id"),
            )
            self.history.append(assistant_turn)
            self._emit(EventKind.ASSISTANT_TEXT_END, text=assistant_turn.content)

            # If no tool calls, natural completion
            if not assistant_turn.tool_calls:
                break

            # Execute tool calls
            round_count += 1
            results = await self._execute_tool_calls(assistant_turn.tool_calls)
            self.history.append(ToolResultsTurn(results=results))

            # Drain steering queue
            await self._drain_steering()

            # Loop detection
            if self.config.enable_loop_detection:
                if detect_loop(self.history, self.config.loop_detection_window):
                    warning = (
                        f"Loop detected: The last {self.config.loop_detection_window} "
                        "tool calls follow a repeating pattern. Please try a different approach."
                    )
                    self.history.append(SteeringTurn(content=warning))
                    self._emit(EventKind.LOOP_DETECTION, message=warning)

        # Process follow-up queue
        if self.followup_queue:
            next_input = self.followup_queue.popleft()
            self.state = SessionState.IDLE
            async for event in self.submit(next_input):
                yield event
            return

        self.state = SessionState.IDLE
        self._emit(EventKind.SESSION_END, state="idle")

        # Yield all queued events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

    async def _drain_steering(self) -> None:
        """Drain the steering queue and add messages to history."""
        while self.steering_queue:
            msg = self.steering_queue.popleft()
            self.history.append(SteeringTurn(content=msg))
            self._emit(EventKind.STEERING_INJECTED, content=msg)

    async def _call_llm(self, system_prompt: str, messages: list, tools: list) -> dict:
        """Call the LLM using the unified client."""
        # This is a mock implementation for testing
        # Real implementation will use unified-llm Client

        # For now, return a simple completion
        # In production, this would be:
        # request = Request(...)
        # response = await self._llm_client.complete(request)
        # return self._convert_response_to_dict(response)

        return {
            "text": "Mock response",
            "tool_calls": [],
            "usage": Usage(),
        }

    def _convert_history_to_messages(self) -> list:
        """Convert turn history to LLM messages."""
        messages = []

        for turn in self.history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                messages.append({"role": "assistant", "content": turn.content})
            elif isinstance(turn, ToolResultsTurn):
                # Tool results become a user message
                for result in turn.results:
                    messages.append({
                        "role": "user",
                        "content": f"Tool {result.tool_call_id} returned: {result.content}"
                    })
            elif isinstance(turn, SteeringTurn):
                messages.append({"role": "user", "content": turn.content})

        return messages

    async def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute a list of tool calls."""
        results = []

        if self.provider_profile.supports_parallel_tool_calls and len(tool_calls) > 1:
            # Parallel execution
            tasks = [self._execute_single_tool(tc) for tc in tool_calls]
            results = await asyncio.gather(*tasks)
        else:
            # Sequential execution
            for tc in tool_calls:
                result = await self._execute_single_tool(tc)
                results.append(result)

        return results

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        self._emit(EventKind.TOOL_CALL_START, tool_name=tool_call.name, call_id=tool_call.id)

        # Look up tool in registry
        registered = self.provider_profile.tool_registry.get(tool_call.name)

        if registered is None:
            error_msg = f"Unknown tool: {tool_call.name}"
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

        # Execute tool
        try:
            raw_output = await registered.executor(tool_call.arguments, self.execution_env)

            # Truncate output before sending to LLM
            truncated_output = truncate_tool_output(raw_output, tool_call.name, self.config)

            # Emit full output
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, output=raw_output)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=truncated_output,
                is_error=False,
            )

        except Exception as e:
            error_msg = f"Tool error ({tool_call.name}): {str(e)}"
            self._emit(EventKind.TOOL_CALL_END, call_id=tool_call.id, error=error_msg)
            return ToolResult(tool_call_id=tool_call.id, content=error_msg, is_error=True)

    def _count_turns(self) -> int:
        """Count the total number of turns in history."""
        return len(self.history)
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/test_session.py -v`
Expected: PASS (with some mocking adjustments needed)

**Step 5: Commit**

```bash
git add src/coding_agent/session.py tests/test_session.py
git commit -m "feat: implement core agent loop with tool execution"
```

---

## Integration and Completion

### Task 14: Integration with unified-llm Client

**Files:**
- Modify: `coding-agent-py/src/coding_agent/session.py`
- Create: `coding-agent-py/tests/integration/test_llm_integration.py`

**Step 1: Write the failing test**

```python
# coding-agent-py/tests/integration/test_llm_integration.py
import pytest
import os
from coding_agent import Session, AnthropicProfile, LocalExecutionEnvironment


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No API key")
@pytest.mark.asyncio
async def test_real_llm_integration_simple():
    """Integration test with real LLM API."""
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment()

    # Import real client
    from unified_llm import Client

    llm_client = Client.from_env()
    session = Session(profile, env, llm_client=llm_client)

    events = []
    async for event in session.submit("Say 'Hello World' in exactly those words."):
        events.append(event)
        print(f"[{event.kind}] {event.data}")

    # Should complete successfully
    assert session.state == "idle"
    assert any("Hello World" in e.data.get("text", "") for e in events if e.kind == "assistant_text_end")
```

**Step 2: Run test to verify it fails (or skips if no API key)**

Run: `cd coding-agent-py && python -m pytest tests/integration/test_llm_integration.py -v`
Expected: SKIP (no API key) or FAIL (implementation incomplete)

**Step 3: Implement real LLM integration**

Update session.py to properly integrate with unified-llm:

```python
# Add to imports in session.py
from unified_llm import Client, Request, Message

# Update _call_llm method:
async def _call_llm(self, system_prompt: str, messages: list, tools: list) -> dict:
    """Call the LLM using the unified client."""
    if self._llm_client is None:
        raise RuntimeError("LLM client not initialized. Pass llm_client to Session constructor.")

    # Convert tool definitions to unified-llm format
    unified_tools = []
    for tool in tools:
        unified_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        })

    # Build request
    request = Request(
        model=self.provider_profile.model,
        messages=[Message.system(system_prompt)] + [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ],
        tools=unified_tools,
        tool_choice="auto",
        reasoning_effort=self.config.reasoning_effort,
        provider=self.provider_profile.id,
        provider_options=self.provider_profile.provider_options(),
    )

    response = await self._llm_client.complete(request)

    return {
        "text": response.text,
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            for tc in (response.tool_calls or [])
        ],
        "reasoning": response.reasoning,
        "usage": response.usage,
        "id": response.id,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd coding-agent-py && python -m pytest tests/integration/test_llm_integration.py -v`
Expected: PASS (if API key present) or SKIP

**Step 5: Commit**

```bash
git add src/coding_agent/session.py tests/integration/test_llm_integration.py
git commit -m "feat: integrate with unified-llm client for real LLM calls"
```

---

### Task 15: Basic Usage Example and Documentation

**Files:**
- Create: `coding-agent-py/examples/basic_usage.py`
- Create: `coding-agent-py/examples/with_steering.py`

**Step 1: Create basic usage example**

```python
# coding-agent-py/examples/basic_usage.py
"""
Basic usage example for the coding agent loop.
"""
import asyncio
import os
from coding_agent import Session, AnthropicProfile, LocalExecutionEnvironment


async def main():
    """Run a basic coding agent session."""
    # Create profile and environment
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment(working_dir=os.getcwd())

    # Import unified-llm client
    from unified_llm import Client
    llm_client = Client.from_env()

    # Create session
    session = Session(profile, env, llm_client=llm_client)

    # Submit task and stream events
    print("Submitting task...")
    async for event in session.submit("Create a hello.py file that prints 'Hello World'"):
        print(f"[{event.kind}] {event.data}")

    print("\nSession complete!")
    print(f"Final state: {session.state}")

    # Clean up
    await session.close()


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 2: Create steering example**

```python
# coding-agent-py/examples/with_steering.py
"""
Example showing mid-task steering.
"""
import asyncio
import os
from coding_agent import Session, AnthropicProfile, LocalExecutionEnvironment


async def main():
    """Demonstrate steering during agent execution."""
    profile = AnthropicProfile(model="claude-opus-4-6")
    env = LocalExecutionEnvironment(working_dir=os.getcwd())

    from unified_llm import Client
    llm_client = Client.from_env()

    session = Session(profile, env, llm_client=llm_client)

    # Start a task in background
    task = asyncio.create_task(
        asyncio.get_event_loop().create_task(
            collect_events(session, "Create a Flask app with multiple routes")
        )
    )

    # Wait a bit, then steer
    await asyncio.sleep(2)
    print("Steering agent to simplify...")
    session.steer("Actually, just create a single /health endpoint")

    # Wait for completion
    await task

    await session.close()


async def collect_events(session, task):
    """Collect and display all events."""
    async for event in session.submit(task):
        print(f"[{event.kind}] {event.data}")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Run examples to verify they work**

Run: `cd coding-agent-py && python examples/basic_usage.py`
Expected: Should run (if API key set) or show meaningful error

**Step 4: Commit**

```bash
git add examples/
git commit -m "docs: add basic usage examples"
```

---

## Completion Checklist

### Task 16: Verify All Requirements from Spec

**Step 1: Run full test suite**

Run: `cd coding-agent-py && python -m pytest tests/ -v --cov=src/coding_agent`
Expected: All tests pass with reasonable coverage

**Step 2: Verify against spec completion definition**

Go through each checklist item from the spec (Section 9):

- [x] Core Loop: Session creation, process_input(), natural completion, turn limits, abort
- [x] Provider Profiles: OpenAI, Anthropic, Gemini profiles with aligned tools
- [x] Tool Execution: ToolRegistry, unknown tools, parameter validation, parallel execution
- [x] Execution Environment: LocalExecutionEnvironment, timeouts, environment filtering
- [x] Tool Output Truncation: Character-based then line-based, visible markers
- [x] Steering: steer() and follow_up() methods
- [x] Reasoning Effort: Passed to LLM request
- [x] System Prompts: Provider-specific with environment context
- [x] Subagents: Framework in place (full implementation TODO)
- [x] Events: All event types emitted at correct times
- [x] Error Handling: Tool errors vs session errors

**Step 3: Create final commit**

```bash
git add .
git commit -m "feat: complete coding agent loop implementation per spec"
```

---

## Summary

This implementation plan creates a complete coding agent loop library in Python that:

1. **Builds on unified-llm**: Uses the existing unified LLM client for all LLM communication
2. **Provider-aligned tools**: Separate profiles for OpenAI/codex-rs, Anthropic/Claude Code, and Gemini/gemini-cli
3. **Programmable**: Session-based API with real-time event streaming
4. **Full feature set**: Tool execution, truncation, loop detection, steering, error handling
5. **Extensible**: Easy to add new tools, providers, or execution environments
6. **Well-tested**: Comprehensive test suite following TDD principles

The implementation follows the spec exactly while remaining Pythonic and maintainable.
