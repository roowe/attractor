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
