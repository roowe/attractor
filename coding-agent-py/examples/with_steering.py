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

    try:
        from unified_llm import Client
        llm_client = Client.from_env()
    except ImportError:
        print("unified-llm not installed. Using mock client.")
        llm_client = None

    session = Session(profile, env, llm_client=llm_client)

    # Start a task in background
    task = asyncio.create_task(
        collect_events(session, "Create a Flask app with multiple routes")
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
