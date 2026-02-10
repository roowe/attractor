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
    try:
        from unified_llm import Client
        llm_client = Client.from_env()
    except ImportError:
        print("unified-llm not installed. Using mock client.")
        llm_client = None

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
