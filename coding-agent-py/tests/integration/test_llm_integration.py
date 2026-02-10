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
