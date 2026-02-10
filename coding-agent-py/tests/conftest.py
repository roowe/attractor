"""Pytest configuration and fixtures."""

import pytest
import asyncio


@pytest.fixture
def event_loop_policy():
    """Use the default event loop policy."""
    return asyncio.DefaultEventLoopPolicy()
