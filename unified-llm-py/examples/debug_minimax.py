"""
调试 MiniMax API 连接
"""
import asyncio
import os
import json

# 使用 httpx 直接测试
from unified_llm.utils._http import HttpClient


async def test_minimax_endpoints():
    """测试不同的 MiniMax endpoint"""

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        print("请设置 MINIMAX_API_KEY")
        return

    client = HttpClient()

    # 测试 1: OpenAI 格式
    print("=== 测试 1: OpenAI 格式 ===")
    print(f"URL: https://api.minimaxi.com/v1/chat/completions")

    try:
        response = await client.request(
            method="POST",
            url="https://api.minimaxi.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "codex-MiniMax-M2.1",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
            },
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text()[:200]}...")
    except Exception as e:
        print(f"错误: {e}")

    print()

    # 测试 2: Anthropic 格式 (标准 /messages)
    print("=== 测试 2: Anthropic 标准格式 ===")
    print(f"URL: https://api.minimaxi.com/anthropic/messages")

    try:
        response = await client.request(
            method="POST",
            url="https://api.minimaxi.com/anthropic/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "MiniMax-M2.1",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text()[:200]}...")
    except Exception as e:
        print(f"错误: {e}")

    print()

    # 测试 3: Anthropic 格式 (带 /v1/)
    print("=== 测试 3: Anthropic 带 /v1/ ===")
    print(f"URL: https://api.minimaxi.com/anthropic/v1/messages")

    try:
        response = await client.request(
            method="POST",
            url="https://api.minimaxi.com/anthropic/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "MiniMax-M2.1",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text()[:200]}...")
    except Exception as e:
        print(f"错误: {e}")

    print()

    # 测试 4: 尝试直接用 messages endpoint (无前缀)
    print("=== 测试 4: 直接 /v1/messages ===")
    print(f"URL: https://api.minimaxi.com/v1/messages")

    try:
        response = await client.request(
            method="POST",
            url="https://api.minimaxi.com/v1/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "MiniMax-M2.1",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text()[:200]}...")
    except Exception as e:
        print(f"错误: {e}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_minimax_endpoints())
