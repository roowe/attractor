"""
MiniMax M2 使用示例

MiniMax API 文档: https://platform.minimaxi.com/document/TextChat%20V2

MiniMax 支持 OpenAI-compatible 和 Anthropic-compatible 两种 API 格式。
"""

import asyncio
import os
from unified_llm import Client, OpenAIAdapter, AnthropicAdapter, Message, Request
from unified_llm.models import StreamEventType


async def main():
    """MiniMax M2 使用示例"""

    # 方式 1: 直接创建 adapter（推荐）
    print("=== 方式 1: 直接创建 MiniMax adapter ===")
    print("-" * 50)

    minimax_adapter = OpenAIAdapter(
        api_key=os.environ.get("MINIMAX_API_KEY", "your-minimax-api-key"),
        base_url="https://api.minimaxi.com/v1",  # MiniMax API endpoint
        default_headers={
            "Authorization": f"Bearer {os.environ.get('MINIMAX_API_KEY', 'your-minimax-api-key')}",
        },
        api_format="chat_completions",  # 使用 Chat Completions API 格式
    )

    client = Client(providers={"minimax": minimax_adapter}, default_provider="minimax")

    # 简单文本生成
    request = Request(
        model="codex-MiniMax-M2.1",  # 或者 "abab6.5s-chat", "abab5.5-chat" 等
        messages=[Message.user("你好，请介绍一下你自己")],
        max_tokens=200,
    )

    response = await client.complete(request)
    print(f"模型: {response.model}")
    print(f"提供商: {response.provider}")
    print(f"回复: {response.text}")
    print(f"Token 使用: {response.usage.total_tokens}\n")

    # 流式生成
    print("=== 流式生成 ===")
    print("-" * 50)

    request = Request(
        model="codex-MiniMax-M2.1",
        messages=[Message.user("写一首关于春天的诗")],
    )

    print("回复: ", end="", flush=True)
    async for event in client.stream(request):
        if event.type == StreamEventType.TEXT_DELTA:
            print(event.delta, end="", flush=True)
    print("\n")

    # 多轮对话
    print("=== 多轮对话 ===")
    print("-" * 50)

    messages = [
        Message.system("你是一个友好的助手，擅长用简洁的语言回答问题。"),
        Message.user("什么是量子计算？"),
        Message.assistant("量子计算是利用量子力学原理进行计算的技术，使用量子比特（qubits）代替传统比特。"),
        Message.user("它有什么优势？"),
    ]

    request = Request(
        model="codex-MiniMax-M2.1",
        messages=messages,
        max_tokens=100,
    )

    response = await client.complete(request)
    print(f"Q: 量子计算有什么优势？")
    print(f"A: {response.text}\n")

    await client.close()


async def main_anthropic_format():
    """使用 Anthropic-compatible API 格式"""
    print("=== 使用 Anthropic API 格式 ===")
    print("-" * 50)

    # 方式 1: 使用 base_url 参数（注意：base_url 需要包含 /v1 路径）
    # MiniMax Anthropic-compatible endpoint
    minimax_adapter = AnthropicAdapter(
        api_key=os.environ.get("MINIMAX_API_KEY", "your-minimax-api-key"),
        base_url="https://api.minimaxi.com/anthropic/v1",  # 注意这里需要 /v1
    )

    # 方式 2: 或者通过环境变量配置（推荐）
    # export ANTHROPIC_API_KEY=your-minimax-key
    # export ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic/v1
    # adapter = AnthropicAdapter(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    client = Client(providers={"minimax": minimax_adapter}, default_provider="minimax")

    request = Request(
        model="MiniMax-M2.1",
        messages=[Message.user("你好，请介绍一下你自己")],
        max_tokens=200,
    )

    response = await client.complete(request)
    print(f"模型: {response.model}")
    print(f"提供商: {response.provider}")
    print(f"回复: {response.text}")
    print(f"Token 使用: {response.usage.total_tokens}\n")

    await client.close()


async def main_with_env():
    """方式 2: 通过环境变量配置"""
    print("=== 方式 2: 使用环境变量 ===")
    print("-" * 50)

    # 设置环境变量后使用 Client.from_env()
    # export OPENAI_API_KEY=your-minimax-key
    # export OPENAI_BASE_URL=https://api.minimax.chat/v1

    client = Client.from_env()

    request = Request(
        model="MiniMax-M2.1",
        messages=[Message.user("你好！")],
    )

    response = await client.complete(request)
    print(f"回复: {response.text}")

    await client.close()


async def main_with_custom_model():
    """方式 3: 尝试不同的 MiniMax 模型"""
    print("=== 方式 3: 使用不同的 MiniMax 模型 ===")
    print("-" * 50)

    minimax_adapter = OpenAIAdapter(
        api_key=os.environ.get("MINIMAX_API_KEY", ""),
        base_url="https://api.minimaxi.com/v1",
        api_format="chat_completions",
    )

    client = Client(providers={"minimax": minimax_adapter}, default_provider="minimax")

    # 可用的 MiniMax 模型（根据官方文档）
    models = [
        "MiniMax-M2.1",      # MiniMax 最新模型
        "abab6.5s-chat",     # abab6.5s-chat
        "abab5.5-chat",      # abab5.5-chat
        "abab5.5-chat-mini", # abab5.5-chat-mini
    ]

    for model in models:
        try:
            request = Request(
                model=model,
                messages=[Message.user(f"你是{model}吗？")],
                max_tokens=50,
            )
            response = await client.complete(request)
            print(f"✓ {model}: {response.text[:50]}...")
        except Exception as e:
            print(f"✗ {model}: {e}")

    await client.close()


if __name__ == "__main__":
    # 检查 API key
    if not os.environ.get("MINIMAX_API_KEY"):
        print("请设置 MINIMAX_API_KEY 环境变量")
        print("\n获取 API key:")
        print("1. 访问 https://platform.minimaxi.com/")
        print("2. 注册并获取 API Key")
        print("3. 运行: export MINIMAX_API_KEY=your-key-here")
        print("\n然后运行: uv run python examples/minimax_example.py")
        exit(1)

    # 运行示例 - 选择你想尝试的方式:

    # 方式 1: OpenAI Chat Completions 格式 (推荐，已验证可用)
    asyncio.run(main())

    # 方式 2: Anthropic Messages API 格式
    # 需要 base_url 包含 /v1 路径: https://api.minimaxi.com/anthropic/v1
    # asyncio.run(main_anthropic_format())

    # 方式 3: 尝试不同模型
    # asyncio.run(main_with_custom_model())
