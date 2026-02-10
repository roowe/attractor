"""
智谱 AI (Zhipu AI / BigModel) 使用示例

智谱 API 文档: https://open.bigmodel.cn/dev/api

智谱支持 OpenAI-compatible 和 Anthropic-compatible 两种 API 格式。
"""

import asyncio
import os
from unified_llm import Client, OpenAIAdapter, AnthropicAdapter, Message, Request
from unified_llm.models import StreamEventType


async def main():
    """智谱 GLM-4.7 使用示例 (OpenAI 格式)"""

    # 方式 1: 直接创建 adapter（推荐）
    print("=== 方式 1: OpenAI 格式 ===")
    print("-" * 50)

    zhipu_adapter = OpenAIAdapter(
        api_key=os.environ.get("ZHIPU_API_KEY", "your-zhipu-api-key"),
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        api_format="chat_completions",
    )

    client = Client(providers={"zhipu": zhipu_adapter}, default_provider="zhipu")

    # 简单文本生成
    request = Request(
        model="GLM-4.7",
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
        model="GLM-4.7",
        messages=[Message.user("写一首关于春天的诗")],
    )

    print("回复: ", end="", flush=True)
    async for event in client.stream(request):
        #print(event)
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
        model="GLM-4.7",
        messages=messages,
        max_tokens=100,
    )

    response = await client.complete(request)
    print(f"Q: 量子计算有什么优势？")
    print(f"A: {response.text}\n")

    await client.close()


async def main_anthropic_format():
    """使用 Anthropic-compatible API 格式"""
    print("=== 方式 2: Anthropic API 格式 ===")
    print("-" * 50)

    zhipu_adapter = AnthropicAdapter(
        api_key=os.environ.get("ZHIPU_API_KEY", "your-zhipu-api-key"),
        base_url="https://open.bigmodel.cn/api/anthropic/v1",
    )

    client = Client(providers={"zhipu": zhipu_adapter}, default_provider="zhipu")

    request = Request(
        model="GLM-4.7",
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
    """方式 3: 通过环境变量配置"""
    print("=== 方式 3: 使用环境变量 ===")
    print("-" * 50)

    # OpenAI 格式环境变量配置:
    # export OPENAI_API_KEY=your-zhipu-key
    # export OPENAI_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4

    # Anthropic 格式环境变量配置:
    # export ANTHROPIC_AUTH_TOKEN=your-zhipu-key
    # export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic

    client = Client.from_env()

    request = Request(
        model="GLM-4.7",
        messages=[Message.user("你好！")],
    )

    response = await client.complete(request)
    print(f"回复: {response.text}")

    await client.close()


async def main_with_custom_model():
    """方式 4: 尝试不同的智谱模型"""
    print("=== 方式 4: 使用不同的智谱模型 ===")
    print("-" * 50)

    zhipu_adapter = OpenAIAdapter(
        api_key=os.environ.get("ZHIPU_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        api_format="chat_completions",
    )

    client = Client(providers={"zhipu": zhipu_adapter}, default_provider="zhipu")

    # 可用的智谱模型（根据官方文档）
    models = [
        "GLM-4.7",           # 最新旗舰模型
        "GLM-4-Flash",       # 高速模型
        "GLM-4-Air",         # 轻量模型
        "GLM-4-Plus",        # 增强模型
        "glm-4-flash",       # 兼容命名
        "glm-4-air",
    ]

    for model in models:
        try:
            request = Request(
                model=model,
                messages=[Message.user(f"你是{model}吗？请简短回答。")],
                max_tokens=50,
            )
            response = await client.complete(request)
            print(f"✓ {model}: {response.text[:50]}...")
        except Exception as e:
            print(f"✗ {model}: {e}")

    await client.close()


if __name__ == "__main__":
    # 检查 API key
    if not os.environ.get("ZHIPU_API_KEY"):
        print("请设置 ZHIPU_API_KEY 环境变量")
        print("\n获取 API key:")
        print("1. 访问 https://open.bigmodel.cn/")
        print("2. 注册并获取 API Key")
        print("3. 运行: export ZHIPU_API_KEY=your-key-here")
        print("\n或者设置环境变量:")
        print("  # OpenAI 格式")
        print("  export OPENAI_API_KEY=your-zhipu-key")
        print("  export OPENAI_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4")
        print("\n  # Anthropic 格式")
        print("  export ANTHROPIC_AUTH_TOKEN=your-zhipu-key")
        print("  export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic")
        print("\n然后运行: uv run python examples/zhipu_example.py")
        exit(1)

    # 运行示例 - 选择你想尝试的方式:

    # 方式 1: OpenAI Chat Completions 格式 (推荐)
    asyncio.run(main())

    # 方式 2: Anthropic Messages API 格式
    #asyncio.run(main_anthropic_format())

    # 方式 3: 尝试不同模型
    # asyncio.run(main_with_custom_model())
