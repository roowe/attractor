# 统一 LLM 客户端规范（精简版）

本文档定义了一个统一的 LLM 客户端接口，将 OpenAI、Anthropic 两家 API 的差异封装在适配器层。

> 注意：该demo不考虑多模态，也不考虑工具调用，返回只需要简单处理message即可。
> 模型调用，不考虑路由，直接使用指定提供商的适配器。
> 如果是生产环境，请参考更完整的规范文档：[统一 LLM 客户端规范（完整版）](unified-llm-full-spec-cn.md)，或者使用现成的库，太多bad case没有覆盖到。

本文档的动机：写个demo，观察不同 LLM 提供商的 API 差异，了解LLM调用的基本流程，并不是想完成一个完整的规范。

## 目录
1. [官方文档](#1-官方文档)
2. [数据模型](#2-数据模型)
3. [请求和响应的映射](#3-请求和响应的映射)
4. [模型适配器接口定义](#4-模型适配器接口定义)

## 1 官方文档

### Anthropic

文档：[https://platform.claude.com/docs/en/api/messages/create](https://platform.claude.com/docs/en/api/messages/create)

```
curl https://api.anthropic.com/v1/messages \
    -H 'Content-Type: application/json' \
    -H 'anthropic-version: 2023-06-01' \
    -H "X-Api-Key: $ANTHROPIC_API_KEY" \
    --max-time 600 \
    -d '{
          "max_tokens": 1024,
          "messages": [
            {
              "content": "Hello, world",
              "role": "user"
            }
          ],
          "model": "claude-opus-4-6"
        }'
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "content": [
    {
      "citations": [
        {
          "cited_text": "cited_text",
          "document_index": 0,
          "document_title": "document_title",
          "end_char_index": 0,
          "file_id": "file_id",
          "start_char_index": 0,
          "type": "char_location"
        }
      ],
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "model": "claude-opus-4-6",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_creation_input_tokens": 2051,
    "cache_read_input_tokens": 2051,
    "inference_geo": "inference_geo",
    "input_tokens": 2095,
    "output_tokens": 503,
    "server_tool_use": {
      "web_search_requests": 0
    },
    "service_tier": "standard"
  }
}
```

### OpenAI

文档：[https://developers.openai.com/api/reference/resources/responses/methods/create](https://developers.openai.com/api/reference/resources/responses/methods/create)

```
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "completed_at": 1741476543,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4.1-2025-04-14",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 87,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}
```


### OpenAI 兼容接口

文档： [https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)

```
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "VAR_chat_model_id",
    "messages": [
      {
        "role": "developer",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
{
  "id": "chatcmpl-B9MBs8CjcvOU2jLn4n570S5qMJKcT",
  "object": "chat.completion",
  "created": 1741569952,
  "model": "gpt-4.1-2025-04-14",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 19,
    "completion_tokens": 10,
    "total_tokens": 29,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "service_tier": "default"
}
```

---
## 2 数据模型

本节定义库使用的所有类型。符号使用语言无关的结构/记录风格。字段类型使用以下约定：

- `String` -- 文本
- `Integer` -- 整数
- `Float` -- 小数
- `Boolean` -- 真/假
- `Bytes` -- 原始二进制数据
- `Dict` -- 键值映射
- `List<T>` -- T 的有序集合
- `T | None` -- 可选（可空）值
- `T | U` -- 并集/任一类型

**Request and Response**

`complete()` 和 `stream()` 的单一输入类型：

```
RECORD Request:
    model             : String                      -- 必需；提供商的原生模型 ID
    messages          : List<Message>               -- 必需；对话
    provider          : String | None               -- 可选；如果省略则使用默认值
    response_format   : ResponseFormat | None       -- 可选；text、json 或 json_schema
    temperature       : Float | None
    top_p             : Float | None
    max_tokens        : Integer | None
    stop_sequences    : List<String> | None
    reasoning_effort  : String | None               -- "none"、"low"、"medium"、"high"
    metadata          : Dict<String, String> | None -- 任意键值对
    provider_options  : Dict | None                 -- 提供商特定参数的转义机制
```

```
RECORD Response:
    id              : String                -- 提供商分配的响应 ID
    model           : String                -- 实际使用的模型（可能与请求的不同）
    provider        : String                -- 哪个提供商完成了请求
    message         : Message               -- 助手的响应作为 Message
    raw             : Dict | None           -- 原始提供商响应 JSON（用于调试）
```


### 2.1 Message

对话的基本单位。对话是一个有序的 `List<Message>`。

```
RECORD Message:
    role          : Role                  -- 谁产生了此消息
    content       : List<ContentPart>     -- 消息主体（多模态）
    name          : String | None         -- 用于工具消息和开发者属性
    tool_call_id  : String | None         -- 将工具结果消息链接到其工具调用
```

### 2.2 Role

五个角色涵盖了所有主要提供商的语义：

```
ENUM Role:
    SYSTEM       -- 高级指令，塑造模型行为。通常是第一个。
    USER         -- 人类输入。文本、图像、音频、文档。
    ASSISTANT    -- 模型输出。文本、工具调用、思考块。
    DEVELOPER    -- 来自应用程序（而非最终用户）的特权指令。
```

角色的提供商映射：

| SDK 角色    | OpenAI                    | Anthropic                        |
|-------------|---------------------------|----------------------------------|
| SYSTEM      | `system` 角色             | 提取到 `system` 参数             | 
| USER        | `user` 角色               | `user` 角色                      |
| ASSISTANT   | `assistant` 角色          | `assistant` 角色                 | 
| DEVELOPER   | `developer` 角色          | 与系统合并                       | 

### 2.3 ContentPart（标记联合）

每个消息包含 ContentPart 对象列表。使用列表而不是单个字符串可以实现多模态消息（文本与图像交错）、结构化助手响应（文本与工具调用和思考块交错）以及包含图像的工具结果。

ContentPart 使用标记联合模式：`kind` 字段确定填充哪个数据字段。

```
RECORD ContentPart:
    kind          : ContentKind | String  -- 判别器标记
    text          : String | None         -- 当 kind == TEXT 时填充
```

注意：`kind` 字段接受枚举和任意字符串。这允许扩展特定于提供商的内容种类，而无需修改核心枚举。

### 2.4 ContentKind

```
ENUM ContentKind:
    TEXT                -- 纯文本。最常见的种类。
```

方向约束：

| 种类              | 可能出现在角色中                      |
|-------------------|---------------------------------------|
| TEXT              | SYSTEM、USER、ASSISTANT、DEVELOPER |



## 3、请求和响应的映射

开发过程中，根据实际情况逐一映射即可，在代码做好注释。


## 4. 模型适配器接口定义

### 4.1 适配器接口

所有提供商适配器实现统一接口：

```
INTERFACE ProviderAdapter:
    PROPERTY name : String             -- 提供商名称，如 "anthropic"、"openai"

    FUNCTION complete(request: Request) -> Response
        -- 发送请求，阻塞直到模型完成，返回完整响应

    FUNCTION stream(request: Request) -> AsyncIterator<StreamEvent>
        -- 发送请求，返回流式事件的异步迭代器
```

其中， `StreamEvent` 根据各家的返回，去做相应的定义和映射。TODO，或者暂时原样输出，demo中不处理。

### 4.2 使用示例

#### 创建适配器

```
# Anthropic
anthropic = AnthropicAdapter(api_key = "sk-ant-...")

# OpenAI (Responses API)
openai = OpenAIAdapter(
    api_key = "sk-...",
    api_format = "responses"  -- 或 "chat_completions"
)

# OpenAI 兼容端点 (第三方服务)
custom = OpenAIAdapter(
    api_key = "...",
    base_url = "https://your-endpoint.example.com/v1",
    api_format = "chat_completions"
)
```

#### 阻塞调用

```
response = anthropic.complete(Request(
    model = "claude-opus-4-6",
    messages = [Message(role="user", content=[...])],
    max_tokens = 500
))

print(response.message.content[0].text)
```

#### 流式调用

```
async for event in anthropic.stream(Request(
    model = "claude-opus-4-6",
    messages = [Message(role="user", content=[...])]
)):
    IF event.type == "text_delta":
        PRINT(event.text)
```

### 4.3 测试友好

直接使用适配器便于测试和依赖注入：

```
# 测试时注入 mock adapter
mock_adapter = MockAnthropicAdapter()
mock_adapter.complete.return_value = MockResponse(...)

# 被测函数接受 adapter 参数
def my_function(adapter: ProviderAdapter) -> str:
    response = adapter.complete(Request(...))
    return response.message.content[0].text
```
