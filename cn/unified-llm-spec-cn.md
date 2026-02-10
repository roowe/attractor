# 统一 LLM 客户端规范

本文档是一个整合的、与语言无关的规范，用于构建统一的客户端库，该库为多个 LLM 提供商（OpenAI、Anthropic、Google Gemini 等）提供单一接口。它被设计为可由任何开发人员或编码代理在任何编程语言中从头开始实现。

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [架构](#2-架构)
3. [数据模型](#3-数据模型)
4. [生成与流式传输](#4-生成与流式传输)
5. [工具调用](#5-工具调用)
6. [错误处理与重试](#6-错误处理与重试)
7. [提供商适配器契约](#7-提供商适配器契约)
8. [完成定义](#8-完成定义)

---

## 1. 概述与目标

### 1.1 问题陈述

使用大型语言模型的应用程序面临一个分散的生态系统。每个提供商——OpenAI、Anthropic、Google Gemini 等——都暴露不同的 HTTP API，具有不同的消息格式、工具调用约定、流式协议、错误形状和身份验证机制。切换提供商或支持多个提供商需要重写请求构建、响应解析、错误处理和流式逻辑。

本规范定义了一个解决此问题的统一客户端库。开发人员编写与提供商无关的代码，并通过更改单个字符串标识符来切换模型。无需重新布线，无需特定于适配器的导入。

### 1.2 设计原则

**提供商无关。** 应用程序代码不应包含特定于提供商的逻辑。统一接口处理所有转换。提供商特定的功能通过显式转义机制提供，而不是通过泄漏的抽象。

**最小表面积。** 库暴露少量类型和函数。开发人员可以在一小时内学习完整的 API。更少的概念意味着更少的错误和更容易的维护。

**流式优先。** 流式传输是一等操作，而不是阻塞调用上的标志。两种生成模式——阻塞和流式——具有独立的方法和不同的返回类型。这使类型系统为开发人员服务。

**可组合。** 横切关注点（日志记录、重试、缓存）通过中间件处理，而不是 baked into 核心。核心客户端是一个薄路由层。

**转义机制胜过虚假抽象。** 当提供商提供不映射到统一模型的独特功能时，库提供传递机制，而不是假装该功能不存在或构建不可靠的 shim。

### 1.3 参考开源项目

以下开源项目解决了相关问题，值得研究其模式、权衡和经验教训。它们不是依赖项；实现者可以从任何组合中汲取灵感。

- **Vercel AI SDK** (https://github.com/vercel/ai) -- TypeScript。多提供商架构，具有版本化的提供商规范。提供商接口与高级便捷 API（`generateText`/`streamText`/`generateObject`）之间清晰分离。展示了开始/增量/结束流式事件模式和可组合的中间件系统。

- **LiteLLM** (https://github.com/BerriAI/litellm) -- Python。在单个 `completion()` 接口后面支持 100 多个提供商。展示了统一调用约定的价值和模型字符串路由模式。展示了如何在规模上处理特定于提供商的怪异问题的长尾。

- **pi-ai** (https://github.com/nicktmro/pipe-ai) -- TypeScript。来自 @mariozechner 的 pi-mono 项目的多提供商 AI 客户端。展示了成本跟踪、使用聚合和清晰的提供商适配器模式，具有显式的推理令牌支持。

---

## 2. 架构

### 2.1 四层架构

库组织成四层，每层都有清晰的职责边界。

```
第 4 层：高级 API         generate(), stream(), generate_object()
          ---------------------------------------------------------------
第 3 层：核心客户端        Client、提供商路由、中间件钩子
          ---------------------------------------------------------------
第 2 层：提供商实用工具    用于构建适配器的共享助手
          ---------------------------------------------------------------
第 1 层：提供商规范        ProviderAdapter 接口、共享类型
```

**第 1 层——提供商规范。** 定义每个提供商适配器必须实现的契约。仅包含接口定义和共享类型定义。没有实现逻辑。这一层是稳定性契约：很少更改，仅在显式版本控制时更改。通过实现此接口来添加新提供商，而不是通过修改它。

**第 2 层——提供商实用工具。** 包含用于构建适配器的共享代码：HTTP 客户端助手、服务器发送事件 (SSE) 解析、重试逻辑、响应规范化实用程序、JSON 模式转换助手。提供商适配器作者导入这一层；应用程序开发人员通常不导入。

**第 3 层——核心客户端。** 主要编排层。`Client` 对象保存已注册的提供商适配器，按提供商标识符路由请求，应用中间件，并管理配置。这是想要直接控制请求的应用程序代码的主要导入。

**第 4 层——高级 API。** 提供便捷函数（`generate()`、`stream()`、`generate_object()`），这些函数使用符合人体工程学的默认值包装 Client。大多数应用程序代码使用这一层。这些函数处理提示标准化、工具执行循环、输出解析、结构化输出验证和自动重试。

### 2.2 客户端配置

#### 基于环境的设置

大多数应用程序的推荐设置读取每个提供商的标准环境变量：

```
client = Client.from_env()
```

环境变量约定：

| 提供商    | 必需变量              | 可选变量                                                  |
|-----------|----------------------|----------------------------------------------------------|
| OpenAI    | OPENAI_API_KEY       | OPENAI_BASE_URL、OPENAI_ORG_ID、OPENAI_PROJECT_ID      |
| Anthropic | ANTHROPIC_API_KEY    | ANTHROPIC_BASE_URL                                      |
| Gemini    | GEMINI_API_KEY       | GEMINI_BASE_URL                                         |

可以接受备用密钥名称（例如，`GOOGLE_API_KEY` 作为 `GEMINI_API_KEY` 的备用）。仅注册环境中存在的密钥的提供商。第一个注册的提供商成为默认提供商。

#### 编程设置

为了完全控制，显式构造适配器并在 Client 中注册：

```
adapter = OpenAIAdapter(
    api_key = "sk-...",
    base_url = "https://custom-endpoint.example.com/v1",
    default_headers = { "X-Custom": "value" },
    timeout = 30.0
)

client = Client(
    providers = { "openai": adapter },
    default_provider = "openai"
)
```

#### 提供商解析

当请求指定 `provider` 字段时，Client 路由到该适配器。当省略 provider 字段时，Client 使用 `default_provider`。如果未设置默认值且未指定提供商，Client 引发配置错误。Client 永远不会猜测。

#### 模型字符串约定

模型标识符是提供商的原生字符串（例如，`"gpt-5.2"`、`"claude-opus-4-6"`、`"gemini-3-flash-preview"`）。库不发明自己的模型命名空间。这避免了映射表的维护负担，并确保新模型立即工作，而无需库更新。如果模型字符串可能存在歧义（多个提供商支持它），请求上的 `provider` 字段会消除歧义。

### 2.3 中间件/拦截器模式

Client 支持用于横切关注点的中间件。中间件包装提供商调用，可以检查或修改请求、检查或修改响应以及执行副作用。

```
FUNCTION logging_middleware(request, next):
    LOG("Request to " + request.provider + "/" + request.model)
    response = next(request)
    LOG("Response: " + response.usage.total_tokens + " tokens")
    RETURN response

client = Client(
    providers = { ... },
    middleware = [logging_middleware]
)
```

**执行顺序。** 中间件在请求阶段按注册顺序运行（第一个注册 = 第一个执行），在响应阶段按相反顺序运行。这是标准的洋葱/责任链模式。

**流式中间件。** 中间件还必须应用于流式请求。对于流式传输，中间件包装事件迭代器，可以观察或转换单个流式事件。中间件接口应支持两种模式：

```
FUNCTION streaming_middleware(request, next):
    event_iterator = next(request)
    FOR EACH event IN event_iterator:
        log_event(event)
        YIELD event
```

**常见的中间件用例：**
- 日志记录
- 请求/响应缓存
- 成本跟踪和预算
- 客户端速率限制
- 提示注入检测
- 断路器模式

### 2.4 提供商适配器接口

每个提供商必须实现此接口：

```
INTERFACE ProviderAdapter:
    PROPERTY name : String             -- 例如，"openai"、"anthropic"、"gemini"

    FUNCTION complete(request: Request) -> Response
        -- 发送请求，阻塞直到模型完成，返回完整响应。

    FUNCTION stream(request: Request) -> AsyncIterator<StreamEvent>
        -- 发送请求，返回流式事件的异步迭代器。
```

**为什么是两种方法，而不是一种。** 拒绝使用带有 `stream: boolean` 标志的单一方法，因为返回类型根本不同。阻塞 `Response` 和异步事件流具有不同的消费模式、错误处理模型和生命周期语义。单独的方法使类型系统为开发人员服务。

**没有单独的 `send_tool_outputs` 方法。** 通过在新 `complete()` 或 `stream()` 调用的消息历史中包含工具结果来发送工具结果。这与 Anthropic 和 Gemini 原生工作方式相匹配。OpenAI 适配器在内部处理任何转换。

#### 可选的适配器方法

推荐这些方法，但不是必需的：

```
FUNCTION close() -> Void
    -- 释放资源（HTTP 连接等）。由 Client.close() 调用。

FUNCTION initialize() -> Void
    -- 在启动时验证配置。注册时由 Client 调用。

FUNCTION supports_tool_choice(mode: String) -> Boolean
    -- 查询是否支持特定的工具选择模式。
```

### 2.5 模块级默认客户端

高级函数（`generate()`、`stream()` 等）使用模块级默认客户端。此客户端在首次使用时从环境变量延迟初始化。应用程序可以覆盖它：

```
set_default_client(my_client)

-- 或每次调用显式传递：
result = generate(model = "...", prompt = "...", client = my_client)
```

### 2.6 并发模型

库是异步优先的。所有提供商调用都是非阻塞的。`complete()` 和 `stream()` 方法是异步的。高级 API 为支持两种范式的语言提供异步和同步包装器。

对不同提供商（或同一提供商）的多个并发请求是安全的。Client 在请求之间不持有可变状态。提供商适配器管理自己的连接池，必须可安全并发使用。

### 2.7 原生 API 使用（关键）

每个提供商适配器必须使用提供商的原生、首选 API——而不是兼容层。这是一个基本的设计要求。使用最小公分母兼容层（例如仅针对 OpenAI Chat Completions API 形状）会失去对提供商特定功能的访问，如推理令牌、扩展思考、提示缓存和高级工具功能。

| 提供商    | 必需 API                           | 为什么不使用兼容层                                              |
|-----------|-----------------------------------|---------------------------------------------------------------|
| OpenAI    | **Responses API** (`/v1/responses`) | Responses API 正确地表露出推理令牌，支持内置工具（Web 搜索、文件搜索、代码解释器），并且是 OpenAI 的前瞻性 API。Chat Completions API 不为推理模型（GPT-5.2 系列等）返回推理令牌，并且缺少服务器端对话状态。 |
| Anthropic | **Messages API** (`/v1/messages`)   | Messages API 支持带有思考块和签名的扩展思考，带有 `cache_control` 的提示缓存，beta 功能标头，以及严格的用户/助手交替模型。没有替代方案。 |
| Gemini    | **Gemini API** (`/v1beta/models/*/generateContent`) | 原生 Gemini API 支持使用 Google Search 进行 grounding、代码执行、系统指令和缓存内容。Gemini 的 OpenAI 兼容端点是有限的 shim。 |

统一的 SDK 抽象了这些不同的 API，以便调用者编写与提供商无关的代码，但内部每个适配器都讲提供商的原生协议。这是整个价值主张：三种不同 API 的复杂性在适配器中处理一次，以便下游消费者（如编码代理）永远不必考虑它。

### 2.8 提供商 Beta 标头和功能标志

提供商经常在 beta 标头或功能标志后面提供新功能。统一的 SDK 必须支持干净地传递这些功能。

**Anthropic beta 标头。** Anthropic 使用 `anthropic-beta` 标头来启用以下功能：
- `max-tokens-3-5-sonnet-2025-04-14` -- 为某些模型启用 1M 令牌上下文
- `interleaved-thinking-2025-05-14` -- 启用交错思考块
- `token-efficient-tools-2025-02-19` -- 更高效的工具令牌使用
- `prompt-caching-2024-07-31` -- 启用提示缓存

这些必须作为 HTTP 标头在请求上传递。适配器应该通过 `provider_options` 接受它们：

```
request = Request(
    model = "claude-opus-4-6",
    messages = [ ... ],
    provider_options = {
        "anthropic": {
            "beta_headers": ["interleaved-thinking-2025-05-14"]
        }
    }
)
```

Anthropic 适配器将这些连接到逗号分隔的 `anthropic-beta` 标头值。

**OpenAI 功能标志。** Responses API 支持通过请求主体启用内置工具和功能（例如，`tools: [{"type": "web_search_preview"}]`）。这些应该通过 `provider_options` 或扩展工具定义来支持。

**Gemini 配置。** Gemini 支持安全设置、grounding 配置和缓存的内容引用作为请求主体的一部分。这些应该可以通过 `provider_options` 传递。

关键原则：统一接口处理 90% 的常见情况。`provider_options` 转义机制处理剩余的 10%，而无需为每个新的提供商功能更改库。

### 2.9 模型目录

SDK 应该附带已知模型的目录，以帮助消费者（尤其是 AI 编码代理）选择有效的模型标识符，而无需猜测或幻觉模型名称。目录是建议性的，而不是限制性的——未知的模型字符串仍然会传递给提供商。

```
RECORD ModelInfo:
    id              : String            -- 模型的 API 标识符（例如，"claude-opus-4-6"）
    provider        : String            -- 哪个提供商提供此模型
    display_name    : String            -- 人类可读的名称（例如，"Claude Opus 4.6"）
    context_window  : Integer           -- 最大总令牌数（输入 + 输出）
    max_output      : Integer | None    -- 最大输出令牌
    supports_tools  : Boolean           -- 模型是否支持工具调用
    supports_vision : Boolean           -- 模型是否接受图像输入
    supports_reasoning : Boolean        -- 模型是否产生推理令牌
    input_cost_per_million  : Float | None  -- 每 100 万输入令牌的成本（美元）
    output_cost_per_million : Float | None  -- 每 100 万输出令牌的成本（美元）
    aliases         : List<String>      -- 简写名称（例如，["sonnet", "claude-sonnet"]）
```

**在撰写本文时（2026 年 2 月）**，每个提供商的 API 提供的顶级模型是：

| 提供商    | 顶级模型                                                 |
|-----------|----------------------------------------------------------|
| Anthropic | **Claude Opus 4.6**、Claude Sonnet 4.5                   |
| OpenAI    | **GPT-5.2 系列**（GPT-5.2、GPT-5.2-codex）              |
| Gemini    | **Gemini 3 Pro (Preview)**、Gemini 3 Flash (Preview) |

实现应该在调用者未指定模型时默认为最新的可用模型，并且应该在任何模型选择逻辑中更喜欢较新的模型。但是，目录还必须包含 API 仍在提供的较旧模型，因为调用者可能出于成本、延迟或兼容性原因需要它们。

示例目录（随着新模型发布保持更新）：

```
MODELS = [
    -- ==========================================================
    -- Anthropic -- 顶级质量首选 Claude Opus 4.6
    -- ==========================================================

    ModelInfo(id="claude-opus-4-6",               provider="anthropic", display_name="Claude Opus 4.6",   context_window=200000, supports_tools=true, supports_vision=true, supports_reasoning=true),
    ModelInfo(id="claude-sonnet-4-5",             provider="anthropic", display_name="Claude Sonnet 4.5", context_window=200000, supports_tools=true, supports_vision=true, supports_reasoning=true),

    -- ==========================================================
    -- OpenAI -- 顶级质量首选 GPT-5.2 系列
    -- ==========================================================

    ModelInfo(id="gpt-5.2",                       provider="openai",    display_name="GPT-5.2",           context_window=1047576, supports_tools=true, supports_vision=true, supports_reasoning=true),
    ModelInfo(id="gpt-5.2-mini",                  provider="openai",    display_name="GPT-5.2 Mini",      context_window=1047576, supports_tools=true, supports_vision=true, supports_reasoning=true),
    ModelInfo(id="gpt-5.2-codex",                 provider="openai",    display_name="GPT-5.2 Codex",     context_window=1047576, supports_tools=true, supports_vision=true, supports_reasoning=true),

    -- ==========================================================
    -- Gemini -- 最新版本首选 Gemini 3 Flash Preview
    -- ==========================================================

    ModelInfo(id="gemini-3-pro-preview",          provider="gemini",    display_name="Gemini 3 Pro (Preview)",   context_window=1048576, supports_tools=true, supports_vision=true, supports_reasoning=true),
    ModelInfo(id="gemini-3-flash-preview",        provider="gemini",    display_name="Gemini 3 Flash (Preview)", context_window=1048576, supports_tools=true, supports_vision=true, supports_reasoning=true),
]
```

**查找函数：**

```
get_model_info(model_id: String) -> ModelInfo | None
    -- 返回模型的目录条目，如果未知则返回 None。

list_models(provider: String | None) -> List<ModelInfo>
    -- 返回所有已知模型，可选择按提供商过滤。

get_latest_model(provider: String, capability: String | None) -> ModelInfo | None
    -- 返回提供商的最新/最佳模型，可选择按功能过滤
    -- （例如，"reasoning"、"vision"、"tools"）。对于想要始终使用最新可用模型的编码代理很有用。
```

**为什么目录对编码代理很重要：** 当 AI 编码代理在此 SDK 之上构建时，它需要按功能选择模型（例如，"选择支持视觉的模型"或"选择支持工具的最便宜模型"）。没有目录，代理必须从其训练数据中幻觉模型标识符，随着提供商发布新模型，这些标识符会过时。目录为代理提供可靠、最新的真实来源。

目录应该作为数据文件（JSON 或类似文件）提供，可以独立于库代码进行更新。考虑从提供商文档或 API 自动生成它。**如有疑问，首选最新模型**——它们通常更有能力，SDK 应该使保持最新变得容易。

### 2.10 提示缓存（对成本至关重要）

提示缓存允许提供商在对话前缀未更改时重用先前请求的计算。对于系统提示和对话历史在许多轮次中相同的代理工作负载，缓存可以将输入令牌成本降低 50-90%。统一的 SDK 必须支持每个提供商的缓存。

| 提供商    | 缓存行为                                              | SDK 需要的操作                              |
|-----------|------------------------------------------------------|-------------------------------------------|
| OpenAI    | 自动——Responses API 在服务器端缓存共享前缀              | 无。使用 Responses API 并从使用情况报告 `cache_read_tokens`。 |
| Gemini    | 自动——重复内容的前缀缓存，以及用于长上下文的显式 `cachedContent` API | 自动操作无。通过 `provider_options` 公开显式缓存。 |
| Anthropic | **不自动。** 需要在内容块上显式 `cache_control` 注释。 | Anthropic 适配器必须为代理工作负载自动注入 `cache_control` 断点。 |

Anthropic 是 SDK 必须做额外工作的唯一提供商。没有 cache_control 注释，每一轮都会以全价重新处理整个系统提示和对话历史。通过正确的缓存，缓存的输入令牌成本降低 90%。这是代理工作负载中 ROI 最高的优化。

所有三个提供商都报告缓存统计信息。SDK 必须将这些映射到 `Usage.cache_read_tokens` 和 `Usage.cache_write_tokens`，以便调用者可以验证缓存是否工作。

---

## 3. 数据模型

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

### 3.1 Message

对话的基本单位。对话是一个有序的 `List<Message>`。

```
RECORD Message:
    role          : Role                  -- 谁产生了此消息
    content       : List<ContentPart>     -- 消息主体（多模态）
    name          : String | None         -- 用于工具消息和开发者属性
    tool_call_id  : String | None         -- 将工具结果消息链接到其工具调用
```

#### 便捷构造函数

对于常见情况，工厂方法创建正确结构的 Message 对象：

```
Message.system("You are a helpful assistant.")
Message.user("What is 2 + 2?")
Message.assistant("The answer is 4.")
Message.tool_result(tool_call_id = "call_123", content = "72F and sunny", is_error = false)
```

#### 文本访问器

Message 上的便捷属性，连接所有文本内容部分的文本：

```
message.text -> String
    -- 返回所有 kind == TEXT 的 ContentPart 条目的连接。
    -- 如果不存在文本部分，则返回空字符串。
```

### 3.2 Role

五个角色涵盖了所有主要提供商的语义：

```
ENUM Role:
    SYSTEM       -- 高级指令，塑造模型行为。通常是第一个。
    USER         -- 人类输入。文本、图像、音频、文档。
    ASSISTANT    -- 模型输出。文本、工具调用、思考块。
    TOOL         -- 工具执行结果，通过 tool_call_id 链接。
    DEVELOPER    -- 来自应用程序（而非最终用户）的特权指令。
```

角色的提供商映射：

| SDK 角色    | OpenAI                    | Anthropic                        | Gemini                |
|-------------|---------------------------|----------------------------------|-----------------------|
| SYSTEM      | `system` 角色             | 提取到 `system` 参数             | `systemInstruction`   |
| USER        | `user` 角色               | `user` 角色                      | `user` 角色           |
| ASSISTANT   | `assistant` 角色          | `assistant` 角色                 | `model` 角色          |
| TOOL        | `tool` 角色               | 用户消息中的 `tool_result` 块     | 用户中的 `functionResponse` |
| DEVELOPER   | `developer` 角色          | 与系统合并                       | 与系统合并            |

### 3.3 ContentPart（标记联合）

每个消息包含 ContentPart 对象列表。使用列表而不是单个字符串可以实现多模态消息（文本与图像交错）、结构化助手响应（文本与工具调用和思考块交错）以及包含图像的工具结果。

ContentPart 使用标记联合模式：`kind` 字段确定填充哪个数据字段。

```
RECORD ContentPart:
    kind          : ContentKind | String  -- 判别器标记
    text          : String | None         -- 当 kind == TEXT 时填充
    image         : ImageData | None      -- 当 kind == IMAGE 时填充
    audio         : AudioData | None      -- 当 kind == AUDIO 时填充
    document      : DocumentData | None   -- 当 kind == DOCUMENT 时填充
    tool_call     : ToolCallData | None   -- 当 kind == TOOL_CALL 时填充
    tool_result   : ToolResultData | None -- 当 kind == TOOL_RESULT 时填充
    thinking      : ThinkingData | None   -- 当 kind == THINKING 或 REDACTED_THINKING 时填充
```

注意：`kind` 字段接受枚举和任意字符串。这允许扩展特定于提供商的内容种类，而无需修改核心枚举。

### 3.4 ContentKind

```
ENUM ContentKind:
    TEXT                -- 纯文本。最常见的种类。
    IMAGE               -- 图像作为 URL、base64 或文件引用。
    AUDIO               -- 音频作为 URL 或带有媒体类型的原始字节。
    DOCUMENT            -- 文档（PDF 等）作为 URL、base64 或文件引用。
    TOOL_CALL           -- 模型发起的工具调用。
    TOOL_RESULT         -- 执行工具调用的结果。
    THINKING            -- 模型推理/思考内容。
    REDACTED_THINKING   -- 编辑的推理 (Anthropic)。不透明，必须逐字往返。
```

方向约束：

| 种类              | 可能出现在角色中                      |
|-------------------|---------------------------------------|
| TEXT              | SYSTEM、USER、ASSISTANT、DEVELOPER、TOOL |
| IMAGE             | USER（输入）、ASSISTANT（生成）        |
| AUDIO             | USER（输入）                          |
| DOCUMENT          | USER（输入）                          |
| TOOL_CALL         | ASSISTANT（输出）                     |
| TOOL_RESULT       | TOOL（响应）                          |
| THINKING          | ASSISTANT（输出）                     |
| REDACTED_THINKING | ASSISTANT（输出）                     |

### 3.5 内容数据结构

#### ImageData

```
RECORD ImageData:
    url         : String | None     -- 指向图像的 URL
    data        : Bytes | None      -- 原始图像字节
    media_type  : String | None     -- MIME 类型，例如 "image/png"、"image/jpeg"
    detail      : String | None     -- 处理保真度提示："auto"、"low"、"high"
```

必须提供 `url` 或 `data` 之一。如果提供商需要，适配器对 `data` 进行 base64 编码。当提供 `data` 且未指定类型时，`media_type` 默认为 `"image/png"`。

**图像上传对多模态功能至关重要。** 许多模型（Claude、GPT-4.1、Gemini）接受图像输入进行分析、代码截图阅读、图表理解等。SDK 必须在所有提供商之间正确处理图像上传：

| 关注点            | OpenAI                                              | Anthropic                                        | Gemini                                             |
|-------------------|-----------------------------------------------------|--------------------------------------------------|----------------------------------------------------|
| URL 图像          | `image_url.url` 字段                                | `source.type = "url"` 与 `url` 字段              | `fileData.fileUri` 字段                           |
| Base64 图像       | `image_url.url` 作为数据 URI (`data:mime;base64,...`) | `source.type = "base64"` 与 `data` + `media_type` | `inlineData` 与 `data` + `mimeType`              |
| 文件路径（本地）  | 读取文件，base64 编码，作为数据 URI 发送             | 读取文件，base64 编码，作为 base64 源发送        | 读取文件，base64 编码，作为 inlineData 发送       |
| 支持的格式        | PNG、JPEG、GIF、WEBP                                | PNG、JPEG、GIF、WEBP                              | PNG、JPEG、GIF、WEBP、HEIC、HEIF                  |
| 最大图像大小      | 20MB                                                | 每个图像约 5MB（base64 编码）                    | 因方法而异                                         |
| 细节/保真度提示   | `detail`："auto"、"low"、"high"                     | 不支持（忽略）                                   | 不支持（忽略）                                     |

**便捷：文件路径支持。** SDK 应该接受本地文件路径作为便捷。当 `url` 看起来像本地文件路径（以 `/`、`./` 或 `~` 开头）时，适配器读取文件，从扩展名推断 MIME 类型，对内容进行 base64 编码，并使用提供商的内联数据格式发送它。这使得编码代理可以轻松发送屏幕截图和图表，而无需手动编码。

#### AudioData

```
RECORD AudioData:
    url         : String | None
    data        : Bytes | None
    media_type  : String | None     -- 例如 "audio/wav"、"audio/mp3"
```

#### DocumentData

```
RECORD DocumentData:
    url         : String | None
    data        : Bytes | None
    media_type  : String | None     -- 例如 "application/pdf"
    file_name   : String | None     -- 可选显示名称
```

#### ToolCallData

```
RECORD ToolCallData:
    id          : String            -- 此调用的唯一标识符（提供商分配）
    name        : String            -- 工具名称
    arguments   : Dict | String     -- 解析的 JSON 参数或原始参数字符串
    type        : String            -- "function"（默认）或 "custom"
```

`id` 字段由提供商分配，对于将工具结果链接回调用是必需的。对于不分配唯一 ID 的提供商（例如 Gemini），适配器必须生成合成唯一 ID（例如，`"call_" + random_uuid()`）并维护到函数名称的映射。

#### ToolResultData

```
RECORD ToolResultData:
    tool_call_id    : String            -- 此结果回答的 ToolCallData.id
    content         : String | Dict     -- 工具的输出（文本或结构化）
    is_error        : Boolean           -- 工具执行是否失败
    image_data      : Bytes | None      -- 可选图像结果
    image_media_type: String | None     -- 图像结果的 MIME 类型
```

当 `is_error` 为 true 时，模型理解工具失败并可以调整其方法。

#### ThinkingData

```
RECORD ThinkingData:
    text        : String            -- 思考/推理内容
    signature   : String | None     -- 用于往返的提供商特定签名
    redacted    : Boolean           -- 如果这是编辑的思考（不透明内容）
```

来自 Anthropic 扩展思考的思考块必须完全按接收保留，并包含在后续消息中。`signature` 字段实现了这一点。编辑的思考块包含无法读取但必须逐字传回的不透明数据。

**跨提供商可移植性：** 带有签名的思考块仅在使用同一提供商和模型继续时有效。切换提供商时，适配器应去除签名，并可选择将思考文本转换为用户可见的上下文消息。

### 3.6 Request

`complete()` 和 `stream()` 的单一输入类型：

```
RECORD Request:
    model             : String                      -- 必需；提供商的原生模型 ID
    messages          : List<Message>               -- 必需；对话
    provider          : String | None               -- 可选；如果省略则使用默认值
    tools             : List<ToolDefinition> | None -- 可选
    tool_choice       : ToolChoice | None           -- 可选；如果存在工具则默认为 AUTO
    response_format   : ResponseFormat | None       -- 可选；text、json 或 json_schema
    temperature       : Float | None
    top_p             : Float | None
    max_tokens        : Integer | None
    stop_sequences    : List<String> | None
    reasoning_effort  : String | None               -- "none"、"low"、"medium"、"high"
    metadata          : Dict<String, String> | None -- 任意键值对
    provider_options  : Dict | None                 -- 提供商特定参数的转义机制
```

#### 提供商选项（转义机制）

`provider_options` 字段传递统一接口未建模的特定于提供商的参数。每个适配器提取它理解的选项并忽略其余选项。

```
request = Request(
    model = "claude-opus-4-6",
    messages = [ ... ],
    provider_options = {
        "anthropic": {
            "thinking": { "type": "enabled", "budget_tokens": 10000 },
            "beta_features": ["interleaved-thinking-2025-05-14"]
        }
    }
)
```

使用 `provider_options` 的代码明确不可移植。库记录此权衡。

### 3.7 Response

```
RECORD Response:
    id              : String                -- 提供商分配的响应 ID
    model           : String                -- 实际使用的模型（可能与请求的不同）
    provider        : String                -- 哪个提供商完成了请求
    message         : Message               -- 助手的响应作为 Message
    finish_reason   : FinishReason          -- 生成停止的原因
    usage           : Usage                 -- 令牌计数
    raw             : Dict | None           -- 原始提供商响应 JSON（用于调试）
    warnings        : List<Warning>         -- 非致命问题（可选，可能为空）
    rate_limit      : RateLimitInfo | None  -- 来自标头的速率限制元数据（可选）
```

Response 上的便捷访问器：

```
response.text        -> String              -- 所有文本部分的连接文本
response.tool_calls  -> List<ToolCall>      -- 从消息中提取的工具调用
response.reasoning   -> String | None       -- 连接的推理/思考文本
```

### 3.8 FinishReason

双重表示，既保留可移植的语义又保留提供商特定的细节：

```
RECORD FinishReason:
    reason  : String        -- 统一：以下值之一
    raw     : String | None -- 提供商的原生完成原因字符串
```

统一原因值：

| 值            | 含义                             |
|--------------|----------------------------------|
| `stop`       | 自然生成结束（模型停止）         |
| `length`     | 输出达到 max_tokens 限制         |
| `tool_calls` | 模型想要调用一个或多个工具       |
| `content_filter` | 响应被安全/内容过滤器阻止    |
| `error`      | 生成期间发生错误                 |
| `other`      | 提供商特定原因，未映射到上面     |

提供商完成原因映射：

| 提供商    | 提供商值        | 统一值         |
|-----------|----------------|----------------|
| OpenAI    | stop           | stop           |
| OpenAI    | length         | length         |
| OpenAI    | tool_calls     | tool_calls     |
| OpenAI    | content_filter | content_filter |
| Anthropic | end_turn       | stop           |
| Anthropic | stop_sequence  | stop           |
| Anthropic | max_tokens     | length         |
| Anthropic | tool_use       | tool_calls     |
| Gemini    | STOP           | stop           |
| Gemini    | MAX_TOKENS     | length         |
| Gemini    | SAFETY         | content_filter |
| Gemini    | RECITATION     | content_filter |
| Gemini    | （有工具调用） | tool_calls     |

注意：Gemini 没有专用的"tool_calls"完成原因。适配器从响应中 `functionCall` 部分的存在推断它。

### 3.9 Usage

```
RECORD Usage:
    input_tokens        : Integer           -- 提示中的令牌
    output_tokens       : Integer           -- 模型生成的令牌
    total_tokens        : Integer           -- 输入 + 输出
    reasoning_tokens    : Integer | None    -- 用于思维链推理的令牌
    cache_read_tokens   : Integer | None    -- 从提示缓存提供的令牌
    cache_write_tokens  : Integer | None    -- 写入提示缓存的令牌
    raw                 : Dict | None       -- 原始提供商使用数据
```

Usage 对象必须支持加法以聚合跨多步操作：

```
usage_a + usage_b -> Usage
    -- 对整数字段求和。
    -- 对于可选字段：如果任一侧非 None，则求和（将 None 视为 0）。
    -- 如果可选字段的两侧都为 None，则结果为 None。
```

提供商使用字段映射：

| SDK 字段           | OpenAI 字段                                        | Anthropic 字段                   | Gemini 字段                          |
|---------------------|---------------------------------------------------|---------------------------------|--------------------------------------|
| input_tokens        | usage.prompt_tokens                                | usage.input_tokens              | usageMetadata.promptTokenCount       |
| output_tokens       | usage.completion_tokens                            | usage.output_tokens             | usageMetadata.candidatesTokenCount   |
| reasoning_tokens    | usage.completion_tokens_details.reasoning_tokens   | （见下文注释）                   | usageMetadata.thoughtsTokenCount     |
| cache_read_tokens   | usage.prompt_tokens_details.cached_tokens          | usage.cache_read_input_tokens   | usageMetadata.cachedContentTokenCount |
| cache_write_tokens  | （未提供）                                         | usage.cache_creation_input_tokens | （未提供）                          |

#### 推理令牌处理（关键）

推理令牌是模型在产生可见输出之前用于内部思维链的令牌。正确跟踪和表露推理令牌对于成本管理和调试至关重要，因为推理令牌作为输出令牌计费，但在响应文本中不可见。

**OpenAI 推理模型（GPT-5.2 系列等）：**
- **Responses API** (`/v1/responses`) 是推理模型所必需的。Chat Completions API 不会为这些模型返回推理令牌细分。Responses API 返回 `usage.output_tokens_details.reasoning_tokens`，它告诉您在推理与可见输出上花费了多少令牌。
- `reasoning_effort` 请求参数（"low"、"medium"、"high"）控制模型进行多少推理。这映射到 Responses API 请求主体中的 `reasoning.effort`。
- 推理内容在响应中不可见（OpenAI 不暴露 GPT-5.2 系列模型的思考文本）。适配器仍应在 Usage 中填充 `reasoning_tokens`，以便调用者可以跟踪成本。

**Anthropic 扩展思考（启用了思考的 Claude）：**
- 扩展思考通过 `thinking` 参数（通过 `provider_options`）启用，并需要特定的 beta 标头。
- Anthropic 将思考作为响应中的显式 `thinking` 内容块表面化。这些块包含实际的推理文本，并计入使用情况中的 `output_tokens`。
- 适配器应该通过求和思考块的令牌长度来填充 `reasoning_tokens`（Anthropic 不提供单独的推理令牌计数，但思考块文本可用于估算）。
- 思考块携带一个 `signature` 字段，必须在后续消息中逐字往返。

**Gemini 思考（Gemini 3 模型）：**
- Gemini 3 Flash 通过 `thinkingConfig` 参数支持"思考"。
- Gemini 在 `usageMetadata` 中报告 `thoughtsTokenCount`，直接映射到 `reasoning_tokens`。
- 思考内容可能会作为响应中的 `thought` 部分返回。

**为什么这很重要：** 在提供商之间切换时，推理令牌使用可能会有很大差异。在 OpenAI GPT-5.2 上使用 500 个推理令牌的查询可能会在 Claude 上使用 2000 个思考令牌。统一的 SDK 必须准确跟踪这一点，以便调用者做出明智的成本决策。尽管推理令牌使直接提供商切换不利（思考风格不同），但 SDK 仍应正确转换，以便高级工具可以进行比较。

### 3.10 ResponseFormat

```
RECORD ResponseFormat:
    type        : String            -- "text"、"json" 或 "json_schema"
    json_schema : Dict | None       -- 当 type 为 "json_schema" 时必需
    strict      : Boolean           -- 为 true 时，提供商严格强制执行架构（默认：false）
```

### 3.11 Warning

```
RECORD Warning:
    message : String                -- 非致命问题的人类可读描述
    code    : String | None         -- 机器可读的警告代码
```

### 3.12 RateLimitInfo

```
RECORD RateLimitInfo:
    requests_remaining  : Integer | None
    requests_limit      : Integer | None
    tokens_remaining    : Integer | None
    tokens_limit        : Integer | None
    reset_at            : Timestamp | None
```

从提供商响应标头（例如，`x-ratelimit-remaining-requests`）填充。此数据仅供参考；库不使用它进行主动节流。

### 3.13 StreamEvent

所有流式事件共享一个 `type` 判别器字段。库将特定于提供商的 SSE 格式规范化为这个统一事件模型。

```
RECORD StreamEvent:
    type              : StreamEventType | String

    -- 文本事件
    delta             : String | None           -- 增量文本
    text_id           : String | None           -- 标识此文本属于哪个段

    -- 推理事件
    reasoning_delta   : String | None           -- 增量推理/思考文本

    -- 工具调用事件
    tool_call         : ToolCall | None         -- 部分或完整的工具调用

    -- 完成事件
    finish_reason     : FinishReason | None
    usage             : Usage | None
    response          : Response | None         -- 完整的累积响应

    -- 错误事件
    error             : SDKError | None

    -- 透传
    raw               : Dict | None             -- 用于透传的原始提供商事件
```

### 3.14 StreamEventType

```
ENUM StreamEventType:
    STREAM_START        -- 流已开始。可能包括警告。
    TEXT_START           -- 新的文本段已开始。包括 text_id。
    TEXT_DELTA           -- 增量文本内容。包括 delta 和 text_id。
    TEXT_END             -- 文本段完成。包括 text_id。
    REASONING_START     -- 模型推理已开始。
    REASONING_DELTA     -- 增量推理内容。
    REASONING_END       -- 推理完成。
    TOOL_CALL_START     -- 工具调用已开始。包括工具名称和调用 ID。
    TOOL_CALL_DELTA     -- 增量工具调用参数（部分 JSON）。
    TOOL_CALL_END       -- 工具调用完全形成并准备好执行。
    FINISH              -- 生成完成。包括 finish_reason、usage、response。
    ERROR               -- 流式传输期间发生错误。
    PROVIDER_EVENT      -- 未映射到统一模型的原始提供商事件。
```

**开始/增量/结束模式。** 文本、推理和工具调用事件遵循一致的开始/增量/结束生命周期。此模式启用：

1. **多个并发段**——响应可以同时包含多个文本段或工具调用。ID 将增量与其段相关联。
2. **资源生命周期**——消费者知道段何时开始和结束，实现正确的缓冲区管理和 UI 更新。
3. **类型完成**——结束事件携带其段的最终累积值。

仅关心增量文本的消费者可以过滤 `TEXT_DELTA` 事件并忽略开始/结束事件。

---

## 4. 生成与流式传输

### 4.1 低级：Client.complete()

基本的阻塞调用。发送请求，阻塞直到模型完成，返回完整响应。

```
response = client.complete(Request(
    model = "claude-opus-4-6",
    messages = [Message.user("Explain photosynthesis in one paragraph")],
    max_tokens = 500,
    temperature = 0.7
))

response.text           -- "Photosynthesis is..."
response.finish_reason  -- FinishReason(reason="stop", raw="end_turn")
response.usage          -- Usage(input_tokens=12, output_tokens=85, ...)
```

**行为：**
- 路由到解析的提供商适配器。
- 阻塞直到模型产生完整响应。
- 返回 Response 对象。
- 在提供商错误时引发异常。
- 不自动重试。重试是第 4 层（高级 API）或应用程序代码的责任。

### 4.2 低级：Client.stream()

基本的流式调用。返回 StreamEvent 对象的异步迭代器。

```
event_stream = client.stream(Request(
    model = "claude-opus-4-6",
    messages = [Message.user("Write a short story")]
))

FOR EACH event IN event_stream:
    IF event.type == TEXT_DELTA:
        PRINT(event.delta)
    ELSE IF event.type == FINISH:
        PRINT("Done. Tokens: " + event.usage.total_tokens)
```

**行为：**
- 立即返回异步迭代器。
- 在从提供商到达时产生 StreamEvent 对象。
- 流以包含完整累积响应的 FINISH 事件终止。
- 必须被消费或显式关闭；在没有关闭的情况下放弃流可能会泄漏连接。
- 不自动重试。

### 4.3 高级：generate()

主要的阻塞生成函数。使用工具执行循环、多步编排、提示标准化和自动重试包装 `Client.complete()`。

```
FUNCTION generate(
    model             : String,
    prompt            : String | None,               -- 简单文本提示
    messages          : List<Message> | None,        -- 完整消息历史
    system            : String | None,               -- 系统消息
    tools             : List<Tool> | None,           -- 带有可选执行处理程序的工具
    tool_choice       : ToolChoice | None,           -- auto/none/required/named
    max_tool_rounds   : Integer = 1,                 -- 最大工具执行循环迭代
    stop_when         : StopCondition | None,        -- 工具循环的自定义停止条件
    response_format   : ResponseFormat | None,
    temperature       : Float | None,
    top_p             : Float | None,
    max_tokens        : Integer | None,
    stop_sequences    : List<String> | None,
    reasoning_effort  : String | None,
    provider          : String | None,
    provider_options  : Dict | None,
    max_retries       : Integer = 2,                 -- 瞬态错误的重试计数
    timeout           : Float | TimeoutConfig | None,
    abort_signal      : AbortSignal | None,          -- 取消信号
    client            : Client | None                -- 覆盖默认客户端
) -> GenerateResult
```

**提示标准化：** 提供 EITHER `prompt`（一个简单字符串，转换为单个用户消息）或 `messages`（完整对话），而不是两者都提供。同时使用两者是错误。`system` 参数始终是单独的，并作为系统消息前置。

**工具执行循环（在第 5 节中详细说明）：** 当提供带有执行处理程序的工具并且模型使用工具调用响应时，`generate()` 自动执行工具，将其结果附加到对话，并再次调用模型。此循环继续，直到模型响应无工具调用、达到 `max_tool_rounds` 或满足停止条件。

**`max_tool_rounds` 语义：** 该值表示执行工具调用并将结果反馈的最大次数。值为 1 表示：进行初始调用，如果模型返回工具调用则执行它们并进行再进行一次调用。值为 0 表示没有自动工具执行（工具返回给调用者）。LLM 调用的总数最多为 `max_tool_rounds + 1`。

#### GenerateResult

```
RECORD GenerateResult:
    text            : String                    -- 最后一步的文本
    reasoning       : String | None             -- 最后一步的推理
    tool_calls      : List<ToolCall>            -- 最后一步的工具调用
    tool_results    : List<ToolResult>          -- 最后一步的工具结果
    finish_reason   : FinishReason
    usage           : Usage                     -- 最后一步的使用情况
    total_usage     : Usage                     -- 所有步骤的聚合使用情况
    steps           : List<StepResult>          -- 每一步的详细结果
    response        : Response                  -- 最终的 Response 对象
    output          : Any | None                -- 解析的结构化输出（用于 generate_object）
```

#### StepResult

```
RECORD StepResult:
    text            : String
    reasoning       : String | None
    tool_calls      : List<ToolCall>
    tool_results    : List<ToolResult>
    finish_reason   : FinishReason
    usage           : Usage
    response        : Response
    warnings        : List<Warning>
```

### 4.4 高级：stream()

主要的流式生成函数。相当于 `generate()`，但增量地产生事件。

```
result = stream(
    model = "claude-opus-4-6",
    prompt = "Write a haiku about coding"
)

FOR EACH event IN result:
    IF event.type == TEXT_DELTA:
        PRINT(event.delta)

-- 迭代后，完整响应可用：
response = result.response()
```

接受与 `generate()` 相同的参数。当提供带有执行处理程序的工具并且模型进行工具调用时，流在工具执行时暂停，发出 `step_finish` 事件，然后恢复流式传输模型的下一个响应。

返回的 StreamResult 提供：
- 事件的异步迭代。
- `response()`——在流结束后返回累积的 Response。
- `text_stream`——仅产生增量文本的异步可迭代对象（便捷）。

#### StreamResult

```
RECORD StreamResult:
    ASYNC ITERATOR over StreamEvent
    FUNCTION response() -> Response         -- 累积响应（流结束后可用）
    PROPERTY text_stream -> AsyncIterator<String>  -- 仅产生增量文本
    PROPERTY partial_response -> Response | None   -- 任何点的当前累积状态
```

#### StreamAccumulator

一个将流式事件收集到完整 Response 中的实用程序：

```
accumulator = StreamAccumulator()

FOR EACH event IN stream:
    accumulator.process(event)

response = accumulator.response()   -- 相当于 complete() 返回的内容
```

这桥接了两种模式：任何与 Response 一起工作的代码都可以通过先累积来与流式传输一起使用。

### 4.5 高级：generate_object()

具有架构验证的结构化输出生成：

```
result = generate_object(
    model = "gpt-5.2",
    prompt = "Extract the person's name and age from: 'Alice is 30 years old'",
    schema = {
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    }
)

result.output   -- { "name": "Alice", "age": 30 }  （解析和验证）
result.text     -- 原始文本响应
```

**按提供商的实现策略：**

| 提供商    | 策略                                                          |
|-----------|---------------------------------------------------------------|
| OpenAI    | 原生 `response_format: { type: "json_schema", ... }` 与严格模式 |
| Gemini    | 原生 `responseMimeType: "application/json"` 与 `responseSchema` |
| Anthropic | 后备：将架构指令注入系统提示，解析输出。或者，使用基于工具的提取（定义一个工具，其输入架构与所需输出匹配，强制模型调用它）。 |

如果解析或验证失败，函数引发 `NoObjectGeneratedError`。

### 4.6 高级：stream_object()

具有部分对象更新的流式结构化输出：

```
result = stream_object(
    model = "gpt-5.2",
    prompt = "Generate a list of 5 recipes",
    schema = recipes_schema
)

FOR EACH partial IN result:
    -- partial 是一个部分解析的对象，随着令牌到达而增长
    PRINT("Recipes so far: " + LENGTH(partial.recipes))

final = result.object()  -- 完整的、验证的对象
```

使用增量 JSON 解析在令牌到达时产生部分对象。这实现了渐进式 UI 渲染。

### 4.7 取消和超时

#### 中止信号

`generate()` 和 `stream()` 都接受中止信号以进行协作取消：

```
controller = AbortController()

-- 在另一个线程/协程中：
controller.abort()

-- 如果取消，generate 调用引发 AbortError：
result = generate(model = "...", prompt = "...", abort_signal = controller.signal)
```

对于流式传输，取消关闭底层连接，流引发 AbortError。

#### 超时

超时可以指定为简单持续时间（总超时）或结构化配置：

```
RECORD TimeoutConfig:
    total       : Float | None      -- 整个多步操作的最大时间
    per_step    : Float | None      -- 每个单独 LLM 调用的最大时间
```

库在适配器级别区分三种超时范围：

```
RECORD AdapterTimeout:
    connect     : Float             -- 建立 HTTP 连接的时间（默认：10s）
    request     : Float             -- 整个请求/响应周期的时间（默认：120s）
    stream_read : Float             -- 连续流式事件之间的最大时间（默认：30s）
```

---

## 5. 工具调用

### 5.1 工具定义

```
RECORD Tool:
    name        : String                    -- 唯一标识符；[a-zA-Z][a-zA-Z0-9_]* 最多 64 个字符
    description : String                    -- 模型的人类可读描述
    parameters  : Dict                      -- 定义输入的 JSON 架构（根必须是 "object"）
    execute     : Function | None           -- 处理程序函数（如果存在，工具是"活动的"）
```

**工具名称约束：** 名称必须是有效标识符：字母数字字符和下划线，以字母开头。最多 64 个字符。这是所有提供商中最严格的公共子集。库在定义时验证名称。

**参数架构：** 参数必须定义为根处具有 `"type": "object"` 的 JSON 架构对象。这是所有提供商的通用要求。库将此架构传递给提供商，提供商使用它来约束参数生成。

**示例：**

```
weather_tool = Tool(
    name = "get_weather",
    description = "Get the current weather for a location",
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco, CA'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    },
    execute = get_weather_function
)
```

### 5.2 工具执行处理程序

`execute` 处理程序是一个可调用对象（同步或异步），接收解析的参数并返回结果：

```
FUNCTION get_weather(location: String, unit: String = "celsius") -> String:
    -- 调用天气 API...
    RETURN "72F and sunny in " + location
```

**处理程序契约：**
- **输入：** 解析的 JSON 参数作为关键字参数，或单个字典。
- **输出：** 字符串、字典、列表或任何 JSON 可序列化的值。
- **错误：** 引发异常以指示工具失败。库捕获它并向模型发送错误结果（带有 `is_error = true`），允许模型恢复。

**工具上下文注入：** 处理程序可以选择接收注入的上下文。库检查处理程序的签名并注入识别的关键字参数：

```
FUNCTION my_tool(
    query        : String,          -- 工具参数
    messages     : List<Message>,   -- 注入：当前对话
    abort_signal : AbortSignal,     -- 注入：取消信号
    tool_call_id : String           -- 注入：此调用的 ID
) -> String:
    ...
```

### 5.3 ToolChoice

控制模型是否以及如何使用工具：

```
RECORD ToolChoice:
    mode        : String            -- "auto"、"none"、"required"、"named"
    tool_name   : String | None     -- 当 mode 为 "named" 时必需
```

| 模式     | 行为                                    |
|----------|-----------------------------------------|
| auto     | 模型决定是否调用工具或以文本响应。      |
| none     | 模型不得调用任何工具，即使已定义。      |
| required | 模型必须至少调用一个工具。              |
| named    | 模型必须调用由 tool_name 标识的特定工具。|

提供商映射：

| SDK 模式  | OpenAI                                              | Anthropic                             | Gemini                                                   |
|-----------|-----------------------------------------------------|---------------------------------------|----------------------------------------------------------|
| auto      | `"auto"`                                            | `{"type": "auto"}`                    | `"AUTO"`                                                 |
| none      | `"none"`                                            | 从请求中省略工具                      | `"NONE"`                                                 |
| required  | `"required"`                                        | `{"type": "any"}`                     | `"ANY"`                                                  |
| named     | `{"type":"function","function":{"name":"..."}}`     | `{"type":"tool","name":"..."}`        | `{"mode":"ANY","allowedFunctionNames":["..."]}`         |

关于 Anthropic `none` 模式的说明：当存在工具时，Anthropic 不支持 `tool_choice: {"type": "none"}`。适配器必须从请求主体中完全省略工具数组。

如果提供商不支持特定模式，适配器引发 `UnsupportedToolChoiceError`。`supports_tool_choice(mode)` 方法允许预先检查功能。

### 5.4 ToolCall 和 ToolResult

从响应中提取并由执行处理程序产生：

```
RECORD ToolCall:
    id              : String            -- 唯一标识符（提供商分配）
    name            : String            -- 工具名称
    arguments       : Dict              -- 解析的 JSON 参数
    raw_arguments   : String | None     -- 解析前的原始参数字符串
```

```
RECORD ToolResult:
    tool_call_id    : String            -- 与 ToolCall.id 相关联
    content         : String | Dict | List  -- 工具的输出
    is_error        : Boolean           -- 如果工具执行失败则为 true
```

### 5.5 活动工具与被动工具

**活动工具**具有 `execute` 处理程序。当与 `generate()` 或 `stream()` 一起使用时，库自动执行它们并循环，直到模型产生最终文本响应。

**被动工具**没有 `execute` 处理程序。工具调用在响应中返回给调用者，调用者使用 `Client.complete()` 手动管理执行循环。

当以下情况时，被动工具很有用：
- 工具执行需要外部协调（人工批准、外部编排）。
- 调用代码有自己的循环和状态管理。
- 工具需要以特定顺序执行或在它们之间有副作用。

### 5.6 多步工具循环

当使用活动工具调用 `generate()` 时，执行以下循环：

```
FUNCTION tool_loop(request, tools, max_tool_rounds, stop_when):
    conversation = request.messages
    steps = []

    FOR round_num FROM 0 TO max_tool_rounds:
        response = client.complete(request_with(conversation))
        tool_calls = response.tool_calls

        -- 如果模型想要调用工具，则执行工具
        IF tool_calls AND response.finish_reason.reason == "tool_calls":
            tool_results = execute_all_tools(tools, tool_calls)  -- 并发
        ELSE:
            tool_results = []

        step = StepResult(response, tool_calls, tool_results, ...)
        steps.APPEND(step)

        -- 检查停止条件
        IF tool_calls is empty OR response.finish_reason.reason != "tool_calls":
            BREAK   -- 模型完成（自然完成）
        IF round_num >= max_tool_rounds:
            BREAK   -- 预算用尽
        IF stop_when is not None AND stop_when(steps) == true:
            BREAK   -- 满足自定义停止条件

        -- 使用工具结果继续对话
        conversation.APPEND(response.message)            -- 带有工具调用的助手消息
        FOR EACH result IN tool_results:
            conversation.APPEND(Message.tool_result(
                tool_call_id = result.tool_call_id,
                content = result.content,
                is_error = result.is_error
            ))

    RETURN GenerateResult from steps
```

### 5.7 并行工具执行

当模型在单个响应中返回多个工具调用时，它们在逻辑上是独立的（模型同时生成它们，而没有看到任何结果）。库必须正确处理此问题：

1. **并发执行所有工具调用。** 同时启动所有执行处理程序（使用异步任务、线程或等效的并发原语）。
2. **在继续之前等待所有结果。** 不要将部分结果发送回模型。继续请求必须包含前一个响应中每个工具调用的结果。
3. **在单个继续请求中发送所有结果。** 将所有工具结果捆绑到消息历史中并进行一次 LLM 调用，而不是每个结果一次调用。
4. **保留排序。** 工具结果应该以与相应工具调用相同的顺序出现，即使执行可能乱序完成。
5. **优雅地处理部分失败。** 如果某些工具执行成功而其他失败，则发送所有结果（失败的 `is_error = true`）。不要因为一个工具失败而中止整个批处理。

```
FUNCTION execute_all_tools(tools, tool_calls):
    -- 并发启动所有执行
    futures = []
    FOR EACH call IN tool_calls:
        tool = find_tool(tools, call.name)
        IF tool AND tool.execute:
            futures.APPEND(async_execute(tool.execute, call.arguments, call.id))
        ELSE:
            futures.APPEND(immediate_error(call.id, "Unknown tool: " + call.name))

    -- 等待所有完成
    results = AWAIT_ALL(futures)

    RETURN results   -- List<ToolResult>，每个 tool_call 一个，按顺序
```

这对于下游消费者（如编码代理）至关重要。当模型要求同时读取三个文件时，SDK 处理并发执行和结果批处理，以便编码代理的代理循环不必管理它。

### 5.8 工具调用验证和修复

在将参数传递给执行处理程序之前，库：

1. 解析 JSON 参数字符串。
2. 可选地根据工具的参数架构进行验证。
3. 如果验证失败并提供了 `repair_tool_call` 函数，则尝试修复（例如，要求模型修复参数）。
4. 如果修复失败或未配置，则向模型发送错误结果。

**未知工具调用：** 当模型调用未在定义中的工具时，库发送错误结果而不是引发异常。这使模型有机会纠正其行为。

### 5.9 使用工具进行流式传输

当使用活动工具进行流式传输时，流在工具调用形成时发出工具调用事件。在步骤之间（工具执行后，下一次模型调用前），发出 `step_finish` 事件。消费者看到跨越多个步骤的连续事件流。

### 5.10 跨提供商的工具结果处理

如何将工具结果转换为每个提供商的格式：

| SDK 格式                           | OpenAI                                    | Anthropic                                   | Gemini                                           |
|------------------------------------|-------------------------------------------|---------------------------------------------|--------------------------------------------------|
| 带有 ToolResultData 的 TOOL 角色消息 | 带有 `tool_call_id` 的单独 `tool` 消息    | `user` 消息中的 `tool_result` 内容块        | `user` 内容中的 `functionResponse` 部分          |

---

## 6. 错误处理与重试

### 6.1 错误分类

所有库错误都继承自单个基础：

```
RECORD SDKError:
    message : String                -- 人类可读的描述
    cause   : Exception | None      -- 根本异常（如果有）
```

错误层次结构：

```
SDKError
 +-- ProviderError                      -- 来自 LLM 提供商的错误
 |    +-- AuthenticationError           -- 401：无效的 API 密钥、过期的令牌
 |    +-- AccessDeniedError             -- 403：权限不足
 |    +-- NotFoundError                 -- 404：未找到模型、未找到端点
 |    +-- InvalidRequestError           -- 400：格式错误的请求、无效的参数
 |    +-- RateLimitError                -- 429：超过速率限制
 |    +-- ServerError                   -- 500-599：提供商内部错误
 |    +-- ContentFilterError            -- 响应被安全过滤器阻止
 |    +-- ContextLengthError            -- 输入 + 输出超过上下文窗口
 |    +-- QuotaExceededError            -- 计费/使用配额已用尽
 +-- RequestTimeoutError                -- 请求或流超时
 +-- AbortError                         -- 通过中止信号取消请求
 +-- NetworkError                       -- 网络级失败
 +-- StreamError                        -- 流消���期间的错误
 +-- InvalidToolCallError               -- 工具调用参数验证失败
 +-- NoObjectGeneratedError             -- 结构化输出解析/验证失败
 +-- ConfigurationError                 -- SDK 配置错误（缺少提供商等）
```

注意：错误类名称选择为避免与常见语言内置名称冲突（例如，`AccessDeniedError` 而不是 `PermissionError`，`NetworkError` 而不是 `ConnectionError`，`RequestTimeoutError` 而不是 `TimeoutError`）。

### 6.2 ProviderError 字段

```
RECORD ProviderError extends SDKError:
    provider    : String                -- 哪个提供商返回了错误
    status_code : Integer | None        -- HTTP 状态代码（如果适用）
    error_code  : String | None         -- 提供商特定的错误代码
    retryable   : Boolean               -- 此错误是否可安全重试
    retry_after : Float | None          -- 重试前等待的秒数
    raw         : Dict | None           -- 来自提供商的原始错误响应主体
```

### 6.3 可重试性分类

每个错误都带有 `retryable` 属性。

**不可重试的错误**（客户端错误——重试无济于事）：

| 错误                  | 状态代码 | 可重试 |
|------------------------|----------|--------|
| AuthenticationError    | 401      | false  |
| AccessDeniedError      | 403      | false  |
| NotFoundError          | 404      | false  |
| InvalidRequestError    | 400、422 | false  |
| ContextLengthError     | 413      | false  |
| QuotaExceededError     | （变化） | false  |
| ContentFilterError     | （变化） | false  |
| ConfigurationError     | （不适用）| false  |

**可重试的错误**（瞬态——重试可能成功）：

| 错误                  | 状态代码 | 可重试 |
|------------------------|----------|--------|
| RateLimitError         | 429      | true   |
| ServerError            | 500-504  | true   |
| RequestTimeoutError    | 408      | true   |
| NetworkError           | （不适用）| true  |
| StreamError            | （不适用）| true  |

**未知错误默认为可重试。** 这是一个经过深思熟虑的保守选择：瞬态网络问题和新提供商错误代码比来自意外代码的永久失败更常见。误重试比误中止更便宜。

### 6.4 HTTP 状态代码映射

适配器使用此表将 HTTP 状态代码映射到错误类型：

| 状态 | 错误类型            | 可重试 |
|------|--------------------|--------|
| 400  | InvalidRequestError | false  |
| 401  | AuthenticationError | false  |
| 403  | AccessDeniedError   | false  |
| 404  | NotFoundError       | false  |
| 408  | RequestTimeoutError | true   |
| 413  | ContextLengthError  | false  |
| 422  | InvalidRequestError | false  |
| 429  | RateLimitError      | true   |
| 500  | ServerError         | true   |
| 502  | ServerError         | true   |
| 503  | ServerError         | true   |
| 504  | ServerError         | true   |

对于 Gemini（可能使用 gRPC 状态代码）：

| gRPC 代码           | 错误类型            |
|---------------------|--------------------|
| NOT_FOUND           | NotFoundError      |
| INVALID_ARGUMENT    | InvalidRequestError |
| UNAUTHENTICATED     | AuthenticationError |
| PERMISSION_DENIED   | AccessDeniedError  |
| RESOURCE_EXHAUSTED  | RateLimitError     |
| UNAVAILABLE         | ServerError        |
| DEADLINE_EXCEEDED   | RequestTimeoutError |
| INTERNAL            | ServerError        |

### 6.5 错误消息分类

对于仅状态代码不足的模糊情况，适配器检查错误消息主体以获取分类信号：

- 包含"not found"或"does not exist"的消息 -> NotFoundError
- 包含"unauthorized"或"invalid key"的消息 -> AuthenticationError
- 包含"context length"或"too many tokens"的消息 -> ContextLengthError
- 包含"content filter"或"safety"的消息 -> ContentFilterError

### 6.6 重试策略

```
RECORD RetryPolicy:
    max_retries         : Integer = 2       -- 总重试尝试次数（不包括初始）
    base_delay          : Float = 1.0       -- 初始延迟（秒）
    max_delay           : Float = 60.0      -- 重试之间的最大延迟
    backoff_multiplier  : Float = 2.0       -- 指数退避因子
    jitter              : Boolean = true    -- 添加随机抖动以防止惊群
    on_retry            : Callback | None   -- 每次重试前调用 (error, attempt, delay)
```

#### 带有抖动的指数退避

尝试 `n`（从 0 开始索引）的延迟计算为：

```
delay = MIN(base_delay * (backoff_multiplier ^ n), max_delay)
IF jitter:
    delay = delay * RANDOM(0.5, 1.5)   -- +/- 50% 抖动
```

使用默认值的示例延迟（base=1.0、multiplier=2.0、max=60.0）：

| 尝试 | 基本延迟 | 带有抖动（大约范围）  |
|------|----------|---------------------|
| 0    | 1.0s     | 0.5s -- 1.5s        |
| 1    | 2.0s     | 1.0s -- 3.0s        |
| 2    | 4.0s     | 2.0s -- 6.0s        |
| 3    | 8.0s     | 4.0s -- 12.0s       |
| 4    | 16.0s    | 8.0s -- 24.0s       |

#### Retry-After 标头

当提供商返回 `Retry-After` 标头（在 429 响应中很常见）时：

- 如果 `Retry-After` 小于 `max_delay`，使用提供商的延迟而不是计算出的退避。
- 如果 `Retry-After` 超过 `max_delay`，则不要重试。立即引发错误，并在异常上设置 `retry_after`。这可以防止静默等待几分钟以清除速率限制。

#### 什么被重试

重试应用于单个 LLM 调用，而不是整个多步操作：

- 带有工具的 `generate()`：每一步的 LLM 调用独立重试。第 3 步的重试不会重新执行第 1 步和第 2 步。
- `stream()`：仅重试初始连接。一旦流式传输开始并传递了部分数据，库就不会重试。相反，流发出错误事件。
- `generate_object()`：LLM 调用被重试。架构验证失败不会被重试（它们表示模型行为问题，而不是瞬态错误）。

#### 在适配器级别重试

提供商适配器默认不重试。重试逻辑存在于第 2 层（提供商实用程序）中，并由第 4 层的高级函数应用。低级 `Client.complete()` 和 `Client.stream()` 永远不会自动重试。使用低级 API 的应用程序可以使用独立的 `retry()` 实用程序组合重试行为：

```
response = retry(
    FUNCTION: client.complete(request),
    policy = RetryPolicy(max_retries = 3)
)
```

#### 禁用重试

设置 `max_retries = 0` 以禁用高级函数中的自动重试。

### 6.7 速率限制处理

当提供商返回 HTTP 429 时，库引发 RateLimitError，其中从响应标头提取 `retry_after`，并且 `retryable = true`。启用自动重试后，速率限制在重试预算内透明处理。

对于需要主动速率限制的应用程序（保持在限制以下而不是达到限制），请使用中间件：

```
FUNCTION rate_limit_middleware(request, next):
    token_bucket.acquire()   -- 阻塞直到预算可用
    RETURN next(request)
```

---

## 7. 提供商适配器契约

本节为实施提供商适配器提供了详细指导。它旨在作为添加新提供商的任何人的参考。

### 7.1 接口摘要

每个适配器必须实现：

```
INTERFACE ProviderAdapter:
    PROPERTY name : String

    FUNCTION complete(request: Request) -> Response
    FUNCTION stream(request: Request) -> AsyncIterator<StreamEvent>
```

推荐的可选方法：

```
    FUNCTION close() -> Void
    FUNCTION initialize() -> Void
    FUNCTION supports_tool_choice(mode: String) -> Boolean
```

### 7.2 请求转换

适配器必须将统一的 `Request` 转换为提供商的原生 API 格式。一般步骤是：

1. **提取系统消息。** 对于 Anthropic：从消息列表中提取，作为 `system` 参数传递。对于 Gemini：提取并作为 `systemInstruction` 传递。对于 OpenAI (Responses API)：提取并作为 `instructions` 参数传递。

2. **转换消息。** 将每个 Message 及其 ContentParts 转换为提供商的格式。

3. **转换工具。** 将 Tool 定义转换为提供商的工具格式。

4. **转换工具选择。** 将统一的 ToolChoice 映射到提供商的格式。

5. **设置生成参数。** 映射 temperature、top_p、max_tokens、stop_sequences 等。

6. **应用响应格式。** 将 ResponseFormat 转换为提供商的结构化输出机制。

7. **应用提供商选项。** 将来自 `request.provider_options[provider_name]` 的任何特定于提供商的选项合并到请求主体中。

### 7.3 消息转换详细信息

#### OpenAI 消息转换 (Responses API)

Responses API 使用的消息格式与 Chat Completions 不同。消息在 `input` 数组中传递，而不是在 `messages` 数组中：

```
统一角色    -> Responses API 处理
SYSTEM          -> 提取到 `instructions` 参数
USER            -> input 项：{ "type": "message", "role": "user", "content": [...] }
ASSISTANT       -> input 项：{ "type": "message", "role": "assistant", "content": [...] }
TOOL            -> input 项：{ "type": "function_call_output", "call_id": "...", "output": "..." }
DEVELOPER       -> 提取到 `instructions` 参数（或 `developer` 角色 input 项）

ContentPart 转换：
  TEXT          -> { "type": "input_text", "text": "..." } (user) 或 { "type": "output_text", "text": "..." } (assistant)
  IMAGE (url)  -> { "type": "input_image", "image_url": "..." }
  IMAGE (data) -> { "type": "input_image", "image_url": "data:<mime>;base64,<data>" }
  TOOL_CALL    -> input 项：{ "type": "function_call", "id": "...", "name": "...", "arguments": "..." }
  TOOL_RESULT  -> input 项：{ "type": "function_call_output", "call_id": "...", "output": "..." }
```

特殊行为：
- 系统消息被提取到 `instructions` 参数，不包含在 `input` 数组中。
- `reasoning.effort` 参数控制 o 系列模型的推理（"low"、"medium"、"high"）。
- 工具调用和结果是顶级 input 项，不嵌套在消息中。
- 对于第三方 OpenAI 兼容端点，请改用 Chat Completions 格式（参见第 7.10 节）。

#### Anthropic 消息转换

```
统一角色    -> Anthropic 处理
SYSTEM          -> 提取到 `system` 参数（不在 messages 数组中）
DEVELOPER       -> 与系统参数合并
USER            -> "user" 角色
ASSISTANT       -> "assistant" 角色
TOOL            -> 带有 tool_result 内容块的 "user" 角色

ContentPart 转换：
  TEXT          -> { "type": "text", "text": "..." }
  IMAGE (url)  -> { "type": "image", "source": { "type": "url", "url": "..." } }
  IMAGE (data) -> { "type": "image", "source": { "type": "base64", "media_type": "...", "data": "..." } }
  TOOL_CALL    -> { "type": "tool_use", "id": "...", "name": "...", "input": { ... } }
  TOOL_RESULT  -> { "type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": ... }
  THINKING     -> { "type": "thinking", "thinking": "...", "signature": "..." }
  REDACTED_THINKING -> { "type": "redacted_thinking", "data": "..." }
```

特殊行为：
- **严格交替：** Anthropic 需要交替的用户/助手消息。适配器必须通过组合其内容数组来合并连续的同角色消息。
- **用户消息中的工具结果：** Anthropic 需要工具结果出现在用户角色消息中，而不是单独的"tool"角色。
- **思考块往返：** 以前响应中的思考和 redacted_thinking 块必须完全按接收保留，并包含在后续助手消息中。
- **需要 max_tokens：** Anthropic 始终需要 `max_tokens`。如果未指定，默认为 4096。

#### Gemini 消息转换

```
统一角色    -> Gemini 处理
SYSTEM          -> 提取到 `systemInstruction` 字段
DEVELOPER       -- 与 systemInstruction 合并
USER            -> "user" 角色
ASSISTANT       -> "model" 角色
TOOL            -> 带有 functionResponse 部分的 "user" 角色

ContentPart 转换：
  TEXT          -> { "text": "..." }
  IMAGE (url)  -> { "fileData": { "mimeType": "...", "fileUri": "..." } }
  IMAGE (data) -> { "inlineData": { "mimeType": "...", "data": "<base64>" } }
  TOOL_CALL    -> { "functionCall": { "name": "...", "args": { ... } } }
  TOOL_RESULT  -> { "functionResponse": { "name": "<function_name>", "response": { ... } } }
```

特殊行为：
- **没有开发者角色：** 与系统相同处理。
- **工具调用 ID：** Gemini 不为函数调用分配唯一 ID。适配器必须生成合成唯一 ID（例如，`"call_" + random_uuid()`），并在发送工具结果时维护从合成 ID 到函数名称的映射。
- **函数响应格式：** Gemini 的 `functionResponse` 使用函数*名称*（而不是调用 ID），并期望响应为字典（如果需要，将字符串包装在 `{"result": "..."}` 中）。
- **流式格式：** Gemini 使用 JSON 块（可选通过 SSE 与 `?alt=sse`），而不是标准 SSE 端点。

### 7.4 工具定义转换

| SDK 格式              | OpenAI                                             | Anthropic                                   | Gemini                                             |
|-----------------------|----------------------------------------------------|---------------------------------------------|----------------------------------------------------|
| Tool.name             | tools[].function.name                              | tools[].name                                | tools[].functionDeclarations[].name                |
| Tool.description      | tools[].function.description                       | tools[].description                         | tools[].functionDeclarations[].description         |
| Tool.parameters       | tools[].function.parameters                        | tools[].input_schema                        | tools[].functionDeclarations[].parameters          |
| 包装器结构            | `{"type":"function","function":{...}}`             | `{"name":...,"description":...,"input_schema":...}` | `{"functionDeclarations":[{...}]}`         |

### 7.5 响应转换

适配器必须将提供商的响应解析为统一的 Response 格式：

1. **提取内容部分。** 将提供商的内容/部分数组解析为带有适当 `ContentKind` 标记的 `List<ContentPart>`。
2. **映射完成原因。** 将提供商的完成/停止原因转换为统一的 `FinishReason`（参见第 3.8 节中的映射表）。
3. **提取使用情况。** 将提供商的令牌计数字段映射到 `Usage`（参见第 3.9 节中的映射表）。
4. **保留原始响应。** 将完整的提供商响应存储在 `Response.raw` 中以进行调试。
5. **提取速率限制信息。** 如果存在，将 `x-ratelimit-*` 标头解析为 `RateLimitInfo`。

### 7.6 错误转换

适配器必须将 HTTP 错误转换为错误层次结构：

1. 解析响应主体以获取错误详细信息（消息、错误代码）。
2. 如果存在，提取 `Retry-After` 标头。
3. 使用第 6.4 节中的表将 HTTP 状态代码映射到适当的错误类型。
4. 对于模糊情况，应用基于消息的分类（第 6.5 节）。
5. 在 `raw` 字段中保留原始错误响应。

```
FUNCTION raise_error(http_response):
    body = parse_json(http_response.body)
    message = body.error.message OR http_response.text
    error_code = body.error.code OR body.error.type

    retry_after = None
    IF http_response.headers["retry-after"] EXISTS:
        retry_after = parse_float(http_response.headers["retry-after"])

    RAISE error_from_status_code(
        status_code = http_response.status,
        message = message,
        provider = self.name,
        error_code = error_code,
        raw = body,
        retry_after = retry_after
    )
```

### 7.7 流式转换

适配器将特定于提供商的流式格式转换为统一的 StreamEvent 模型。

#### SSE 解析

大多数提供商使用服务器发送事件 (SSE)。适当的 SSE 解析器必须处理：

- `event:` 行（事件类型）
- `data:` 行（负载，可能跨越多行）
- `retry:` 行（重新连接间隔）
- 注释行（以 `:` 开头）
- 空行（事件边界）

解析器产生 `(event_type, data)` 元组。许多提供商在 JSON 负载以及 SSE 事件字段中包含事件类型；为了可靠性，首选 JSON 负载字段。

#### OpenAI 流式传输 (Responses API)

Responses API 使用的流式格式与 Chat Completions 不同：

```
提供商格式 (Responses API)：
    event: response.created        -- 响应对象已创建
    event: response.in_progress    -- 生成已开始
    event: response.output_text.delta  -- 增量文本
    event: response.function_call_arguments.delta  -- 增量工具调用参数
    event: response.output_item.done   -- 输出项完成
    event: response.completed      -- 生成完成，包括带有 reasoning_tokens 的使用情况

转换：
    output_text.delta              -> TEXT_DELTA 事件（在第一个时发出 TEXT_START）
    function_call_arguments.delta  -> TOOL_CALL_DELTA 事件
    output_item.done (text)        -> TEXT_END 事件
    output_item.done (function)    -> TOOL_CALL_END 事件
    response.completed             -> 带有使用情况的 FINISH 事件（包括 reasoning_tokens）
```

Responses API 流式格式在最终的 `response.completed` 事件中提供推理令牌计数，这就是推理模型需要它的原因。

对于 OpenAI 兼容适配器 (Chat Completions)，流式格式是：

```
提供商格式 (Chat Completions，用于第三方端点)：
    data: {"choices": [{"delta": {"content": "text"}, "finish_reason": null}]}
    data: {"choices": [{"delta": {"tool_calls": [{"index": 0, ...}]}}]}
    data: {"usage": {...}}
    data: [DONE]
```

#### Anthropic 流式传输

```
提供商格式 (SSE 事件)：
    event: message_start       -- 包含消息元数据和输入令牌计数
    event: content_block_start -- 新内容块（text、tool_use、thinking）
    event: content_block_delta -- 块内的增量内容
    event: content_block_stop  -- 块完成
    event: message_delta       -- 完成原因和输出使用情况
    event: message_stop        -- 流完成

转换：
    content_block_start (type=text)     -> TEXT_START
    content_block_delta (type=text)     -> TEXT_DELTA
    content_block_stop  (type=text)     -> TEXT_END
    content_block_start (type=tool_use) -> TOOL_CALL_START
    content_block_delta (type=tool_use) -> TOOL_CALL_DELTA
    content_block_stop  (type=tool_use) -> TOOL_CALL_END
    content_block_start (type=thinking) -> REASONING_START
    content_block_delta (type=thinking) -> REASONING_DELTA
    content_block_stop  (type=thinking) -> REASONING_END
    message_stop                        -> 带有累积响应的 FINISH
```

#### Gemini 流式传输

Gemini 使用 SSE（带有 `?alt=sse` 查询参数）或换行符分隔的 JSON 块。

```
提供商格式 (SSE)：
    data: {"candidates": [{"content": {"parts": [{"text": "..."}]}}], "usageMetadata": {...}}

转换：
    parts[].text 存在               -> TEXT_DELTA（在第一个时发出 TEXT_START）
    parts[].functionCall 存在       -> TOOL_CALL_START + TOOL_CALL_END（一个块中的完整调用）
    candidate.finishReason 存在     -> TEXT_END
    最终块                         -> 带有累积响应的 FINISH
```

注意：Gemini 通常在单个块中将函数调用作为完整对象传递，而不是增量地。为每个函数调用发出 TOOL_CALL_START 和 TOOL_CALL_END。

### 7.8 提供商怪癖参考

适配器必须处理的特定于提供商的行为摘要：

| 关注点                      | OpenAI                           | Anthropic                              | Gemini                              |
|------------------------------|----------------------------------|----------------------------------------|-------------------------------------|
| **原生 API**               | **Responses API** (`/v1/responses`) | **Messages API** (`/v1/messages`)   | **Gemini API** (`/v1beta/...generateContent`) |
| 系统消息处理      | `instructions` 参数         | 提取到 `system` 参数        | 提取到 `systemInstruction`    |
| 开发者角色               | `instructions` 或 `developer` 角色 | 与系统合并                   | 与系统合并                  |
| 消息交替          | 无严格要求            | 严格的用户/助手交替      | 无严格要求               |
| 推理令牌             | 通过 `output_tokens_details`；需要 Responses API | 通过思考块（文本可见） | 通过 `thoughtsTokenCount`          |
| 工具调用 ID                | 提供商分配的唯一 ID     | 提供商分配的唯一 ID           | 无唯一 ID（使用函数名称）   |
| 工具结果格式           | 单独的 `tool` 角色消息    | 用户消息中的 `tool_result` 块  | `functionResponse` 在用户内容  |
| 工具选择 "none"           | `"none"`                         | 完全从请求中省略工具       | `"NONE"`                            |
| max_tokens                   | 可选                         | 必需（默认为 4096）             | 可选（作为 `maxOutputTokens`）     |
| 思考块              | 未暴露（o 系列内部）  | `thinking` / `redacted_thinking` 块| `thought` 部分（2.5 模型）       |
| 结构化输出            | 原生 json_schema 模式          | 提示工程或工具提取  | 原生 responseSchema               |
| 流式协议           | 带有 `data:` 行的 SSE           | 带有事件类型 + 数据行的 SSE       | SSE（带有 `?alt=sse`）或 JSON       |
| 流式终止           | `data: [DONE]`                   | `message_stop` 事件                   | 最终块（无显式信号）    |
| 工具的完成原因      | `tool_calls`                     | `tool_use`                             | 无专用原因（从部分推断）|
| 图像输入                  | `image_url` 中的数据 URI          | 带有 `media_type` 的 `base64` 源      | 带有 `mimeType` 的 `inlineData`        |
| 提示缓存               | 自动（免费，50% 折扣）   | 需要显式 `cache_control` 块（90% 折扣） | 自动（免费前缀缓存）   |
| Beta/功能标头         | 不适用（功能在请求主体中）   | `anthropic-beta` 标头（逗号分隔） | 不适用（功能在请求主体中）   |
| 身份验证               | Authorization 中的不记名令牌    | `x-api-key` 标头                     | `key` 查询参数               |
| API 版本控制               | 通过 URL 路径 (/v1/)              | `anthropic-version` 标头             | 通过 URL 路径 (/v1beta/)             |

### 7.9 添加新提供商

要添加对新提供商的支持：

1. **实现 ProviderAdapter 接口。** 创建一个带有 `name`、`complete()` 和 `stream()` 的类。
2. **编写请求转换。** 按照 7.3 节中的模式，将统一 Request 映射到提供商的 API 格式。
3. **编写响应转换。** 按照 7.5 节，将提供商的响应映射到统一 Response。
4. **编写错误转换。** 按照 7.6 节，将 HTTP 错误映射到错误层次结构。
5. **编写流式转换。** 按照 7.7 节，将提供商的流式格式映射到 StreamEvent 对象。
6. **处理提供商怪癖。** 记录任何特定于提供商的行为（如 Anthropic 的严格交替或 Gemini 的缺少工具调用 ID）并在适配器中处理它们。
7. **注册适配器。** 使用适当的环境变量检查将其添加到 `Client.from_env()`，或允许用户以编程方式注册它。

### 7.10 OpenAI 兼容端点

许多第三方服务（vLLM、Ollama、Together AI、Groq 等）暴露 OpenAI 兼容的 Chat Completions API。对于这些服务，提供单独的 `OpenAICompatibleAdapter`，使用 Chat Completions 端点 (`/v1/chat/completions`) 而不是 Responses API：

```
adapter = OpenAICompatibleAdapter(
    api_key = "...",
    base_url = "https://my-vllm-instance.example.com/v1"
)
```

此适配器与主要的 OpenAI 适配器（使用 Responses API）不同，因为第三方服务通常仅实现 Chat Completions 协议。兼容适配器不支持推理令牌、内置工具或其他 Responses API 功能。

---

## 附录 A：对话示例

### A.1 简单文本对话

```
messages = [
    Message(role = SYSTEM, content = [ContentPart(kind = TEXT, text = "You are a helpful assistant.")]),
    Message(role = USER,   content = [ContentPart(kind = TEXT, text = "What is 2 + 2?")])
]
```

### A.2 多模态对话

```
messages = [
    Message(role = USER, content = [
        ContentPart(kind = TEXT, text = "What do you see in this image?"),
        ContentPart(kind = IMAGE, image = ImageData(url = "https://example.com/photo.jpg"))
    ])
]
```

### A.3 工具使用对话

```
messages = [
    Message(role = USER, content = [
        ContentPart(kind = TEXT, text = "What is the weather in San Francisco?")
    ]),
    Message(role = ASSISTANT, content = [
        ContentPart(kind = TOOL_CALL, tool_call = ToolCallData(
            id = "call_123",
            name = "get_weather",
            arguments = { "city": "San Francisco" }
        ))
    ]),
    Message(role = TOOL, content = [
        ContentPart(kind = TOOL_RESULT, tool_result = ToolResultData(
            tool_call_id = "call_123",
            content = "72F, sunny",
            is_error = false
        ))
    ], tool_call_id = "call_123"),
    Message(role = ASSISTANT, content = [
        ContentPart(kind = TEXT, text = "The weather in San Francisco is 72F and sunny.")
    ])
]
```

### A.4 思考块 (Anthropic 扩展思考)

```
messages = [
    Message(role = USER, content = [
        ContentPart(kind = TEXT, text = "Solve this complex math problem...")
    ]),
    Message(role = ASSISTANT, content = [
        ContentPart(kind = THINKING, thinking = ThinkingData(
            text = "Let me work through this step by step...",
            signature = "sig_abc123"
        )),
        ContentPart(kind = TEXT, text = "The answer is 42.")
    ])
]
```

当继续包含思考块的对话时，思考内容部分必须包含在消息历史中，以便提供商可以验证其完整性。

---

## 附录 B：高级 API 使用示例

### B.1 简单生成

```
result = generate(model = "claude-opus-4-6", prompt = "Explain quantum computing")
PRINT(result.text)
PRINT(result.usage.total_tokens)
```

### B.2 带有工具的生成

```
result = generate(
    model = "claude-opus-4-6",
    system = "You are a helpful assistant with access to weather data.",
    prompt = "What is the weather in San Francisco?",
    tools = [weather_tool],
    max_tool_rounds = 5
)

PRINT(result.text)                              -- 所有工具轮次后的最终文本
PRINT(LENGTH(result.steps))                     -- 采取的步数
PRINT(result.total_usage.total_tokens)          -- 聚合令牌计数
```

### B.3 流式传输

```
result = stream(model = "claude-opus-4-6", prompt = "Write a poem")

FOR EACH event IN result:
    IF event.type == TEXT_DELTA:
        PRINT(event.delta)

response = result.response()
PRINT(response.usage)
```

### B.4 结构化输出

```
result = generate_object(
    model = "gpt-5.2",
    prompt = "Extract the person's name and age from: 'Alice is 30 years old'",
    schema = {
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"]
    }
)

PRINT(result.output)    -- { "name": "Alice", "age": 30 }
```

### B.5 提供商回退模式

```
TRY:
    result = generate(model = "claude-opus-4-6", prompt = "...")
CATCH ProviderError:
    result = generate(model = "gpt-5.2", provider = "openai", prompt = "...")
```

### B.6 日志记录中间件

```
FUNCTION logging_middleware(request, next):
    start_time = NOW()
    LOG_INFO("LLM request: provider=" + request.provider + " model=" + request.model)
    response = next(request)
    elapsed = NOW() - start_time
    LOG_INFO("LLM response: tokens=" + response.usage.total_tokens + " latency=" + elapsed)
    RETURN response

client = Client(
    providers = { "anthropic": AnthropicAdapter(...) },
    middleware = [logging_middleware]
)
```

---

## 附录 C：设计决策基本原理

本附录总结了关键的设计决策及其背后的基本原理。提供这些是为了让实现者理解"为什么"，并在他们的语言或上下文需要不同选择时做出明智的权衡。

**为什么是单一 Request 类型而不是每个方法的参数列表？** 单个 Request 对象比许多关键字参数更容易构造、传递、修改和序列化。它使中间件能够统一地检查和修改请求。高级函数如 `generate(model=..., prompt=...)` 提供符合人体工程学的简写。

**为什么运送模型目录（如果模型字符串可以按原样工作）？** 模型字符串适用于知道存在哪些模型的开发人员。但是，在此 SDK 之上构建的 AI 编码代理经常从陈旧的训练数据中幻觉模型标识符。目录为它们提供了有效模型 ID 和功能的可靠、最新来源。未知的模型字符串仍然传递——目录是建议性的，而不是限制性的。

**为什么在 Request 上显式提供商而不是基于模型的路由？** 几个提供商提供具有重叠名称的模型。显式路由避免了歧义。对于常见情况，`default_provider` 删除了样板文件。

**为什么单独的 generate() 和 stream()？** 返回类型根本不同：GenerateResult 与 StreamResult。布尔标志失去类型安全性。

**为什么是开始/增量/结束事件而不是平面增量？** 当响应包含多个文本段或交错的工具调用时，平面增量会丢失结构信息。该模式增加了最小的开销，但实现了对复杂响应的正确处理。

**为什么是 max_tool_rounds 而不是无限循环？** 无界循环有无限循环的风险。默认值 1 是安全的。较高的值是显式选择加入。

**为什么工具参数使用 JSON 架构而不是语言原生类型？** JSON 架构是所有提供商的通用参数描述格式。语言原生助手可以生成 JSON 架构，但 JSON 架构是规范格式。

**为什么向模型发送错误结果而不是引发异常？** 工具失败时引发会中止整个生成。发送错误结果使模型有机会重试、使用不同的工具或解释失败。

**为什么默认重试未知错误？** 来自意外代码的瞬态失败比永久失败更常见。误重试比误中止更便宜。

**为什么默认不重试超时请求？** 超时表示操作本质上很慢，而不是它瞬态失败。应用程序可以选择加入超时重试。

**为什么使用每个提供商的原生 API 而不是仅针对 Chat Completions 到处都是？** Chat Completions API 是一个 OpenAI 特定协议，其他提供商部分模仿它作为便捷 shim。将其用作通用传输会失去关键功能：OpenAI 自己的 Responses API 暴露了 Chat Completions 隐藏的推理令牌；Anthropic 的 Messages API 支持思考块、提示缓存和 beta 标头；Gemini 的原生 API 支持 grounding 和代码执行。统一的 SDK 的价值恰恰在于抽象这些不同的原生 API，以便调用者不必这样做。使用兼容层会违背目的。

**为什么在 SDK 中处理并行工具执行而不是将其留给调用者？** 当模型返回 5 个并行工具调用时，正确的行为是并发执行所有 5 个，等待所有完成，并在一个继续中发送所有 5 个结果。这很难正确实现（错误处理、排序、超时管理），并且对每个消费者都是相同的。在 SDK 中做一次意味着编码代理和其他下游工具可以免费获得它。

---

## 8. 完成定义

本节定义了如何验证此规范的实现是完整和正确的。在开发过程中将其作为清单使用。当每个项目都被勾选时，实现被认为完成。

### 8.1 核心基础设施

- [ ] `Client` 可以从环境变量构造 (`Client.from_env()`)
- [ ] `Client` 可以使用显式适配器实例以编程方式构造
- [ ] 提供商路由工作：请求根据 `provider` 字段分派到正确的适配器
- [ ] 当从请求中省略 `provider` 时使用默认提供商
- [ ] 当未配置提供商且未设置默认值时引发 `ConfigurationError`
- [ ] 中间件链按正确顺序执行（请求：注册顺序，响应：相反顺序）
- [ ] 模块级默认客户端工作（`set_default_client()` 和隐式延迟初始化）
- [ ] 模型目录填充了当前模型，并且 `get_model_info()` / `list_models()` 返回正确数据

### 8.2 提供商适配器

对于每个提供商 (OpenAI、Anthropic、Gemini)，验证：

- [ ] 适配器使用提供商的**原生 API** (OpenAI: Responses API、Anthropic: Messages API、Gemini: Gemini API)——而不是兼容 shim
- [ ] 身份验证工作（来自 env var 或显式配置的 API 密钥）
- [ ] `complete()` 发送请求并返回正确填充的 `Response`
- [ ] `stream()` 返回正确类型的 `StreamEvent` 对象的异步迭代器
- [ ] 系统消息根据提供商约定提取/处理
- [ ] 所有 5 个角色 (SYSTEM、USER、ASSISTANT、TOOL、DEVELOPER) 都正确转换
- [ ] `provider_options` 转义机制传递特定于提供商的参数
- [ ] 支持 Beta 标头（尤其是 Anthropic 的 `anthropic-beta` 标头）
- [ ] HTTP 错误转换为正确的错误层次结构类型
- [ ] `Retry-After` 标头被解析并在错误对象上设置

### 8.3 消息和内容模型

- [ ] 仅文本内容的消息在所有提供商上工作
- [ ] **图像输入工作**：作为 URL、base64 数据和本地文件路径发送的图像根据每个提供商正确转换
- [ ] 音频和文档内容部分被处理（或在提供商不支持时优雅地拒绝）
- [ ] 工具调用内容部分正确往返（带有工具调用的助手消息 -> 工具结果消息 -> 下一个助手消息）
- [ ] 思考块 (Anthropic) 被保留，并带有完整签名的往返
- [ ] 编辑的思考块逐字传递
- [ ] 多模态消息（同一消息中的文本 + 图像）工作

### 8.4 生成

- [ ] `generate()` 使用简单的文本 `prompt` 工作
- [ ] `generate()` 使用完整的 `messages` 列表工作
- [ ] `generate()` 同时提供 `prompt` 和 `messages` 时拒绝
- [ ] `stream()` 产生连接到完整响应文本的 `TEXT_DELTA` 事件
- [ ] `stream()` 产生带有正确元数据的 `STREAM_START` 和 `FINISH` 事件
- [ ] 流式传输遵循文本段的开始/增量/结束模式
- [ ] `generate_object()` 返回解析的、验证的结构化输出
- [ ] `generate_object()` 在解析/验证失败时引发 `NoObjectGeneratedError`
- [ ] 通过中止信号取消对 `generate()` 和 `stream()` 工作
- [ ] 超时工作（总超时和每步超时）

### 8.5 推理令牌

- [ ] OpenAI 推理模型（GPT-5.2 系列等）通过 Responses API 在 `Usage` 中返回 `reasoning_tokens`
- [ ] `reasoning_effort` 参数正确传递给 OpenAI 推理模型
- [ ] Anthropic 扩展思考块在启用时作为 `THINKING` 内容部分返回
- [ ] 思考块 `signature` 字段为往返保留
- [ ] Gemini 思考令牌 (`thoughtsTokenCount`) 映射到 `Usage` 中的 `reasoning_tokens`
- [ ] `Usage` 正确报告 `reasoning_tokens` 与 `output_tokens` 不同

### 8.6 提示缓存

- [ ] **OpenAI**：通过 Responses API 自动缓存工作（无需客户端配置）
- [ ] **OpenAI**：`Usage.cache_read_tokens` 从 `usage.prompt_tokens_details.cached_tokens` 填充
- [ ] **Anthropic**：适配器自动在系统提示、工具定义和对话前缀上注入 `cache_control` 断点
- [ ] **Anthropic**：存在 cache_control 时自动包含 `prompt-caching-2024-07-31` beta 标头
- [ ] **Anthropic**：`Usage.cache_read_tokens` 和 `Usage.cache_write_tokens` 正确填充
- [ ] **Anthropic**：可以通过 `provider_options.anthropic.auto_cache = false` 禁用自动缓存
- [ ] **Gemini**：自动前缀缓存工作（无需客户端配置）
- [ ] **Gemini**：`Usage.cache_read_tokens` 从 `usageMetadata.cachedContentTokenCount` 填充
- [ ] 多轮代理会话：验证第 5+ 轮显示所有三个提供商的显著 cache_read_tokens（>输入令牌的 50%）

### 8.7 工具调用

- [ ] 带有 `execute` 处理程序的工具（活动工具）触发自动工具执行循环
- [ ] 没有 `execute` 处理程序的工具（被动工具）将工具调用返回给调用者而不循环
- [ ] `max_tool_rounds` 受到尊重：循环在配置的轮次数后停止
- [ ] `max_tool_rounds = 0` 完全禁用自动执行
- [ ] **并行工具调用**：当模型在一个响应中返回 N 个工具调用时，所有 N 个并发执行
- [ ] **并行工具结果**：所有 N 个结果在单个继续请求中发送回（而不是一次一个）
- [ ] 工具执行错误作为错误结果发送给模型（`is_error = true`），而不是作为异常引发
- [ ] 未知工具调用（模型调用未在定义中的工具）发送错误结果，而不是异常
- [ ] `ToolChoice` 模式（auto、none、required、named）根据每个提供商正确转换
- [ ] 工具调用参数 JSON 在传递给执行处理程序之前被解析和验证
- [ ] `StepResult` 对象跟踪每一步的工具调用、结果和使用情况

### 8.8 错误处理和重试

- [ ] 层次结构中的所有错误都针对正确的 HTTP 状态代码引发（参见第 6.4 节表）
- [ ] `retryable` 标志在每个错误类型上正确设置
- [ ] 带有抖动的指数退避工作：延迟每次尝试正确增加
- [ ] `Retry-After` 标头在存在时（并在 `max_delay` 内）覆盖计算出的退避
- [ ] `max_retries = 0` 禁用自动重试
- [ ] 速率限制错误 (429) 透明重试
- [ ] 不可重试的错误 (401、403、404) 立即引发而不重试
- [ ] 重试应用于每步，而不是整个多步操作
- [ ] 流式传输在传递部分数据后不重试

### 8.9 跨提供商同等性

运行此验证矩阵——每个单元格必须通过：

| 测试用例                                | OpenAI | Anthropic | Gemini |
|------------------------------------------|--------|-----------|--------|
| 简单文本生成                   | [ ]    | [ ]       | [ ]    |
| 流式文本生成                | [ ]    | [ ]       | [ ]    |
| 图像输入 (base64)                     | [ ]    | [ ]       | [ ]    |
| 图像输入 (URL)                        | [ ]    | [ ]       | [ ]    |
| 单个工具调用 + 执行             | [ ]    | [ ]       | [ ]    |
| 多个并行工具调用             | [ ]    | [ ]       | [ ]    |
| 多步工具循环 (3+ 轮)         | [ ]    | [ ]       | [ ]    |
| 带有工具调用的流式传输                | [ ]    | [ ]       | [ ]    |
| 结构化输出 (generate_object)      | [ ]    | [ ]       | [ ]    |
| 推理/思考令牌报告       | [ ]    | [ ]       | [ ]    |
| 错误处理 (无效 API 密钥 -> 401)  | [ ]    | [ ]       | [ ]    |
| 错误处理 (速率限制 -> 429)       | [ ]    | [ ]       | [ ]    |
| 使用令牌计数准确          | [ ]    | [ ]       | [ ]    |
| 提示缓存 (第 2+ 轮的 cache_read_tokens > 0) | [ ] | [ ]  | [ ]    |
| 提供商特定选项传递   | [ ]    | [ ]       | [ ]    |

### 8.10 集成冒烟测试

最终验证：使用真实 API 密钥对所有三个提供商运行此端到端测试。

```
-- 1. 跨所有提供商的基本生成
FOR EACH provider IN ["anthropic", "openai", "gemini"]:
    result = generate(
        model = get_latest_model(provider).id,
        prompt = "Say hello in one sentence.",
        max_tokens = 100,
        provider = provider
    )
    ASSERT result.text is not empty
    ASSERT result.usage.input_tokens > 0
    ASSERT result.usage.output_tokens > 0
    ASSERT result.finish_reason.reason == "stop"

-- 2. 流式传输
stream_result = stream(model = "claude-opus-4-6", prompt = "Write a haiku.")
text_chunks = []
FOR EACH event IN stream_result:
    IF event.type == TEXT_DELTA:
        text_chunks.APPEND(event.delta)
ASSERT JOIN(text_chunks) == stream_result.response().text

-- 3. 带有并行执行的工具调用
result = generate(
    model = "claude-opus-4-6",
    prompt = "What is the weather in San Francisco and New York?",
    tools = [weather_tool],    -- 返回模拟天气数据的工具
    max_tool_rounds = 3
)
ASSERT LENGTH(result.steps) >= 2               -- 至少：初始调用 + 工具结果后
ASSERT result.text contains "San Francisco"
ASSERT result.text contains "New York"

-- 4. 图像输入
result = generate(
    model = "claude-opus-4-6",
    messages = [Message(role=USER, content=[
        ContentPart(kind=TEXT, text="What do you see?"),
        ContentPart(kind=IMAGE, image=ImageData(data=<png_bytes>, media_type="image/png"))
    ])]
)
ASSERT result.text is not empty

-- 5. 结构化输出
result = generate_object(
    model = "gpt-5.2",
    prompt = "Extract: Alice is 30 years old",
    schema = {"type":"object", "properties":{"name":{"type":"string"},"age":{"type":"integer"}}, "required":["name","age"]}
)
ASSERT result.output.name == "Alice"
ASSERT result.output.age == 30

-- 6. 错误处理
TRY:
    generate(model = "nonexistent-model-xyz", prompt = "test", provider = "openai")
    FAIL("Should have raised an error")
CATCH NotFoundError:
    PASS   -- 正确的错误类型
```

如果勾选了本节中的所有项目，则统一的 LLM 库已完成，并可用作编码代理或任何其他 LLM 驱动应用程序的基础。
