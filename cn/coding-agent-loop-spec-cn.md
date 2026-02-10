# 编码代理循环规范

本文档是一个与编程语言无关的规范，用于构建编码代理——一个通过代理循环将大语言模型与开发工具配对的自主系统。它的设计使得任何开发者或编码代理都能从零开始，使用任何编程语言实现它。

本规范建立在[统一 LLM 客户端规范](./unified-llm-spec.md)之上，该规范处理所有 LLM 通信。代理循环直接使用 SDK 的底层 `Client.complete()` 和 `Client.stream()` 方法，实现自己的回合循环，以交错执行工具调用、输出截断、引导、事件和循环检测。

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [代理循环](#2-代理循环)
3. [提供商对齐的工具集](#3-提供商对齐的工具集)
4. [工具执行环境](#4-工具执行环境)
5. [工具输出与上下文管理](#5-工具输出与上下文管理)
6. [系统提示词与环境上下文](#6-系统提示词与环境上下文)
7. [子代理](#7-子代理)
8. [超出范围（锦上添花的功能）](#8-超出范围锦上添花的功能)
9. [完成定义](#9-完成定义)

---

## 1. 概述与目标

### 1.1 问题陈述

编码代理是一个系统，它接收自然语言指令（"修复登录 bug"、"添加深色模式"、"为该模块编写测试"），规划解决方案，并通过读取文件、编辑代码、运行命令和迭代来执行，直到任务完成。核心挑战是编排一个代理循环，协调 LLM 调用、工具执行、上下文管理和特定提供商的行为，形成一个可靠的自主工作流程。

每个 LLM 提供商的模型都针对特定的工具接口和系统提示词进行了训练和优化。GPT-5.2 和 GPT-5.2-codex 系列与 codex-rs 使用相同的工具和提示词时效果最佳。Gemini 模型与 gemini-cli 使用相同的工具和提示词时效果最佳。Anthropic 模型与 Claude Code 使用相同的工具和提示词时效果最佳。一个好的编码代理应该尊重这一现实，而不是强制所有模型使用通用工具集。

### 1.2 为什么是库而不是 CLI

Claude Code、Codex CLI 和 Gemini CLI 等工具作为最终用户 CLI 存在。您可以在非交互模式下运行它们并管道输出，但这会给您一个黑盒：文本输入，文本输出。您无法在运行时以编程方式检查对话、在工具调用之间注入引导消息、交换执行环境、动态更改推理工作量、观察单个工具调用的发生，或将代理组合到更大的系统中。

本规范定义了一个**库**——一个可编程的代理循环，主机应用程序在每个步骤都控制它。主机可以：

- **提交输入并观察每个事件**，当代理思考、调用工具和产生输出时——不是事后，而是实时发生。
- **在任务中途引导代理**，通过在工具回合之间注入消息，无需重启即可重定向它。
- **动态更改配置**——推理工作量、模型、超时——在任何两个回合之间。
- **交换工具运行位置**，通过提供不同的执行环境（本地、Docker、Kubernetes、WASM、SSH），无需更改任何工具逻辑。
- **组合代理**，通过为并行工作生成子代理，每个代理都有自己的历史记录，但共享同一个文件系统。
- **在此基础上构建**——CLI、IDE、Web UI、批处理系统、CI 管道、评估工具和代理到代理的协调系统都使用同一个库。

控制保真度是关键。每个编码代理 CLI 内部都建立在代理循环之上；本规范使该循环成为一流的、可编程的接口。

### 1.3 设计原则

**可编程优先。**代理是一个库，而不是 CLI。循环的每个方面——工具执行、事件传递、引导、配置——都可以通过编程访问。CLI 是一种可能的主机应用程序，而不是主要接口。

**提供商对齐。**每个模型家族与其原生代理的工具和系统提示词配合使用效果最佳。规范定义了特定于提供商的工具配置文件，而不是单一的通用集。从提供商的原生工具集开始并扩展它。

**可扩展执行。**工具执行被抽象在 `ExecutionEnvironment` 接口后面。默认在本地运行。实现可以针对 Docker、Kubernetes、WASM 或任何远程主机。更改工具运行位置不应需要更改工具本身。

**事件驱动。**每个代理操作都会发出一个类型化事件，用于 UI 渲染、日志记录和集成。事件流是主机应用程序的主要接口。

**可黑客攻击。**合理的默认值，到处都有覆盖点——超时、输出大小、工具集、执行环境、系统提示词、推理工作量。规范规定默认值，而不是上限。

**与语言无关。**所有代码都是伪代码。数据结构使用中性表示法。不假设特定的编程语言。

### 1.3 架构

```
+--------------------------------------------------+
|  主机应用程序 (CLI、IDE、Web UI)                   |
+--------------------------------------------------+
        |                            ^
        | submit(input)              | events
        v                            |
+--------------------------------------------------+
|  编码代理循环                                      |
|  +--------------------+  +---------------------+ |
|  | Session            |  | Provider Profiles   | |
|  |  - history         |  |  - OpenAI (codex)   | |
|  |  - steering queue  |  |  - Anthropic (cc)   | |
|  |  - event emitter   |  |  - Gemini (cli)     | |
|  +--------------------+  +---------------------+ |
|  +--------------------+  +---------------------+ |
|  | Tool Registry      |  | Execution Env       | |
|  |  - tool dispatch   |  |  - local (default)  | |
|  |  - truncation      |  |  - docker           | |
|  |  - validation      |  |  - k8s / wasm / ssh | |
|  +--------------------+  +---------------------+ |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|  统一 LLM SDK (Client.complete / stream)          |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|  LLM Provider APIs                                |
|  (OpenAI Responses / Anthropic Messages / Gemini) |
+--------------------------------------------------+
```

代理循环**不使用**统一 LLM SDK 的 `generate()` 高级函数（它有自己的工具循环）。它使用底层的 `Client.complete()` 并实现自己的循环，因为它需要交错执行工具调用与输出截断、引导消息注入、事件发射、超时强制执行和循环检测——这些是 SDK 通用工具循环不处理的关注点。

### 1.4 参考项目

以下开源项目解决了相关问题，任何实现本规范的人都值得研究。

- **codex-rs** (https://github.com/openai/codex/tree/main/codex-rs) -- Rust。OpenAI 的编码代理。演示了异步基于回合的循环、15+ 工具包括 `apply_patch`（v4a diff 格式）、使用头/尾分割的输出截断（1 MiB 上限）、10 秒默认命令超时、特定平台的沙箱、子代理生成和环境变量过滤。

- **pi-agent-core** (https://github.com/badlogic/pi-mono/tree/main/packages/agent) -- TypeScript。@mariozechner 的最小代理核心。演示了 4 工具极简主义（read、write、edit、bash）、显式 `steer()` 和 `followUp()` 队列用于回合间消息注入、15+ 事件类型、可配置的思维级别、上下文转换钩子和中止信号支持。

- **gemini-cli** (https://github.com/google-gemini/gemini-cli) -- TypeScript。Google 的 CLI 代理。演示了 ReAct 循环、18+ 内置工具包括 Web 搜索和 Web 获取、GEMINI.md 用于项目特定指令、无头/非交互模式用于自动化，以及多种身份验证方法。

### 1.5 与统一 LLM SDK 的关系

本规范假设伴随的统一 LLM 客户端规范已实现。代理循环直接导入和使用这些类型：

- `Client`、`Request`、`Response` -- 用于 LLM 通信
- `Message`、`ContentPart`、`Role` -- 用于对话历史
- `Tool`、`ToolCall`、`ToolResult` -- 用于工具定义和结果
- `StreamEvent` -- 用于流式响应
- `Usage` -- 用于令牌跟踪
- `FinishReason` -- 用于停止条件检测

代理构建 `Request` 对象，调用 `Client.complete()` 或 `Client.stream()`，处理 `Response`，通过执行环境执行任何 `ToolCall` 对象，构建 `ToolResult` 对象，将它们追加到对话中，并循环。

---

## 2. 代理循环

### 2.1 会话

会话是中央协调器。它持有对话状态、调度工具调用、管理事件流并强制执行限制。

```
RECORD Session:
    id                : String                  -- UUID，在创建时分配
    provider_profile  : ProviderProfile         -- 活跃模型的工具 + 系统提示词
    execution_env     : ExecutionEnvironment    -- 工具运行位置
    history           : List<Turn>              -- 有序的对话回合
    event_emitter     : EventEmitter            -- 将事件传递给主机应用程序
    config            : SessionConfig           -- 限制、超时、设置
    state             : SessionState            -- 当前生命周期状态
    llm_client        : Client                  -- 来自统一 LLM SDK
    steering_queue    : Queue<String>           -- 在工具回合之间注入的消息
    followup_queue    : Queue<String>           -- 当前输入完成后处理的消息
    subagents         : Map<String, SubAgent>   -- 活跃的子代理
```

### 2.2 会话配置

```
RECORD SessionConfig:
    max_turns                   : Integer = 0       -- 0 = 无限制
    max_tool_rounds_per_input   : Integer = 200     -- 每个用户输入，而不是每个会话
    default_command_timeout_ms  : Integer = 10000   -- 10 秒
    max_command_timeout_ms      : Integer = 600000  -- 10 分钟
    reasoning_effort            : String | None     -- "low"、"medium"、"high" 或 null
    tool_output_limits          : Map<String, Integer>  -- 每工具字符限制（见第 5 节）
    enable_loop_detection       : Boolean = true
    loop_detection_window       : Integer = 10      -- 警告前的连续相同调用次数
    max_subagent_depth          : Integer = 1       -- 子代理的最大嵌套级别
```

### 2.3 会话生命周期

```
ENUM SessionState:
    IDLE              -- 等待用户输入
    PROCESSING        -- 运行代理循环
    AWAITING_INPUT    -- 模型向用户提问
    CLOSED            -- 会话终止（正常或错误）
```

状态转换：

```
IDLE -> PROCESSING          -- 在 submit() 时
PROCESSING -> PROCESSING    -- 工具循环继续
PROCESSING -> AWAITING_INPUT -- 模型向用户提问（无工具调用，开放式）
PROCESSING -> IDLE          -- 自然完成或回合限制
PROCESSING -> CLOSED        -- 不可恢复的错误
IDLE -> CLOSED              -- 显式 close()
any -> CLOSED               -- 中止信号
AWAITING_INPUT -> PROCESSING -- 用户提供答案
```

### 2.4 回合类型

回合是对话历史中的单个条目。

```
RECORD UserTurn:
    content     : String
    timestamp   : Timestamp

RECORD AssistantTurn:
    content     : String            -- 文本输出
    tool_calls  : List<ToolCall>    -- 模型请求的工具调用
    reasoning   : String | None     -- 思考/推理文本（如果可用）
    usage       : Usage             -- 此回合的令牌计数
    response_id : String | None     -- 提供商响应 ID
    timestamp   : Timestamp

RECORD ToolResultsTurn:
    results     : List<ToolResult>  -- 每个工具调用一个
    timestamp   : Timestamp

RECORD SystemTurn:
    content     : String
    timestamp   : Timestamp

RECORD SteeringTurn:
    content     : String            -- 注入的引导消息
    timestamp   : Timestamp
```

### 2.5 核心代理循环

这是规范的核心。循环运行直到模型生成仅文本响应（无工具调用）、达到限制或触发中止信号。

```
FUNCTION process_input(session, user_input):
    session.state = PROCESSING
    session.history.APPEND(UserTurn(content = user_input))
    session.emit(USER_INPUT, content = user_input)

    -- 在第一次 LLM 调用之前排空任何待处理的引导消息
    drain_steering(session)

    round_count = 0

    LOOP:
        -- 1. 检查限制
        IF round_count >= session.config.max_tool_rounds_per_input:
            session.emit(TURN_LIMIT, round = round_count)
            BREAK

        IF session.config.max_turns > 0 AND count_turns(session) >= session.config.max_turns:
            session.emit(TURN_LIMIT, total_turns = count_turns(session))
            BREAK

        IF session.abort_signaled:
            BREAK

        -- 2. 使用提供商配置文件构建 LLM 请求
        system_prompt = session.provider_profile.build_system_prompt(
            environment = session.execution_env,
            project_docs = discover_project_docs(session.execution_env.working_directory())
        )
        messages = convert_history_to_messages(session.history)
        tool_defs = session.provider_profile.tools()

        request = Request(
            model           = session.provider_profile.model,
            messages        = [Message.system(system_prompt)] + messages,
            tools           = tool_defs,
            tool_choice     = "auto",
            reasoning_effort = session.config.reasoning_effort,
            provider        = session.provider_profile.id,
            provider_options = session.provider_profile.provider_options()
        )

        -- 3. 通过统一 LLM SDK 调用 LLM（单次，无 SDK 级工具循环）
        response = session.llm_client.complete(request)

        -- 4. 记录助手回合
        assistant_turn = AssistantTurn(
            content     = response.text,
            tool_calls  = response.tool_calls,
            reasoning   = response.reasoning,
            usage       = response.usage,
            response_id = response.id
        )
        session.history.APPEND(assistant_turn)
        session.emit(ASSISTANT_TEXT_END, text = response.text, reasoning = response.reasoning)

        -- 5. 如果没有工具调用，自然完成
        IF response.tool_calls IS EMPTY:
            BREAK

        -- 6. 通过执行环境执行工具调用
        round_count += 1
        results = execute_tool_calls(session, response.tool_calls)
        session.history.APPEND(ToolResultsTurn(results = results))

        -- 7. 排空在工具执行期间注入的引导消息
        drain_steering(session)

        -- 8. 循环检测
        IF session.config.enable_loop_detection:
            IF detect_loop(session.history, session.config.loop_detection_window):
                warning = "检测到循环：最后 " + session.config.loop_detection_window
                        + " 次工具调用遵循重复模式。请尝试不同的方法。"
                session.history.APPEND(SteeringTurn(content = warning))
                session.emit(LOOP_DETECTION, message = warning)

    END LOOP

    -- 如果有排队的后续消息，则处理它们
    IF session.followup_queue IS NOT EMPTY:
        next_input = session.followup_queue.DEQUEUE()
        process_input(session, next_input)
        RETURN

    session.state = IDLE
    session.emit(SESSION_END)


FUNCTION drain_steering(session):
    WHILE session.steering_queue IS NOT EMPTY:
        msg = session.steering_queue.DEQUEUE()
        session.history.APPEND(SteeringTurn(content = msg))
        session.emit(STEERING_INJECTED, content = msg)


FUNCTION execute_tool_calls(session, tool_calls):
    results = []

    -- 执行工具调用（如果配置文件支持并行执行，则并发执行）
    IF session.provider_profile.supports_parallel_tool_calls AND LENGTH(tool_calls) > 1:
        results = AWAIT_ALL([
            execute_single_tool(session, tc) FOR tc IN tool_calls
        ])
    ELSE:
        FOR EACH tc IN tool_calls:
            result = execute_single_tool(session, tc)
            results.APPEND(result)
    RETURN results


FUNCTION execute_single_tool(session, tool_call):
    session.emit(TOOL_CALL_START, tool_name = tool_call.name, call_id = tool_call.id)

    -- 在注册表中查找工具
    registered = session.provider_profile.tool_registry.get(tool_call.name)
    IF registered IS None:
        error_msg = "未知工具：" + tool_call.name
        session.emit(TOOL_CALL_END, call_id = tool_call.id, error = error_msg)
        RETURN ToolResult(tool_call_id = tool_call.id, content = error_msg, is_error = true)

    -- 通过执行环境执行
    TRY:
        raw_output = registered.execute(tool_call.arguments, session.execution_env)

        -- 在发送到 LLM 之前截断输出（先基于字符，然后基于行）
        truncated_output = truncate_tool_output(raw_output, tool_call.name, session.config)

        -- 通过事件流发出完整输出（未截断）
        session.emit(TOOL_CALL_END, call_id = tool_call.id, output = raw_output)

        RETURN ToolResult(
            tool_call_id = tool_call.id,
            content = truncated_output,
            is_error = false
        )

    CATCH error:
        error_msg = "工具错误（" + tool_call.name + "）：" + str(error)
        session.emit(TOOL_CALL_END, call_id = tool_call.id, error = error_msg)
        RETURN ToolResult(tool_call_id = tool_call.id, content = error_msg, is_error = true)
```

### 2.6 引导

引导允许主机应用程序在工具回合之间向对话中注入消息。这是用户可以在任务中途重定向代理而无需等待其完成的方式。

```
session.steer(message: String)
    -- 将消息排队以在当前工具回合完成后注入。
    -- 消息成为历史记录中的 SteeringTurn，转换为
    -- 下次调用的用户消息。
    -- 如果代理处于空闲状态，则在下次 submit() 时传递消息。

session.follow_up(message: String)
    -- 将消息排队以在当前输入完全处理后处理
    -- （模型已生成仅文本响应）。触发新的处理周期。
```

在构建 LLM 请求时，SteeringTurn 被转换为用户角色消息。这意味着模型将它们视为额外的用户指令。

### 2.7 推理工作量

`reasoning_effort` 配置控制模型进行多少推理/思考。它直接映射到统一 LLM SDK 的 Request 上的 `reasoning_effort` 字段。

| 值     | 效果                                                           |
|--------|--------------------------------------------------------------|
| "low"  | 最小推理。更快、更便宜。适用于简单任务。                        |
| "medium" | 平衡推理。大多数任务的默认值。                                |
| "high" | 深度推理。更慢、更昂贵。适用于复杂任务。                        |
| null   | 提供商默认值（无覆盖）。                                       |

在会话中途更改 `reasoning_effort` 在下次 LLM 调用时生效。对于 OpenAI 推理模型（GPT-5.2 系列），这控制推理令牌预算。对于具有扩展思考的 Anthropic 模型，这映射到思考预算。对于具有思考的 Gemini 模型，这映射到 thinkingConfig。

### 2.8 停止条件

当满足以下任何条件时，循环退出：

1. **自然完成。**模型仅以文本响应（无工具调用）。模型完成。
2. **回合限制。**达到 `max_tool_rounds_per_input`。代理停止并返回其已有的内容。
3. **回合限制。**达到整个会话的 `max_turns`。
4. **中止信号。**主机应用程序发出取消信号。当前 LLM 流关闭，正在运行的进程被终止，会话转换到 CLOSED。
5. **不可恢复的错误。**身份验证错误、上下文溢出或其他不可重试的错误。会话转换到 CLOSED。

### 2.9 事件系统

每个代理操作都会发出一个类型化事件。事件通过异步迭代器（或语言适当的等效项）传递给主机应用程序。

```
RECORD SessionEvent:
    kind        : EventKind
    timestamp   : Timestamp
    session_id  : String
    data        : Map<String, Any>

ENUM EventKind:
    SESSION_START           -- 会话创建
    SESSION_END             -- 会话关闭（包括最终状态）
    USER_INPUT              -- 用户提交输入
    ASSISTANT_TEXT_START     -- 模型开始生成文本
    ASSISTANT_TEXT_DELTA     -- 增量文本令牌
    ASSISTANT_TEXT_END       -- 模型完成文本（包括完整文本）
    TOOL_CALL_START         -- 工具执行开始（包括工具名称、调用 ID）
    TOOL_CALL_OUTPUT_DELTA  -- 增量工具输出（用于流式工具）
    TOOL_CALL_END           -- 工具执行完成（包括完整未截断输出）
    STEERING_INJECTED       -- 引导消息已添加到历史记录
    TURN_LIMIT              -- 达到回合限制
    LOOP_DETECTION          -- 检测到循环模式
    ERROR                   -- 发生错误
```

**关键设计决策：** `TOOL_CALL_END` 事件携带完整的未截断工具输出。LLM 接收截断版本。这意味着主机应用程序（UI、日志）始终可以访问完整输出，即使模型看到的是缩写版本。

### 2.10 循环检测

跟踪每个工具调用的签名（名称 + 参数哈希）。如果最后 N 次调用（默认：10）包含重复模式（例如，相同的 2-3 次调用循环），则注入警告作为 SteeringTurn，告诉模型尝试不同的方法。

```
FUNCTION detect_loop(history, window_size) -> Boolean:
    recent_calls = extract_tool_call_signatures(history, last = window_size)
    IF LENGTH(recent_calls) < window_size: RETURN false

    -- 检查长度为 1、2 或 3 的重复模式
    FOR pattern_len IN [1, 2, 3]:
        IF window_size % pattern_len != 0: CONTINUE
        pattern = recent_calls[0..pattern_len]
        all_match = true
        FOR i FROM pattern_len TO window_size STEP pattern_len:
            IF recent_calls[i..i+pattern_len] != pattern:
                all_match = false
                BREAK
        IF all_match: RETURN true

    RETURN false
```

---

## 3. 提供商对齐的工具集

### 3.1 提供商对齐原则

模型针对特定工具接口进行了训练和优化。OpenAI 的模型在 codex-rs 的 apply_patch 格式和工具架构上进行训练。Anthropic 的模型在 Claude Code 的 old_string/new_string 编辑和工具架构上进行训练。Gemini 模型在 gemini-cli 的工具集上进行训练。

使用提供商的原生工具格式比强制通用格式产生更好的结果。**每个提供商的初始基础应该是提供商参考代理的 1:1 副本——完全相同的系统提示词，完全相同的工具定义，逐字节。**不是类似的提示词。不是类似的工具。而是模型被评估和优化的实际提示词和工具。然后使用其他功能（如子代理）进行扩展。不要让所有提供商都符合单一工具接口。

### 3.2 ProviderProfile 接口

```
INTERFACE ProviderProfile:
    id              : String            -- "openai"、"anthropic"、"gemini"
    model           : String            -- 模型标识符（例如，"gpt-5.2-codex"）
    tool_registry   : ToolRegistry      -- 此配置文件可用的所有工具

    FUNCTION build_system_prompt(environment, project_docs) -> String
    FUNCTION tools() -> List<ToolDefinition>
    FUNCTION provider_options() -> Map | None

    -- 能力标志
    supports_reasoning           : Boolean
    supports_streaming           : Boolean
    supports_parallel_tool_calls : Boolean
    context_window_size          : Integer
```

### 3.3 共享核心工具

所有配置文件都包括这些基本工具。参数架构和输出格式可能在配置文件之间有所不同（以匹配提供商的原生约定），但功能相同。

#### read_file

读取带有行号的文件内容。

```
TOOL read_file:
    description: "从文件系统读取文件。返回带行号的内容。"
    parameters:
        file_path   : String (required)     -- 文件的绝对路径
        offset      : Integer (optional)    -- 从 1 开始的行号，从该行开始读取
        limit       : Integer (optional)    -- 要读取的最大行数（默认：2000）
    returns: "NNN | content" 格式的带行号文本内容
    errors: 文件未找到、权限被拒绝、二进制文件
```

行为：读取文件，前置行号，尊重 offset/limit。对于图像文件，返回多模态模型的图像数据。对于没有 offset/limit 的非常大的文件，工具输出将被截断层截断（第 5 节）。

#### write_file

将内容写入文件，如果不存在则创建。

```
TOOL write_file:
    description: "将内容写入文件。如果需要，创建文件和父目录。"
    parameters:
        file_path   : String (required)     -- 绝对路径
        content     : String (required)     -- 完整的文件内容
    returns: 确认消息，包含写入的字节数
    errors: 权限被拒绝、磁盘已满
```

#### edit_file

在文件中搜索精确字符串并替换它。这是 Anthropic 模型的原生编辑格式。

```
TOOL edit_file:
    description: "替换文件中的精确字符串出现。"
    parameters:
        file_path   : String (required)
        old_string  : String (required)     -- 要查找的精确文本
        new_string  : String (required)     -- 替换文本
        replace_all : Boolean (optional)    -- 替换所有出现（默认：false）
    returns: 确认，包含所做的替换次数
    errors: 文件未找到、未找到 old_string、old_string 不唯一（当 replace_all=false 时）
```

行为：精确字符串匹配。如果未精确找到 `old_string`，实现可能会尝试模糊匹配（空白规范化、Unicode 等效）并报告匹配。如果 `old_string` 匹配多个位置且 `replace_all` 为 false，则返回错误，要求模型提供更多上下文。

#### shell

在系统 shell 中执行命令。

```
TOOL shell:
    description: "执行 shell 命令。返回 stdout、stderr 和退出代码。"
    parameters:
        command     : String (required)     -- 要运行的命令
        timeout_ms  : Integer (optional)    -- 覆盖默认超时
        description : String (optional)     -- 人类可读的描述，说明此命令的作用
    returns: 命令输出（stdout + stderr）、退出代码、持续时间
    errors: 超时、权限被拒绝、未找到命令
```

行为：在新进程组中运行。强制执行超时（默认来自 SessionConfig，可每次调用覆盖）。超时时：SIGTERM，等待 2 秒，SIGKILL。返回收集的输出加上超时消息。应用环境变量过滤（见第 4 节）。

#### grep

按模式搜索文件内容。

```
TOOL grep:
    description: "使用正则表达式模式搜索文件内容。"
    parameters:
        pattern         : String (required)     -- 正则表达式模式
        path            : String (optional)     -- 要搜索的目录或文件（默认：工作目录）
        glob_filter     : String (optional)     -- 文件模式过滤器（例如，"*.py"）
        case_insensitive: Boolean (optional)    -- 默认：false
        max_results     : Integer (optional)    -- 默认：100
    returns: 带有文件路径和行号的匹配行
    errors: 无效的正则表达式、未找到路径
```

#### glob

按名称模式查找文件。

```
TOOL glob:
    description: "查找匹配 glob 模式的文件。"
    parameters:
        pattern     : String (required)     -- glob 模式（例如，"**/*.ts"）
        path        : String (optional)     -- 基本目录（默认：工作目录）
    returns: 匹配的文件路径列表，按修改时间排序（最新的在前）
    errors: 无效的模式、未找到路径
```

### 3.4 OpenAI 配置文件（codex-rs 对齐）

适用于 GPT-5.2、GPT-5.2-codex 和其他 OpenAI 模型。镜像 codex-rs 工具集。

**关键区别：`apply_patch` 替换 `edit_file` 和 `write_file` 进行文件修改。** OpenAI 模型专门针对这种格式进行训练，使用它时会产生明显更好的编辑。

共享核心之外的其他/修改工具：

#### apply_patch（OpenAI 特定）

```
TOOL apply_patch:
    description: "使用补丁格式应用代码更改。支持在单个操作中创建、删除和修改文件。"
    parameters:
        patch       : String (required)     -- v4a 格式的补丁内容
    returns: 受影响的文件路径列表和执行的操作
    errors: 解析错误、未找到文件（对于更新）、验证失败
```

补丁格式在[附录 A](#附录-a-apply_patch-v4a-格式参考)中完整定义。

**OpenAI 的配置文件工具列表：**
- `read_file`（与共享核心相同，映射到 codex-rs `read_file`）
- `apply_patch`（替换 `edit_file` 和 `write_file` 进行修改）
- `write_file`（保留用于创建新文件，无需补丁开销）
- `shell`（映射到 codex-rs `exec_command`，10s 默认超时）
- `grep`（映射到 codex-rs `grep_files`）
- `glob`（映射到 codex-rs `list_dir`）
- `spawn_agent`、`send_input`、`wait`、`close_agent`（子代理工具，第 7 节）

**系统提示词：**应镜像 codex-rs 系统提示词结构。涵盖身份、工具使用指南、apply_patch 格式期望和编码最佳实践。

**提供商选项：** OpenAI 配置文件应在配置 `reasoning_effort` 时在 Responses API 请求上设置 `reasoning.effort`。

### 3.5 Anthropic 配置文件（Claude Code 对齐）

适用于 Claude Opus 4.6、Opus 4.5、Sonnet 4.5、Haiku 4.5 和较旧的 Claude 模型。镜像 Claude Code 工具集。

**关键区别：带有 `old_string`/`new_string` 的 `edit_file` 是原生编辑格式。** Anthropic 模型专门针对这种精确匹配搜索和替换模式进行训练。不要将 apply_patch 与 Anthropic 模型一起使用。

**Anthropic 的配置文件工具列表：**
- `read_file`（带行号的输出，支持 offset/limit）
- `write_file`（完整文件写入）
- `edit_file`（old_string/new_string -- 这是原生格式）
- `shell`（bash 执行，120s 默认超时，按照 Claude Code 约定）
- `grep`（支持 ripgrep，输出模式：content、files_with_matches、count）
- `glob`（按 mtime 排序的文件模式匹配）
- 子代理工具（映射到 Claude Code 的 Task 工具模式，第 7 节）

**系统提示词：**应镜像 Claude Code 系统提示词结构。涵盖身份、工具选择指南、edit_file 格式（解释 `old_string` 必须是唯一的）、文件操作首选项（编辑现有文件而不是创建新文件）和编码最佳实践。

**提供商选项：** Anthropic 配置文件应通过 `provider_options.anthropic.beta_headers` 传递 beta 标头（例如，用于扩展思考、1M 上下文）。

### 3.6 Gemini 配置文件（gemini-cli 对齐）

适用于 Gemini 3 Flash、Gemini 2.5 Pro/Flash 和其他 Gemini 模型。镜像 gemini-cli 工具集。

**Gemini 的配置文件工具列表：**
- `read_file` / `read_many_files`（批量读取支持）
- `write_file`
- `edit_file`（搜索和替换样式，匹配 gemini-cli 约定）
- `shell`（命令执行，10s 默认超时）
- `grep`（ripgrep 语义）
- `glob`（文件模式匹配）
- `list_dir`（带有深度选项的目录列表）
- `web_search`（可选 -- Gemini 模型具有原生接地能力）
- `web_fetch`（可选 -- 从 URL 获取和提取内容）
- 子代理工具（第 7 节）

**系统提示词：**应镜像 gemini-cli 系统提示词结构。涵盖身份、工具使用、GEMINI.md 约定和编码最佳实践。

**提供商选项：** Gemini 配置文件应通过 `provider_options.gemini` 配置安全设置和接地。

### 3.7 使用自定义工具扩展配置文件

加载提供商配置文件后，可以注册其他工具：

```
profile = create_openai_profile(model = "gpt-5.2-codex")

-- 在配置文件之上添加自定义工具
profile.tool_registry.register(RegisteredTool(
    definition = ToolDefinition(
        name = "run_tests",
        description = "运行项目的测试套件",
        parameters = { "type": "object", "properties": { "filter": { "type": "string" } } }
    ),
    executor = run_tests_function
))
```

名称冲突由最新获胜解决：与配置文件工具同名的自定义工具会覆盖它。

### 3.8 工具注册表

```
RECORD ToolDefinition:
    name        : String            -- 唯一标识符
    description : String            -- 用于 LLM
    parameters  : Dict              -- JSON Schema（根必须是 "object"）

RECORD RegisteredTool:
    definition  : ToolDefinition
    executor    : Function          -- (arguments, execution_env) -> String

RECORD ToolRegistry:
    _tools      : Map<String, RegisteredTool>

    register(tool)                  -- 添加或替换工具
    unregister(name)                -- 删除工具
    get(name) -> RegisteredTool | None
    definitions() -> List<ToolDefinition>
    names() -> List<String>
```

**工具执行管道：**

```
1. LOOKUP      -- 按名称查找 RegisteredTool
2. VALIDATE    -- 根据架构解析和验证参数
3. EXECUTE     -- 使用 (arguments, execution_env) 调用执行器
4. TRUNCATE    -- 应用输出大小限制（第 5 节）
5. EMIT        -- 发出带有完整输出的 TOOL_CALL_END 事件
6. RETURN      -- 将截断的输出作为 ToolResult 返回
```

---

## 4. 工具执行环境

### 4.1 执行环境抽象

所有工具操作都通过 `ExecutionEnvironment` 接口传递。这将工具逻辑与其运行位置解耦。默认在本地运行。交换不同的实现以在 Docker、Kubernetes pod、SSH 或 WASM 中运行相同的工具。

```
INTERFACE ExecutionEnvironment:
    -- 文件操作
    read_file(path: String, offset: Integer | None, limit: Integer | None) -> String
    write_file(path: String, content: String) -> void
    file_exists(path: String) -> Boolean
    list_directory(path: String, depth: Integer) -> List<DirEntry>

    -- 命令执行
    exec_command(
        command     : String,
        timeout_ms  : Integer,
        working_dir : String | None,
        env_vars    : Map<String, String> | None
    ) -> ExecResult

    -- 搜索操作
    grep(pattern: String, path: String, options: GrepOptions) -> String
    glob(pattern: String, path: String) -> List<String>

    -- 生命周期
    initialize() -> void
    cleanup() -> void

    -- 元数据
    working_directory() -> String
    platform() -> String           -- "darwin"、"linux"、"windows"、"wasm"
    os_version() -> String

RECORD ExecResult:
    stdout      : String
    stderr      : String
    exit_code   : Integer
    timed_out   : Boolean
    duration_ms : Integer

RECORD DirEntry:
    name        : String
    is_dir      : Boolean
    size        : Integer | None
```

### 4.2 LocalExecutionEnvironment（必需实现）

默认实现。在本地计算机上运行所有内容。

**文件操作：**直接文件系统访问。路径相对于 `working_directory()` 解析。

**命令执行：**
- 在新进程组中生成以实现干净的终止
- 使用平台的默认 shell（Linux/macOS 上的 `/bin/bash -c`，Windows 上的 `cmd.exe /c`）
- 强制执行超时：超时时，向进程组发送 SIGTERM，等待 2 秒，然后 SIGKILL
- 分别捕获 stdout 和 stderr，然后为结果组合
- 记录挂钟持续时间

**环境变量过滤：**
- 默认情况下，排除匹配以下模式的变量：`*_API_KEY`、`*_SECRET`、`*_TOKEN`、`*_PASSWORD`、`*_CREDENTIAL`（不区分大小写）
- 始终包括：`PATH`、`HOME`、`USER`、`SHELL`、`LANG`、`TERM`、`TMPDIR`、特定于语言的路径（`GOPATH`、`CARGO_HOME`、`NVM_DIR` 等）
- 可通过环境变量策略自定义：继承全部、不继承（从干净开始）或仅继承核心

**搜索操作：**如果可用，使用 `ripgrep` 进行 grep，否则回退到语言原生的正则表达式搜索。对 glob 使用文件系统 globbing。

### 4.3 替代环境（扩展点）

这些不是必需的实现。它们演示了接口的可扩展性。

**DockerExecutionEnvironment:**
```
-- 命令在 Docker 容器内执行
exec_command(cmd, ...) -> docker exec <container_id> sh -c <cmd>
-- 文件操作使用卷挂载或 docker cp
read_file(path) -> docker cp <container_id>:<path> - | read
write_file(path, content) -> pipe content | docker cp - <container_id>:<path>
```

**KubernetesExecutionEnvironment:**
```
-- 命令在 Kubernetes pod 中执行
exec_command(cmd, ...) -> kubectl exec <pod> -- sh -c <cmd>
-- 文件操作使用 kubectl cp
read_file(path) -> kubectl cp <pod>:<path> /dev/stdout
```

**WASMExecutionEnvironment:**
```
-- 用于浏览器或嵌入式使用
-- 文件操作使用内存文件系统（例如，memfs）
-- 命令执行通过 WASI 受限或模拟
```

**RemoteSSHExecutionEnvironment:**
```
-- 命令通过 SSH 执行
exec_command(cmd, ...) -> ssh <host> <cmd>
-- 文件操作使用 SCP/SFTP
read_file(path) -> sftp get <host>:<path>
```

### 4.4 组合环境

可以包装执行环境以处理横切关注点：

```
-- 日志记录包装器
LoggingExecutionEnvironment(inner: ExecutionEnvironment):
    exec_command(cmd, ...):
        LOG("exec: " + cmd)
        result = inner.exec_command(cmd, ...)
        LOG("exit: " + result.exit_code + " in " + result.duration_ms + "ms")
        RETURN result

-- 只读包装器（拒绝所有写入）
ReadOnlyExecutionEnvironment(inner: ExecutionEnvironment):
    write_file(path, content):
        RAISE "只读模式下禁用写入操作"
    exec_command(cmd, ...):
        -- 可以分析命令的写入意图，或允许所有
        RETURN inner.exec_command(cmd, ...)
```

---

## 5. 工具输出与上下文管理

### 5.1 工具输出截断

当工具输出超过配置的限制时，必须在发送到 LLM 之前截断它。完整输出始终通过事件流（`TOOL_CALL_END` 事件）可用。

**截断算法（头/尾分割）：**

```
FUNCTION truncate_output(output: String, max_chars: Integer, mode: String) -> String:
    IF LENGTH(output) <= max_chars:
        RETURN output

    IF mode == "head_tail":
        half = max_chars / 2
        removed = LENGTH(output) - max_chars
        RETURN output[0..half]
             + "\n\n[警告：工具输出已被截断。"
             + removed + " 个字符已从中间删除。"
             + " 完整输出在事件流中可用。"
             + " 如果您需要查看特定部分，请使用更有针对性的参数重新运行工具。]\n\n"
             + output[-half..]

    IF mode == "tail":
        removed = LENGTH(output) - max_chars
        RETURN "[警告：工具输出已被截断。前 "
             + removed + " 个字符已删除。"
             + " 完整输出在事件流中可用。]\n\n"
             + output[-max_chars..]
```

截断消息明确告诉模型输出已被截断、删除了多少以及完整输出的位置。这可以防止模型在不完整信息不知情的情况下根据不完整信息做出决策。

### 5.2 默认输出大小限制

| 工具         | 默认最大值（字符） | 截断模式 | 基本原理                                           |
|--------------|--------------------|----------|--------------------------------------------------|
| read_file    | 50,000             | head_tail| 保留开头（导入/类型）和结尾（最近的代码）            |
| shell        | 30,000             | head_tail| 开头有启动信息，结尾有结果                        |
| grep         | 20,000             | tail     | 保留最近/相关的匹配                               |
| glob         | 20,000             | tail     | 最近修改的文件在前                                |
| edit_file    | 10,000             | tail     | 确认输出，通常很短                                |
| apply_patch  | 10,000             | tail     | 补丁结果，通常很短                                |
| write_file   | 1,000              | tail     | 确认，总是很短                                   |
| spawn_agent  | 20,000             | head_tail| 子代理结果                                        |

这些默认值可通过 `SessionConfig.tool_output_limits` 覆盖。

### 5.3 截断顺序（重要）

基于字符的截断（第 5.1 节）是主要保障，必须始终首先运行。它处理所有情况，包括像每行 10MB 的 2 行 CSV 这样的病理情况。基于行的截断是字符截断后运行的次要可读性传递。

每个工具输出的完整管道：

```
FUNCTION truncate_tool_output(output, tool_name, config) -> String:
    max_chars = config.tool_output_limits.get(tool_name, DEFAULT_TOOL_LIMITS[tool_name])

    -- 步骤 1：基于字符的截断（始终运行，处理所有大小问题）
    result = truncate_output(output, max_chars, DEFAULT_TRUNCATION_MODES[tool_name])

    -- 步骤 2：基于行的截断（次要，用于可读性）
    max_lines = config.tool_line_limits.get(tool_name, DEFAULT_LINE_LIMITS[tool_name])
    IF max_lines IS NOT None:
        result = truncate_lines(result, max_lines)

    RETURN result
```

**默认行限制**（在字符截断后应用）：

| 工具         | 默认最大行数 | 基本原理                         |
|--------------|--------------|----------------------------------|
| shell        | 256          | 带有许多短行的命令输出            |
| grep         | 200          | 搜索结果，每行一个               |
| glob         | 500          | 文件列表，每行一个路径           |
| read_file    | None         | 字符限制足够                     |
| edit_file    | None         | 字符限制足够                     |

基于行的截断使用相同的头/尾分割：

```
FUNCTION truncate_lines(output: String, max_lines: Integer) -> String:
    lines = SPLIT(output, "\n")
    IF LENGTH(lines) <= max_lines:
        RETURN output

    head_count = max_lines / 2
    tail_count = max_lines - head_count
    omitted = LENGTH(lines) - head_count - tail_count

    RETURN JOIN(lines[0..head_count], "\n")
         + "\n[... 省略 " + omitted + " 行 ...]\n"
         + JOIN(lines[-tail_count..], "\n")
```

**为什么字符截断必须首先进行：**文件可能有 2 行，每行 10MB。基于行的截断会看到"只有 2 行"并原样传递，从而炸毁上下文窗口。字符截断会捕获这一点，因为它操作原始大小，而不是行数。始终首先按大小截断，然后按行数截断。

### 5.4 默认命令超时

每个命令执行都有默认超时。模型可以通过 shell 工具的 `timeout_ms` 参数每次调用覆盖超时。

| 设置                      | 默认值   | 目的                                 |
|---------------------------|----------|--------------------------------------|
| default_command_timeout_ms | 10,000   | 未设置 timeout_ms 时应用             |
| max_command_timeout_ms     | 600,000  | 上限（10 分钟）                      |

触发超时时：
1. 向进程组发送 SIGTERM
2. 等待 2 秒以实现优雅关闭
3. 如果进程仍在运行，则发送 SIGKILL
4. 返回到目前为止收集的输出加上超时消息

发送到 LLM 的超时消息：
```
[错误：命令在 {X}ms 后超时。上面显示了部分输出。
您可以通过设置 timeout_ms 参数使用更长的超时重试。]
```

### 5.5 上下文窗口感知

代理应使用启发式方法跟踪大约的令牌使用情况：1 个令牌约 4 个字符。当使用量超过提供商配置文件的 `context_window_size` 的 80% 时，发出警告事件。

这仅供参考。代理**不**执行自动压缩或摘要（这超出了本规范的范围）。主机应用程序可以使用此信号来实现自己的上下文管理策略。

```
FUNCTION check_context_usage(session):
    approx_tokens = total_chars_in_history(session.history) / 4
    threshold = session.provider_profile.context_window_size * 0.8
    IF approx_tokens > threshold:
        session.emit(WARNING, message = "上下文使用率约为 ~"
            + ROUND(approx_tokens / session.provider_profile.context_window_size * 100)
            + "% 的上下文窗口")
```

---

## 6. 系统提示词与环境上下文

### 6.1 分层系统提示词构建

系统提示词从多个层组装而成，后层优先：

```
final_system_prompt =
    1. 特定于提供商的基本指令             (来自 ProviderProfile)
  + 2. 环境上下文                         (平台、git、工作目录、日期、模型信息)
  + 3. 工具描述                           (来自活动配置文件的工具集)
  + 4. 项目特定指令                       (AGENTS.md、CLAUDE.md、GEMINI.md 等)
  + 5. 用户指令覆盖                       (最后追加，最高优先级)
```

### 6.2 特定于提供商的基本指令

每个配置文件提供自己的针对模型系列调整的基本提示词。基本指令应紧密镜像提供商原生代理的系统提示词：

- **OpenAI 配置文件：**镜像 codex-rs 系统提示词。涵盖身份、工具使用（尤其是 apply_patch 约定）、编码最佳实践、错误处理指南。
- **Anthropic 配置文件：**镜像 Claude Code 系统提示词。涵盖身份、工具选择指南（编辑前先读、编辑胜过写入）、edit_file 格式（old_string 必须是唯一的）、文件操作首选项。
- **Gemini 配置文件：**镜像 gemini-cli 系统提示词。涵盖身份、工具使用、GEMINI.md 约定、编码最佳实践。

规范**不**规定完整的系统提示词文本——这些是经常变化的实现细节。它指定提示词必须涵盖的主题。

### 6.3 环境上下文块

包含带有运行时信息的结构化块：

```
<environment>
工作目录：{working_directory}
是 git 仓库：{true/false}
Git 分支：{current_branch}
平台：{darwin/linux/windows}
操作系统版本：{os_version_string}
今天的日期：{YYYY-MM-DD}
模型：{model_display_name}
知识截止日期：{knowledge_cutoff_date}
</environment>
```

此块在会话开始时生成，并包含在每个系统提示词中。

### 6.4 Git 上下文

在会话开始时快照。包括：
- 当前分支
- 简短状态（修改/未跟踪文件计数，不是完整 diff）
- 最近的提交消息（最后 5-10 个）

模型始终可以通过 shell 工具运行 `git status`、`git diff` 等以获取当前状态。快照提供初始方向。

### 6.5 项目文档发现

从 git 根目录（如果不在 git 仓库中，则为工作目录）走到当前工作目录。识别的指令文件：

| 文件名                  | 约定           |
|-------------------------|----------------|
| `AGENTS.md`             | 通用           |
| `CLAUDE.md`             | Anthropic 对齐 |
| `GEMINI.md`             | Gemini 对齐    |
| `.codex/instructions.md`| OpenAI 对齐    |

**加载规则：**
- 首先加载根级别的文件
- 追加子目录文件（更深 = 更高优先级）
- 总字节预算：32KB。如果超出，使用标记截断："[项目指令在 32KB 处截断]"
- 仅加载与活动提供商配置文件匹配的文件（例如，Anthropic 配置文件加载 AGENTS.md 和 CLAUDE.md，而不是 GEMINI.md）
- AGENTS.md 始终加载，无论提供商如何

---

## 7. 子代理

### 7.1 概念

子代理是由父代生成的子会话，用于处理范围任务。子代理运行自己的代理循环，拥有自己的对话历史，但共享父代的执行环境（相同的文件系统，相同的工作目录或子目录）。这实现了并行工作和任务分解。

### 7.2 生成接口

```
TOOL spawn_agent:
    description: "生成子代理以自主处理范围任务。"
    parameters:
        task            : String (required)     -- 自然语言任务描述
        working_dir     : String (optional)     -- 将代理限制到的子目录
        model           : String (optional)     -- 模型覆盖（默认：父代的模型）
        max_turns       : Integer (optional)    -- 回合限制（默认：50）
    returns: 代理 ID 和初始状态

TOOL send_input:
    description: "向正在运行的子代理发送消息。"
    parameters:
        agent_id        : String (required)
        message         : String (required)
    returns: 确认

TOOL wait:
    description: "等待子代理完成并返回其结果。"
    parameters:
        agent_id        : String (required)
    returns: SubAgentResult（输出文本、成功布尔值、使用的回合数）

TOOL close_agent:
    description: "终止子代理。"
    parameters:
        agent_id        : String (required)
    returns: 最终状态
```

### 7.3 SubAgent 生命周期

```
RECORD SubAgentHandle:
    id          : String
    session     : Session           -- 具有自己历史的独立会话
    status      : "running" | "completed" | "failed"

RECORD SubAgentResult:
    output      : String            -- 来自子代理的最终文本输出
    success     : Boolean
    turns_used  : Integer
```

子代理：
- 获得自己的带有独立对话历史的 Session
- 共享父代的 `ExecutionEnvironment`（相同的文件系统）
- 使用父代的 `ProviderProfile`（或覆盖的模型）
- 有自己的回合限制（可配置，默认：50）
- 不能生成子子代理（深度限制，默认最大深度：1，可通过 `max_subagent_depth` 配置）

### 7.4 用例

- **并行探索：**生成多个代理以同时调查代码库的不同部分
- **专注重构：**将代理限制到具有特定任务的单个模块
- **测试执行：**生成代理以运行和修复测试，而父代继续其他工作
- **替代方法：**生成代理以尝试不同的解决方案并选择最佳方案

---

## 8. 超出范围（锦上添花的功能）

以下功能有意从此核心规范中排除。它们是有价值的扩展，可以在此处定义的架构之上添加。规范的设计为每个功能都有自然的扩展点。

**MCP（模型上下文协议）。** MCP 客户端可以使用来自外部服务器（GitHub、数据库、Slack 等）的工具扩展代理。工具注册表支持使用命名空间名称注册 MCP 发现的工具（例如，`github__create_pr`）。这是一个自然的扩展，但对于功能性编码代理不是核心要求。

**技能 / 自定义命令。**存储为带有 YAML 前言的 markdown 文件的可重用提示词模板。技能标准化常见工作流程（例如，`/commit`、`/review-pr`），可以从项目目录或用户主目录加载。系统提示词层具有技能描述的自然插入点。

**沙箱 / 安全策略。** 操作系统级沙箱（macOS Seatbelt、Linux Landlock/Seccomp、Windows 受限令牌）限制文件和网络访问。`ExecutionEnvironment` 抽象提供了一个自然的钩子——`SandboxedLocalExecutionEnvironment` 可以包装默认环境。对于更强的隔离，使用 `DockerExecutionEnvironment`。

**压缩 / 上下文摘要。** 接近上下文限制时的自动对话历史摘要。这是一个具有重大权衡的复杂功能（信息丢失、摘要成本、固定回合）。上下文窗口感知信号（第 5.5 节）为主机应用程序提供了实现自己策略所需的信息。

**批准 / 权限系统。** 敏感操作（文件写入、shell 命令、破坏性操作）的用户批准门。工具执行管道（第 3.8 节）在 VALIDATE 和 EXECUTE 之间有一个自然的扩展点，可以插入批准步骤。

**写入前读取护栏。** 跟踪哪些文件已被读取并阻止对未读文件的写入。可以实现的启发式安全网，作为包装执行环境的工具执行中间件。

---

## 9. 完成定义

本节定义如何验证此规范的实现是完整和正确的。当每个项目都被勾选时，实现就完成了。

### 9.1 核心循环

- [ ] 可以使用 ProviderProfile 和 ExecutionEnvironment 创建会话
- [ ] `process_input()` 运行代理循环：LLM 调用 -> 工具执行 -> 循环直到自然完成
- [ ] 自然完成：模型仅以文本响应（无工具调用）并且循环退出
- [ ] 回合限制：`max_tool_rounds_per_input` 在达到时停止循环
- [ ] 会话回合限制：`max_turns` 在所有输入中停止循环
- [ ] 中止信号：取消停止循环，终止正在运行的进程，转换到 CLOSED
- [ ] 循环检测：连续相同的工具调用模式触发警告 SteeringTurn
- [ ] 多个顺序输入工作：提交，等待完成，再次提交

### 9.2 提供商配置文件

- [ ] OpenAI 配置文件提供 codex-rs 对齐的工具，包括 `apply_patch`（v4a 格式）
- [ ] Anthropic 配置文件提供 Claude Code 对齐的工具，包括 `edit_file`（old_string/new_string）
- [ ] Gemini 配置文件提供 gemini-cli 对齐的工具
- [ ] 每个配置文件生成特定于提供商的系统提示词，涵盖身份、工具使用和编码指南
- [ ] 可以在任何配置文件上注册自定义工具
- [ ] 工具名称冲突已解决：自定义注册覆盖配置文件默认值

### 9.3 工具执行

- [ ] 工具调用通过 ToolRegistry 调度
- [ ] 未知工具调用向 LLM 返回错误结果（不是异常）
- [ ] 工具参数 JSON 根据工具的参数架构解析和验证
- [ ] 工具执行错误被捕获并作为错误结果返回（`is_error = true`）
- [ ] 当配置文件的 `supports_parallel_tool_calls` 为 true 时，并行工具执行工作

### 9.4 执行环境

- [ ] `LocalExecutionEnvironment` 实现所有文件和命令操作
- [ ] 命令超时默认为 10 秒
- [ ] 命令超时可通过 shell 工具的 `timeout_ms` 参数每次调用覆盖
- [ ] 超时命令：进程组接收 SIGTERM，2 秒后接收 SIGKILL
- [ ] 环境变量过滤默认排除敏感变量（`*_API_KEY`、`*_SECRET` 等）
- [ ] 消费者可以为自定义环境实现 `ExecutionEnvironment` 接口（Docker、K8s、WASM、SSH）

### 9.5 工具输出截断

- [ ] 基于字符的截断首先在所有工具输出上运行（处理病理情况，如 10MB 单行 CSV）
- [ ] 基于行的截断在配置的情况下第二运行（shell：256、grep：200、glob：500）
- [ ] 截断插入可见标记：`[警告：工具输出已被截断。删除了 N 个字符...]`
- [ ] 完整的未截断输出通过 `TOOL_CALL_END` 事件可用
- [ ] 默认字符限制与第 5.2 节中的表匹配（read_file：50k、shell：30k、grep：20k 等）
- [ ] 字符和行限制都可通过 `SessionConfig` 覆盖

### 9.6 引导

- [ ] `steer()` 将消息排队，在当前工具回合后注入
- [ ] `follow_up()` 将消息排队，在当前输入完成后处理
- [ ] 引导消息作为历史记录中的 SteeringTurn 出现
- [ ] SteeringTurn 被转换为 LLM 的用户角色消息

### 9.7 推理工作量

- [ ] `reasoning_effort` 被传递到 LLM SDK Request
- [ ] 在会话中途更改 `reasoning_effort` 在下次 LLM 调用时生效
- [ ] 有效值："low"、"medium"、"high"、null（提供商默认值）（某些提供商可能有其他选项，如 `xhigh`）

### 9.8 系统提示词

- [ ] 系统提示词包括特定于提供商的基本指令
- [ ] 系统提示词包括环境上下文（平台、git、工作目录、日期、模型信息）
- [ ] 系统提示词包括来自活动配置文件的工具描述
- [ ] 项目文档文件（AGENTS.md + 特定于提供商的文件）被发现并包含
- [ ] 用户指令覆盖最后追加（最高优先级）
- [ ] 仅加载相关的项目文件（例如，Anthropic 配置文件加载 CLAUDE.md，而不是 GEMINI.md）

### 9.9 子代理

- [ ] 可以通过 `spawn_agent` 工具使用范围任务生成子代理
- [ ] 子代理共享父代的执行环境（相同的文件系统）
- [ ] 子代理维护独立的对话历史
- [ ] 深度限制防止递归生成（默认最大深度：1）
- [ ] 子代理结果作为工具结果返回给父代
- [ ] `send_input`、`wait` 和 `close_agent` 工具正常工作

### 9.10 事件系统

- [ ] 第 2.9 节中列出的所有事件类型都在正确的时间发出
- [ ] 事件通过异步迭代器或语言适当的等效项传递
- [ ] `TOOL_CALL_END` 事件携带完整的未截断工具输出
- [ ] 会话生命周期事件（SESSION_START、SESSION_END）括起会话

### 9.11 错误处理

- [ ] 工具执行错误 -> 向 LLM 发送错误结果（模型可以恢复）
- [ ] LLM API 瞬态错误（429、500-503）-> 使用退避重试（由统一 LLM SDK 层处理）
- [ ] 身份验证错误 -> 立即显示，不重试，会话转换到 CLOSED
- [ ] 上下文窗口溢出 -> 发出警告事件（无自动压缩）
- [ ] 优雅关闭：中止信号 -> 取消 LLM 流 -> 终止正在运行的进程 -> 刷新事件 -> 发出 SESSION_END

### 9.12 跨提供商同等矩阵

运行此验证矩阵——每个单元格都必须通过：

| 测试用例                                    | OpenAI | Anthropic | Gemini |
|---------------------------------------------|--------|-----------|--------|
| 简单文件创建任务                             | [ ]    | [ ]       | [ ]    |
| 读取文件，然后编辑它                         | [ ]    | [ ]       | [ ]    |
| 一个会话中的多文件编辑                       | [ ]    | [ ]       | [ ]    |
| Shell 命令执行                               | [ ]    | [ ]       | [ ]    |
| Shell 命令超时处理                           | [ ]    | [ ]       | [ ]    |
| Grep + glob 查找文件                         | [ ]    | [ ]       | [ ]    |
| 多步骤任务（读取 -> 分析 -> 编辑）           | [ ]    | [ ]       | [ ]    |
| 工具输出截断（大文件）                       | [ ]    | [ ]       | [ ]    |
| 并行工具调用（如果支持）                      | [ ]    | [ ]       | [ ]    |
| 任务中途引导                                 | [ ]    | [ ]       | [ ]    |
| 推理工作量更改                               | [ ]    | [ ]       | [ ]    |
| 子代理生成和等待                             | [ ]    | [ ]       | [ ]    |
| 循环检测触发警告                             | [ ]    | [ ]       | [ ]    |
| 错误恢复（工具失败，模型重试）               | [ ]    | [ ]       | [ ]    |
| 特定于提供商的编辑格式工作                   | [ ]    | [ ]       | [ ]    |

### 9.13 集成冒烟测试

使用真实 API 密钥的端到端测试：

```
FOR EACH profile IN [openai_profile, anthropic_profile, gemini_profile]:
    env = LocalExecutionEnvironment(working_dir = temp_directory())
    session = Session(profile, env)

    -- 1. 简单文件创建
    session.submit("创建一个名为 hello.py 的文件，打印 'Hello World'")
    ASSERT env.file_exists("hello.py")
    ASSERT env.read_file("hello.py") CONTAINS "Hello"

    -- 2. 读取和编辑
    session.submit("读取 hello.py 并添加第二条打印语句，说 'Goodbye'")
    content = env.read_file("hello.py")
    ASSERT content CONTAINS "Hello"
    ASSERT content CONTAINS "Goodbye"

    -- 3. Shell 执行
    session.submit("运行 hello.py 并显示输出")
    -- 验证代理执行了命令（检查事件流中的 shell 工具调用）

    -- 4. 截断验证
    env.write_file("big.txt", REPEAT("x", 100000))
    session.submit("读取 big.txt")
    -- 验证 TOOL_CALL_END 事件有完整的 100k 字符
    -- 验证发送到 LLM 的 ToolResult 有截断标记

    -- 5. 引导
    session.submit("创建一个带有多个路由的 Flask Web 应用程序")
    session.steer("实际上，现在只创建一个 /health 端点")
    -- 验证代理调整了其方法

    -- 6. 子代理
    session.submit("生成一个子代理为 hello.py 编写测试，然后审查其输出")
    -- 验证子代理工具调用出现在事件流中

    -- 7. 超时处理
    session.submit("使用默认超时运行 'sleep 30'")
    -- 验证命令在 10 秒后超时，并且代理优雅地处理它
```

---

## 附录 A：apply_patch v4a 格式参考

`apply_patch` 工具（由 OpenAI 配置文件使用）接受 v4a 格式的补丁。此格式支持在单个补丁中创建、删除、更新和重命名文件。

### 语法

```
patch       = "*** Begin Patch\n" operations "*** End Patch\n"
operations  = (add_file | delete_file | update_file)*

add_file    = "*** Add File: " path "\n" added_lines
delete_file = "*** Delete File: " path "\n"
update_file = "*** Update File: " path "\n" [move_line] hunks

move_line   = "*** Move to: " new_path "\n"
added_lines = ("+" line "\n")*
hunks       = hunk+
hunk        = "@@ " [context_hint] "\n" hunk_lines
hunk_lines  = (context_line | delete_line | add_line)+
context_line = " " line "\n"           -- 空格前缀 = 未更改的行
delete_line  = "-" line "\n"           -- 减号前缀 = 删除此行
add_line     = "+" line "\n"           -- 加号前缀 = 添加此行
eof_marker   = "*** End of File\n"     -- 可选，标记最后一个 hunk 的结束
```

### 操作

**添加文件：**创建一个新文件。所有行都以 `+` 为前缀。
```
*** Begin Patch
*** Add File: src/utils/helpers.py
+def greet(name):
+    return f"Hello, {name}!"
*** End Patch
```

**删除文件：**完全删除文件。
```
*** Begin Patch
*** Delete File: src/old_module.py
*** End Patch
```

**更新文件：**使用基于上下文的 hunk 修改现有文件。
```
*** Begin Patch
*** Update File: src/main.py
@@ def main():
     print("Hello")
-    return 0
+    print("World")
+    return 1
*** End Patch
```

**更新 + 重命名：**在一个操作中修改和重命名。
```
*** Begin Patch
*** Update File: old_name.py
*** Move to: new_name.py
@@ import os
 import sys
-import old_dep
+import new_dep
*** End Patch
```

### Hunk 匹配

`@@` 行提供上下文提示（通常是函数签名或更改附近的可识别行）。实现使用此提示加上上下文行（带空格前缀）来定位文件中的正确位置。约定：在每个更改上方和下方显示 3 行上下文。

当精确匹配失败时，实现应在报告错误之前尝试模糊匹配（空白规范化、Unicode 标点等效）。

### 多 Hunk 更新

单个更新文件块可以包含多个 `@@` hunk：

```
*** Begin Patch
*** Update File: src/config.py
@@ DEFAULT_TIMEOUT = 30
-DEFAULT_TIMEOUT = 30
+DEFAULT_TIMEOUT = 60
@@ def load_config():
     config = {}
-    config["debug"] = False
+    config["debug"] = True
*** End Patch
```

---

## 附录 B：错误处理

### 工具级别错误

工具执行错误被代理捕获并作为错误结果发送到 LLM（`is_error = true`）。这给了模型恢复、重试或尝试不同方法的机会。

| 错误类型          | 示例                                      | 恢复                         |
|-------------------|-------------------------------------------|------------------------------|
| FileNotFound      | 对不存在的路径进行 read_file              | 模型可以搜索正确路径         |
| EditConflict      | 未找到 old_string 或不唯一                | 模型可以读取文件并重试       |
| ShellExitError    | 命令返回非零退出代码                      | 模型可以检查输出并修复       |
| ShellTimeout      | 命令超过 timeout_ms                       | 模型可以使用更长的超时重试   |
| PermissionDenied  | 写入受保护的路径                           | 模型可以选择不同的路径       |
| ValidationError   | 工具的无效 JSON 参数                      | 模型可以修复参数             |
| UnknownTool       | 模型调用了注册表中没有的工具              | 错误结果告诉模型工具名称错误 |

### 会话级别错误

这些错误影响会话本身，而不是单个工具调用。

| 错误类型              | 可重试 | 行为                                         |
|-----------------------|--------|----------------------------------------------|
| ProviderError (429)   | 是     | 使用退避重试（由统一 LLM SDK 处理）          |
| ProviderError (500-503)| 是     | 使用退避重试（由统一 LLM SDK 处理）          |
| AuthenticationError   | 否     | 立即显示，会话 -> CLOSED                     |
| ContextLengthError    | 否     | 发出警告，会话 -> CLOSED                     |
| NetworkError          | 是     | 使用退避重试（由统一 LLM SDK 处理）          |
| TurnLimitExceeded     | 否     | 发出 TURN_LIMIT 事件，会话 -> IDLE           |

### 优雅关闭序列

当触发中止信号或发生不可恢复的错误时：

```
1. 取消任何进行中的 LLM 流
2. 向所有正在运行的命令进程组发送 SIGTERM
3. 等待 2 秒
4. 向任何剩余进程发送 SIGKILL
5. 刷新待处理的事件
6. 发出带有最终状态的 SESSION_END 事件
7. 清理子代理（在所有活动子代理上 close_agent）
8. 将会话转换到 CLOSED
```

---

## 附录 C：设计决策基本原理

**为什么提供商对齐的工具集而不是通用工具集？** 模型可能针对特定工具格式进行了训练。GPT-5.2-codex 在 apply_patch 上进行了训练；强迫它使用 old_string/new_string 编辑会产生更差的结果。Claude 在 old_string/new_string 上进行了训练；强迫它使用 apply_patch 会产生更差的结果。每个提供商配置文件的初始基础应该是该提供商参考代理的确切系统提示词和工具——不是类似的提示词，不是类似的工具，而是原始提示词和工具定义的 1:1 逐字节副本作为起点。然后从那里扩展。从原生工具集开始可提供最佳的基线体验，因为模型正是针对该工具进行评估和优化的。

**为什么可扩展执行环境而不是固定的本地实现？** 只能在本地计算机上运行的编码代理是有限的。通过将工具执行抽象在接口后面，相同的代理逻辑可以在 Docker（用于沙箱）、Kubernetes（用于云执行）、SSH（用于远程开发）或 WASM（用于基于浏览器的代理）中工作。抽象几乎不需要复杂性成本，但开辟了主要的部署灵活性。

**为什么头/尾截断而不是仅从末尾截断？** 文件的开头（导入、类型定义、模块文档字符串）或命令输出的开头（启动消息、标头）通常与结尾（最终结果）一样重要。头/尾分割保留两者。显式截断标记准确地告诉模型发生了什么，以便它可以在需要时请求特定部分。

**为什么不自动压缩？** 压缩（总结对话历史以释放上下文空间）是复杂的、有损的和特定于实现的。不同的主机应用程序有不同的要求——CLI 可能希望激进的压缩，IDE 可能希望重新启动会话，批处理系统可能希望失败。规范提供了上下文窗口感知信号，并将策略留给主机应用程序。

**为什么循环使用 Client.complete() 而不是 SDK 的 generate()？** SDK 的 `generate()` 函数有自己的工具执行循环，但它不处理：带有显式标记的输出截断、在工具回合之间注入引导消息、用于 UI 渲染的事件发射、每个工具的超时强制执行、循环检测或执行环境抽象。代理循环需要所有这些，因此它使用较低级别的 `Client.complete()` 管理自己的循环。

**为什么 10 秒默认命令超时？** 这与 codex-rs 匹配。大多数开发人员命令（编译、lint、测试单个文件、git 操作）在 10 秒内完成。长时间运行的命令（完整测试套件、构建）应明确请求更长的超时。默认值保护 against 异常进程，而不会太短导致正常操作失败。

**为什么默认排除敏感环境变量？** 环境中的 API 密钥、机密和令牌不应被 LLM 可见（它可能在响应中包含它们或记录它们）。默认值排除 `*_API_KEY`、`*_SECRET`、`*_TOKEN`、`*_PASSWORD` 模式。这是一个安全默认值，而不是安全边界——如果需要，代理仍然可以运行通过 shell 自己的环境访问这些变量的命令。
