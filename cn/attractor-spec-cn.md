# Attractor 规范

基于 DOT 的流水线执行器，使用有向图（使用 Graphviz DOT 语法定义）来编排多阶段 AI 工作流。图中的每个节点都是一个 AI 任务（LLM 调用、人工审核、条件分支、并行分发等），边定义了它们之间的流转关系。

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [DOT DSL 模式](#2-dot-dsl-模式)
3. [流水线执行引擎](#3-流水线执行引擎)
4. [节点处理器](#4-节点处理器)
5. [状态与上下文](#5-状态与上下文)
6. [人机协作（面试官模式）](#6-人机协作面试官模式)
7. [验证与检查](#7-验证与检查)
8. [模型样式表](#8-模型样式表)
9. [转换与扩展性](#9-转换与扩展性)
10. [条件表达式语言](#10-条件表达式语言)
11. [完成定义](#11-完成定义)

---

## 1. 概述与目标

### 1.1 问题陈述

AI 驱动的软件工作流——代码生成、代码审查、测试、部署规划——通常需要多个 LLM 调用，这些调用通过条件逻辑、人工审批和并行执行串联在一起。如果没有结构化的编排层，开发者要么编写脆弱的命令式脚本，要么构建临时的状态机，这些状态机难以可视化、版本控制或调试。

Attractor 通过让流水线作者使用 Graphviz DOT 语法定义多阶段 AI 工作流为有向图来解决这个问题。图即工作流：节点是任务，边是转换，属性配置行为。结果是一个声明式的、可视化的、可版本控制的流水线定义，执行引擎可以确定性遍历。

### 1.2 为何选择 DOT 语法

选择 DOT 作为流水线定义格式有以下几个原因：

- **DOT 本质上是一种图描述语言。** 工作流流水线是有向图。使用 DOT 意味着结构（节点和边）直接映射到语言的主要构造，而不是编码在 YAML 或 JSON 等没有图原生概念的数据格式中。
- **现有工具。** DOT 文件可以使用标准 Graphviz 工具渲染为 SVG/PNG，为流水线作者提供即时的可视化反馈。编辑器、检查器和解析器已经存在。
- **声明式且可读。** `.dot` 文件是一个完整的、自包含的工作流定义，可以进行版本控制、差异比较和在拉取请求中审查。
- **受限的可扩展性。** 通过限制为定义良好的 DOT 子集（仅限有向图、类型化属性、无 HTML 标签），DSL 保持可预测，同时可以通过自定义属性进行扩展。

有关 DOT 语法的参考，请参阅 Graphviz DOT 语言规范：https://graphviz.org/doc/info/lang.html

### 1.3 设计原则

**声明式流水线。** `.dot` 文件声明工作流的外观以及每个阶段应该做什么。执行引擎决定如何以及何时运行每个阶段。流水线作者不编写控制流；他们声明图结构。

**可插拔处理器。** 每个节点类型（LLM 调用、人工门控、并行分发）都由实现公共接口的处理器支持。通过注册新处理器来添加新节点类型。执行引擎不了解处理器内部。

**检查点与恢复。** 每个节点完成后，执行引擎保存可序列化的检查点。如果进程崩溃，执行将从最后一个检查点恢复。

**人机协作。** 流水线可以在指定节点暂停，向人工操作员展示选择，并根据人工的决定进行路由。这支持审批门控、代码审查和手动覆盖——对于自动化判断可能不足的 AI 工作流至关重要。

**基于边的路由。** 节点之间的转换由边上的条件、标签和权重控制，具有运行时条件评估。

### 1.4 分层与 LLM 后端

Attractor 定义了编排层：图定义、遍历、状态管理和可扩展性。它不要求任何特定的 LLM 集成。codergen 处理器（第 4.5 节）需要一种调用 LLM 并获取响应的方法——如何提供这一点取决于您。

codergen 处理器接受符合 `CodergenBackend` 接口的后端（第 4.5 节）。该后端内部做什么完全由实现者决定——使用配套的 [Coding Agent Loop](./coding-agent-loop-spec.md) 和 [Unified LLM Client](./unified-llm-spec.md) 规范、在子进程中生成 CLI 代理（Claude Code、Codex、Gemini CLI）、在 tmux 窗格中运行代理并由管理器附加、直接调用 LLM API 或其他任何方式。流水线定义（DOT 文件）不会因后端选择而改变。

Attractor 流水线由事件流驱动（第 9.6 节）。TUI、Web 和 IDE 前端消费事件并提交人机协作答案。流水线引擎是无头的；表示层是分离的。

---

## 2. DOT DSL 模式

### 2.1 支持的子集

Attractor 接受 Graphviz DOT 语言的严格子集。这些限制的存在是为了可预测性：每个文件一个图、仅限有向边、无 HTML 标签、具有默认值的类型化属性。

### 2.2 BNF 风格语法

```
Graph           ::= 'digraph' Identifier '{' Statement* '}'

Statement       ::= GraphAttrStmt
                   | NodeDefaults
                   | EdgeDefaults
                   | SubgraphStmt
                   | NodeStmt
                   | EdgeStmt
                   | GraphAttrDecl

GraphAttrStmt   ::= 'graph' AttrBlock ';'?
NodeDefaults    ::= 'node' AttrBlock ';'?
EdgeDefaults    ::= 'edge' AttrBlock ';'?
GraphAttrDecl   ::= Identifier '=' Value ';'?

SubgraphStmt    ::= 'subgraph' Identifier? '{' Statement* '}'

NodeStmt        ::= Identifier AttrBlock? ';'?
EdgeStmt        ::= Identifier ( '->' Identifier )+ AttrBlock? ';'?

AttrBlock       ::= '[' Attr ( ',' Attr )* ']'
Attr            ::= Key '=' Value

Key             ::= Identifier | QualifiedId
QualifiedId     ::= Identifier ( '.' Identifier )+

Value           ::= String | Integer | Float | Boolean | Duration
Identifier      ::= [A-Za-z_][A-Za-z0-9_]*
String          ::= '"' ( '\\"' | '\\n' | '\\t' | '\\\\' | [^"\\] )* '"'
Integer         ::= '-'? [0-9]+
Float           ::= '-'? [0-9]* '.' [0-9]+
Boolean         ::= 'true' | 'false'
Duration        ::= Integer ( 'ms' | 's' | 'm' | 'h' | 'd' )

Direction       ::= 'TB' | 'LR' | 'BT' | 'RL'
```

### 2.3 关键约束

- **每个文件一个有向图。** 拒绝多个图、无向图和 `strict` 修饰符。
- **节点 ID 使用裸标识符。** 节点 ID 必须匹配 `[A-Za-z_][A-Za-z0-9_]*`。人类可读的名称放在 `label` 属性中。
- **属性之间需要逗号。** 在属性块内，逗号分隔键值对以进行明确解析。
- **仅限有向边。** `->` 是唯一的边运算符。`--`（无向）被拒绝。
- **支持注释。** 在解析之前剥离 `// 行` 和 `/* 块 */` 注释。
- **分号可选。** 接受语句终止分号但不是必需的。

### 2.4 值类型

| 类型     | 语法                          | 示例                             |
|----------|--------------------------------|----------------------------------|
| String   | 双引号带转义字符              | `"Hello world"`, `"line1\nline2"`    |
| Integer  | 可选符号加数字                | `42`, `-1`, `0`                      |
| Float    | 小数                          | `0.5`, `-3.14`                       |
| Boolean  | 字面关键字                    | `true`, `false`                      |
| Duration | 整数加单位后缀                | `900s`, `15m`, `2h`, `250ms`, `1d`   |

### 2.5 图级属性

图属性在 `graph [ ... ]` 块中声明或作为顶级 `key = value` 声明。它们配置整个工作流。

| Key                       | Type     | Default   | Description |
|---------------------------|----------|-----------|-------------|
| `goal`                    | String   | `""`      | 流水线的人类可读目标。在提示模板中公开为 `$goal`，并在运行上下文中镜像为 `graph.goal`。 |
| `label`                   | String   | `""`      | 图的显示名称（用于可视化）。 |
| `model_stylesheet`        | String   | `""`      | 用于每节点 LLM 模型/提供商默认值的类 CSS 样式表。见第 8 节。 |
| `default_max_retry`       | Integer  | `50`      | 省略 `max_retries` 的节点的全局重试上限。 |
| `retry_target`            | String   | `""`      | 如果在目标门未满足时到达出口，则跳转到的节点 ID。 |
| `fallback_retry_target`   | String   | `""`      | 如果 `retry_target` 缺失或无效的辅助跳转目标。 |
| `default_fidelity`        | String   | `""`      | 默认上下文保真度模式（见第 5.4 节）。 |

### 2.6 节点属性

| Key                 | Type     | Default         | Description |
|---------------------|----------|-----------------|-------------|
| `label`             | String   | node ID         | 在 UI、提示和遥测中显示的名称。 |
| `shape`             | String   | `"box"`         | Graphviz 形状。确定默认处理器类型（见下面的映射表）。 |
| `type`              | String   | `""`            | 显式处理器类型覆盖。优先于基于形状的解析。 |
| `prompt`            | String   | `""`            | 阶段的主要指令。支持 `$goal` 变量扩展。对于 LLM 阶段，如果为空则回退到 `label`。 |
| `max_retries`       | Integer  | `0`             | 除初始执行外的额外尝试次数。`max_retries=3` 意味着最多执行 4 次。 |
| `goal_gate`         | Boolean  | `false`         | 如果为 `true`，此节点必须在流水线退出前达到 SUCCESS。 |
| `retry_target`      | String   | `""`            | 如果此节点失败且重试耗尽，则跳转到的节点 ID。 |
| `fallback_retry_target` | String | `""`          | 辅助重试目标。 |
| `fidelity`          | String   | inherited       | 此节点 LLM 会话的上下文保真度模式。见第 5.4 节。 |
| `thread_id`         | String   | derived         | 用于在 `full` 保真度下重用 LLM 会话的显式线程标识符。 |
| `class`             | String   | `""`            | 用于模型样式表定位的逗号分隔类名。 |
| `timeout`           | Duration | unset           | 此节点的最大执行时间。 |
| `llm_model`         | String   | inherited       | LLM 模型标识符。可被样式表覆盖。 |
| `llm_provider`      | String   | auto-detected   | LLM 提供商密钥。如果未设置，则从模型自动检测。 |
| `reasoning_effort`  | String   | `"high"`        | LLM 推理强度：`low`、`medium`、`high`。 |
| `auto_status`       | Boolean  | `false`         | 如果为 `true` 且处理器未写入状态，引擎自动生成 SUCCESS 结果。 |
| `allow_partial`     | Boolean  | `false`         | 在重试耗尽时接受 PARTIAL_SUCCESS 而不是失败。 |

### 2.7 边属性

| Key          | Type     | Default | Description |
|--------------|----------|---------|-------------|
| `label`      | String   | `""`    | 面向人类的标题和路由键。用于边选择中的首选标签匹配。 |
| `condition`  | String   | `""`    | 针对当前上下文和结果评估的布尔保护表达式。见第 10 节。 |
| `weight`     | Integer  | `0`     | 边选择的数字优先级。在同等符合条件的边中，权重越高者获胜。 |
| `fidelity`   | String   | unset   | 覆盖目标节点的保真度模式。在保真度解析中具有最高优先级。 |
| `thread_id`  | String   | unset   | 覆盖目标节点会话重用的线程 ID。 |
| `loop_restart` | Boolean | `false` | 当为 `true` 时，终止当前运行并使用新的日志目录重新启动。 |

### 2.8 形状到处理器类型映射

节点上的 `shape` 属性确定哪个处理器执行它，除非被显式的 `type` 属性覆盖。此表定义了规范映射：

| Shape             | Handler Type          | Description |
|-------------------|-----------------------|-------------|
| `Mdiamond`        | `start`               | 流水线入口点。无操作处理器。每个图必须恰好有一个。 |
| `Msquare`         | `exit`                | 流水线出口点。无操作处理器。每个图必须恰好有一个。 |
| `box`             | `codergen`            | LLM 任务（代码生成、分析、规划）。所有没有显式形状的节点的默认值。 |
| `hexagon`         | `wait.human`          | 人机协作门控。阻塞直到人工选择一个选项。 |
| `diamond`         | `conditional`         | 条件路由点。基于针对当前上下文的边条件进行路由。 |
| `component`       | `parallel`            | 并行分发。并发执行多个分支。 |
| `tripleoctagon`   | `parallel.fan_in`     | 并行汇聚。等待所有分支并合并结果。 |
| `parallelogram`   | `tool`                | 外部工具执行（shell 命令、API 调用）。 |
| `house`           | `stack.manager_loop`  | 主管循环。编排子流水线的观察/引导/等待周期。 |

### 2.9 链式边

链式边声明是语法糖。语句：

```
A -> B -> C [label="next"]
```

扩展为两条边：

```
A -> B [label="next"]
B -> C [label="next"]
```

链式声明中的边属性应用于链中的所有边。

### 2.10 子图

子图有两个用途：**作用域默认值**和**派生类**用于模型样式表。

**作用域默认值：** 在子图的 `node [ ... ]` 块中声明的属性应用于该子图内的节点，除非节点显式覆盖它们。

```
subgraph cluster_loop {
    label = "Loop A"
    node [thread_id="loop-a", timeout="900s"]

    Plan      [label="Plan next step"]
    Implement [label="Implement", timeout="1800s"]
}
```

这里 `Plan` 继承 `thread_id="loop-a"` 和 `timeout="900s"`，而 `Implement` 继承 `thread_id` 但覆盖 `timeout`。

**类派生：** 子图标签可以为模型样式表匹配生成类 CSS 的类。子图内的节点接收派生的类。类名通过小写标签、用连字符替换空格并剥离非字母数字字符（连字符除外）派生。例如，`label="Loop A"` 生成类 `loop-a`。

### 2.11 节点和边默认块

默认块在其作用域内为所有后续节点或边设置基线属性：

```
node [shape=box, timeout="900s"]
edge [weight=0]
```

单个节点或边上的显式属性覆盖这些默认值。

### 2.12 类属性

`class` 属性为一个或多个类 CSS 的类名分配给节点，用于模型样式表定位：

```
review_code [shape=box, class="code,critical", prompt="Review the code"]
```

类用逗号分隔。它们可以在模型样式表中用点前缀选择器（`.code`、`.critical`）引用。

### 2.13 最小示例

**简单线性工作流：**

```
digraph Simple {
    graph [goal="Run tests and report"]
    rankdir=LR

    start [shape=Mdiamond, label="Start"]
    exit  [shape=Msquare, label="Exit"]

    run_tests [label="Run Tests", prompt="Run the test suite and report results"]
    report    [label="Report", prompt="Summarize the test results"]

    start -> run_tests -> report -> exit
}
```

**带条件的分支工作流：**

```
digraph Branch {
    graph [goal="Implement and validate a feature"]
    rankdir=LR
    node [shape=box, timeout="900s"]

    start     [shape=Mdiamond, label="Start"]
    exit      [shape=Msquare, label="Exit"]
    plan      [label="Plan", prompt="Plan the implementation"]
    implement [label="Implement", prompt="Implement the plan"]
    validate  [label="Validate", prompt="Run tests"]
    gate      [shape=diamond, label="Tests passing?"]

    start -> plan -> implement -> validate -> gate
    gate -> exit      [label="Yes", condition="outcome=success"]
    gate -> implement [label="No", condition="outcome!=success"]
}
```

**人工门控：**

```
digraph Review {
    rankdir=LR

    start [shape=Mdiamond, label="Start"]
    exit  [shape=Msquare, label="Exit"]

    review_gate [
        shape=hexagon,
        label="Review Changes",
        type="wait.human"
    ]

    start -> review_gate
    review_gate -> ship_it [label="[A] Approve"]
    review_gate -> fixes   [label="[F] Fix"]
    ship_it -> exit
    fixes -> review_gate
}
```

---

## 3. 流水线执行引擎

### 3.1 运行生命周期

执行生命周期经历五个阶段：

```
PARSE -> VALIDATE -> INITIALIZE -> EXECUTE -> FINALIZE
```

1. **解析：** 读取 `.dot` 源并生成内存中的图模型（节点、边、属性）。
2. **验证：** 运行检查规则（第 7 节）。拒绝无效的图。对可疑模式发出警告。
3. **初始化：** 创建运行目录、初始上下文和检查点。将图属性镜像到上下文中。应用转换（样式表、变量扩展）。
4. **执行：** 从起始节点遍历图，执行处理器并选择边。
5. **完成：** 写入最终检查点、发出完成事件并清理资源（关闭会话、释放文件）。

### 3.2 核心执行循环

以下伪代码定义了执行引擎的遍历算法。这是系统的核心。

```
FUNCTION run(graph, config):
    context = new Context()
    mirror_graph_attributes(graph, context)
    checkpoint = new Checkpoint()
    completed_nodes = []
    node_outcomes = {}

    current_node = find_start_node(graph)
        -- 解析方式：(1) shape=Mdiamond, (2) id="start" 或 "Start"
        -- 如果未找到则引发错误

    WHILE true:
        node = graph.nodes[current_node.id]

        -- 步骤 1：检查终端节点
        IF is_terminal(node):
            gate_ok, failed_gate = check_goal_gates(graph, node_outcomes)
            IF NOT gate_ok AND failed_gate exists:
                retry_target = get_retry_target(failed_gate, graph)
                IF retry_target exists:
                    current_node = graph.nodes[retry_target]
                    CONTINUE
                ELSE:
                    RAISE "Goal gate unsatisfied and no retry target"
            BREAK  -- 退出循环；流水线完成

        -- 步骤 2：使用重试策略执行节点处理器
        retry_policy = build_retry_policy(node, graph)
        outcome = execute_with_retry(node, context, graph, retry_policy)

        -- 步骤 3：记录完成
        completed_nodes.append(node.id)
        node_outcomes[node.id] = outcome

        -- 步骤 4：应用结果中的上下文更新
        FOR EACH (key, value) IN outcome.context_updates:
            context.set(key, value)
        context.set("outcome", outcome.status)
        IF outcome.preferred_label is not empty:
            context.set("preferred_label", outcome.preferred_label)

        -- 步骤 5：保存检查点
        checkpoint = create_checkpoint(context, current_node.id, completed_nodes)
        save_checkpoint(checkpoint, logs_root)

        -- 步骤 6：选择下一条边
        next_edge = select_edge(node, outcome, context, graph)
        IF next_edge is NONE:
            IF outcome.status == FAIL:
                RAISE "Stage failed with no outgoing fail edge"
            BREAK

        -- 步骤 7：处理 loop_restart
        IF next_edge has loop_restart=true:
            restart_run(graph, config, start_at=next_edge.target)
            RETURN

        -- 步骤 8：前进到下一个节点
        current_node = graph.nodes[next_edge.to_node]

    RETURN last_outcome
```

### 3.3 边选择算法

节点完成后，引擎从节点的出边中选择下一条边。选择是确定性的，遵循五步优先级顺序：

**步骤 1：条件匹配边。** 针对当前上下文和结果评估每条边的 `condition` 表达式（见第 10 节）。条件评估为 `true` 的边是符合条件的。没有条件的边在此步骤中不考虑；它们进入后续步骤。

**步骤 2：首选标签匹配。** 如果节点的结果包含 `preferred_label`，查找第一个符合条件的边（通过条件或无条件）的 `label` 在规范化后匹配。标签规范化：小写、修剪空白、剥离加速器前缀（如 `[Y] `、`Y) `、`Y - ` 等模式）。

**步骤 3：建议的下一个 ID。** 如果没有标签匹配且结果包含 `suggested_next_ids`，查找第一个符合条件的边，其目标节点 ID 出现在列表中。

**步骤 4：最高权重。** 在剩余符合条件的无条件边中，选择具有最高 `weight` 属性的边（默认为 0）。

**步骤 5：词法决胜。** 如果权重相等，选择目标节点 ID 在词法上排在第一位的边。

```
FUNCTION select_edge(node, outcome, context, graph):
    edges = graph.outgoing_edges(node.id)
    IF edges is empty:
        RETURN NONE

    -- 步骤 1：条件匹配
    condition_matched = []
    FOR EACH edge IN edges:
        IF edge.condition is not empty:
            IF evaluate_condition(edge.condition, outcome, context) == true:
                condition_matched.append(edge)
    IF condition_matched is not empty:
        RETURN best_by_weight_then_lexical(condition_matched)

    -- 步骤 2：首选标签
    IF outcome.preferred_label is not empty:
        FOR EACH edge IN edges:
            IF normalize_label(edge.label) == normalize_label(outcome.preferred_label):
                RETURN edge

    -- 步骤 3：建议的下一个 ID
    IF outcome.suggested_next_ids is not empty:
        FOR EACH suggested_id IN outcome.suggested_next_ids:
            FOR EACH edge IN edges:
                IF edge.to_node == suggested_id:
                    RETURN edge

    -- 步骤 4 & 5：带词法决胜的权重（仅无条件边）
    unconditional = [e FOR e IN edges WHERE e.condition is empty]
    IF unconditional is not empty:
        RETURN best_by_weight_then_lexical(unconditional)

    -- 回退：任何边
    RETURN best_by_weight_then_lexical(edges)


FUNCTION best_by_weight_then_lexical(edges):
    SORT edges BY (weight DESCENDING, to_node ASCENDING)
    RETURN edges[0]
```

### 3.4 目标门强制执行

具有 `goal_gate=true` 的节点表示必须在流水线退出前成功的关键阶段。当遍历到达终端节点（shape=Msquare）时：

1. 检查所有已访问的具有 `goal_gate=true` 的节点。
2. 如果任何目标门节点具有非成功结果（不是 SUCCESS 或 PARTIAL_SUCCESS），流水线无法退出。
3. 相反，跳转到未满足目标门节点的 `retry_target`。如果未设置，尝试 `fallback_retry_target`。如果也未设置，尝试图级的 `retry_target` 和 `fallback_retry_target`。
4. 如果在任何级别都不存在重试目标，流水线失败并显示错误。

```
FUNCTION check_goal_gates(graph, node_outcomes):
    FOR EACH (node_id, outcome) IN node_outcomes:
        node = graph.nodes[node_id]
        IF node.goal_gate == true:
            IF outcome.status NOT IN {SUCCESS, PARTIAL_SUCCESS}:
                RETURN (false, node)
    RETURN (true, NONE)
```

### 3.5 重试逻辑

每个节点都有一个重试策略，由以下因素决定：

1. 节点属性 `max_retries`（如果设置）-- 初始执行之外的额外尝试次数
2. 图属性 `default_max_retry`（回退）
3. 内置默认值：0（无重试）

`max_retries` 属性指定额外尝试。因此 `max_retries=3` 意味着总共执行 4 次（1 次初始 + 3 次重试）。内部映射到 `max_attempts = max_retries + 1`。

```
FUNCTION execute_with_retry(node, context, graph, retry_policy):
    FOR attempt FROM 1 TO retry_policy.max_attempts:
        TRY:
            outcome = handler.execute(node, context, graph, logs_root)
        CATCH exception:
            IF retry_policy.should_retry(exception) AND attempt < retry_policy.max_attempts:
                delay = retry_policy.backoff.delay_for_attempt(attempt)
                sleep(delay)
                CONTINUE
            ELSE:
                RETURN Outcome(status=FAIL, failure_reason=str(exception))

        IF outcome.status IN {SUCCESS, PARTIAL_SUCCESS}:
            reset_retry_counter(node.id)
            RETURN outcome

        IF outcome.status == RETRY:
            IF attempt < retry_policy.max_attempts:
                increment_retry_counter(node.id)
                delay = retry_policy.backoff.delay_for_attempt(attempt)
                sleep(delay)
                CONTINUE
            ELSE:
                IF node.allow_partial == true:
                    RETURN Outcome(status=PARTIAL_SUCCESS, notes="retries exhausted, partial accepted")
                RETURN Outcome(status=FAIL, failure_reason="max retries exceeded")

        IF outcome.status == FAIL:
            RETURN outcome

    RETURN Outcome(status=FAIL, failure_reason="max retries exceeded")
```

### 3.6 重试策略

```
RetryPolicy:
    max_attempts    : Integer         -- 最小值 1（1 表示无重试）
    backoff         : BackoffConfig   -- 重试之间的延迟计算
    should_retry    : Function(Error) -> Boolean  -- 可重试错误的谓词

BackoffConfig:
    initial_delay_ms  : Integer   -- 首次重试延迟（毫秒）（默认：200）
    backoff_factor    : Float     -- 后续延迟的乘数（默认：2.0）
    max_delay_ms      : Integer   -- 延迟上限（毫秒）（默认：60000）
    jitter            : Boolean   -- 添加随机抖动以防止惊群效应（默认：true）
```

**延迟计算：**

```
FUNCTION delay_for_attempt(attempt, config):
    -- attempt 是 1 索引的（首次重试是 attempt=1）
    delay = config.initial_delay_ms * (config.backoff_factor ^ (attempt - 1))
    delay = MIN(delay, config.max_delay_ms)
    IF config.jitter:
        delay = delay * random_uniform(0.5, 1.5)
    RETURN delay
```

**预设策略：**

| Name         | Max Attempts | Initial Delay | Factor | Description |
|--------------|-------------|---------------|--------|-------------|
| `none`       | 1           | --            | --     | 无重试。出错时立即失败。 |
| `standard`   | 5           | 200ms         | 2.0    | 通用。延迟：200, 400, 800, 1600, 3200ms。 |
| `aggressive`  | 5           | 500ms         | 2.0    | 用于不可靠的操作。延迟：500, 1000, 2000, 4000, 8000ms。 |
| `linear`     | 3           | 500ms         | 1.0    | 尝试之间固定延迟。延迟：500, 500, 500ms。 |
| `patient`    | 3           | 2000ms        | 3.0    | 长时间运行的操作。延迟：2000, 6000, 18000ms。 |

**默认 should_retry 谓词：** 对网络错误、速率限制错误（HTTP 429）、服务器错误（HTTP 5xx）和提供商报告的瞬态故障返回 `true`。对身份验证错误（HTTP 401、403）、错误请求错误（HTTP 400）、验证错误和配置错误返回 `false`。

### 3.7 失败路由

当阶段返回 FAIL（或重试耗尽）时，引擎按以下顺序尝试失败路由：

1. **失败边：** 具有 `condition="outcome=fail"` 的出边。如果找到，遵循它。
2. **重试目标：** 节点属性 `retry_target`。跳转到该节点。
3. **回退重试目标：** 节点属性 `fallback_retry_target`。跳转到该节点。
4. **流水线终止：** 未找到失败路由。流水线因阶段的失败原因而失败。

### 3.8 并发模型

图遍历是单线程的。顶级图中一次只执行一个节点。这简化了上下文状态的推理并避免了竞争条件。

并行性存在于管理内部并发执行的特定节点处理器（`parallel`、`parallel.fan_in`）中。每个并行分支接收上下文的隔离克隆。收集分支结果，但单个分支上下文更改不会合并回父级——仅应用处理器的结果及其 `context_updates`。

---

## 4. 节点处理器

### 4.1 处理器接口

每个节点处理器都实现一个公共接口。执行引擎根据节点的 `type` 属性（如果 `type` 为空，则基于基于形状的解析）分派到适当的处理器。

```
INTERFACE Handler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome

    -- 参数：
    --   node      : 具有所有其属性的已解析节点
    --   context   : 流水线运行的共享键值上下文（读/写）
    --   graph     : 完整的已解析图（用于读取出边等）
    --   logs_root : 此运行的日志/制品目录的文件系统路径

    -- 返回：
    --   Outcome   : 执行的结果（见第 5.2 节）
```

### 4.2 处理器注册表

处理器注册表将类型字符串映射到处理器实例。解析遵循以下顺序：

1. 节点上的**显式 `type` 属性**（例如，`type="wait.human"`）
2. **基于形状的解析**，使用形状到处理器类型映射表（第 2.8 节）
3. **默认处理器**（codergen/LLM 处理器）

```
HandlerRegistry:
    handlers        : Map<String, Handler>   -- 类型字符串 -> 处理器实例
    default_handler : Handler                -- 回退处理器（通常是 codergen）

    FUNCTION register(type_string, handler):
        handlers[type_string] = handler
        -- 为已注册类型注册会替换先前的处理器

    FUNCTION resolve(node) -> Handler:
        -- 1. 显式 type 属性
        IF node.type is not empty AND node.type IN handlers:
            RETURN handlers[node.type]

        -- 2. 基于形状的解析
        handler_type = SHAPE_TO_TYPE[node.shape]
        IF handler_type IN handlers:
            RETURN handlers[handler_type]

        -- 3. 默认
        RETURN default_handler
```

### 4.3 起始处理器

流水线入口点的无操作处理器。立即返回 SUCCESS，不执行任何工作。

```
StartHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        RETURN Outcome(status=SUCCESS)
```

每个图必须恰好有一个起始节点（shape=Mdiamond）。检查规则强制执行此操作。

### 4.4 退出处理器

流水线出口点的无操作处理器。立即返回 SUCCESS。目标门强制执行由执行引擎处理（第 3.4 节），而不是由此处理器处理。

```
ExitHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        RETURN Outcome(status=SUCCESS)
```

每个图必须恰好有一个退出节点（shape=Msquare）。

### 4.5 Codergen 处理器（LLM 任务）

codergen 处理器是所有调用 LLM 的节点的默认值。它读取节点的提示、扩展模板变量、调用 LLM 后端（有关后端选项，见第 1.4 节）、将提示和响应写入日志目录，并返回结果。

```
CodergenHandler:
    backend : CodergenBackend | None
        -- LLM 执行后端。CodergenBackend 接口的任何实现（第 4.5 节）。
        -- None = 模拟模式。

    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        -- 1. 构建提示
        prompt = node.prompt
        IF prompt is empty:
            prompt = node.label
        prompt = expand_variables(prompt, graph, context)

        -- 2. 将提示写入日志
        stage_dir = logs_root + "/" + node.id + "/"
        create_directory(stage_dir)
        write_file(stage_dir + "prompt.md", prompt)

        -- 3. 调用 LLM 后端
        IF backend is not NONE:
            TRY:
                result = backend.run(node, prompt, context)
                IF result is an Outcome:
                    write_status(stage_dir, result)
                    RETURN result
                response_text = string(result)
            CATCH exception:
                RETURN Outcome(status=FAIL, failure_reason=str(exception))
        ELSE:
            response_text = "[Simulated] Response for stage: " + node.id

        -- 4. 将响应写入日志
        write_file(stage_dir + "response.md", response_text)

        -- 5. 写入状态并返回结果
        outcome = Outcome(
            status=SUCCESS,
            notes="Stage completed: " + node.id,
            context_updates={
                "last_stage": node.id,
                "last_response": truncate(response_text, 200)
            }
        )
        write_status(stage_dir, outcome)
        RETURN outcome
```

**变量扩展：** 唯一的内置模板变量是 `$goal`，它解析为图级的 `goal` 属性。变量扩展是简单的字符串替换，而不是模板引擎。

**状态文件：** 处理器在阶段目录中写入 `status.json`，其中 Outcome 字段序列化为 JSON。此文件用作审计跟踪并启用状态文件契约：外部工具或代理可以编写 `status.json` 以将结果传达回引擎。

#### CodergenBackend 接口

```
INTERFACE CodergenBackend:
    FUNCTION run(node: Node, prompt: String, context: Context) -> String | Outcome
```

如何实现此接口取决于您。流水线引擎只关心它获取 String 或 Outcome。

### 4.6 等待人工处理器

阻塞流水线执行，直到人工从节点的出边派生的选项中进行选择。这实现了人机协作模式（有关完整的面试官协议，见第 6 节）。

```
WaitForHumanHandler:
    interviewer : Interviewer  -- 人类交互前端

    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        -- 1. 从出边派生选项
        edges = graph.outgoing_edges(node.id)
        choices = []
        FOR EACH edge IN edges:
            label = edge.label OR edge.to_node
            key = parse_accelerator_key(label)
            choices.append(Choice(key=key, label=label, to=edge.to_node))

        IF choices is empty:
            RETURN Outcome(status=FAIL, failure_reason="No outgoing edges for human gate")

        -- 2. 从选项中构建问题
        options = [Option(key=c.key, label=c.label) FOR c IN choices]
        question = Question(
            text=node.label OR "Select an option:",
            type=MULTIPLE_CHOICE,
            options=options,
            stage=node.id
        )

        -- 3. 呈现给面试官并等待答案
        answer = interviewer.ask(question)

        -- 4. 处理超时/跳过
        IF answer is TIMEOUT:
            default_choice = node.attrs["human.default_choice"]
            IF default_choice exists:
                -- 使用默认值
            ELSE:
                RETURN Outcome(status=RETRY, failure_reason="human gate timeout, no default")

        IF answer is SKIPPED:
            RETURN Outcome(status=FAIL, failure_reason="human skipped interaction")

        -- 5. 查找匹配的选项
        selected = find_choice_matching(answer, choices)
        IF selected is NONE:
            selected = choices[0]  -- 回退到第一个

        -- 6. 在上下文中记录并返回
        RETURN Outcome(
            status=SUCCESS,
            suggested_next_ids=[selected.to],
            context_updates={
                "human.gate.selected": selected.key,
                "human.gate.label": selected.label
            }
        )
```

**加速器键解析** 使用以下模式从边标签中提取快捷键：

| Pattern           | Example           | Extracted Key |
|-------------------|-------------------|---------------|
| `[K] Label`       | `[Y] Yes, deploy` | `Y`           |
| `K) Label`        | `Y) Yes, deploy`  | `Y`           |
| `K - Label`       | `Y - Yes, deploy` | `Y`           |
| First character   | `Yes, deploy`     | `Y`           |

### 4.7 条件处理器

对于充当条件路由点的菱形节点。处理器本身是一个返回 SUCCESS 的无操作；实际路由由执行引擎的边选择算法处理（第 3.3 节），该算法评估出边上的条件。

```
ConditionalHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        RETURN Outcome(
            status=SUCCESS,
            notes="Conditional node evaluated: " + node.id
        )
```

此设计将路由逻辑保留在引擎中（在那里它可以是确定性的和可检查的），而不是在处理器中。

### 4.8 并行处理器

并发地将执行分发到多个分支。每个并行分支接收父上下文的隔离克隆并独立运行。处理器等待所有分支完成（或应用可配置的加入策略）然后返回。

```
ParallelHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        -- 1. 识别分发边（来自此节点的所有出边）
        branches = graph.outgoing_edges(node.id)

        -- 2. 从节点属性确定加入策略
        join_policy = node.attrs.get("join_policy", "wait_all")
        error_policy = node.attrs.get("error_policy", "continue")
        max_parallel = integer(node.attrs.get("max_parallel", "4"))

        -- 3. 使用有界并行并发执行分支
        results = []
        FOR EACH branch IN branches (最多 max_parallel 个分支):
            branch_context = context.clone()
            branch_outcome = execute_subgraph(branch.to_node, branch_context, graph, logs_root)
            results.append(branch_outcome)

        -- 4. 评估加入策略
        success_count = count(r FOR r IN results WHERE r.status == SUCCESS)
        fail_count = count(r FOR r IN results WHERE r.status == FAIL)

        IF join_policy == "wait_all":
            IF fail_count == 0:
                RETURN Outcome(status=SUCCESS)
            ELSE:
                RETURN Outcome(status=PARTIAL_SUCCESS)

        IF join_policy == "first_success":
            IF success_count > 0:
                RETURN Outcome(status=SUCCESS)
            ELSE:
                RETURN Outcome(status=FAIL)

        -- 5. 在上下文中存储结果以供下游汇聚
        context.set("parallel.results", serialize_results(results))
        RETURN Outcome(status=SUCCESS)
```

**加入策略：**

| Policy           | Behavior |
|------------------|----------|
| `wait_all`       | 所有分支必须完成。当全部完成时加入满足。 |
| `k_of_n`         | 至少 K 个分支必须成功。 |
| `first_success`  | 一旦一个分支成功就满足加入。其他分支可能被取消。 |
| `quorum`         | 至少可配置分数的分支必须成功。 |

**错误策略：**

| Policy              | Behavior |
|---------------------|----------|
| `fail_fast`         | 在首次失败时取消所有剩余分支。 |
| `continue`          | 继续剩余分支。收集所有结果。 |
| `ignore`            | 完全忽略失败。仅返回成功结果。 |

### 4.9 汇聚处理器

合并前一个并行节点的结果并选择最佳候选。

```
FanInHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        -- 1. 读取并行结果
        results = context.get("parallel.results")
        IF results is empty:
            RETURN Outcome(status=FAIL, failure_reason="No parallel results to evaluate")

        -- 2. 评估候选
        IF node.prompt is not empty:
            -- 基于 LLM 的评估：调用 LLM 对候选进行排名
            best = llm_evaluate(node.prompt, results)
        ELSE:
            -- 启发式：按结果状态排名，然后按分数
            best = heuristic_select(results)

        -- 3. 在上下文中记录获胜者
        context_updates = {
            "parallel.fan_in.best_id": best.id,
            "parallel.fan_in.best_outcome": best.outcome
        }

        RETURN Outcome(
            status=SUCCESS,
            context_updates=context_updates,
            notes="Selected best candidate: " + best.id
        )


FUNCTION heuristic_select(candidates):
    outcome_rank = {SUCCESS: 0, PARTIAL_SUCCESS: 1, RETRY: 2, FAIL: 3}
    SORT candidates BY (outcome_rank[c.outcome], -c.score, c.id)
    RETURN candidates[0]
```

只要至少有一个候选可用，即使某些候选失败，汇聚也会运行。仅当所有候选都失败时，汇聚才返回 FAIL。

### 4.10 工具处理器

执行通过节点属性配置的外部工具（shell 命令、API 调用或其他非 LLM 操作）。

```
ToolHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        command = node.attrs.get("tool_command", "")
        IF command is empty:
            RETURN Outcome(status=FAIL, failure_reason="No tool_command specified")

        -- 执行命令
        TRY:
            result = run_shell_command(command, timeout=node.timeout)
            RETURN Outcome(
                status=SUCCESS,
                context_updates={"tool.output": result.stdout},
                notes="Tool completed: " + command
            )
        CATCH exception:
            RETURN Outcome(status=FAIL, failure_reason=str(exception))
```

### 4.11 管理器循环处理器

通过监督子流水线来编排基于迭代的迭代。管理器观察子级的遥测、通过守卫函数评估进度，并可选择通过干预来引导子级。

```
ManagerLoopHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        child_dotfile = graph.attrs.get("stack.child_dotfile")
        poll_interval = parse_duration(node.attrs.get("manager.poll_interval", "45s"))
        max_cycles = integer(node.attrs.get("manager.max_cycles", "1000"))
        stop_condition = node.attrs.get("manager.stop_condition", "")
        actions = split(node.attrs.get("manager.actions", "observe,wait"), ",")

        -- 1. 如果配置了则自动启动子级
        IF node.attrs.get("stack.child_autostart", "true") == "true":
            start_child_pipeline(child_dotfile)

        -- 2. 观察循环
        FOR cycle FROM 1 TO max_cycles:
            IF "observe" IN actions:
                ingest_child_telemetry(context)

            IF "steer" IN actions AND steer_cooldown_elapsed():
                steer_child(context, node)

            -- 评估停止条件
            child_status = context.get_string("context.stack.child.status")
            IF child_status IN {"completed", "failed"}:
                child_outcome = context.get_string("context.stack.child.outcome")
                IF child_outcome == "success":
                    RETURN Outcome(status=SUCCESS, notes="Child completed")
                IF child_status == "failed":
                    RETURN Outcome(status=FAIL, failure_reason="Child failed")

            IF stop_condition is not empty:
                IF evaluate_condition(stop_condition, ..., context):
                    RETURN Outcome(status=SUCCESS, notes="Stop condition satisfied")

            IF "wait" IN actions:
                sleep(poll_interval)

        RETURN Outcome(status=FAIL, failure_reason="Max cycles exceeded")
```

管理器模式实现**主管架构**，其中：
- **观察** 摄取工作者遥测（活动阶段、结果、重试计数、制品）
- **守卫** 对工作者进展进行评分并路由到继续、干预或升级
- **引导** 将干预指令写入子级的活动阶段目录

### 4.12 自定义处理器

通过实现 Handler 接口并向注册表注册来添加新的处理器类型：

```
-- 定义自定义处理器
MyCustomHandler:
    FUNCTION execute(node, context, graph, logs_root) -> Outcome:
        -- 自定义逻辑
        RETURN Outcome(status=SUCCESS)

-- 注册它
registry.register("my_custom_type", MyCustomHandler())

-- 在 DOT 文件中引用
my_node [type="my_custom_type", shape=box, custom_attr="value"]
```

**处理器契约：**
- 处理器必须是无状态的或使用同步保护共享可变状态。
- 处理器恐慌/异常必须由引擎捕获并转换为 FAIL 结果。
- 处理器不应嵌入特定于提供商的逻辑；LLM 编组委托给集成的 SDK。

---

## 5. 状态与上下文

### 5.1 上下文

上下文是一个线程安全的键值存储，在流水线运行期间在所有阶段之间共享。它是节点之间传递数据的主要机制。

```
Context:
    values : Map<String, Any>      -- 键值存储
    lock   : ReadWriteLock         -- 并行访问的线程安全
    logs   : List<String>          -- 仅追加运行日志

    FUNCTION set(key, value):
        ACQUIRE write lock
        values[key] = value
        RELEASE write lock

    FUNCTION get(key, default=NONE) -> Any:
        ACQUIRE read lock
        result = values.get(key, default)
        RELEASE read lock
        RETURN result

    FUNCTION get_string(key, default="") -> String:
        value = get(key)
        IF value is NONE: RETURN default
        RETURN string(value)

    FUNCTION append_log(entry):
        ACQUIRE write lock
        logs.append(entry)
        RELEASE write lock

    FUNCTION snapshot() -> Map<String, Any>:
        -- 返回所有值的可序列化副本
        ACQUIRE read lock
        result = shallow_copy(values)
        RELEASE read lock
        RETURN result

    FUNCTION clone() -> Context:
        -- 并行分支隔离的深度复制
        ACQUIRE read lock
        new_context = new Context()
        new_context.values = shallow_copy(values)
        new_context.logs = copy(logs)
        RELEASE read lock
        RETURN new_context

    FUNCTION apply_updates(updates):
        -- 将更新字典合并到上下文中
        ACQUIRE write lock
        FOR EACH (key, value) IN updates:
            values[key] = value
        RELEASE write lock
```

**引擎设置的内置上下文键：**

| Key                                   | Type    | Set By   | Description |
|---------------------------------------|---------|----------|-------------|
| `outcome`                             | String  | Engine   | 最后一个处理器结果状态（`success`、`fail` 等） |
| `preferred_label`                     | String  | Engine   | 最后一个处理器的首选边标签 |
| `graph.goal`                          | String  | Engine   | 从图 `goal` 属性镜像 |
| `current_node`                        | String  | Engine   | 当前执行节点的 ID |
| `last_stage`                          | String  | Handler  | 最后完成的阶段的 ID |
| `last_response`                       | String  | Handler  | 最后一个 LLM 响应的截断文本 |
| `internal.retry_count.<node_id>`      | Integer | Engine   | 特定节点的重试计数器 |

**上下文键命名空间约定：**

| Prefix        | Purpose                                        |
|---------------|------------------------------------------------|
| `context.*`   | 节点之间共享的语义状态                          |
| `graph.*`     | 初始化时镜像的图属性                            |
| `internal.*`  | 引擎簿记（重试计数器、计时）                    |
| `parallel.*`  | 并行处理器状态（结果、计数）                    |
| `stack.*`     | 主管/工作者状态                                 |
| `human.gate.*`| 人类交互状态                                    |
| `work.*`      | 并行工作项的每项上下文                          |

### 5.2 结果

结果是执行节点处理器的结果。它驱动路由决策和状态更新。

```
Outcome:
    status             : StageStatus     -- SUCCESS, FAIL, PARTIAL_SUCCESS, RETRY, SKIPPED
    preferred_label    : String          -- 要遵循的边标签（可选）
    suggested_next_ids : List<String>    -- 显式下一个节点 ID（可选）
    context_updates    : Map<String, Any> -- 要合并到上下文中的键值对
    notes              : String          -- 人类可读的执行摘要
    failure_reason     : String          -- 失败原因（当状态为 FAIL 或 RETRY 时）
```

**StageStatus 值：**

| Status             | Meaning |
|--------------------|---------|
| `SUCCESS`          | 阶段完成了其工作。前进到下一条边。重置重试计数器。 |
| `PARTIAL_SUCCESS`  | 阶段完成但有警告。对于路由视为成功，但注释描述了未完成的内容。 |
| `RETRY`            | 阶段请求重新执行。引擎增加重试计数器并在限制内重新执行。 |
| `FAIL`             | 阶段永久失败。引擎查找失败边或终止流水线。 |
| `SKIPPED`          | 阶段被跳过（例如，条件未满足）。继续而不记录结果。 |

### 5.3 检查点

执行状态的可序列化快照，在每个节点完成后保存。启用崩溃恢复和恢复。

```
Checkpoint:
    timestamp       : Timestamp              -- 创建此检查点的时间
    current_node    : String                  -- 最后完成的节点的 ID
    completed_nodes : List<String>            -- 按顺序排列的所有已完成节点的 ID
    node_retries    : Map<String, Integer>    -- 每节点的重试计数器
    context_values  : Map<String, Any>        -- 上下文的序列化快照
    logs            : List<String>            -- 运行日志条目

    FUNCTION save(path):
        -- 序列化为 JSON 并写入文件系统
        data = {
            "timestamp": timestamp,
            "current_node": current_node,
            "completed_nodes": completed_nodes,
            "node_retries": node_retries,
            "context": serialize_to_json(context_values),
            "logs": logs
        }
        write_json_file(path, data)

    FUNCTION load(path) -> Checkpoint:
        -- 从 JSON 文件反序列化
        data = read_json_file(path)
        RETURN new Checkpoint from data
```

**恢复行为：**

1. 从 `{logs_root}/checkpoint.json` 加载检查点。
2. 从 `context_values` 恢复上下文状态。
3. 恢复 `completed_nodes` 以跳过已完成的工作。
4. 从 `node_retries` 恢复重试计数器。
5. 确定要执行的下一个节点（遍历中 `current_node` 之后的节点）。
6. 如果前一个节点使用 `full` 保真度，则为第一个恢复的节点降级为 `summary:high`，因为内存 LLM 会话无法序列化。在这一降级跳跃之后，后续节点可以再次使用 `full` 保真度。

### 5.4 上下文保真度

上下文保真度控制有多少先前的对话和状态被携带到下一个节点的 LLM 会话中。这是管理多阶段流水线中上下文窗口使用的核心机制。

```
FidelityMode ::= 'full'
               | 'truncate'
               | 'compact'
               | 'summary:low'
               | 'summary:medium'
               | 'summary:high'
```

| Mode             | Session | Context Carried                                         | Approximate Token Budget |
|------------------|---------|---------------------------------------------------------|--------------------------|
| `full`           | 重用（同一线程） | 保留完整的对话历史                                      | 无限制（使用压缩） |
| `truncate`       | 新     | 最小：仅图目标和运行 ID                                 | 最小 |
| `compact`        | 新     | 结构化项目符号摘要：已完成阶段、结果、关键上下文值      | 中等 |
| `summary:low`    | 新     | 简短文本摘要，带有最小事件计数                          | ~600 tokens |
| `summary:medium` | 新     | 中等细节：最近阶段结果、活动上下文值、值得注意的事件    | ~1500 tokens |
| `summary:high`   | 新     | 详细：许多最近事件、工具调用摘要、综合上下文             | ~3000 tokens |

**保真度解析优先级（从高到低）：**

1. 边 `fidelity` 属性（在入边上）
2. 目标节点 `fidelity` 属性
3. 图 `default_fidelity` 属性
4. 默认值：`compact`

**线程解析（用于 `full` 保真度）：**

当保真度解析为 `full` 时，引擎为会话重用确定线程密钥：

1. 目标节点 `thread_id` 属性
2. 边 `thread_id` 属性
3. 图级默认线程
4. 来自封闭子图的派生类
5. 回退：前一个节点 ID

共享相同线程密钥的节点重用同一个 LLM 会话。具有不同线程密钥的节点启动新会话。

### 5.5 制品存储

制品存储为不属于上下文的大型阶段输出提供命名、类型化存储（上下文应仅包含用于路由和检查点序列化的小标量值）。

```
ArtifactStore:
    artifacts : Map<String, (ArtifactInfo, Any)>
    lock      : ReadWriteLock
    base_dir  : String or NONE   -- 文件支持的制品的文件系统目录

    FUNCTION store(artifact_id, name, data) -> ArtifactInfo:
        size = byte_size(data)
        is_file_backed = (size > FILE_BACKING_THRESHOLD) AND (base_dir is not NONE)
        IF is_file_backed:
            write data to "{base_dir}/artifacts/{artifact_id}.json"
            stored_data = file_path
        ELSE:
            stored_data = data
        info = ArtifactInfo(id=artifact_id, name=name, size=size, is_file_backed=is_file_backed)
        artifacts[artifact_id] = (info, stored_data)
        RETURN info

    FUNCTION retrieve(artifact_id) -> Any:
        IF artifact_id NOT IN artifacts:
            RAISE "Artifact not found"
        (info, data) = artifacts[artifact_id]
        IF info.is_file_backed:
            RETURN read_json_file(data)
        RETURN data

    FUNCTION has(artifact_id) -> Boolean
    FUNCTION list() -> List<ArtifactInfo>
    FUNCTION remove(artifact_id)
    FUNCTION clear()

ArtifactInfo:
    id              : String
    name            : String
    size_bytes      : Integer
    stored_at       : Timestamp
    is_file_backed  : Boolean
```

默认文件支持阈值为 100KB。低于此阈值的制品存储在内存中；高于此阈值的制品写入磁盘。

### 5.6 运行目录结构

每次流水线执行都会生成一个用于日志记录、检查点和制品的目录树：

```
{logs_root}/
    checkpoint.json              -- 每个节点后的序列化检查点
    manifest.json                -- 流水线元数据（名称、目标、开始时间）
    {node_id}/
        status.json              -- 节点执行结果
        prompt.md                -- 发送到 LLM 的呈现提示
        response.md              -- LLM 响应文本
    artifacts/
        {artifact_id}.json       -- 文件支持的制品
```

---

## 6. 人机协作（面试官模式）

### 6.1 面试官接口

Attractor 中的所有人类交互都通过面试官接口进行。这种抽象允许流水线通过任何前端向人类提出问题并接收答案：CLI、Web UI、Slack 机器人或用于测试的编程队列。

```
INTERFACE Interviewer:
    FUNCTION ask(question: Question) -> Answer
    FUNCTION ask_multiple(questions: List<Question>) -> List<Answer>
    FUNCTION inform(message: String, stage: String) -> Void
```

### 6.2 问题模型

```
Question:
    text            : String              -- 向人类提出的问题
    type            : QuestionType        -- 确定 UI 和有效答案
    options         : List<Option>        -- 用于 MULTIPLE_CHOICE 类型
    default         : Answer or NONE      -- 超时/跳过时的默认值
    timeout_seconds : Float or NONE       -- 最大等待时间
    stage           : String              -- 来源阶段名称（用于显示）
    metadata        : Map<String, Any>    -- 任意键值对

QuestionType:
    YES_NO              -- 是/否二元选择
    MULTIPLE_CHOICE     -- 从选项列表中选择一个
    FREEFORM            -- 自由文本输入
    CONFIRMATION        -- 是/否确认（在语义上与 YES_NO 不同）

Option:
    key   : String    -- 加速器键（例如，"Y"、"A"）
    label : String    -- 显示文本（例如，"Yes, deploy to production"）
```

### 6.3 答案模型

```
Answer:
    value           : String or AnswerValue   -- 选定的值
    selected_option : Option or NONE          -- 完整的选定选项（用于 MULTIPLE_CHOICE）
    text            : String                  -- 自由文本响应（用于 FREEFORM）

AnswerValue:
    YES       -- 肯定
    NO        -- 否定
    SKIPPED   -- 人类跳过了问题
    TIMEOUT   -- 超时内无响应
```

### 6.4 内置面试官实现

**AutoApproveInterviewer：** 始终为是/否问题选择 YES，为多选问题选择第一个选项。用于自动化测试和 CI/CD 流水线，其中没有人类可用。

```
AutoApproveInterviewer:
    FUNCTION ask(question) -> Answer:
        IF question.type IN {YES_NO, CONFIRMATION}:
            RETURN Answer(value=YES)
        IF question.type == MULTIPLE_CHOICE AND question.options is not empty:
            RETURN Answer(value=question.options[0].key, selected_option=question.options[0])
        RETURN Answer(value="auto-approved", text="auto-approved")
```

**ConsoleInterviewer (CLI)：** 从标准输入读取。显示带有选项键的格式化提示。通过非阻塞读取支持超时。

```
ConsoleInterviewer:
    FUNCTION ask(question) -> Answer:
        print("[?] " + question.text)
        IF question.type == MULTIPLE_CHOICE:
            FOR EACH option IN question.options:
                print("  [" + option.key + "] " + option.label)
            response = read_input("Select: ")
            RETURN find_matching_option(response, question.options)
        IF question.type == YES_NO:
            response = read_input("[Y/N]: ")
            RETURN Answer(value=YES if response is "y" ELSE NO)
        IF question.type == FREEFORM:
            response = read_input("> ")
            RETURN Answer(text=response)
```

**CallbackInterviewer：** 将问题回答委托给提供的回调函数。用于与外部系统集成（Slack、Web UI、API）。

```
CallbackInterviewer:
    callback : Function(Question) -> Answer

    FUNCTION ask(question) -> Answer:
        RETURN callback(question)
```

**QueueInterviewer：** 从预填充的答案队列中读取答案。用于确定性测试和重放。

```
QueueInterviewer:
    answers : Queue<Answer>

    FUNCTION ask(question) -> Answer:
        IF answers is not empty:
            RETURN answers.dequeue()
        RETURN Answer(value=SKIPPED)
```

**RecordingInterviewer：** 包装另一个面试官并记录所有问答对。用于重放、调试和审计跟踪。

```
RecordingInterviewer:
    inner      : Interviewer
    recordings : List<(Question, Answer)>

    FUNCTION ask(question) -> Answer:
        answer = inner.ask(question)
        recordings.append((question, answer))
        RETURN answer
```

### 6.5 超时处理

如果人类在配置的 `timeout_seconds` 内没有响应：

1. 如果问题有 `default` 答案，使用它。
2. 如果没有默认值，返回 `Answer(value=TIMEOUT)`。
3. 处理器决定如何处理超时（重试问题、失败或按假设继续）。

对于 `wait.human` 节点，节点属性 `human.default_choice` 指定在超时时选择哪个边目标。

---

## 7. 验证与检查

### 7.1 诊断模型

验证生成诊断列表，每个诊断都有一个严重性级别。引擎必须拒绝执行具有错误严重性诊断的流水线。

```
Diagnostic:
    rule     : String                    -- 规则标识符（例如，"start_node"）
    severity : Severity                  -- ERROR、WARNING 或 INFO
    message  : String                    -- 人类可读的描述
    node_id  : String                    -- 相关节点 ID（可选）
    edge     : (String, String) or NONE  -- 相关边作为 (from, to)（可选）
    fix      : String                    -- 建议的修复（可选）

Severity:
    ERROR     -- 流水线不会执行
    WARNING   -- 流水线将执行，但行为可能出乎意料
    INFO      -- 信息性说明
```

### 7.2 内置检查规则

| Rule ID                  | Severity | Description |
|--------------------------|----------|-------------|
| `start_node`             | ERROR    | 流水线必须恰好有一个起始节点（shape=Mdiamond 或 id 匹配 `start`/`Start`）。 |
| `terminal_node`          | ERROR    | 流水线必须至少有一个终端节点（shape=Msquare 或 id 匹配 `exit`/`end`）。 |
| `reachability`           | ERROR    | 所有节点必须可以通过 BFS/DFS 遍历从起始节点到达。 |
| `edge_target_exists`     | ERROR    | 每个边目标必须引用现有的节点 ID。 |
| `start_no_incoming`      | ERROR    | 起始节点必须没有入边。 |
| `exit_no_outgoing`       | ERROR    | 退出节点必须没有出边。 |
| `condition_syntax`       | ERROR    | 边条件表达式必须正确解析（有效的运算符和键）。 |
| `stylesheet_syntax`      | ERROR    | `model_stylesheet` 属性必须解析为有效的样式表规则。 |
| `type_known`             | WARNING  | 节点 `type` 值应被处理器注册表识别。 |
| `fidelity_valid`         | WARNING  | 保真度模式值必须是以下之一：`full`、`truncate`、`compact`、`summary:low`、`summary:medium`、`summary:high`。 |
| `retry_target_exists`    | WARNING  | `retry_target` 和 `fallback_retry_target` 必须引用现有节点。 |
| `goal_gate_has_retry`    | WARNING  | 具有 `goal_gate=true` 的节点应该有 `retry_target` 或 `fallback_retry_target`。 |
| `prompt_on_llm_nodes`    | WARNING  | 解析为 codergen 处理器的节点应该有 `prompt` 或 `label` 属性。 |

### 7.3 验证 API

```
FUNCTION validate(graph, extra_rules=NONE) -> List<Diagnostic>:
    rules = BUILT_IN_RULES
    IF extra_rules is not NONE:
        rules = rules + extra_rules
    diagnostics = []
    FOR EACH rule IN rules:
        diagnostics.extend(rule.apply(graph))
    RETURN diagnostics


FUNCTION validate_or_raise(graph, extra_rules=NONE):
    diagnostics = validate(graph, extra_rules)
    errors = [d FOR d IN diagnostics WHERE d.severity == ERROR]
    IF errors is not empty:
        RAISE ValidationError with error messages
    RETURN diagnostics
```

### 7.4 自定义检查规则

实现可以通过实现规则接口来注册自定义检查规则：

```
INTERFACE LintRule:
    name : String
    FUNCTION apply(graph) -> List<Diagnostic>
```

自定义规则附加到内置规则并在验证期间运行。

---

## 8. 模型样式表

### 8.1 概述

`model_stylesheet` 图属性为在节点上设置默认 LLM 配置提供类似 CSS 的规则。这集中了模型选择，以便单个节点不需要在每个节点上指定 `llm_model`、`llm_provider` 和 `reasoning_effort`。

### 8.2 样式表语法

```
Stylesheet    ::= Rule+
Rule          ::= Selector '{' Declaration ( ';' Declaration )* ';'? '}'
Selector      ::= '*' | '#' Identifier | '.' ClassName
ClassName     ::= [a-z0-9-]+
Declaration   ::= Property ':' PropertyValue
Property      ::= 'llm_model' | 'llm_provider' | 'reasoning_effort'
PropertyValue ::= String | 'low' | 'medium' | 'high'
```

### 8.3 选择器和特异性

| Selector      | Matches                      | Specificity |
|---------------|------------------------------|-------------|
| `*`           | 所有节点                     | 0（最低）   |
| `.class_name` | 具有该类的节点               | 1（中等）   |
| `#node_id`    | 按 ID 的特定节点             | 2（最高）   |

相同特异性的后续规则覆盖先前的规则。显式节点属性始终覆盖样式表值（最高优先级）。

### 8.4 识别的属性

| Property           | Values                     | Description |
|--------------------|----------------------------|-------------|
| `llm_model`        | 任何模型标识符字符串        | 提供商原生模型 ID（例如，`gpt-5.2`、`claude-opus-4-6`） |
| `llm_provider`     | 提供商密钥字符串            | `openai`、`anthropic`、`gemini` 等。 |
| `reasoning_effort`  | `low`、`medium`、`high`    | 控制 LLM 的推理/思考深度 |

### 8.5 应用顺序

节点上任何模型相关属性的解析顺序为：

1. 显式节点属性（例如，节点上的 `llm_model="gpt-5.2"`）-- 最高优先级
2. 按特异性匹配的样式表规则（ID > class > universal）
3. 图级默认属性
4. 处理器/系统默认值

样式表在解析之后和验证之前作为转换应用。转换遍历所有节点并应用匹配的规则，但仅设置节点尚未显式拥有的属性。

### 8.6 示例

```
digraph Pipeline {
    graph [
        goal="Implement feature X",
        model_stylesheet="
            * { llm_model: claude-sonnet-4-5; llm_provider: anthropic; }
            .code { llm_model: claude-opus-4-6; llm_provider: anthropic; }
            #critical_review { llm_model: gpt-5.2; llm_provider: openai; reasoning_effort: high; }
        "
    ]

    start [shape=Mdiamond]
    exit  [shape=Msquare]

    plan            [label="Plan", class="planning"]
    implement       [label="Implement", class="code"]
    critical_review [label="Critical Review", class="code"]

    start -> plan -> implement -> critical_review -> exit
}
```

在此示例中：
- `plan` 从 `*` 规则获取 `claude-sonnet-4-5`（没有 `.code` 的类匹配）。
- `implement` 从 `.code` 规则获取 `claude-opus-4-6`。
- `critical_review` 从 `#critical_review` 规则获取 `gpt-5.2`（最高特异性），覆盖 `.code` 类匹配。

---

## 9. 转换与扩展性

### 9.1 AST 转换

转换是在解析之后和验证之前修改流水线图的函数。它们启用预处理、优化和结构重写，而无需修改原始 DOT 文件。

```
INTERFACE Transform:
    FUNCTION apply(graph) -> Graph
    -- 返回新的或修改的图。不应修改输入图。
```

转换在解析之后和验证之前按定义的顺序应用：

```
FUNCTION prepare_pipeline(dot_source):
    graph = parse(dot_source)
    FOR EACH transform IN transforms:
        graph = transform.apply(graph)
    diagnostics = validate(graph)
    RETURN (graph, diagnostics)
```

### 9.2 内置转换

**变量扩展转换：** 将节点 `prompt` 属性中的 `$goal` 扩展为图级 `goal` 属性值。

```
VariableExpansionTransform:
    FUNCTION apply(graph) -> Graph:
        FOR EACH node IN graph.nodes:
            IF node.prompt contains "$goal":
                node.prompt = replace(node.prompt, "$goal", graph.goal)
        RETURN graph
```

**样式表应用转换：** 应用 `model_stylesheet` 来解析每个节点的 `llm_model`、`llm_provider` 和 `reasoning_effort`。详情见第 8 节。

**前导转换：** 为不使用 `full` 保真度的阶段合成上下文携带文本。在执行时应用（而不是在解析时），因为它依赖于运行时状态。

### 9.3 自定义转换

实现可以注册自定义转换：

```
runner.register_transform(MyCustomTransform())
```

自定义转换在内置转换之后运行。自定义转换的顺序遵循注册顺序。

**自定义转换的用例：**
- 将日志记录或审计节点注入图中
- 在某些节点类型周围添加重试包装器
- 将多个图合并到单个流水线中
- 应用组织特定的默认值

### 9.4 流水线组合

Attractor 支持通过以下方式组合多个 DOT 图：

**子流水线节点：** 其处理器将整个子图作为其执行运行的节点。管理器循环处理器（第 4.11 节）就是此模式的一个示例。

**图合并（通过转换）：** 自定义转换可以将一个图的节点和边合并到另一个图中，从而实现模块化流水线定义。

### 9.5 HTTP 服务器模式

实现可以将流水线引擎公开为 HTTP 服务，用于基于 Web 的管理、远程人工交互和与外部系统集成。

**核心端点：**

| Method | Path                                    | Description |
|--------|-----------------------------------------|-------------|
| `POST` | `/pipelines`                            | 提交 DOT 源并开始执行。返回流水线 ID。 |
| `GET`  | `/pipelines/{id}`                       | 获取流水线状态和进度。 |
| `GET`  | `/pipelines/{id}/events`                | 流水线事件的实时 SSE 流。 |
| `POST` | `/pipelines/{id}/cancel`                | 取消正在运行的流水线。 |
| `GET`  | `/pipelines/{id}/graph`                 | 获取渲染的图可视化（SVG）。 |
| `GET`  | `/pipelines/{id}/questions`             | 获取待处理的人工交互问题。 |
| `POST` | `/pipelines/{id}/questions/{qid}/answer`| 向待处理问题提交答案。 |
| `GET`  | `/pipelines/{id}/checkpoint`            | 获取当前检查点状态。 |
| `GET`  | `/pipelines/{id}/context`               | 获取当前上下文键值存储。 |

人工门必须可以通过 Web 控件以及 CLI 操作。服务器维护 SSE 连接以进行实时事件流式传输。

### 9.6 可观察性和事件

引擎在执行期间发出类型化事件，用于 UI、日志记录和指标集成：

**流水线生命周期事件：**
- `PipelineStarted(name, id)` -- 流水线开始
- `PipelineCompleted(duration, artifact_count)` -- 流水线成功
- `PipelineFailed(error, duration)` -- 流水线失败

**阶段生命周期事件：**
- `StageStarted(name, index)` -- 阶段开始
- `StageCompleted(name, index, duration)` -- 阶段成功
- `StageFailed(name, index, error, will_retry)` -- 阶段失败
- `StageRetrying(name, index, attempt, delay)` -- 阶段重试

**并行执行事件：**
- `ParallelStarted(branch_count)` -- 并行块开始
- `ParallelBranchStarted(branch, index)` -- 分支开始
- `ParallelBranchCompleted(branch, index, duration, success)` -- 分支完成
- `ParallelCompleted(duration, success_count, failure_count)` -- 所有分支完成

**人类交互事件：**
- `InterviewStarted(question, stage)` -- 问题呈现
- `InterviewCompleted(question, answer, duration)` -- 收到答案
- `InterviewTimeout(question, stage, duration)` -- 达到超时

**检查点事件：**
- `CheckpointSaved(node_id)` -- 检查点已写入

事件可以通过观察者/回调模式或异步流使用：

```
-- 观察者模式
runner.on_event = FUNCTION(event):
    log(event.description)

-- 流模式（用于异步运行时）
FOR EACH event IN pipeline.events():
    process(event)
```

### 9.7 工具调用挂钩

图级或节点级属性 `tool_hooks.pre` 和 `tool_hooks.post` 指定围绕每个 LLM 工具调用执行的 shell 命令：

- **前挂钩：** 在每次 LLM 工具调用之前执行。通过环境变量和 stdin JSON 接收工具元数据。退出代码 0 表示继续；非零表示跳过工具调用。
- **后挂钩：** 在每次 LLM 工具调用之后执行。接收工具元数据和结果。主要用于日志记录和审计。

挂钩失败（非零退出）不会阻止工具调用，但会记录在阶段日志中。

---

## 10. 条件表达式语言

### 10.1 概述

边条件使用最小的布尔表达式语言来在路由期间限制边资格。该语言故意简单，以保持路由确定性和可检查。

### 10.2 语法

```
ConditionExpr  ::= Clause ( '&&' Clause )*
Clause         ::= Key Operator Literal
Key            ::= 'outcome'
                 | 'preferred_label'
                 | 'context.' Path
Path           ::= Identifier ( '.' Identifier )*
Operator       ::= '=' | '!='
Literal        ::= String | Integer | Boolean
```

### 10.3 语义

- 子句通过 AND 组合，从左到右评估。
- `outcome` 指的是执行节点的结果状态（`success`、`retry`、`fail`、`partial_success`）。
- `preferred_label` 指的是节点结果中的 `preferred_label` 值。
- `context.*` 键从运行上下文中查找值。缺失的键比较为空字符串（永远不等于非空值）。
- 字符串比较是精确的且区分大小写。
- 所有子句必须评估为 true 才能使条件通过。

### 10.4 变量解析

```
FUNCTION resolve_key(key, outcome, context) -> String:
    IF key == "outcome":
        RETURN outcome.status as string
    IF key == "preferred_label":
        RETURN outcome.preferred_label
    IF key starts with "context.":
        value = context.get(key)
        IF value is not NONE:
            RETURN string(value)
        -- 也尝试不带 "context." 前缀以便于使用
        value = context.get(key without "context." prefix)
        IF value is not NONE:
            RETURN string(value)
        RETURN ""
    -- 非限定键的直接上下文查找
    value = context.get(key)
    IF value is not NONE:
        RETURN string(value)
    RETURN ""
```

### 10.5 评估

```
FUNCTION evaluate_condition(condition, outcome, context) -> Boolean:
    IF condition is empty:
        RETURN true  -- 无条件意味着始终符合条件

    clauses = split(condition, "&&")
    FOR EACH clause IN clauses:
        clause = trim(clause)
        IF clause is empty:
            CONTINUE
        IF NOT evaluate_clause(clause, outcome, context):
            RETURN false
    RETURN true


FUNCTION evaluate_clause(clause, outcome, context) -> Boolean:
    IF clause contains "!=":
        (key, value) = split(clause, "!=", max=1)
        RETURN resolve_key(trim(key), outcome, context) != trim(value)
    ELSE IF clause contains "=":
        (key, value) = split(clause, "=", max=1)
        RETURN resolve_key(trim(key), outcome, context) == trim(value)
    ELSE:
        -- 裸键：检查是否为真
        RETURN bool(resolve_key(trim(clause), outcome, context))
```

### 10.6 示例

```
-- 成功时路由
plan -> implement [condition="outcome=success"]

-- 失败时路由
plan -> fix [condition="outcome=fail"]

-- 成功时路由且上下文标志
validate -> deploy [condition="outcome=success && context.tests_passed=true"]

-- 上下文值不是特定值时路由
review -> iterate [condition="context.loop_state!=exhausted"]

-- 基于首选标签路由
gate -> fix [condition="preferred_label=Fix"]
```

### 10.7 扩展运算符（未来）

当前条件语言仅支持 `=`（等于）和 `!=`（不等于）与 AND（`&&`） conjunction。未来版本可能会添加：

- `contains` -- 子字符串或集成员资格
- `matches` -- 正则表达式匹配
- `OR` -- disjunction
- `NOT` -- 否定
- `>`、`<`、`>=`、`<=` -- 数字比较

这些在此处记录为潜在扩展。实现不应在不更新语法和验证规则的情况下添加它们。

---

## 11. 完成定义

本节定义如何验证此规范的实现是完整且正确的。当每个项目都被勾选时，实现就完成了。

### 11.1 DOT 解析

- [ ] 解析器接受支持的 DOT 子集（带有 graph/node/edge 属性块的 digraph）
- [ ] 图级属性（`goal`、`label`、`model_stylesheet`）被正确提取
- [ ] 节点属性被解析，包括多行属性块（属性跨越 `[...]` 内的多行）
- [ ] 边属性（`label`、`condition`、`weight`）被正确解析
- [ ] 链式边（`A -> B -> C`）为每对生成单独的边
- [ ] 节点/边默认块（`node [...]`、`edge [...]`）应用于后续声明
- [ ] 子图块被展平（保留内容，移除包装器）
- [ ] 节点上的 `class` 属性合并来自样式表的属性
- [ ] 引用和未引用的属性值都可以工作
- [ ] 注释（`//` 和 `/* */`）在解析之前被剥离

### 11.2 验证与检查

- [ ] 恰好一个起始节点（shape=Mdiamond）是必需的
- [ ] 恰好一个退出节点（shape=Msquare）是必需的
- [ ] 起始节点没有入边
- [ ] 退出节点没有出边
- [ ] 所有节点都可以从起始节点到达（没有孤立节点）
- [ ] 所有边都引用有效的节点 ID
- [ ] Codergen 节点（shape=box）具有非空的 `prompt` 属性（如果缺失则发出警告）
- [ ] 边上的条件表达式正确解析
- [ ] `validate_or_raise()` 在错误严重性违规时抛出
- [ ] 检查结果包括规则名称、严重性（错误/警告）、节点/边 ID 和消息

### 11.3 执行引擎

- [ ] 引擎解析起始节点并从那里开始执行
- [ ] 每个节点的处理器通过形状到处理器类型映射解析
- [ ] 使用 (node, context, graph, logs_root) 调用处理器并返回 Outcome
- [ ] Outcome 被写入 `{logs_root}/{node_id}/status.json`
- [ ] 边选择遵循 5 步优先级：条件匹配 -> 首选标签 -> 建议 ID -> 权重 -> 词法
- [ ] 引擎循环：执行节点 -> 选择边 -> 前进到下一个节点 -> 重复
- [ ] 终端节点（shape=Msquare）停止执行
- [ ] 如果所有 goal_gate 节点成功，流水线结果为 "success"，否则为 "fail"

### 11.4 目标门强制执行

- [ ] 在执行期间跟踪具有 `goal_gate=true` 的节点
- [ ] 在通过终端节点允许退出之前，引擎检查所有目标门节点是否具有状态 SUCCESS
- [ ] 如果任何目标门节点未成功，引擎路由到 `retry_target`（如果已配置）而不是退出
- [ ] 如果没有 retry_target 且目标门未满足，流水线结果为 "fail"

### 11.5 重试逻辑

- [ ] 具有 `max_retries > 0` 的节点在 RETRY 或 FAIL 结果时重试
- [ ] 每个节点跟踪重试计数并遵守配置的限制
- [ ] 重试之间的退避有效（根据配置为常量、线性或指数）
- [ ] 配置时抖动应用于退避延迟
- [ ] 重试耗尽后，节点的最终结果用于边选择

### 11.6 节点处理器

- [ ] **起始处理器：** 立即返回 SUCCESS（无操作）
- [ ] **退出处理器：** 立即返回 SUCCESS（无操作，引擎检查目标门）
- [ ] **Codergen 处理器：** 扩展提示中的 `$goal`，调用 `CodergenBackend.run()`，将 prompt.md 和 response.md 写入阶段目录
- [ ] **Wait.human 处理器：** 向面试官呈现出边标签作为选项，返回选定标签作为 preferred_label
- [ ] **条件处理器：** 传递通过；引擎针对结果/上下文评估边条件
- [ ] **并行处理器：** 并发分发到多个目标节点（或顺序作为回退）
- [ ] **汇聚处理器：** 在继续之前等待所有并行分支完成
- [ ] **工具处理器：** 执行配置的工具/命令并返回结果
- [ ] 可以通过类型字符串注册自定义处理器

### 11.7 状态与上下文

- [ ] 上下文是所有处理器可访问的键值存储
- [ ] 处理器可以读取上下文并在结果中返回 `context_updates`
- [ ] 上下文更新在每个节点执行后合并
- [ ] 每个节点完成后保存检查点（current_node、completed_nodes、context、重试计数）
- [ ] 从检查点恢复：加载检查点 -> 恢复状态 -> 从 current_node 继续
- [ ] 制品写入 `{logs_root}/{node_id}/`（prompt.md、response.md、status.json）

### 11.8 人机协作

- [ ] 面试官接口工作：`ask(question) -> Answer`
- [ ] 问题支持类型：SINGLE_SELECT、MULTI_SELECT、FREE_TEXT、CONFIRM
- [ ] AutoApproveInterviewer 始终选择第一个选项（用于自动化/测试）
- [ ] ConsoleInterviewer 在终端中提示并读取用户输入
- [ ] CallbackInterviewer 委托给提供的函数
- [ ] QueueInterviewer 从预填充的答案队列读取（用于测试）

### 11.9 条件表达式

- [ ] `=`（等于）运算符适用于字符串比较
- [ ] `!=`（不等于）运算符有效
- [ ] `&&`（AND） conjunction 适用于多个子句
- [ ] `outcome` 变量解析为当前节点的结果状态
- [ ] `preferred_label` 变量解析为结果的首选标签
- [ ] `context.*` 变量解析为上下文值（缺失键 = 空字符串）
- [ ] 空条件始终评估为 true（无条件边）

### 11.10 模型样式表

- [ ] 样式表从图的 `model_stylesheet` 属性解析
- [ ] 按形状名称的选择器有效（例如，`box { model = "claude-opus-4-6" }`）
- [ ] 按类名的选择器有效（例如，`.fast { model = "gemini-3-flash-preview" }`）
- [ ] 按节点 ID 的选择器有效（例如，`#review { reasoning_effort = "high" }`）
- [ ] 特异性顺序：universal < shape < class < ID
- [ ] 样式表属性被显式节点属性覆盖

### 11.11 转换与扩展性

- [ ] AST 转换可以在解析和验证之间修改图
- [ ] 转换接口：`transform(graph) -> graph`
- [ ] 内置变量扩展转换替换提示中的 `$goal`
- [ ] 可以注册自定义转换并按顺序运行
- [ ] HTTP 服务器模式（如果实现）：POST /run 启动流水线，GET /status 检查状态，POST /answer 提交人工输入

### 11.12 跨功能奇偶校验矩阵

运行此验证矩阵 -- 每个单元格必须通过：

| Test Case                                        | Pass |
|--------------------------------------------------|------|
| 解析简单线性流水线（start -> A -> B -> done）      | [ ]   |
| 解析带有图级属性的流水线（goal、label）            | [ ]   |
| 解析多行节点属性                                  | [ ]   |
| 验证：缺少起始节点 -> 错误                        | [ ]   |
| 验证：缺少退出节点 -> 错误                         | [ ]   |
| 验证：孤立节点 -> 警告                            | [ ]   |
| 端到端执行线性 3 节点流水线                        | [ ]   |
| 使用条件分支执行（成功/失败路径）                  | [ ]   |
| 失败时执行重试（max_retries=2）                   | [ ]   |
| 目标门在未满足时阻止退出                           | [ ]   |
| 目标门在全部满足时允许退出                         | [ ]   |
| Wait.human 呈现选项并根据选择路由                  | [ ]   |
| 边选择：条件匹配胜过权重                           | [ ]   |
| 边选择：权重打破无条件边的平局                     | [ ]   |
| 边选择：词法决胜作为最终回退                       | [ ]   |
| 来自一个节点的上下文更新对下一个节点可见            | [ ]   |
| 检查点保存和恢复产生相同结果                       | [ ]   |
| 样式表按形状应用模型覆盖到节点                     | [ ]   |
| 提示变量扩展（$goal）有效                          | [ ]   |
| 并行分发和汇聚正确完成                             | [ ]   |
| 自定义处理器注册和执行有效                         | [ ]   |
| 具有 10+ 个节点的流水线成功完成，没有错误          | [ ]   |

### 11.13 集成冒烟测试

使用真实 LLM 回调的端到端测试：

```
-- 测试流水线：plan -> implement -> review -> done
DOT = """
digraph test_pipeline {
    graph [goal="Create a hello world Python script"]

    start       [shape=Mdiamond]
    plan        [shape=box, prompt="Plan how to create a hello world script for: $goal"]
    implement   [shape=box, prompt="Write the code based on the plan", goal_gate=true]
    review      [shape=box, prompt="Review the code for correctness"]
    done        [shape=Msquare]

    start -> plan
    plan -> implement
    implement -> review [condition="outcome=success"]
    implement -> plan   [condition="outcome=fail", label="Retry"]
    review -> done      [condition="outcome=success"]
    review -> implement [condition="outcome=fail", label="Fix"]
}
"""

-- 1. 解析
graph = parse_dot(DOT)
ASSERT graph.goal == "Create a hello world Python script"
ASSERT LENGTH(graph.nodes) == 5
ASSERT LENGTH(edges_total(graph)) == 6

-- 2. 验证
lint_results = validate(graph)
ASSERT no error-severity results in lint_results

-- 3. 使用 LLM 回调执行
context = Context()
outcome = run_pipeline(graph, context, llm_callback = real_llm_callback)

-- 4. 验证
ASSERT outcome.status == "success"
ASSERT "implement" in outcome.completed_nodes
ASSERT artifacts_exist(logs_root, "plan", ["prompt.md", "response.md", "status.json"])
ASSERT artifacts_exist(logs_root, "implement", ["prompt.md", "response.md", "status.json"])
ASSERT artifacts_exist(logs_root, "review", ["prompt.md", "response.md", "status.json"])

-- 5. 验证目标门
ASSERT goal_gate_satisfied(graph, outcome, "implement")

-- 6. 验证检查点
checkpoint = load_checkpoint(logs_root)
ASSERT checkpoint.current_node == "done"
ASSERT "plan" IN checkpoint.completed_nodes
ASSERT "implement" IN checkpoint.completed_nodes
ASSERT "review" IN checkpoint.completed_nodes
```

---

## 附录 A：完整属性参考

### 图属性

| Key                     | Type     | Default | Description |
|-------------------------|----------|---------|-------------|
| `goal`                  | String   | `""`    | 流水线级目标描述 |
| `label`                 | String   | `""`    | 图的显示名称 |
| `model_stylesheet`      | String   | `""`    | 类 CSS 的 LLM 模型/提供商样式表 |
| `default_max_retry`     | Integer  | `50`    | 全局重试上限 |
| `default_fidelity`      | String   | `""`    | 默认上下文保真度模式 |
| `retry_target`          | String   | `""`    | 未满足退出时跳转到的节点 |
| `fallback_retry_target` | String   | `""`    | 辅助跳转目标 |
| `stack.child_dotfile`   | String   | `""`    | 用于监督的子 DOT 文件路径 |
| `stack.child_workdir`   | String   | cwd     | 子运行的工作目录 |
| `tool_hooks.pre`        | String   | `""`    | 每次工具调用前的 shell 命令 |
| `tool_hooks.post`       | String   | `""`    | 每次工具调用后的 shell 命令 |

### 节点属性

| Key                     | Type     | Default       | Description |
|-------------------------|----------|---------------|-------------|
| `label`                 | String   | node ID       | 显示名称 |
| `shape`                 | String   | `"box"`       | Graphviz 形状（确定处理器类型） |
| `type`                  | String   | `""`          | 显式处理器类型覆盖 |
| `prompt`                | String   | `""`          | LLM 提示（支持 `$goal` 扩展） |
| `max_retries`           | Integer  | `0`           | 额外重试尝试 |
| `goal_gate`             | Boolean  | `false`       | 必须在流水线退出前成功 |
| `retry_target`          | String   | `""`          | 失败时的跳转目标 |
| `fallback_retry_target` | String   | `""`          | 辅助跳转目标 |
| `fidelity`              | String   | inherited     | 上下文保真度模式 |
| `thread_id`             | String   | derived       | 会话重用密钥 |
| `class`                 | String   | `""`          | 样式表类名（逗号分隔） |
| `timeout`               | Duration | unset         | 最大执行时间 |
| `llm_model`             | String   | inherited     | LLM 模型覆盖 |
| `llm_provider`          | String   | auto-detected | LLM 提供商覆盖 |
| `reasoning_effort`      | String   | `"high"`      | 推理深度：low/medium/high |
| `auto_status`           | Boolean  | `false`       | 如果未写入状态则自动生成 SUCCESS |
| `allow_partial`         | Boolean  | `false`       | 在重试耗尽时接受 PARTIAL_SUCCESS |

### 边属性

| Key            | Type     | Default | Description |
|----------------|----------|---------|-------------|
| `label`        | String   | `""`    | 显示标题和路由密钥 |
| `condition`    | String   | `""`    | 布尔保护表达式 |
| `weight`       | Integer  | `0`     | 边选择的优先级（越高者获胜） |
| `fidelity`     | String   | unset   | 覆盖目标节点的保真度 |
| `thread_id`    | String   | unset   | 覆盖目标节点的线程 ID |
| `loop_restart` | Boolean  | `false` | 使用新的日志目录重新启动流水线 |

---

## 附录 B：形状到处理器类型映射

| Shape           | Handler Type        | Default Behavior |
|-----------------|---------------------|------------------|
| `Mdiamond`      | `start`             | 无操作入口点 |
| `Msquare`       | `exit`              | 无操作出口点（引擎中的目标门检查） |
| `box`           | `codergen`          | LLM 任务（所有节点的默认值） |
| `hexagon`       | `wait.human`        | 阻塞以等待人工选择 |
| `diamond`       | `conditional`       | 传递通过；引擎评估边条件 |
| `component`     | `parallel`          | 并发分支执行 |
| `tripleoctagon` | `parallel.fan_in`   | 合并并行结果 |
| `parallelogram` | `tool`              | 外部工具执行 |
| `house`         | `stack.manager_loop`| 主管轮询循环 |

---

## 附录 C：状态文件契约

每个非终端节点在其阶段目录中写入 `status.json` 文件。此文件驱动路由决策并提供审计跟踪。

```
{
    "outcome": "success | retry | fail | partial_success",
    "preferred_next_label": "<edge label or empty>",
    "suggested_next_ids": ["<node_id>", ...],
    "context_updates": {
        "key": "value",
        "nested.key": "value"
    },
    "notes": "Human-readable execution summary"
}
```

| Field                  | Type            | Required | Description |
|------------------------|-----------------|----------|-------------|
| `outcome`              | String (enum)   | Yes      | 结果状态。驱动路由和目标检查。 |
| `preferred_next_label` | String          | No       | 要为下一个转换优先考虑的边标签。 |
| `suggested_next_ids`   | List of Strings | No       | 如果没有标签匹配的回退目标节点 ID。 |
| `context_updates`      | Map             | No       | 合并到运行上下文中的键值对。 |
| `notes`                | String          | No       | 人类可读的日志条目。 |

当节点上 `auto_status=true` 且处理器未写入 `status.json` 时，引擎合成：`{"outcome": "success", "notes": "auto-status: handler completed without writing status"}`。

---

## 附录 D：错误类别

流水线执行期间的每个错误都属于以下三个类别之一：

**可重试错误** 是重新执行可能成功的瞬态故障。例如：LLM 速率限制、网络超时、临时服务不可用。引擎根据节点的重试策略自动重试这些错误。

**终端错误** 是重新执行无济于事的永久性故障。例如：无效提示、缺少必需上下文、身份验证失败。引擎不会重试这些；它立即路由到失败路径。

**流水线错误** 是流水线本身的结构性故障。例如：没有起始节点、不可达的节点、无效条件。这些在验证期间（执行前）尽可能检测到。运行时检测导致流水线立即终止。
