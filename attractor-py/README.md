# Attractor

基于 DOT 的流水线执行器，用于编排多阶段 AI 工作流。

## 功能

- 使用 Graphviz DOT 语法定义流水线
- 声明式图遍历执行
- 可插拔的节点处理器
- 检查点与恢复
- 人机协作（面试官模式）
- 条件路由
- 重试策略

## 安装

```bash
uv pip install attractor
```

## 快速开始

```python
from attractor import parse_dot, PipelineExecutor, ExecutorConfig

# 定义流水线
dot_source = '''
digraph MyPipeline {
    graph [goal="Complete the task"]
    start [shape=Mdiamond]
    task [shape=box, prompt="Do the work"]
    exit [shape=Msquare]

    start -> task -> exit
}
'''

# 解析
graph = parse_dot(dot_source)

# 配置并执行
config = ExecutorConfig(logs_root="./logs")
executor = PipelineExecutor(config)
result = executor.run(graph)

print(f"Status: {result.status}")
print(f"Completed: {result.completed_nodes}")
```

## 节点类型

| 形状 | 处理器 | 描述 |
|------|--------|------|
| `Mdiamond` | start | 流水线入口点 |
| `Msquare` | exit | 流水线出口点 |
| `box` | codergen | LLM 任务 |
| `hexagon` | wait.human | 人机协作门控 |
| `diamond` | conditional | 条件路由 |
| `parallelogram` | tool | 外部工具执行 |

## 许可证

MIT
