# Attractor

本仓库包含 [NLSpecs](#terminology)，用于构建您自己的 Attractor 版本，以创建您自己的软件工厂。

虽然构建您自己的 Attractor 不一定要自建代理循环（agentic loop）和统一的 LLM SDK，但我们强烈建议您控制整个技术栈，这样您就能拥有坚实的基础。

## 规范

- [Attractor 规范](./attractor-spec.md)
- [编码代理循环规范](./coding-agent-loop-spec.md)
- [统一 LLM 客户端规范](./unified-llm-spec.md)

## 构建 Attractor

向现代编码代理（Claude Code、Codex、OpenCode、Amp、Cursor 等）提供以下提示：

```
codeagent> 按照 https://factory.strongdm.ai/ 的描述实现 Attractor
```

## 术语

- **NLSpec**（自然语言规范）：一种人类可读的规范，旨在由编码代理直接用于实现/验证行为。
