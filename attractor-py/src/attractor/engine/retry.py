# src/attractor/engine/retry.py
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..models import Context, Graph, Node, Outcome, StageStatus


@dataclass
class BackoffConfig:
    """重试退避配置"""

    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60000
    jitter: bool = True


@dataclass
class RetryPolicy:
    """重试策略"""

    max_attempts: int
    backoff: BackoffConfig = None
    should_retry: Optional[Callable[[Exception], bool]] = None

    def __post_init__(self):
        if self.backoff is None:
            self.backoff = BackoffConfig()


def execute_with_retry(
    handler, node: Node, context: Context, graph: Graph, logs_root: str, policy: RetryPolicy
) -> Outcome:
    """使用重试策略执行节点"""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            outcome = handler.execute(node, context, graph, logs_root)
        except Exception as e:
            if policy.should_retry and policy.should_retry(e) and attempt < policy.max_attempts:
                delay = _delay_for_attempt(attempt, policy.backoff)
                time.sleep(delay / 1000)
                continue
            else:
                return Outcome(status=StageStatus.FAIL, failure_reason=str(e))

        # 成功或部分成功
        if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
            return outcome

        # 请求重试
        if outcome.status == StageStatus.RETRY:
            if attempt < policy.max_attempts:
                delay = _delay_for_attempt(attempt, policy.backoff)
                time.sleep(delay / 1000)
                continue
            else:
                # 重试耗尽
                if node.allow_partial:
                    return Outcome(
                        status=StageStatus.PARTIAL_SUCCESS,
                        notes="retries exhausted, partial accepted",
                    )
                return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")

        # 失败
        if outcome.status == StageStatus.FAIL:
            return outcome

    return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")


def _delay_for_attempt(attempt: int, config: BackoffConfig) -> int:
    """计算重试延迟（毫秒）"""
    delay = config.initial_delay_ms * (config.backoff_factor ** (attempt - 1))
    delay = min(delay, config.max_delay_ms)
    if config.jitter:
        delay = delay * random.uniform(0.5, 1.5)
    return int(delay)
