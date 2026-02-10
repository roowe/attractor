# src/attractor/handlers/human.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union

from ..models import Context, Graph, Node, Outcome, StageStatus
from .interface import Handler


class QuestionType(Enum):
    """问题类型"""

    YES_NO = auto()
    MULTIPLE_CHOICE = auto()
    FREEFORM = auto()
    CONFIRMATION = auto()


@dataclass
class Option:
    """多选项选项"""

    key: str
    label: str


@dataclass
class Question:
    """人类交互问题"""

    text: str
    type: QuestionType
    options: List[Option] = None
    default: Optional["Answer"] = None
    timeout_seconds: Optional[float] = None
    stage: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.metadata is None:
            self.metadata = {}


class AnswerValue(Enum):
    """答案值"""

    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Answer:
    """问题答案"""

    value: Union[str, AnswerValue]
    selected_option: Optional[Option] = None
    text: str = ""


class Interviewer(ABC):
    """面试官接口"""

    @abstractmethod
    def ask(self, question: Question) -> Answer:
        """提出问题并等待答案"""
        pass


class WaitForHumanHandler(Handler):
    """等待人工输入处理器"""

    def __init__(self, interviewer: Interviewer) -> None:
        self._interviewer = interviewer

    def execute(self, node: Node, context: Context, graph: Graph, logs_root: str) -> Outcome:
        # 1. 从出边派生选项
        edges = [e for e in graph.edges if e.from_node == node.id]

        if not edges:
            return Outcome(
                status=StageStatus.FAIL, failure_reason="No outgoing edges for human gate"
            )

        choices = []
        for edge in edges:
            label = edge.label or edge.to_node
            key = self._parse_accelerator_key(label)
            choices.append((key, label, edge.to_node))

        # 2. 构建问题
        options = [Option(key=k, label=l) for k, l, _ in choices]
        question = Question(
            text=node.label or "Select an option:",
            type=QuestionType.MULTIPLE_CHOICE,
            options=options,
            stage=node.id,
        )

        # 3. 呈现给面试官并等待答案
        answer = self._interviewer.ask(question)

        # 4. 处理答案
        selected = self._find_matching_choice(answer, choices)
        if selected is None:
            selected = choices[0]  # 回退到第一个

        _, label, to_node = selected

        # 5. 返回结果
        return Outcome(
            status=StageStatus.SUCCESS,
            suggested_next_ids=[to_node],
            context_updates={"human.gate.selected": selected[0], "human.gate.label": label},
        )

    def _parse_accelerator_key(self, label: str) -> str:
        """从标签解析加速器键"""
        import re

        # 模式: [K] Label, K) Label, K - Label
        patterns = [
            r"\[([A-Za-z0-9])\]",  # [K] Label
            r"([A-Za-z0-9])\)",  # K) Label
            r"([A-Za-z0-9])\s*-",  # K - Label
        ]
        for pattern in patterns:
            match = re.match(pattern, label.strip())
            if match:
                return match.group(1).upper()
        # 默认: 首字符
        return label[0].upper() if label else ""

    def _find_matching_choice(self, answer: Answer, choices: list) -> Optional[tuple]:
        """查找匹配的选择"""
        answer_key = str(answer.value).upper()

        # 首先尝试精确键匹配
        for key, label, to_node in choices:
            if key == answer_key:
                return (key, label, to_node)

        # 然后尝试标签匹配
        for key, label, to_node in choices:
            if label.upper().startswith(answer_key):
                return (key, label, to_node)

        return None
