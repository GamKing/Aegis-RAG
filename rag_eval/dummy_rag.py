"""RAG 后端抽象与 Dummy 实现。

BaseRAG 是真实系统的接入点：把你们的检索器 + 生成器包进
retrieve_and_generate，即可直接复用整套评测逻辑。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from .models import EvalSample


@dataclass(frozen=True)
class RAGResponse:
    """RAG 链路的单次产出：检索上下文 + 生成答案。"""

    context: str
    answer: str


class BaseRAG(abc.ABC):
    """评测所依赖的最小后端接口。"""

    @abc.abstractmethod
    def retrieve_and_generate(self, sample: EvalSample) -> RAGResponse:
        raise NotImplementedError


class DummyRAG(BaseRAG):
    """查表式假 RAG。

    - 命中预设：返回预设的 (context, answer)，用于离线构造各类失败模式；
    - 未命中：返回空检索 + 拒答，模拟最差链路，保证 harness 不会因未知样本崩溃。
    """

    def __init__(self, presets: dict) -> None:
        self._presets = dict(presets)

    def retrieve_and_generate(self, sample: EvalSample) -> RAGResponse:
        preset = self._presets.get(sample.id)
        if preset is not None:
            return preset
        return RAGResponse(context="", answer="抱歉，未能检索到相关政策内容。")
