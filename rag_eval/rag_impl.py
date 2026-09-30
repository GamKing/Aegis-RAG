"""将真实检索模块与受控生成流水线组装为可评测的 RAG 后端。

ProductionRAG 复用 pipeline.PipelineRAG 的受控生成（受限抽取 + Tier1 质检 +
受控合成），仅将检索阶段替换为检索模块的实际召回结果，保持 BaseRAG /
RAGResponse 契约不变，从而直接接进 runner.py 的既有评测链路。
"""
from __future__ import annotations

from typing import Sequence

from .dummy_rag import BaseRAG, RAGResponse
from .models import EvalSample
from .retrieval.base import BaseRetriever
from .retrieval.source_scope import select_context
from .pipeline import (
    PipelineRAG,
    RuleBasedFactExtractor,
    ControlledSynthesizer,
    Tier1CodeVerifier,
)


class ProductionRAG(BaseRAG):
    """检索模块 + 受控生成流水线的组装体。"""

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 3,
        generator: PipelineRAG | None = None,
        self_corrector=None,
    ) -> None:
        self.retriever = retriever
        self.top_k = top_k
        # 生成端默认走既有受控流水线；可注入自定义 PipelineRAG 以便替换抽取/合成器。
        # self_corrector 为可选的自愈重试器（如 NLIAndSelfCorrector 挂载本地模型），
        # 注入后 Tier-1 失败会自动反哺抽取端重试一次，形成自愈闭环。
        self.generator = generator or PipelineRAG(
            context_provider=lambda sample: "",
            extractor=RuleBasedFactExtractor(),
            synthesizer=ControlledSynthesizer(),
            verifier=Tier1CodeVerifier(),
            self_corrector=self_corrector,
        )

    def _retrieve_context(self, sample: EvalSample) -> str:
        return self.retriever.retrieve_context(sample.question, top_k=self.top_k)

    def retrieve_top_k(self, query: str, k: int) -> list[str]:
        """返回检索 Top-K 块的内容文本列表（供 Hit@K 评分）。"""
        results = self.retriever.retrieve(query, top_k=k)
        return [r.chunk.content for r in results]

    def retrieve_and_generate(self, sample: EvalSample,
                              requested_sources: Sequence[str] | None = None) -> RAGResponse:
        hits = self.retriever.retrieve(sample.question, top_k=self.top_k)
        # 仅消费用户问题/显式请求来源，不读取 sample.source_id 或标准答案。
        selection = select_context(sample.question, hits, requested_sources)
        return self.generator.generate_from_context(sample, selection.context)
