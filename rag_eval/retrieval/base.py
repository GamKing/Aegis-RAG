"""检索器统一接口契约。"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class DocumentChunk:
    """检索文档单元。

    metadata 约定字段：
    - index: 块在文档中的序号
    - doc_id: 源文档标识
    - section: 所属章节（法律场景）
    - kind: 块类型（legal_clause / financial_table_row 等）
    - page: 页码（PDF 场景）
    - line_start / line_end: 行号范围（溯源用）
    """

    chunk_id: str
    content: str
    metadata: dict | None = field(default_factory=dict)


@dataclass
class SearchResult:
    """带得分的检索结果项。"""

    chunk: DocumentChunk
    score: float


class BaseChunker(abc.ABC):
    """分块器基类。

    行业插槽 1：各行业通过继承并实现 split_text，在不改动主链路的前提下
    提供"行业感知解析"。例如：
    - 法律：LegalHierarchyChunker 按"第X条"切分并保留父标题；
    - 金融财报：FinancialTableChunker 将财务报表逐行转成语义描述。
    """

    @abc.abstractmethod
    def split_text(self, text: str, doc_id: str = "doc") -> list[DocumentChunk]:
        """将一段文本切分为若干文档块。"""
        raise NotImplementedError


class BaseRetriever(abc.ABC):
    """检索器基类。"""

    @abc.abstractmethod
    def index_documents(self, documents: Sequence[DocumentChunk]) -> None:
        """建立索引。"""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """执行召回。"""
        raise NotImplementedError

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """直接组装为供 RAG 生成或评测用的合并字符串。"""
        results = self.retrieve(query, top_k=top_k)
        return "\n\n".join(r.chunk.content for r in results)