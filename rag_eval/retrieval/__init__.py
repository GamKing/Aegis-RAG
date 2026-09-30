"""检索模块：从 Query 到高质量 actual_context 的召回链路。

三层架构：
- 稀疏检索 BM25（专有名词/数字/编码的字面匹配）
- 稠密向量（语义相似度，可插拔，未配置时退化为纯 BM25）
- RRF 倒数融合排序 + Top-K 截断

对外仅暴露稳定契约 DocumentChunk / SearchResult / BaseRetriever，
以及开箱即用的 RecursiveChunker 与 HybridRetriever。
"""
from __future__ import annotations

from .base import BaseChunker, BaseRetriever, DocumentChunk, SearchResult
from .chunker import (
    FinancialTableChunker,
    LegalHierarchyChunker,
    RecursiveChunker,
)
from .hybrid_retriever import HybridRetriever, SimpleBM25

__all__ = [
    "BaseChunker",
    "BaseRetriever",
    "DocumentChunk",
    "SearchResult",
    "RecursiveChunker",
    "LegalHierarchyChunker",
    "FinancialTableChunker",
    "HybridRetriever",
    "SimpleBM25",
]