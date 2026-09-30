"""双路混合检索与 RRF 融合：纯 Python 实现，零外部依赖可离线运行。

核心思路：BM25 做字面匹配（专有名词/数字/编码搜得准），
向量引擎做语义相似度（意图匹配，可插拔），两者经 RRF 倒数融合排序，
取 Top-K 作为 actual_context。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .base import BaseRetriever, DocumentChunk, SearchResult


class SimpleBM25:
    """轻量级纯 Python BM25，无外部 C 依赖，可离线运行。

    行业插槽 2：可通过 tokenizer 注入行业专有词表，确保"奥沙利铂"、
    "索拉非尼"这类专有名词作为不可分割的单 token 参与检索。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None) -> None:
        self.k1 = k1
        self.b = b
        # 自定义分词器；未注入时使用默认的 CJK bigram + 英文/数字
        self.tokenizer = tokenizer
        self.corpus: list[DocumentChunk] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """中文二元分词 (CJK bigram) + 英文/数字 token。

        若注入了自定义 tokenizer（行业词表），则先做专有名词保护。
        """
        if self.tokenizer is not None:
            protected = self.tokenizer.protect(text)
            # 对保护后的文本跑默认 bigram 分词，行业专有名词以整体 token 保留
            tokens: list[str] = []
            tokens.extend(re.findall(r"[a-zA-Z0-9_]+", protected.lower()))
            chinese_chars = re.findall(r"[\u4e00-\u9fff]", protected)
            for i in range(len(chinese_chars) - 1):
                tokens.append(chinese_chars[i] + chinese_chars[i + 1])
            return tokens
        return self._default_tokenize(text)

    def _default_tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        return tokens

    def fit(self, documents: Sequence[DocumentChunk]) -> None:
        self.corpus = list(documents)
        total_len = 0
        self.doc_freqs = Counter()

        for doc in self.corpus:
            tokens = set(self._tokenize(doc.content))
            for t in tokens:
                self.doc_freqs[t] += 1
            length = len(tokens)
            self.doc_lengths.append(length)
            total_len += length

        n_docs = len(self.corpus)
        self.avg_doc_len = total_len / n_docs if n_docs > 0 else 0.0

        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        q_tokens = self._tokenize(query)
        scores: list[float] = []

        for idx, doc in enumerate(self.corpus):
            doc_tokens = self._tokenize(doc.content)
            tf = Counter(doc_tokens)
            doc_len = self.doc_lengths[idx]
            score = 0.0

            for t in q_tokens:
                if t not in tf:
                    continue
                term_tf = tf[t]
                idf = self.idf.get(t, 0.0)
                denom = term_tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
                )
                score += idf * (term_tf * (self.k1 + 1.0)) / (denom or 1.0)

            scores.append(score)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            SearchResult(chunk=self.corpus[i], score=scores[i])
            for i in ranked_indices
            if scores[i] > 0
        ]


class HybridRetriever(BaseRetriever):
    """混合检索器：BM25 词面匹配 + 向量相似度（按需插拔）+ RRF 融合。

    未配置 vector_engine 时自动退化为纯 BM25，零外部依赖直接跑通。

    行业插槽 2：可通过 tokenizer 参数注入行业专有词表（如医学本体库），
    保证专有名词作为整体参与 BM25 检索。
    """

    def __init__(self, vector_engine=None, rrf_k: int = 60, tokenizer=None) -> None:
        self.bm25 = SimpleBM25(tokenizer=tokenizer)
        self.vector_engine = vector_engine  # 预留可注入向量库接口
        self.rrf_k = rrf_k
        self.documents: list[DocumentChunk] = []

    def index_documents(self, documents: Sequence[DocumentChunk]) -> None:
        self.documents = list(documents)
        self.bm25.fit(documents)
        if self.vector_engine:
            self.vector_engine.index_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        # 未配置向量引擎，直接退化为纯 BM25 检索
        if not self.vector_engine:
            return bm25_results[:top_k]

        vector_results = self.vector_engine.search(query, top_k=top_k * 2)

        # RRF (Reciprocal Rank Fusion) 排名倒数加权融合
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, res in enumerate(bm25_results):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, res in enumerate(vector_results):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_cids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )[:top_k]
        return [
            SearchResult(chunk=chunk_map[cid], score=rrf_scores[cid])
            for cid in sorted_cids
        ]