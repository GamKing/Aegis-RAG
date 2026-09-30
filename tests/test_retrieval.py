"""检索模块的确定性单元测试：分块器、BM25、混合检索、ProductionRAG 装配。"""
from __future__ import annotations

import unittest

from rag_eval.dataset import load_samples
from rag_eval.evaluators import ContextRecallEvaluator
from rag_eval.models import EvalSample
from rag_eval.rag_impl import ProductionRAG
from rag_eval.retrieval.base import BaseRetriever, DocumentChunk, SearchResult
from rag_eval.retrieval.chunker import RecursiveChunker
from rag_eval.retrieval.hybrid_retriever import HybridRetriever, SimpleBM25
from rag_eval.runner import EvalConfig, build_default_evaluators, evaluate_sample


def _sample(**kwargs) -> EvalSample:
    base = dict(
        id="unit-test",
        question="q",
        ground_truth_context="c",
        ground_truth_answer="a",
        expected_entities=[],
    )
    base.update(kwargs)
    return EvalSample(**base)


class RecursiveChunkerTests(unittest.TestCase):
    def test_paragraph_first_splitting(self):
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = RecursiveChunker(chunk_size=100).split_text(text, doc_id="d")
        # 三个短段落应合并成一块（总长 < chunk_size）
        self.assertEqual(len(chunks), 1)
        self.assertIn("第一段", chunks[0].content)
        self.assertIn("第三段", chunks[0].content)

    def test_long_paragraph_falls_back_to_sentence_split(self):
        text = "甲。乙。丙。丁。戊。己。庚。辛。"
        chunks = RecursiveChunker(chunk_size=6, chunk_overlap=0).split_text(text, doc_id="d")
        self.assertGreater(len(chunks), 1)

    def test_overlap_window_carries_tail_into_next_chunk(self):
        # 两段各 10 字，chunk_size=10，overlap=3：第二块应带上前一块末尾 3 字
        text = "一二三四五六七八九十\n\n甲乙丙丁戊己庚辛壬癸"
        chunks = RecursiveChunker(chunk_size=10, chunk_overlap=3).split_text(
            text, doc_id="d"
        )
        self.assertGreaterEqual(len(chunks), 2)
        # 第二块内容应包含第一块末尾的字符
        self.assertIn("八九十", chunks[1].content)

    def test_chunk_id_and_metadata(self):
        chunks = RecursiveChunker(chunk_size=50).split_text("段一。\n\n段二。", doc_id="docX")
        self.assertTrue(all(c.chunk_id.startswith("docX_c") for c in chunks))
        self.assertEqual(chunks[0].metadata["doc_id"], "docX")


class SimpleBM25Tests(unittest.TestCase):
    def test_cjk_bigram_and_ascii_tokenize(self):
        bm25 = SimpleBM25()
        tokens = bm25._tokenize("医保报销70%与eGFR")
        # 英文/数字 token
        self.assertIn("egfr", tokens)
        self.assertIn("70", tokens)
        # 中文 bigram
        self.assertIn("医保", tokens)
        self.assertIn("报销", tokens)

    def test_discriminative_ranking(self):
        docs = [
            DocumentChunk("d0", "高血压门诊慢特病保障范围"),
            DocumentChunk("d1", "城乡居民医保住院报销比例"),
            DocumentChunk("d2", "高血压门诊慢特病统筹基金限额"),
        ]
        bm25 = SimpleBM25()
        bm25.fit(docs)
        results = bm25.search("高血压门诊", top_k=2)
        self.assertEqual(len(results), 2)
        # 含"高血压门诊"的两篇应排在前面
        top_ids = {r.chunk.chunk_id for r in results}
        self.assertEqual(top_ids, {"d0", "d2"})

    def test_empty_query_returns_no_results(self):
        bm25 = SimpleBM25()
        bm25.fit([DocumentChunk("d0", "内容")])
        self.assertEqual(bm25.search("", top_k=5), [])


class HybridRetrieverTests(unittest.TestCase):
    def test_pure_bm25_fallback_when_no_vector_engine(self):
        docs = [DocumentChunk("d0", "医保报销70%"), DocumentChunk("d1", "其他内容")]
        retriever = HybridRetriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("医保报销", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.chunk_id, "d0")

    def test_top_k_truncation(self):
        docs = [DocumentChunk(f"d{i}", f"内容{i}") for i in range(10)]
        retriever = HybridRetriever()
        retriever.index_documents(docs)
        results = retriever.retrieve("内容", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_rrf_fusion_with_fake_vector_engine(self):
        # 构造一个假向量引擎：给 d0 高分，BM25 给 d1 高分，RRF 应让两者都进入 top_k
        class FakeVector:
            def index_documents(self, documents):
                self.docs = list(documents)

            def search(self, query, top_k=10):
                # 让 d1 排在第一位
                return [
                    SearchResult(self.docs[1], score=0.9),
                    SearchResult(self.docs[0], score=0.1),
                ]

        docs = [DocumentChunk("d0", "BM25命中"), DocumentChunk("d1", "向量命中")]
        retriever = HybridRetriever(vector_engine=FakeVector())
        retriever.index_documents(docs)
        results = retriever.retrieve("测试", top_k=2)
        # 两路都召回，RRF 后应都出现在结果中
        ids = {r.chunk.chunk_id for r in results}
        self.assertEqual(ids, {"d0", "d1"})


class ProductionRAGTests(unittest.TestCase):
    def test_implements_base_rag_contract(self):
        from rag_eval.dummy_rag import BaseRAG

        docs = [DocumentChunk("d0", "医保报销比例70%")]
        retriever = HybridRetriever()
        retriever.index_documents(docs)
        rag = ProductionRAG(retriever, top_k=1)
        self.assertIsInstance(rag, BaseRAG)
        sample = _sample(question="医保报销比例")
        response = rag.retrieve_and_generate(sample)
        # 返回的 context 应来自检索
        self.assertIn("医保", response.context)

    def test_wires_through_runner(self):
        samples = load_samples()
        chunks = []
        for s in samples:
            chunks.extend(
                RecursiveChunker(chunk_size=300, chunk_overlap=50).split_text(
                    s.ground_truth_context, doc_id=s.id
                )
            )
        retriever = HybridRetriever()
        retriever.index_documents(chunks)
        rag = ProductionRAG(retriever, top_k=3)

        config = EvalConfig()
        evaluators = build_default_evaluators(config)
        for s in samples:
            res = evaluate_sample(s, rag, evaluators, config)
            # 召回应接近 1.0（库内只有 5 个短文档）
            self.assertGreaterEqual(res.context_recall_score, 0.9)


class RetrievalIntegrationTests(unittest.TestCase):
    def test_full_retrieval_eval_against_dataset(self):
        """用真实检索模块跑完整评测链路，验证召回率全绿。"""
        from rag_eval.run_retrieval_eval import run_retrieval_eval

        summary = run_retrieval_eval(config=EvalConfig(), top_k=3)
        self.assertEqual(summary.total, 5)
        # 召回率应全部 >= 0.9（库内文档短，top_k=3 能覆盖大部分）
        for r in summary.results:
            self.assertGreaterEqual(
                r.context_recall_score, 0.9, f"{r.sample_id} recall too low"
            )


if __name__ == "__main__":
    unittest.main()