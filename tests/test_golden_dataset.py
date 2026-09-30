"""自动化评测集工程（Day 1-2）测试：数据生成、Hit@K、剪刀差排障。"""
import unittest

from rag_eval.data_generator import (
    MUTATIONS,
    generate_dataset,
    mutate_digit_swap,
    mutate_drop_condition,
    mutate_drop_entity,
    mutate_inject_fabrication,
)
from rag_eval.dataset import SAMPLE_001
from rag_eval.evaluators import HitAtKEvaluator


class TestDataGenerator(unittest.TestCase):
    def test_generates_100_samples(self):
        ds = generate_dataset(seed=42)
        self.assertGreaterEqual(len(ds), 100)

    def test_deterministic_with_seed(self):
        ds1 = generate_dataset(seed=7)
        ds2 = generate_dataset(seed=7)
        self.assertEqual(
            [s.id for s in ds1],
            [s.id for s in ds2],
        )

    def test_has_three_categories(self):
        ds = generate_dataset()
        ids = [s.id for s in ds]
        has_synth = any(i.startswith("syn-") for i in ids)
        has_edge = any(i.startswith("edge-") for i in ids)
        has_adv = any(i.startswith("adv-") for i in ids)
        self.assertTrue(has_synth)
        self.assertTrue(has_edge)
        self.assertTrue(has_adv)

    def test_samples_have_full_fields(self):
        for s in generate_dataset()[:10]:
            self.assertTrue(s.question)
            self.assertTrue(s.ground_truth_context)
            self.assertTrue(s.ground_truth_answer)
            self.assertTrue(s.expected_entities)


class TestMutations(unittest.TestCase):
    """对抗变异应构造出与标准样本不同的答案。"""

    def test_digit_swap_changes_answer(self):
        mutated = mutate_digit_swap(SAMPLE_001)
        self.assertNotEqual(mutated.ground_truth_answer, SAMPLE_001.ground_truth_answer)

    def test_drop_entity_removes_entity(self):
        mutated = mutate_drop_entity(SAMPLE_001)
        # 至少删除一个期望实体
        dropped = [e for e in SAMPLE_001.expected_entities
                   if e not in mutated.ground_truth_answer]
        self.assertTrue(dropped)

    def test_drop_condition(self):
        s = SAMPLE_001  # 无例外条款，应保留原样
        mutated = mutate_drop_condition(s)
        self.assertEqual(mutated.ground_truth_answer, s.ground_truth_answer)

    def test_inject_fabrication_appends(self):
        mutated = mutate_inject_fabrication(SAMPLE_001)
        self.assertIn("补充商业保险", mutated.ground_truth_answer)
        self.assertIn("补充商业保险", mutated.ground_truth_answer)

    def test_four_mutation_types(self):
        names = {m.name for m in MUTATIONS}
        self.assertEqual(
            names,
            {"无关注入", "数字篡改", "条件丢失", "实体缺失"},
        )


class TestHitAtK(unittest.TestCase):
    def test_full_coverage_when_context_present(self):
        ev = HitAtKEvaluator(threshold=0.80, k=1)
        outcome = ev.evaluate(
            SAMPLE_001,
            actual_context=SAMPLE_001.ground_truth_context,
            actual_answer="",
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)

    def test_low_coverage_when_context_absent(self):
        ev = HitAtKEvaluator(threshold=0.80, k=1)
        outcome = ev.evaluate(
            SAMPLE_001,
            actual_context="无关内容。",
            actual_answer="",
        )
        self.assertFalse(outcome.passed)

    def test_with_top_k_provider(self):
        class Provider:
            def retrieve_top_k(self, query, k):
                return [SAMPLE_001.ground_truth_context]

        ev = HitAtKEvaluator(threshold=0.80, k=1, top_k_provider=Provider())
        outcome = ev.evaluate(SAMPLE_001, actual_context="", actual_answer="")
        self.assertTrue(outcome.passed)


class TestProductionRetrieveTopK(unittest.TestCase):
    def test_retrieve_top_k_returns_blocks(self):
        from rag_eval.retrieval.chunker import RecursiveChunker
        from rag_eval.retrieval.hybrid_retriever import HybridRetriever
        from rag_eval.rag_impl import ProductionRAG

        chunker = RecursiveChunker(chunk_size=100)
        chunks = chunker.split_text(SAMPLE_001.ground_truth_context, doc_id="s1")
        retriever = HybridRetriever()
        retriever.index_documents(chunks)
        rag = ProductionRAG(retriever, top_k=3)

        blocks = rag.retrieve_top_k(SAMPLE_001.question, 2)
        self.assertGreaterEqual(len(blocks), 1)
        self.assertIsInstance(blocks[0], str)


if __name__ == "__main__":
    unittest.main()