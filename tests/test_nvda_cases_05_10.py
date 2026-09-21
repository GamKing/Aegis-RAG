"""NVIDIA Case 05-10 回归测试。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from rag_eval.dummy_rag import DummyRAG, RAGResponse
from rag_eval.evaluators import CompletenessEvaluator, FaithfulnessEvaluator
from rag_eval.models import EvalSample
from rag_eval.runner import EvalConfig, run_eval_pipeline

FIXTURE = Path(__file__).with_name("nvda_cases_05_10.json")


def load_samples() -> list[EvalSample]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [EvalSample.model_validate(item) for item in payload]


class NvidiaCases0510Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load_samples()

    def test_fixture_contract(self):
        self.assertEqual(len(self.samples), 6)
        self.assertEqual(self.samples[0].id, "nvda_case_05_datacenter")
        self.assertEqual(self.samples[-1].id, "nvda_case_10_negative_unmentioned")

    def test_faithfulness_expected_boundaries(self):
        by_id = {}
        for sample in self.samples:
            by_id[sample.id] = FaithfulnessEvaluator().evaluate(
                sample, sample.ground_truth_context, sample.ground_truth_answer
            )

        # 05：89.0 billion <-> 890 亿，数字与百分比均有出处。
        self.assertTrue(by_id["nvda_case_05_datacenter"].passed)
        # 06：26.0/99.0 billion <-> 260/990 亿。
        self.assertTrue(by_id["nvda_case_06_share_repurchase"].passed)
        # 07：税率区间与排除前提没有新增数字，应该通过。
        self.assertTrue(by_id["nvda_case_07_full_year_tax"].passed)
        # 08：上下文为 7.2 billion，答案篡改为 9.5 billion，应失败。
        self.assertFalse(by_id["nvda_case_08_adversarial_hallucination"].passed)
        # 09：9.2/9.0 billion <-> 92/90 亿，应通过。
        self.assertTrue(by_id["nvda_case_09_operating_expenses"].passed)
        # 10：答案没有具体数字，Faithfulness 对空数字集平凡通过。
        self.assertTrue(by_id["nvda_case_10_negative_unmentioned"].passed)

    def test_completeness_expected_boundaries(self):
        by_id = {}
        for sample in self.samples:
            by_id[sample.id] = CompletenessEvaluator(threshold=0.80).evaluate(
                sample, sample.ground_truth_context, sample.ground_truth_answer
            )

        self.assertEqual(by_id["nvda_case_05_datacenter"].score, 1.0)
        self.assertEqual(by_id["nvda_case_06_share_repurchase"].score, 1.0)
        # 中英文 expected_entities 与中文答案存在语言不一致，严格字面匹配会部分失败。
        self.assertLess(by_id["nvda_case_07_full_year_tax"].score, 1.0)
        self.assertEqual(by_id["nvda_case_08_adversarial_hallucination"].score, 0.0)
        self.assertEqual(by_id["nvda_case_09_operating_expenses"].score, 1.0)
        self.assertEqual(by_id["nvda_case_10_negative_unmentioned"].score, 0.0)

    def test_pipeline_batch_runs_all_six(self):
        presets = {
            sample.id: RAGResponse(
                context=sample.ground_truth_context,
                answer=sample.ground_truth_answer,
            )
            for sample in self.samples
        }
        summary = run_eval_pipeline(self.samples, DummyRAG(presets), EvalConfig())
        self.assertEqual(summary.total, 6)
        self.assertEqual(len(summary.results), 6)
        self.assertEqual(
            {result.sample_id for result in summary.results},
            {sample.id for sample in self.samples},
        )


if __name__ == "__main__":
    unittest.main()
