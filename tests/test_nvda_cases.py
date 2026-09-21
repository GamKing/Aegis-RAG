"""NVIDIA 外部样本回归测试。

这些测试不是把英文上下文强行判为中文答案的满分，而是记录当前 v1
纯代码评分器的真实行为，防止后续修改悄悄改变结果。
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from rag_eval.evaluators import (
    CompletenessEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
)
from rag_eval.models import EvalSample
from rag_eval.runner import EvalConfig, run_eval_pipeline
from rag_eval.dummy_rag import DummyRAG, RAGResponse


class AlwaysEntailedJudge:
    def check_faithfulness(self, context: str, claim: str) -> bool:
        return "not assuming" in context and "未计入" in claim


FIXTURE = Path(__file__).with_name("nvda_cases.json")


def load_nvda_samples() -> list[EvalSample]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [EvalSample.model_validate(item) for item in payload]


class NvidiaFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load_nvda_samples()

    def test_fixture_has_four_contract_valid_samples(self):
        self.assertEqual(len(self.samples), 4)
        self.assertEqual(
            [s.id for s in self.samples],
            ["nvda_case_01", "nvda_case_02", "nvda_case_03", "nvda_case_04"],
        )
        for sample in self.samples:
            self.assertTrue(sample.question)
            self.assertTrue(sample.ground_truth_context)
            self.assertTrue(sample.ground_truth_answer)
            self.assertGreater(len(sample.expected_entities), 0)

    def test_ground_truth_answers_expose_current_metric_boundaries(self):
        """标准答案作为 actual answer 时，记录 v1 的可解释边界行为。"""
        outcomes = {}
        for sample in self.samples:
            recall = ContextRecallEvaluator().evaluate(
                sample, sample.ground_truth_context, sample.ground_truth_answer
            )
            completeness = CompletenessEvaluator(threshold=0.80).evaluate(
                sample, sample.ground_truth_context, sample.ground_truth_answer
            )
            faithfulness = FaithfulnessEvaluator().evaluate(
                sample, sample.ground_truth_context, sample.ground_truth_answer
            )
            outcomes[sample.id] = (recall, completeness, faithfulness)

        # 01：英文上下文中的 million 与答案中的“亿”已归一化，忠实性通过。
        self.assertEqual(outcomes["nvda_case_01"][1].score, 1.0)
        self.assertTrue(outcomes["nvda_case_01"][2].passed)

        # 02：英文月份与中文数字月份已归一化，忠实性通过。
        self.assertEqual(outcomes["nvda_case_02"][1].score, 1.0)
        self.assertTrue(outcomes["nvda_case_02"][2].passed)

        # 03：标准答案不再额外推断“0”，因此数字忠实性通过；
        # 但英文 “not assuming” 与中文“未计入”不是同一字面实体，完整性仍按
        # 当前严格子串规则保守判定为未完全命中。
        self.assertTrue(outcomes["nvda_case_03"][2].passed)
        self.assertLess(outcomes["nvda_case_03"][1].score, 1.0)

        # 04：答案将 $108.0 billion 改写为 1080 亿，经过单位归一化后通过。
        self.assertTrue(outcomes["nvda_case_04"][2].passed)
        self.assertEqual(outcomes["nvda_case_04"][1].score, 1.0)

    def test_case_03_can_pass_with_nli_semantic_judge(self):
        sample = next(s for s in self.samples if s.id == "nvda_case_03")
        candidate_answer = sample.ground_truth_answer + "（假设为 0）"
        outcome = FaithfulnessEvaluator(nli_judge=AlwaysEntailedJudge()).evaluate(
            sample, sample.ground_truth_context, candidate_answer
        )
        self.assertTrue(outcome.passed)
        self.assertTrue(any("NLI" in detail for detail in outcome.details))

    def test_pipeline_runs_all_four_cases_without_crashing(self):
        presets = {
            s.id: RAGResponse(context=s.ground_truth_context, answer=s.ground_truth_answer)
            for s in self.samples
        }
        summary = run_eval_pipeline(self.samples, DummyRAG(presets), EvalConfig())
        self.assertEqual(summary.total, 4)
        self.assertEqual(len(summary.results), 4)
        self.assertEqual({r.sample_id for r in summary.results}, {s.id for s in self.samples})


if __name__ == "__main__":
    unittest.main()
