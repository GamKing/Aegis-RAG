"""评分器与主管线的确定性单元测试（stdlib unittest，无额外依赖）。"""
from __future__ import annotations

import unittest

from rag_eval import text_utils as tu
from rag_eval.evaluators import (
    CompletenessEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
)
from rag_eval.main import build_demo_rag
from rag_eval.models import EvalSample
from rag_eval.runner import run_eval_pipeline
from rag_eval.dataset import load_samples


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


class FaithfulnessEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = FaithfulnessEvaluator()
        self.sample = _sample()

    def test_supported_numbers_pass(self):
        outcome = self.evaluator.evaluate(
            self.sample,
            "报销比例为70%，限额1500元",
            "报销比例为70%，限额1500元。",
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.details, [])

    def test_tampered_numbers_fail_with_two_details(self):
        outcome = self.evaluator.evaluate(
            self.sample,
            "限额1500元，报销比例70%",
            "限额2500元，报销比例85%。",
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(len(outcome.details), 2)

    def test_prefix_substring_not_counted(self):
        # 上下文是 15000，答案说 1500 —— 不得因子串前缀而误判为有出处
        outcome = self.evaluator.evaluate(
            self.sample, "限额15000元", "限额1500元。"
        )
        self.assertFalse(outcome.passed)

    def test_decimal_boundary_not_counted(self):
        # 上下文是 13.5，答案说 3.5 —— 小数点边界必须拦截
        outcome = self.evaluator.evaluate(
            self.sample, "费用13.5元", "费用3.5元。"
        )
        self.assertFalse(outcome.passed)

    def test_percent_unit_mismatch_rejected(self):
        # 答案 85% 不得被上下文 85元 洗白
        outcome = self.evaluator.evaluate(
            self.sample, "费用85元", "报销比例为85%。"
        )
        self.assertFalse(outcome.passed)

    def test_plain_number_accepts_percent_context(self):
        # 普通数值允许命中上下文中的百分数形式（文档化规则）
        outcome = self.evaluator.evaluate(
            self.sample, "比例为70%", "报销比例达到70。"
        )
        self.assertTrue(outcome.passed)

    def test_thousands_separator_and_fullwidth_normalized(self):
        outcome = self.evaluator.evaluate(
            self.sample, "限额为1,500元", "限额为１５００元。"
        )
        self.assertTrue(outcome.passed)

    def test_empty_answer_passes_vacuously(self):
        outcome = self.evaluator.evaluate(self.sample, "上下文", "")
        self.assertTrue(outcome.passed)


class CompletenessEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = CompletenessEvaluator(threshold=0.80)

    def test_fullwidth_and_whitespace_normalized(self):
        sample = _sample(expected_entities=["70%", "1500 元"])
        outcome = self.evaluator.evaluate(
            sample, "ctx", "报销比例７０％，限额 1500 元。"
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)

    def test_strict_contiguous_match(self):
        # "异地就医备案" 不得被 "异地就医办理备案" 稀释命中
        sample = _sample(expected_entities=["异地就医备案"])
        outcome = self.evaluator.evaluate(
            sample, "ctx", "需异地就医办理备案后方可结算。"
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.score, 0.0)

    def test_partial_hit_ratio(self):
        sample = _sample(expected_entities=["A1", "B2", "C3", "D4"])
        outcome = self.evaluator.evaluate(sample, "ctx", "包含 A1 与 B2。")
        self.assertAlmostEqual(outcome.score, 0.5)
        self.assertFalse(outcome.passed)

    def test_no_entities_vacuous_pass(self):
        outcome = self.evaluator.evaluate(_sample(), "ctx", "任意答案")
        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.score, 1.0)


class ContextRecallEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = ContextRecallEvaluator(threshold=0.90)

    def test_identical_context_scores_one(self):
        sample = _sample(ground_truth_context="异地就医备案后可直接结算，报销比例70%。")
        outcome = self.evaluator.evaluate(sample, sample.ground_truth_context, "ans")
        self.assertEqual(outcome.score, 1.0)
        self.assertTrue(outcome.passed)

    def test_partial_context_fails_and_reports_segments(self):
        sample = _sample(
            ground_truth_context="参保人员未备案自行外出就医的报销比例降低20个百分点"
        )
        outcome = self.evaluator.evaluate(
            sample, "参保人员外出就医按比例报销", "ans"
        )
        self.assertFalse(outcome.passed)
        self.assertLess(outcome.score, 0.8)
        joined = "; ".join(outcome.details)
        self.assertIn("未备", joined)  # 缺失片段可读化
        self.assertIn("20", joined)  # 缺失数字被点名

    def test_empty_ground_truth_vacuous_pass(self):
        outcome = self.evaluator.evaluate(_sample(ground_truth_context="，。"), "ctx", "a")
        self.assertTrue(outcome.passed)


class TextUtilTests(unittest.TestCase):
    def test_display_width(self):
        self.assertEqual(tu.display_width("abc"), 3)
        self.assertEqual(tu.display_width("医保"), 4)
        self.assertEqual(tu.display_width("医保70%"), 7)

    def test_truncate_display_keeps_alignment(self):
        out = tu.truncate_display("职工医保普通门诊报销比例", 10)
        self.assertLessEqual(tu.display_width(out), 10)
        self.assertTrue(out.endswith("..."))

    def test_extract_numbers_dedup(self):
        self.assertEqual(
            tu.extract_numbers("70%的70%与1500元"),
            [("70", True), ("1500", False)],
        )

    def test_faithfulness_normalizer_converts_months(self):
        self.assertIn("9月", tu.normalize_faithfulness_text("September 10"))
        self.assertIn("10月", tu.normalize_faithfulness_text("Oct 1"))

    def test_faithfulness_normalizer_converts_scales(self):
        normalized = tu.normalize_faithfulness_text("59,688 million and 108.0 billion")
        self.assertIn("596.88亿", normalized)
        self.assertIn("1080亿", normalized)


class PipelineTests(unittest.TestCase):
    def test_demo_pipeline_metrics(self):
        samples = load_samples()
        summary = run_eval_pipeline(samples, build_demo_rag())
        self.assertEqual(summary.total, 5)

        by_id = {r.sample_id: r for r in summary.results}

        # 完美链路：全绿
        perfect = by_id["med-ins-001"]
        self.assertTrue(summary.is_case_passed(perfect))

        # 检索缺失：召回/完整性塌陷，忠实性通过（没有编造）
        retrieval_gap = by_id["med-ins-002"]
        self.assertLess(retrieval_gap.context_recall_score, 0.8)
        self.assertLess(retrieval_gap.completeness_score, 0.8)
        self.assertTrue(retrieval_gap.faithfulness_pass)

        # 生成端遗漏：召回 1.0、完整性 0.5、忠实性通过
        omission = by_id["med-ins-003"]
        self.assertAlmostEqual(omission.context_recall_score, 1.0)
        self.assertAlmostEqual(omission.completeness_score, 0.5)
        self.assertTrue(omission.faithfulness_pass)

        # 前置条件失守
        precondition = by_id["med-ins-004"]
        self.assertFalse(summary.is_case_passed(precondition))

        # 对抗样本：召回/完整性全绿，仅忠实性一票否决（剪刀差）
        adversarial = by_id["med-ins-005"]
        self.assertAlmostEqual(adversarial.context_recall_score, 1.0)
        self.assertAlmostEqual(adversarial.completeness_score, 1.0)
        self.assertFalse(adversarial.faithfulness_pass)
        self.assertEqual(len(adversarial.error_details), 2)

        # 聚合指标
        self.assertEqual(summary.passed, 1)
        self.assertEqual(summary.failed, 4)
        self.assertAlmostEqual(summary.faithfulness_pass_rate, 0.8)


if __name__ == "__main__":
    unittest.main()
