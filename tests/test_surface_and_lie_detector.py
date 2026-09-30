"""受控表面实现 + Logprobs 测谎仪（Day 5-6）测试。"""
import math
import unittest

from rag_eval.surface import (
    EntityAwareRealizer,
    build_constrained_realizer_prompt,
)
from rag_eval.verifiers import (
    LieDetectorReport,
    LogprobsLieDetector,
    TokenLogprobItem,
)
from rag_eval.dataset import SAMPLE_001


class TestEntityAwareRealizer(unittest.TestCase):
    def test_empty_facts_returns_fallback(self):
        r = EntityAwareRealizer()
        out = r.realize([], "问题", "上下文")
        self.assertIn("未查询到", out)

    def test_reconstructs_subject_and_metric(self):
        ctx = SAMPLE_001.ground_truth_context
        r = EntityAwareRealizer()
        out = r.realize(
            [{"value": "70%", "key": "numeric_fact"}],
            SAMPLE_001.question,
            ctx,
        )
        # 回溯出实体与指标词
        self.assertIn("报销比例", out)
        self.assertIn("70%", out)
        self.assertNotIn("numeric_fact", out)

    def test_never_outputs_unlisted_numbers(self):
        ctx = "报销比例为70%。"
        r = EntityAwareRealizer()
        out = r.realize([{"value": "70%"}], "q", ctx)
        # 不应出现未在 facts 里的数字
        self.assertNotIn("95", out)
        self.assertIn("70%", out)

    def test_prefix_and_clause_separator(self):
        r = EntityAwareRealizer(prefix="根据规定：")
        out = r.realize([{"value": "200元"}], "q", "起付线为200元")
        self.assertTrue(out.startswith("根据规定："))
        self.assertTrue(out.endswith("。"))

    def test_fallback_when_no_clause(self):
        r = EntityAwareRealizer()
        # 事实值在上下文中无任何匹配，无法回溯
        out = r.realize([{"value": "999元"}], "q", "完全不相关的内容")
        # 仍会兜底输出（规定标准为999元），但保证有返回值
        self.assertIsInstance(out, str)
        self.assertTrue(out)


class TestConstrainedRealizerPrompt(unittest.TestCase):
    def test_prompt_contains_red_lines(self):
        facts = [{"value": "70%"}]
        prompt = build_constrained_realizer_prompt(facts, "问题")
        self.assertIn("严禁添加", prompt)
        self.assertIn("已核实事实列表", prompt)
        # 规则本身禁止输出内部标签，故提示词必须包含该约束
        self.assertIn("内部 JSON 标签", prompt)


class TestLogprobsLieDetector(unittest.TestCase):
    def test_cannot_act_as_sole_veto_by_default(self):
        # 默认不能作为单独一票否决：低置信度只标记风险预警，不单独判负
        stream = [
            TokenLogprobItem("报销", -0.05),
            TokenLogprobItem("比例", -0.02),
            TokenLogprobItem("70", -0.03),
            TokenLogprobItem("%", -0.01),
            TokenLogprobItem("40", -1.90),  # 瞎猜，置信度 15%
            TokenLogprobItem("00", -0.10),
            TokenLogprobItem("元", -0.05),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50)
        report = detector.analyze(stream)
        # 不能单独一票否决
        self.assertTrue(report.passed)
        self.assertTrue(report.has_risk)
        self.assertTrue(any("40" in e["entity"] or "4000" in e["entity"]
                            for e in report.flagged_entities))
        self.assertTrue(any("置信度极低" in w for w in report.warnings))

    def test_joint_veto_when_context_unsupported(self):
        # 协同裁决：低置信度 + 上下文无出处 -> 双重确凿，触发协同否决
        stream = [
            TokenLogprobItem("报销", -0.05),
            TokenLogprobItem("比例", -0.02),
            TokenLogprobItem("70", -0.03),
            TokenLogprobItem("%", -0.01),
            TokenLogprobItem("40", -1.90),  # 瞎猜，置信度 15%
            TokenLogprobItem("00", -0.10),
            TokenLogprobItem("元", -0.05),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50)
        # 上下文中只有 3000 元，无 4000 元
        context = "门诊报销比例70%，最高限额3000元。"
        report = detector.analyze(stream, context=context)
        self.assertFalse(report.passed)
        self.assertTrue(report.has_risk)
        self.assertTrue(any("协同否决" in r for r in report.rejection_reasons))

    def test_avoids_false_rejection_when_context_supports(self):
        # 防误杀：即使模型生成该数字时置信度低，但若上下文有确凿证据，则放行
        stream = [
            TokenLogprobItem("报销", -0.05),
            TokenLogprobItem("比例", -0.02),
            TokenLogprobItem("70", -0.03),
            TokenLogprobItem("%", -0.01),
            TokenLogprobItem("40", -1.90),  # 置信度偏低（可能因分词边界或概率分散）
            TokenLogprobItem("00", -0.10),
            TokenLogprobItem("元", -0.05),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50)
        # 上下文中明确包含 4000 元
        context = "门诊报销比例70%，特殊补贴4000元。"
        report = detector.analyze(stream, context=context)
        self.assertTrue(report.passed)
        self.assertTrue(report.has_risk)
        self.assertEqual(len(report.rejection_reasons), 0)

    def test_explicit_sole_veto_mode(self):
        # 显式开启 allow_sole_veto=True 时，仍支持单独一票否决（向下兼容/特定强管控场景）
        stream = [
            TokenLogprobItem("报销", -0.05),
            TokenLogprobItem("40", -1.90),
            TokenLogprobItem("00", -0.10),
            TokenLogprobItem("元", -0.05),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50, allow_sole_veto=True)
        report = detector.analyze(stream)
        self.assertFalse(report.passed)
        self.assertTrue(any("置信度极低" in r for r in report.rejection_reasons))

    def test_passes_when_all_high_confidence(self):
        stream = [
            TokenLogprobItem("报销", -0.05),
            TokenLogprobItem("70", -0.03),
            TokenLogprobItem("%", -0.01),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50)
        report = detector.analyze(stream)
        self.assertTrue(report.passed)
        self.assertFalse(report.has_risk)
        self.assertGreater(report.avg_confidence, 0.90)

    def test_ppl_exceeds_threshold(self):
        # 极低概率 token 拉高 PPL
        stream = [
            TokenLogprobItem("a", -5.0),
            TokenLogprobItem("b", -5.0),
            TokenLogprobItem("c", -5.0),
        ]
        # 默认模式：PPL 偏高仅作为软风险提示
        detector = LogprobsLieDetector(max_ppl_threshold=8.0)
        report = detector.analyze(stream)
        self.assertTrue(report.passed)
        self.assertTrue(report.has_risk)
        self.assertGreater(report.overall_ppl, 8.0)
        self.assertTrue(any("困惑度" in w for w in report.warnings))

        # 强管控模式：允许单独否决
        strict_detector = LogprobsLieDetector(max_ppl_threshold=8.0, allow_sole_veto=True)
        strict_report = strict_detector.analyze(stream)
        self.assertFalse(strict_report.passed)
        self.assertTrue(any("困惑度" in r for r in strict_report.rejection_reasons))

    def test_empty_stream_fails(self):
        detector = LogprobsLieDetector()
        report = detector.analyze([])
        self.assertFalse(report.passed)
        self.assertIn("无 Token", report.rejection_reasons[0])

    def test_confidence_conversion(self):
        item = TokenLogprobItem("x", math.log(0.7))
        self.assertAlmostEqual(item.prob, 0.7, places=3)

    def test_critical_entities_scan(self):
        stream = [
            TokenLogprobItem("统筹", -0.05),
            TokenLogprobItem("基金", -0.02),
            TokenLogprobItem("为", -0.01),
            TokenLogprobItem("1500", -1.80),  # 低置信
            TokenLogprobItem("元", -0.05),
        ]
        detector = LogprobsLieDetector(token_confidence_threshold=0.50)
        report = detector.analyze(stream, critical_entities=["统筹基金"])
        self.assertTrue(report.passed)
        self.assertTrue(report.has_risk)
        self.assertTrue(report.flagged_entities)

        # 结合上下文协同：若上下文无 1500，则触发协同否决
        bad_ctx_report = detector.analyze(
            stream,
            critical_entities=["统筹基金"],
            context="统筹基金起付线为200元。",
        )
        self.assertFalse(bad_ctx_report.passed)


if __name__ == "__main__":
    unittest.main()