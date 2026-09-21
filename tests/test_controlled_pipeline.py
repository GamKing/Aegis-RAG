"""受限抽取、Tier1、Tier2 自愈与受控合成测试。"""
from __future__ import annotations

import unittest

from rag_eval import (
    ControlledSynthesizer,
    EvalSample,
    ExtractionPayload,
    ExtractedFact,
    NLIAndSelfCorrector,
    RuleBasedFactExtractor,
    Tier1CodeVerifier,
    run_controlled_pipeline,
)


class FakeJudge:
    def __init__(self, verdict: bool):
        self.verdict = verdict
        self.calls = 0

    def check_faithfulness(self, context: str, claim: str) -> bool:
        self.calls += 1
        return self.verdict


def sample(context: str, entities: list[str] | None = None) -> EvalSample:
    return EvalSample(
        id="pipeline-test",
        question="金额是多少？",
        ground_truth_context=context,
        ground_truth_answer="answer",
        expected_entities=entities or [],
    )


class ControlledPipelineTests(unittest.TestCase):
    def test_rule_extractor_and_tier1_pass(self):
        context = "Revenue was $108.0 billion, up 18%."
        item = sample(context)
        payload = RuleBasedFactExtractor().extract(item.question, context)
        result = Tier1CodeVerifier().verify(item, context, payload)
        self.assertTrue(result.passed)
        self.assertEqual(len(payload.facts), 2)

    def test_tier1_blocks_unsupported_number(self):
        context = "Revenue was $108.0 billion."
        item = sample(context)
        payload = ExtractionPayload(
            facts=[ExtractedFact(key="revenue", value="$95.0 billion", is_numeric=True)],
            claims=["revenue: $95.0 billion"],
        )
        result = Tier1CodeVerifier().verify(item, context, payload)
        self.assertFalse(result.passed)
        self.assertEqual(result.findings[0].code, "UNSUPPORTED_NUMBER")

    def test_tier1_blocks_unsupported_fact(self):
        context = "Revenue was $108.0 billion."
        item = sample(context)
        payload = ExtractionPayload(
            facts=[ExtractedFact(key="business", value="Edge Computing")],
            claims=["Edge Computing"],
        )
        result = Tier1CodeVerifier().verify(item, context, payload)
        self.assertFalse(result.passed)
        self.assertEqual(result.findings[0].code, "UNSUPPORTED_FACT")

    def test_tier1_direct_path_skips_nli(self):
        context = "Revenue was $108.0 billion."
        item = sample(context)
        judge = FakeJudge(True)
        result = run_controlled_pipeline(
            item,
            context,
            RuleBasedFactExtractor(),
            ControlledSynthesizer(),
            Tier1CodeVerifier(),
            NLIAndSelfCorrector(judge),
        )
        self.assertTrue(result.verification.passed)
        self.assertFalse(result.trace.tier2_triggered)
        self.assertEqual(judge.calls, 0)

    def test_nli_self_correction_keeps_entailed_fact_then_rechecks(self):
        context = "NVIDIA is not assuming any Data Center compute revenue from China."
        item = sample(context)
        payload = ExtractionPayload(
            facts=[ExtractedFact(key="china_revenue", value="0", is_numeric=True)],
            claims=["china_revenue: 0"],
        )
        judge = FakeJudge(True)
        result = run_controlled_pipeline(
            item,
            context,
            lambda_query_extractor(payload),
            ControlledSynthesizer(),
            Tier1CodeVerifier(),
            NLIAndSelfCorrector(judge),
        )
        # NLI 证明“未计入收入”可合理直接推导为 0，Tier2 标记后由 Tier1 复验通过。
        self.assertTrue(result.trace.tier2_triggered)
        self.assertTrue(result.verification.passed)
        self.assertIn("0", result.answer)
        self.assertEqual(judge.calls, 1)

    def test_nli_rejects_hallucinated_number(self):
        context = "Edge Computing revenue was $7.2 billion."
        item = sample(context)
        payload = ExtractionPayload(
            facts=[ExtractedFact(key="revenue", value="$9.5 billion", is_numeric=True)],
            claims=["revenue: $9.5 billion"],
        )
        judge = FakeJudge(False)
        result = run_controlled_pipeline(
            item,
            context,
            lambda_query_extractor(payload),
            ControlledSynthesizer(),
            Tier1CodeVerifier(),
            NLIAndSelfCorrector(judge),
        )
        self.assertFalse(result.verification.passed)
        self.assertEqual(result.answer, "")
        self.assertEqual(judge.calls, 1)

    def test_synthesizer_only_outputs_verified_facts(self):
        payload = ExtractionPayload(
            facts=[ExtractedFact(key="revenue", value="1080亿", is_numeric=True)]
        )
        answer = ControlledSynthesizer().synthesize("q", payload)
        self.assertEqual(answer, "revenue: 1080亿")
        self.assertNotIn("95", answer)


def lambda_query_extractor(payload: ExtractionPayload):
    class FixedExtractor:
        def extract(self, query: str, context: str) -> ExtractionPayload:
            return payload

    return FixedExtractor()


if __name__ == "__main__":
    unittest.main()
