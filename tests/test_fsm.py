"""受限解码状态机（Day 3-4）测试：FSM 闭环、JSON Schema 约束、单次打回重试。"""
import unittest

from rag_eval.fsm import (
    DecodingState,
    GenerationFSM,
    JsonSchemaConstraint,
    run_fsm_pipeline,
)
from rag_eval.pipeline import (
    ControlledSynthesizer,
    NLIAndSelfCorrector,
    RuleBasedFactExtractor,
    Tier1CodeVerifier,
)
from rag_eval.dataset import SAMPLE_001
from rag_eval.models import ExtractedFact, ExtractionPayload


def _failing_payload():
    """构造一个必失败的 payload：数字无上下文出处。"""
    return ExtractionPayload(
        facts=[
            ExtractedFact(
                key="numeric_fact",
                value="999999元",
                source_text="999999",
                is_numeric=True,
            )
        ],
        claims=["999999元"],
        raw_json={"query": "q", "facts": [], "atomic_facts": []},
    )


class _AcceptingJudge:
    def check_faithfulness(self, context, claim):
        return True


class _RejectingJudge:
    def check_faithfulness(self, context, claim):
        return False


class _NoopCorrector:
    def correct(self, context, payload, findings):
        return payload


class _FailingExtractor:
    def extract(self, query, context):
        return _failing_payload()


class TestJsonSchemaConstraint(unittest.TestCase):
    def test_rejects_unknown_top_level_key(self):
        payload = ExtractionPayload(
            facts=[], claims=[],
            raw_json={"query": "q", "evil_key": 123, "atomic_facts": []},
        )
        result = JsonSchemaConstraint().coerce(payload)
        self.assertTrue(result.repaired)
        self.assertNotIn("evil_key", result.payload.raw_json)

    def test_normalizes_wrong_field_type(self):
        payload = ExtractionPayload(
            facts=[], claims=[],
            raw_json={
                "query": "q",
                "atomic_facts": [
                    {
                        "subject": "医保", "predicate": "报销", "object_value": "70%",
                        "fact_type": "numeric", "is_negative": "yes", "source_text": "x",
                    }
                ],
            },
        )
        result = JsonSchemaConstraint().coerce(payload)
        self.assertTrue(result.repaired)
        self.assertFalse(result.payload.raw_json["atomic_facts"][0]["is_negative"])

    def test_clean_payload_no_violation(self):
        payload = ExtractionPayload(
            facts=[], claims=[],
            raw_json={"query": "q", "atomic_facts": []},
        )
        result = JsonSchemaConstraint().coerce(payload)
        self.assertFalse(result.repaired)


class TestGenerationFSM(unittest.TestCase):
    def test_accept_path(self):
        fsm = GenerationFSM(
            extractor=RuleBasedFactExtractor(),
            verifier=Tier1CodeVerifier(),
            synthesizer=ControlledSynthesizer(),
        )
        r = fsm.run(SAMPLE_001, SAMPLE_001.ground_truth_context)
        self.assertTrue(r.accepted)
        self.assertEqual(r.trace[-1], DecodingState.ACCEPT)
        self.assertEqual(r.repair_count, 0)
        # 直接通过：extract->validate->decide->accept
        self.assertEqual(
            [s.value for s in r.trace],
            ["extract", "validate", "decide", "accept"],
        )

    def test_repair_retry_accept(self):
        """单次打回重试：Tier-1 失败 → NLI 兜底修复 → 重跑 Tier-1 通过。"""
        fsm = GenerationFSM(
            extractor=_FailingExtractor(),
            verifier=Tier1CodeVerifier(),
            synthesizer=ControlledSynthesizer(),
            self_corrector=NLIAndSelfCorrector(_AcceptingJudge()),
        )
        r = fsm.run(SAMPLE_001, SAMPLE_001.ground_truth_context)
        self.assertTrue(r.accepted)
        self.assertEqual(r.repair_count, 1)
        # 状态机闭环完整
        self.assertEqual(
            [s.value for s in r.trace],
            ["extract", "validate", "decide", "repair", "revalidate", "decide", "accept"],
        )

    def test_reject_path(self):
        """NLI 兜底无效，仍失败 → 安全拒绝。"""
        fsm = GenerationFSM(
            extractor=_FailingExtractor(),
            verifier=Tier1CodeVerifier(),
            synthesizer=ControlledSynthesizer(),
            self_corrector=_NoopCorrector(),
        )
        r = fsm.run(SAMPLE_001, SAMPLE_001.ground_truth_context)
        self.assertFalse(r.accepted)
        self.assertEqual(r.trace[-1], DecodingState.REJECT)

    def test_illegal_transition_is_guarded(self):
        fsm = GenerationFSM(
            extractor=RuleBasedFactExtractor(),
            verifier=Tier1CodeVerifier(),
            synthesizer=ControlledSynthesizer(),
        )
        # 不允许从 ACCEPT 转移到任何状态
        self.assertFalse(fsm._transition(DecodingState.ACCEPT, DecodingState.REPAIR))


class TestRunFsmPipeline(unittest.TestCase):
    def test_returns_pipeline_contract(self):
        result = run_fsm_pipeline(
            SAMPLE_001,
            SAMPLE_001.ground_truth_context,
            RuleBasedFactExtractor(),
            ControlledSynthesizer(),
            Tier1CodeVerifier(),
        )
        self.assertTrue(result.accepted)
        self.assertTrue(result.answer)
        self.assertEqual(result.repair_count, 0)
        self.assertEqual(
            result.trace_states,
            ["extract", "validate", "decide", "accept"],
        )


if __name__ == "__main__":
    unittest.main()