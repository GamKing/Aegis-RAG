"""NVIDIA 结构化抽取契约 Case 01-07 测试。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

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

FIXTURE = Path(__file__).with_name("nvda_structured_cases.json")


class EntailedJudge:
    def __init__(self, verdict: bool = True):
        self.verdict = verdict
        self.calls = 0

    def check_faithfulness(self, context: str, claim: str) -> bool:
        self.calls += 1
        return self.verdict


def load_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def to_sample(case: dict) -> EvalSample:
    return EvalSample(
        id=case["id"],
        question=case["question"],
        ground_truth_context=case["ground_truth_context"],
        ground_truth_answer=case["ground_truth_answer"],
        expected_entities=case["expected_entities"],
    )


class NvidiaStructuredCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_all_structured_contracts_have_required_fields(self):
        self.assertEqual(len(self.cases), 7)
        for case in self.cases:
            self.assertTrue(case["id"])
            self.assertTrue(case["category"])
            self.assertTrue(case["expected_chunk_ids"])
            extracted = case["expected_extracted_json"]
            self.assertIn("direct_answer", extracted)
            self.assertIn("numerical_facts", extracted)
            self.assertIn("conditions_and_exceptions", extracted)
            self.assertIn("source_quote", extracted)
            for fact in extracted["numerical_facts"]:
                self.assertEqual(
                    set(fact), {"entity_name", "value", "unit"}
                )
                self.assertTrue(fact["entity_name"])
                self.assertTrue(fact["value"])
                self.assertTrue(fact["unit"])

    def test_expected_numeric_facts_have_context_support(self):
        verifier = Tier1CodeVerifier()
        for case in self.cases:
            sample = to_sample(case)
            facts = [
                ExtractedFact(
                    key=item["entity_name"],
                    value=item["value"] + ("%" if item["unit"] == "%" else (" " + item["unit"] if item["unit"] in {"million USD", "billion USD"} else "")),
                    source_text=case["expected_extracted_json"]["source_quote"] or None,
                    is_numeric=True,
                )
                for item in case["expected_extracted_json"]["numerical_facts"]
            ]
            payload = ExtractionPayload(facts=facts, claims=[f.value for f in facts])
            result = verifier.verify(sample, case["ground_truth_context"], payload)
            if case["id"] == "nvda_case_03_nli_entailment":
                # 0 是语义推导，不是字面出处，必须留给 Tier2。
                self.assertFalse(result.passed)
                self.assertTrue(
                    any(f.code == "UNSUPPORTED_NUMBER" for f in result.findings)
                )
            else:
                self.assertFalse(
                    any(f.code == "UNSUPPORTED_NUMBER" for f in result.findings),
                    msg=case["id"],
                )

    def test_case_03_nli_entailment_can_retain_zero(self):
        case = self.by_id["nvda_case_03_nli_entailment"]
        sample = to_sample(case)
        payload = ExtractionPayload(
            facts=[
                ExtractedFact(
                    key="China Data Center Compute Revenue Outlook",
                    value="0",
                    is_numeric=True,
                )
            ],
            claims=["China Data Center Compute Revenue Outlook: 0"],
        )
        judge = EntailedJudge(True)

        class FixedExtractor:
            def extract(self, query: str, context: str) -> ExtractionPayload:
                return payload

        result = run_controlled_pipeline(
            sample,
            case["ground_truth_context"],
            FixedExtractor(),
            ControlledSynthesizer(),
            Tier1CodeVerifier(),
            NLIAndSelfCorrector(judge),
        )
        self.assertTrue(result.trace.tier2_triggered)
        self.assertTrue(result.verification.passed)
        self.assertIn("0", result.answer)
        self.assertEqual(judge.calls, 1)

    def test_case_06_adversarial_expected_extraction_is_clean(self):
        case = self.by_id["nvda_case_06_adversarial_extraction"]
        sample = to_sample(case)
        extracted = case["expected_extracted_json"]
        payload = ExtractionPayload(
            facts=[
                ExtractedFact(
                    key=item["entity_name"],
                    value=item["value"] + ("%" if item["unit"] == "%" else (" " + item["unit"] if item["unit"] in {"million USD", "billion USD"} else "")),
                    is_numeric=True,
                )
                for item in extracted["numerical_facts"]
            ]
        )
        result = Tier1CodeVerifier().verify(
            sample, case["ground_truth_context"], payload
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(result.findings), 0)

    def test_case_07_has_no_numeric_facts_and_does_not_invent_revenue(self):
        case = self.by_id["nvda_case_07_unmentioned_boundary"]
        self.assertEqual(case["expected_extracted_json"]["numerical_facts"], [])
        self.assertEqual(case["expected_extracted_json"]["source_quote"], "")
        self.assertNotIn("DRIVE Thor", case["ground_truth_context"])


if __name__ == "__main__":
    unittest.main()
