"""评测基线的反例测试：避免数据污染和指标假绿。"""
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rag_eval.data_generator import generate_dataset
from rag_eval.dataset import SAMPLE_001
from rag_eval.scissors_check import (
    audit_dataset, build_index, classify_failure, evaluate_baseline, main,
)


class TestQualityDataset(unittest.TestCase):
    def test_exact_size_full_determinism_and_clean_gold(self):
        samples = generate_dataset(seed=42)
        self.assertEqual(len(samples), 100)
        self.assertEqual([s.model_dump() for s in samples],
                         [s.model_dump() for s in generate_dataset(seed=42)])
        self.assertEqual(audit_dataset(samples), [])

    def test_zero_boundary_is_real_and_consistent(self):
        zeros = [s for s in generate_dataset() if s.id.startswith("edge-")
                 and int(s.id.split("-")[1]) % 3 == 0]
        self.assertEqual(len(zeros), 9)
        for sample in zeros:
            self.assertIn("起付线0元", sample.ground_truth_context)
            self.assertIn("起付线0元", sample.ground_truth_answer)
            self.assertIn("0元", sample.expected_entities)

    def test_adversarial_answers_do_not_replace_gold(self):
        adversarial = [s for s in generate_dataset() if s.category == "adversarial"]
        self.assertEqual(len(adversarial), 15)
        for sample in adversarial:
            self.assertNotEqual(sample.candidate_answer, sample.ground_truth_answer)
            self.assertEqual(audit_dataset([sample]), [])
            if sample.mutation_type == "条件丢失":
                self.assertIn("20个百分点", sample.ground_truth_answer)
                self.assertNotIn("20个百分点", sample.candidate_answer)

    def test_conflicting_versions_and_invalid_gold_are_rejected(self):
        original = generate_dataset()[0]
        conflict = original.model_copy(update={"id": "conflict",
            "ground_truth_context": "没有任何政策数值。"})
        codes = {i["code"] for i in audit_dataset([original, conflict])}
        self.assertTrue({"ambiguous_question", "conflicting_source", "invalid_gold"} <= codes)

    def test_index_deduplicates_shared_sources(self):
        original = generate_dataset()[0]
        duplicate = original.model_copy(update={"id": "variant"})
        _, count = build_index([original, duplicate])
        self.assertEqual(count, 1)

    def test_invalid_counts(self):
        for kwargs in ({"synth_count": -1}, {"synth_count": 0, "edge_count": 1}):
            with self.assertRaises(ValueError):
                generate_dataset(**kwargs)


class TestBaselineDiagnosis(unittest.TestCase):
    def test_attribution_uses_provenance_and_oracle(self):
        self.assertEqual(classify_failure(True, True, True, True, True), "data")
        self.assertEqual(classify_failure(False, False, True, True, True), "retrieval_source_miss")
        self.assertEqual(classify_failure(False, True, False, False, True), "generation_with_gold_context")
        self.assertEqual(classify_failure(False, True, True, True, False), "context_contamination")
        self.assertEqual(classify_failure(False, True, True, False, True), "generation_with_retrieved_context")

    def test_foreign_numbers_cannot_be_washed_by_retrieved_context(self):
        first = SAMPLE_001.model_copy(update={"id": "first", "source_id": "policy_a",
            "question": "policy_a 职工医保报销比例是多少？",
            "ground_truth_context": "policy_a 职工医保报销比例为70%。",
            "ground_truth_answer": "职工医保报销比例70%。", "expected_entities": ["70%"]})
        other = first.model_copy(update={"id": "other", "source_id": "policy_b",
            "question": "policy_b 职工医保报销比例是多少？",
            "ground_truth_context": "policy_b 职工医保报销比例为85%。",
            "ground_truth_answer": "职工医保报销比例85%。", "expected_entities": ["85%"]})
        report = evaluate_baseline([first, other], top_k=2)
        self.assertEqual(report["metrics"]["case_pass_rate"], 1.0)
        self.assertEqual(report["classifications"], {"context_contamination": 2})
        self.assertFalse(report["gate_passed"])
        self.assertTrue(evaluate_baseline([first, other], top_k=1)["gate_passed"])

    def test_data_issues_fail_before_running_pipeline(self):
        report = evaluate_baseline([SAMPLE_001])
        self.assertFalse(report["gate_passed"])
        self.assertNotIn("metrics", report)

    def test_cli_quality_failure_returns_nonzero_and_writes_evidence(self):
        with TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "invalid.json"
            output = Path(tmp) / "report.json"
            dataset.write_text(json.dumps([SAMPLE_001.model_dump()]), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--dataset", str(dataset), "--n", "1",
                                       "--output", str(output)]), 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))["gate_passed"])

    def test_invalid_sample_limit_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--n", "0"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
