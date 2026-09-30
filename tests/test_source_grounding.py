"""Source scope, mandatory coverage, and attribution regression tests."""
import unittest

from rag_eval.data_generator import generate_dataset
from rag_eval.dummy_rag import DummyRAG, RAGResponse
from rag_eval.evaluators import CompletenessEvaluator, RequiredEntitiesEvaluator
from rag_eval.models import AtomicFact, EvalSample, FactType
from rag_eval.pipeline import PipelineRAG, RuleBasedFactExtractor, ControlledSynthesizer
from rag_eval.rag_impl import ProductionRAG
from rag_eval.retrieval.base import DocumentChunk, SearchResult
from rag_eval.retrieval.source_scope import SourceScopeError, select_context
from rag_eval.runner import run_eval_pipeline
from rag_eval.report import render_report
from rag_eval.scissors_check import evaluate_baseline
from rag_eval.verification_rules import NumericRule


def hit(doc, text, scope=None):
    meta = {"doc_id": doc}
    if scope:
        meta["scope_id"] = scope
    return SearchResult(DocumentChunk(doc, text, meta), 1.0)


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k=3):
        return self.hits[:top_k]


class TestSourceScope(unittest.TestCase):
    def setUp(self):
        self.hits = [hit("a-main", "报销比例70%。", "policy-a"),
                     hit("b-main", "报销比例85%。", "policy-b"),
                     hit("a-exceptions", "急诊起付线0元。", "policy-a")]

    def test_multiple_blocks_in_one_scope_survive_top_three(self):
        selection = select_context("【来源 policy-a】报销比例和急诊起付线？", self.hits)
        self.assertIn("70%", selection.context)
        self.assertIn("起付线0元", selection.context)
        self.assertNotIn("85%", selection.context)
        self.assertEqual(selection.discarded_sources, ("policy-b",))

    def test_unknown_scope_does_not_fall_back_to_top_one(self):
        with self.assertRaises(SourceScopeError) as error:
            select_context("[source: missing] 报销？", self.hits)
        self.assertEqual(error.exception.code, "REQUESTED_SOURCE_NOT_FOUND")

    def test_ambiguous_policy_versions_require_clarification(self):
        with self.assertRaises(SourceScopeError) as error:
            select_context("报销比例是多少？", self.hits)
        self.assertEqual(error.exception.code, "AMBIGUOUS_SOURCE")

    def test_explicit_multiple_sources_keep_both(self):
        selected = select_context("比较 [source: policy-a] 与 [source: policy-b]", self.hits)
        self.assertEqual(set(selected.selected_sources), {"policy-a", "policy-b"})
        self.assertIn("85%", selected.context)
        self.assertIn("70%", selected.context)
        self.assertIn("【来源 policy-a】", selected.context)
        self.assertIn("【来源 policy-b】", selected.context)

    def test_single_requested_scope_can_span_multiple_documents(self):
        selected = select_context("报销？", self.hits, sources=["policy-a"])
        self.assertIn("急诊", selected.context)
        self.assertNotIn("85%", selected.context)

    def test_missing_one_of_multiple_requested_sources_fails(self):
        with self.assertRaises(SourceScopeError):
            select_context("比较", self.hits, sources=["policy-a", "missing"])

    def test_plain_documents_still_support_multi_document_retrieval(self):
        selected = select_context("医保", [hit("outpatient", "门诊70%。"), hit("inpatient", "住院80%。")])
        self.assertIn("门诊", selected.context)
        self.assertIn("住院", selected.context)

    def test_metadata_content_conflict_rejected(self):
        with self.assertRaises(SourceScopeError):
            select_context("[source: a]", [hit("doc", "【来源 b】70%", "a")])

    def test_gold_source_label_cannot_steer_generation(self):
        rag = ProductionRAG(FixedRetriever(self.hits), top_k=3)
        sample = EvalSample(id="test", question="[source: policy-a] 报销比例？",
                            ground_truth_context="不应读取", ground_truth_answer="不应读取",
                            source_id="policy-b")
        answer = rag.retrieve_and_generate(sample).answer
        self.assertIn("70%", answer)
        self.assertNotIn("85%", answer)

    def test_shared_context_provider_is_not_overwritten(self):
        provider = lambda sample: "original"
        pipeline = PipelineRAG(context_provider=provider)
        rag = ProductionRAG(FixedRetriever(self.hits), generator=pipeline)
        sample = EvalSample(id="test", question="[source: policy-a] 比例？",
                            ground_truth_context="", ground_truth_answer="")
        rag.retrieve_and_generate(sample)
        self.assertIs(pipeline.context_provider, provider)


class TestMandatoryCoverage(unittest.TestCase):
    def setUp(self):
        self.sample = EvalSample(id="required", question="待遇？",
            ground_truth_context="病种 门诊 基金 限额 比例", ground_truth_answer="病种 门诊 基金 限额 比例",
            expected_entities=["病种", "门诊", "基金", "限额", "比例"], required_entities=["病种"])
        self.answer = "门诊 基金 限额 比例"

    def test_eighty_percent_does_not_compensate_missing_required_field(self):
        comp = CompletenessEvaluator(0.8).evaluate(self.sample, "", self.answer)
        self.assertEqual(comp.score, 0.8)
        self.assertTrue(comp.passed)
        self.assertFalse(RequiredEntitiesEvaluator().evaluate(self.sample, "", self.answer).passed)

    def test_optional_missing_field_keeps_coverage_threshold(self):
        sample = self.sample.model_copy(update={"required_entities": ["基金"]})
        self.assertTrue(RequiredEntitiesEvaluator().evaluate(sample, "", self.answer).passed)
        self.assertTrue(CompletenessEvaluator(0.8).evaluate(sample, "", self.answer).passed)

    def test_runner_summary_and_custom_evaluators_respect_required_gate(self):
        rag = DummyRAG({self.sample.id: RAGResponse(self.sample.ground_truth_context, self.answer)})
        for evaluators in (None, [CompletenessEvaluator(0.8)]):
            summary = run_eval_pipeline([self.sample], rag, evaluators=evaluators)
            self.assertFalse(summary.results[0].required_entities_pass)
            self.assertFalse(summary.is_case_passed(summary.results[0]))
            self.assertEqual(summary.failed, 1)
            self.assertIn("REQUIRED", render_report(summary))
            self.assertTrue(any("required_entities" in error for error in summary.results[0].error_details))


class TestNumericAttribution(unittest.TestCase):
    def setUp(self):
        self.rule = NumericRule()
        self.context = "一级医院起付线200元，报销比例85%；三级医院起付线1200元，报销比例55%。"

    def fact(self, **kwargs):
        return AtomicFact(subject="一级医院", predicate="起付线", object_value="200元",
                          fact_type=FactType.NUMERIC).model_copy(update=kwargs)

    def test_correct_binding_passes(self):
        self.assertTrue(self.rule.verify(self.fact(), self.context).passed)

    def test_number_elsewhere_cannot_support_wrong_subject(self):
        result = self.rule.verify(self.fact(object_value="1200元"), self.context)
        self.assertFalse(result.passed)
        self.assertIn("NUMERIC_BINDING_MISMATCH", [finding.code for finding in result.findings])

    def test_same_clause_wrong_subject_is_rejected(self):
        context = "一级医院起付线200元，三级医院起付线1200元。"
        self.assertFalse(self.rule.verify(self.fact(object_value="1200元"), context).passed)

    def test_wrong_attribute_is_rejected(self):
        context = "一级医院起付线200元，年度限额1200元。"
        self.assertFalse(self.rule.verify(self.fact(object_value="1200元"), context).passed)

    def test_fabricated_quote_rejected_even_when_number_exists(self):
        result = self.rule.verify(self.fact(source_text="一级医院起付线1200元。", object_value="1200元"), self.context)
        self.assertFalse(result.passed)
        self.assertEqual(result.findings[0].code, "INVALID_SOURCE_TEXT")

    def test_wrong_real_quote_rejected(self):
        self.assertFalse(self.rule.verify(self.fact(source_text="三级医院起付线1200元", object_value="1200元"), self.context).passed)

    def test_valid_local_quote_passes(self):
        self.assertTrue(self.rule.verify(self.fact(source_text="一级医院起付线200元"), self.context).passed)


class TestFixedBaseline(unittest.TestCase):
    def test_verbatim_synthesis_preserves_binding_without_orphan_numbers(self):
        context = "【来源 policy-a】一级医院起付线200元。"
        payload = RuleBasedFactExtractor().extract("起付线？", context)
        answer = ControlledSynthesizer().synthesize("起付线？", payload)
        self.assertEqual(answer.count("200"), 1)
        self.assertIn("【来源 policy-a】一级医院起付线200元", answer)

    def test_all_scoped_cases_and_adversarial_candidates_pass(self):
        report = evaluate_baseline(generate_dataset(), top_k=3)
        self.assertTrue(report["gate_passed"])
        self.assertEqual(report["classifications"], {"pass": 100})
        self.assertEqual(report["adversarial_detected"], 15)
        self.assertEqual(sum(a["legacy_detected"] for a in report["adversarial"]), 13)


if __name__ == "__main__":
    unittest.main()
