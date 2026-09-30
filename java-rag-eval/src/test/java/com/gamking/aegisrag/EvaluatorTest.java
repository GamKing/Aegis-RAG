package com.gamking.aegisrag;

import com.gamking.aegisrag.evaluator.CompletenessEvaluator;
import com.gamking.aegisrag.evaluator.ContextRecallEvaluator;
import com.gamking.aegisrag.evaluator.FaithfulnessEvaluator;
import com.gamking.aegisrag.model.EvalSample;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 三项评测器单元测试
 */
public class EvaluatorTest {

    private static EvalSample sample(String context, String answer, List<String> entities) {
        return new EvalSample("test", "q", context, answer, entities);
    }

    @Test
    public void test_contextRecall_identical_scores_one() {
        String context = "异地就医备案后可直接结算，报销比例70%。";
        EvalSample s = sample(context, "ans", List.of());
        var outcome = new ContextRecallEvaluator().evaluate(s, context, "ans");
        assertEquals(1.0, outcome.score());
        assertTrue(outcome.passed());
    }

    @Test
    public void test_contextRecall_partial_fails() {
        String ground = "参保人员未备案自行外出就医的报销比例降低20个百分点";
        EvalSample s = sample(ground, "ans", List.of());
        var outcome = new ContextRecallEvaluator().evaluate(s, "参保人员外出就医按比例报销", "ans");
        assertFalse(outcome.passed());
        assertTrue(outcome.score() < 0.8);
    }

    @Test
    public void test_contextRecall_empty_ground_truth_pass() {
        EvalSample s = sample("，。", "ctx", List.of());
        var outcome = new ContextRecallEvaluator().evaluate(s, "ctx", "a");
        assertTrue(outcome.passed());
    }

    @Test
    public void test_completeness_fullwidth_normalized() {
        EvalSample s = sample("ctx", "报销比例７０％，限额 1500 元。", List.of("70%", "1500 元"));
        var outcome = new CompletenessEvaluator().evaluate(s, "ctx", "报销比例７０％，限额 1500 元。");
        assertEquals(1.0, outcome.score());
    }

    @Test
    public void test_completeness_strict_contiguous() {
        EvalSample s = sample("ctx", "需异地就医办理备案后方可结算。", List.of("异地就医备案"));
        var outcome = new CompletenessEvaluator().evaluate(s, "ctx", "需异地就医办理备案后方可结算。");
        assertEquals(0.0, outcome.score());
    }

    @Test
    public void test_completeness_partial_ratio() {
        EvalSample s = sample("ctx", "包含 A1 与 B2。", List.of("A1", "B2", "C3", "D4"));
        var outcome = new CompletenessEvaluator().evaluate(s, "ctx", "包含 A1 与 B2。");
        assertEquals(0.5, outcome.score());
        assertFalse(outcome.passed());
    }

    @Test
    public void test_faithfulness_supported_numbers_pass() {
        EvalSample s = sample("报销比例为70%，限额1500元", "报销比例为70%，限额1500元。", List.of());
        var outcome = new FaithfulnessEvaluator().evaluate(s, "报销比例为70%，限额1500元", "报销比例为70%，限额1500元。");
        assertTrue(outcome.passed());
    }

    @Test
    public void test_faithfulness_tampered_numbers_fail() {
        EvalSample s = sample("限额1500元，报销比例70%", "限额2500元，报销比例85%。", List.of());
        var outcome = new FaithfulnessEvaluator().evaluate(s, "限额1500元，报销比例70%", "限额2500元，报销比例85%。");
        assertFalse(outcome.passed());
        assertEquals(2, outcome.details().size());
    }

    @Test
    public void test_faithfulness_prefix_substring_not_counted() {
        EvalSample s = sample("限额15000元", "限额1500元。", List.of());
        var outcome = new FaithfulnessEvaluator().evaluate(s, "限额15000元", "限额1500元。");
        assertFalse(outcome.passed());
    }

    @Test
    public void test_faithfulness_percent_unit_mismatch_rejected() {
        EvalSample s = sample("费用85元", "报销比例为85%。", List.of());
        var outcome = new FaithfulnessEvaluator().evaluate(s, "费用85元", "报销比例为85%。");
        assertFalse(outcome.passed());
    }
}