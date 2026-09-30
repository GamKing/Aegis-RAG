package com.gamking.aegisrag.evaluator;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.MetricOutcome;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 评估 actual_context 对 ground_truth_context 的关键内容覆盖率。
 * 
 * 方法：双端抽取词项（CJK bigram + ASCII 词/数字），计算
 * 
 *     recall = |GT 词项 ∩ ACTUAL 词项| / |GT 词项|
 */
public class ContextRecallEvaluator implements Evaluator {

    private final double threshold;

    public ContextRecallEvaluator() {
        this(0.80);
    }

    public ContextRecallEvaluator(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public MetricOutcome evaluate(EvalSample sample, String actualContext, String actualAnswer) {
        Set<String> gtTerms = TextNormalizer.extractTerms(sample.groundTruthContext());
        Set<String> actualTerms = TextNormalizer.extractTerms(actualContext);

        if (gtTerms.isEmpty()) {
            return MetricOutcome.pass("context_recall", 1.0);
        }

        Set<String> missing = new LinkedHashSet<>(gtTerms);
        missing.removeAll(actualTerms);

        double score = 1.0 - (double) missing.size() / gtTerms.size();
        score = Math.round(score * 10000.0) / 10000.0;
        boolean passed = score >= threshold;

        List<String> details = new ArrayList<>();
        if (!passed) {
            details.add(String.format("[context_recall] 得分 %.3f 低于阈值 %.2f; 缺失 %d 个词项",
                    score, threshold, missing.size()));
        }

        return passed ? MetricOutcome.pass("context_recall", score)
                      : MetricOutcome.fail("context_recall", score, details);
    }
}
