package com.gamking.aegisrag.evaluator;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.MetricOutcome;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 严格实体命中检查：命中数 / 期望总数。
 * 
 * 匹配口径：对实体与答案做归一化（全角->半角、去空白、忽略大小写）后，
 * 要求实体以连续子串形式出现在答案中。
 */
public class CompletenessEvaluator implements Evaluator {

    private final double threshold;

    public CompletenessEvaluator() {
        this(0.80);
    }

    public CompletenessEvaluator(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public MetricOutcome evaluate(EvalSample sample, String actualContext, String actualAnswer) {
        List<String> entities = sample.expectedEntities();
        if (entities.isEmpty()) {
            return MetricOutcome.pass("completeness", 1.0);
        }

        String answerNorm = TextNormalizer.normalizeForMatch(actualAnswer);
        List<String> missed = new ArrayList<>();
        for (String entity : entities) {
            String entityNorm = TextNormalizer.normalizeForMatch(entity);
            if (!answerNorm.contains(entityNorm)) {
                missed.add(entity);
            }
        }

        int hit = entities.size() - missed.size();
        double score = (double) hit / entities.size();
        score = Math.round(score * 10000.0) / 10000.0;
        boolean passed = score >= threshold;

        List<String> details = new ArrayList<>();
        if (!passed) {
            details.add(String.format("[completeness] 实体命中率 %d/%d = %.3f 低于阈值 %.2f; 未命中: %s",
                    hit, entities.size(), score, threshold, String.join(", ", missed)));
        }

        return passed ? MetricOutcome.pass("completeness", score)
                      : MetricOutcome.fail("completeness", score, details);
    }
}
