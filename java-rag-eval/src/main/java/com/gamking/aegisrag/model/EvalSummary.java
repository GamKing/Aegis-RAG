package com.gamking.aegisrag.model;

import java.time.LocalDateTime;
import java.util.List;

/** 整批样本的聚合结果与阈值快照。 */
public record EvalSummary(
        LocalDateTime generatedAt,
        int total,
        int passed,
        int failed,
        double avgContextRecall,
        double avgCompleteness,
        double faithfulnessPassRate,
        double casePassRate,
        double contextRecallThreshold,
        double completenessThreshold,
        List<EvalResult> results
) {
    public EvalSummary {
        if (total < 0) throw new IllegalArgumentException("total must be >= 0");
        if (passed < 0 || failed < 0) throw new IllegalArgumentException("passed/failed must be >= 0");
        results = results == null ? List.of() : List.copyOf(results);
    }

    /** 判定单条结果是否通过（recall/completeness 达标且忠实）。 */
    public boolean isCasePassed(EvalResult result) {
        return result.contextRecallScore() >= contextRecallThreshold
                && result.completenessScore() >= completenessThreshold
                && result.faithfulnessPass();
    }

    /** 返回失败的结果列表。 */
    public List<EvalResult> failedResults() {
        return results.stream()
                .filter(r -> !isCasePassed(r))
                .toList();
    }
}
