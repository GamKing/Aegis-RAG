package com.gamking.aegisrag.model;

/** 单个评分器的判定产出。score 为 null 表示纯布尔型指标（忠实性）。 */
public record MetricOutcome(
        String evaluatorName,
        Double score,
        boolean passed,
        java.util.List<String> details
) {
    public MetricOutcome {
        if (evaluatorName == null || evaluatorName.isBlank()) {
            throw new IllegalArgumentException("evaluatorName must not be blank");
        }
        details = details == null ? java.util.List.of() : java.util.List.copyOf(details);
    }

    public static MetricOutcome pass(String evaluatorName, Double score) {
        return new MetricOutcome(evaluatorName, score, true, java.util.List.of());
    }

    public static MetricOutcome fail(String evaluatorName, Double score, java.util.List<String> details) {
        return new MetricOutcome(evaluatorName, score, false, details);
    }
}
