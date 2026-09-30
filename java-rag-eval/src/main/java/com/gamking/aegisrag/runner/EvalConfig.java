package com.gamking.aegisrag.runner;

/** 评测通过门槛配置。 */
public record EvalConfig(
        double contextRecallThreshold,
        double completenessThreshold
) {
    public static final double DEFAULT_RECALL = 0.80;
    public static final double DEFAULT_COMPLETENESS = 0.80;

    public EvalConfig {
        if (contextRecallThreshold < 0.0 || contextRecallThreshold > 1.0) {
            throw new IllegalArgumentException("contextRecallThreshold must be in [0.0, 1.0]");
        }
        if (completenessThreshold < 0.0 || completenessThreshold > 1.0) {
            throw new IllegalArgumentException("completenessThreshold must be in [0.0, 1.0]");
        }
    }

    public static EvalConfig defaults() {
        return new EvalConfig(DEFAULT_RECALL, DEFAULT_COMPLETENESS);
    }
}