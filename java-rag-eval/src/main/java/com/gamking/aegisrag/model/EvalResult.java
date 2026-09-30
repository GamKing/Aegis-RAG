package com.gamking.aegisrag.model;

import java.util.List;

/** 单条样本的评测产出。 */
public record EvalResult(
        String sampleId,
        String actualContext,
        String actualAnswer,
        double contextRecallScore,
        double completenessScore,
        boolean faithfulnessPass,
        List<String> errorDetails
) {
    public EvalResult {
        if (sampleId == null || sampleId.isBlank()) {
            throw new IllegalArgumentException("sampleId must not be blank");
        }
        if (actualContext == null || actualAnswer == null) {
            throw new IllegalArgumentException("actualContext/actualAnswer must not be null");
        }
        if (contextRecallScore < 0.0 || contextRecallScore > 1.0) {
            throw new IllegalArgumentException("contextRecallScore must be in [0.0, 1.0]");
        }
        if (completenessScore < 0.0 || completenessScore > 1.0) {
            throw new IllegalArgumentException("completenessScore must be in [0.0, 1.0]");
        }
        errorDetails = errorDetails == null ? List.of() : List.copyOf(errorDetails);
    }
}
