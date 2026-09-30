package com.gamking.aegisrag.model;

/** 受控流水线的最终产出。 */
public record PipelineResult(
        String answer,
        ExtractionPayload payload,
        VerificationResult verification,
        PipelineTrace trace
) {
    public PipelineResult {
        if (answer == null) answer = "";
        if (payload == null) {
            throw new IllegalArgumentException("payload must not be null");
        }
        if (verification == null) {
            throw new IllegalArgumentException("verification must not be null");
        }
        if (trace == null) {
            throw new IllegalArgumentException("trace must not be null");
        }
    }
}
