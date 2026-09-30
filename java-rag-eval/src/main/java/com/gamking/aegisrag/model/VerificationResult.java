package com.gamking.aegisrag.model;

import java.util.List;

/** 一次质检的结果。 */
public record VerificationResult(
        boolean passed,
        List<VerificationFinding> findings,
        String tier
) {
    public VerificationResult {
        if (tier == null || tier.isBlank()) {
            throw new IllegalArgumentException("tier must not be blank");
        }
        findings = findings == null ? List.of() : List.copyOf(findings);
    }

    public static VerificationResult pass(String tier) {
        return new VerificationResult(true, List.of(), tier);
    }

    public static VerificationResult fail(String tier, List<VerificationFinding> findings) {
        return new VerificationResult(false, findings, tier);
    }
}
