package com.gamking.aegisrag.model;

/** Tier 1/Tier 2 的结构化质检发现。 */
public record VerificationFinding(
        String code,
        String message,
        String field,
        String severity
) {
    public VerificationFinding {
        if (code == null || code.isBlank()) {
            throw new IllegalArgumentException("code must not be blank");
        }
        if (message == null || message.isBlank()) {
            throw new IllegalArgumentException("message must not be blank");
        }
        severity = severity == null ? "error" : severity;
        field = field == null ? "" : field;
    }

    public static VerificationFinding error(String code, String message, String field) {
        return new VerificationFinding(code, message, field, "error");
    }

    public static VerificationFinding warning(String code, String message, String field) {
        return new VerificationFinding(code, message, field, "warning");
    }
}
