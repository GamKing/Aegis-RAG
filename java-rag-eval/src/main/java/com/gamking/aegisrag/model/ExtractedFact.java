package com.gamking.aegisrag.model;

/** 受限抽取后的单个原子事实。 */
public record ExtractedFact(
        String key,
        String value,
        String sourceText,
        boolean isNumeric
) {
    public ExtractedFact {
        if (key == null || key.isBlank()) {
            throw new IllegalArgumentException("key must not be blank");
        }
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("value must not be blank");
        }
    }
}
