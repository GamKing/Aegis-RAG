package com.gamking.aegisrag.model;

import java.util.Objects;

/**
 * 通用原子事实：主体-谓词-客体三元组，支持多类型校验。
 */
public record AtomicFact(
        String subject,
        String predicate,
        String objectValue,
        FactType factType,
        boolean isNegative,
        String sourceText
) {
    public AtomicFact {
        Objects.requireNonNull(subject, "subject must not be null");
        Objects.requireNonNull(predicate, "predicate must not be null");
        Objects.requireNonNull(objectValue, "objectValue must not be null");
        factType = factType == null ? FactType.NUMERIC : factType;
        sourceText = sourceText == null ? "" : sourceText;
    }

    public AtomicFact(String subject, String predicate, String objectValue, FactType factType) {
        this(subject, predicate, objectValue, factType, false, "");
    }

    public AtomicFact(String subject, String predicate, String objectValue, FactType factType, boolean isNegative) {
        this(subject, predicate, objectValue, factType, isNegative, "");
    }
}