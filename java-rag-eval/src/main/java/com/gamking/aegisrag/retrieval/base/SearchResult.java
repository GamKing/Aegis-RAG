package com.gamking.aegisrag.retrieval.base;

import java.util.Objects;

/**
 * 带得分的检索结果项。
 */
public record SearchResult(
        DocumentChunk chunk,
        double score
) {
    public SearchResult {
        Objects.requireNonNull(chunk, "chunk must not be null");
    }
}
