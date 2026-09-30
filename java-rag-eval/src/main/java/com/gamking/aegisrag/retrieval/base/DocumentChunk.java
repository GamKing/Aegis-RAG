package com.gamking.aegisrag.retrieval.base;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 检索文档单元。
 * 
 * metadata 约定字段：
 * - index: 块在文档中的序号
 * - doc_id: 源文档标识
 * - section: 所属章节（法律场景）
 * - kind: 块类型（legal_clause / financial_table_row 等）
 * - page: 页码（PDF 场景）
 * - line_start / line_end: 行号范围（溯源用）
 */
public record DocumentChunk(
        String chunkId,
        String content,
        Map<String, String> metadata
) {
    public DocumentChunk {
        Objects.requireNonNull(chunkId, "chunkId must not be null");
        Objects.requireNonNull(content, "content must not be null");
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }

    public DocumentChunk(String chunkId, String content) {
        this(chunkId, content, Map.of());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String chunkId;
        private String content;
        private final Map<String, String> metadata = new HashMap<>();

        public Builder chunkId(String chunkId) {
            this.chunkId = chunkId;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder metadata(String key, String value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, String> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public DocumentChunk build() {
            return new DocumentChunk(chunkId, content, metadata);
        }
    }
}
