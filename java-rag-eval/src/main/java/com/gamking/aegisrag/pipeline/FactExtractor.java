package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.ExtractionPayload;

/** 受限事实抽取接口，可由规则实现或 Ollama JSON Schema 实现。 */
public interface FactExtractor {
    ExtractionPayload extract(String query, String context);
}
