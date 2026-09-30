package com.gamking.aegisrag.model;

/** RAG 链路的单次产出：检索上下文 + 生成答案。 */
public record RagResponse(
        String context,
        String answer
) {
    public RagResponse {
        if (context == null) context = "";
        if (answer == null) answer = "";
    }
}
