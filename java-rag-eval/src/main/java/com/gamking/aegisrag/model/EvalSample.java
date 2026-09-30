package com.gamking.aegisrag.model;

import java.util.List;

/** 单条评测样本：问题、金标准上下文、标准答案和必须命中的实体。 */
public record EvalSample(
        String id,
        String question,
        String groundTruthContext,
        String groundTruthAnswer,
        List<String> expectedEntities
) {
    public EvalSample {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("sample id must not be blank");
        }
        if (question == null || groundTruthContext == null || groundTruthAnswer == null) {
            throw new IllegalArgumentException("question/context/answer must not be null");
        }
        expectedEntities = expectedEntities == null ? List.of() : List.copyOf(expectedEntities);
    }
}
