package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.ExtractionPayload;

/** 受控答案合成接口，只接收已经验证的结构化事实。 */
public interface AnswerSynthesizer {
    String synthesize(String query, ExtractionPayload payload);
}
