package com.gamking.aegisrag.rag;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.RagResponse;

import java.util.Map;

/** 查表式假 RAG，用于离线演示和测试。 */
public class DummyRag implements BaseRag {

    private final Map<String, RagResponse> presets;

    public DummyRag(Map<String, RagResponse> presets) {
        this.presets = presets;
    }

    @Override
    public RagResponse retrieveAndGenerate(EvalSample sample) {
        RagResponse preset = presets.get(sample.id());
        if (preset != null) {
            return preset;
        }
        return new RagResponse("", "抱歉，未能检索到相关政策内容。");
    }
}