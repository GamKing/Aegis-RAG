package com.gamking.aegisrag.rag;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.RagResponse;

/** 被测 RAG 系统的最小接口。 */
public interface BaseRag {
    RagResponse retrieveAndGenerate(EvalSample sample);
}
