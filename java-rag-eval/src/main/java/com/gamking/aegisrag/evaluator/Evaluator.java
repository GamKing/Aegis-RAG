package com.gamking.aegisrag.evaluator;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.MetricOutcome;

/** 评测器接口。 */
public interface Evaluator {
    /**
     * 评估单个样本。
     * 
     * @param sample 评测样本
     * @param actualContext RAG 实际检索到的上下文
     * @param actualAnswer RAG 实际生成的答案
     * @return 评测结果
     */
    MetricOutcome evaluate(EvalSample sample, String actualContext, String actualAnswer);
}
