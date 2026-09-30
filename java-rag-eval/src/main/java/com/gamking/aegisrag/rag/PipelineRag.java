package com.gamking.aegisrag.rag;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.PipelineResult;
import com.gamking.aegisrag.model.RagResponse;
import com.gamking.aegisrag.pipeline.*;

import java.util.function.Function;

/**
 * 把检索上下文接入受控流水线，并保持 BaseRag 接口。
 * 
 * contextProvider 只负责返回检索上下文；抽取、验证、自愈和合成由受控流水线统一管理。
 */
public class PipelineRag implements BaseRag {

    private final Function<EvalSample, String> contextProvider;
    private final ControlledPipeline pipeline;

    private PipelineResult lastResult;

    public PipelineRag(
            Function<EvalSample, String> contextProvider,
            FactExtractor extractor,
            AnswerSynthesizer synthesizer,
            Tier1CodeVerifier verifier,
            SelfCorrector selfCorrector
    ) {
        this.contextProvider = contextProvider;
        this.pipeline = new ControlledPipeline(
                extractor != null ? extractor : new RuleBasedFactExtractor(),
                synthesizer != null ? synthesizer : new ControlledSynthesizer(),
                verifier != null ? verifier : new Tier1CodeVerifier(),
                selfCorrector
        );
    }

    @Override
    public RagResponse retrieveAndGenerate(EvalSample sample) {
        String context = contextProvider.apply(sample);
        PipelineResult result = pipeline.run(sample, context);
        this.lastResult = result;
        return new RagResponse(context, result.answer());
    }

    public PipelineResult getLastResult() {
        return lastResult;
    }
}