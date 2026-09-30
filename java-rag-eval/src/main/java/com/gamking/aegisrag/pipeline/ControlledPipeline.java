package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.PipelineResult;
import com.gamking.aegisrag.model.PipelineTrace;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;

import java.util.ArrayList;
import java.util.List;

/**
 * 受控流水线：受限抽取 -> Tier1 质检 -> Tier2 自愈 -> Tier1 复验 -> 受控合成。
 */
public class ControlledPipeline {

    private final FactExtractor extractor;
    private final AnswerSynthesizer synthesizer;
    private final Tier1CodeVerifier verifier;
    private final SelfCorrector selfCorrector;

    public ControlledPipeline(
            FactExtractor extractor,
            AnswerSynthesizer synthesizer,
            Tier1CodeVerifier verifier,
            SelfCorrector selfCorrector
    ) {
        this.extractor = extractor;
        this.synthesizer = synthesizer;
        this.verifier = verifier;
        this.selfCorrector = selfCorrector;
    }

    public PipelineResult run(EvalSample sample, String context) {
        ExtractionPayload payload = extractor.extract(sample.question(), context);
        VerificationResult first = verifier.verify(sample, context, payload);
        PipelineTrace trace = PipelineTrace.initial(first.passed(), first.findings());
        VerificationResult finalResult = first;

        if (!first.passed() && selfCorrector != null) {
            int originalCount = payload.facts().size();
            payload = selfCorrector.correct(context, payload, first.findings());
            finalResult = verifier.verify(sample, context, payload);

            // 若 Tier2 删除了事实，则安全失败
            if (payload.facts().size() < originalCount) {
                List<VerificationFinding> combined = new ArrayList<>(first.findings());
                combined.add(VerificationFinding.error(
                        "REMOVED_UNVERIFIED_FACT",
                        "Tier2 删除了未被上下文支持的事实，进入安全失败",
                        "pipeline"));
                finalResult = VerificationResult.fail("tier1", combined);
            }

            List<VerificationFinding> allFindings = new ArrayList<>(first.findings());
            allFindings.addAll(finalResult.findings());
            trace = trace.withTier2(allFindings, finalResult.passed());
        }

        String answer = finalResult.passed()
                ? synthesizer.synthesize(sample.question(), payload)
                : "";
        return new PipelineResult(answer, payload, finalResult, trace);
    }
}