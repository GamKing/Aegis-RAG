package com.gamking.aegisrag;

import com.gamking.aegisrag.model.*;
import com.gamking.aegisrag.pipeline.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** 受控流水线测试：Tier1 直通、Tier2 自愈和对抗数字拦截。 */
public class PipelineTest {

    private static EvalSample sample(String context) {
        return new EvalSample("pipeline-test", "金额是多少？", context, "answer", List.of());
    }

    private static class FakeJudge implements com.gamking.aegisrag.nli.NliJudge {
        private final boolean verdict;
        private int calls;

        private FakeJudge(boolean verdict) {
            this.verdict = verdict;
        }

        @Override
        public boolean checkFaithfulness(String context, String claim) {
            calls++;
            return verdict;
        }
    }

    @Test
    public void tier1_direct_path_skips_nli() {
        String context = "Revenue was $108.0 billion, up 18%.";
        FakeJudge judge = new FakeJudge(true);
        ControlledPipeline pipeline = new ControlledPipeline(
                new RuleBasedFactExtractor(), new ControlledSynthesizer(),
                new Tier1CodeVerifier(), new NliSelfCorrector(judge));

        PipelineResult result = pipeline.run(sample(context), context);
        assertTrue(result.verification().passed());
        assertFalse(result.trace().tier2Triggered());
        assertEquals(0, judge.calls);
    }

    @Test
    public void tier1_blocks_unsupported_number() {
        String context = "Revenue was $108.0 billion.";
        ExtractionPayload payload = ExtractionPayload.of(
                List.of(new ExtractedFact("revenue", "$95.0 billion", null, true)),
                List.of("revenue: $95.0 billion"));
        FactExtractor extractor = (query, ignored) -> payload;
        VerificationResult result = new Tier1CodeVerifier().verify(
                sample(context), context, extractor.extract("q", context));
        assertFalse(result.passed());
        assertTrue(result.findings().stream().anyMatch(f -> f.code().equals("UNSUPPORTED_NUMBER")));
    }

    @Test
    public void nli_entailment_can_retain_semantic_zero() {
        String context = "NVIDIA is not assuming any Data Center compute revenue from China.";
        ExtractionPayload payload = ExtractionPayload.of(
                List.of(new ExtractedFact("China revenue", "0", null, true)),
                List.of("China revenue: 0"));
        FakeJudge judge = new FakeJudge(true);
        ControlledPipeline pipeline = new ControlledPipeline(
                (query, ignored) -> payload, new ControlledSynthesizer(),
                new Tier1CodeVerifier(), new NliSelfCorrector(judge));

        PipelineResult result = pipeline.run(sample(context), context);
        assertTrue(result.trace().tier2Triggered());
        assertTrue(result.verification().passed());
        assertTrue(result.answer().contains("0"));
        assertEquals(1, judge.calls);
    }

    @Test
    public void nli_rejects_adversarial_number() {
        String context = "Edge Computing revenue was $7.2 billion.";
        ExtractionPayload payload = ExtractionPayload.of(
                List.of(new ExtractedFact("revenue", "$9.5 billion", null, true)),
                List.of("revenue: $9.5 billion"));
        FakeJudge judge = new FakeJudge(false);
        ControlledPipeline pipeline = new ControlledPipeline(
                (query, ignored) -> payload, new ControlledSynthesizer(),
                new Tier1CodeVerifier(), new NliSelfCorrector(judge));

        PipelineResult result = pipeline.run(sample(context), context);
        assertFalse(result.verification().passed());
        assertEquals("", result.answer());
    }
}