package com.gamking.aegisrag.model;

import java.util.List;

/** 分层流水线审计轨迹。 */
public record PipelineTrace(
        boolean tier1InitialPassed,
        boolean tier2Triggered,
        int repairCount,
        boolean tier1FinalPassed,
        List<VerificationFinding> findings
) {
    public PipelineTrace {
        findings = findings == null ? List.of() : List.copyOf(findings);
    }

    public static PipelineTrace initial(boolean tier1Passed, List<VerificationFinding> findings) {
        return new PipelineTrace(tier1Passed, false, 0, tier1Passed, findings);
    }

    public PipelineTrace withTier2(List<VerificationFinding> allFindings, boolean finalPassed) {
        return new PipelineTrace(
                this.tier1InitialPassed,
                true,
                1,
                finalPassed,
                allFindings
        );
    }
}
