package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.VerificationFinding;

import java.util.List;

/** Tier 2 自愈接口。 */
public interface SelfCorrector {
    ExtractionPayload correct(
            String context,
            ExtractionPayload payload,
            List<VerificationFinding> findings
    );
}
