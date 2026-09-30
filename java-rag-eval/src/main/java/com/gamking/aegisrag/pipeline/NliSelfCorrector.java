package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.nli.NliJudge;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/** 对 Tier1 失败事实逐条做 NLI 判定，并最多执行一次修复。 */
public class NliSelfCorrector implements SelfCorrector {

    private final NliJudge judge;

    public NliSelfCorrector(NliJudge judge) {
        this.judge = judge;
    }

    @Override
    public ExtractionPayload correct(
            String context,
            ExtractionPayload payload,
            List<VerificationFinding> findings
    ) {
        Set<String> unsupportedFields = findings.stream()
                .filter(f -> f.code().equals("UNSUPPORTED_NUMBER") || f.code().equals("UNSUPPORTED_FACT"))
                .map(VerificationFinding::field)
                .filter(f -> !f.isEmpty())
                .collect(Collectors.toSet());

        List<ExtractedFact> kept = new ArrayList<>();
        List<ExtractedFact> nliVerified = new ArrayList<>();

        for (int index = 0; index < payload.facts().size(); index++) {
            ExtractedFact fact = payload.facts().get(index);
            String field = "facts[" + index + "]";
            if (!unsupportedFields.contains(field)) {
                kept.add(fact);
                continue;
            }
            String claim = fact.key() + ": " + fact.value();
            boolean accepted;
            try {
                accepted = judge.checkFaithfulness(context, claim);
            } catch (Exception e) {
                accepted = false;
            }
            if (accepted) {
                kept.add(fact);
                nliVerified.add(fact);
            }
        }

        return new ExtractionPayload(kept,
                kept.stream().map(ExtractedFact::value).toList(),
                "{\"self_corrected\":true}");
    }
}