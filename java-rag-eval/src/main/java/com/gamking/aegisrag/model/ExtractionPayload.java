package com.gamking.aegisrag.model;

import java.util.List;

/** LLM JSON Schema 或规则抽取器的统一输出。 */
public record ExtractionPayload(
        List<ExtractedFact> facts,
        List<String> claims,
        String rawJson,
        List<AtomicFact> atomicFacts
) {
    public ExtractionPayload {
        facts = facts == null ? List.of() : List.copyOf(facts);
        claims = claims == null ? List.of() : List.copyOf(claims);
        rawJson = rawJson == null ? "" : rawJson;
        atomicFacts = atomicFacts == null ? List.of() : List.copyOf(atomicFacts);
    }

    public ExtractionPayload(List<ExtractedFact> facts, List<String> claims, String rawJson) {
        this(facts, claims, rawJson, List.of());
    }

    public static ExtractionPayload of(List<ExtractedFact> facts, List<String> claims) {
        return new ExtractionPayload(facts, claims, "", List.of());
    }

    public static ExtractionPayload of(List<ExtractedFact> facts, List<String> claims, List<AtomicFact> atomicFacts) {
        return new ExtractionPayload(facts, claims, "", atomicFacts);
    }
}