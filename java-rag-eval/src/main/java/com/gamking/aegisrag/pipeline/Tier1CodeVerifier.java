package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;
import com.gamking.aegisrag.verification.RuleMatrix;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tier 1 纯代码质检：使用规则矩阵验证多种类型的事实。
 *
 * 支持四种事实类型：NUMERIC、ENTITY_REL、CONDITION、CATEGORICAL。
 * 同时保持旧式 ExtractedFact 的验证逻辑，确保向后兼容。
 */
public class Tier1CodeVerifier {

    private final RuleMatrix ruleMatrix;

    public Tier1CodeVerifier() {
        this(new RuleMatrix());
    }

    public Tier1CodeVerifier(RuleMatrix ruleMatrix) {
        this.ruleMatrix = ruleMatrix == null ? new RuleMatrix() : ruleMatrix;
    }

    public VerificationResult verify(EvalSample sample, String context, ExtractionPayload payload) {
        List<VerificationFinding> findings = new ArrayList<>();
        String normalizedContext = TextNormalizer.normalizeText(context);
        Set<String> seen = new HashSet<>();

        // 1. 验证 AtomicFact（规则矩阵）
        for (int index = 0; index < payload.atomicFacts().size(); index++) {
            AtomicFact fact = payload.atomicFacts().get(index);
            String field = "atomicFacts[" + index + "]";

            if (fact.objectValue().isBlank()) {
                findings.add(VerificationFinding.error("EMPTY_FACT", "事实值为空", field));
                continue;
            }

            VerificationResult ruleResult = ruleMatrix.verify(fact, context);
            findings.addAll(ruleResult.findings());
        }

        // 2. 验证旧式 ExtractedFact（保持向后兼容）
        for (int index = 0; index < payload.facts().size(); index++) {
            ExtractedFact fact = payload.facts().get(index);
            String field = "facts[" + index + "]";
            String valueNorm = TextNormalizer.normalizeForMatch(fact.value());

            if (valueNorm.isEmpty()) {
                findings.add(VerificationFinding.error("EMPTY_FACT", "事实值为空", field));
                continue;
            }
            if (!seen.add(valueNorm)) {
                findings.add(VerificationFinding.error("DUPLICATE_FACT", "事实重复", field));
            }

            List<TextNormalizer.NumberUnit> numbers = TextNormalizer.extractNumbers(fact.value());
            if (!numbers.isEmpty()) {
                for (TextNormalizer.NumberUnit unit : numbers) {
                    if (!TextNormalizer.numberSupportedInContext(
                            unit.value(), unit.isPercent(), normalizedContext)) {
                        String raw = unit.value() + (unit.isPercent() ? "%" : "");
                        findings.add(VerificationFinding.error(
                                "UNSUPPORTED_NUMBER",
                                "数值 " + raw + " 在上下文中无出处",
                                field));
                    }
                }
            } else if (!normalizedContext.toLowerCase().contains(valueNorm.toLowerCase())) {
                findings.add(VerificationFinding.error(
                        "UNSUPPORTED_FACT",
                        "事实 " + fact.value() + " 在上下文中无出处",
                        field));
            }
        }

        // 3. 验证预期实体（保持原有逻辑）
        String contextNorm = TextNormalizer.normalizeForMatch(context);
        for (String entity : sample.expectedEntities()) {
            if (!contextNorm.contains(TextNormalizer.normalizeForMatch(entity))) {
                findings.add(VerificationFinding.warning(
                        "MISSING_EXPECTED_ENTITY",
                        "标准实体 " + entity + " 未在上下文中出现",
                        "expected_entities"));
            }
        }

        boolean passed = findings.stream().noneMatch(f -> "error".equals(f.severity()));
        return passed
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }
}