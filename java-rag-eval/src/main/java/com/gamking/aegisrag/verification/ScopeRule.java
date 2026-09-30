package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 限制/条件事实校验规则。
 *
 * 检查条件关键词与具体条件值是否在上下文中出现。
 */
public final class ScopeRule implements VerificationRule {

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();
        String contextNorm = TextNormalizer.normalizeForMatch(context);
        String conditionText = fact.predicate() + " " + fact.objectValue();

        for (String keyword : TextNormalizer.extractConditionKeywords(conditionText)) {
            if (!contextNorm.contains(TextNormalizer.normalizeForMatch(keyword))) {
                findings.add(VerificationFinding.error(
                        "CONDITION_NOT_FOUND",
                        "条件关键词 '" + keyword + "' 在上下文中无出处",
                        "predicate"));
            }
        }

        if (!fact.objectValue().isBlank()
                && !contextNorm.contains(TextNormalizer.normalizeForMatch(fact.objectValue()))) {
            findings.add(VerificationFinding.error(
                    "CONDITION_VALUE_NOT_FOUND",
                    "条件值 '" + fact.objectValue() + "' 在上下文中无出处",
                    "objectValue"));
        }

        if (fact.isNegative() && !TextNormalizer.detectNegation(conditionText)) {
            findings.add(VerificationFinding.error(
                    "NEGATION_MISSING",
                    "事实标记为否定型，但未检测到否定词",
                    "isNegative"));
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }
}