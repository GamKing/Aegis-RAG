package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 分类/枚举事实校验规则：检查分类标签是否在上下文中出现。
 */
public final class CategoricalRule implements VerificationRule {

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();

        if (fact.objectValue().isBlank()) {
            return VerificationResult.pass("tier1");
        }

        if (!TextNormalizer.normalizeForMatch(context)
                .contains(TextNormalizer.normalizeForMatch(fact.objectValue()))) {
            findings.add(VerificationFinding.error(
                    "CATEGORY_NOT_FOUND",
                    "分类标签 '" + fact.objectValue() + "' 在上下文中无出处",
                    "objectValue"));
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }
}