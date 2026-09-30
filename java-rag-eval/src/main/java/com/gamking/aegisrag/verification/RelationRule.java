package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 实体关系事实校验规则：检查实体对是否共现。
 *
 * 校验逻辑：
 * 1. 主体（subject）必须出现在上下文中；
 * 2. 客体（objectValue）必须出现在上下文中；
 * 3. 如果是否定型事实，还需检查否定词是否同现。
 */
public final class RelationRule implements VerificationRule {

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();

        // 检查主体和客体是否在上下文中出现
        if (!fact.subject().isBlank() && !TextNormalizer.normalizeForMatch(context).contains(TextNormalizer.normalizeForMatch(fact.subject()))) {
            findings.add(VerificationFinding.error(
                    "ENTITY_NOT_FOUND",
                    "主体 '" + fact.subject() + "' 在上下文中无出处",
                    "subject"));
        }
        if (!fact.objectValue().isBlank() && !TextNormalizer.normalizeForMatch(context).contains(TextNormalizer.normalizeForMatch(fact.objectValue()))) {
            findings.add(VerificationFinding.error(
                    "RELATION_OBJECT_NOT_FOUND",
                    "关系客体 '" + fact.objectValue() + "' 在上下文中无出处",
                    "objectValue"));
        }

        // 如果是否定型事实，检查否定词是否同现
        if (fact.isNegative()) {
            String relationText = fact.predicate() + " " + fact.objectValue();
            if (!TextNormalizer.detectNegation(relationText)) {
                findings.add(VerificationFinding.error(
                        "NEGATION_NOT_FOUND",
                        "否定谓词 '" + fact.predicate() + "' 在上下文中无出处",
                        "predicate"));
            }
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }
}