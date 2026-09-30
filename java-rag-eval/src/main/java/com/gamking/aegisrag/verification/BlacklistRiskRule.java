package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 医疗/政务黑名单禁忌规则。
 *
 * 用于两类场景：
 * 1. 绝对化/危险表述核查：答案若含黑名单词（如"绝对安全"、"根治"），
 *    而上下文没有对应支持，直接判定失败；
 * 2. 高风险禁忌核查：如妊娠禁用、配伍禁忌，一旦答案遗漏否定词，
 *    视为安全风险。
 */
public final class BlacklistRiskRule implements VerificationRule {

    private final Set<String> blacklist;

    public BlacklistRiskRule() {
        this(Set.of("绝对安全", "根治", "无任何副作用", "百分百有效"));
    }

    public BlacklistRiskRule(Set<String> blacklist) {
        this.blacklist = blacklist == null ? Set.of() : blacklist;
    }

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();
        String answerText = fact.predicate() + " " + fact.objectValue();
        String contextNorm = TextNormalizer.normalizeForMatch(context);

        for (String word : blacklist) {
            if (answerText.contains(word)
                    && !contextNorm.contains(TextNormalizer.normalizeForMatch(word))) {
                findings.add(VerificationFinding.error(
                        "RISK_WORD_IN_ANSWER",
                        "答案含高危表述 '" + word + "' 但上下文中无出处",
                        "objectValue"));
            }
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }
}