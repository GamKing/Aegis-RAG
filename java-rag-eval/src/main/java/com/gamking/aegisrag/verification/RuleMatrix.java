package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.FactType;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * 校验规则矩阵：根据事实类型分发到对应规则。
 *
 * 行业插槽 3：通过 register 方法挂载行业专属规则插件
 * （如金融量纲换算 UnitConversionRule、医疗黑名单 BlacklistRiskRule）。
 */
public final class RuleMatrix {

    private final Map<FactType, VerificationRule> rules = new EnumMap<>(FactType.class);
    private final List<VerificationRule> extraRules = new ArrayList<>();

    public RuleMatrix() {
        rules.put(FactType.NUMERIC, new NumericRule());
        rules.put(FactType.ENTITY_REL, new RelationRule());
        rules.put(FactType.CONDITION, new ScopeRule());
        rules.put(FactType.CATEGORICAL, new CategoricalRule());
    }

    /**
     * 注册一个行业专属规则插件（作用于全部事实类型的交叉检查）。
     */
    public void register(VerificationRule rule) {
        if (rule != null) {
            extraRules.add(rule);
        }
    }

    /**
     * 根据事实类型分发到对应规则校验，并叠加行业额外规则。
     */
    public VerificationResult verify(AtomicFact fact, String context) {
        VerificationRule rule = rules.get(fact.factType());
        List<VerificationFinding> findings = new ArrayList<>();

        if (rule == null) {
            findings.add(VerificationFinding.warning(
                    "UNKNOWN_FACT_TYPE",
                    "未知事实类型: " + fact.factType(),
                    "factType"));
        } else {
            findings.addAll(rule.verify(fact, context).findings());
        }

        // 叠加行业额外规则
        for (VerificationRule extra : extraRules) {
            findings.addAll(extra.verify(fact, context).findings());
        }

        boolean hasErrors = findings.stream()
                .anyMatch(f -> "error".equals(f.severity()));
        return new VerificationResult(!hasErrors, findings, "tier1");
    }
}