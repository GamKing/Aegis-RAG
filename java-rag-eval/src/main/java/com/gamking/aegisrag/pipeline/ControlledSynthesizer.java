package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.FactType;

import java.util.ArrayList;
import java.util.List;

/**
 * 只输出已验证的事实，不重新自由生成数字。
 *
 * 支持两种输入：
 * - 旧式 ExtractedFact（key/value 对）：保持向后兼容，输出 "key: value"
 * - 新式 AtomicFact（主体/谓词/客体三元组）：输出自然语言句子
 */
public class ControlledSynthesizer implements AnswerSynthesizer {

    @Override
    public String synthesize(String query, ExtractionPayload payload) {
        // 优先使用 AtomicFact（新式三元组）
        if (!payload.atomicFacts().isEmpty()) {
            return synthesizeAtomic(payload.atomicFacts());
        }

        // 回退到旧式 ExtractedFact
        if (!payload.facts().isEmpty()) {
            return payload.facts().stream()
                    .map(f -> f.key() + ": " + f.value())
                    .reduce((a, b) -> a + "；" + b)
                    .orElse("");
        }

        return "";
    }

    private String synthesizeAtomic(List<AtomicFact> atomicFacts) {
        List<String> sentences = new ArrayList<>();
        for (AtomicFact fact : atomicFacts) {
            if (fact.objectValue().isBlank()) continue;

            String neg = fact.isNegative() ? "不得" : "";
            String subject = fact.subject();
            String predicate = fact.predicate();
            String objectValue = fact.objectValue();

            if (FactType.ENTITY_REL.equals(fact.factType()) && subject.isBlank()) {
                sentences.add(neg + predicate + objectValue);
            } else if (subject.isBlank()) {
                sentences.add(predicate + objectValue);
            } else {
                sentences.add(subject + neg + predicate + objectValue);
            }
        }
        return String.join("；", sentences) + (sentences.isEmpty() ? "" : "。");
    }
}