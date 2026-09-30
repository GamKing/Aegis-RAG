package com.gamking.aegisrag.pipeline;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.model.FactType;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 离线安全抽取器：从上下文原文提取多种类型的事实。
 *
 * 支持四种事实类型：
 * - NUMERIC: 数值型（百分比、金额、日期等）
 * - ENTITY_REL: 实体关系型（含否定词的关系）
 * - CONDITION: 条件限制型（含"仅限"、"必须"等关键词）
 * - CATEGORICAL: 分类枚举型
 */
public class RuleBasedFactExtractor implements FactExtractor {

    private static final Pattern NUMBER_WITH_SCALE = Pattern.compile(
            "(?<raw>\\$?\\d[\\d,]*(?:\\.\\d+)?\\s*(?:million|billion|亿|万)?%?)",
            Pattern.CASE_INSENSITIVE
    );

    private static final Pattern CJK_RUN = Pattern.compile("[\\u4e00-\\u9fff]+");

    @Override
    public ExtractionPayload extract(String query, String context) {
        String normalized = TextNormalizer.normalizeText(context);
        List<ExtractedFact> facts = new ArrayList<>();
        List<AtomicFact> atomicFacts = new ArrayList<>();
        Set<String> seen = new LinkedHashSet<>();

        // 1. 数值型事实
        Matcher matcher = NUMBER_WITH_SCALE.matcher(normalized);
        while (matcher.find()) {
            String value = matcher.group("raw").trim();
            String key = TextNormalizer.normalizeForMatch(value);
            if (!value.isBlank() && seen.add(key)) {
                facts.add(new ExtractedFact("numeric_fact", value, matcher.group(0), true));
            }
        }

        // 2. 实体关系型 + 条件型（含否定词）
        for (String negWord : TextNormalizer.NEGATION_WORDS) {
            int idx = normalized.indexOf(negWord);
            if (idx < 0) continue;
            String before = normalized.substring(Math.max(0, idx - 10), idx);
            String after = normalized.substring(
                    idx + negWord.length(),
                    Math.min(normalized.length(), idx + negWord.length() + 10));
            List<String> beforeEntities = extractCjkTerms(before, 2, 6);
            List<String> afterEntities = extractCjkTerms(after, 2, 6);
            if (!beforeEntities.isEmpty() && !afterEntities.isEmpty()) {
                String subject = beforeEntities.get(beforeEntities.size() - 1);
                String objectValue = afterEntities.get(0);
                atomicFacts.add(new AtomicFact(
                        subject, negWord, objectValue, FactType.ENTITY_REL, true));
            }
        }

        // 3. 条件限制型事实
        for (String condWord : TextNormalizer.CONDITION_KEYWORDS) {
            int idx = normalized.indexOf(condWord);
            if (idx < 0) continue;
            String after = normalized.substring(idx, Math.min(normalized.length(), idx + 20));
            List<String> condEntities = extractCjkTerms(after, 2, 10);
            if (!condEntities.isEmpty()) {
                atomicFacts.add(new AtomicFact(
                        "", condWord, condEntities.get(0), FactType.CONDITION, false));
            }
        }

        List<String> claims = facts.stream().map(ExtractedFact::value).toList();
        return ExtractionPayload.of(facts, claims, atomicFacts);
    }

    /** 提取中文连续段中长度 [minLen, maxLen] 的词组。 */
    private static List<String> extractCjkTerms(String text, int minLen, int maxLen) {
        List<String> terms = new ArrayList<>();
        Matcher matcher = CJK_RUN.matcher(text);
        while (matcher.find()) {
            String run = matcher.group();
            for (int i = 0; i <= run.length() - minLen; i++) {
                for (int len = minLen; len <= Math.min(maxLen, run.length() - i); len++) {
                    terms.add(run.substring(i, i + len));
                }
            }
        }
        return terms;
    }
}