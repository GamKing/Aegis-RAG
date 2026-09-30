package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationFinding;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 金融量纲换算规则。
 *
 * 识别财报中的单位（万元/亿元/美元）并做等价映射。允许答案中的数值以
 * 等价量纲出现在上下文中。例如：
 * - 答案 "0.5亿" 可被上下文 "5000万" 支持；
 * - 答案 "108.0 billion" 可被上下文 "108000000000" 支持。
 */
public final class UnitConversionRule implements VerificationRule {

    /** 量纲换算表：统一折算到"元" */
    private static final Map<String, Double> SCALES = Map.ofEntries(
            Map.entry("元", 1.0),
            Map.entry("万", 1e4),
            Map.entry("万元", 1e4),
            Map.entry("亿", 1e8),
            Map.entry("亿元", 1e8),
            Map.entry("千万", 1e7),
            Map.entry("百万", 1e6),
            Map.entry("million", 1e6),
            Map.entry("千", 1e3),
            Map.entry("thousand", 1e3),
            Map.entry("k", 1e3),
            Map.entry("m", 1e6),
            Map.entry("b", 1e9),
            Map.entry("billion", 1e9)
    );

    private static final Pattern NUM_RE = Pattern.compile(
            "(?<value>\\d+(?:\\.\\d+)?)\\s*(?<unit>亿元|万元|亿|万|元|千万|百万|billion|million|thousand|k|m|b)?");

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();
        List<Double> factAmounts = normalizeNumbersWithScale(fact.objectValue());

        if (factAmounts.isEmpty()) {
            return VerificationResult.pass("tier1");
        }

        List<Double> contextAmounts = normalizeNumbersWithScale(context);
        for (double amount : factAmounts) {
            boolean supported = false;
            for (double ctxAmount : contextAmounts) {
                double tolerance = Math.max(Math.abs(amount) * 1e-6, 1e-6);
                if (Math.abs(amount - ctxAmount) < tolerance) {
                    supported = true;
                    break;
                }
            }
            if (!supported) {
                findings.add(VerificationFinding.error(
                        "UNIT_MISMATCH",
                        "数值/量纲 '" + fact.objectValue() + "' 在上下文中无等价出处",
                        "objectValue"));
            }
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }

    /**
     * 把文本中的带单位数值统一折算为元。
     */
    private static List<Double> normalizeNumbersWithScale(String text) {
        List<Double> results = new ArrayList<>();
        Matcher matcher = NUM_RE.matcher(text.toLowerCase());
        while (matcher.find()) {
            double value = Double.parseDouble(matcher.group("value"));
            String unit = matcher.group("unit");
            double scale = SCALES.getOrDefault(unit == null ? "元" : unit, 1.0);
            results.add(value * scale);
        }
        return results;
    }
}