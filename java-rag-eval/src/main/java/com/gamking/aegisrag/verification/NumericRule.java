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
 * 数值型事实校验规则：边界敏感匹配 + 量纲换算等价支持。
 *
 * 除了常规的边界敏感数字匹配，还支持量纲换算（如"0.5亿" ⇔ "5000万"、
 * "108.0 billion" ⇔ "108000000000"），适用于金融财报类场景。
 */
public final class NumericRule implements VerificationRule {

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

    private static final Pattern NUM_WITH_UNIT = Pattern.compile(
            "(?<value>\\d+(?:\\.\\d+)?)\\s*(?<unit>亿元|万元|亿|万|元|千万|百万|billion|million|thousand|k|m|b)?");

    @Override
    public VerificationResult verify(AtomicFact fact, String context) {
        List<VerificationFinding> findings = new ArrayList<>();
        List<TextNormalizer.NumberUnit> numbers = TextNormalizer.extractNumbers(fact.objectValue());

        String normalizedContext = TextNormalizer.normalizeText(context);
        double tolerance = 1e-6;

        // 逐个数值检查：先边界敏感匹配，再量纲换算等价匹配
        for (TextNormalizer.NumberUnit number : numbers) {
            boolean directlySupported = TextNormalizer.numberSupportedInContext(
                    number.value(), number.isPercent(), normalizedContext);
            if (directlySupported) continue;

            // 尝试量纲换算等价匹配
            boolean scaleSupported = isScaleEquivalent(number, fact.objectValue(), context, tolerance);
            if (!scaleSupported) {
                String raw = number.value() + (number.isPercent() ? "%" : "");
                findings.add(VerificationFinding.error(
                        "UNSUPPORTED_NUMBER",
                        "数值 " + raw + " 在上下文中无出处（含量纲换算）",
                        "objectValue"));
            }
        }

        return findings.isEmpty()
                ? VerificationResult.pass("tier1")
                : VerificationResult.fail("tier1", findings);
    }

    private static boolean isScaleEquivalent(
            TextNormalizer.NumberUnit number, String factValue, String context, double tolerance) {
        // 只对带单位的非百分数做量纲换算（百分数无量纲换算概念）
        if (number.isPercent()) return false;
        double factAmount = Double.parseDouble(number.value());
        // 换算 fact 中该数值可能带的单位
        double factScale = findScaleInText(factValue, number.value());
        double factTotal = factAmount * factScale;

        List<Double> contextAmounts = normalizeNumbersWithScale(context);
        double localTolerance = Math.max(Math.abs(factTotal) * 1e-6, tolerance);
        for (double ctxAmount : contextAmounts) {
            if (Math.abs(factTotal - ctxAmount) < localTolerance) {
                return true;
            }
        }
        return false;
    }

    /** 在文本里找到与特定数值相邻的单位倍率。 */
    private static double findScaleInText(String text, String value) {
        Matcher matcher = NUM_WITH_UNIT.matcher(text.toLowerCase());
        while (matcher.find()) {
            if (matcher.group("value").equals(value)) {
                String unit = matcher.group("unit");
                return SCALES.getOrDefault(unit == null ? "元" : unit, 1.0);
            }
        }
        return 1.0;
    }

    private static List<Double> normalizeNumbersWithScale(String text) {
        List<Double> results = new ArrayList<>();
        Matcher matcher = NUM_WITH_UNIT.matcher(text.toLowerCase());
        while (matcher.find()) {
            double value = Double.parseDouble(matcher.group("value"));
            String unit = matcher.group("unit");
            double scale = SCALES.getOrDefault(unit == null ? "元" : unit, 1.0);
            results.add(value * scale);
        }
        return results;
    }
}