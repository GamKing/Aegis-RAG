package com.gamking.aegisrag.ollama;

import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.pipeline.FactExtractor;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Ollama JSON 抽取器。
 * 
 * 模型只负责把原始上下文压缩成 JSON；JSON 解析成功后仍必须经过 Tier 1。
 */
public class OllamaFactExtractor implements FactExtractor {

    private final OllamaClient client;

    public OllamaFactExtractor(OllamaClient client) {
        this.client = client;
    }

    @Override
    public ExtractionPayload extract(String query, String context) {
        String prompt = """
                你是一个受限事实抽取器。只从参考材料中提取与问题有关的原子事实。
                严禁创造材料中没有的数字、日期、机构或条件。
                只输出合法 JSON，不要 Markdown，不要解释。格式必须是：
                {"facts":[{"key":"字段名","value":"原文值","sourceText":"原文片段","isNumeric":true}],"claims":[]}

                【问题】
                %s

                【参考材料】
                %s
                """.formatted(query, context);

        String raw = client.chat(prompt, 0.0, 800);
        try {
            String cleaned = stripCodeFence(raw);
            List<ExtractedFact> facts = parseFacts(cleaned);
            List<String> claims = parseClaims(cleaned);
            return new ExtractionPayload(facts, claims, raw);
        } catch (Exception e) {
            // 解析失败返回空 payload，由上层安全失败，不允许绕过 Tier 1。
            return new ExtractionPayload(List.of(), List.of(), raw);
        }
    }

    private List<ExtractedFact> parseFacts(String json) {
        List<ExtractedFact> facts = new ArrayList<>();
        // 匹配 {"facts": [...]} 中的每个 fact 对象
        Pattern factPattern = Pattern.compile(
            "\\{[^{}]*\"key\"\\s*:\\s*\"([^\"]+)\"[^{}]*\"value\"\\s*:\\s*\"([^\"]+)\"[^{}]*\"sourceText\"\\s*:\\s*\"([^\"]*)\"[^{}]*\"isNumeric\"\\s*:\\s*(true|false)[^{}]*\\}",
            Pattern.CASE_INSENSITIVE
        );
        Matcher matcher = factPattern.matcher(json);
        while (matcher.find()) {
            String key = matcher.group(1);
            String value = matcher.group(2);
            String sourceText = matcher.group(3);
            boolean isNumeric = Boolean.parseBoolean(matcher.group(4));
            facts.add(new ExtractedFact(key, value, sourceText, isNumeric));
        }
        return facts;
    }

    private List<String> parseClaims(String json) {
        List<String> claims = new ArrayList<>();
        // 匹配 "claims": ["...", "..."]
        Pattern claimsPattern = Pattern.compile("\"claims\"\\s*:\\s*\\[([^\\]]*)\\]");
        Matcher claimsMatcher = claimsPattern.matcher(json);
        if (claimsMatcher.find()) {
            String claimsArray = claimsMatcher.group(1);
            Pattern stringPattern = Pattern.compile("\"([^\"]+)\"");
            Matcher stringMatcher = stringPattern.matcher(claimsArray);
            while (stringMatcher.find()) {
                claims.add(stringMatcher.group(1));
            }
        }
        return claims;
    }

    private static String stripCodeFence(String raw) {
        String text = raw == null ? "" : raw.trim();
        if (text.startsWith("```")) {
            text = text.replaceFirst("^```(?:json)?\\s*", "");
            text = text.replaceFirst("\\s*```$", "");
        }
        return text.trim();
    }
}
