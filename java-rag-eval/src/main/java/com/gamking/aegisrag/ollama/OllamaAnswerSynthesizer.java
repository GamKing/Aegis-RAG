package com.gamking.aegisrag.ollama;

import com.gamking.aegisrag.model.ExtractedFact;
import com.gamking.aegisrag.model.ExtractionPayload;
import com.gamking.aegisrag.pipeline.AnswerSynthesizer;

import java.util.List;

/**
 * Ollama 受控润色器。
 * 
 * 注意：只接收已经通过 Tier 1 的结构化事实，不接收原始长上下文，
 * 以降低最终生成阶段重新引入新事实的风险。
 */
public class OllamaAnswerSynthesizer implements AnswerSynthesizer {

    private final OllamaClient client;

    public OllamaAnswerSynthesizer(OllamaClient client) {
        this.client = client;
    }

    @Override
    public String synthesize(String query, ExtractionPayload payload) {
        try {
            String factsJson = buildFactsJson(payload.facts());
            String prompt = """
                    你是一个受控答案润色器。
                    只根据下面已经通过代码核验的事实回答问题。
                    不得增加任何新数字、新日期、新实体或新条件。
                    如果事实不足，请明确说材料不足，不要猜测。
                    只输出一句简洁、准确的中文回答。

                    【问题】
                    %s

                    【已核验事实 JSON】
                    %s
                    """.formatted(query, factsJson);
            return client.chat(prompt, 0.0, 300).trim();
        } catch (Exception e) {
            // Ollama 失败时不自由生成，返回空字符串交给上层安全处理。
            return "";
        }
    }

    private String buildFactsJson(List<ExtractedFact> facts) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < facts.size(); i++) {
            if (i > 0) sb.append(",");
            ExtractedFact fact = facts.get(i);
            sb.append("{\"key\":\"").append(escapeJson(fact.key())).append("\",")
              .append("\"value\":\"").append(escapeJson(fact.value())).append("\",")
              .append("\"sourceText\":\"").append(escapeJson(fact.sourceText())).append("\",")
              .append("\"isNumeric\":").append(fact.isNumeric()).append("}");
        }
        sb.append("]");
        return sb.toString();
    }

    private String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
