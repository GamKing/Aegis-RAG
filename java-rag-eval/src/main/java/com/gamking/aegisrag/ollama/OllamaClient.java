package com.gamking.aegisrag.ollama;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * 本地 Ollama 统一客户端。
 * 
 * 通过 HTTP 调用 Ollama 的 /api/chat 接口，不依赖任何第三方 SDK。
 * 默认地址 localhost:11434，可用环境变量 OLLAMA_BASE_URL / OLLAMA_MODEL 覆盖。
 */
public class OllamaClient {

    public static final String DEFAULT_BASE_URL = "http://localhost:11434";
    public static final String DEFAULT_MODEL = "Kimi2.5:1.5b";

    private final HttpClient httpClient;
    private final String baseUrl;
    private final String model;
    private final Duration timeout;

    public OllamaClient() {
        this(readEnv("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
             readEnv("OLLAMA_MODEL", DEFAULT_MODEL));
    }

    public OllamaClient(String baseUrl, String model) {
        this(baseUrl, model, Duration.ofSeconds(60));
    }

    public OllamaClient(String baseUrl, String model, Duration timeout) {
        this.baseUrl = normalizeBaseUrl(baseUrl);
        this.model = model == null || model.isBlank() ? DEFAULT_MODEL : model;
        this.timeout = timeout;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(timeout)
                .build();
    }

    public String getModel() {
        return model;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    /**
     * 发送一次聊天请求并返回模型的纯文本回复。
     *
     * @param prompt      用户提示词
     * @param temperature 采样温度
     * @param numPredict  最大生成 token 数
     * @return 模型生成的纯文本
     */
    public String chat(String prompt, double temperature, int numPredict) {
        try {
            String payload = buildPayload(prompt, temperature, numPredict);
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/api/chat"))
                    .timeout(timeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(payload, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new IllegalStateException(
                        "Ollama 请求失败: HTTP " + response.statusCode() + " " + response.body());
            }

            return extractMessage(response.body());
        } catch (java.net.ConnectException e) {
            throw new IllegalStateException(
                    "无法连接 Ollama (" + baseUrl + ")。请确认 Ollama 已启动，并已拉取模型 " + model, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Ollama 请求被中断", e);
        } catch (java.io.IOException e) {
            throw new IllegalStateException("Ollama 请求 I/O 异常: " + e.getMessage(), e);
        }
    }

    private String buildPayload(String prompt, double temperature, int numPredict) {
        // 使用 /api/chat 的 messages 结构
        String content = escapeJson(prompt);
        return "{\"model\":\"" + escapeJson(model) + "\","
                + "\"stream\":false,"
                + "\"messages\":[{\"role\":\"user\",\"content\":\"" + content + "\"}],"
                + "\"options\":{\"temperature\":" + temperature + ",\"num_predict\":" + numPredict + "}}";
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private String extractMessage(String responseBody) {
        // 简单解析 JSON 响应，提取 message.content 字段
        // 响应格式: {"message":{"content":"..."},"done":true,...}
        int contentIdx = responseBody.indexOf("\"content\":\"");
        if (contentIdx == -1) {
            return "";
        }
        int start = contentIdx + "\"content\":\"".length();
        int end = start;
        while (end < responseBody.length()) {
            char c = responseBody.charAt(end);
            if (c == '"' && responseBody.charAt(end - 1) != '\\') {
                break;
            }
            end++;
        }
        String raw = responseBody.substring(start, end);
        return unescapeJson(raw).trim();
    }

    private static String unescapeJson(String s) {
        return s.replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t");
    }

    private static String normalizeBaseUrl(String url) {
        if (url == null || url.isBlank()) {
            return DEFAULT_BASE_URL;
        }
        String trimmed = url.trim();
        if (trimmed.endsWith("/")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }

    private static String readEnv(String key, String defaultValue) {
        String value = System.getenv(key);
        return (value == null || value.isBlank()) ? defaultValue : value;
    }
}