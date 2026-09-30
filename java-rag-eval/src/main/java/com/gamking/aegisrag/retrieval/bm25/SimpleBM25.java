package com.gamking.aegisrag.retrieval.bm25;

import com.gamking.aegisrag.retrieval.base.DocumentChunk;
import com.gamking.aegisrag.retrieval.base.SearchResult;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 轻量级纯 Java BM25 检索器，无外部依赖，可离线运行。
 *
 * 中文二元分词 (CJK bigram) + 英文/数字 token。支持注入行业专有分词器。
 */
public final class SimpleBM25 {

    private final double k1;
    private final double b;
    private final Tokenizer tokenizer;

    private List<DocumentChunk> corpus = new ArrayList<>();
    private List<Integer> docLengths = new ArrayList<>();
    private double avgDocLen = 0.0;
    private Map<String, Integer> docFreqs = new HashMap<>();
    private Map<String, Double> idf = new HashMap<>();

    private static final Pattern ASCII_TOKEN = Pattern.compile("[a-zA-Z0-9_]+");
    private static final Pattern CJK_CHARS = Pattern.compile("[\\u4e00-\\u9fff]");

    public SimpleBM25() {
        this(1.5, 0.75, null);
    }

    public SimpleBM25(double k1, double b) {
        this(k1, b, null);
    }

    public SimpleBM25(double k1, double b, Tokenizer tokenizer) {
        this.k1 = k1;
        this.b = b;
        this.tokenizer = tokenizer;
    }

    /**
     * 建立索引。
     */
    public void fit(List<DocumentChunk> documents) {
        corpus = new ArrayList<>(documents);
        docLengths = new ArrayList<>();
        docFreqs = new HashMap<>();
        idf = new HashMap<>();

        int totalLen = 0;

        for (DocumentChunk doc : corpus) {
            Set<String> tokens = new HashSet<>(tokenize(doc.content()));
            for (String t : tokens) {
                docFreqs.put(t, docFreqs.getOrDefault(t, 0) + 1);
            }
            int length = tokens.size();
            docLengths.add(length);
            totalLen += length;
        }

        int nDocs = corpus.size();
        avgDocLen = nDocs > 0 ? (double) totalLen / nDocs : 0.0;

        for (Map.Entry<String, Integer> entry : docFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            idf.put(word, Math.log((nDocs - freq + 0.5) / (freq + 0.5) + 1.0));
        }
    }

    /**
     * 执行检索。
     */
    public List<SearchResult> search(String query, int topK) {
        List<String> qTokens = tokenize(query);
        List<Double> scores = new ArrayList<>();

        for (int idx = 0; idx < corpus.size(); idx++) {
            DocumentChunk doc = corpus.get(idx);
            Map<String, Integer> tf = new HashMap<>();
            for (String t : tokenize(doc.content())) {
                tf.put(t, tf.getOrDefault(t, 0) + 1);
            }

            int docLen = docLengths.get(idx);
            double score = 0.0;

            for (String t : qTokens) {
                if (!tf.containsKey(t)) continue;
                int termTf = tf.get(t);
                double idfVal = idf.getOrDefault(t, 0.0);
                double docLenRatio = avgDocLen > 0 ? docLen / avgDocLen : 1.0;
                double denom = termTf + k1 * (1.0 - b + b * docLenRatio);
                score += idfVal * (termTf * (k1 + 1.0)) / (denom > 0 ? denom : 1.0);
            }

            scores.add(score);
        }

        // 按分数降序排列
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < scores.size(); i++) indices.add(i);
        indices.sort((a, bIdx) -> Double.compare(scores.get(bIdx), scores.get(a)));

        List<SearchResult> results = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, indices.size()); i++) {
            int idx = indices.get(i);
            if (scores.get(idx) <= 0) continue;
            results.add(new SearchResult(corpus.get(idx), scores.get(idx)));
        }
        return results;
    }

    /**
     * 分词：中文 bigram + 英文/数字 token，支持行业专有词保护。
     */
    private List<String> tokenize(String text) {
        if (tokenizer != null) {
            String protectedText = tokenizer.protect(text);
            return tokenizeDefault(protectedText);
        }
        return tokenizeDefault(text);
    }

    private List<String> tokenizeDefault(String text) {
        List<String> tokens = new ArrayList<>();
        Matcher asciiMatcher = ASCII_TOKEN.matcher(text.toLowerCase());
        while (asciiMatcher.find()) {
            String token = asciiMatcher.group();
            if (!token.isBlank()) {
                tokens.add(token);
            }
        }

        List<Character> chineseChars = new ArrayList<>();
        Matcher cjkMatcher = CJK_CHARS.matcher(text);
        while (cjkMatcher.find()) {
            chineseChars.add(cjkMatcher.group().charAt(0));
        }
        for (int i = 0; i < chineseChars.size() - 1; i++) {
            tokens.add("" + chineseChars.get(i) + chineseChars.get(i + 1));
        }
        return tokens;
    }
}