package com.gamking.aegisrag.retrieval;

import com.gamking.aegisrag.retrieval.base.DocumentChunk;
import com.gamking.aegisrag.retrieval.base.SearchResult;
import com.gamking.aegisrag.retrieval.base.BaseRetriever;
import com.gamking.aegisrag.retrieval.bm25.SimpleBM25;
import com.gamking.aegisrag.retrieval.bm25.Tokenizer;

import java.util.List;
import java.util.*;

/**
 * 混合检索器：BM25 词面匹配 + 向量相似度（可插拔）+ RRF 融合。
 * 
 * 未配置向量引擎时自动退化为纯 BM25，零外部依赖直接跑通。
 * 支持注入行业专有分词器以保护专有名词。
 */
public class HybridRetriever implements BaseRetriever {
    
    private final SimpleBM25 bm25;
    private final VectorEngine vectorEngine;
    private final int rrfK;
    private List<DocumentChunk> documents = new ArrayList<>();
    
    public HybridRetriever() {
        this(null, 60, null);
    }
    
    public HybridRetriever(Tokenizer tokenizer) {
        this(null, 60, tokenizer);
    }
    
    public HybridRetriever(VectorEngine vectorEngine, int rrfK) {
        this(vectorEngine, rrfK, null);
    }
    
    public HybridRetriever(VectorEngine vectorEngine, int rrfK, Tokenizer tokenizer) {
        this.vectorEngine = vectorEngine;
        this.rrfK = rrfK;
        this.bm25 = tokenizer == null ? new SimpleBM25() : new SimpleBM25(1.5, 0.75, tokenizer);
    }
    
    /**
     * 建立索引。
     */
    public void indexDocuments(List<DocumentChunk> documents) {
        this.documents = new ArrayList<>(documents);
        bm25.fit(documents);
        if (vectorEngine != null) {
            vectorEngine.indexDocuments(documents);
        }
    }
    
    @Override
    public List<SearchResult> retrieve(String query, int topK) {
        List<SearchResult> bm25Results = bm25.search(query, topK * 2);
        
        // 未配置向量引擎，直接退化为纯 BM25
        if (vectorEngine == null) {
            return bm25Results.size() > topK ? bm25Results.subList(0, topK) : bm25Results;
        }
        
        List<SearchResult> vectorResults = vectorEngine.search(query, topK * 2);
        
        // RRF (Reciprocal Rank Fusion) 排名倒数加权融合
        Map<String, Double> rrfScores = new HashMap<>();
        Map<String, DocumentChunk> chunkMap = new HashMap<>();
        
        for (int rank = 0; rank < bm25Results.size(); rank++) {
            SearchResult res = bm25Results.get(rank);
            String cid = res.chunk().chunkId();
            chunkMap.put(cid, res.chunk());
            rrfScores.put(cid, rrfScores.getOrDefault(cid, 0.0) + 1.0 / (rrfK + rank + 1));
        }
        
        for (int rank = 0; rank < vectorResults.size(); rank++) {
            SearchResult res = vectorResults.get(rank);
            String cid = res.chunk().chunkId();
            chunkMap.put(cid, res.chunk());
            rrfScores.put(cid, rrfScores.getOrDefault(cid, 0.0) + 1.0 / (rrfK + rank + 1));
        }
        
        // 按 RRF 分数降序排列
        List<Map.Entry<String, Double>> sorted = new ArrayList<>(rrfScores.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        
        List<SearchResult> results = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, sorted.size()); i++) {
            String cid = sorted.get(i).getKey();
            results.add(new SearchResult(chunkMap.get(cid), sorted.get(i).getValue()));
        }
        return results;
    }
}