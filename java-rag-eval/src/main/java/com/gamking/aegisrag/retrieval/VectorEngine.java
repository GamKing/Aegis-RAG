package com.gamking.aegisrag.retrieval;

import com.gamking.aegisrag.retrieval.base.DocumentChunk;
import com.gamking.aegisrag.retrieval.base.SearchResult;
import java.util.List;

/**
 * 向量引擎抽象接口。
 * 
 * 行业插槽：可接入任何向量数据库或嵌入模型实现，如 Milvus、Pinecone 等。
 */
public interface VectorEngine {
    /** 建立向量索引 */
    void indexDocuments(List<DocumentChunk> documents);
    
    /** 向量检索 */
    List<SearchResult> search(String query, int topK);
}