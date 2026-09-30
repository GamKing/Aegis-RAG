package com.gamking.aegisrag.retrieval.base;

import java.util.List;

/**
 * 检索器抽象基类。
 */
public interface BaseRetriever {
    
    /**
     * 执行检索，返回最相关的文档块。
     * 
     * @param query 查询文本
     * @param topK 返回结果数量
     * @return 检索结果列表，按相关性降序排列
     */
    List<SearchResult> retrieve(String query, int topK);
    
    /**
     * 执行检索，返回最相关的文档块（默认返回 5 个）。
     * 
     * @param query 查询文本
     * @return 检索结果列表，按相关性降序排列
     */
    default List<SearchResult> retrieve(String query) {
        return retrieve(query, 5);
    }
    
    /**
     * 执行检索并返回合并后的上下文文本。
     * 
     * @param query 查询文本
     * @param topK 返回结果数量
     * @return 合并后的上下文文本
     */
    default String retrieveContext(String query, int topK) {
        List<SearchResult> results = retrieve(query, topK);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < results.size(); i++) {
            if (i > 0) {
                sb.append("\n\n");
            }
            sb.append(results.get(i).chunk().content());
        }
        return sb.toString();
    }
}
