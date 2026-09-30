package com.gamking.aegisrag.retrieval.bm25;

/**
 * 行业专有分词器接口。
 * 
 * 行业插槽 2：各行业通过实现本接口注入行业词表，确保专有名词
 * （如"奥沙利铂"、"索拉非尼"）作为不可分割的整体参与检索。
 */
public interface Tokenizer {
    
    /**
     * 对文本做行业专有名词保护，返回受保护的文本。
     * 
     * 实现应当将专有名词用特殊标记包裹（如【】【】），使下游的 bigram
     * 分词能够保留整体语义。
     */
    String protect(String text);
}