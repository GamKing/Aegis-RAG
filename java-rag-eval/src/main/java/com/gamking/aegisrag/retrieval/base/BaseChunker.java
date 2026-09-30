package com.gamking.aegisrag.retrieval.base;

import java.util.List;

/**
 * 分块器抽象基类。
 * 
 * 行业插槽 1：各行业通过继承并实现 splitText，在不改动主链路的前提下
 * 提供"行业感知解析"。例如：
 * - 法律：LegalHierarchyChunker 按"第X条"切分并保留父标题；
 * - 金融财报：FinancialTableChunker 将财务报表逐行转成语义描述。
 */
public interface BaseChunker {
    
    /**
     * 将一段文本切分为若干文档块。
     * 
     * @param text 待切分的文本
     * @param docId 文档标识
     * @return 文档块列表
     */
    List<DocumentChunk> splitText(String text, String docId);
}
