package com.gamking.aegisrag.retrieval.chunker;

import com.gamking.aegisrag.retrieval.base.BaseChunker;
import com.gamking.aegisrag.retrieval.base.DocumentChunk;

import java.util.ArrayList;
import java.util.List;

/**
 * 财报表格分块器。
 * 
 * 将财务报表逐行转为独立的语义描述，适用于财务报表、数据表格等结构化数据。
 * 每行转为 "指标名为数值" 的格式，便于检索和验证。
 */
public class FinancialTableChunker implements BaseChunker {
    
    @Override
    public List<DocumentChunk> splitText(String text, String docId) {
        String[] rows = text.split("\n");
        List<DocumentChunk> chunks = new ArrayList<>();
        
        for (int idx = 0; idx < rows.length; idx++) {
            String row = rows[idx].trim();
            if (row.isEmpty()) continue;
            
            // 按逗号或制表符分割单元格
            String[] cells = row.split("[,，\t]");
            StringBuilder content = new StringBuilder();
            
            if (cells.length >= 2) {
                String subject = cells[0].trim();
                StringBuilder rest = new StringBuilder();
                for (int i = 1; i < cells.length; i++) {
                    if (i > 1) rest.append("，");
                    rest.append(cells[i].trim());
                }
                content.append(subject).append("为").append(rest);
            } else {
                content.append(row);
            }
            
            chunks.add(DocumentChunk.builder()
                    .chunkId(docId + "_f" + idx)
                    .content(content.toString())
                    .metadata("index", String.valueOf(idx))
                    .metadata("doc_id", docId)
                    .metadata("row", String.valueOf(idx))
                    .metadata("kind", "financial_table_row")
                    .build());
        }
        
        return chunks;
    }
}
