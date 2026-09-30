package com.gamking.aegisrag.retrieval.chunker;

import com.gamking.aegisrag.retrieval.base.BaseChunker;
import com.gamking.aegisrag.retrieval.base.DocumentChunk;

import java.util.ArrayList;
import java.util.List;

/**
 * 带重叠窗口的递归分块器。
 * 
 * 优先按段落（双换行）切分，支持重叠窗口，保证跨块的政策条件与例外条款
 * 能完整出现在某一块中。
 */
public class RecursiveChunker implements BaseChunker {
    
    private final int chunkSize;
    private final int chunkOverlap;
    
    public RecursiveChunker() {
        this(400, 60);
    }
    
    public RecursiveChunker(int chunkSize, int chunkOverlap) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("chunkSize must be positive");
        }
        if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
            throw new IllegalArgumentException("chunkOverlap must be in [0, chunkSize)");
        }
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
    }
    
    @Override
    public List<DocumentChunk> splitText(String text, String docId) {
        List<String> chunks = splitIntoChunks(text);
        
        // 构建重叠窗口
        if (chunkOverlap > 0) {
            List<String> windowed = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                String c = chunks.get(i);
                if (i > 0) {
                    String overlapText = chunks.get(i - 1);
                    if (overlapText.length() > chunkOverlap) {
                        overlapText = overlapText.substring(overlapText.length() - chunkOverlap);
                    }
                    c = overlapText + c;
                }
                windowed.add(c);
            }
            chunks = windowed;
        }
        
        List<DocumentChunk> result = new ArrayList<>();
        for (int idx = 0; idx < chunks.size(); idx++) {
            String chunkText = chunks.get(idx);
            result.add(DocumentChunk.builder()
                    .chunkId(docId + "_c" + idx)
                    .content(chunkText)
                    .metadata("index", String.valueOf(idx))
                    .metadata("doc_id", docId)
                    .build());
        }
        return result;
    }
    
    private List<String> splitIntoChunks(String text) {
        String[] paragraphs = text.split("\n\n");
        List<String> chunks = new ArrayList<>();
        StringBuilder currentChunk = new StringBuilder();
        
        for (String para : paragraphs) {
            para = para.trim();
            if (para.isEmpty()) continue;
            
            if (currentChunk.length() + para.length() <= chunkSize) {
                if (currentChunk.length() > 0) {
                    currentChunk.append("\n\n");
                }
                currentChunk.append(para);
            } else {
                if (currentChunk.length() > 0) {
                    chunks.add(currentChunk.toString());
                }
                if (para.length() > chunkSize) {
                    chunks.addAll(splitBySentences(para));
                    currentChunk = new StringBuilder();
                } else {
                    currentChunk = new StringBuilder(para);
                }
            }
        }
        
        if (currentChunk.length() > 0) {
            chunks.add(currentChunk.toString());
        }
        
        return chunks;
    }
    
    private List<String> splitBySentences(String text) {
        String[] sentences = text.split("(?<=[。；\n])");
        List<String> result = new ArrayList<>();
        StringBuilder buf = new StringBuilder();
        
        for (String sentence : sentences) {
            buf.append(sentence);
            if (buf.length() >= chunkSize) {
                result.add(buf.toString());
                buf = new StringBuilder();
            }
        }
        
        if (buf.length() > 0) {
            result.add(buf.toString());
        }
        
        return result;
    }
}
