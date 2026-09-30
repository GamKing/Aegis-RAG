package com.gamking.aegisrag.retrieval.chunker;

import com.gamking.aegisrag.retrieval.base.BaseChunker;
import com.gamking.aegisrag.retrieval.base.DocumentChunk;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 法律层级分块器。
 * 
 * 按"第X条"作为切割边界，强制保留父标题，确保法律文本的层级结构完整。
 * 适用于法律法规、合同条款等具有明确章节结构的文本。
 */
public class LegalHierarchyChunker implements BaseChunker {
    
    private static final Pattern ARTICLE_PATTERN = Pattern.compile("第[一二三四五六七八九十百千0-9]+条");
    private static final Pattern TITLE_PATTERN = Pattern.compile("^(第[一二三四五六七八九十百0-9]+[章节编]|[一二三四五六七八九十]+、)");
    
    @Override
    public List<DocumentChunk> splitText(String text, String docId) {
        String[] lines = text.split("\n");
        String currentTitle = "";
        List<Section> sections = new ArrayList<>();
        StringBuilder bufLines = new StringBuilder();
        
        for (String line : lines) {
            String stripped = line.trim();
            if (stripped.isEmpty()) continue;
            
            Matcher titleMatcher = TITLE_PATTERN.matcher(stripped);
            Matcher articleMatcher = ARTICLE_PATTERN.matcher(stripped);
            
            if (titleMatcher.find() && !articleMatcher.find()) {
                // 新的章节标题
                if (bufLines.length() > 0) {
                    sections.add(new Section(currentTitle, bufLines.toString().trim()));
                    bufLines = new StringBuilder();
                }
                currentTitle = stripped;
            } else {
                if (bufLines.length() > 0) {
                    bufLines.append("\n");
                }
                bufLines.append(stripped);
            }
        }
        
        if (bufLines.length() > 0) {
            sections.add(new Section(currentTitle, bufLines.toString().trim()));
        }
        
        List<DocumentChunk> chunks = new ArrayList<>();
        for (int idx = 0; idx < sections.size(); idx++) {
            Section section = sections.get(idx);
            if (section.content.isEmpty()) continue;
            
            String content = section.title.isEmpty() 
                    ? section.content 
                    : section.title + "\n" + section.content;
            
            chunks.add(DocumentChunk.builder()
                    .chunkId(docId + "_l" + idx)
                    .content(content)
                    .metadata("index", String.valueOf(idx))
                    .metadata("doc_id", docId)
                    .metadata("section", section.title)
                    .metadata("kind", "legal_clause")
                    .build());
        }
        
        return chunks;
    }
    
    private record Section(String title, String content) {}
}
