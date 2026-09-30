package com.gamking.aegisrag;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.FactType;
import com.gamking.aegisrag.model.VerificationResult;
import com.gamking.aegisrag.retrieval.base.DocumentChunk;
import com.gamking.aegisrag.retrieval.chunker.FinancialTableChunker;
import com.gamking.aegisrag.retrieval.chunker.LegalHierarchyChunker;
import com.gamking.aegisrag.retrieval.chunker.RecursiveChunker;
import com.gamking.aegisrag.retrieval.HybridRetriever;
import com.gamking.aegisrag.retrieval.bm25.Tokenizer;
import com.gamking.aegisrag.verification.BlacklistRiskRule;
import com.gamking.aegisrag.verification.RuleMatrix;
import com.gamking.aegisrag.verification.UnitConversionRule;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 检索模块与通用事实质检引擎测试。
 */
public class RetrievalAndVerificationTest {

    // ---------------------------------------------------------------------------
    // 分块器
    // ---------------------------------------------------------------------------

    @Test
    void recursiveChunkerProducesChunks() {
        List<DocumentChunk> chunks = new RecursiveChunker(300, 50)
                .splitText("第一段。\n\n第二段。", "doc_001");
        assertFalse(chunks.isEmpty());
        assertEquals("doc_001_c0", chunks.get(0).chunkId());
        assertTrue(chunks.get(0).content().contains("第一段"));
    }

    @Test
    void legalChunkerPreservesParentTitle() {
        String text = "第一章 总则\n第一条 为规范市场行为，制定本法。\n第二条 应当遵守本法。\n第二章 市场准入\n第三条 应依法登记。";
        List<DocumentChunk> chunks = new LegalHierarchyChunker().splitText(text, "law_001");
        assertEquals(2, chunks.size());
        assertEquals("第一章 总则", chunks.get(0).metadata().get("section"));
        assertEquals("legal_clause", chunks.get(0).metadata().get("kind"));
        assertTrue(chunks.get(0).content().contains("第一条"));
    }

    @Test
    void financialChunkerTransformsRows() {
        String table = "营业收入,10000万元,15%\n营业成本,6000万元,10%";
        List<DocumentChunk> chunks = new FinancialTableChunker().splitText(table, "fin_2024");
        assertEquals(2, chunks.size());
        assertTrue(chunks.get(0).content().contains("营业收入"));
        assertTrue(chunks.get(0).content().contains("10000万元"));
        assertEquals("financial_table_row", chunks.get(0).metadata().get("kind"));
    }

    // ---------------------------------------------------------------------------
    // 检索器
    // ---------------------------------------------------------------------------

    @Test
    void bm25RanksRelevantDocFirst() {
        DocumentChunk d0 = new DocumentChunk("d0", "高血压门诊慢特病保障范围");
        DocumentChunk d1 = new DocumentChunk("d1", "城乡居民医保住院报销比例");
        HybridRetriever retriever = new HybridRetriever();
        retriever.indexDocuments(List.of(d0, d1));

        var results = retriever.retrieve("高血压门诊", 2);
        assertFalse(results.isEmpty());
        assertEquals("d0", results.get(0).chunk().chunkId());
    }

    @Test
    void hybridRetrieverAcceptsTokenizer() {
        Tokenizer tokenizer = text -> text.replace("奥沙利铂", "【奥沙利铂】");
        HybridRetriever retriever = new HybridRetriever(tokenizer);

        DocumentChunk d0 = new DocumentChunk("d0", "使用奥沙利铂治疗");
        DocumentChunk d1 = new DocumentChunk("d1", "其他内容");
        retriever.indexDocuments(List.of(d0, d1));

        var results = retriever.retrieve("奥沙利铂", 2);
        assertFalse(results.isEmpty());
        assertEquals("d0", results.get(0).chunk().chunkId());
    }

    // ---------------------------------------------------------------------------
    // 验证规则
    // ---------------------------------------------------------------------------

    @Test
    void unitConversionRuleMapsEquivalentScales() {
        RuleMatrix matrix = new RuleMatrix();
        matrix.register(new UnitConversionRule());

        AtomicFact fact = new AtomicFact("营收", "为", "0.5亿元", FactType.NUMERIC);
        VerificationResult result = matrix.verify(fact, "公司营收达到5000万元");
        assertTrue(result.passed());
    }

    @Test
    void unitConversionRuleRejectsMismatch() {
        UnitConversionRule rule = new UnitConversionRule();
        AtomicFact fact = new AtomicFact("营收", "为", "1亿元", FactType.NUMERIC);
        VerificationResult result = rule.verify(fact, "公司营收达到5000万元");
        assertFalse(result.passed());
    }

    @Test
    void blacklistRuleRejectsRiskWord() {
        BlacklistRiskRule rule = new BlacklistRiskRule();
        AtomicFact fact = new AtomicFact("疗法", "效果", "可以根治", FactType.CATEGORICAL);
        VerificationResult result = rule.verify(fact, "该疗法有效率为80%");
        assertFalse(result.passed());
        assertEquals("RISK_WORD_IN_ANSWER", result.findings().get(0).code());
    }

    @Test
    void ruleMatrixRegistersExtraRules() {
        RuleMatrix matrix = new RuleMatrix();
        matrix.register(new BlacklistRiskRule(java.util.Set.of("根治")));

        AtomicFact fact = new AtomicFact("疗法", "效果", "可以根治", FactType.CATEGORICAL);
        VerificationResult result = matrix.verify(fact, "该疗法有效率为80%");
        assertFalse(result.passed());
    }
}