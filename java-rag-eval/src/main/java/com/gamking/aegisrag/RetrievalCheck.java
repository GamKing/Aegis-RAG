package com.gamking.aegisrag;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.FactType;
import com.gamking.aegisrag.retrieval.HybridRetriever;
import com.gamking.aegisrag.retrieval.base.DocumentChunk;
import com.gamking.aegisrag.retrieval.chunker.FinancialTableChunker;
import com.gamking.aegisrag.retrieval.chunker.LegalHierarchyChunker;
import com.gamking.aegisrag.retrieval.chunker.RecursiveChunker;
import com.gamking.aegisrag.verification.BlacklistRiskRule;
import com.gamking.aegisrag.verification.RuleMatrix;
import com.gamking.aegisrag.verification.UnitConversionRule;

import java.util.List;

/**
 * 检索模块与通用事实质检引擎的独立运行时验证（不依赖 JUnit）。
 */
public class RetrievalCheck {

    private static int passed = 0;
    private static int failed = 0;

    private static void check(String name, boolean condition) {
        if (condition) {
            passed++;
            System.out.println("[PASS] " + name);
        } else {
            failed++;
            System.out.println("[FAIL] " + name);
        }
    }

    public static void main(String[] args) {
        testChunkers();
        testRetrieval();
        testRules();
        System.out.println("\n==== " + passed + " passed, " + failed + " failed ====");
        if (failed > 0) System.exit(1);
    }

    private static void testChunkers() {
        List<DocumentChunk> recursive = new RecursiveChunker(300, 50)
                .splitText("第一段。\n\n第二段。", "doc_001");
        check("RecursiveChunker 分块", !recursive.isEmpty()
                && recursive.get(0).chunkId().equals("doc_001_c0"));

        String legal = "第一章 总则\n第一条 为规范市场行为，制定本法。\n第二条 应当遵守本法。\n第二章 市场准入\n第三条 应依法登记。";
        List<DocumentChunk> legalChunks = new LegalHierarchyChunker().splitText(legal, "law_001");
        check("LegalChunker 保留父标题", legalChunks.size() == 2
                && legalChunks.get(0).metadata().get("section").equals("第一章 总则")
                && legalChunks.get(0).content().contains("第一条"));

        String table = "营业收入,10000万元,15%\n营业成本,6000万元,10%";
        List<DocumentChunk> finChunks = new FinancialTableChunker().splitText(table, "fin_2024");
        check("FinancialChunker 转写行", finChunks.size() == 2
                && finChunks.get(0).content().contains("营业收入")
                && finChunks.get(0).content().contains("10000万元"));
    }

    private static void testRetrieval() {
        HybridRetriever retriever = new HybridRetriever();
        retriever.indexDocuments(List.of(
                new DocumentChunk("d0", "高血压门诊慢特病保障范围"),
                new DocumentChunk("d1", "城乡居民医保住院报销比例")));
        var results = retriever.retrieve("高血压门诊", 2);
        check("BM25 相关文档排第一", !results.isEmpty()
                && results.get(0).chunk().chunkId().equals("d0"));
    }

    private static void testRules() {
        // 量纲换算
        RuleMatrix matrix = new RuleMatrix();
        matrix.register(new UnitConversionRule());
        AtomicFact equiv = new AtomicFact("营收", "为", "0.5亿元", FactType.NUMERIC);
        check("量纲换算等价通过", matrix.verify(equiv, "公司营收达到5000万元").passed());

        AtomicFact mismatch = new AtomicFact("营收", "为", "1亿元", FactType.NUMERIC);
        check("量纲换算不匹配拒绝", !new UnitConversionRule().verify(mismatch, "公司营收达到5000万元").passed());

        // 黑名单
        AtomicFact risk = new AtomicFact("疗法", "效果", "可以根治", FactType.CATEGORICAL);
        check("黑名单禁忌拒绝", !new BlacklistRiskRule().verify(risk, "该疗法有效率为80%").passed());
    }
}