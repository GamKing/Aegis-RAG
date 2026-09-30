"""行业插件插槽测试。"""
import unittest
from rag_eval.models import AtomicFact, FactType
from rag_eval.retrieval import (
    BaseChunker,
    DocumentChunk,
    FinancialTableChunker,
    HybridRetriever,
    LegalHierarchyChunker,
    RecursiveChunker,
)
from rag_eval.verification_rules import (
    BlacklistRiskRule,
    RuleMatrix,
    UnitConversionRule,
)


class TestBaseChunker(unittest.TestCase):
    """插槽 1：分块器抽象基类测试。"""

    def test_recursive_chunker_is_base_chunker(self):
        chunker = RecursiveChunker(chunk_size=100)
        self.assertIsInstance(chunker, BaseChunker)

    def test_legal_chunker_is_base_chunker(self):
        chunker = LegalHierarchyChunker()
        self.assertIsInstance(chunker, BaseChunker)

    def test_financial_chunker_is_base_chunker(self):
        chunker = FinancialTableChunker()
        self.assertIsInstance(chunker, BaseChunker)


class TestLegalHierarchyChunker(unittest.TestCase):
    """法律层级分块器测试。"""

    def test_split_by_articles(self):
        text = """第一章 总则
第一条 为了规范市场行为，根据宪法，制定本法。
第二条 在中华人民共和国境内从事市场活动，应当遵守本法。
第二章 市场准入
第三条 市场主体应当依法登记。"""
        chunker = LegalHierarchyChunker()
        chunks = chunker.split_text(text, doc_id="law_001")

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, "law_001_l0")
        self.assertIn("总则", chunks[0].content)
        self.assertIn("第一条", chunks[0].content)
        self.assertEqual(chunks[0].metadata["section"], "第一章 总则")
        self.assertEqual(chunks[0].metadata["kind"], "legal_clause")

    def test_preserves_parent_title(self):
        text = """第三章 合同
第十条 合同应当采用书面形式。
第十一条 合同自签订之日起生效。"""
        chunker = LegalHierarchyChunker()
        chunks = chunker.split_text(text, doc_id="contract_law")

        self.assertEqual(len(chunks), 1)
        self.assertIn("第三章", chunks[0].content)
        self.assertIn("第十条", chunks[0].content)
        self.assertEqual(chunks[0].metadata["section"], "第三章 合同")


class TestFinancialTableChunker(unittest.TestCase):
    """财报表格分块器测试。"""

    def test_split_table_rows(self):
        text = """营业收入,10000万元,15%
营业成本,6000万元,10%
净利润,3000万元,20%"""
        chunker = FinancialTableChunker()
        chunks = chunker.split_text(text, doc_id="finance_2024")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_id, "finance_2024_f0")
        self.assertIn("营业收入", chunks[0].content)
        self.assertIn("10000万元", chunks[0].content)
        self.assertEqual(chunks[0].metadata["kind"], "financial_table_row")
        self.assertEqual(chunks[0].metadata["row"], 0)

    def test_handles_single_column(self):
        text = """资产负债表
流动资产
固定资产"""
        chunker = FinancialTableChunker()
        chunks = chunker.split_text(text, doc_id="balance_sheet")

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].content, "资产负债表")


class TestTokenizerInjection(unittest.TestCase):
    """插槽 2：分词器注入测试。"""

    def test_hybrid_retriever_accepts_tokenizer(self):
        class MedicalTokenizer:
            def protect(self, text: str) -> str:
                # 模拟医学专有名词保护
                return text.replace("奥沙利铂", "【奥沙利铂】")

        tokenizer = MedicalTokenizer()
        retriever = HybridRetriever(tokenizer=tokenizer)
        self.assertIsNotNone(retriever.bm25.tokenizer)

    def test_tokenizer_protects_terms(self):
        class SimpleTokenizer:
            def protect(self, text: str) -> str:
                # 简单保护：在专有名词前后加标记
                return text.replace("心衰", "【心衰】")

        docs = [
            DocumentChunk("d0", "患者出现心衰症状"),
            DocumentChunk("d1", "心功能不全的治疗方案"),
        ]
        tokenizer = SimpleTokenizer()
        retriever = HybridRetriever(tokenizer=tokenizer)
        retriever.index_documents(docs)

        results = retriever.retrieve("心衰", top_k=2)
        self.assertGreater(len(results), 0)


class TestIndustryRules(unittest.TestCase):
    """插槽 3：行业规则插件测试。"""

    def test_unit_conversion_rule_pass(self):
        rule = UnitConversionRule()
        fact = AtomicFact(
            subject="营收",
            predicate="为",
            object_value="0.5亿元",
            fact_type=FactType.NUMERIC,
        )
        context = "公司营收达到5000万元"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_unit_conversion_rule_fail(self):
        rule = UnitConversionRule()
        fact = AtomicFact(
            subject="营收",
            predicate="为",
            object_value="1亿元",
            fact_type=FactType.NUMERIC,
        )
        context = "公司营收达到5000万元"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)

    def test_unit_conversion_with_english_units(self):
        rule = UnitConversionRule()
        fact = AtomicFact(
            subject="revenue",
            predicate="is",
            object_value="108.0 billion",
            fact_type=FactType.NUMERIC,
        )
        context = "Revenue reached 108000000000 dollars"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_blacklist_rule_pass(self):
        rule = BlacklistRiskRule(blacklist=("绝对安全", "根治"))
        fact = AtomicFact(
            subject="药物",
            predicate="疗效",
            object_value="有效率为90%",
            fact_type=FactType.NUMERIC,
        )
        context = "临床试验显示药物有效率为90%"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_blacklist_rule_fail(self):
        rule = BlacklistRiskRule(blacklist=("绝对安全", "根治"))
        fact = AtomicFact(
            subject="药物",
            predicate="疗效",
            object_value="绝对安全",
            fact_type=FactType.CATEGORICAL,
        )
        context = "临床试验显示药物有效率为90%"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)

    def test_rule_matrix_register_extra_rules(self):
        matrix = RuleMatrix()
        blacklist_rule = BlacklistRiskRule(blacklist=("根治",))
        matrix.register(blacklist_rule)

        fact = AtomicFact(
            subject="疗法",
            predicate="效果",
            object_value="可以根治",
            fact_type=FactType.CATEGORICAL,
        )
        context = "该疗法有效率为80%"
        result = matrix.verify(fact, context)
        self.assertFalse(result.passed)
        # 检查最后一个 finding（额外规则在基础规则之后执行）
        self.assertEqual(result.findings[-1].code, "RISK_WORD_IN_ANSWER")


class TestCitationMetadata(unittest.TestCase):
    """插槽 4：溯源元数据测试。"""

    def test_chunk_metadata_supports_citation(self):
        chunk = DocumentChunk(
            chunk_id="doc_001_c0",
            content="合同条款内容",
            metadata={
                "index": 0,
                "doc_id": "doc_001",
                "section": "第三章 合同",
                "kind": "legal_clause",
                "page": 15,
                "line_start": 120,
                "line_end": 125,
            },
        )
        self.assertEqual(chunk.metadata["page"], 15)
        self.assertEqual(chunk.metadata["line_start"], 120)
        self.assertEqual(chunk.metadata["line_end"], 125)


if __name__ == "__main__":
    unittest.main()
