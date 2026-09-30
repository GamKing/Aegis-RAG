"""通用事实校验规则矩阵测试。"""
import unittest
from rag_eval.models import AtomicFact, FactType
from rag_eval.verification_rules import (
    NumericRule,
    RelationRule,
    ScopeRule,
    CategoricalRule,
    RuleMatrix,
)


class TestNumericRule(unittest.TestCase):
    def test_numeric_rule_pass(self):
        rule = NumericRule()
        fact = AtomicFact(
            subject="报销比例",
            predicate="为",
            object_value="70%",
            fact_type=FactType.NUMERIC,
        )
        context = "职工医保报销比例为70%"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_numeric_rule_fail(self):
        rule = NumericRule()
        fact = AtomicFact(
            subject="报销比例",
            predicate="为",
            object_value="80%",
            fact_type=FactType.NUMERIC,
        )
        context = "职工医保报销比例为70%"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)


class TestRelationRule(unittest.TestCase):
    def test_relation_rule_pass(self):
        rule = RelationRule()
        fact = AtomicFact(
            subject="阿司匹林",
            predicate="禁止与",
            object_value="布洛芬",
            fact_type=FactType.ENTITY_REL,
            is_negative=True,
        )
        context = "阿司匹林禁止与布洛芬合用"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_relation_rule_fail(self):
        rule = RelationRule()
        fact = AtomicFact(
            subject="阿司匹林",
            predicate="禁止与",
            object_value="维生素C",
            fact_type=FactType.ENTITY_REL,
            is_negative=True,
        )
        context = "阿司匹林禁止与布洛芬合用"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)


class TestScopeRule(unittest.TestCase):
    def test_scope_rule_pass(self):
        rule = ScopeRule()
        fact = AtomicFact(
            subject="申报",
            predicate="仅限",
            object_value="2026年前",
            fact_type=FactType.CONDITION,
        )
        context = "本项目申报仅限2026年前提交"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_scope_rule_fail(self):
        rule = ScopeRule()
        fact = AtomicFact(
            subject="申报",
            predicate="仅限",
            object_value="2025年前",
            fact_type=FactType.CONDITION,
        )
        context = "本项目申报仅限2026年前提交"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)


class TestCategoricalRule(unittest.TestCase):
    def test_categorical_rule_pass(self):
        rule = CategoricalRule()
        fact = AtomicFact(
            subject="医院",
            predicate="等级",
            object_value="三级甲等",
            fact_type=FactType.CATEGORICAL,
        )
        context = "该医院为三级甲等综合医院"
        result = rule.verify(fact, context)
        self.assertTrue(result.passed)

    def test_categorical_rule_fail(self):
        rule = CategoricalRule()
        fact = AtomicFact(
            subject="医院",
            predicate="等级",
            object_value="二级甲等",
            fact_type=FactType.CATEGORICAL,
        )
        context = "该医院为三级甲等综合医院"
        result = rule.verify(fact, context)
        self.assertFalse(result.passed)


class TestRuleMatrix(unittest.TestCase):
    def test_rule_matrix_dispatch(self):
        matrix = RuleMatrix()
        
        # 数值型
        numeric_fact = AtomicFact(
            subject="比例",
            predicate="为",
            object_value="70%",
            fact_type=FactType.NUMERIC,
        )
        result = matrix.verify(numeric_fact, "报销比例为70%")
        self.assertTrue(result.passed)
        
        # 关系型
        relation_fact = AtomicFact(
            subject="A药",
            predicate="禁止与",
            object_value="B药",
            fact_type=FactType.ENTITY_REL,
            is_negative=True,
        )
        result = matrix.verify(relation_fact, "A药禁止与B药合用")
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
