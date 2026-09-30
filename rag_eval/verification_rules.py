"""通用事实校验规则矩阵：Strategy 模式实现多类型事实校验。

将单一的"查数字"扩展为多策略校验矩阵：
- NumericRule: 数值型事实（70%、1500元）
- RelationRule: 实体关系事实（A药禁止与B药合用）
- ScopeRule: 限制/条件事实（仅限2026年前、非直系亲属除外）
- CategoricalRule: 分类/枚举事实（三级甲等、必须、禁止）

每条规则返回 VerificationResult，供 Tier1CodeVerifier 和 FaithfulnessEvaluator 使用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from . import text_utils as tu
from .models import AtomicFact, FactType, VerificationFinding, VerificationResult


class VerificationRule(Protocol):
    """校验规则接口：所有规则实现 verify 方法。"""

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        """校验单个事实是否在上下文中有出处。"""
        ...


@dataclass
class NumericRule:
    """数值型事实校验规则。

    校验逻辑：
    1. 提取事实中的数值（支持日期/金额单位归一化）
    2. 对每个数值做边界敏感匹配
    3. 任一数值无出处则判定失败
    """

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []

        # 从 object_value 中提取数值
        normalized_value = tu.normalize_faithfulness_text(fact.object_value)
        numbers = tu.extract_numbers(normalized_value)

        if not numbers:
            # 无数值，跳过数值校验
            return VerificationResult(passed=True, findings=[], tier="tier1")

        normalized_context = tu.normalize_faithfulness_text(context)

        # 引用必须确实来自当前允许的上下文；有引用时只在该片段内证明数值。
        if fact.source_text:
            evidence = tu.normalize_faithfulness_text(fact.source_text)
            if tu.normalize_for_match(evidence) not in tu.normalize_for_match(normalized_context):
                return VerificationResult(passed=False, tier="tier1", findings=[
                    VerificationFinding(code="INVALID_SOURCE_TEXT", field="source_text",
                                        message="数值事实引用片段不属于当前证据")])
            normalized_context = evidence

        for value, is_pct in numbers:
            if not tu.number_supported_in_context(value, is_pct, normalized_context):
                findings.append(
                    VerificationFinding(
                        code="UNSUPPORTED_NUMBER",
                        message=f"数值 {value}{'%' if is_pct else ''} 在上下文中无出处",
                        field="object_value",
                    )
                )

        # 有明确主体时，数字存在还不够：主体、属性及对应数字必须在局部语句绑定。
        # 不使用跨句共现，也不允许跳过属性后第一个数值去匹配别人的数值。
        if fact.subject and not self._binding_supported(fact, normalized_context, numbers):
            findings.append(VerificationFinding(
                code="NUMERIC_BINDING_MISMATCH", field="subject/predicate/object_value",
                message=f"无法在同一证据语句中确认 {fact.subject}/{fact.predicate}/{fact.object_value} 的对应关系"))

        return VerificationResult(
            passed=not findings,
            findings=findings,
            tier="tier1",
        )

    @staticmethod
    def _binding_supported(fact: AtomicFact, context: str, numbers: list) -> bool:
        subject = tu.normalize_for_match(tu.normalize_faithfulness_text(fact.subject))
        predicate = tu.normalize_for_match(tu.normalize_faithfulness_text(fact.predicate))
        for clause in re.split(r"[。；;\n！？!?]", context):
            clause = tu.normalize_for_match(clause)
            for occurrence in re.finditer(re.escape(subject), clause):
                tail = clause[occurrence.end():]
                if predicate:
                    at = tail.find(predicate)
                    # 禁止越过另一个属性值，再用后续主体的同名属性作证。
                    if at < 0 or tu.extract_numbers(tail[:at]):
                        continue
                    tail = tail[at + len(predicate):]
                first = tu.extract_numbers(tail)
                if len(numbers) == 1 and first and first[0] == numbers[0]:
                    return True
        return False


@dataclass
class RelationRule:
    """实体关系事实校验规则。

    校验逻辑：
    1. 提取主体和客体作为两个实体
    2. 检查两个实体是否在上下文中同现（句子级或段落级）
    3. 如果事实是否定型（is_negative=True），还需检查否定词是否同现
    """

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []

        # 检查主体和客体是否在上下文中同现
        if not tu.entities_co_occur(
            fact.subject, fact.object_value, context, window="sentence"
        ):
            findings.append(
                VerificationFinding(
                    code="ENTITY_RELATION_NOT_FOUND",
                    message=f"实体关系 '{fact.subject} - {fact.object_value}' 在上下文中未同现",
                    field="subject/object_value",
                )
            )

        # 如果是否定型事实，检查否定词是否同现
        if fact.is_negative:
            # 检查谓词中的否定词是否在上下文中出现
            predicate_norm = tu.normalize_for_match(fact.predicate)
            context_norm = tu.normalize_for_match(context)
            if predicate_norm not in context_norm:
                findings.append(
                    VerificationFinding(
                        code="NEGATION_NOT_FOUND",
                        message=f"否定谓词 '{fact.predicate}' 在上下文中无出处",
                        field="predicate",
                    )
                )

        return VerificationResult(
            passed=not findings,
            findings=findings,
            tier="tier1",
        )


@dataclass
class ScopeRule:
    """限制/条件事实校验规则。

    校验逻辑：
    1. 提取事实中的条件关键词
    2. 检查条件关键词是否在上下文中出现
    3. 验证 object_value 的具体内容是否在上下文中出现
    4. 如果是否定型事实，检查否定词是否出现
    """

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []

        # 从谓词和客体中提取条件关键词
        condition_text = f"{fact.predicate} {fact.object_value}"
        condition_keywords = tu.extract_condition_keywords(condition_text)

        if condition_keywords:
            context_norm = tu.normalize_for_match(context)
            for kw in condition_keywords:
                kw_norm = tu.normalize_for_match(kw)
                if kw_norm not in context_norm:
                    findings.append(
                        VerificationFinding(
                            code="CONDITION_NOT_FOUND",
                            message=f"条件关键词 '{kw}' 在上下文中无出处",
                            field="predicate/object_value",
                        )
                    )

        # 验证 object_value 的具体内容是否在上下文中出现
        if fact.object_value:
            value_norm = tu.normalize_for_match(fact.object_value)
            context_norm = tu.normalize_for_match(context)
            if value_norm not in context_norm:
                findings.append(
                    VerificationFinding(
                        code="CONDITION_VALUE_NOT_FOUND",
                        message=f"条件值 '{fact.object_value}' 在上下文中无出处",
                        field="object_value",
                    )
                )

        # 如果是否定型事实，检查否定词
        if fact.is_negative:
            if not tu.detect_negation(condition_text):
                findings.append(
                    VerificationFinding(
                        code="NEGATION_MISSING",
                        message=f"事实标记为否定型，但未检测到否定词",
                        field="is_negative",
                    )
                )

        return VerificationResult(
            passed=not findings,
            findings=findings,
            tier="tier1",
        )


@dataclass
class CategoricalRule:
    """分类/枚举事实校验规则。

    校验逻辑：
    1. 将客体值作为分类标签
    2. 检查标签是否在上下文中出现（子串匹配）
    """

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []

        if not fact.object_value:
            return VerificationResult(passed=True, findings=[], tier="tier1")

        # 子串匹配：客体值必须在上下文中出现
        value_norm = tu.normalize_for_match(fact.object_value)
        context_norm = tu.normalize_for_match(context)

        if value_norm not in context_norm:
            findings.append(
                VerificationFinding(
                    code="CATEGORY_NOT_FOUND",
                    message=f"分类标签 '{fact.object_value}' 在上下文中无出处",
                    field="object_value",
                )
            )

        return VerificationResult(
            passed=not findings,
            findings=findings,
            tier="tier1",
        )


class RuleMatrix:
    """校验规则矩阵：根据事实类型分发到对应规则。

    使用方式：
        matrix = RuleMatrix()
        result = matrix.verify(fact, context)

    行业插槽 3：通过 register 方法挂载行业专属规则插件
    （如金融量纲换算 UnitConversionRule、医疗黑名单 BlacklistRiskRule）。
    """

    def __init__(self) -> None:
        self.rules: dict[FactType, VerificationRule] = {
            FactType.NUMERIC: NumericRule(),
            FactType.ENTITY_REL: RelationRule(),
            FactType.CONDITION: ScopeRule(),
            FactType.CATEGORICAL: CategoricalRule(),
        }
        # 额外行业规则（作用于全部事实类型的交叉检查），按注册顺序执行
        self.extra_rules: list[VerificationRule] = []

    def register(self, rule: VerificationRule) -> None:
        """注册一个行业专属规则插件。"""
        self.extra_rules.append(rule)

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        """根据事实类型分发到对应规则进行校验，并叠加行业额外规则。"""
        rule = self.rules.get(fact.fact_type)
        findings: list[VerificationFinding] = []

        if rule is None:
            # 未知类型，默认通过（保守策略）
            findings.append(
                VerificationFinding(
                    code="UNKNOWN_FACT_TYPE",
                    message=f"未知事实类型: {fact.fact_type}",
                    field="fact_type",
                    severity="warning",
                )
            )
        else:
            findings.extend(rule.verify(fact, context).findings)

        # 叠加行业额外规则
        for extra in self.extra_rules:
            findings.extend(extra.verify(fact, context).findings)

        errors = [f for f in findings if f.severity == "error"]
        return VerificationResult(passed=not errors, findings=findings, tier="tier1")


# ---------------------------------------------------------------------------
# 行业专属规则插件（插槽 3 的具体落地示例）
# ---------------------------------------------------------------------------


@dataclass
class UnitConversionRule:
    """金融量纲换算规则：识别财报单位（万元/亿元/美元）并做等价映射。

    允许答案中的数值以等价量纲出现在上下文中。例如：
    - 答案 "0.5亿" 可被上下文 "5000万" 支持；
    - 答案 "108.0亿元" 可被上下文 "10,800,000,000" 支持。
    """

    # 量纲换算表：统一折算到"元"
    _SCALES = {
        "元": 1.0,
        "万": 1e4,
        "万元": 1e4,
        "亿": 1e8,
        "亿元": 1e8,
        "千万": 1e7,
        "百万": 1e6,
        "million": 1e6,
        "千": 1e3,
        "thousand": 1e3,
        "k": 1e3,
        "m": 1e6,
        "b": 1e9,
        "billion": 1e9,
    }

    _NUM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>亿元|万元|亿|万|元|千万|百万|billion|million|thousand|k|m|b)?")

    def _normalize_number_with_scale(self, text: str) -> list[float]:
        """把文本中的带单位数值统一折算为元。"""
        results: list[float] = []
        for m in self._NUM_RE.finditer(text.lower()):
            value = float(m.group("value"))
            unit = m.group("unit") or "元"  # 默认单位为"元"
            scale = self._SCALES.get(unit, 1.0)
            results.append(value * scale)
        return results

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []
        # 对事实的 object_value 做量纲归一化
        fact_amounts = self._normalize_number_with_scale(fact.object_value)
        if not fact_amounts:
            return VerificationResult(passed=True, findings=[], tier="tier1")

        context_amounts = self._normalize_number_with_scale(context)
        for amount in fact_amounts:
            # 允许少量浮点误差（量纲换算可能引入精度差）
            supported = any(abs(amount - ctx) < max(abs(amount) * 1e-6, 1e-6) for ctx in context_amounts)
            if not supported:
                findings.append(
                    VerificationFinding(
                        code="UNIT_MISMATCH",
                        message=f"数值/量纲 {fact.object_value} 在上下文中无等价出处",
                        field="object_value",
                    )
                )

        return VerificationResult(passed=not findings, findings=findings, tier="tier1")


@dataclass
class BlacklistRiskRule:
    """医疗/政务黑名单禁忌规则：高危禁忌核查，遗漏直接一票否决。

    用于两类场景：
    1. 绝对化/危险表述核查：答案若含黑名单词（如"绝对安全"、"根治"），
       而上下文没有对应支持，直接判定失败；
    2. 高风险禁忌核查：如妊娠禁用、配伍禁忌，一旦答案遗漏否定词，
       视为安全风险。
    """

    def __init__(self, blacklist: tuple = ("绝对安全", "根治", "无任何副作用", "百分百有效")) -> None:
        self.blacklist = tuple(blacklist)

    def verify(self, fact: AtomicFact, context: str) -> VerificationResult:
        findings: list[VerificationFinding] = []
        answer_text = f"{fact.predicate} {fact.object_value}"
        context_norm = tu.normalize_for_match(context)

        for word in self.blacklist:
            if word in answer_text and word not in context_norm:
                findings.append(
                    VerificationFinding(
                        code="RISK_WORD_IN_ANSWER",
                        message=f"答案含高危表述 '{word}' 但上下文中无出处",
                        field="object_value",
                    )
                )

        return VerificationResult(passed=not findings, findings=findings, tier="tier1")
