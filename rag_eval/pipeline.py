"""受限抽取 -> Tier 1 质检 -> Tier 2 自愈 -> 受控合成流水线。

默认实现完全离线。生产环境可替换 FactExtractor / AnswerSynthesizer，接入
JSON Schema 强制解码模型与轻量 NLI，但所有模型输出都必须重新经过 Tier 1。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .dummy_rag import BaseRAG, RAGResponse
from .models import (
    AtomicFact,
    EvalSample,
    ExtractedFact,
    ExtractionPayload,
    FactType,
    PipelineTrace,
    VerificationFinding,
    VerificationResult,
)
from . import text_utils as tu
from .verification_rules import RuleMatrix


class FactExtractor(Protocol):
    def extract(self, query: str, context: str) -> ExtractionPayload: ...


class AnswerSynthesizer(Protocol):
    def synthesize(self, query: str, payload: ExtractionPayload) -> str: ...


class SelfCorrector(Protocol):
    def correct(
        self,
        context: str,
        payload: ExtractionPayload,
        findings: Sequence[VerificationFinding],
    ) -> ExtractionPayload: ...


_NUMBER_WITH_SCALE_RE = re.compile(
    r"(?P<raw>\$?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|亿|万|元|个百分点)?%?)",
    re.IGNORECASE,
)


def _cjk_bigrams(text: str) -> set[str]:
    """提取文本的中文二元组，用于 query 与段落的相关性判定。"""
    bigrams = set()
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chars) - 1):
        bigrams.add(chars[i] + chars[i + 1])
    return bigrams


def _extract_relevant_paragraphs(query: str, context: str) -> list[str]:
    """提取与 query 相关的完整段落，保留名词实体语义完整性。

    策略：按双换行切分段落，保留含数字或与 query 共享中文二元组的段落。
    未命中时回退为全量段落，保证不漏关键内容。
    """
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    query_bigrams = _cjk_bigrams(query)
    relevant = []
    for para in paragraphs:
        has_number = re.search(r"\d", para)
        has_query_term = bool(_cjk_bigrams(para) & query_bigrams)
        if has_number or has_query_term:
            relevant.append(para)
    return relevant or list(paragraphs)


class RuleBasedFactExtractor:
    """离线安全抽取器：从 context 原文提取多种类型的事实。

    支持四种事实类型：
    - NUMERIC: 数值型（百分比、金额、日期等）
    - ENTITY_REL: 实体关系型（含否定词的关系）
    - CONDITION: 条件限制型（含"仅限"、"必须"等关键词）
    - CATEGORICAL: 分类枚举型（等级、类型等）

    向后兼容：同时生成旧式 ExtractedFact（用于现有测试）和新式 AtomicFact
    （用于通用质检引擎）。
    """

    def extract(self, query: str, context: str) -> ExtractionPayload:
        facts: list[ExtractedFact] = []
        atomic_facts: list[dict] = []
        normalized = tu.normalize_faithfulness_text(context)

        # 0. 保留与 query 相关的完整段落（保证名词实体如"定点医疗机构"、
        #    "急诊"、"医保目录"不被丢弃，从而支撑 Completeness 评分）。
        relevant_paragraphs = _extract_relevant_paragraphs(query, context)
        for pidx, para in enumerate(relevant_paragraphs):
            atomic_facts.append({
                "subject": "",
                "predicate": "",
                "object_value": para,
                "fact_type": "categorical",
                "is_negative": False,
                "source_text": para,
            })

        # 1. 抽取数值型事实（保持原有逻辑）
        for match in _NUMBER_WITH_SCALE_RE.finditer(normalized):
            value = match.group("raw").strip()
            if not value:
                continue

            # 提取数值周围的上下文作为 subject/predicate
            start = max(0, match.start() - 20)
            end = min(len(normalized), match.end() + 10)
            surrounding = normalized[start:end]

            # 简单启发式：数值前的中文词作为 predicate
            before_text = normalized[max(0, match.start() - 10):match.start()]
            predicate_match = re.findall(r"[\u4e00-\u9fff]{2,6}$", before_text)
            predicate = predicate_match[0] if predicate_match else "数值"

            facts.append(
                ExtractedFact(
                    key="numeric_fact",
                    value=value,
                    source_text=match.group(0),
                    is_numeric=True,
                )
            )

            atomic_facts.append({
                "subject": "",  # 数值型通常无明确 subject
                "predicate": predicate,
                "object_value": value,
                "fact_type": "numeric",
                "is_negative": False,
                "source_text": match.group(0),
            })

        # 注：entity_rel / condition 的纯启发式抽取（用否定词/条件词切出前后实体）
        # 对长段落不可靠（会把"非医保目录内费用不纳入"误判为实体关系），且与
        # 上面已保留的相关完整段落重复。完整段落已覆盖名词实体，故此处不再生成
        # 这两类原子事实，交由更稳健的段落级 categorical 承载。

        # 去重（按 value 去重）
        unique: list[ExtractedFact] = []
        seen: set[str] = set()
        for fact in facts:
            key = tu.normalize_for_match(fact.value)
            if key not in seen:
                seen.add(key)
                unique.append(fact)

        claims = [f.value for f in unique]
        return ExtractionPayload(
            facts=unique,
            claims=claims,
            raw_json={
                "extraction_mode": "verbatim_evidence",
                "query": query,
                "facts": [f.model_dump() for f in unique],
                "atomic_facts": atomic_facts,
            },
        )


class ControlledSynthesizer:
    """只输出已经通过代码验证的事实，不重新自由生成数字。

    支持两种输入：
    - 旧式 ExtractedFact（key/value 对）：保持向后兼容，输出 "key: value"
    - 新式 AtomicFact（主体/谓词/客体三元组）：输出自然语言句子

    生成逻辑完全确定性（无 LLM），通过模板拼装保证每个词都可追溯到上下文。
    """

    def synthesize(self, query: str, payload: ExtractionPayload) -> str:
        # 优先使用 AtomicFact（新式三元组）
        atomic_facts = payload.raw_json.get("atomic_facts", [])
        if payload.raw_json.get("extraction_mode") == "verbatim_evidence":
            # 原文段落已经包含数值及其主体，不再追加脱离来源的数值碎片。
            paragraphs = [fact for fact in atomic_facts
                          if fact.get("fact_type") == "categorical"
                          and not fact.get("subject") and not fact.get("predicate")
                          and fact.get("object_value") == fact.get("source_text")]
            if paragraphs:
                return self._synthesize_atomic(paragraphs)
        if atomic_facts:
            return self._synthesize_atomic(atomic_facts)

        # 回退到旧式 ExtractedFact
        if payload.facts:
            return "；".join(f"{fact.key}: {fact.value}" for fact in payload.facts)

        return ""

    def _synthesize_atomic(self, atomic_facts: list[dict]) -> str:
        """将 AtomicFact 列表拼装为自然语言句子。"""
        sentences: list[str] = []
        for f in atomic_facts:
            subject = f.get("subject", "")
            predicate = f.get("predicate", "")
            object_value = f.get("object_value", "")
            is_negative = f.get("is_negative", False)
            fact_type = f.get("fact_type", "numeric")

            if not object_value:
                continue

            # 根据事实类型选择模板
            if fact_type == "numeric":
                sentences.append(f"{subject}{predicate}{object_value}")
            elif fact_type == "entity_rel":
                neg = "不得" if is_negative else ""
                sentences.append(f"{subject}{neg}{predicate}{object_value}")
            elif fact_type == "condition":
                neg = "不得" if is_negative else ""
                sentences.append(f"{subject}{neg}{predicate}{object_value}")
            elif fact_type == "categorical":
                # 整段原文：直接输出，避免空谓词拼接破坏语义
                if not predicate:
                    sentences.append(object_value)
                else:
                    sentences.append(f"{subject}{predicate}{object_value}")
            else:
                sentences.append(f"{subject}{predicate}{object_value}")

        return "；".join(sentences) + "。" if sentences else ""


class Tier1CodeVerifier:
    """毫秒级确定性质检：使用规则矩阵验证多种类型的事实。

    支持四种事实类型的校验：
    - NUMERIC: 数值边界敏感匹配
    - ENTITY_REL: 实体共现检测
    - CONDITION: 条件关键词检测
    - CATEGORICAL: 分类标签匹配

    向后兼容：同时验证旧式 ExtractedFact 和新式 AtomicFact。
    """

    def __init__(self) -> None:
        self.rule_matrix = RuleMatrix()

    def verify(
        self,
        sample: EvalSample,
        context: str,
        payload: ExtractionPayload,
    ) -> VerificationResult:
        findings: list[VerificationFinding] = []
        normalized_context = tu.normalize_faithfulness_text(context)
        literal_context = tu.normalize_text(context)
        seen: set[str] = set()

        # 1. 验证新式 AtomicFact（使用规则矩阵）
        atomic_facts = payload.raw_json.get("atomic_facts", [])
        for index, fact_dict in enumerate(atomic_facts):
            field = f"atomic_facts[{index}]"

            # 转换为 AtomicFact 对象
            fact = AtomicFact(
                subject=fact_dict.get("subject", ""),
                predicate=fact_dict.get("predicate", ""),
                object_value=fact_dict.get("object_value", ""),
                fact_type=FactType(fact_dict.get("fact_type", "numeric")),
                is_negative=fact_dict.get("is_negative", False),
                source_text=fact_dict.get("source_text"),
            )

            # 检查空值
            if not fact.object_value:
                findings.append(
                    VerificationFinding(code="EMPTY_FACT", message="事实值为空", field=field)
                )
                continue

            # 使用规则矩阵验证
            rule_result = self.rule_matrix.verify(fact, context)
            findings.extend(rule_result.findings)

        # 2. 验证旧式 ExtractedFact（保持向后兼容）
        nli_verified = set(payload.raw_json.get("nli_verified_fields", []))
        for index, fact in enumerate(payload.facts):
            field = f"facts[{index}]"
            value_norm = tu.normalize_for_match(fact.value)
            if not value_norm:
                findings.append(
                    VerificationFinding(code="EMPTY_FACT", message="事实值为空", field=field)
                )
                continue
            if value_norm in seen:
                findings.append(
                    VerificationFinding(code="DUPLICATE_FACT", message="事实重复", field=field)
                )
            seen.add(value_norm)

            # 如果该字段已被 NLI 验证通过，跳过后续检查
            if field in nli_verified:
                continue

            # 数值型事实：边界敏感匹配
            if fact.is_numeric:
                normalized_fact = tu.normalize_faithfulness_text(fact.value)
                numbers = tu.extract_faithfulness_numbers(normalized_fact)
                for value, is_pct in numbers:
                    if not tu.number_supported_in_context(
                        value, is_pct, normalized_context
                    ):
                        findings.append(
                            VerificationFinding(
                                code="UNSUPPORTED_NUMBER",
                                message=f"数值 {value}{'%' if is_pct else ''} 在上下文中无出处",
                                field=field,
                            )
                        )
            # 非数值型事实：子串匹配
            elif value_norm not in tu.normalize_for_match(normalized_context):
                findings.append(
                    VerificationFinding(
                        code="UNSUPPORTED_FACT",
                        message=f"事实 {fact.value!r} 在上下文中无出处",
                        field=field,
                    )
                )

        # 3. 检查预期实体（保持原有逻辑）
        context_norm = tu.normalize_for_match(literal_context)
        for entity in sample.expected_entities:
            if tu.normalize_for_match(entity) not in context_norm:
                findings.append(
                    VerificationFinding(
                        code="MISSING_EXPECTED_ENTITY",
                        message=f"标准实体 {entity!r} 未在上下文中出现",
                        field="expected_entities",
                        severity="warning",
                    )
                )

        errors = [f for f in findings if f.severity == "error"]
        return VerificationResult(
            passed=not errors,
            findings=findings,
            tier="tier1",
        )


class NLIAndSelfCorrector:
    """对 Tier1 失败事实逐条做 NLI 判定，并最多执行一次修复。"""

    def __init__(self, judge) -> None:
        self.judge = judge

    def correct(
        self,
        context: str,
        payload: ExtractionPayload,
        findings: Sequence[VerificationFinding],
    ) -> ExtractionPayload:
        unsupported_fields = {
            finding.field
            for finding in findings
            if finding.code in {"UNSUPPORTED_NUMBER", "UNSUPPORTED_FACT"}
            and finding.field
        }
        kept: list[ExtractedFact] = []
        nli_verified_fields: list[str] = []
        for index, fact in enumerate(payload.facts):
            field = f"facts[{index}]"
            if field not in unsupported_fields:
                kept.append(fact)
                continue
            claim = f"{fact.key}: {fact.value}"
            try:
                accepted = self.judge.check_faithfulness(context, claim)
            except Exception:
                accepted = False
            if accepted:
                kept.append(fact)
                nli_verified_fields.append(field)
        return ExtractionPayload(
            facts=kept,
            claims=[f.value for f in kept],
            raw_json={
                **payload.raw_json,
                "self_corrected": True,
                "nli_verified_fields": nli_verified_fields,
            },
        )


@dataclass(frozen=True)
class PipelineResult:
    answer: str
    payload: ExtractionPayload
    verification: VerificationResult
    trace: PipelineTrace


def run_controlled_pipeline(
    sample: EvalSample,
    context: str,
    extractor: FactExtractor,
    synthesizer: AnswerSynthesizer,
    verifier: Tier1CodeVerifier,
    self_corrector: SelfCorrector | None = None,
) -> PipelineResult:
    """执行完整五步流程，Tier2 后必须重新经过 Tier1。"""
    # 阶段一：抽取和代码质检先形成安全边界，合成器只能消费已验证的 payload。
    payload = extractor.extract(sample.question, context)
    first = verifier.verify(sample, context, payload)
    trace = PipelineTrace(
        tier1_initial_passed=first.passed,
        tier1_final_passed=first.passed,
        findings=list(first.findings),
    )
    final = first

    # 阶段二：仅在 Tier 1 失败时自愈，且修复后必须回到同一质检边界重新验收。
    if not first.passed and self_corrector is not None:
        trace = trace.model_copy(update={"tier2_triggered": True, "repair_count": 1})
        original_fact_count = len(payload.facts)
        payload = self_corrector.correct(context, payload, first.findings)
        final = verifier.verify(sample, context, payload)
        if len(payload.facts) < original_fact_count:
            final = final.model_copy(
                update={
                    "passed": False,
                    "findings": list(final.findings)
                    + [
                        VerificationFinding(
                            code="REMOVED_UNVERIFIED_FACT",
                            message="Tier2 删除了未被上下文支持的事实，进入安全失败",
                            severity="error",
                        )
                    ],
                }
            )
        trace = trace.model_copy(
            update={
                "tier1_final_passed": final.passed,
                "findings": list(first.findings) + list(final.findings),
            }
        )

    answer = synthesizer.synthesize(sample.question, payload) if final.passed else ""
    return PipelineResult(answer=answer, payload=payload, verification=final, trace=trace)


class PipelineRAG(BaseRAG):
    """把检索上下文接入受控流水线，并保持现有 RAGResponse 接口。"""

    def __init__(
        self,
        context_provider,
        extractor: FactExtractor | None = None,
        synthesizer: AnswerSynthesizer | None = None,
        verifier: Tier1CodeVerifier | None = None,
        self_corrector: SelfCorrector | None = None,
    ) -> None:
        self.context_provider = context_provider
        self.extractor = extractor or RuleBasedFactExtractor()
        self.synthesizer = synthesizer or ControlledSynthesizer()
        self.verifier = verifier or Tier1CodeVerifier()
        self.self_corrector = self_corrector
        self.last_trace: PipelineTrace | None = None

    def retrieve_and_generate(self, sample: EvalSample) -> RAGResponse:
        # 适配器只负责提供上下文；抽取、验证和合成仍由受控流水线统一管理。
        context = self.context_provider(sample)
        return self.generate_from_context(sample, context)

    def generate_from_context(self, sample: EvalSample, context: str) -> RAGResponse:
        """按请求传入已筛选证据，不修改共享的 context_provider。"""
        result = run_controlled_pipeline(
            sample,
            context,
            self.extractor,
            self.synthesizer,
            self.verifier,
            self.self_corrector,
        )
        self.last_trace = result.trace
        return RAGResponse(context=context, answer=result.answer)
