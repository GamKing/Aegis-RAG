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
    EvalSample,
    ExtractedFact,
    ExtractionPayload,
    PipelineTrace,
    VerificationFinding,
    VerificationResult,
)
from . import text_utils as tu


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
    r"(?P<raw>\$?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|亿|万)?%?)",
    re.IGNORECASE,
)


class RuleBasedFactExtractor:
    """离线安全抽取器：只从 context 原文提取数字/百分比事实。"""

    def extract(self, query: str, context: str) -> ExtractionPayload:
        facts: list[ExtractedFact] = []
        normalized = tu.normalize_faithfulness_text(context)
        for match in _NUMBER_WITH_SCALE_RE.finditer(normalized):
            value = match.group("raw").strip()
            if not value:
                continue
            facts.append(
                ExtractedFact(
                    key="numeric_fact",
                    value=value,
                    source_text=match.group(0),
                    is_numeric=True,
                )
            )
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
            raw_json={"query": query, "facts": [f.model_dump() for f in unique]},
        )


class ControlledSynthesizer:
    """只输出已经通过代码验证的事实，不重新自由生成数字。"""

    def synthesize(self, query: str, payload: ExtractionPayload) -> str:
        return "；".join(f"{fact.key}: {fact.value}" for fact in payload.facts)


class Tier1CodeVerifier:
    """毫秒级确定性质检：字段、实体和数字必须有上下文出处。"""

    def verify(
        self,
        sample: EvalSample,
        context: str,
        payload: ExtractionPayload,
    ) -> VerificationResult:
        findings: list[VerificationFinding] = []
        normalized_context = tu.normalize_faithfulness_text(context)
        seen: set[str] = set()

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

            nli_verified = set(payload.raw_json.get("nli_verified_fields", []))
            numbers = tu.extract_faithfulness_numbers(fact.value)
            if field in nli_verified:
                continue
            if numbers:
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
            elif value_norm not in tu.normalize_for_match(normalized_context):
                findings.append(
                    VerificationFinding(
                        code="UNSUPPORTED_FACT",
                        message=f"事实 {fact.value!r} 在上下文中无出处",
                        field=field,
                    )
                )

        context_norm = tu.normalize_for_match(normalized_context)
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
    payload = extractor.extract(sample.question, context)
    first = verifier.verify(sample, context, payload)
    trace = PipelineTrace(
        tier1_initial_passed=first.passed,
        tier1_final_passed=first.passed,
        findings=list(first.findings),
    )
    final = first

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
        context = self.context_provider(sample)
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
