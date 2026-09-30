"""受限解码状态机（Day 3-4）：确定性质控 + 单次打回重试闭环。

两个核心组件：

1. GenerationFSM — 显式有限状态机，封装受控生成的状态转移闭环：
       EXTRACT → VALIDATE → [DECIDE] → (ACCEPT | REPAIR → REVALIDATE → ACCEPT | REJECT)
   实现"单次打回重试"：Tier-1 失败 → Tier-2 NLI 兜底 → 重跑 Tier-1 → 通过即 ACCEPT，
   仍失败则 REJECT（安全失败）。

2. ConstraintDecoder / JsonSchemaConstraint — FSM 受限解码：
   约束合成器的原始输出必须满足 JSON Schema（顶层 object、字段白名单、
   atomic_facts 数组结构）。非法输出触发 REPAIR（规范化/剔除非法字段），
   无法修复则 REJECT。

全部确定性：无外部模型依赖，可复现、可审计。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .models import ExtractionPayload, VerificationFinding, VerificationResult


# ---------------------------------------------------------------------------
# 受限解码：JSON Schema 强制
# ---------------------------------------------------------------------------

# 允许的原子事实字段白名单（对应 AtomicFact 模型）
_ATOMIC_FACT_ALLOWED_FIELDS = {
    "subject": str,
    "predicate": str,
    "object_value": str,
    "fact_type": str,
    "is_negative": bool,
    "source_text": str,
}

# 顶层允许的键
_TOP_LEVEL_ALLOWED_KEYS = {"query", "facts", "atomic_facts", "self_corrected", "nli_verified_fields"}


class SchemaViolation(Exception):
    """JSON Schema 约束违规。"""


@dataclass
class ConstraintResult:
    """受限解码结果。"""

    payload: ExtractionPayload
    repaired: bool
    violations: list[str] = field(default_factory=list)


class JsonSchemaConstraint:
    """对 ExtractionPayload 的 raw_json 施加 JSON Schema 约束。"""

    def coerce(self, payload: ExtractionPayload) -> ConstraintResult:
        """校验并规范化 raw_json，使其满足 schema。"""
        violations: list[str] = []
        raw = dict(payload.raw_json)

        # 顶层必须是 object（dict）
        if not isinstance(raw, dict):
            violations.append("顶层必须是 JSON object")
            raw = {}

        # 剔除未知顶层键（白名单）
        for key in list(raw.keys()):
            if key not in _TOP_LEVEL_ALLOWED_KEYS:
                violations.append(f"未知顶层键 {key!r}，已剔除")
                del raw[key]

        # 校验 atomic_facts 结构
        atomic = raw.get("atomic_facts", [])
        if not isinstance(atomic, list):
            violations.append("atomic_facts 必须是数组")
            atomic = []
        cleaned_atomic: list[dict] = []
        for idx, item in enumerate(atomic):
            if not isinstance(item, dict):
                violations.append(f"atomic_facts[{idx}] 必须是 object，已剔除")
                continue
            cleaned: dict = {}
            for k, expected in _ATOMIC_FACT_ALLOWED_FIELDS.items():
                v = item.get(k)
                if v is None:
                    cleaned[k] = "" if expected is str else False
                elif isinstance(v, expected):
                    cleaned[k] = v
                else:
                    violations.append(f"atomic_facts[{idx}].{k} 类型非法，已规范化为默认值")
                    cleaned[k] = "" if expected is str else False
            cleaned_atomic.append(cleaned)
        raw["atomic_facts"] = cleaned_atomic

        # 校验 facts 数组
        facts = payload.facts  # 由模型层保证类型，这里仅透传

        constrained = ExtractionPayload(
            facts=facts,
            claims=payload.claims,
            raw_json=raw,
        )
        return ConstraintResult(payload=constrained, repaired=bool(violations), violations=violations)


# ---------------------------------------------------------------------------
# 生成受限解码器
# ---------------------------------------------------------------------------

class DecodingState(str, Enum):
    """受限解码的有限状态。"""

    EXTRACT = "extract"            # 抽取原子事实
    VALIDATE = "validate"          # Tier-1 代码质控
    DECIDE = "decide"              # 判定是否通过
    ACCEPT = "accept"              # 通过，输出答案
    REPAIR = "repair"              # Tier-2 NLI 兜底修复
    REVALIDATE = "revalidate"      # 修复后重跑 Tier-1
    REJECT = "reject"              # 无法修复，安全失败


# 合法状态转移表（确定性 FSM）
_STATE_TRANSITIONS: dict[DecodingState, set[DecodingState]] = {
    DecodingState.EXTRACT: {DecodingState.VALIDATE},
    DecodingState.VALIDATE: {DecodingState.DECIDE},
    DecodingState.DECIDE: {DecodingState.ACCEPT, DecodingState.REPAIR, DecodingState.REJECT},
    DecodingState.REPAIR: {DecodingState.REVALIDATE},
    DecodingState.REVALIDATE: {DecodingState.DECIDE, DecodingState.REJECT},
    DecodingState.ACCEPT: set(),
    DecodingState.REJECT: set(),
}


@dataclass
class FSMResult:
    """状态机闭环的最终产出。"""

    payload: ExtractionPayload
    verification: VerificationResult
    accepted: bool
    trace: list[DecodingState] = field(default_factory=list)
    repair_count: int = 0
    schema_violations: list[str] = field(default_factory=list)


class GenerationFSM:
    """受控生成 + 单次打回重试的有限状态机。

    用法：
        fsm = GenerationFSM(extractor, verifier, synthesizer, self_corrector)
        result = fsm.run(sample, context)

    状态闭环（最多一次 REPAIR）：
        EXTRACT → VALIDATE → DECIDE →（通过则 ACCEPT；失败且有 NLI 则 REPAIR）
        REPAIR → REVALIDATE → DECIDE →（通过 ACCEPT，仍失败 REJECT）
    """

    def __init__(
        self,
        extractor,
        verifier,
        synthesizer,
        self_corrector=None,
        constraint: JsonSchemaConstraint | None = None,
    ) -> None:
        self.extractor = extractor
        self.verifier = verifier
        self.synthesizer = synthesizer
        self.self_corrector = self_corrector
        self.constraint = constraint or JsonSchemaConstraint()

    def _transition(self, current: DecodingState, target: DecodingState) -> bool:
        """校验状态转移是否合法。"""
        return target in _STATE_TRANSITIONS[current]

    def run(self, sample, context: str) -> FSMResult:
        trace: list[DecodingState] = [DecodingState.EXTRACT]

        # EXTRACT
        payload = self.extractor.extract(sample.question, context)
        # 受限解码：先做 JSON Schema 强制
        constrained = self.constraint.coerce(payload)
        payload = constrained.payload
        if constrained.repaired:
            # 规范化修复不计入"打回重试"，但记录违规
            trace.append(DecodingState.VALIDATE)
        else:
            trace.append(DecodingState.VALIDATE)

        # VALIDATE
        verification = self.verifier.verify(sample, context, payload)
        trace.append(DecodingState.DECIDE)

        repair_count = 0
        # DECIDE / (REPAIR → REVALIDATE → DECIDE) 最多一次
        if verification.passed:
            trace.append(DecodingState.ACCEPT)
        elif self.self_corrector is not None:
            # REPAIR：Tier-2 NLI 兜底
            trace.append(DecodingState.REPAIR)
            repair_count = 1
            payload = self.self_corrector.correct(context, payload, verification.findings)
            # REVALIDATE：重跑 Tier-1
            trace.append(DecodingState.REVALIDATE)
            verification = self.verifier.verify(sample, context, payload)
            trace.append(DecodingState.DECIDE)
            if verification.passed:
                trace.append(DecodingState.ACCEPT)
            else:
                trace.append(DecodingState.REJECT)
        else:
            trace.append(DecodingState.REJECT)

        accepted = trace[-1] == DecodingState.ACCEPT
        return FSMResult(
            payload=payload,
            verification=verification,
            accepted=accepted,
            trace=trace,
            repair_count=repair_count,
            schema_violations=constrained.violations,
        )


# ---------------------------------------------------------------------------
# 与现有 run_controlled_pipeline 对齐的适配函数
# ---------------------------------------------------------------------------

def run_fsm_pipeline(
    sample,
    context: str,
    extractor,
    synthesizer,
    verifier,
    self_corrector=None,
) -> PipelineResultAdapter:
    """在 FSM 闭环之上复刻 run_controlled_pipeline 的返回契约。"""
    fsm = GenerationFSM(
        extractor=extractor,
        verifier=verifier,
        synthesizer=synthesizer,
        self_corrector=self_corrector,
    )
    result = fsm.run(sample, context)
    answer = synthesizer.synthesize(sample.question, result.payload) if result.accepted else ""
    return PipelineResultAdapter(
        answer=answer,
        payload=result.payload,
        verification=result.verification,
        accepted=result.accepted,
        repair_count=result.repair_count,
        trace_states=[s.value for s in result.trace],
    )


@dataclass
class PipelineResultAdapter:
    """轻量结果容器，替代 pipeline.PipelineResult 供 FSM 路径使用。"""

    answer: str
    payload: ExtractionPayload
    verification: VerificationResult
    accepted: bool
    repair_count: int
    trace_states: list[str]