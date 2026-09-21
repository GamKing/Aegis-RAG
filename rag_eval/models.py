"""评测数据契约（Pydantic v2）。

契约字段与任务规范严格一致；EvalSummary 是 Runner 层新增的聚合视图，
用于承载整体平均指标与通过率。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """单条评测样本：问题 + 标准上下文 + 标准答案 + 必须命中的实体清单。"""

    id: str = Field(..., min_length=1, description="样本唯一标识")
    question: str = Field(..., description="用户问题")
    ground_truth_context: str = Field(..., description="标准（金标准）检索上下文")
    ground_truth_answer: str = Field(..., description="标准答案")
    expected_entities: list[str] = Field(
        default_factory=list,
        description="答案中必须命中的实体（金额、比例、机构名、政策术语等）",
    )


class ExtractedFact(BaseModel):
    """受限抽取后的单个原子事实。"""

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    source_text: str | None = None
    is_numeric: bool = False


class ExtractionPayload(BaseModel):
    """LLM JSON Schema 或规则抽取器的统一输出。"""

    facts: list[ExtractedFact] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    raw_json: dict[str, Any] = Field(default_factory=dict)


class VerificationFinding(BaseModel):
    """Tier 1/Tier 2 的结构化质检发现。"""

    code: str
    message: str
    field: str | None = None
    severity: Literal["error", "warning"] = "error"


class VerificationResult(BaseModel):
    """一次质检的结果。"""

    passed: bool
    findings: list[VerificationFinding] = Field(default_factory=list)
    tier: Literal["tier1", "tier2"]


class PipelineTrace(BaseModel):
    """分层流水线审计轨迹。"""

    tier1_initial_passed: bool = False
    tier2_triggered: bool = False
    repair_count: int = 0
    tier1_final_passed: bool = False
    findings: list[VerificationFinding] = Field(default_factory=list)


class EvalResult(BaseModel):
    """单条样本的评测产出。"""

    sample_id: str
    actual_context: str = Field(..., description="RAG 实际检索到的上下文")
    actual_answer: str = Field(..., description="RAG 实际生成的答案")
    context_recall_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    faithfulness_pass: bool
    error_details: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    """整批样本的聚合结果与阈值快照。"""

    generated_at: datetime = Field(default_factory=datetime.now)
    total: int = Field(..., ge=0)
    passed: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    avg_context_recall: float = Field(..., ge=0.0, le=1.0)
    avg_completeness: float = Field(..., ge=0.0, le=1.0)
    faithfulness_pass_rate: float = Field(..., ge=0.0, le=1.0)
    case_pass_rate: float = Field(..., ge=0.0, le=1.0)
    thresholds: dict[str, float] = Field(default_factory=dict)
    results: list[EvalResult] = Field(default_factory=list)

    def is_case_passed(self, result: EvalResult) -> bool:
        """按阈值快照判定单条结果是否通过（recall/completeness 达标且忠实）。"""
        recall_gate = result.context_recall_score >= self.thresholds.get(
            "context_recall", 0.0
        )
        completeness_gate = result.completeness_score >= self.thresholds.get(
            "completeness", 0.0
        )
        return recall_gate and completeness_gate and result.faithfulness_pass

    def failed_results(self) -> list[EvalResult]:
        return [r for r in self.results if not self.is_case_passed(r)]

    def model_dump_json_safe(self) -> dict[str, Any]:
        """供日志/上报使用的可序列化视图。"""
        return self.model_dump(mode="json")
