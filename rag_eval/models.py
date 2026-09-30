"""评测数据契约（Pydantic v2）。

契约字段与任务规范严格一致；EvalSummary 是 Runner 层新增的聚合视图，
用于承载整体平均指标与通过率。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class FactType(str, Enum):
    """事实类型枚举：将校验从单一"查数字"泛化为多类型命题校验。"""

    NUMERIC = "numeric"        # 标量事实：70%、1500元、200元
    ENTITY_REL = "entity_rel"  # 实体关系：A药禁止与B药合用
    CONDITION = "condition"    # 限制/时效：仅限2026年前、非直系亲属除外
    CATEGORICAL = "categorical"  # 分类/枚举：三级甲等、必须、禁止


class AtomicFact(BaseModel):
    """通用原子事实：主体-谓词-客体三元组，支持多类型校验。

    与 ExtractedFact 的关系：ExtractedFact 是简单的 key/value 对，
    适合数值型事实；AtomicFact 是结构化的三元组，适合跨类型事实校验。
    两者共存于 ExtractionPayload 中，向后兼容。
    """

    subject: str = Field(default="", description="事实主体，如'职工医保门诊'")
    predicate: str = Field(default="", description="谓词/属性，如'报销比例'")
    object_value: str = Field(default="", description="客体/值，如'70%'")
    fact_type: FactType = FactType.NUMERIC
    is_negative: bool = Field(default=False, description="是否包含否定/例外")
    source_text: str | None = Field(default=None, description="原文出处片段")


class EvalSample(BaseModel):
    """单条评测样本：问题 + 标准上下文 + 标准答案 + 必须命中的实体清单。

    这是评测链路的输入边界：后续评分器只依赖这份稳定契约，不直接读取数据集细节。
    """

    id: str = Field(..., min_length=1, description="样本唯一标识")
    question: str = Field(..., description="用户问题")
    ground_truth_context: str = Field(..., description="标准（金标准）检索上下文")
    ground_truth_answer: str = Field(..., description="标准答案")
    expected_entities: list[str] = Field(
        default_factory=list,
        description="答案中必须命中的实体（金额、比例、机构名、政策术语等）",
    )
    source_id: str | None = Field(default=None, description="金标准来源文档，用于检索溯源评测")
    required_entities: list[str] = Field(default_factory=list, description="必须全部命中的评测要点，独立于平均完整性")
    category: str = "standard"
    candidate_answer: str | None = Field(default=None, description="独立的对抗候选答案，不得覆盖金标准")
    mutation_type: str | None = None


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
    """单条样本的评测产出。

    保留实际上下文和答案，便于指标失败时回溯数据流，而不只留下一个分数。
    """

    sample_id: str
    actual_context: str = Field(..., description="RAG 实际检索到的上下文")
    actual_answer: str = Field(..., description="RAG 实际生成的答案")
    context_recall_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    faithfulness_pass: bool
    required_entities_pass: bool = True
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
        """按阈值快照判定单条结果是否通过（recall/completeness 达标且忠实）。

        阈值随汇总结果保存，确保报告复核时不会受运行后配置变化影响。
        """
        recall_gate = result.context_recall_score >= self.thresholds.get(
            "context_recall", 0.0
        )
        completeness_gate = result.completeness_score >= self.thresholds.get(
            "completeness", 0.0
        )
        return recall_gate and completeness_gate and result.faithfulness_pass and result.required_entities_pass

    def failed_results(self) -> list[EvalResult]:
        return [r for r in self.results if not self.is_case_passed(r)]

    def model_dump_json_safe(self) -> dict[str, Any]:
        """供日志/上报使用的可序列化视图。"""
        return self.model_dump(mode="json")
