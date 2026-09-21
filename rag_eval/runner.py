"""评测主流程：批量执行 -> 单样本判定 -> 聚合。

工程约定：
- 单个评分器异常只污染该指标（缺失指标记 0 分 / 忠实性记失败），不拖垮整批；
- RAG 链路级异常将该样本记为全指标失败并写入 error_details；
- 聚合口径：recall/completeness 取算术平均，faithfulness 取通过率。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

from .dummy_rag import BaseRAG
from .evaluators import (
    BaseEvaluator,
    CompletenessEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
)
from .models import EvalResult, EvalSample, EvalSummary

logger = logging.getLogger("rag_eval.runner")


@dataclass(frozen=True)
class EvalConfig:
    """通过门槛配置。faithfulness 为布尔指标，恒为必须通过。"""

    context_recall_threshold: float = 0.80
    completeness_threshold: float = 0.80

    def as_dict(self) -> dict:
        return {
            "context_recall": self.context_recall_threshold,
            "completeness": self.completeness_threshold,
        }


def build_default_evaluators(config: EvalConfig) -> list:
    """默认评分器组合：召回 + 完整性 + 忠实性。"""
    return [
        ContextRecallEvaluator(threshold=config.context_recall_threshold),
        CompletenessEvaluator(threshold=config.completeness_threshold),
        FaithfulnessEvaluator(),
    ]


def _case_passed(result: EvalResult, config: EvalConfig) -> bool:
    return (
        result.context_recall_score >= config.context_recall_threshold
        and result.completeness_score >= config.completeness_threshold
        and result.faithfulness_pass
    )


def evaluate_sample(
    sample: EvalSample,
    rag: BaseRAG,
    evaluators: Sequence[BaseEvaluator],
    config: EvalConfig,
) -> EvalResult:
    """跑通单个样本：调用 RAG -> 依次执行评分器 -> 汇总为 EvalResult。"""
    response = rag.retrieve_and_generate(sample)

    error_details: list = []
    scores: dict = {}
    faithfulness_pass = True

    for evaluator in evaluators:
        try:
            outcome = evaluator.evaluate(sample, response.context, response.answer)
        except Exception as exc:  # noqa: BLE001 —— 单评分器故障需要被记录而非中断
            logger.exception(
                "评测器 %s 在样本 %s 上执行异常", evaluator.name, sample.id
            )
            error_details.append(
                f"[harness] 评测器 {evaluator.name} 执行异常: {exc!r}"
            )
            if evaluator.name == "faithfulness":
                faithfulness_pass = False  # 忠实性不可判定时保守判失败
            else:
                scores[evaluator.name] = 0.0
            continue

        if evaluator.name == "faithfulness":
            faithfulness_pass = outcome.passed
        else:
            scores[evaluator.name] = (
                outcome.score if outcome.score is not None else 0.0
            )
        if not outcome.passed:
            error_details.extend(outcome.details)

    return EvalResult(
        sample_id=sample.id,
        actual_context=response.context,
        actual_answer=response.answer,
        context_recall_score=round(scores.get("context_recall", 0.0), 4),
        completeness_score=round(scores.get("completeness", 0.0), 4),
        faithfulness_pass=faithfulness_pass,
        error_details=error_details,
    )


def run_eval_pipeline(
    samples: Sequence[EvalSample],
    rag: BaseRAG,
    config: Optional[EvalConfig] = None,
    evaluators: Optional[Sequence[BaseEvaluator]] = None,
) -> EvalSummary:
    """批量评测入口。

    参数：
        samples:    评测样本集；
        rag:        被测 RAG 后端（实现 BaseRAG 即可，通常是 Dummy 或真实系统的适配器）；
        config:     通过门槛，缺省为 recall/completeness 均 >= 0.80；
        evaluators: 自定义评分器组合，缺省为默认三件套。

    返回：
        EvalSummary：逐样本结果 + 整体平均指标 + 通过率。
    """
    config = config or EvalConfig()
    if evaluators is None:
        evaluators = build_default_evaluators(config)

    results: list = []
    for sample in samples:
        try:
            results.append(evaluate_sample(sample, rag, evaluators, config))
        except Exception as exc:  # noqa: BLE001 —— 链路级异常记为全指标失败
            logger.exception("样本 %s 评测链路异常", sample.id)
            results.append(
                EvalResult(
                    sample_id=sample.id,
                    actual_context="",
                    actual_answer="",
                    context_recall_score=0.0,
                    completeness_score=0.0,
                    faithfulness_pass=False,
                    error_details=[f"[harness] 样本评测链路异常: {exc!r}"],
                )
            )

    total = len(results)
    if total == 0:
        return EvalSummary(
            generated_at=datetime.now(),
            total=0,
            passed=0,
            failed=0,
            avg_context_recall=0.0,
            avg_completeness=0.0,
            faithfulness_pass_rate=0.0,
            case_pass_rate=0.0,
            thresholds=config.as_dict(),
            results=[],
        )

    passed = sum(1 for r in results if _case_passed(r, config))
    avg_recall = sum(r.context_recall_score for r in results) / total
    avg_completeness = sum(r.completeness_score for r in results) / total
    faith_rate = sum(1 for r in results if r.faithfulness_pass) / total

    return EvalSummary(
        generated_at=datetime.now(),
        total=total,
        passed=passed,
        failed=total - passed,
        avg_context_recall=round(avg_recall, 4),
        avg_completeness=round(avg_completeness, 4),
        faithfulness_pass_rate=round(faith_rate, 4),
        case_pass_rate=round(passed / total, 4),
        thresholds=config.as_dict(),
        results=results,
    )
