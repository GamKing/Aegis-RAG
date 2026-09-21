"""rag_eval —— 离线 RAG 评测套件（基础版）。

快速上手：
    from rag_eval import load_samples, run_eval_pipeline, render_report
    summary = run_eval_pipeline(load_samples(), my_rag)
    print(render_report(summary))
"""
from .dataset import load_samples
from .dummy_rag import BaseRAG, DummyRAG, RAGResponse
from .evaluators import (
    BaseEvaluator,
    CompletenessEvaluator,
    ContextRecallEvaluator,
    FaithfulnessEvaluator,
    MetricOutcome,
)
from .models import (
    EvalResult,
    EvalSample,
    EvalSummary,
    ExtractedFact,
    ExtractionPayload,
    PipelineTrace,
    VerificationFinding,
    VerificationResult,
)
from .nli_judge import NLIJudge
from .pipeline import (
    ControlledSynthesizer,
    NLIAndSelfCorrector,
    PipelineRAG,
    PipelineResult,
    RuleBasedFactExtractor,
    Tier1CodeVerifier,
    run_controlled_pipeline,
)
from .report import render_report
from .runner import EvalConfig, build_default_evaluators, run_eval_pipeline

__version__ = "0.1.0"

__all__ = [
    "BaseEvaluator",
    "BaseRAG",
    "CompletenessEvaluator",
    "ContextRecallEvaluator",
    "DummyRAG",
    "EvalConfig",
    "EvalResult",
    "EvalSample",
    "EvalSummary",
    "FaithfulnessEvaluator",
    "MetricOutcome",
    "NLIJudge",
    "ControlledSynthesizer",
    "NLIAndSelfCorrector",
    "PipelineRAG",
    "PipelineResult",
    "RuleBasedFactExtractor",
    "Tier1CodeVerifier",
    "run_controlled_pipeline",
    "RAGResponse",
    "build_default_evaluators",
    "load_samples",
    "render_report",
    "run_eval_pipeline",
]
