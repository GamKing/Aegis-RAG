"""数据审计、来源命中、正确上下文对照及对抗检出基线。

python -m rag_eval.scissors_check --n 100 --output reports/baseline.json
退出码：0 全部门禁通过；1 质量不达标；2 参数错误。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from .data_generator import generate_dataset
from .evaluators import CompletenessEvaluator, FaithfulnessEvaluator, RequiredEntitiesEvaluator
from .models import EvalSample
from .pipeline import PipelineRAG
from .rag_impl import ProductionRAG
from .retrieval.chunker import RecursiveChunker
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.source_scope import requested_sources
from .runner import run_eval_pipeline


def audit_dataset(samples):
    """检查结构和标签一致性，不替代人工语义审查。"""
    issues = []
    questions = defaultdict(set)
    sources = defaultdict(set)
    ids = Counter(s.id for s in samples)
    for sid, count in ids.items():
        if count > 1:
            issues.append({"sample_id": sid, "code": "duplicate_id"})
    for s in samples:
        questions[s.question].add(s.ground_truth_context)
        if not s.source_id:
            issues.append({"sample_id": s.id, "code": "missing_source"})
        else:
            sources[s.source_id].add(s.ground_truth_context)
        if not s.question.strip() or not s.ground_truth_context.strip() or not s.ground_truth_answer.strip():
            issues.append({"sample_id": s.id, "code": "empty_required_text"})
        comp = CompletenessEvaluator(threshold=1.0).evaluate(s, s.ground_truth_context, s.ground_truth_answer)
        faith = FaithfulnessEvaluator().evaluate(s, s.ground_truth_context, s.ground_truth_answer)
        required = RequiredEntitiesEvaluator().evaluate(s, s.ground_truth_context, s.ground_truth_answer)
        if not comp.passed or not faith.passed or not required.passed:
            issues.append({"sample_id": s.id, "code": "invalid_gold", "details": comp.details + faith.details + required.details})
        if s.category == "adversarial" and (not s.candidate_answer or s.candidate_answer == s.ground_truth_answer):
            issues.append({"sample_id": s.id, "code": "ineffective_mutation"})
    for question, contexts in questions.items():
        if len(contexts) > 1:
            issues.append({"code": "ambiguous_question", "question": question, "versions": len(contexts)})
    for source_id, contexts in sources.items():
        if len(contexts) > 1:
            issues.append({"code": "conflicting_source", "source_id": source_id})
    return issues


def build_index(samples):
    """同源文档只建库一次，避免重复用例占据 Top-K。"""
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=50)
    chunks = []
    documents = {}
    for s in samples:
        source = s.source_id or s.id
        if source in documents and documents[source] != s.ground_truth_context:
            raise ValueError(f"来源文档内容冲突: {source}")
        documents[source] = s.ground_truth_context
    for source, context in documents.items():
        document_chunks = chunker.split_text(context, doc_id=source)
        scopes = requested_sources(context)
        if len(scopes) == 1:
            for chunk in document_chunks:
                chunk.metadata["scope_id"] = scopes[0]
        chunks.extend(document_chunks)
    retriever = HybridRetriever()
    retriever.index_documents(chunks)
    return retriever, len(chunks)


def classify_failure(data_invalid, source_hit, oracle_passed, actual_passed, gold_supported):
    if data_invalid:
        return "data"
    if not source_hit:
        return "retrieval_source_miss"
    if not oracle_passed:
        return "generation_with_gold_context"
    if not gold_supported:
        return "context_contamination"
    if not actual_passed:
        return "generation_with_retrieved_context"
    return "pass"


def evaluate_baseline(samples, top_k=3):
    if not samples:
        raise ValueError("评测集不能为空")
    if top_k < 1:
        raise ValueError("top_k 必须为正整数")
    issues = audit_dataset(samples)
    # 数据审计失败时禁止继续输出质量结论。
    if issues:
        return {"schema_version": 2, "total": len(samples), "data_issues": issues, "gate_passed": False}
    retriever, n_chunks = build_index(samples)
    actual = run_eval_pipeline(samples, ProductionRAG(retriever, top_k=top_k))
    oracle = run_eval_pipeline(samples, PipelineRAG(context_provider=lambda s: s.ground_truth_context))
    rows = []
    for s, result, control in zip(samples, actual.results, oracle.results):
        hits = retriever.retrieve(s.question, top_k=top_k)
        sources = [h.chunk.metadata.get("doc_id") for h in hits]
        hit = s.source_id in sources
        # 同时检查实际上下文与金标准文档的数值支持，暴露其他版本数值混入。
        supported = FaithfulnessEvaluator().evaluate(s, s.ground_truth_context, result.actual_answer)
        category = classify_failure(False, hit, oracle.is_case_passed(control),
                                    actual.is_case_passed(result), supported.passed)
        rows.append({
            "sample_id": s.id, "category": s.category, "question": s.question,
            "expected_source": s.source_id, "retrieved_sources": sources,
            "source_hit": hit, "oracle_passed": oracle.is_case_passed(control),
            "metric_passed": actual.is_case_passed(result),
            "gold_numeric_supported": supported.passed,
            "classification": category,
            "actual": result.model_dump(mode="json"),
            "oracle": control.model_dump(mode="json"),
            "gold_support_details": supported.details,
        })
    adversarial = []
    for s in samples:
        if s.candidate_answer is None:
            continue
        comp = CompletenessEvaluator(threshold=0.8).evaluate(s, s.ground_truth_context, s.candidate_answer)
        faith = FaithfulnessEvaluator().evaluate(s, s.ground_truth_context, s.candidate_answer)
        required = RequiredEntitiesEvaluator().evaluate(s, s.ground_truth_context, s.candidate_answer)
        adversarial.append({"sample_id": s.id, "mutation_type": s.mutation_type,
                            "detected": not (comp.passed and faith.passed and required.passed),
                            "legacy_detected": not (comp.passed and faith.passed),
                            "required_entities_pass": required.passed,
                            "completeness": comp.score, "faithfulness": faith.passed,
                            "candidate_answer": s.candidate_answer,
                            "details": comp.details + faith.details + required.details})
    counts = dict(Counter(r["classification"] for r in rows))
    detected = sum(r["detected"] for r in adversarial)
    return {
        "schema_version": 3, "total": len(samples), "top_k": top_k,
        "indexed_chunks": n_chunks, "data_issues": issues,
        "categories": dict(Counter(s.category for s in samples)),
        "metrics": {"context_recall": actual.avg_context_recall,
                    "completeness": actual.avg_completeness,
                    "faithfulness": actual.faithfulness_pass_rate,
                    "case_pass_rate": actual.case_pass_rate,
                    "source_hit_rate": sum(r["source_hit"] for r in rows) / len(rows),
                    "oracle_case_pass_rate": oracle.case_pass_rate,
                    "grounded_case_pass_rate": counts.get("pass", 0) / len(rows)},
        "classifications": counts,
        "adversarial_total": len(adversarial), "adversarial_detected": detected,
        "gate_passed": counts.get("pass", 0) == len(rows) and detected == len(adversarial),
        "limitations": ["合成政策版本检索，不代表真实业务泛化质量。",
                        "忠实性仅检查数值出处，不证明事实归属、非数值语义或答案相关性。",
                        "失败分类是诊断线索，仍需逐例检查混入文档与生成选择。"],
        "cases": rows, "adversarial": adversarial,
    }


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    parser = argparse.ArgumentParser(description="数据审计与可信质量基线")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--dataset", type=Path, help="读取固定 JSON 数据集，默认确定性生成")
    parser.add_argument("--output", type=Path, help="保存完整 JSON 报告")
    args = parser.parse_args(argv)
    samples = ([EvalSample.model_validate(s) for s in json.loads(args.dataset.read_text(encoding="utf-8"))]
               if args.dataset else generate_dataset(seed=42))
    if not 1 <= args.n <= len(samples) or args.top_k < 1:
        parser.error(f"n 必须介于 1 和 {len(samples)}；top-k 必须为正整数")
    report = evaluate_baseline(samples[:args.n], top_k=args.top_k)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key not in ("cases", "adversarial")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
