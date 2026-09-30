"""检索模块闭环验收：切块建库 -> 混合检索 -> 全链路评测。

既可作为独立脚本运行，也可供 main.py 以 --mode retrieval 调度。
"""
from __future__ import annotations

from .dataset import load_samples
from .evaluators import ContextRecallEvaluator
from .rag_impl import ProductionRAG
from .retrieval.chunker import RecursiveChunker
from .retrieval.hybrid_retriever import HybridRetriever
from .runner import EvalConfig, run_eval_pipeline
from .models import EvalSummary


def build_index(chunk_size: int = 300, chunk_overlap: int = 50):
    """将全部样本的标准上下文切块建库，返回 (样本集, 检索器, 块清单)。"""
    samples = load_samples()
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    for s in samples:
        all_chunks.extend(chunker.split_text(s.ground_truth_context, doc_id=s.id))

    retriever = HybridRetriever()
    retriever.index_documents(all_chunks)
    return samples, retriever, all_chunks


def build_production_rag(
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    top_k: int = 3,
) -> ProductionRAG:
    """装配真实检索模块 + 受控生成流水线，作为可评测的 RAG 后端。"""
    _, retriever, _ = build_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return ProductionRAG(retriever, top_k=top_k)


def run_retrieval_eval(
    config: EvalConfig | None = None,
    top_k: int = 3,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> EvalSummary:
    """跑完整评测链路（召回 + 完整性 + 忠实性），返回聚合摘要。"""
    samples = load_samples()
    rag = build_production_rag(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k
    )
    return run_eval_pipeline(samples, rag, config=config or EvalConfig())


def test_retrieval_quality(top_k: int = 2, threshold: float = 0.80) -> None:
    """单项召回率验收：只看 ContextRecallEvaluator 的分数。"""
    samples, retriever, all_chunks = build_index()
    recall_evaluator = ContextRecallEvaluator(threshold=threshold)

    print("\n" + "=" * 60)
    print("检索模块 (Retrieval) 单项召回率验收")
    print(f"库内块数: {len(all_chunks)} | top_k: {top_k} | 阈值: {threshold}")
    print("=" * 60)

    for s in samples:
        actual_context = retriever.retrieve_context(s.question, top_k=top_k)
        outcome = recall_evaluator.evaluate(
            s, actual_context=actual_context, actual_answer=""
        )
        status = "[PASS]" if outcome.passed else "[FAIL]"
        print(f"{status} 样本: {s.id} | 召回得分: {outcome.score:.4f}")
        if not outcome.passed:
            for d in outcome.details:
                print(f"       {d}")
    print("=" * 60)


if __name__ == "__main__":
    test_retrieval_quality()