"""演示入口（组合根）：装配内置数据集与 RAG 后端，运行评测并打印报表。

支持两种模式（--mode）：
- demo（默认）：使用 Dummy RAG 预设，刻意构造四种失败模式，演示三类指标的
  「剪刀差」——见下方各预设的失败动机；
- retrieval：使用真实检索模块（BM25/混合 + RRF）建库召回，接受控生成流水线，
  验证从 Query 到 actual_context 再到答案的整条链路。
"""
from __future__ import annotations

import sys

from .dataset import SAMPLE_001, SAMPLE_002, SAMPLE_003, SAMPLE_004, SAMPLE_005, load_samples
from .dummy_rag import DummyRAG, RAGResponse
from .report import render_report
from .runner import EvalConfig, run_eval_pipeline


def build_demo_rag() -> DummyRAG:
    """构造带预设的 Dummy RAG，覆盖完美链路 + 四种典型失败模式。"""
    presets = {
        # 完美链路：检索命中金标准上下文，答案完整且全部数值有出处 -> 三项全绿
        "med-ins-001": RAGResponse(
            context=SAMPLE_001.ground_truth_context,
            answer=SAMPLE_001.ground_truth_answer,
        ),
        # 检索缺失：丢掉三级医院条款与年度封顶线；生成端如实转述残缺上下文，
        # 没有编造 -> 忠实性通过，但召回/完整性双双塌陷
        "med-ins-002": RAGResponse(
            context=(
                "城乡居民基本医疗保险住院待遇标准：一级及以下医疗机构起付线"
                "300元，政策范围内费用报销比例85%；二级医疗机构起付线600元，"
                "报销比例70%。"
            ),
            answer=(
                "城乡居民医保住院：一级及以下医疗机构起付线300元，报销85%；"
                "二级医疗机构起付线600元，报销70%。三级医疗机构与年度封顶线的"
                "政策暂未查询到。"
            ),
        ),
        # 生成端信息损耗：上下文完整（召回 1.0），但答案漏掉困难群体倾斜条款，
        # 且未编造任何数字 -> 忠实性通过、完整性失败
        "med-ins-003": RAGResponse(
            context=SAMPLE_003.ground_truth_context,
            answer=(
                "大病保险：一个自然年度内，个人自付合规医疗费用累计超过起付线"
                "1.5万元的部分纳入支付范围，报销比例为60%，不设封顶线。"
            ),
        ),
        # 前置条件失守：检索丢掉例外条款，生成端给出「备案即可报销」的绝对化
        # 答案，未提及降比例惩罚与急诊视同备案
        "med-ins-004": RAGResponse(
            context=(
                "异地就医直接结算政策：参保人员办理异地就医备案后，在备案地"
                "开通联网结算的定点医疗机构住院，可直接结算，按参保地政策报销。"
            ),
            answer=(
                "可以，但需先办理异地就医备案：备案后在开通联网结算的定点医疗"
                "机构住院，可直接结算并按参保地政策正常报销。"
            ),
        ),
        # 对抗样本（数字篡改）：上下文与金标准一致（召回 1.0），非数值实体全部
        # 命中（完整性 1.0），但年度限额 1500元->2500元、报销比例 70%->85%，
        # 两个数值均无出处 -> 忠实性一票否决
        "med-ins-005": RAGResponse(
            context=SAMPLE_005.ground_truth_context,
            answer=(
                "高血压病（Ⅱ级及以上）属于门诊慢特病保障范围，统筹基金年度"
                "支付限额为每人每年2500元，政策范围内费用报销比例为85%；待遇"
                "认定须二级及以上定点医疗机构确诊，并经医保经办机构审核。"
            ),
        ),
    }
    return DummyRAG(presets)


def main(argv: list | None = None) -> int:
    # Windows 旧控制台可能是 GBK 编码，强制 UTF-8 输出避免报表乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    argv = list(sys.argv[1:] if argv is None else argv)
    mode = "demo"
    if "--mode" in argv:
        idx = argv.index("--mode")
        if idx + 1 < len(argv):
            mode = argv[idx + 1]

    # demo 模式用 Dummy 预设演示指标剪刀差；retrieval 模式走真实检索链路。
    if mode == "retrieval":
        from .run_retrieval_eval import run_retrieval_eval

        summary = run_retrieval_eval(config=EvalConfig())
    else:
        samples = load_samples()
        rag = build_demo_rag()
        summary = run_eval_pipeline(samples, rag, config=EvalConfig())

    questions = {s.id: s.question for s in load_samples()}
    print(render_report(summary, questions=questions))

    if summary.failed:
        print(
            f"\n结论: {summary.failed}/{summary.total} 个样本未达标，"
            f"失败原因见上方归因区。"
        )
        return 1
    print(f"\n结论: 全部 {summary.total} 个样本通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
