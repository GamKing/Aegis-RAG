"""结构化 ASCII 报表渲染（CJK 宽度感知，含中文时列仍然对齐）。"""
from __future__ import annotations

from typing import Mapping, Optional

from . import text_utils as tu
from .models import EvalSummary

_DIV = "=" * 88


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_table(headers, rows, aligns) -> str:
    """渲染带边框的 ASCII 表格；列宽按显示宽度（中文占 2 列）计算。"""
    n = len(headers)
    widths = [tu.display_width(h) for h in headers]
    for row in rows:
        for i in range(n):
            widths[i] = max(widths[i], tu.display_width(row[i]))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(cells) -> str:
        padded = [
            tu.pad_display(c, widths[i], aligns[i]) for i, c in enumerate(cells)
        ]
        return "|" + "|".join(f" {p} " for p in padded) + "|"

    lines = [sep, fmt(headers), sep]
    lines.extend(fmt(row) for row in rows)
    lines.append(sep)
    return "\n".join(lines)


def render_report(summary: EvalSummary, questions: Optional[Mapping] = None) -> str:
    """渲染完整评测报表：逐样本明细 -> 聚合指标 -> 失败归因。"""
    questions = questions or {}
    lines: list = []

    lines.append(_DIV)
    lines.append("离线 RAG 评测报告 / OFFLINE RAG EVALUATION REPORT")
    lines.append(
        f"运行时间: {summary.generated_at:%Y-%m-%d %H:%M:%S} | "
        f"样本总数: {summary.total} | 通过: {summary.passed} | 失败: {summary.failed}"
    )
    t = summary.thresholds
    lines.append(
        f"通过阈值: ContextRecall >= {t.get('context_recall', 0.0):.2f} | "
        f"Completeness >= {t.get('completeness', 0.0):.2f} | "
        f"Faithfulness == PASS"
    )
    lines.append(_DIV)
    lines.append("")
    lines.append("额外硬门槛: 已声明的必答项必须全部命中（REQUIRED == PASS）")

    # [1] 逐样本明细 ----------------------------------------------------------
    lines.append("[1] 逐样本明细")
    headers = ["CASE ID", "问题(截断)", "RECALL", "COMPLETE", "FAITH", "REQUIRED", "VERDICT"]
    aligns = ["left", "left", "right", "right", "center", "center", "center"]
    rows = []
    for r in summary.results:
        rows.append(
            [
                r.sample_id,
                tu.truncate_display(questions.get(r.sample_id, "-"), 22),
                _fmt(r.context_recall_score),
                _fmt(r.completeness_score),
                "PASS" if r.faithfulness_pass else "FAIL",
                "PASS" if r.required_entities_pass else "FAIL",
                "PASS" if summary.is_case_passed(r) else "FAIL",
            ]
        )
    lines.append(render_table(headers, rows, aligns))
    lines.append("")

    # [2] 整体聚合指标 --------------------------------------------------------
    lines.append("[2] 整体聚合指标")
    lines.append(f"  平均 Context Recall : {_fmt(summary.avg_context_recall)}")
    lines.append(f"  平均 Completeness   : {_fmt(summary.avg_completeness)}")
    lines.append(
        f"  Faithfulness 通过率 : {summary.faithfulness_pass_rate:.1%}"
    )
    lines.append(
        f"  用例整体通过率      : {summary.passed}/{summary.total} "
        f"({summary.case_pass_rate:.1%})"
    )
    lines.append("")

    # [3] 失败案例与归因 ------------------------------------------------------
    lines.append("[3] 失败案例与归因")
    failed = summary.failed_results()
    if not failed:
        lines.append("  (无 —— 全部样本通过全部门槛)")
    for r in failed:
        lines.append(
            f"  [FAIL] {r.sample_id} | recall={_fmt(r.context_recall_score)} | "
            f"completeness={_fmt(r.completeness_score)} | "
            f"faithfulness={'PASS' if r.faithfulness_pass else 'FAIL'}"
        )
        for detail in r.error_details:
            lines.append(f"      - {detail}")
        # 指标剪刀差提示：检索覆盖达标但答案数值无出处 -> 指向生成端问题
        if (
            not r.faithfulness_pass
            and r.context_recall_score >= t.get("context_recall", 0.0)
        ):
            lines.append(
                "      * 剪刀差警示: 检索覆盖达标但答案数值无出处 -> "
                "生成端幻觉/数字篡改"
            )
    lines.append("")
    lines.append(_DIV)
    return "\n".join(lines)
