package com.gamking.aegisrag.report;

import com.gamking.aegisrag.model.EvalResult;
import com.gamking.aegisrag.model.EvalSummary;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** 结构化 ASCII 报表渲染器。 */
public class AsciiReportRenderer {

    private static final String DIV = "=".repeat(88);

    public String render(EvalSummary summary, Map<String, String> questions) {
        List<String> lines = new ArrayList<>();
        lines.add(DIV);
        lines.add("离线 RAG 评测报告 / OFFLINE RAG EVALUATION REPORT");
        lines.add(String.format("样本总数: %d | 通过: %d | 失败: %d",
                summary.total(), summary.passed(), summary.failed()));
        lines.add(String.format("通过阈值: ContextRecall >= %.2f | Completeness >= %.2f | Faithfulness == PASS",
                summary.contextRecallThreshold(), summary.completenessThreshold()));
        lines.add(DIV);
        lines.add("");
        lines.add("[1] 逐样本明细");
        lines.add(renderTable(summary, questions));
        lines.add("");
        lines.add("[2] 整体聚合指标");
        lines.add(String.format("  平均 Context Recall : %.3f", summary.avgContextRecall()));
        lines.add(String.format("  平均 Completeness   : %.3f", summary.avgCompleteness()));
        lines.add(String.format("  Faithfulness 通过率 : %.1f%%", summary.faithfulnessPassRate() * 100));
        lines.add(String.format("  用例整体通过率      : %d/%d (%.1f%%)",
                summary.passed(), summary.total(), summary.casePassRate() * 100));
        lines.add("");
        lines.add("[3] 失败案例与归因");

        if (summary.failedResults().isEmpty()) {
            lines.add("  (无 —— 全部样本通过全部门槛)");
        } else {
            for (EvalResult result : summary.failedResults()) {
                lines.add(String.format("  [FAIL] %s | recall=%.3f | completeness=%.3f | faithfulness=%s",
                        result.sampleId(), result.contextRecallScore(), result.completenessScore(),
                        result.faithfulnessPass() ? "PASS" : "FAIL"));
                result.errorDetails().forEach(detail -> lines.add("      - " + detail));
            }
        }
        lines.add("");
        lines.add(DIV);
        return String.join("\n", lines);
    }

    private String renderTable(EvalSummary summary, Map<String, String> questions) {
        StringBuilder sb = new StringBuilder();
        sb.append("+-------------+-----------------------+--------+----------+-------+---------+\n");
        sb.append("| CASE ID     | 问题(截断)            | RECALL | COMPLETE | FAITH | VERDICT |\n");
        sb.append("+-------------+-----------------------+--------+----------+-------+---------+\n");
        for (EvalResult result : summary.results()) {
            String question = TextNormalizer.truncateDisplay(
                    questions.getOrDefault(result.sampleId(), "-"), 22);
            String verdict = summary.isCasePassed(result) ? "PASS" : "FAIL";
            sb.append(String.format("| %-11s | %-21s | %6.3f | %8.3f | %5s | %7s |\n",
                    result.sampleId(), question, result.contextRecallScore(),
                    result.completenessScore(), result.faithfulnessPass() ? "PASS" : "FAIL", verdict));
        }
        sb.append("+-------------+-----------------------+--------+----------+-------+---------+");
        return sb.toString();
    }
}
