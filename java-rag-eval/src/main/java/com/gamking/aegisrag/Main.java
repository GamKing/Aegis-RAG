package com.gamking.aegisrag;

import com.gamking.aegisrag.dataset.DemoDataset;
import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.EvalSummary;
import com.gamking.aegisrag.rag.DummyRag;
import com.gamking.aegisrag.report.AsciiReportRenderer;
import com.gamking.aegisrag.runner.EvalConfig;
import com.gamking.aegisrag.runner.EvalRunner;

import java.util.Map;
import java.util.stream.Collectors;

/** Java 17 演示入口。 */
public final class Main {
    private Main() {}

    public static void main(String[] args) {
        var samples = DemoDataset.samples();
        var rag = new DummyRag(DemoDataset.demoPresets());
        EvalSummary summary = EvalRunner.run(samples, rag, EvalConfig.defaults());
        Map<String, String> questions = samples.stream()
                .collect(Collectors.toMap(EvalSample::id, EvalSample::question));

        System.out.println(new AsciiReportRenderer().render(summary, questions));
        if (summary.failed() > 0) {
            System.out.printf("%n结论: %d/%d 个样本未达标。%n", summary.failed(), summary.total());
            System.exit(1);
        }
        System.out.printf("%n结论: 全部 %d 个样本通过。%n", summary.total());
    }
}
