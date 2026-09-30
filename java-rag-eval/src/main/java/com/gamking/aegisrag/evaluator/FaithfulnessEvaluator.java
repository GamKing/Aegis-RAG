package com.gamking.aegisrag.evaluator;

import com.gamking.aegisrag.model.EvalSample;
import com.gamking.aegisrag.model.MetricOutcome;
import com.gamking.aegisrag.nli.NliJudge;
import com.gamking.aegisrag.text.TextNormalizer;

import java.util.ArrayList;
import java.util.List;

/**
 * 数值忠实性：答案中的每个数值/百分比都必须能在检索上下文中找到出处。
 * 
 * 纯代码断言，一票否决制：
 * - 提取答案的全部数值单元（含是否百分比标记）；
 * - 逐一在上下文中做边界敏感匹配；
 * - 只要存在一个无出处数值 => passed=False。
 */
public class FaithfulnessEvaluator implements Evaluator {

    private final NliJudge nliJudge;

    public FaithfulnessEvaluator() {
        this(null);
    }

    public FaithfulnessEvaluator(NliJudge nliJudge) {
        this.nliJudge = nliJudge;
    }

    @Override
    public MetricOutcome evaluate(EvalSample sample, String actualContext, String actualAnswer) {
        List<TextNormalizer.NumberUnit> numbers = TextNormalizer.extractNumbers(actualAnswer);
        List<TextNormalizer.NumberUnit> unsupported = new ArrayList<>();

        for (TextNormalizer.NumberUnit unit : numbers) {
            if (!TextNormalizer.numberSupportedInContext(unit.value(), unit.isPercent(), actualContext)) {
                unsupported.add(unit);
            }
        }

        boolean passed = unsupported.isEmpty();

        // 如果有未支持数值且注入了 NLI judge，尝试语义复核
        if (!passed && nliJudge != null) {
            try {
                boolean nliPass = nliJudge.checkFaithfulness(actualContext, actualAnswer);
                if (nliPass) {
                    passed = true;
                    unsupported.clear();
                }
            } catch (Exception e) {
                // NLI 异常时保守判失败
            }
        }

        List<String> details = new ArrayList<>();
        if (!passed) {
            for (TextNormalizer.NumberUnit unit : unsupported) {
                String raw = unit.value() + (unit.isPercent() ? "%" : "");
                details.add(String.format("[faithfulness] 答案数值 「%s」 在检索上下文中无出处（数值幻觉/篡改嫌疑），一票否决", raw));
            }
        }

        return passed ? MetricOutcome.pass("faithfulness", null)
                      : MetricOutcome.fail("faithfulness", null, details);
    }
}
