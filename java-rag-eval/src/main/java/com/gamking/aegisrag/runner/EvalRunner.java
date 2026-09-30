package com.gamking.aegisrag.runner;

import com.gamking.aegisrag.evaluator.CompletenessEvaluator;
import com.gamking.aegisrag.evaluator.ContextRecallEvaluator;
import com.gamking.aegisrag.evaluator.Evaluator;
import com.gamking.aegisrag.evaluator.FaithfulnessEvaluator;
import com.gamking.aegisrag.model.*;
import com.gamking.aegisrag.nli.NliJudge;
import com.gamking.aegisrag.rag.BaseRag;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 评测主流程：批量执行、单样本判定、异常隔离、聚合。
 */
public class EvalRunner {

    /**
     * 批量评测入口。
     */
    public static EvalSummary run(
            List<EvalSample> samples,
            BaseRag rag,
            EvalConfig config,
            NliJudge nliJudge
    ) {
        config = config == null ? EvalConfig.defaults() : config;

        ContextRecallEvaluator recall = new ContextRecallEvaluator(config.contextRecallThreshold());
        CompletenessEvaluator completeness = new CompletenessEvaluator(config.completenessThreshold());
        FaithfulnessEvaluator faithfulness = nliJudge != null
                ? new FaithfulnessEvaluator(nliJudge)
                : new FaithfulnessEvaluator();

        List<EvalResult> results = new ArrayList<>();
        for (EvalSample sample : samples) {
            results.add(evaluateSample(sample, rag, recall, completeness, faithfulness));
        }

        int total = results.size();
        if (total == 0) {
            return new EvalSummary(LocalDateTime.now(), 0, 0, 0,
                    0.0, 0.0, 0.0, 0.0,
                    config.contextRecallThreshold(), config.completenessThreshold(),
                    List.of());
        }

        int passed = 0;
        double recallSum = 0, completenessSum = 0, faithSum = 0;
        for (EvalResult r : results) {
            if (isCasePassed(r, config)) passed++;
            recallSum += r.contextRecallScore();
            completenessSum += r.completenessScore();
            if (r.faithfulnessPass()) faithSum++;
        }

        return new EvalSummary(
                LocalDateTime.now(),
                total, passed, total - passed,
                round(recallSum / total),
                round(completenessSum / total),
                round(faithSum / total),
                round((double) passed / total),
                config.contextRecallThreshold(), config.completenessThreshold(),
                results);
    }

    /** 兼容无 NLI 的调用。 */
    public static EvalSummary run(List<EvalSample> samples, BaseRag rag, EvalConfig config) {
        return run(samples, rag, config, null);
    }

    private static EvalResult evaluateSample(
            EvalSample sample,
            BaseRag rag,
            ContextRecallEvaluator recall,
            CompletenessEvaluator completeness,
            FaithfulnessEvaluator faithfulness
    ) {
        RagResponse response;
        try {
            response = rag.retrieveAndGenerate(sample);
        } catch (Exception e) {
            return new EvalResult(sample.id(), "", "",
                    0.0, 0.0, false,
                    List.of("[harness] RAG 链路异常: " + e));
        }

        MetricOutcome recallOutcome = recall.evaluate(sample, response.context(), response.answer());
        MetricOutcome completenessOutcome = completeness.evaluate(sample, response.context(), response.answer());
        MetricOutcome faithfulnessOutcome = faithfulness.evaluate(sample, response.context(), response.answer());

        List<String> errors = new ArrayList<>();
        if (!recallOutcome.passed()) errors.addAll(recallOutcome.details());
        if (!completenessOutcome.passed()) errors.addAll(completenessOutcome.details());
        if (!faithfulnessOutcome.passed()) errors.addAll(faithfulnessOutcome.details());

        return new EvalResult(
                sample.id(),
                response.context(),
                response.answer(),
                recallOutcome.score() != null ? recallOutcome.score() : 0.0,
                completenessOutcome.score() != null ? completenessOutcome.score() : 0.0,
                faithfulnessOutcome.passed(),
                errors);
    }

    private static boolean isCasePassed(EvalResult r, EvalConfig config) {
        return r.contextRecallScore() >= config.contextRecallThreshold()
                && r.completenessScore() >= config.completenessThreshold()
                && r.faithfulnessPass();
    }

    private static double round(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }
}