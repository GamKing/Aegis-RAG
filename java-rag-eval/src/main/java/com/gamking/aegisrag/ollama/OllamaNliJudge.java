package com.gamking.aegisrag.ollama;

import com.gamking.aegisrag.nli.NliJudge;

/**
 * 基于本地 Ollama 的 NLI 裁判。
 * 
 * 让模型只做单选：A=蕴含 / B=矛盾 / C=中立。只有 A 才算通过质检。
 * 非法输出保守转为 C（中立），确保不绕过代码质检。
 */
public class OllamaNliJudge implements NliJudge {

    private final OllamaClient client;
    private final double temperature;

    public OllamaNliJudge(OllamaClient client) {
        this(client, 0.0);
    }

    public OllamaNliJudge(OllamaClient client, double temperature) {
        this.client = client;
        this.temperature = temperature;
    }

    /**
     * 对单个 claim 做 A/B/C 分类。
     *
     * @return "A" 蕴含 / "B" 矛盾 / "C" 中立
     */
    public String classify(String context, String claim) {
        String prompt = buildPrompt(context, claim);
        String verdict = "";
        try {
            verdict = client.chat(prompt, temperature, 1);
        } catch (Exception e) {
            // 调用异常时保守返回 C（中立），不输出未经验证的事实
            return "C";
        }
        String upper = verdict.trim().toUpperCase();
        if (upper.startsWith("A")) return "A";
        if (upper.startsWith("B")) return "B";
        return "C";
    }

    @Override
    public boolean checkFaithfulness(String context, String claim) {
        return classify(context, claim).equals("A");
    }

    private static String buildPrompt(String context, String claim) {
        return "你是一个严肃的逻辑事实裁判。判断【待核实主张】是否能够由【参考材料】严格推导出来。\n\n"
                + "【参考材料】: " + context + "\n"
                + "【待核实主张】: " + claim + "\n\n"
                + "请仅输出一个选项字母：\n"
                + "A. 蕴含（主张的事实完全源自材料，或属于合理直接推导）\n"
                + "B. 矛盾（主张与材料明确冲突）\n"
                + "C. 中立（材料未提及，属于主张自己脑补的信息）\n\n"
                + "你的选择是 (仅回复 A、B 或 C):";
    }
}