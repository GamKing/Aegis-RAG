package com.gamking.aegisrag.nli;

/** NLI 裁判接口。实现类负责判断一个 claim 是否可以被 context 蕴含。 */
public interface NliJudge {

    /**
     * 判断 claim 是否由 context 严格推导（蕴含）。
     *
     * @param context 参考材料（前提）
     * @param claim   待核实主张（假设）
     * @return true 表示蕴含，false 表示矛盾或中立
     */
    boolean checkFaithfulness(String context, String claim);
}