package com.gamking.aegisrag.verification;

import com.gamking.aegisrag.model.AtomicFact;
import com.gamking.aegisrag.model.VerificationResult;

/**
 * 校验规则接口：所有规则实现 verify 方法。
 */
public interface VerificationRule {

    /**
     * 校验单个事实是否在上下文中有出处。
     *
     * @param fact 待校验的事实
     * @param context 上下文文本
     * @return 校验结果
     */
    VerificationResult verify(AtomicFact fact, String context);
}