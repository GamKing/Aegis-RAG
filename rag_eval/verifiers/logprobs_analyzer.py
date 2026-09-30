"""Logprobs 测谎仪（Day 5-6 机制 3）：底层概率可解释性幻觉检测。

在严肃生产环境，大模型可能"以极其笃定的语气胡说八道"。logprobs（对数
概率）是模型生成每个 Token 时的底层概率对数值：

    logprob = ln(P(token))

- 有把握的 token：P → 1.0，logprob → 0.0（接近 0）；
- 凭空捏造的 token：P 骤降（如 0.05），logprob ≈ -2.99。

两个核心指标：
1. 局部实体置信度（Token Minimum Confidence）：数字/核心实体各 Token 中
   最低概率。任一个低于阈值（如 0.40）即标记高风险幻觉。
2. 全文困惑度（Perplexity, PPL）：量化整段回答的混乱程度。PPL 越小越
   稳定自然（通常 1.0~3.0），偏大（如 >10）往往在拼凑虚假逻辑。

【架构定位与否决权设计原则】：
Logprobs 测谎仪反映的是大模型内在的不确定性（Token 概率分布），属于软性启发式特征，
不能作为单独的一票否决硬断言！
- 避免误杀（False Rejection）：如果某个关键数字在检索上下文（Context）中有确凿证据支持，
  即使生成该 Token 时概率较低（例如受到词表切分、前缀标点或同义词分布影响），也不能将其判定为编造；
- 协同裁决（Joint Verification）：测谎仪默认不拥有单独否决权（allow_sole_veto=False）。
  出具风险标记（Warning/has_risk）。只有在与上下文出处比对结合时，即「低置信度 + 上下文无出处」
  双重确凿时，才构成协同否决；若上下文中存在明确出处，则放行避免误杀；
- 保持可配置性：在特定强风控场景下，可通过显式指定 allow_sole_veto=True 开启单独立即否决。

【机制碰撞特别警示：Logit Masking 与 Logprobs 的冲突】：
现代工业界常引入基于 FSM 的 Logit Masking 强制锁定输出 JSON Schema。
然而，Logit Masking 会在 Softmax 计算前将所有非法词打分强制改写为 -inf。
分母抹杀导致：即使模型对某数字极其心虚（原生概率仅 0.03%），分母被强行抹除其他候选后，
Softmax 计算出的概率会被人工拉升至 100%（Logprob = 0.0），造成测谎仪“彻底致盲”！
因此，工业落地采取双链路物理隔离：
1. 线上实时链路：开启 FSM 前验硬约束，关闭 logprobs=True（省带宽降延迟），由 Tier-1 纯代码
   确定性 Verifier 执行真正的一票否决（哪怕模型 100% 置信度撒谎，无出处即拦截）；
2. 线下/异步链路：剥离 JSON 约束，自由作答（开启 logprobs=True），作为监控大盘与人机协作高亮核对。

纯 Python、零依赖，可直接接入 API 响应或本地推理引擎输出。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class TokenLogprobItem:
    """单个 Token 及其对数概率信息。"""

    token: str
    logprob: float
    prob: float = field(init=False)

    def __post_init__(self):
        # 将 logprob（对数概率）转换为真实百分比概率（0.0 ~ 1.0）
        self.prob = math.exp(self.logprob)


@dataclass
class LieDetectorReport:
    """测谎诊断报告。"""

    passed: bool
    overall_ppl: float  # 全文困惑度
    avg_confidence: float  # 全文平均置信度
    flagged_entities: list[dict[str, Any]]  # 判定为"心虚/编造"的实体或数字
    rejection_reasons: list[str] = field(default_factory=list)  # 触发硬否决的具体归因
    warnings: list[str] = field(default_factory=list)  # 软预警：置信度低但未单独否决的警告信息
    has_risk: bool = False  # 是否存在低置信度实体或高困惑度风险


def _is_entity_supported_in_context(entity_str: str, context: str) -> bool:
    """检查可疑实体在上下文中是否存在出处支持（协同裁决）。"""
    try:
        from .. import text_utils as tu
    except Exception:
        try:
            from rag_eval import text_utils as tu
        except Exception:
            tu = None

    if tu is not None:
        norm_ctx = tu.normalize_faithfulness_text(context)
        # 1. 尝试提取数值进行边界敏感匹配
        numbers = tu.extract_numbers(entity_str)
        if numbers:
            return all(
                tu.number_supported_in_context(val, is_pct, norm_ctx)
                for val, is_pct in numbers
            )
        # 2. 文本实体做归一化子串匹配
        return tu.normalize_for_match(entity_str) in tu.normalize_for_match(context)
    else:
        # 无 text_utils 时的轻量回退匹配
        cleaned = entity_str.strip()
        return cleaned in context


class LogprobsLieDetector:
    """Logprobs 测谎仪核心组件。

    架构原则：不能作为单独一票否决的硬断言。
    默认 allow_sole_veto=False，作为风险预警（has_risk=True, warnings）及协同裁决信号。
    """

    def __init__(
        self,
        token_confidence_threshold: float = 0.50,  # 关键实体最小置信度阈值（低于 50% 报警）
        max_ppl_threshold: float = 8.0,  # 文本整体最大困惑度容忍度
        allow_sole_veto: bool = False,  # 是否允许单独作为一票否决权（默认 False，禁止单独否决）
    ) -> None:
        self.token_confidence_threshold = token_confidence_threshold
        self.max_ppl_threshold = max_ppl_threshold
        self.allow_sole_veto = allow_sole_veto

    def analyze(
        self,
        tokens_logprobs: Sequence[TokenLogprobItem],
        critical_entities: list[str] | None = None,
        context: str | None = None,
        allow_sole_veto: bool | None = None,
    ) -> LieDetectorReport:
        """分析 API 返回的 logprobs 序列，评估是否在编造关键数据。

        Args:
            tokens_logprobs: 大模型 API 返回的 token 及其 logprob 列表。
            critical_entities: 需要重点审查的核心采分点/数字白名单（可选）。
            context: 检索上下文原文。若提供，可进行出处协同校验（避免上下文已有的实体被误杀）。
            allow_sole_veto: 是否允许单独一票否决。未指定时继承实例的 self.allow_sole_veto（默认 False）。

        Returns:
            LieDetectorReport 诊断报告。
        """
        if not tokens_logprobs:
            return LieDetectorReport(
                passed=False,
                overall_ppl=float("inf"),
                avg_confidence=0.0,
                flagged_entities=[],
                rejection_reasons=["无 Token 对数概率数据输入"],
                warnings=[],
                has_risk=False,
            )

        total_logprob = sum(item.logprob for item in tokens_logprobs)
        n = len(tokens_logprobs)

        # 1. 计算全文困惑度 (PPL) = exp(- (1/N) * sum(logprob))
        avg_neg_logprob = -(total_logprob / n)
        overall_ppl = math.exp(avg_neg_logprob)

        # 2. 平均概率置信度
        avg_confidence = sum(item.prob for item in tokens_logprobs) / n

        # 3. 扫描并定位关键 Token（数字、百分比、关键专有名词）
        flagged_entities: list[dict[str, Any]] = []
        warnings: list[str] = []
        rejection_reasons: list[str] = []

        full_text = "".join(item.token for item in tokens_logprobs)

        # 扫描目标：数字/百分比/带单位数字，以及传入的重点关注词
        target_patterns = [r"\d+(?:\.\d+)?%?", r"\d+元", r"\d+万"]
        if critical_entities:
            for ce in critical_entities:
                if ce:
                    target_patterns.append(re.escape(ce))

        combined_regex = re.compile("|".join(target_patterns))

        # 构建字符索引 → token 索引映射，精确定位哪个 token 属于哪个实体
        current_char_pos = 0
        token_spans: list[tuple[int, int, TokenLogprobItem]] = []
        for item in tokens_logprobs:
            start = current_char_pos
            end = start + len(item.token)
            token_spans.append((start, end, item))
            current_char_pos = end

        # 逐个匹配检查敏感实体区域的 token 概率
        for m in combined_regex.finditer(full_text):
            m_start, m_end = m.span()
            entity_str = m.group(0)

            overlapping_tokens = [
                item
                for (t_s, t_e, item) in token_spans
                if max(m_start, t_s) < min(m_end, t_e)
            ]
            if not overlapping_tokens:
                continue

            # 找出实体内部最"心虚"（概率最低）的那个 Token
            min_token = min(overlapping_tokens, key=lambda t: t.prob)

            if min_token.prob < self.token_confidence_threshold:
                flagged = {
                    "entity": entity_str,
                    "suspect_token": min_token.token,
                    "confidence": round(min_token.prob, 4),
                    "logprob": round(min_token.logprob, 4),
                }
                flagged_entities.append(flagged)
                warnings.append(
                    f"关键实体「{entity_str}」生成置信度极低 ({min_token.prob:.1%})，"
                    f"Token「{min_token.token}」疑似幻觉编造"
                )

        ppl_exceeded = overall_ppl > self.max_ppl_threshold
        if ppl_exceeded:
            warnings.append(
                f"全文困惑度 (PPL={overall_ppl:.2f}) 超过安全阈值 ({self.max_ppl_threshold})"
            )

        has_risk = bool(flagged_entities) or ppl_exceeded

        # 4. 判定是否通过（遵循“不能单独作为一票否决权”原则）
        sole_veto_allowed = (
            self.allow_sole_veto if allow_sole_veto is None else allow_sole_veto
        )

        passed = True
        if sole_veto_allowed:
            # 开启单独一票否决模式（向后兼容/特定强风控场景）
            if ppl_exceeded or flagged_entities:
                passed = False
                rejection_reasons.extend(warnings)
        else:
            # 默认：禁止单独否决权
            if context is not None:
                # 提供了上下文：执行协同裁决（Joint Verification）
                unsupported_flagged = [
                    fe
                    for fe in flagged_entities
                    if not _is_entity_supported_in_context(fe["entity"], context)
                ]
                if unsupported_flagged:
                    # 既低置信度（心虚），且在检索上下文中无出处支持 -> 双重确凿，触发协同否决
                    passed = False
                    for fe in unsupported_flagged:
                        rejection_reasons.append(
                            f"关键实体「{fe['entity']}」生成置信度极低 ({fe['confidence']:.1%})，"
                            f"且在检索上下文中无出处支持，触发协同否决"
                        )
                # 若 flagged_entities 全部在上下文中存在确凿出处，则放行（避免因模型采样概率波动而误杀）
            else:
                # 未提供上下文：测谎仪仅出具软预警与风险标记，不能单凭 logprob 否定结果
                passed = True

        return LieDetectorReport(
            passed=passed,
            overall_ppl=round(overall_ppl, 2),
            avg_confidence=round(avg_confidence, 4),
            flagged_entities=flagged_entities,
            rejection_reasons=rejection_reasons,
            warnings=warnings,
            has_risk=has_risk,
        )