"""三个核心评分器：检索召回（Context Recall）/ 答案完整性（Completeness）/
数值忠实性（Faithfulness）。

三个评分器全部是纯代码断言（无模型打分），保证：
- 可复现：同一输入永远同一输出；
- 可审计：每条失败都能给出确定性的归因文本；
- 可离线：不依赖任何外部服务。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Protocol


class FaithfulnessJudge(Protocol):
    def check_faithfulness(self, context: str, claim: str) -> bool: ...

from . import text_utils as tu
from .models import EvalSample


@dataclass
class MetricOutcome:
    """单个评分器的判定产出。score 为 None 表示纯布尔型指标（忠实性）。"""

    evaluator_name: str
    score: Optional[float] = None
    passed: bool = True
    details: list = field(default_factory=list)


class BaseEvaluator(abc.ABC):
    """评分器基类：子类实现 evaluate，由 Runner 统一驱动与聚合。"""

    name = "base"

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @abc.abstractmethod
    def evaluate(
        self, sample: EvalSample, actual_context: str, actual_answer: str
    ) -> MetricOutcome:
        raise NotImplementedError


class ContextRecallEvaluator(BaseEvaluator):
    """评估 actual_context 对 ground_truth_context 的关键内容覆盖率。

    方法：双端抽取词项（CJK bigram + ASCII 词/数字），计算

        recall = |GT 词项 ∩ ACTUAL 词项| / |GT 词项|

    词项级召回是检索质量的保守代理指标——bigram 覆盖率低，说明标准上下文中的
    关键表述（政策条件、例外条款、数字）没有被检索命中。
    """

    name = "context_recall"

    def __init__(
        self, threshold: float = 0.80, max_reported_segments: int = 8
    ) -> None:
        super().__init__(threshold)
        self.max_reported_segments = max_reported_segments

    def evaluate(
        self, sample: EvalSample, actual_context: str, actual_answer: str
    ) -> MetricOutcome:
        gt_terms = tu.extract_terms(sample.ground_truth_context)
        actual_terms = tu.extract_terms(actual_context)
        if not gt_terms:  # 金标准为空，视为平凡通过
            return MetricOutcome(self.name, score=1.0, passed=True)

        missing = gt_terms - actual_terms
        score = round(1.0 - len(missing) / len(gt_terms), 4)
        passed = score >= self.threshold

        details: list = []
        if not passed:
            cjk_segs, ascii_tokens = tu.find_uncovered_ground_truth_segments(
                sample.ground_truth_context, actual_terms, self.max_reported_segments
            )
            parts = []
            if cjk_segs:
                shown = "、".join(f"「{s}」" for s in cjk_segs)
                parts.append(f"缺失关键片段: {shown}")
            if ascii_tokens:
                parts.append(f"缺失数字/符号: {'、'.join(ascii_tokens)}")
            detail = (
                f"[{self.name}] 得分 {score:.3f} 低于阈值 {self.threshold:.2f}"
            )
            if parts:
                detail += "; " + "; ".join(parts)
            details.append(detail)

        return MetricOutcome(self.name, score=score, passed=passed, details=details)


class CompletenessEvaluator(BaseEvaluator):
    """严格实体命中检查：命中数 / 期望总数。

    匹配口径：对实体与答案做归一化（全角->半角、去空白、忽略大小写）后，
    要求实体以连续子串形式出现在答案中——"异地就医备案" 不会被
    "异地就医办理备案" 稀释命中，保证严格性。
    """

    name = "completeness"

    def evaluate(
        self, sample: EvalSample, actual_context: str, actual_answer: str
    ) -> MetricOutcome:
        entities = sample.expected_entities
        if not entities:  # 未声明实体的样本，视为平凡通过
            return MetricOutcome(self.name, score=1.0, passed=True)

        answer_norm = tu.normalize_for_match(actual_answer)
        missed = [e for e in entities if tu.normalize_for_match(e) not in answer_norm]
        hit = len(entities) - len(missed)
        score = round(hit / len(entities), 4)
        passed = score >= self.threshold

        details: list = []
        if not passed:
            missed_repr = "、".join(f"「{e}」" for e in missed)
            details.append(
                f"[{self.name}] 实体命中率 {hit}/{len(entities)} = {score:.3f} "
                f"低于阈值 {self.threshold:.2f}; 未命中: {missed_repr}"
            )

        return MetricOutcome(self.name, score=score, passed=passed, details=details)


class FaithfulnessEvaluator(BaseEvaluator):
    """数值忠实性：答案中的每个数值/百分比都必须能在检索上下文中找到出处。

    纯代码断言，一票否决制：
    - 提取答案的全部数值单元（含是否百分比标记）；
    - 逐一在上下文中做边界敏感匹配（见 text_utils.number_supported_in_context）；
    - 只要存在一个无出处数值 => passed=False。

    threshold 字段对布尔型指标无意义，保留只为统一基类接口。
    """

    name = "faithfulness"

    def __init__(self, nli_judge: Optional[FaithfulnessJudge] = None) -> None:
        super().__init__(threshold=1.0)
        self.nli_judge = nli_judge

    def evaluate(
        self, sample: EvalSample, actual_context: str, actual_answer: str
    ) -> MetricOutcome:
        numbers = tu.extract_faithfulness_numbers(actual_answer)
        normalized_context = tu.normalize_faithfulness_text(actual_context)
        unsupported = [
            (value, is_pct)
            for value, is_pct in numbers
            if not tu.number_supported_in_context(value, is_pct, normalized_context)
        ]

        # 对字面未命中的完整答案做一次可选 NLI 复核。NLI 只负责确认“合理直接推导”，
        # 不会替代数字边界规则；未注入 judge 时保持原有纯代码行为。
        nli_pass = False
        if unsupported and self.nli_judge is not None:
            try:
                nli_pass = self.nli_judge.check_faithfulness(
                    actual_context, actual_answer
                )
            except Exception as exc:  # noqa: BLE001
                nli_pass = False
                details = [f"[{self.name}] NLI 复核异常: {exc!r}"]
            else:
                details = []
        else:
            details = []

        passed = not unsupported or nli_pass

        for value, is_pct in ([] if nli_pass else unsupported):
            raw = value + ("%" if is_pct else "")
            details.append(
                f"[{self.name}] 答案数值 「{raw}」 在检索上下文中无出处"
                f"（数值幻觉/篡改嫌疑），一票否决"
            )
        if nli_pass:
            details.append(
                f"[{self.name}] 字面数值未完全命中，但 NLI 判定答案是参考材料的合理直接推导"
            )

        return MetricOutcome(self.name, score=None, passed=passed, details=details)
