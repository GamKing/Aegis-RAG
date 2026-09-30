"""受控表面实现（Surface Realization）引擎。

把两段式受控流水线断点的"最后一步"补齐：

    检索 Context → 抽取 → 质检 → 表面实现 → 合规自然语言答复

职责：安全、合规地将核验通过的结构化事实还原为通顺自然语言，**不给大模型
发散胡编的空间**，确保采分点（实体 + 数值）全覆盖。

两种落地方式：
- EntityAwareRealizer：纯规则模板拼装（零 LLM，离线/低延迟，默认）；
- ConstrainedRealizerPrompt：受控润色的系统提示词（供接入 LLM 时使用）。

核心算法（EntityAwareRealizer）：
1. 提取事实的底层数值（剥除内部标签）；
2. 在检索到的上下文中回溯该数值所在句的修饰成分（主语实体 + 指标词）；
3. 用"主语 + 的 + 指标 + 为 + 数值"的合规句式缝合；
4. 严禁产生列表以外的任何数字。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


# 高权重实体词表（用于从上下文回溯主语）
_ENTITY_KEYWORDS = [
    "一级及以下医疗机构", "二级医疗机构", "三级医疗机构", "定点医疗机构",
    "参保职工", "城乡居民", "异地就医", "大病保险", "门诊费用",
    "住院医疗费用", "退休人员", "在职职工", "困难群体", "特困人员",
    "低保对象", "门诊慢特病", "统筹基金", "急诊抢救人员",
]

# 指标/谓词词表（用于从上下文回溯数值的属性）
_METRIC_KEYWORDS = [
    "报销比例", "支付比例", "起付线", "起付标准", "最高支付限额",
    "自负比例", "降低幅度", "补偿比例", "年度限额", "封顶线",
]


@dataclass
class SurfaceClause:
    """一次表面实现的输出子句。"""

    value: str
    subject: str
    metric: str
    rendered: str


class EntityAwareRealizer:
    """基于上下文实体回溯的规则模板表面实现引擎（做法 A，零 LLM）。"""

    def __init__(self, prefix: str = "根据相关政策规定，") -> None:
        self.prefix = prefix

    def realize(
        self,
        verified_facts: Sequence[Any],
        question: str,
        retrieved_context: str,
    ) -> str:
        """把校验过的事实还原为包含完整实体的自然语言。

        Args:
            verified_facts: 已核验的事实列表（ExtractedFact 或含 value 的对象）。
            question: 用户原始问题。
            retrieved_context: 检索到的参考上下文。

        Returns:
            合规的自然语言答复。
        """
        if not verified_facts:
            return "根据提供的参考政策文件，未查询到相关标准或具体规定。"

        clauses: list[str] = []
        for fact in verified_facts:
            raw_val = self._extract_clean_value(fact)
            if not raw_val:
                continue
            subject, metric = self._resolve_entity_and_metric(raw_val, retrieved_context)
            clauses.append(self._assemble(subject, metric, raw_val))

        if not clauses:
            return "未能从参考资料中整理出确切的标准信息。"

        return self.prefix + "；".join(clauses) + "。"

    def _assemble(self, subject: str, metric: str, value: str) -> str:
        """缝合主谓宾。"""
        if subject and metric:
            return f"{subject}的{metric}为{value}"
        if subject:
            return f"{subject}为{value}"
        return f"相关规定标准为{value}"

    def _extract_clean_value(self, fact: Any) -> str:
        """清洗提取数值，剥除内部标签。"""
        if hasattr(fact, "value"):
            return str(fact.value).strip()
        if isinstance(fact, dict):
            return str(fact.get("value", "")).strip()
        s = str(fact).strip()
        if ":" in s:
            s = s.split(":", 1)[-1].strip()
        return s

    def _resolve_entity_and_metric(self, val: str, context: str) -> tuple[str, str]:
        """纯代码实体回溯：在 context 中定位该数值，向前抓取该句的修饰成分。"""
        escaped_val = re.escape(val)
        # 匹配该数值前同一句内的文本（最多回溯 35 字）
        pattern = rf"([^。；\n]{{2,35}}?){escaped_val}"
        match = re.search(pattern, context)
        if not match:
            return "", ""
        prefix_text = match.group(1).strip("，,：: ")

        # 核心实体匹配（优先提取高权重实体）
        found_entity = ""
        for kw in _ENTITY_KEYWORDS:
            if kw in prefix_text:
                found_entity = kw
                break

        # 指标词匹配
        found_metric = ""
        for kw in _METRIC_KEYWORDS:
            if kw in prefix_text:
                found_metric = kw
                break

        return found_entity, found_metric


# ---------------------------------------------------------------------------
# 做法 B：受限润色的系统提示词（接入 LLM 时使用）
# ---------------------------------------------------------------------------

CONSTRAINED_REALIZER_PROMPT = """【系统角色】
你是一个严格受控的公文润色助手（Surface Realizer）。

【输入】
1. 已核实事实列表：{facts}
2. 用户问题：{question}

【生成硬性红线】
1. 你的唯一任务是用通顺的汉语将【已核实事实列表】中的内容串联成一段回答；
2. 严禁添加列表中未列出的任何数字、时间、百分比、机构名称或例外条款；
3. 不得包含任何形如 "numeric_fact" 或内部 JSON 标签的字符；
4. 若列表中无事实，必须且只能回答："根据参考资料，未查询到相关政策标准。"
"""


def build_constrained_realizer_prompt(facts: Sequence[Any], question: str) -> str:
    """构造受控润色的系统提示词（做法 B）。"""
    rendered_facts = [str(f) for f in facts]
    return CONSTRAINED_REALIZER_PROMPT.format(
        facts=rendered_facts,
        question=question,
    )