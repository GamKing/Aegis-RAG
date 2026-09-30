"""自动化评测集（Golden Dataset）工程：合成 + 变异。

目标：产出 100 条评测样本，覆盖标准用例、长尾边界与对抗样本三类。

- 标准用例（synthetic）：基于医保模板做参数化合成，
  替换机构级别 / 报销比例 / 病种 / 条款细节，验证常规语义下链路稳健；
- 长尾边界（edge）：零起付线、问题表述扰动、全称缩写混用；
- 对抗样本（adversarial）：注入数字篡改、实体缺失、条件丢失、
  无关事实注入，独立保存错误候选，测试评分器的检出能力。

确定性：固定随机种子，同一参数必然产出同一评测集。
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass

from .dataset import (
    SAMPLE_001,
    SAMPLE_002,
    SAMPLE_003,
    SAMPLE_004,
    SAMPLE_005,
)
from .models import EvalSample

# ---------------------------------------------------------------------------
# 合成模板：可替换的字段槽
# ---------------------------------------------------------------------------

_HOSPITAL_LEVELS = ["一级及以下", "二级", "三级"]
_DISEASES = [
    "高血压", "糖尿病", "冠心病", "慢性肾病", "慢阻肺", "类风湿关节炎",
    "强直性脊柱炎", "乙肝", "甲状腺功能减退", "帕金森病",
]
_INCOME_LIMITS = [1500, 2000, 2500, 3000, 4000]
_RATIOS = [55, 60, 65, 70, 75, 80, 85, 90]
_DEDUCTIBLES = [100, 200, 300, 500, 600, 800, 1000, 1200]

# 合成模板：{question, context, answer, entities} 含 {} 槽位
_SYNTH_TEMPLATES = [
    {
        "question": "城乡居民医保住院在{level}医疗机构的起付线、报销比例和年度封顶线是多少？",
        "context": (
            "城乡居民基本医疗保险住院待遇标准：{level}医疗机构起付线{deductible}元，"
            "政策范围内费用报销比例{ratio}%；一个自然年度内，基本医保统筹基金"
            "最高支付限额（封顶线）为{limit}万元。"
        ),
        "answer": (
            "城乡居民医保住院在{level}医疗机构起付线{deductible}元、报销{ratio}%；"
            "一个自然年度内统筹基金封顶线{limit}万元。"
        ),
        "entities": ["{deductible}元", "{ratio}%", "{limit}万"],
    },
    {
        "question": "{disease}是否纳入门诊慢特病保障范围？统筹基金年度限额和报销比例是多少？",
        "context": (
            "{disease}病纳入门诊慢特病保障范围。统筹基金支付限额为每人每年{limit}元，"
            "政策范围内费用报销比例为{ratio}%。待遇认定须经二级及以上定点医疗机构"
            "确诊，并由医保经办机构审核通过。"
        ),
        "answer": (
            "{disease}病纳入门诊慢特病保障范围，统筹基金支付限额每人每年{limit}元，"
            "政策范围内费用报销比例{ratio}%。"
        ),
        "entities": ["{disease}", "门诊慢特病", "统筹基金", "{limit}元", "{ratio}%"],
    },
    {
        "question": "大病保险的基础起付线、报销比例是多少，对困难群体有没有倾斜？",
        "context": (
            "大病保险的保障对象为参加城乡居民基本医疗保险的人员。一个自然年度内，"
            "个人自付的合规医疗费用累计超过起付线{deductible}元的部分，纳入大病"
            "保险支付范围，报销比例为{ratio}%，不设封顶线。特困人员、低保对象等"
            "困难群体起付线降低50%，报销比例提高5个百分点。"
        ),
        "answer": (
            "大病保险起付线为{deductible}元，个人自付合规费用超过起付线的部分报销"
            "{ratio}%，不设封顶线；困难群体起付线降低50%，报销比例提高5个百分点。"
        ),
        "entities": ["{deductible}元", "{ratio}%", "困难群体", "5个百分点"],
    },
]


def _render(template: dict, **kwargs) -> EvalSample:
    """渲染一个合成样本。"""
    question = template["question"].format(**kwargs)
    context = template["context"].format(**kwargs)
    answer = template["answer"].format(**kwargs)
    entities = [e.format(**kwargs) for e in template["entities"]]
    return EvalSample(
        id="unscoped-template",
        question=question,
        ground_truth_context=context,
        ground_truth_answer=answer,
        expected_entities=entities,
    )


# ---------------------------------------------------------------------------
# 变异器：注入失败模式
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """一种对抗变异：对标准样本注入缺陷。"""

    name: str
    apply: callable  # EvalSample -> EvalSample（变异后的 ground_truth_answer）


def mutate_digit_swap(sample: EvalSample) -> EvalSample:
    """对抗1：数字篡改——把答案里的一个数字替换为邻近值。"""
    numbers = re.findall(r"\d+", sample.ground_truth_answer)
    if not numbers:
        return copy.deepcopy(sample)
    target = numbers[0]
    tampered = str(int(target) + 1)
    new_answer = sample.ground_truth_answer.replace(target, tampered, 1)
    return EvalSample(
        id=sample.id + "-mut-digit",
        question=sample.question,
        ground_truth_context=sample.ground_truth_context,
        ground_truth_answer=new_answer,
        expected_entities=sample.expected_entities,
    )


def mutate_drop_entity(sample: EvalSample) -> EvalSample:
    """对抗2：实体缺失——从答案中删掉一个期望实体。"""
    if not sample.expected_entities:
        return copy.deepcopy(sample)
    drop = sample.expected_entities[0]
    new_answer = sample.ground_truth_answer.replace(drop, "")
    return EvalSample(
        id=sample.id + "-mut-drop-entity",
        question=sample.question,
        ground_truth_context=sample.ground_truth_context,
        ground_truth_answer=new_answer,
        expected_entities=sample.expected_entities,
    )


def mutate_drop_condition(sample: EvalSample) -> EvalSample:
    """对抗3：条件丢失——把答案里的例外/前置条件从句删除。"""
    # 简单启发式：删除"未备案""例外""不纳入"所在的分句
    new_answer = sample.ground_truth_answer
    for marker in ("未备案自行外出就医的", "非医保目录内费用不纳入报销范围", "例外情形"):
        if marker in new_answer:
            idx = new_answer.find(marker)
            new_answer = new_answer[:idx].rstrip("，；。 ")
            break
    return EvalSample(
        id=sample.id + "-mut-drop-cond",
        question=sample.question,
        ground_truth_context=sample.ground_truth_context,
        ground_truth_answer=new_answer,
        expected_entities=sample.expected_entities,
    )


def mutate_inject_fabrication(sample: EvalSample) -> EvalSample:
    """对抗4：无关事实注入——在答案末尾追加上下文没有的内容。"""
    fabricated = "同时，参保人员可享受补充商业保险额外10%的报销优惠。"
    new_answer = sample.ground_truth_answer + fabricated
    return EvalSample(
        id=sample.id + "-mut-fabricate",
        question=sample.question,
        ground_truth_context=sample.ground_truth_context,
        ground_truth_answer=new_answer,
        expected_entities=sample.expected_entities,
    )


MUTATIONS: list[Mutation] = [
    Mutation("数字篡改", mutate_digit_swap),
    Mutation("实体缺失", mutate_drop_entity),
    Mutation("条件丢失", mutate_drop_condition),
    Mutation("无关注入", mutate_inject_fabrication),
]


# ---------------------------------------------------------------------------
# 主入口：合成 100 条
# ---------------------------------------------------------------------------

def _scope(sample: EvalSample, source_id: str) -> EvalSample:
    """问题和文档显式标注虚构政策版本，不通过样本 ID 偷选文档。"""
    label = f"【模拟政策 {source_id}】"
    return sample.model_copy(update={
        "source_id": source_id,
        "required_entities": list(sample.expected_entities),
        "question": label + sample.question,
        "ground_truth_context": label + sample.ground_truth_context,
    })


def generate_dataset(
    seed: int = 42,
    synth_count: int = 55,
    edge_count: int = 25,
    adv_count: int = 15,
) -> list[EvalSample]:
    """默认 100 条：55 合成 + 25 边界 + 15 对抗 + 5 种子。

    错误候选单独存入 candidate_answer，ground_truth_answer 保持正确。
    虚构政策版本仅用于离线检索实验，不代表真实医保政策。
    """
    if min(synth_count, edge_count, adv_count) < 0:
        raise ValueError("样本数量不能为负")
    if synth_count == 0 and (edge_count or adv_count):
        raise ValueError("边界/对抗样本需要至少一条合成样本")
    rng = random.Random(seed)
    samples: list[EvalSample] = []
    for i in range(synth_count):
        template = _SYNTH_TEMPLATES[i % len(_SYNTH_TEMPLATES)]
        if "level" in template["question"]:
            kwargs = dict(level=rng.choice(_HOSPITAL_LEVELS),
                          deductible=rng.choice(_DEDUCTIBLES),
                          ratio=rng.choice(_RATIOS), limit=rng.choice([10, 15, 20, 25, 30]))
        elif "disease" in template["question"]:
            kwargs = dict(disease=rng.choice(_DISEASES),
                          limit=rng.choice(_INCOME_LIMITS), ratio=rng.choice(_RATIOS))
        else:
            kwargs = dict(deductible=rng.choice(_DEDUCTIBLES), ratio=rng.choice(_RATIOS))
        sample = _render(template, **kwargs).model_copy(update={
            "id": f"syn-{i:03d}", "category": "synthetic",
        })
        # 字母版本号避免标识数字被抽取为业务事实。
        code = "".join(chr(97 + int(d)) for d in str(i))
        samples.append(_scope(sample, "policy_" + code))

    for i in range(edge_count):
        base = samples[i % synth_count]
        if i % 3 == 0:
            # 真正构造零起付线，同步上下文、答案和实体，并创建独立版本。
            template = _SYNTH_TEMPLATES[0]
            base = _render(template, level="一级及以下", deductible=0,
                           ratio=70, limit=20)
            code = "".join(chr(97 + int(d)) for d in str(i))
            base = _scope(base, "edge_" + code)
            question = base.question + "请明确说明零起付线。"
        elif i % 3 == 1:
            # 表述扰动，不添加上下文无法回答的退休待遇问题。
            question = "请依据所指政策回答：" + base.question
        else:
            question = base.question.replace("城乡居民", "居保").replace("大病保险", "大病险")
        samples.append(base.model_copy(update={
            "id": f"edge-{i:03d}", "question": question, "category": "edge",
        }))

    for i in range(adv_count):
        mutation = MUTATIONS[i % len(MUTATIONS)]
        base = samples[i % synth_count]
        if mutation.name == "条件丢失":
            base = _scope(SAMPLE_004, "seed_d")
        candidate = mutation.apply(base).ground_truth_answer
        if candidate == base.ground_truth_answer:
            raise ValueError(f"对抗变异未生效: {mutation.name}")
        samples.append(base.model_copy(update={
            "id": f"adv-{i:03d}-{mutation.name}", "category": "adversarial",
            "candidate_answer": candidate, "mutation_type": mutation.name,
        }))

    for letter, base in zip("abcde", (SAMPLE_001, SAMPLE_002, SAMPLE_003, SAMPLE_004, SAMPLE_005)):
        samples.append(_scope(base, "seed_" + letter).model_copy(update={"category": "seed"}))
    return samples


def save_dataset(samples: list[EvalSample], path: str) -> None:
    """将评测集保存为 JSON。"""
    import json

    data = [s.model_dump(mode="json") for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(data)} 条评测样本到 {path}")


if __name__ == "__main__":
    ds = generate_dataset()
    save_dataset(ds, "golden_dataset_100.json")
    print(f"合计 {len(ds)} 条")
