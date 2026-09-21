"""可选的轻量 NLI 语义裁判适配器。

NLIJudge 不参与默认离线评测，只有显式注入 client 时才启用；这样默认流程仍然
完全离线、可复现。模型只允许返回 A/B/C：蕴含、矛盾、中立。
"""
from __future__ import annotations


class NLIJudge:
    def __init__(self, client, model: str = "Kimi2.5:1.5b") -> None:
        self.client = client
        self.model = model

    def classify(self, context: str, claim: str) -> str:
        prompt = f"""你是一个严肃的逻辑事实裁判。判断【待核实主张】是否能够由【参考材料】严格推导出来。

【参考材料】: {context}
【待核实主张】: {claim}

请仅输出一个选项字母：
A. 蕴含（主张的事实完全源自材料，或属于合理直接推导）
B. 矛盾（主张与材料明确冲突）
C. 中立（材料未提及，属于主张自己脑补的信息）

你的选择是 (仅回复 A、B 或 C):"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().upper()
        return verdict if verdict in {"A", "B", "C"} else "C"

    def check_faithfulness(self, context: str, claim: str) -> bool:
        return self.classify(context, claim) == "A"
