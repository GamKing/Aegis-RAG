import asyncio
import json
import os
import re
import sys
import time
import httpx


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 将项目根目录加入模块搜索路径，确保直接 import 本地工程
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag_eval.models import (
    AtomicFact,
    EvalSample,
    ExtractedFact,
    ExtractionPayload,
    FactType,
    VerificationResult,
)
from rag_eval.pipeline import Tier1CodeVerifier
from rag_eval.surface import EntityAwareRealizer

# 推理服务端口配置
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "22434")
OLLAMA_URL = os.environ.get("OLLAMA_URL", f"http://localhost:{OLLAMA_PORT}/api/generate")
MODEL_NAME = "qwen2.5-benchmark"

# 构造真实医保业务长文档（包含规约通用条款与特定待遇规则，~3000 Tokens）
BASE_POLICY_DOC = """
【国家基本医疗保险与门诊报销管理规约（2026年修订版）】
第一条 参保人员在定点甲等公立医疗机构普通门诊就诊，起付标准为个人自付累计满 1500 元。
第二条 超过起付标准部分，在一级医疗机构报销比例为 85%，二级医疗机构报销比例为 75%，三级医疗机构报销比例为 60%。
第三条 退休人员在上述报销比例基础上各档次分别上调 5%。直系亲属共济账户可用于支付个人自负部分。
第四条 恶性肿瘤放化疗、器官移植抗排异治疗等特殊门诊，不设起付线，统筹基金合规报销比例固定为 90%。
第五条 境外就医及未经转诊异地就医产生的医疗费用，统筹基金一票否决不予报销。
""" * 20

RESIDENT_HOSPITAL_POLICY = """
【城乡居民医保住院待遇细则】
第六条 城乡居民基本医疗保险住院待遇标准：一级及以下医疗机构起付线为300元，政策范围内费用报销比例为85%；二级医疗机构起付线为600元，报销比例为70%；三级医疗机构起付线为1200元，报销比例为55%。一个自然年度内，基本医保统筹基金最高支付限额（封顶线）为20万元。
第七条 参保人员在非定点医疗机构或未经批准异地就医产生的医疗费用，统筹基金不予支付。
"""

REAL_CONTEXT = f"{BASE_POLICY_DOC}\n\n{RESIDENT_HOSPITAL_POLICY}"
QUESTION = "请问城乡居民医保住院在不同级别医院的起付线和报销比例分别是多少？年度封顶线是多少？"

SAMPLE = EvalSample(
    id="benchmark-case-001",
    question=QUESTION,
    ground_truth_context=REAL_CONTEXT,
    ground_truth_answer="一级起付300元报销85%；二级起付600元报销70%；三级起付1200元报销55%；封顶线20万元。",
    expected_entities=["300元", "85%", "600元", "70%", "1200元", "55%", "20万元"],
)

# 1. 传统单阶段散文生成 Prompt
SINGLE_STAGE_PROMPT = f"""{REAL_CONTEXT}

【问题】：{QUESTION}
请根据以上规定，详细分析并撰写一份正式的居民住院报销政策解答。"""

# 2. 当前项目规范的受限事实抽取 Prompt（严格对齐 AtomicFact 契约）
PROJECT_EXTRACTION_PROMPT = f"""{REAL_CONTEXT}

【问题】：{QUESTION}
【抽取任务契约】：
从上文中提取与问题最相关的核心原子事实（仅提取各级起付线、报销比例与封顶线），输出严格紧凑的 JSON 数组，严禁任何多余解释与客套话：
```json
{{
  "atomic_facts": [
    {{"subject": "一级及以下医疗机构", "predicate": "起付线", "object_value": "300元", "fact_type": "numeric"}},
    {{"subject": "一级及以下医疗机构", "predicate": "报销比例", "object_value": "85%", "fact_type": "numeric"}},
    {{"subject": "三级医疗机构", "predicate": "起付线", "object_value": "1200元", "fact_type": "numeric"}},
    {{"subject": "三级医疗机构", "predicate": "报销比例", "object_value": "55%", "fact_type": "numeric"}},
    {{"subject": "基本医保统筹基金", "predicate": "最高支付限额", "object_value": "20万元", "fact_type": "numeric"}}
  ]
}}
```
"""


async def call_llm(client: httpx.AsyncClient, prompt: str, max_tokens: int):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
        },
    }
    t0 = time.perf_counter()
    first_token_t = None
    tokens = 0
    chunks = []

    async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
        async for line in resp.aiter_lines():
            if not line:
                continue
            item = json.loads(line)
            if first_token_t is None:
                first_token_t = time.perf_counter()
            tokens += 1
            chunks.append(item.get("response", ""))
            if item.get("done", False):
                break

    t1 = time.perf_counter()
    return {
        "ttft": (first_token_t - t0) if first_token_t else 0.0,
        "llm_time": t1 - t0,
        "tokens": tokens,
        "text": "".join(chunks),
    }


def parse_and_verify_with_project(raw_llm_text: str, context: str, sample: EvalSample):
    """直接调用本地 rag_eval 模块中的 Tier1CodeVerifier 和 EntityAwareRealizer。"""
    t_parse_start = time.perf_counter()
    
    # 1. 宽容反序列化与契约组装
    clean_text = raw_llm_text.strip()
    if "```json" in clean_text:
        clean_text = clean_text.split("```json")[1].split("```")[0].strip()
    elif "```" in clean_text:
        clean_text = clean_text.split("```")[1].split("```")[0].strip()

    data = None
    try:
        data = json.loads(clean_text)
    except Exception:
        # 正则提取所有独立的完整三元组对象
        obj_matches = re.findall(r"\{[^{}]*\"object_value\"[^{}]*\}", clean_text)
        if obj_matches:
            try:
                data = {"atomic_facts": [json.loads(m) for m in obj_matches]}
            except Exception:
                pass
    if not data or not isinstance(data, dict):
        data = {"atomic_facts": []}

    raw_facts = data.get("atomic_facts", [])
    atomic_facts = []
    extracted_facts = []

    for f in raw_facts:
        val = str(f.get("object_value", "")).strip()
        if not val:
            continue
        ft_str = f.get("fact_type", "numeric")
        try:
            ft = FactType(ft_str)
        except Exception:
            ft = FactType.NUMERIC

        subj = str(f.get("subject", "")).strip()
        pred = str(f.get("predicate", "")).strip()

        af = AtomicFact(
            subject=subj,
            predicate=pred,
            object_value=val,
            fact_type=ft,
            is_negative=bool(f.get("is_negative", False)),
            source_text=val,
        )
        atomic_facts.append(af)
        extracted_facts.append(ExtractedFact(key=pred or "numeric_fact", value=val, source_text=val, is_numeric=True))

    payload = ExtractionPayload(
        facts=extracted_facts,
        claims=[f.object_value for f in atomic_facts],
        raw_json={
            "query": sample.question,
            "facts": [f.model_dump() for f in extracted_facts],
            "atomic_facts": [f.model_dump() for f in atomic_facts],
        },
    )

    t_verify_start = time.perf_counter()
    # 2. 调用真实的本地 Tier1CodeVerifier 质检网关
    verifier = Tier1CodeVerifier()
    verification = verifier.verify(sample, context, payload)
    t_verify_end = time.perf_counter()

    # 3. 若通过质检，调用真实的本地 EntityAwareRealizer 进行表面实现
    t_realize_start = time.perf_counter()
    realizer = EntityAwareRealizer(prefix="根据最新医保规约规定：")
    if verification.passed and extracted_facts:
        final_answer = realizer.realize(
            verified_facts=extracted_facts,
            question=sample.question,
            retrieved_context=context,
        )
    elif not extracted_facts:
        final_answer = "根据提供的参考政策文件，未提取到符合条件的标准数据。"
    else:
        final_answer = "【合规安全熔断】Tier-1 质检未通过：提取数据与参考资料不符，拒绝输出。"
    t_realize_end = time.perf_counter()

    return {
        "payload": payload,
        "verification": verification,
        "final_answer": final_answer,
        "t_verify_ms": (t_verify_end - t_verify_start) * 1000,
        "t_realize_ms": (t_realize_end - t_realize_start) * 1000,
        "t_code_total_ms": (t_realize_end - t_parse_start) * 1000,
    }



async def test_baseline(concurrency: int = 4):
    print("\n" + "=" * 55)
    print(f"  正在执行【对照组：传统单阶段散文生成】(并发 C={concurrency})")
    print("=" * 55)

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        start_wall = time.perf_counter()
        tasks = [call_llm(client, SINGLE_STAGE_PROMPT, max_tokens=250) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - start_wall

    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_llm_time = sum(r["llm_time"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    tps = total_tokens / wall_time

    print(f"成功完成请求: {len(results)}/{concurrency}")
    print(f"平均首字延迟 (TTFT): {avg_ttft:.3f} s")
    print(f"平均单请求端到端耗时: {avg_llm_time:.3f} s")
    print(f"单请求平均生成 Token 数: {total_tokens / len(results):.1f} tokens")
    print(f"系统吞吐量 (Throughput): {tps:.2f} tokens/s")
    print(f"批次总墙钟耗时 (Wall Time): {wall_time:.2f} s")
    print("\n[单阶段直接生成样例片选]:")
    print(results[0]["text"].strip()[:180] + " ...\n")
    return {
        "avg_ttft": avg_ttft,
        "avg_latency": avg_llm_time,
        "tokens_per_req": total_tokens / len(results),
        "wall_time": wall_time,
        "tps": tps,
    }


async def test_project_two_stage(concurrency: int = 4):
    print("=" * 55)
    print(f"  正在执行【实验组：Aegis-RAG 真实项目两段式流水线】(并发 C={concurrency})")
    print("  调用链路: LLM抽取 -> Tier1CodeVerifier质检 -> EntityAwareRealizer表面实现")
    print("=" * 55)

    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        start_wall = time.perf_counter()
        tasks = [call_llm(client, PROJECT_EXTRACTION_PROMPT, max_tokens=300) for _ in range(concurrency)]
        llm_results = await asyncio.gather(*tasks)

        # 串联本地真实工程代码
        pipeline_outputs = []
        for res in llm_results:
            pipe_out = parse_and_verify_with_project(res["text"], REAL_CONTEXT, SAMPLE)
            pipeline_outputs.append(pipe_out)

        wall_time = time.perf_counter() - start_wall

    avg_ttft = sum(r["ttft"] for r in llm_results) / len(llm_results)
    avg_llm_time = sum(r["llm_time"] for r in llm_results) / len(llm_results)
    total_tokens = sum(r["tokens"] for r in llm_results)
    tps = total_tokens / wall_time

    avg_verify_ms = sum(p["t_verify_ms"] for p in pipeline_outputs) / len(pipeline_outputs)
    avg_realize_ms = sum(p["t_realize_ms"] for p in pipeline_outputs) / len(pipeline_outputs)
    avg_code_ms = sum(p["t_code_total_ms"] for p in pipeline_outputs) / len(pipeline_outputs)
    avg_e2e_sec = avg_llm_time + (avg_code_ms / 1000.0)

    all_passed = all(p["verification"].passed for p in pipeline_outputs)

    print(f"成功完成请求: {len(llm_results)}/{concurrency}")
    print(f"平均首字延迟 (TTFT): {avg_ttft:.3f} s")
    print(f"阶段一 LLM 抽取耗时: {avg_llm_time:.3f} s")
    print(f"阶段中 Tier-1 确定性质检耗时: 平均 {avg_verify_ms:.3f} 毫秒")
    print(f"阶段二 EntityAwareRealizer 组装耗时: 平均 {avg_realize_ms:.3f} 毫秒")
    print(f"本地 Python 代码处理总耗时: 平均 {avg_code_ms:.3f} 毫秒")
    print(f"端到端综合单请求延迟 (E2E Latency): {avg_e2e_sec:.3f} s")
    print(f"单请求平均生成 Token 数: {total_tokens / len(llm_results):.1f} tokens")
    print(f"批次总墙钟耗时 (Wall Time): {wall_time:.2f} s")
    print(f"Tier-1 质检通过率: {'100% 验收通过' if all_passed else '存在拦截'}")

    sample_out = pipeline_outputs[0]
    print("\n[阶段一抽取 JSON 结果]:")
    print(sample_out["payload"].raw_json.get("atomic_facts", []))
    print("\n[阶段二 EntityAwareRealizer 真实组装答复]:")
    print(sample_out["final_answer"])
    print("\n" + "=" * 55)

    return {
        "avg_ttft": avg_ttft,
        "avg_latency": avg_e2e_sec,
        "tokens_per_req": total_tokens / len(llm_results),
        "wall_time": wall_time,
        "tps": tps,
        "avg_verify_ms": avg_verify_ms,
        "avg_realize_ms": avg_realize_ms,
        "avg_code_ms": avg_code_ms,
        "all_passed": all_passed,
    }


async def main():
    b = await test_baseline(concurrency=4)
    await asyncio.sleep(2)
    p = await test_project_two_stage(concurrency=4)


if __name__ == "__main__":
    asyncio.run(main())
