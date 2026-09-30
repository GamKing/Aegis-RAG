import asyncio
import json
import os
import sys
import time
import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 优先支持环境变量配置，默认指向当前环境监听的 22434 端口
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "22434")
OLLAMA_URL = os.environ.get("OLLAMA_URL", f"http://localhost:{OLLAMA_PORT}/api/generate")
MODEL_NAME = "qwen2.5-benchmark"

# 同样的长文档上下文
MOCK_CONTEXT = """
【国家基本医疗保险与门诊报销管理规约（2026年修订版）】
第一条 参保人员在定点甲等公立医疗机构普通门诊就诊，起付标准为个人自付累计满 1500 元。
第二条 超过起付标准部分，在一级医疗机构报销比例为 85%，二级医疗机构报销比例为 75%，三级医疗机构报销比例为 60%。
第三条 退休人员在上述报销比例基础上各档次分别上调 5%。直系亲属共济账户可用于支付个人自负部分。
第四条 恶性肿瘤放化疗、器官移植抗排异治疗等特殊门诊，不设起付线，统筹基金合规报销比例固定为 90%。
第五条 境外就医及未经转诊异地就医产生的医疗费用，统筹基金一票否决不予报销。
""" * 30

# 1. 传统单阶段 Prompt：要求大模型输出完整解答长文
SINGLE_STAGE_PROMPT = f"{MOCK_CONTEXT}\n\n请详细列出三级医院在职人员的报销比例、特殊门诊待遇以及异地就医的拒赔规则，并输出完整分析解答。"

# 2. Aegis-RAG 抽取阶段 Prompt：受限格式，仅输出压缩 JSON
EXTRACTION_PROMPT = f"""{MOCK_CONTEXT}

【任务要求】：从上文中抽取事实，仅输出如下严格紧凑的 JSON，严禁散文叙事：
{{"lv3_ratio": "数值", "special_clinic_ratio": "数值", "remote_medical_covered": false}}
"""

# Aegis-RAG 第二阶段：纯代码表面实现器（零显卡算力开销）
def pure_code_realizer(fact_json_str: str) -> str:
    try:
        # 去除 markdown 标记
        clean_json = fact_json_str.strip().replace("```json", "").replace("```", "")
        facts = json.loads(clean_json)
        # 模板化安全装配（100% 确定性，防篡改）
        ans = (
            f"根据最新规约规定："
            f"1. 三级公立医院普通门诊报销比例为 {facts.get('lv3_ratio', '未知')}；"
            f"2. 特殊门诊（如恶性肿瘤放化疗）报销比例为 {facts.get('special_clinic_ratio', '未知')} 且免起付线；"
            f"3. 异地就医政策："
            f"{'统筹基金一票否决不予报销' if not facts.get('remote_medical_covered', True) else '可按规定报销'}。"
        )
        return ans
    except Exception:
        return "规则装配异常，触发保底合规提示。"


async def run_single_request(client: httpx.AsyncClient, prompt: str, max_tokens: int):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0
        }
    }
    t0 = time.perf_counter()
    first_token_t = None
    tokens = 0
    full_text = []

    async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
        async for line in resp.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if first_token_t is None:
                first_token_t = time.perf_counter()
            tokens += 1
            full_text.append(chunk.get("response", ""))
            if chunk.get("done", False):
                break

    t1 = time.perf_counter()
    return {
        "ttft": (first_token_t - t0) if first_token_t else 0.0,
        "total_latency": t1 - t0,
        "tokens": tokens,
        "text": "".join(full_text)
    }


async def test_mode(mode_name: str, prompt: str, max_tokens: int, concurrency: int = 4, is_two_stage: bool = False):
    print(f"\n==========================================")
    print(f"  正在测试模式: {mode_name} (并发 C={concurrency})")
    print(f"==========================================")

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=4)
    async with httpx.AsyncClient(limits=limits) as client:
        start_wall = time.perf_counter()
        tasks = [run_single_request(client, prompt, max_tokens) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        
        # 若为两段式，执行第二阶段（纯代码组装）
        code_times = []
        assembled_texts = []
        if is_two_stage:
            for r in results:
                t_code_start = time.perf_counter()
                assembled = pure_code_realizer(r["text"])
                code_times.append(time.perf_counter() - t_code_start)
                assembled_texts.append(assembled)

        wall_time = time.perf_counter() - start_wall

    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_latency = sum(r["total_latency"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    tps = total_tokens / wall_time

    print(f"平均首字延迟 (TTFT): {avg_ttft:.3f} s")
    print(f"平均单请求端到端耗时: {avg_latency:.3f} s")
    print(f"单请求平均生成 Token 数: {total_tokens / len(results):.1f} tokens")
    print(f"系统整体吞吐量: {tps:.2f} tokens/s")
    print(f"总墙钟耗时: {wall_time:.2f} s")
    if is_two_stage and code_times:
        print(f"阶段二纯代码装配耗时: 平均 {sum(code_times)/len(code_times)*1000:.3f} 毫秒")
        print("\n[样例抽取输出 JSON] ->", results[0]["text"].strip())
        print("[样例纯代码组装文本] ->", assembled_texts[0])
    elif not is_two_stage and results:
        print("\n[单阶段直接生成样例片选] ->", results[0]["text"].strip()[:150], "...")


async def main():
    # 1. 跑对照组：传统单阶段生成（限制 250 tokens 解答）
    await test_mode("对照组: 传统单阶段散文生成", SINGLE_STAGE_PROMPT, max_tokens=250, concurrency=4, is_two_stage=False)

    # 预留 2 秒让显存和引擎沉降
    await asyncio.sleep(2)

    # 2. 跑实验组：Aegis-RAG 两段式（抽取受限 JSON + 纯代码 Realizer）
    await test_mode("实验组: Aegis-RAG 两段式流水线", EXTRACTION_PROMPT, max_tokens=80, concurrency=4, is_two_stage=True)


if __name__ == "__main__":
    asyncio.run(main())
