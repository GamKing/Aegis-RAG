import asyncio
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

# 构造约 3000 字的医保与财务规约长上下文（模拟真实 RAG 注入）
MOCK_CONTEXT = """
【国家基本医疗保险与门诊报销管理规约（2026年修订版）】
第一条 参保人员在定点甲等公立医疗机构普通门诊就诊，起付标准为个人自付累计满 1500 元。
第二条 超过起付标准部分，在一级医疗机构报销比例为 85%，二级医疗机构报销比例为 75%，三级医疗机构报销比例为 60%。
第三条 退休人员在上述报销比例基础上各档次分别上调 5%。直系亲属共济账户可用于支付个人自负部分。
第四条 恶性肿瘤放化疗、器官移植抗排异治疗等特殊门诊，不设起付线，统筹基金合规报销比例固定为 90%。
第五条 境外就医及未经转诊异地就医产生的医疗费用，统筹基金一票否决不予报销。
""" * 30  # 复制30次，构造高长度 Prefill 压力

BENCHMARK_PROMPT = f"{MOCK_CONTEXT}\n\n请详细列出三级医院在职人员的报销比例、特殊门诊待遇以及异地就医的拒赔规则。"


async def send_single_request(client: httpx.AsyncClient, req_id: int):
    payload = {
        "model": MODEL_NAME,
        "prompt": BENCHMARK_PROMPT,
        "stream": True,
        "options": {
            "num_predict": 250  # 限制生成 250 字左右，贴合 RAG 答案长度
        }
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    generated_tokens = 0

    try:
        async with client.stream("POST", OLLAMA_URL, json=payload, timeout=180.0) as response:
            if response.status_code != 200:
                print(f"[Req {req_id}] 错误状态码: {response.status_code}")
                return None

            async for line in response.aiter_lines():
                if not line:
                    continue
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                generated_tokens += 1

        end_time = time.perf_counter()
        ttft = (first_token_time - start_time) if first_token_time else 0.0
        total_time = end_time - start_time
        tps = generated_tokens / total_time if total_time > 0 else 0.0

        return {
            "req_id": req_id,
            "ttft": ttft,
            "total_time": total_time,
            "tokens": generated_tokens,
            "tps": tps
        }
    except Exception as e:
        print(f"[Req {req_id}] 请求异常: {e}")
        return None


async def run_batch(concurrency: int):
    print(f"\n==========================================")
    print(f"  开始压测: 并发量 (Concurrency) = {concurrency}")
    print(f"==========================================")
    
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [send_single_request(client, i + 1) for i in range(concurrency)]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("所有请求均未成功完成。")
        return

    avg_ttft = sum(r["ttft"] for r in valid_results) / len(valid_results)
    max_ttft = max(r["ttft"] for r in valid_results)
    total_tokens = sum(r["tokens"] for r in valid_results)
    system_throughput = total_tokens / wall_time

    print(f"成功完成请求: {len(valid_results)}/{concurrency}")
    print(f"平均首字延迟 (Avg TTFT): {avg_ttft:.3f} 秒")
    print(f"最大首字延迟 (Max TTFT): {max_ttft:.3f} 秒")
    print(f"系统整体吞吐量 (Throughput): {system_throughput:.2f} tokens/s")
    print(f"总压测墙钟耗时 (Wall Time): {wall_time:.2f} 秒")


if __name__ == "__main__":
    # 依次压测 1、2、4 三个梯度
    for c in [1, 2, 4]:
        asyncio.run(run_batch(c))
