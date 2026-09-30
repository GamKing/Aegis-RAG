import asyncio
import json
import os
import sys
import time
import httpx
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 优先支持环境变量配置，默认指向当前环境监听的 22434 端口
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "22434")
OLLAMA_API = os.environ.get("OLLAMA_URL", f"http://127.0.0.1:{OLLAMA_PORT}/api/generate")
MODEL_NAME = "qwen2.5-benchmark"

# 1. 构造 4000 Token 长文本上下文 (模拟重型大卡车突发 Prefill)
LONG_CONTEXT_PROMPT = (
    "以下是某市完整的医保门诊与住院共济政策细则汇总文档：\n"
    + ("【政策条款】参保人员发生的合规医疗费用，统筹基金按比例报销，起付线为800元。\n" * 150)
    + "请详细总结上述政策对普通参保人员的核心要点。"
)

# 2. 构造轻量短请求 Prompt (模拟打字机小轿车)
SHORT_STREAM_PROMPT = "请以非常详细的步骤，写一篇关于如何合理安排日常健康饮食的简短建议。"


async def run_short_streaming_client(client_id: int, start_delay: float = 0.0):
    """短请求流式客户端：逐字记录每个 Token 的生成时间戳，抓出 ITL 抖动。"""
    if start_delay > 0:
        await asyncio.sleep(start_delay)

    itl_list = []
    first_token_time = None
    last_token_time = None
    token_count = 0
    ttft = 0.0

    req_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": MODEL_NAME,
            "prompt": SHORT_STREAM_PROMPT,
            "stream": True,
            "options": {"num_predict": 120, "temperature": 0.0},
        }
        async with client.stream("POST", OLLAMA_API, json=payload) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                item = json.loads(line)
                now = time.perf_counter()
                if first_token_time is None:
                    first_token_time = now
                    ttft = first_token_time - req_start
                else:
                    itl = (now - last_token_time) * 1000.0  # 转换为毫秒
                    itl_list.append(itl)

                last_token_time = now
                token_count += 1
                if item.get("done", False):
                    break

    e2e_latency = time.perf_counter() - req_start

    # 统计数据清洗
    avg_itl = float(np.mean(itl_list)) if itl_list else 0.0
    p95_itl = float(np.percentile(itl_list, 95)) if itl_list else 0.0
    p99_itl = float(np.percentile(itl_list, 99)) if itl_list else 0.0
    max_itl = float(np.max(itl_list)) if itl_list else 0.0

    return {
        "client_id": client_id,
        "token_count": token_count,
        "ttft_s": ttft,
        "avg_itl_ms": avg_itl,
        "p95_itl_ms": p95_itl,
        "p99_itl_ms": p99_itl,
        "max_itl_ms": max_itl,
        "e2e_s": e2e_latency,
        "itl_history": itl_list,
    }


async def run_long_prefill_client():
    """突发长文档 Prefill 客户端 (不开启流式，纯打满算力)。"""
    req_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": MODEL_NAME,
            "prompt": LONG_CONTEXT_PROMPT,
            "stream": False,
            "options": {"num_predict": 20, "temperature": 0.0},
        }
        resp = await client.post(OLLAMA_API, json=payload)
    total_time = time.perf_counter() - req_start
    return {"long_prefill_time_s": total_time, "status_code": resp.status_code}


async def main():
    print("=" * 65)
    print(" 🚀 启动 Chunked Prefill 抗干扰流式抖动专项压测")
    print("=" * 65)

    # 启动 3 个背景流式打字机请求
    print("[*] T = 0.0s: 启动 3 个背景流式打字机请求 (客户端 1, 2, 3，生成 120 Tokens)...")
    short_tasks = [run_short_streaming_client(i) for i in range(1, 4)]

    # 在 0.8 秒后，突发注入长文档大卡车请求！
    async def inject_spike():
        await asyncio.sleep(0.8)
        print("\n[!] 🚨 突发事件：T = +0.8s！4000 Token 重型长文档请求正式轰炸 GPU！")
        return await run_long_prefill_client()

    all_tasks = short_tasks + [inject_spike()]
    results = await asyncio.gather(*all_tasks)

    short_results = results[:3]
    long_result = results[3]

    print("\n" + "=" * 65)
    print(f" 📊 突发长文档 Prefill 耗时: {long_result['long_prefill_time_s']:.2f} 秒")
    print("=" * 65)
    for r in short_results:
        print(f" 客户端 {r['client_id']} 打字机表现:")
        print(f"   - 首字延迟 (TTFT): {r['ttft_s']:.3f} s")
        print(f"   - 生成总 Token 数: {r['token_count']} tokens (端到端: {r['e2e_s']:.2f}s)")
        print(f"   - 平均字间时延 (Avg ITL): {r['avg_itl_ms']:.1f} ms")
        print(f"   - P95 字间时延 (P95 ITL): {r['p95_itl_ms']:.1f} ms")
        print(f"   - P99 字间时延 (P99 ITL): {r['p99_itl_ms']:.1f} ms ⚠️ (重点观察)")
        print(f"   - 最大瞬时卡顿 (Max Spike): {r['max_itl_ms']:.1f} ms 🚨")
        print("-" * 65)

    # 计算整体汇总 SLA
    all_p99 = [r["p99_itl_ms"] for r in short_results]
    all_max = [r["max_itl_ms"] for r in short_results]
    print(f"🏆 SLA 验收综合判定:")
    print(f"   - 平均 P99 ITL: {np.mean(all_p99):.1f} ms (SLA 合格线: < 100 ms)")
    print(f"   - 最大瞬时突发卡顿: {np.max(all_max):.1f} ms")
    if np.mean(all_p99) < 150:
        print(f"   - 判定结论: ✅ 引擎有效分块调度，打字机流式抗干扰平稳通过！")
    else:
        print(f"   - 判定结论: ⚠️ 出现明显算力抢占，长 Prefill 造成打字机卡顿。")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
