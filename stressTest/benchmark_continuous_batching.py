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

# 推理服务端口配置
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "22434")
OLLAMA_URL = os.environ.get("OLLAMA_URL", f"http://localhost:{OLLAMA_PORT}/api/generate")
MODEL_NAME = "qwen2.5-benchmark"

# 同样的长文档上下文（~3000 Tokens）
MOCK_CONTEXT = """
【国家基本医疗保险与门诊报销管理规约（2026年修订版）】
第一条 参保人员在定点甲等公立医疗机构普通门诊就诊，起付标准为个人自付累计满 1500 元。
第二条 超过起付标准部分，在一级医疗机构报销比例为 85%，二级医疗机构报销比例为 75%，三级医疗机构报销比例为 60%。
第三条 退休人员在上述报销比例基础上各档次分别上调 5%。直系亲属共济账户可用于支付个人自负部分。
第四条 恶性肿瘤放化疗、器官移植抗排异治疗等特殊门诊，不设起付线，统筹基金合规报销比例固定为 90%。
第五条 境外就医及未经转诊异地就医产生的医疗费用，统筹基金一票否决不予报销。
""" * 30

SHORT_PROMPT = f"{MOCK_CONTEXT}\n\n简短回答：三级医院普通门诊在职职工报销比例是多少？请用一句话回答。"
LONG_PROMPT = f"{MOCK_CONTEXT}\n\n请详细列出各级医院报销比例、特殊门诊待遇、退休人员补偿及异地拒赔规则，并撰写一份详尽的长篇政策解读报告。"


async def send_request(client: httpx.AsyncClient, req_id: str, prompt: str, max_tokens: int, start_reference: float):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
        },
    }
    
    t_send = time.perf_counter()
    first_token_t = None
    tokens = 0
    full_text = []

    try:
        async with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                tokens += 1
                full_text.append(item.get("response", ""))
                if item.get("done", False):
                    break
        t_done = time.perf_counter()
        
        return {
            "req_id": req_id,
            "tokens": tokens,
            "send_offset": t_send - start_reference,
            "ttft_relative": (first_token_t - start_reference) if first_token_t else 0.0,
            "ttft_from_send": (first_token_t - t_send) if first_token_t else 0.0,
            "e2e_from_send": t_done - t_send,
            "done_relative": t_done - start_reference,
            "text_preview": "".join(full_text)[:80].replace("\n", " "),
        }
    except Exception as e:
        print(f"[{req_id}] 异常: {e}")
        return None


async def run_static_batching_simulation():
    """测试场景 A：模拟传统 Static Batching（整批等待 / 队头木桶阻塞）。
    
    规则：
    - 槽位 1~3 运行短请求 (25 tokens)；
    - 槽位 4 运行长木桶请求 (250 tokens)；
    - 新用户【请求 5】在第 0.2 秒到达，但因为是静态整批调度，必须等待整批 (1~4) 全部完成才能上车！
    """
    print("\n" + "=" * 65)
    print("  【场景 A 压测】：模拟传统 Static Batching（整批静态等待 / 队头木桶阻塞）")
    print("=" * 65)

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits) as client:
        start_global = time.perf_counter()
        
        # 1. 发射批次：3 个短任务 + 1 个长木桶任务
        batch_tasks = [
            send_request(client, "槽位1(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global),
            send_request(client, "槽位2(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global),
            send_request(client, "槽位3(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global),
            send_request(client, "槽位4(长木桶)", LONG_PROMPT, max_tokens=250, start_reference=start_global),
        ]
        
        # 模拟新用户【请求 5】到达时刻
        req5_arrival_offset = 0.2
        print(f"[*] 初始批次启动：槽位1~3 为短请求 (25 tokens)，槽位4 为长请求 (250 tokens)")
        print(f"[*] 门外等待者：新用户【请求 5】在 T = +{req5_arrival_offset:.1f}s 到达队列")
        print(f"[*] [静态批处理策略]：槽位1~3 跑完后显存被锁死 Padding，强制等待最慢的槽位4！")

        # 等待整批全部完成
        batch_results = await asyncio.gather(*batch_tasks)
        t_batch_done = time.perf_counter()
        
        # 整批跑完后，才允许向显卡下发请求 5
        print(f"[!] 静态整批跑完！耗时 {t_batch_done - start_global:.2f}s。此时才放行【请求 5】上车...")
        req5_res = await send_request(client, "请求5(新用户)", SHORT_PROMPT, max_tokens=25, start_reference=start_global)

    # 统计打印
    for r in batch_results:
        print(f"  - {r['req_id']}: 生成 {r['tokens']} tokens | 完成时间 T=+{r['done_relative']:.2f}s | 端到端 {r['e2e_from_send']:.2f}s")
    
    # 用户 5 的真实感受（从他第 0.2 秒到达，到拿到首字和完成）
    req5_perceived_ttft = req5_res["ttft_relative"] - req5_arrival_offset
    req5_perceived_e2e = req5_res["done_relative"] - req5_arrival_offset
    print("-" * 65)
    print(f"🚨 【请求 5 (新用户) 惨痛体验】:")
    print(f"  - 用户实际体感首字等待 (TTFT): {req5_perceived_ttft:.2f} 秒 (被槽位4长请求硬生生拖垮!)")
    print(f"  - 用户实际体感端到端总耗时: {req5_perceived_e2e:.2f} 秒")
    print(f"  - 显存/槽位闲置浪费: 槽位1~3 闲置干等了约 {batch_results[3]['done_relative'] - batch_results[0]['done_relative']:.2f} 秒！")

    return {
        "req5_ttft": req5_perceived_ttft,
        "req5_e2e": req5_perceived_e2e,
        "long_req_duration": batch_results[3]["e2e_from_send"],
    }


async def run_continuous_batching_test():
    """测试场景 B：现代 Continuous Batching（迭代级动态插队调度）。
    
    规则：
    - 槽位 1~3 运行短请求 (25 tokens)；
    - 槽位 4 运行长木桶请求 (250 tokens)；
    - 新用户【请求 5】在第 0.2 秒到达，直接提交到推理队列；
    - 一旦槽位 1 跑完 (约 0.8s)，调度器立刻踢出槽位 1，抓取【请求 5】插队无缝上车！
    """
    print("\n" + "=" * 65)
    print("  【场景 B 压测】：现代 Continuous Batching（迭代级动态插队 / 槽位即时回收）")
    print("=" * 65)

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits) as client:
        start_global = time.perf_counter()
        
        # 1. 并发发射前 4 个任务
        t1 = asyncio.create_task(send_request(client, "槽位1(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global))
        t2 = asyncio.create_task(send_request(client, "槽位2(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global))
        t3 = asyncio.create_task(send_request(client, "槽位3(短)", SHORT_PROMPT, max_tokens=25, start_reference=start_global))
        t4 = asyncio.create_task(send_request(client, "槽位4(长木桶)", LONG_PROMPT, max_tokens=250, start_reference=start_global))

        # 2. 模拟新用户【请求 5】在第 0.2 秒到达并直接提交
        await asyncio.sleep(0.2)
        req5_arrival_offset = 0.2
        print(f"[*] 初始批次启动：槽位1~3 (短请求)，槽位4 (长请求)")
        print(f"[*] ⚡ [动态插队]：新用户【请求 5】在 T = +0.2s 到达，直接注入底层调度器！")
        t5 = asyncio.create_task(send_request(client, "请求5(插队用户)", SHORT_PROMPT, max_tokens=25, start_reference=start_global))

        # 等待所有任务并发动态完成
        results = await asyncio.gather(t1, t2, t3, t4, t5)

    # 排序打印各请求完成时间
    results_sorted = sorted(results, key=lambda x: x["done_relative"])
    for r in results_sorted:
        print(f"  - {r['req_id']}: 生成 {r['tokens']} tokens | 完成时间 T=+{r['done_relative']:.2f}s | 首字吐出 T=+{r['ttft_relative']:.2f}s")

    # 提取请求 5 的数据
    req5 = next(r for r in results if "请求5" in r["req_id"])
    req5_perceived_ttft = req5["ttft_relative"] - req5_arrival_offset
    req5_perceived_e2e = req5["done_relative"] - req5_arrival_offset
    
    print("-" * 65)
    print(f"🎉 【请求 5 (连续批处理动态插队) 极速体验】:")
    print(f"  - 用户实际体感首字等待 (TTFT): {req5_perceived_ttft:.2f} 秒 (槽位1刚释放，立即插队上车！)")
    print(f"  - 用户实际体感端到端总耗时: {req5_perceived_e2e:.2f} 秒")
    print(f"  - 对比长请求 4 还在吭哧吭哧跑: 此时槽位4完成时间为 T=+{results[3]['done_relative']:.2f}s！")

    return {
        "req5_ttft": req5_perceived_ttft,
        "req5_e2e": req5_perceived_e2e,
        "slot1_done": results[0]["done_relative"],
        "long_req_done": results[3]["done_relative"],
    }


async def main():
    # 场景 A: 静态批处理模拟
    res_static = await run_static_batching_simulation()
    
    # 冷却 2 秒
    await asyncio.sleep(2)
    
    # 场景 B: 连续批处理动态插队实测
    res_continuous = await run_continuous_batching_test()

    # 最终综合对比
    print("\n" + "=" * 65)
    print("          🏆 Continuous Batching 核心收益大比拼")
    print("=" * 65)
    speedup_ttft = res_static["req5_ttft"] / res_continuous["req5_ttft"] if res_continuous["req5_ttft"] > 0 else 0
    speedup_e2e = res_static["req5_e2e"] / res_continuous["req5_e2e"] if res_continuous["req5_e2e"] > 0 else 0
    print(f"1. 新用户【请求 5】首字时延 (TTFT):")
    print(f"   - 传统 Static Batching: {res_static['req5_ttft']:.2f} 秒 (死等整批长木桶)")
    print(f"   - 现代 Continuous Batching: {res_continuous['req5_ttft']:.2f} 秒 (短请求一走，立刻插队)")
    print(f"   - 🚀 TTFT 首字等待提速: {speedup_ttft:.2f} 倍 (时延暴降 {(1 - res_continuous['req5_ttft']/res_static['req5_ttft'])*100:.1f}%)")
    print(f"\n2. 新用户【请求 5】端到端总耗时 (E2E Latency):")
    print(f"   - 传统 Static Batching: {res_static['req5_e2e']:.2f} 秒")
    print(f"   - 现代 Continuous Batching: {res_continuous['req5_e2e']:.2f} 秒")
    print(f"   - 🚀 端到端交互提速: {speedup_e2e:.2f} 倍 (时延暴降 {(1 - res_continuous['req5_e2e']/res_static['req5_e2e'])*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
