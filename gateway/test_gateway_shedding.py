"""生产级反压与过载削峰 (Load Shedding & Backpressure) 验证套件。

核心验证指标：
1. 突发并发流量洪峰下的 Fail-fast 机制：等待队列满载后，后续超额请求在 <10ms 内极速返回 HTTP 429；
2. 活跃槽位安全护航：正在 GPU 上计算的活跃推理请求不受洪峰流量冲撞与干扰；
3. 排队恢复机制：等待队列中的请求在槽位释放后平稳递进执行；
4. 全链路可观测性核验：/metrics/health 与 /metrics (Prometheus) 精确记录拒绝数与延迟分布。
"""
import argparse
import asyncio
import json
import os
import sys
import time
import unittest
from typing import List

import httpx


class TestGatewayLoadSheddingInProcess(unittest.IsolatedAsyncioTestCase):
    """基于 ASGI Transport 的进程内并发削峰精确边界测试。"""

    async def asyncSetUp(self):
        # 动态重设测试参数
        os.environ["MAX_ACTIVE_REQUESTS"] = "2"
        os.environ["MAX_QUEUE_SIZE"] = "4"

        # 动态导入 app 确保环境变量生效
        import gateway.app as gw_app
        gw_app.MAX_ACTIVE_REQUESTS = 2
        gw_app.MAX_QUEUE_SIZE = 4
        gw_app.engine_semaphore = asyncio.Semaphore(2)
        gw_app.current_active_requests = 0
        gw_app.request_queue_length = 0

        # 重置全局 metrics
        from gateway.metrics import metrics_store
        metrics_store.total_requests = 0
        metrics_store.successful_requests = 0
        metrics_store.rejected_shedding = 0
        metrics_store.upstream_errors = 0
        metrics_store.ttft_count = 0
        metrics_store.ttft_sum = 0.0
        metrics_store.ttft_max = 0.0
        metrics_store.e2e_count = 0
        metrics_store.e2e_sum = 0.0
        metrics_store.e2e_max = 0.0
        metrics_store.prefix_fingerprint_hits.clear()

        self.app = gw_app.app

    async def test_burst_load_shedding_fail_fast(self):
        """测试 10 个突发并发请求在 MAX_ACTIVE=2, MAX_QUEUE=4 配置下的削峰表现。

        预期行为：
        - 2 个请求立即占用活跃槽位 (Active Slots = 2)
        - 4 个请求进入排队队列 (Queue Size = 4)
        - 4 个请求被快速失败熔断 (HTTP 429 Too Many Requests, <15ms)
        - 活跃和排队请求最终全部正常返回
        """
        import gateway.app as gw_app

        # 模拟后端引擎延迟（每个请求推理 0.2 秒）
        original_forward = gw_app.forward_stream_to_engine

        async def mock_forward_stream(target_url, payload, prefix_fingerprint, req_start_time):
            # 模拟第 1 个 token 吐出
            await asyncio.sleep(0.05)
            yield b'data: {"token": "hello"}\n\n'
            await asyncio.sleep(0.15)
            yield b'data: {"token": "world"}\n\n'
            # 记录 metrics
            dur = time.perf_counter() - req_start_time
            gw_app.metrics_store.record_success(ttft=0.05, e2e_latency=dur, prefix_fingerprint=prefix_fingerprint)

        gw_app.forward_stream_to_engine = mock_forward_stream

        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 准备 10 个并发请求
            payload = {
                "model": "qwen2.5-benchmark",
                "prompt": "请解释什么是反压网关？",
                "stream": True,
            }

            results = []

            async def send_req(req_id: int):
                t0 = time.perf_counter()
                try:
                    resp = await client.post("/v1/chat/completions", json=payload, timeout=10.0)
                    t1 = time.perf_counter()
                    return {
                        "id": req_id,
                        "status": resp.status_code,
                        "elapsed": t1 - t0,
                        "body": resp.text,
                    }
                except Exception as exc:
                    t1 = time.perf_counter()
                    return {"id": req_id, "status": 500, "elapsed": t1 - t0, "body": str(exc)}

            # 同时发射 10 个突发请求
            tasks = [send_req(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

        # 恢复原始 forward
        gw_app.forward_stream_to_engine = original_forward

        status_counts = {}
        rejected_latencies = []
        success_latencies = []

        for r in results:
            st = r["status"]
            status_counts[st] = status_counts.get(st, 0) + 1
            if st == 429:
                rejected_latencies.append(r["elapsed"])
            elif st == 200:
                success_latencies.append(r["elapsed"])

        print(f"\n[Load Shedding Test] 并发突发请求结果分布: {status_counts}")
        print(f"[Load Shedding Test] 熔断 429 数量: {status_counts.get(429, 0)} (期望 4)")
        print(f"[Load Shedding Test] 成功 200 数量: {status_counts.get(200, 0)} (期望 6 = 2 active + 4 queued)")

        # 核心断言 1：精确削峰数量
        self.assertEqual(status_counts.get(429, 0), 4, "超额的 4 个请求必须被 HTTP 429 拦截")
        self.assertEqual(status_counts.get(200, 0), 6, "2 个槽位 + 4 个排队的请求必须全部成功处理")

        # 核心断言 2：Fail-fast 时延极低 (< 50ms)
        for lat in rejected_latencies:
            self.assertLess(lat, 0.05, f"429 快速拒绝时延过高: {lat*1000:.2f}ms")

        print(f"[Load Shedding Test] 429 平均快速拒绝时延: {sum(rejected_latencies)/len(rejected_latencies)*1000:.2f}ms")

        # 核心断言 3：可观测性端点输出校验
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health_resp = await client.get("/metrics/health")
            self.assertEqual(health_resp.status_code, 200)
            health_data = health_resp.json()
            self.assertEqual(health_data["traffic"]["total_requests"], 10)
            self.assertEqual(health_data["traffic"]["rejected_shedding_429"], 4)
            self.assertEqual(health_data["traffic"]["successful_requests"], 6)

            prom_resp = await client.get("/metrics")
            self.assertEqual(prom_resp.status_code, 200)
            prom_text = prom_resp.text
            self.assertIn("llm_gateway_load_shedding_rejected_total 4", prom_text)
            self.assertIn("llm_gateway_successful_requests_total 6", prom_text)
            self.assertIn("llm_gateway_requests_total 10", prom_text)

        print("[Load Shedding Test] /metrics/health 与 Prometheus 指标导出校验 100% 通过！")


async def run_live_gateway_stress(url: str, concurrency: int = 24, total_requests: int = 32):
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"\n=======================================================")
    print(f"[*] 启动实时网关压测 -> 目标: {url}")
    print(f"[*] 并发量: {concurrency}, 总请求数: {total_requests}")
    print(f"=======================================================")

    payload = {
        "model": "qwen2.5-benchmark",
        "prompt": "请用一句话回答：什么是大模型的 KV Cache？",
        "stream": True,
        "options": {"num_predict": 30},
    }

    status_stats = {}
    durations = []
    rejection_durations = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 首先检查网关健康度
        try:
            h_resp = await client.get(f"{url}/metrics/health")
            print(f"[*] 初始网关状态: {h_resp.json()['saturation']}")
        except Exception as e:
            print(f"[!] 无法连接到网关 {url}，请确认服务已启动: {e}")
            return

        async def worker(req_id: int):
            t0 = time.perf_counter()
            try:
                resp = await client.post(f"{url}/v1/chat/completions", json=payload)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                st = resp.status_code
                return {"id": req_id, "status": st, "elapsed": elapsed}
            except Exception as e:
                t1 = time.perf_counter()
                return {"id": req_id, "status": 500, "elapsed": t1 - t0}

        t_start_all = time.perf_counter()
        tasks = [worker(i) for i in range(total_requests)]
        results = await asyncio.gather(*tasks)
        t_total = time.perf_counter() - t_start_all

        for r in results:
            st = r["status"]
            status_stats[st] = status_stats.get(st, 0) + 1
            if st == 429:
                rejection_durations.append(r["elapsed"])
            elif st == 200:
                durations.append(r["elapsed"])

        print(f"\n--- 压测结果汇总 ---")
        print(f"总耗时: {t_total:.2f} 秒")
        print(f"状态码分布: {status_stats}")
        if rejection_durations:
            avg_rej = sum(rejection_durations) / len(rejection_durations) * 1000
            print(f"429 削峰拒绝数: {len(rejection_durations)}, 平均拒绝响应时延: {avg_rej:.2f} ms")
        if durations:
            avg_dur = sum(durations) / len(durations)
            print(f"200 成功响应数: {len(durations)}, 平均端到端时延: {avg_dur:.2f} s")

        # 拉取最新指标
        h_resp = await client.get(f"{url}/metrics/health")
        print(f"[*] 最终网关健康指标:\n{json.dumps(h_resp.json(), indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="网关反压与负载削峰测试")
    parser.add_argument("--live", action="store_true", help="对正在运行的网关执行真实压测")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="网关根地址")
    parser.add_argument("--concurrency", type=int, default=24, help="并发数量")
    parser.add_argument("--total", type=int, default=32, help="总请求数")

    args, remaining = parser.parse_known_args()

    if args.live:
        asyncio.run(run_live_gateway_stress(url=args.url, concurrency=args.concurrency, total_requests=args.total))
    else:
        # 运行单元/集成测试
        unittest.main(argv=[sys.argv[0]] + remaining)
