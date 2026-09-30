"""生产级高可用反压与自适应削峰网关 (LLM Production Resilient Gateway).

核心特性：
1. 并发槽位硬约束 (Concurrency Semaphore)：严格限制打入引擎的并发数不超过安全阈值
2. 队列过载秒级削峰 (Load Shedding)：等待队列超限立即触发 Fail-fast (HTTP 429)，守护活跃请求
3. 前缀感知路由 (Prefix-Aware Routing)：基于一致性哈希将长前缀请求归拢至同一 GPU 节点
4. 全链路可观测性 (Prometheus Metrics)：四大黄金指标、TTFT/E2E 延时分布与健康状态输出
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .metrics import metrics_store
from .router import PrefixAwareRouter

# ----------------- 网关弹性控制参数 (支持环境变量注入) -----------------
MAX_ACTIVE_REQUESTS = int(os.environ.get("MAX_ACTIVE_REQUESTS", "8"))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "16"))

# 推理后端节点地址（支持多节点逗号分隔，如 "http://127.0.0.1:22434/api/generate"）
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "22434")
DEFAULT_HOST = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"
BACKEND_HOSTS = [h.strip() for h in os.environ.get("INFERENCE_ENGINE_HOSTS", DEFAULT_HOST).split(",") if h.strip()]

# 并发控制原语
engine_semaphore = asyncio.Semaphore(MAX_ACTIVE_REQUESTS)
current_active_requests = 0
request_queue_length = 0

# 节点级细粒度活跃槽位跟踪 (用于热点防倾斜与主动缓存复制)
node_active_slots: dict[str, int] = {node: 0 for node in BACKEND_HOSTS}


def is_node_available(node: str) -> bool:
    """检查节点当前活跃并发数是否小于单节点活跃上限。"""
    return node_active_slots.get(node, 0) < MAX_ACTIVE_REQUESTS


# 前缀感知路由器
router = PrefixAwareRouter(BACKEND_HOSTS)

# 全局高复用 HTTP 客户端连接池
http_limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None or http_client.is_closed:
        http_client = httpx.AsyncClient(limits=http_limits, timeout=httpx.Timeout(180.0, connect=10.0))
    return http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_http_client()
    yield
    global http_client
    if http_client and not http_client.is_closed:
        await http_client.aclose()


app = FastAPI(
    title="Aegis-RAG Resilient Gateway",
    description="大模型生产级高可用反压网关，提供 Load Shedding、前缀感知路由与全链路可观测性度量",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    model: str = Field(default="qwen2.5-benchmark", description="目标模型名称")
    prompt: str = Field(description="用户输入提示词或检索上下文拼接文本")
    stream: bool = Field(default=True, description="是否开启 SSE 流式输出")
    options: dict = Field(default_factory=dict, description="采样与长度控制参数 (如 num_predict)")


@app.middleware("http")
async def track_queue_and_shedding(request: Request, call_next):
    """网关反压与过载削峰 (Load Shedding) 中间件。

    物理防护机制：
    1. 快速失败保护 (Fail-fast)：当等待队列超限时，在 <1ms 内直接抛出 HTTP 429；
    2. 绝不盲目接收请求打爆后端显存；
    3. 守护已经在跑的活跃请求不受并发抖动影响。
    """
    global request_queue_length, current_active_requests

    if request.url.path not in ("/v1/chat/completions", "/api/generate"):
        return await call_next(request)

    metrics_store.record_request_start()

    t_mw_entry = time.perf_counter()

    # 1. 快速失败保护: 队列超限直接拒绝，阻止雪崩打穿显存
    if request_queue_length >= MAX_QUEUE_SIZE:
        metrics_store.record_rejection()
        shedding_latency_ms = (time.perf_counter() - t_mw_entry) * 1000.0
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "Engine capacity saturated (Load shedding triggered)",
                "code": "CAPACITY_EXCEEDED",
                "message": f"排队深度已达安全上限 ({MAX_QUEUE_SIZE})，网关主动熔断保护后端显存，请稍后重试。",
                "active_slots": f"{current_active_requests}/{MAX_ACTIVE_REQUESTS}",
                "pending_queue": f"{request_queue_length}/{MAX_QUEUE_SIZE}",
                "shedding_latency_ms": round(shedding_latency_ms, 3),
            },
            headers={
                "x-llm-gateway-active-slots": str(current_active_requests),
                "x-llm-gateway-pending-queue": str(request_queue_length),
                "x-llm-shedding-time-ms": f"{shedding_latency_ms:.3f}",
                "retry-after": "1",
            },
        )

    request_queue_length += 1
    t_queue_enter = time.perf_counter()

    try:
        # 2. 排队等待空闲算力槽位
        await engine_semaphore.acquire()
    except Exception:
        request_queue_length -= 1
        metrics_store.record_upstream_error()
        raise

    request_queue_length -= 1
    current_active_requests += 1
    t_slot_acquired = time.perf_counter()
    request.state.queue_wait_seconds = t_slot_acquired - t_queue_enter

    slot_released = False

    def release_slot():
        nonlocal slot_released
        if not slot_released:
            slot_released = True
            global current_active_requests
            current_active_requests -= 1
            engine_semaphore.release()

    try:
        response = await call_next(request)
        orig_body_iterator = response.body_iterator

        async def wrapped_body_iterator():
            try:
                async for chunk in orig_body_iterator:
                    yield chunk
            finally:
                release_slot()

        response.body_iterator = wrapped_body_iterator()
        return response
    except Exception:
        release_slot()
        metrics_store.record_upstream_error()
        raise


async def forward_stream_to_engine(
    target_url: str,
    payload: dict,
    prefix_fingerprint: str,
    req_start_time: float,
) -> AsyncGenerator[bytes, None]:
    """流式转发并透传推理引擎输出，保持最低开销与平滑 ITL。"""
    first_token_time = None
    client = get_http_client()

    try:
        async with client.stream("POST", target_url, json=payload) as upstream_resp:
            if upstream_resp.status_code != 200:
                metrics_store.record_upstream_error()
                err_msg = json.dumps({"error": f"Upstream error HTTP {upstream_resp.status_code}"})
                yield f"data: {err_msg}\n\n".encode("utf-8")
                return

            async for chunk in upstream_resp.aiter_bytes():
                if first_token_time is None and chunk:
                    first_token_time = time.perf_counter()
                yield chunk

        e2e_duration = time.perf_counter() - req_start_time
        ttft = (first_token_time - req_start_time) if first_token_time else 0.0
        metrics_store.record_success(ttft=ttft, e2e_latency=e2e_duration, prefix_fingerprint=prefix_fingerprint)
    except Exception as exc:
        metrics_store.record_upstream_error()
        err_msg = json.dumps({"error": f"Stream forwarding failed: {str(exc)}"})
        yield f"data: {err_msg}\n\n".encode("utf-8")
    finally:
        node_active_slots[target_url] = max(0, node_active_slots.get(target_url, 1) - 1)


@app.post("/v1/chat/completions")
@app.post("/api/generate")
async def handle_chat_completion(payload: ChatRequest, request: Request):
    """统一生产级路由转发入口 (支持流式 SSE、动态负载溢出保护与主动缓存分裂)。"""
    req_start = time.perf_counter()

    # 1. 前缀感知与负载感知动态路由决策 (Load-Aware Fallback / Cache Replication)
    decision = router.route(payload.prompt, node_load_checker=is_node_available)
    target_url = decision.target_node
    prefix_fingerprint = decision.prefix_fingerprint

    if decision.is_replicated:
        metrics_store.record_cache_replication()

    # 占用目标节点活跃槽位
    node_active_slots[target_url] = node_active_slots.get(target_url, 0) + 1

    engine_payload = {
        "model": payload.model,
        "prompt": payload.prompt,
        "stream": payload.stream,
        "options": payload.options,
    }

    queue_wait_ms = getattr(request.state, "queue_wait_seconds", 0.0) * 1000.0

    headers = {
        "x-llm-gateway-active-slots": str(current_active_requests),
        "x-llm-gateway-pending-queue": str(request_queue_length),
        "x-llm-node-active-slots": f"{node_active_slots.get(target_url, 1)}/{MAX_ACTIVE_REQUESTS}",
        "x-llm-queue-wait-ms": f"{queue_wait_ms:.2f}",
        "x-llm-prefix-fingerprint": prefix_fingerprint,
        "x-llm-routed-target": target_url,
        "x-llm-primary-target": decision.primary_node,
        "x-llm-cache-replicated": "true" if decision.is_replicated else "false",
    }

    # 2. 流式转发
    if payload.stream:
        return StreamingResponse(
            forward_stream_to_engine(
                target_url=target_url,
                payload=engine_payload,
                prefix_fingerprint=prefix_fingerprint,
                req_start_time=req_start,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    # 3. 非流式单次请求
    client = get_http_client()
    try:
        resp = await client.post(target_url, json=engine_payload)
        e2e_duration = time.perf_counter() - req_start
        if resp.status_code == 200:
            metrics_store.record_success(ttft=e2e_duration, e2e_latency=e2e_duration, prefix_fingerprint=prefix_fingerprint)
            return Response(content=resp.content, status_code=200, media_type="application/json", headers=headers)
        else:
            metrics_store.record_upstream_error()
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
    finally:
        node_active_slots[target_url] = max(0, node_active_slots.get(target_url, 1) - 1)


@app.get("/metrics/health")
async def get_system_health():
    """网关实时负载与健康可观测指标 (JSON 结构化视图)。"""
    return metrics_store.get_summary_dict(
        active_slots=current_active_requests,
        max_slots=MAX_ACTIVE_REQUESTS,
        queue_len=request_queue_length,
        max_queue=MAX_QUEUE_SIZE,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """标准 Prometheus 格式导出接口，供 Prometheus Server 和 Grafana 面板定时拉取。"""
    return metrics_store.to_prometheus_format(
        active_slots=current_active_requests,
        queue_len=request_queue_length,
    )
