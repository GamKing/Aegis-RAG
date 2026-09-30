"""全链路可观测性 (Observability) 指标与 Prometheus 格式导出器。

聚焦四大四色黄金指标：
1. 饱和度 (Saturation): 活跃并发槽位占用率、队列饱和度
2. 流量与队列 (Traffic): 活跃请求数、排队深度、请求总吞吐
3. 延迟分布 (Latency): 首字延迟 (TTFT)、端到端时延 (E2E)、流式字间时延 (ITL)
4. 错误与削峰 (Errors): Load Shedding 拦截数 (HTTP 429)、上游 5xx 异常
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GatewayMetricsStore:
    """网关内存级高性能指标聚合器。"""

    total_requests: int = 0
    successful_requests: int = 0
    rejected_shedding: int = 0
    upstream_errors: int = 0

    # 延迟直方数据统计
    ttft_count: int = 0
    ttft_sum: float = 0.0
    ttft_max: float = 0.0

    e2e_count: int = 0
    e2e_sum: float = 0.0
    e2e_max: float = 0.0

    # 前缀路由命中统计与热点主动分裂 (Cache Replication)
    prefix_fingerprint_hits: dict[str, int] = field(default_factory=dict)
    cache_replications: int = 0

    def record_request_start(self) -> None:
        self.total_requests += 1

    def record_rejection(self) -> None:
        self.rejected_shedding += 1

    def record_upstream_error(self) -> None:
        self.upstream_errors += 1

    def record_cache_replication(self) -> None:
        self.cache_replications += 1

    def record_success(self, ttft: float, e2e_latency: float, prefix_fingerprint: str = "") -> None:
        self.successful_requests += 1

        if ttft > 0:
            self.ttft_count += 1
            self.ttft_sum += ttft
            if ttft > self.ttft_max:
                self.ttft_max = ttft

        if e2e_latency > 0:
            self.e2e_count += 1
            self.e2e_sum += e2e_latency
            if e2e_latency > self.e2e_max:
                self.e2e_max = e2e_latency

        if prefix_fingerprint:
            self.prefix_fingerprint_hits[prefix_fingerprint] = (
                self.prefix_fingerprint_hits.get(prefix_fingerprint, 0) + 1
            )

    def get_summary_dict(self, active_slots: int, max_slots: int, queue_len: int, max_queue: int) -> dict[str, Any]:
        """返回结构化 JSON 健康与负载概览。"""
        avg_ttft = (self.ttft_sum / self.ttft_count) if self.ttft_count > 0 else 0.0
        avg_e2e = (self.e2e_sum / self.e2e_count) if self.e2e_count > 0 else 0.0
        rejection_rate = (self.rejected_shedding / max(1, self.total_requests)) * 100.0

        return {
            "status": "healthy" if active_slots < max_slots else "saturated",
            "saturation": {
                "active_concurrent_slots": f"{active_slots}/{max_slots}",
                "slot_utilization_rate": f"{(active_slots / max_slots) * 100.0:.1f}%",
                "pending_queue_size": f"{queue_len}/{max_queue}",
                "queue_saturation_rate": f"{(queue_len / max_queue) * 100.0:.1f}%",
            },
            "traffic": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "rejected_shedding_429": self.rejected_shedding,
                "upstream_errors": self.upstream_errors,
                "load_shedding_rate": f"{rejection_rate:.2f}%",
            },
            "latency": {
                "avg_ttft_seconds": round(avg_ttft, 3),
                "max_ttft_seconds": round(self.ttft_max, 3),
                "avg_e2e_seconds": round(avg_e2e, 3),
                "max_e2e_seconds": round(self.e2e_max, 3),
            },
            "prefix_routing": {
                "unique_prefixes_tracked": len(self.prefix_fingerprint_hits),
                "replicated_cache_fallbacks": self.cache_replications,
                "top_prefixes": sorted(self.prefix_fingerprint_hits.items(), key=lambda x: x[1], reverse=True)[:5],
            },
        }

    def to_prometheus_format(self, active_slots: int, queue_len: int) -> str:
        """导出符合 Prometheus 规范的标准文本度量格式。"""
        lines = [
            "# HELP llm_gateway_requests_total Total number of incoming requests received by gateway",
            "# TYPE llm_gateway_requests_total counter",
            f"llm_gateway_requests_total {self.total_requests}",
            "",
            "# HELP llm_gateway_successful_requests_total Total successful requests forwarded and responded",
            "# TYPE llm_gateway_successful_requests_total counter",
            f"llm_gateway_successful_requests_total {self.successful_requests}",
            "",
            "# HELP llm_gateway_load_shedding_rejected_total Total requests rejected due to queue saturation (HTTP 429)",
            "# TYPE llm_gateway_load_shedding_rejected_total counter",
            f"llm_gateway_load_shedding_rejected_total {self.rejected_shedding}",
            "",
            "# HELP llm_gateway_cache_replication_total Total requests dynamically spilled over to alternative nodes (Cache Replication)",
            "# TYPE llm_gateway_cache_replication_total counter",
            f"llm_gateway_cache_replication_total {self.cache_replications}",
            "",
            "# HELP llm_gateway_upstream_errors_total Total errors received from backend inference engine",
            "# TYPE llm_gateway_upstream_errors_total counter",
            f"llm_gateway_upstream_errors_total {self.upstream_errors}",
            "",
            "# HELP llm_gateway_active_slots Current active inference slots occupied in engine",
            "# TYPE llm_gateway_active_slots gauge",
            f"llm_gateway_active_slots {active_slots}",
            "",
            "# HELP llm_gateway_queue_length Current pending requests waiting in queue",
            "# TYPE llm_gateway_queue_length gauge",
            f"llm_gateway_queue_length {queue_len}",
            "",
            "# HELP llm_gateway_ttft_seconds Time to first token summary",
            "# TYPE llm_gateway_ttft_seconds summary",
            f"llm_gateway_ttft_seconds_count {self.ttft_count}",
            f"llm_gateway_ttft_seconds_sum {self.ttft_sum:.4f}",
            "",
            "# HELP llm_gateway_e2e_seconds End-to-end request duration summary",
            "# TYPE llm_gateway_e2e_seconds summary",
            f"llm_gateway_e2e_seconds_count {self.e2e_count}",
            f"llm_gateway_e2e_seconds_sum {self.e2e_sum:.4f}",
            "",
        ]
        return "\n".join(lines)


# 全局单例指标仓
metrics_store = GatewayMetricsStore()
