"""前缀感知路由 (Prefix-Aware Routing) 与一致性哈希模块。

大模型推理具有状态强绑定（Stateful KV Cache）特性：
相同 System Prompt 或长上下文文档的请求，若被随机轮询分发到不同 GPU 节点，
会导致 PagedAttention 前缀缓存（Prefix Cache）命中率彻底归零。

本模块提供：
1. 前缀特征指纹提取 (Prefix Fingerprint Extractor)
2. 一致性哈希环 (Consistent Hash Ring with Virtual Nodes)
3. 前缀亲和性路由决策 (Affinity Router)
"""
from __future__ import annotations

import bisect
import hashlib
from typing import Sequence


class ConsistentHashRing:
    """带虚拟节点的高性能一致性哈希环。"""

    def __init__(self, nodes: Sequence[str] | None = None, virtual_replicas: int = 64) -> None:
        self.virtual_replicas = virtual_replicas
        self.ring: list[int] = []
        self.ring_map: dict[int, str] = {}
        self.nodes: set[str] = set()

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """使用 MD5 生成 32 位整型哈希值。"""
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        """向环中添加物理节点及其虚拟节点副本。"""
        self.nodes.add(node)
        for i in range(self.virtual_replicas):
            v_key = f"{node}#VN{i}"
            v_hash = self._hash(v_key)
            bisect.insort(self.ring, v_hash)
            self.ring_map[v_hash] = node

    def remove_node(self, node: str) -> None:
        """从环中移除物理节点及其虚拟副本。"""
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        new_ring: list[int] = []
        for v_hash in self.ring:
            if self.ring_map.get(v_hash) == node:
                del self.ring_map[v_hash]
            else:
                new_ring.append(v_hash)
        self.ring = new_ring

    def get_node(self, key: str) -> str | None:
        """根据路由键二分查找目标节点。"""
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.ring_map[self.ring[idx]]

    def get_candidate_nodes(self, key: str) -> list[str]:
        """按一致性哈希环的顺时针顺序返回所有唯一的候选物理节点列表。

        首个节点为最高优先级主亲和节点；后续节点为主节点满载时的顺位备选（用于主动缓存复制与分流）。
        """
        if not self.ring:
            return []
        h = self._hash(key)
        start_idx = bisect.bisect_right(self.ring, h)
        total = len(self.ring)

        ordered_nodes: list[str] = []
        seen: set[str] = set()
        for offset in range(total):
            idx = (start_idx + offset) % total
            node = self.ring_map[self.ring[idx]]
            if node not in seen:
                seen.add(node)
                ordered_nodes.append(node)
                if len(seen) == len(self.nodes):
                    break
        return ordered_nodes


def extract_prefix_fingerprint(prompt: str, prefix_char_limit: int = 1500) -> str:
    """从 Prompt 中提取公共长上下文特征指纹。

    策略：
    - RAG 场景中长文档通常位于 Prompt 前部或中部；
    - 截取前 prefix_char_limit 个字符（通常覆盖公共规约/长文档前置段落）；
    - 计算 SHA256 作为前缀唯一标识指纹。
    """
    cleaned = prompt.strip()
    prefix_sample = cleaned[:prefix_char_limit]
    return hashlib.sha256(prefix_sample.encode("utf-8")).hexdigest()[:16]


class RouteDecision:
    """路由决策结果对象，封装目标节点、指纹与分流状态。

    同时兼容二元组解构: target_url, prefix_fp = router.route(...)
    """

    def __init__(
        self,
        target_node: str,
        prefix_fingerprint: str,
        is_replicated: bool = False,
        primary_node: str = "",
    ) -> None:
        self.target_node = target_node
        self.prefix_fingerprint = prefix_fingerprint
        self.is_replicated = is_replicated
        self.primary_node = primary_node or target_node

    def __iter__(self):
        """保持向后兼容：支持 (target_node, prefix_fingerprint) 解构。"""
        return iter((self.target_node, self.prefix_fingerprint))

    def __getitem__(self, index: int):
        return (self.target_node, self.prefix_fingerprint, self.is_replicated)[index]

    def __repr__(self) -> str:
        return (
            f"RouteDecision(target={self.target_node}, fp={self.prefix_fingerprint}, "
            f"replicated={self.is_replicated}, primary={self.primary_node})"
        )


class PrefixAwareRouter:
    """前缀感知与动态负载感知路由器 (Prefix & Load-Aware Router)。

    调度核心能力：
    1. 前缀亲和性优先 (Prefix-Affinity First)：同长上下文默认 100% 路由至固定主节点，打满 PagedAttention 缓存命中；
    2. 动态负载溢出保护 (Load-Aware Fallback / Cache Replication)：当主卡活跃槽位打满时，自动分裂并溢出分流给空闲卡；
    3. 集群雪崩自适应回退：当全集群打满时，回退至主节点交由网关排队队列或 Load Shedding (HTTP 429) 熔断兜底。
    """

    def __init__(self, backend_endpoints: list[str]) -> None:
        self.endpoints = backend_endpoints
        self.ring = ConsistentHashRing(backend_endpoints)

    def route(
        self,
        prompt: str,
        node_load_checker: Any = None,
    ) -> RouteDecision:
        """为请求挑选最佳推理后端节点。

        Args:
            prompt: 用户输入或 RAG 长文本上下文
            node_load_checker: 可调用的节点负载探针函数 (node_url: str) -> bool。
                              返回 True 表示该节点仍有空闲并发槽位；返回 False 表示已满载。

        Returns:
            RouteDecision 对象
        """
        if not self.endpoints:
            raise RuntimeError("无可用推理后端节点")

        fingerprint = extract_prefix_fingerprint(prompt)

        # 单节点模式：直接返回
        if len(self.endpoints) == 1:
            return RouteDecision(self.endpoints[0], fingerprint, is_replicated=False)

        candidates = self.ring.get_candidate_nodes(fingerprint)
        primary_node = candidates[0] if candidates else self.endpoints[0]

        # 若未提供负载检查探针，退化为经典纯一致性哈希路由
        if not node_load_checker:
            return RouteDecision(primary_node, fingerprint, is_replicated=False, primary_node=primary_node)

        # 1. 亲和性优先：若主亲和节点仍有容量，100% 锁定主节点以复用 KV Cache
        if node_load_checker(primary_node):
            return RouteDecision(primary_node, fingerprint, is_replicated=False, primary_node=primary_node)

        # 2. 动态负载平衡（防热点倾斜）：主卡已打满！沿哈希环顺位探测空闲卡，触发“主动分裂 / Cache Replication”
        for alt_node in candidates[1:]:
            if node_load_checker(alt_node):
                return RouteDecision(alt_node, fingerprint, is_replicated=True, primary_node=primary_node)

        # 3. 集群全满：降级归拢回主亲和节点排队/熔断
        return RouteDecision(primary_node, fingerprint, is_replicated=False, primary_node=primary_node)

