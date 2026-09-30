"""前缀感知路由 (Prefix-Aware Routing) 与一致性哈希测试套件。

验证核心能力：
1. 前缀指纹稳定性：相同长文档上下文 + 不同用户提问，提取出的指纹 100% 相同；
2. 亲和性路由一致性：相同指纹请求在集群中始终路由至同一目标节点；
3. 节点故障迁移稳定性：当集群某节点离线时，仅迁移该节点承载的 key，其余节点映射保持稳定；
4. 节点分布均衡性：多节点下请求指纹在一致性哈希环上的虚拟节点分布均匀度。
"""
import sys
import unittest
from gateway.router import ConsistentHashRing, PrefixAwareRouter, extract_prefix_fingerprint


class TestPrefixAwareRouting(unittest.TestCase):

    def setUp(self):
        self.nodes = [
            "http://gpu-worker-01:22434/api/generate",
            "http://gpu-worker-02:22434/api/generate",
            "http://gpu-worker-03:22434/api/generate",
        ]
        self.router = PrefixAwareRouter(self.nodes)

        # 模拟一份 3000 字的企业级长法律/技术制度前缀
        self.shared_document = (
            "【企业数据安全与反洗钱内部合规指引 2026 版】\n"
            "第一条：为防范洗钱及恐怖融资风险，全行各级营业机构应当严格遵循反洗钱合规准则。\n"
            "第二条：对大额交易与可疑交易实行实时监测机制，单笔交易金额超过 50,000 元人民币应留存凭证。\n"
            + ("合规条文细节正文补充说明..." * 200)
        )

    def test_prefix_fingerprint_determinism(self):
        """验证相同长上下文下的不同提问是否能提取出完全一致的前缀指纹。"""
        q1 = f"{self.shared_document}\n\n问题 1：大额交易的监控阈值是多少？"
        q2 = f"{self.shared_document}\n\n问题 2：全行各级营业机构由哪个部门牵头合规检查？"
        q3 = f"{self.shared_document}\n\n问题 3：请根据指引第 1 条给出总结。"

        fp1 = extract_prefix_fingerprint(q1)
        fp2 = extract_prefix_fingerprint(q2)
        fp3 = extract_prefix_fingerprint(q3)

        self.assertEqual(fp1, fp2, "不同提问的前缀指纹应保持完全一致")
        self.assertEqual(fp2, fp3, "不同提问的前缀指纹应保持完全一致")
        self.assertTrue(len(fp1) == 16, "指纹应为 16 位 16 进制字符串")

    def test_routing_affinity_consistency(self):
        """验证相同前缀的不同请求 100% 路由到同一节点（缓存亲和命中）。"""
        target_nodes = set()
        for i in range(20):
            prompt = f"{self.shared_document}\n\n随机提问序号 {i}：请解释条文中的特定规定。"
            target, fp = self.router.route(prompt)
            target_nodes.add(target)

        self.assertEqual(
            len(target_nodes),
            1,
            f"所有共享该长前缀的请求必须被路由到唯一的固定节点，实际分配到了: {target_nodes}",
        )
        print(f"\n[Prefix-Aware Routing] 20 次同前缀请求全部精确锁定目标节点: {list(target_nodes)[0]}")

    def test_distinct_prefixes_distribution(self):
        """验证不同前缀的文档在多个节点之间具有良好的离散分布特性。"""
        node_counter = {node: 0 for node in self.nodes}

        for doc_id in range(60):
            distinct_prompt = f"【企业制度文档编号-{doc_id}】" + ("独立业务规范正文..." * 50) + "请分析业务合规性。"
            target, _ = self.router.route(distinct_prompt)
            node_counter[target] += 1

        print(f"[Prefix-Aware Routing] 60 篇不同文档的节点离散分布: {node_counter}")
        for node, count in node_counter.items():
            self.assertGreater(count, 0, f"节点 {node} 应该分担到至少一部分请求，不能饥饿")

    def test_node_removal_resilience(self):
        """验证一致性哈希在单节点宕机时的最小扰动特性（只迁移故障节点流量）。"""
        ring = ConsistentHashRing(self.nodes)

        # 记录 100 个请求在 3 节点下的初始路由映射
        initial_mapping = {}
        for i in range(100):
            key = f"document_key_{i}"
            initial_mapping[key] = ring.get_node(key)

        removed_node = self.nodes[0]
        ring.remove_node(removed_node)

        migrated_keys = 0
        intact_keys = 0

        for key, old_node in initial_mapping.items():
            new_node = ring.get_node(key)
            if old_node == removed_node:
                # 原本在故障节点的请求必须迁移到新节点
                self.assertNotEqual(new_node, removed_node)
                migrated_keys += 1
            else:
                # 原本在存活节点的请求应该尽量保持不动（哈希环局部单调性）
                if new_node == old_node:
                    intact_keys += 1
                else:
                    migrated_keys += 1

        print(f"[ConsistentHash] 移除节点 {removed_node} 后: 保持未动比例 = {intact_keys}/100, 迁移 = {migrated_keys}/100")
        self.assertGreater(intact_keys, 50, "大部分存活节点的映射关系应当保持稳定")

    def test_load_aware_fallback_cache_replication(self):
        """验证动态负载平衡与主动缓存分裂 (Load-Aware Fallback / Cache Replication)：
        当主亲和卡槽位满载时，后续同前缀请求自动溢出分流给空闲备用卡。
        """
        prompt = f"{self.shared_document}\n\n热点突发提问：紧急合规条例查询。"

        # 1. 初始状态：所有节点均空闲
        node_loads = {node: 0 for node in self.nodes}

        def load_checker(node_url: str) -> bool:
            return node_loads.get(node_url, 0) < 8  # 模拟单卡 8 并发上限

        # 决策 1：正常亲和路由至主卡
        decision_1 = self.router.route(prompt, node_load_checker=load_checker)
        primary_node = decision_1.target_node
        self.assertFalse(decision_1.is_replicated, "未满载时严禁发生缓存分裂")
        self.assertEqual(decision_1.primary_node, primary_node)

        # 2. 模拟突发流量：主卡瞬间被打满（8 槽位全占满）
        node_loads[primary_node] = 8

        # 决策 2：相同前缀的新请求打入
        decision_2 = self.router.route(prompt, node_load_checker=load_checker)
        replicated_node = decision_2.target_node

        # 核心断言：
        # - 目标节点不能再是已打满的主卡
        self.assertNotEqual(replicated_node, primary_node, "主卡打满时，同前缀请求必须溢出分流")
        # - 必须标记为主动分裂 (Cache Replication)
        self.assertTrue(decision_2.is_replicated, "溢出请求必须标记为 is_replicated=True")
        # - primary_node 属性保持记录原始亲和节点
        self.assertEqual(decision_2.primary_node, primary_node)
        # - 候选物理节点列表中必须包含主卡和复制卡
        candidates = self.router.ring.get_candidate_nodes(decision_2.prefix_fingerprint)
        self.assertEqual(candidates[0], primary_node)
        self.assertEqual(candidates[1], replicated_node)

        print(f"\n[Load-Aware Routing] 主卡 {primary_node} 满载 (8/8)，同前缀请求成功主动分裂至空闲卡: {replicated_node}")

    def test_cluster_all_saturated_fallback(self):
        """验证集群全节点满载时的降级保护：回退至主亲和节点排队/熔断。"""
        prompt = f"{self.shared_document}\n\n全集群极度饱和测试。"

        # 模拟所有节点均满载
        node_loads = {node: 8 for node in self.nodes}

        def load_checker(node_url: str) -> bool:
            return node_loads.get(node_url, 0) < 8

        decision = self.router.route(prompt, node_load_checker=load_checker)
        # 集群全满时应回退到主亲和节点统一排队或熔断
        self.assertEqual(decision.target_node, decision.primary_node)
        self.assertFalse(decision.is_replicated)
        print(f"[Load-Aware Routing] 集群全节点满载时，平稳回退至主亲和节点排队: {decision.target_node}")


if __name__ == "__main__":
    unittest.main()

