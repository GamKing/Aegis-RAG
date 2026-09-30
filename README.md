# Offline RAG Eval Harness（离线 RAG 评测套件 · 基础版）

## 当前质量基线（2026-09-30）

修复原因、方案与验证见 [来源约束与必答项修复报告](reports/SOURCE_GROUNDING_FIX.md)。
修复前审计保留在 [质量基线审计报告](reports/QUALITY_BASELINE.md)。
默认数据集已修正为恰好 100 条：55 合成、25 边界、15 对抗、5 种子。
问题显式指定虚构政策版本；对抗候选与正确标准答案分开存储。

```bash
# 固定数据集：审计 + Top-3 检索 + 正确上下文对照 + 对抗检出
python -m rag_eval.scissors_check --dataset golden_dataset_100.json --output reports/baseline_v3.json

# Top-1 仅作诊断对照，不替代默认 Top-3 验收
python -m rag_eval.scissors_check --dataset golden_dataset_100.json --top-k 1 --output reports/baseline_v3_top1.json
```

当前基线门禁预期返回 **0**：Top-3 召回后按用户指定来源筛选，100 条问答通过；
必答项独立门槛使 15 条错误候选全部检出，完整性覆盖阈值仍为 0.8。
新增 `source_id` 用于来源命中核验，`candidate_answer` 用于独立评分器测试；
该合成数据集只证明机制行为，不代表真实政策问答质量。

生产接入时请在块元数据填写 `scope_id`（同一政策版本的多个块共享该值），
并通过问题中的 `【来源 id】` / `[source: id]` 或
`ProductionRAG.retrieve_and_generate(sample, requested_sources=[...])` 指定来源。
`sample.source_id` 仅供评测使用，不参与生产来源选择。多个版本缺少选择条件时抛出
`SourceScopeError(code="AMBIGUOUS_SOURCE")`，由调用层提示用户澄清。
`required_entities` 是评测侧必答项；实际业务应根据需求维护这些标注。

面向工业级 RAG 系统的离线自动化评测套件：不依赖任何在线模型服务，用纯代码断言
对「检索质量 / 答案完整性 / 数值忠实性」三个维度做可复现评测，并输出结构化
ASCII 报表（含失败归因与指标剪刀差提示）。

## 快速开始

```bash
pip install -r requirements.txt

# 运行演示评测（内置 5 条医保政策样本 + Dummy RAG 预设四种失败模式）
python -m rag_eval

# 运行单元测试
python -m unittest discover -s tests -v
```

`main()` 在存在失败样本时以退出码 `1` 结束，可直接接入 CI 流水线。

## 模块结构

| 模块 | 职责 |
| --- | --- |
| `rag_eval/models.py` | Pydantic 数据契约：`EvalSample` / `EvalResult` / `EvalSummary` |
| `rag_eval/text_utils.py` | 文本归一化、CJK bigram 分词、边界敏感数值匹配、CJK 宽度对齐 |
| `rag_eval/evaluators.py` | 三个核心评分器（召回 / 完整性 / 忠实性） |
| `rag_eval/dataset.py` | 内置 5 条医保政策模拟样本（3 标准 + 1 前置条件 + 1 对抗） |
| `rag_eval/dummy_rag.py` | Dummy RAG 后端与真实系统接入点（`BaseRAG`） |
| `rag_eval/runner.py` | `run_eval_pipeline`：批量执行、单样本判定、异常隔离、聚合 |
| `rag_eval/report.py` | 结构化 ASCII 报表渲染 |
| `rag_eval/main.py` | 演示入口（组合根：装配数据集、Dummy 预设、运行、打印） |

## 指标定义

| 指标 | 口径 | 通过条件（默认） |
| --- | --- | --- |
| ContextRecall | 标准上下文词项（CJK bigram + ASCII 词/数字）在检索上下文中的覆盖率 | ≥ 0.80 |
| Completeness | `expected_entities` 归一化后严格子串命中率（命中数 / 总数） | ≥ 0.80 |
| Faithfulness | 答案中全部数值/百分比必须能在检索上下文中找到出处，一个无出处即否决 | == True |

单样本通过 = 三项全部达标；任一失败会在 `error_details` 中给出确定性的归因文本。

## 忠实性校验的严格规则

1. **数字边界**：`1500` 不会命中 `15000` / `31500` 的子串；
2. **单位一致**：答案中的百分数必须以百分数形式出现在上下文（`85%` 不能被
   `85元` 洗白）；普通数值允许命中普通数值或百分数；
3. **表述归一**：千分位逗号与全角数字/百分号先归一化（`1,500` ≡ `1500`，
   `７０％` ≡ `70%`）；英文月份归一为数字月份（`September` ≡ `9月`），
   `million`/`billion` 归一为“亿”（`59,688 million` ≡ `596.88亿`，
   `$108.0 billion` ≡ `1080亿`）。

## 内置演示的失败模式（指标剪刀差）

| 样本 | 检索 | 生成 | 展示点 |
| --- | --- | --- | --- |
| med-ins-001 | 完美 | 完美 | 全绿基线 |
| med-ins-002 | 缺失（三级条款+封顶线） | 如实转述 | 检索问题沿链路传导到完整性 |
| med-ins-003 | 完美 | 遗漏倾斜条款 | 召回 1.0 但完整性失败：检索指标看不见生成端损耗 |
| med-ins-004 | 丢失例外条款 | 绝对化答案 | 政策前置条件/例外失守 |
| med-ins-005 | 完美 | 数字被篡改 | 召回/完整性全绿，仅忠实性一票否决 |

## 已知限制（v1）

- 忠实性只校验数值出处，不校验非数值的事实性主张（因果、归属等）；
- 不支持中文数字（`百分之七十`、大写金额）与万亿单位换算（`3.5万` ≠ `35000`）；
- ContextRecall 是词项级代理指标，不做语义等价判断（同义改写会拉低得分）。

## 五步受控生成流水线

项目新增 `rag_eval/pipeline.py`，将 RAG 输出拆成可审计的五步：

```text
Query + 原始 Context
  -> FactExtractor（默认 RuleBasedFactExtractor；生产可替换为 JSON Schema 强制解码）
  -> Tier1CodeVerifier（纯代码数值/实体出处校验）
  -> Tier2 NLIAndSelfCorrector（仅 Tier1 失败时触发，最多一次）
  -> Tier1 复验
  -> ControlledSynthesizer（只使用通过校验的结构化事实）
```

核心接口：

- `ExtractionPayload` / `ExtractedFact`：受限抽取中间契约；
- `Tier1CodeVerifier`：复用现有月份、million/billion、数字边界和实体归一化；
- `NLIAndSelfCorrector`：逐个失败事实调用可选 NLI，A/蕴含才保留；
- `ControlledSynthesizer`：不重新自由生成事实，只拼接已验证字段；
- `PipelineRAG`：保持现有 `BaseRAG` / `RAGResponse` 兼容。

默认流水线不调用外部模型，完全离线。注入 `NLIJudge` 后，Tier2 才会启用；NLI
失败、返回拒绝或调用异常时安全降级，不绕过 Tier1。修复后的 payload 必须再次经过
Tier1；数字篡改（如 `7.2 billion -> 9.5 billion`）仍会被拦截。

示例：

```python
from rag_eval import (
    NLIAndSelfCorrector, NLIJudge, PipelineRAG, run_eval_pipeline,
)

# context_provider(sample) 只负责返回检索上下文
pipeline_rag = PipelineRAG(
    context_provider=lambda sample: retriever.search_text(sample.question),
    self_corrector=NLIAndSelfCorrector(NLIJudge(client)),
)
summary = run_eval_pipeline(samples, pipeline_rag)
```

## 接入真实 RAG

实现 `BaseRAG` 接口即可复用整套评测：

```python
from rag_eval import BaseRAG, EvalSample, RAGResponse, load_samples, run_eval_pipeline, render_report


class MyRAG(BaseRAG):
    def retrieve_and_generate(self, sample: EvalSample) -> RAGResponse:
        docs = my_retriever.search(sample.question, top_k=5)
        answer = my_llm.generate(sample.question, docs)
        return RAGResponse(context="\n".join(d.text for d in docs), answer=answer)


summary = run_eval_pipeline(load_samples(), MyRAG())
print(render_report(summary))
```
