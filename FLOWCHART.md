# Offline RAG Eval Harness 流程图

本文档描述当前项目从启动、数据加载、Dummy/真实 RAG 调用、三项评分，到聚合报表和退出码的完整流程。

## 1. 端到端执行流程

```mermaid
flowchart TD
    A([开始\npython -m rag_eval]) --> B[main.main]
    B --> C[load_samples\n加载 Pydantic EvalSample 数据集]
    C --> D[build_demo_rag\n装配 DummyRAG 预设]
    D --> E[run_eval_pipeline]

    E --> F{遍历每个 EvalSample}
    F --> G[evaluate_sample]
    G --> H[rag.retrieve_and_generate\n获得 RAGResponse]
    H --> I[actual_context\nactual_answer]

    I --> J[ContextRecallEvaluator\n上下文关键词/短句覆盖率]
    I --> K[CompletenessEvaluator\nexpected_entities 命中率]
    I --> L[FaithfulnessEvaluator\n答案数值/百分比出处校验]

    J --> M[汇总 MetricOutcome]
    K --> M
    L --> M
    M --> N[生成 EvalResult\n记录分数、布尔结果、error_details]
    N --> F

    F -->|全部样本完成| O[聚合 EvalSummary\n平均 Recall / Completeness\nFaithfulness 通过率 / 用例通过率]
    O --> P[render_report\n生成结构化 ASCII 报表]
    P --> Q[打印报表与失败归因]
    Q --> R{summary.failed > 0?}
    R -->|是| S([退出码 1\n评测未全部通过])
    R -->|否| T([退出码 0\n全部样本通过])
```

## 2. 五步受控生成流程

```mermaid
flowchart TD
    Q[用户 Query] --> R[检索与上下文组装]
    R --> X[受限信息抽取\nFactExtractor + JSON Schema]
    X --> T1[Tier 1 代码质检\nNormalizer + 数值/实体出处]
    T1 -->|通过| S[受控润色输出\nControlledSynthesizer]
    T1 -->|未命中/存疑| T2[Tier 2 NLI 与单次自愈\nA=蕴含 / B=矛盾 / C=中立]
    T2 --> V[Tier 1 复验]
    V -->|通过| S
    V -->|仍失败| F[安全失败\n不输出未经验证事实]
    S --> O[最终 RAGResponse\n进入现有三项评测]
```

## 3. 单样本评分流程

```mermaid
flowchart LR
    A[EvalSample\nquestion / ground_truth_context\nexpected_entities] --> B[BaseRAG.retrieve_and_generate]
    B --> C[RAGResponse\nactual_context / actual_answer]

    C --> D[ContextRecallEvaluator]
    D --> D1[normalize_text]
    D1 --> D2[CJK bigram + ASCII term extraction]
    D2 --> D3[交集数量 / 标准上下文词项总数]
    D3 --> D4[context_recall_score]

    C --> E[CompletenessEvaluator]
    E --> E1[normalize_for_match]
    E1 --> E2[逐个检查 expected_entities]
    E2 --> E3[命中数 / 实体总数]
    E3 --> E4[completeness_score]

    C --> F[FaithfulnessEvaluator]
    F --> F1[extract_numbers]
    F1 --> F2[提取答案中的数值与百分比]
    F2 --> F3{每个数值是否存在于 actual_context?}
    F3 -->|全部存在| F4[faithfulness_pass = true]
    F3 -->|存在新数值| F5[faithfulness_pass = false\n记录数字幻觉/篡改详情]

    D4 --> G[EvalResult]
    E4 --> G
    F4 --> G
    F5 --> G
```

## 3. 失败判定与指标剪刀差

```mermaid
flowchart TD
    A[EvalResult] --> B{Context Recall >= 0.80?}
    B -->|否| C[检索覆盖不足\nContext Recall 失败]
    B -->|是| D{Completeness >= 0.80?}
    D -->|否| E[答案遗漏实体\nCompleteness 失败]
    D -->|是| F{Faithfulness PASS?}
    F -->|否| G[答案出现上下文未支持的数值\n数字幻觉/篡改]
    F -->|是| H([Case PASS])

    B -->|是| F
    G --> I[剪刀差警示\n检索覆盖达标但生成端不忠实]
```

## 4. 模块关系图

```mermaid
flowchart TB
    MAIN[main.py\n演示入口] --> DATA[dataset.py\n内置样本]
    MAIN --> RAG[dummy_rag.py\nDummyRAG / BaseRAG]
    MAIN --> RUNNER[runner.py\nrun_eval_pipeline]
    MAIN --> REPORT[report.py\nrender_report]

    RUNNER --> MODELS[models.py\nEvalSample / EvalResult / EvalSummary]
    RUNNER --> EVAL[evaluators.py\n三类 Evaluator]
    RUNNER --> RAG
    EVAL --> MODELS
    EVAL --> TEXT[text_utils.py\n归一化 / 分词 / 数值匹配]
    REPORT --> MODELS
    REPORT --> TEXT

    TESTS[tests/] -.回归验证.-> MODELS
    TESTS -.回归验证.-> EVAL
    TESTS -.回归验证.-> RUNNER
    TESTS -.使用 fixture.-> NVDA[tests/nvda_cases.json\nNVIDIA 样本]
```

## 5. 当前演示数据的处理路径

```mermaid
flowchart LR
    A[med-ins-001\n完美链路] --> P[三项通过]
    B[med-ins-002\n检索缺失] --> Q[Recall + Completeness 失败]
    C[med-ins-003\n生成遗漏] --> R[Recall 通过\nCompleteness 失败]
    D[med-ins-004\n前置条件/例外丢失] --> S[Recall + Completeness 失败]
    E[med-ins-005\n数字篡改] --> T[Recall + Completeness 通过\nFaithfulness 失败]
```

## 6. 真实 RAG 接入点

真实系统只需要实现 `BaseRAG`：

```mermaid
sequenceDiagram
    participant Runner as run_eval_pipeline
    participant Adapter as 自定义 BaseRAG 适配器
    participant Retriever as 真实 Retriever
    participant LLM as 真实 Generator / LLM
    participant Evaluator as 三个纯代码 Evaluator

    Runner->>Adapter: retrieve_and_generate(sample)
    Adapter->>Retriever: search(sample.question, top_k)
    Retriever-->>Adapter: documents
    Adapter->>LLM: generate(question, documents)
    LLM-->>Adapter: answer
    Adapter-->>Runner: RAGResponse(context, answer)
    Runner->>Evaluator: evaluate(sample, context, answer)
    Evaluator-->>Runner: MetricOutcome
    Runner-->>Runner: EvalResult / EvalSummary / report
```
