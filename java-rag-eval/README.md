# Aegis-RAG Java 17 评测系统

这是一个基于 Java 17 实现的工业级 RAG（检索增强生成）评测系统，提供确定性的事实核验和受控生成流水线。

## 系统架构

### 核心模块

```
com.gamking.aegisrag
├── model/              # 数据模型
│   ├── EvalSample      # 评测样本
│   ├── EvalResult      # 评测结果
│   ├── EvalSummary     # 评测汇总
│   ├── ExtractedFact   # 抽取的事实
│   └── ExtractionPayload  # 抽取结果
├── text/               # 文本处理
│   └── TextNormalizer  # 文本归一化（全角/半角、日期、金额单位）
├── evaluator/          # 评测器
│   ├── ContextRecallEvaluator      # 上下文召回率
│   ├── CompletenessEvaluator       # 完整性
│   └── FaithfulnessEvaluator       # 忠实度
├── pipeline/           # 受控流水线
│   ├── FactExtractor   # 事实抽取接口
│   ├── RuleBasedFactExtractor  # 基于规则的事实抽取
│   ├── Tier1CodeVerifier       # Tier 1 代码验证
│   ├── NliSelfCorrector        # Tier 2 NLI 自愈
│   └── ControlledPipeline      # 受控流水线
├── nli/                # NLI 裁判
│   └── NliJudge        # NLI 裁判接口
├── ollama/             # Ollama 集成
│   ├── OllamaClient    # Ollama HTTP 客户端
│   ├── OllamaNliJudge  # 基于 Ollama 的 NLI 裁判
│   ├── OllamaFactExtractor  # 基于 Ollama 的事实抽取
│   └── OllamaAnswerSynthesizer  # 基于 Ollama 的答案合成
├── rag/                # RAG 接口
│   ├── BaseRag         # RAG 基础接口
│   ├── DummyRag        # 虚拟 RAG（用于测试）
│   └── PipelineRag     # 流水线 RAG
├── runner/             # 运行器
│   ├── EvalConfig      # 评测配置
│   └── EvalRunner      # 评测运行器
├── report/             # 报告生成
│   └── AsciiReportRenderer  # ASCII 报表渲染器
├── dataset/            # 数据集
│   └── DemoDataset     # 演示数据集
└── Main.java           # 主程序入口
```

### 流水线流程

```
用户问题 + 检索上下文
    ↓
[1] 事实抽取（RuleBasedFactExtractor 或 OllamaFactExtractor）
    ↓
[2] Tier 1 代码验证（确定性规则检查）
    ├─ 通过 → [4] 受控合成
    └─ 失败 → [3] Tier 2 NLI 自愈
                 ↓
              Tier 1 复验
                 ├─ 通过 → [4] 受控合成
                 └─ 失败 → 安全失败
    ↓
[4] 受控合成（ControlledSynthesizer 或 OllamaAnswerSynthesizer）
    ↓
最终答案
```

## 环境要求

- **JDK 17** 或更高版本
- **Maven 3.6+**（用于构建和测试）
- **Ollama**（可选，用于 LLM 集成）

## 快速开始

### 1. 编译项目

```bash
cd java-rag-eval
mvn clean compile
```

### 2. 运行演示

```bash
mvn exec:java
```

这将运行内置的医保政策演示数据集，展示评测系统的完整流程。

### 3. 运行测试

```bash
mvn test
```

测试覆盖：
- 文本归一化（全角/半角、日期、金额单位）
- 三项评测器（Context Recall、Completeness、Faithfulness）
- 受控流水线（Tier 1 直通、Tier 2 自愈、对抗样本拦截）

## 配置 Ollama

如果需要使用 Ollama 进行 LLM 集成，需要设置环境变量：

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=Kimi2.5:1.5b
```

### 使用 Ollama 流水线

```java
// 创建 Ollama 客户端
OllamaClient client = new OllamaClient();

// 创建 Ollama NLI 裁判
OllamaNliJudge nliJudge = new OllamaNliJudge(client);

// 创建 Ollama 事实抽取器
OllamaFactExtractor extractor = new OllamaFactExtractor(client);

// 创建 Ollama 答案合成器
OllamaAnswerSynthesizer synthesizer = new OllamaAnswerSynthesizer(client);

// 创建受控流水线
ControlledPipeline pipeline = new ControlledPipeline(
    extractor,
    synthesizer,
    new Tier1CodeVerifier(),
    new NliSelfCorrector(nliJudge)
);

// 创建 PipelineRag
PipelineRag rag = new PipelineRag(contextProvider, pipeline);
```

## 核心特性

### 1. 确定性评测

- **Context Recall**：基于 CJK bigram 和 ASCII token 的上下文覆盖率
- **Completeness**：关键实体的严格子串匹配
- **Faithfulness**：数值出处验证（边界敏感正则）

### 2. 双层质检

- **Tier 1**：纯代码验证，毫秒级响应
  - 数值边界检查（防止 1500 误匹配 15000）
  - 单位一致性（85% 不会被 85元 支持）
  - 日期和金额单位归一化
  
- **Tier 2**：NLI 语义复核
  - 处理合理推导（如 "not assuming" → 0）
  - 拒绝矛盾和中立结论
  - 异常时安全降级

### 3. 受控生成

- 只使用已验证的事实进行合成
- 不引入新的数字、日期或实体
- 支持基于规则和基于 LLM 的合成器

### 4. 安全机制

- Tier 1 失败时自动触发 Tier 2
- Tier 2 修复后必须重新通过 Tier 1
- 无法验证时返回空答案，不输出未经验证的内容

## 测试数据集

### 医保政策数据集

内置 5 条医保政策样本，覆盖：
- 正常样本
- 检索缺失
- 生成遗漏
- 前置条件/例外丢失
- 数字篡改

### NVIDIA 财务数据集

从 Python 版本迁移的测试数据：
- `nvda_cases.json`：Case 01-04
- `nvda_cases_05_10.json`：Case 05-10
- `nvda_structured_cases.json`：结构化抽取 Case 01-07

## 与 Python 版本的对比

| 特性 | Python 版本 | Java 版本 |
|------|------------|-----------|
| 数据模型 | Pydantic | Java 17 Records |
| HTTP 客户端 | httpx | java.net.http.HttpClient |
| JSON 处理 | 内置 | 简单字符串处理 |
| 并发模型 | asyncio | 同步（可扩展） |
| 依赖管理 | pip | Maven |

## 扩展指南

### 添加新的评测器

实现 `Evaluator` 接口：

```java
public class MyEvaluator implements Evaluator {
    @Override
    public MetricOutcome evaluate(EvalSample sample, String context, String answer) {
        // 实现评测逻辑
        return new MetricOutcome("my_metric", score, passed, details);
    }
}
```

### 添加新的事实抽取器

实现 `FactExtractor` 接口：

```java
public class MyFactExtractor implements FactExtractor {
    @Override
    public ExtractionPayload extract(String query, String context) {
        // 实现抽取逻辑
        return ExtractionPayload.of(facts, claims);
    }
}
```

### 添加新的答案合成器

实现 `AnswerSynthesizer` 接口：

```java
public class MySynthesizer implements AnswerSynthesizer {
    @Override
    public String synthesize(String query, ExtractionPayload payload) {
        // 实现合成逻辑
        return answer;
    }
}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
