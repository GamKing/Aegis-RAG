# Aegis-RAG 压测基准与性能实战 (Stress Test Suite)

本目录承载 **Day 10 - Day 13：低成本压测基准跑通与两段式开销实测（Infra & Systems 实战篇）** 的全部压测代码、模型配置与基准数据报告。

---

## 目录结构

```text
stressTest/
├── Modelfile                                     # 4096 上下文专属基准镜像构建文件
├── benchmark_concurrency.py                      # 任务一：并发阶梯压测脚本 (C=1, 2, 4, 8)
├── benchmark_two_stage_comparison.py             # 任务二：单阶段 vs 两段式开销对比脚本 (C=4)
├── RTX4080并发阶梯压测与性能基准报告.md          # 任务一：阶梯压测与物理拐点分析报告
├── RAG生成端吞吐与显存容量基准压测报告.md        # 任务二：两段式开销实测与算力经济学报告 (Day 13 交付物)
└── README.md                                     # 本说明文档
```

---

## 核心压测脚本与快速复现

### 1. 设置环境变量并启动推理服务

```powershell
# 允许最大处理 4 个并行请求槽位
$env:OLLAMA_NUM_PARALLEL="4"

# 强制开启 FlashAttention (RTX 4080 Ada 架构必备)
$env:OLLAMA_FLASH_ATTENTION="1"

# 模型存储路径与服务监听地址
$env:OLLAMA_MODELS="D:\ollama\models"
$env:OLLAMA_HOST="127.0.0.1:22434"

# 启动服务
ollama serve
```

### 2. 构建基准镜像

```powershell
ollama create qwen2.5-benchmark -f ./Modelfile
```

### 3. 执行压测任务

* **任务一：并发阶梯压测与物理拐点探测**
  ```powershell
  python benchmark_concurrency.py
  ```
  详见报告：[RTX4080并发阶梯压测与性能基准报告.md](file:///D:/source/evalProject/stressTest/RTX4080并发阶梯压测与性能基准报告.md)

* **任务二（轻量版）：单阶段 vs 两段式开销对比脚本**
  ```powershell
  python benchmark_two_stage_comparison.py
  ```
  详见报告：[RAG生成端吞吐与显存容量基准压测报告.md](file:///D:/source/evalProject/stressTest/RAG生成端吞吐与显存容量基准压测报告.md)

* **任务二（深度集成版）：直接调用当前项目真实代码库（Tier1CodeVerifier + EntityAwareRealizer）**
  ```powershell
  python benchmark_project_pipeline.py
  ```
  详见报告：[基于Aegis_RAG真实代码库的两段式端到端性能压测报告.md](file:///D:/source/evalProject/基于Aegis_RAG真实代码库的两段式端到端性能压测报告.md)

* **连续批处理专项：Continuous Batching 动态插队与迭代级调度压测**
  ```powershell
  python benchmark_continuous_batching.py
  ```
  详见报告：[Continuous_Batching动态连续批处理真机实测与深度剖析.md](file:///D:/source/evalProject/Continuous_Batching动态连续批处理真机实测与深度剖析.md)

* **分页与前缀缓存专项：PagedAttention 超额并发防爆与前缀零拷贝共享压测**
  ```powershell
  python benchmark_paged_attention.py
  ```
  详见报告：[PagedAttention专项压测报告——超额并发防爆与前缀零拷贝共享实测.md](file:///D:/source/evalProject/PagedAttention专项压测报告——超额并发防爆与前缀零拷贝共享实测.md)

* **算力切片与流式 SLA 专项：Chunked Prefill 抗干扰与 P99 ITL 抖动压测**
  ```powershell
  python benchmark_chunked_prefill.py
  ```
  详见报告：[Chunked_Prefill专项压测报告——流式打字机P99_ITL与算力抢占抗干扰实测.md](file:///D:/source/evalProject/Chunked_Prefill专项压测报告——流式打字机P99_ITL与算力抢占抗干扰实测.md)





---

## 任务二核心实测成果速览 (并发 C=4)

| 核心评估指标 | 对照组：传统单阶段散文生成 | 实验组：Aegis-RAG 两段式 | 差异增益 (Delta) |
| :--- | :---: | :---: | :---: |
| **平均端到端耗时** | **8.296 秒** | **2.613 秒** | **-68.5% (提速 3.17 倍)** |
| **单请求输出 Tokens** | **244.0 tokens** | **29.0 tokens** | **-88.1% (节省近 9 成)** |
| **批次总墙钟耗时** | **8.35 秒** | **2.67 秒** | **-68.0% (周转率升 3.12 倍)** |
| **第二阶段装配耗时** | 无 | **0.006 毫秒 (6 微秒)** | 纯代码执行，零 GPU 消耗 |
