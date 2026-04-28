---
title: "模型部署"
description: "Ollama/llama.cpp/vLLM/TGI 框架对比、Docker 部署、性能调优"
topics: [deployment, Ollama, llama-cpp, vLLM, TGI, TensorRT-LLM, Docker, GGUF]
prereqs: [engineering/quantization]
---
# 模型部署

::: info 一句话总结
模型部署是将训练好的大模型转化为可服务的 API，涉及推理框架选型（vLLM / TGI / llama.cpp / Ollama / TensorRT-LLM）、量化推理、容器化、性能调优和成本估算。
:::


## 在大模型体系中的位置

```
预训练 → SFT/RLHF → 量化/压缩 → 【模型部署】 → 上线服务
                                      ↑
                               你在这里
                     ├── 推理框架选型
                     ├── 服务化 API
                     ├── 容器化部署
                     ├── 性能监控
                     └── 成本优化
```

模型部署是大模型从"实验室里能跑"到"生产环境能用"的关键一步。选错部署方案可能导致数倍的成本差距或无法满足延迟要求。

## 部署形态总览

| 部署形态 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| **本地部署** | 数据隐私要求高、离线场景 | 数据不出域、无网络延迟 | 需自购硬件、维护成本高 |
| **云端部署** | 弹性需求、快速上线 | 按需扩缩容、运维省心 | 数据传输风险、长期成本高 |
| **边缘部署** | 端侧推理、低延迟场景 | 极低延迟、无网络依赖 | 模型大小受限、算力有限 |

### 不同规模模型的硬件需求估算

```
模型显存估算公式：
  显存 ≈ 参数量 × 每参数字节 × 1.2（KV Cache + 运行时开销系数）

推理吞吐估算（decode 阶段，memory-bound）：
  tokens/s ≈ 显存带宽(GB/s) / 模型大小(GB)
```

| 模型规模 | FP16 显存 | INT4 显存 | 推荐硬件（FP16）| 推荐硬件（INT4）|
|---------|----------|----------|---------------|---------------|
| **1.5B** | 3 GB | 1 GB | RTX 3060 | 任意 GPU / CPU |
| **7B** | 14 GB | 4 GB | RTX 4090 (24GB) | RTX 3060 (12GB) |
| **13B** | 26 GB | 7 GB | A100 40GB | RTX 4090 (24GB) |
| **34B** | 68 GB | 18 GB | 2×A100 40GB | A100 40GB |
| **70B** | 140 GB | 36 GB | 2×A100 80GB | A100 80GB / 2×RTX 4090 |

> 注意：以上为纯模型权重的估算，实际运行时 KV Cache 会额外占用大量显存，特别是长上下文场景。

---

## 推理框架对比

这是选型决策中最关键的一环。

| 特性 | **vLLM** | **TGI** | **llama.cpp** | **Ollama** | **TensorRT-LLM** |
|------|---------|---------|--------------|-----------|-----------------|
| 开发方 | UC Berkeley | Hugging Face | Georgi Gerganov | Ollama Inc | NVIDIA |
| 语言 | Python | Rust + Python | C/C++ | Go + C++ | C++ + Python |
| 核心优势 | PagedAttention | 生产级稳定 | CPU/低端GPU | 开箱即用 | 极致GPU性能 |
| GPU 支持 | CUDA | CUDA | CUDA/Metal/Vulkan | CUDA/Metal | 仅 CUDA |
| CPU 推理 | 不支持 | 不支持 | **最佳** | 支持 | 不支持 |
| 量化格式 | AWQ/GPTQ/FP8 | GPTQ/AWQ/BnB | **GGUF** | GGUF | FP8/INT4/INT8 |
| OpenAI 兼容 API | 原生支持 | 支持 | 通过 server 模式 | 原生支持 | 通过 Triton |
| 并发处理 | Continuous Batching | Continuous Batching | 有限 | 有限 | Inflight Batching |
| 学习曲线 | 中等 | 中等 | 低 | **最低** | 高 |
| 适用场景 | 高并发服务 | HF 生态集成 | 本地/边缘 | 本地体验 | 大规模生产 |

### 选型决策树

```
你的场景是什么？
├── 本地体验/开发测试 → Ollama（最简单）
├── 本地部署 + CPU 推理 → llama.cpp
├── 高并发线上服务
│   ├── NVIDIA GPU → vLLM（首选）或 TensorRT-LLM（极致性能）
│   └── HF 生态深度集成 → TGI
└── 边缘设备 / 移动端 → llama.cpp（GGUF 量化）
```

---

## Ollama 使用指南

Ollama 是目前**最简单的本地 LLM 部署方案**，一行命令即可运行。

### 安装与基础使用

```bash
# macOS / Linux 安装
curl -fsSL https://ollama.ai/install.sh | sh

# 或 macOS 直接下载 App
# https://ollama.ai/download

# 运行模型（自动下载）
ollama run llama3.1:8b

# 查看已下载的模型
ollama list

# 拉取模型（不运行）
ollama pull qwen2.5:7b

# 删除模型
ollama rm llama3.1:8b

# 查看模型详细信息
ollama show llama3.1:8b
```

### API 调用

Ollama 启动后默认监听 `http://localhost:11434`。

```python
import requests
import json

# === 基础 Chat API ===
def chat_with_ollama(prompt: str, model: str = "llama3.1:8b") -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
    )
    return response.json()["message"]["content"]

print(chat_with_ollama("什么是 Transformer？"))


# === 流式输出 ===
def chat_stream(prompt: str, model: str = "llama3.1:8b"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        },
        stream=True,
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if not data.get("done"):
                print(data["message"]["content"], end="", flush=True)
    print()

chat_stream("用三句话解释量子计算")


# === OpenAI 兼容 API（Ollama 原生支持）===
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # 任意值即可
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)


# === Embedding API ===
def get_embedding(text: str, model: str = "nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text},
    )
    return response.json()["embedding"]

emb = get_embedding("什么是大语言模型？")
print(f"Embedding 维度: {len(emb)}")
```

### 自定义 Modelfile

Modelfile 是 Ollama 的模型配置文件，类似 Dockerfile。

```dockerfile
# 文件名: Modelfile

# 基于已有模型
FROM llama3.1:8b

# 设置系统提示词
SYSTEM """你是一位资深的 Python 编程导师。
你的回答应该：
1. 包含完整的可运行代码
2. 详细解释每一步
3. 指出常见陷阱
4. 用中文回答
"""

# 调整推理参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 2048
PARAMETER stop "<|eot_id|>"

# 设置模板（自定义 prompt 格式）
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
```

```bash
# 从 Modelfile 创建自定义模型
ollama create python-tutor -f Modelfile

# 运行自定义模型
ollama run python-tutor

# 从 GGUF 文件创建模型
# Modelfile 内容：FROM ./my-model.gguf
ollama create my-local-model -f Modelfile
```

---

## llama.cpp 量化推理

llama.cpp 是 LLM 领域最重要的开源项目之一，它让大模型在 CPU 和消费级硬件上成为可能。

### GGUF 格式

GGUF（GPT-Generated Unified Format）是 llama.cpp 的模型格式，取代了旧的 GGML 格式。

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)                    # CPU 编译
# 或
make -j$(nproc) LLAMA_CUDA=1      # CUDA GPU 加速
# 或
make -j$(nproc) LLAMA_METAL=1     # macOS Metal 加速
```

### 量化级别对比

| 量化类型 | 每参数位数 | 7B 模型大小 | 质量损失 | 推荐场景 |
|---------|-----------|-----------|---------|---------|
| **Q2_K** | ~2.6 bit | ~2.7 GB | 明显 | 极端资源受限 |
| **Q3_K_M** | ~3.4 bit | ~3.3 GB | 较大 | 低端设备 |
| **Q4_K_M** | ~4.5 bit | ~4.1 GB | 轻微 | **性价比最优** |
| **Q5_K_M** | ~5.5 bit | ~4.8 GB | 极小 | 质量优先 |
| **Q6_K** | ~6.6 bit | ~5.5 GB | 几乎无 | 接近无损 |
| **Q8_0** | 8 bit | ~7.0 GB | 无 | 无损压缩基线 |
| **F16** | 16 bit | ~14 GB | 无 | 原始精度 |

> **推荐**：大多数场景使用 **Q4_K_M**，在质量和大小之间取得最佳平衡。

### 模型转换与量化

```bash
# 1. 将 HuggingFace 模型转换为 GGUF
python convert_hf_to_gguf.py \
    /path/to/hf-model \
    --outfile model-f16.gguf \
    --outtype f16

# 2. 量化
./llama-quantize model-f16.gguf model-q4km.gguf Q4_K_M

# 3. 运行推理
./llama-cli -m model-q4km.gguf \
    -p "What is machine learning?" \
    -n 256 \
    --temp 0.7 \
    -ngl 99        # GPU offload 层数，99 表示尽可能多

# 4. 启动 API 服务（OpenAI 兼容）
./llama-server -m model-q4km.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 4096 \       # 上下文长度
    --parallel 4    # 并发槽位数
```

### Python 调用 llama.cpp

```python
# 通过 llama-cpp-python 绑定调用
# pip install llama-cpp-python

from llama_cpp import Llama

# 加载量化模型
llm = Llama(
    model_path="./models/llama-3.1-8b-q4km.gguf",
    n_ctx=4096,         # 上下文长度
    n_gpu_layers=-1,    # -1 表示全部 offload 到 GPU
    n_threads=8,        # CPU 线程数（仅 CPU 推理时有效）
    verbose=False,
)

# 补全模式
output = llm(
    "The meaning of life is",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    echo=False,
)
print(output["choices"][0]["text"])

# Chat 模式
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "解释什么是注意力机制"},
    ],
    max_tokens=512,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])
```

---

## vLLM 服务化部署

vLLM 是目前**高并发 LLM 服务的首选框架**，其核心创新 PagedAttention 将 KV Cache 的显存利用率提升了数倍。

### 基础部署

```bash
# 安装
pip install vllm

# 最简启动（OpenAI 兼容 API）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# 带量化模型
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-7B-Chat-AWQ \
    --quantization awq \
    --port 8000

# 多 GPU Tensor Parallel
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000
```

### 关键参数调优

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    \
    # === 显存管理 ===
    --gpu-memory-utilization 0.90 \    # GPU 显存使用比例（默认 0.9）
    --max-model-len 8192 \             # 最大序列长度
    --block-size 16 \                  # PagedAttention block 大小
    \
    # === 并发与调度 ===
    --max-num-seqs 256 \               # 最大并发序列数
    --max-num-batched-tokens 8192 \    # 单个 batch 最大 token 数
    \
    # === 多 GPU ===
    --tensor-parallel-size 1 \         # TP 并行度
    --pipeline-parallel-size 1 \       # PP 并行度（vLLM 0.4+）
    \
    # === 量化 ===
    --quantization awq \               # 量化方法：awq, gptq, fp8
    --dtype float16                    # 数据类型
```

### Python SDK 调用

```python
# === 在线服务模式：通过 OpenAI 兼容 API 调用 ===
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-placeholder",
)

# 普通请求
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain PagedAttention in 3 sentences."},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)

# 流式请求
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)


# === 离线批量推理模式 ===
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

prompts = [
    "What is deep learning?",
    "Explain attention mechanism.",
    "What is RLHF?",
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated = output.outputs[0].text
    print(f"Prompt: {prompt[:50]}...")
    print(f"Output: {generated[:100]}...\n")
```

### vLLM 性能基准测试

```python
"""vLLM 吞吐量和延迟基准测试脚本"""
import time
import asyncio
import aiohttp
import statistics

async def send_request(session, url, payload):
    """发送单个请求并返回延迟"""
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        result = await resp.json()
        latency = time.perf_counter() - start
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        return latency, tokens

async def benchmark(
    url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_requests: int = 100,
    concurrency: int = 10,
    max_tokens: int = 128,
):
    """并发压测 vLLM 服务"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Tell me a short joke."}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    total_tokens = 0
    
    async def bounded_request(session):
        nonlocal total_tokens
        async with semaphore:
            latency, tokens = await send_request(session, url, payload)
            latencies.append(latency)
            total_tokens += tokens
    
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session) for _ in range(num_requests)]
        await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    print(f"=== vLLM Benchmark Results ===")
    print(f"并发数: {concurrency}")
    print(f"总请求数: {num_requests}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"QPS: {num_requests / total_time:.2f}")
    print(f"总生成 tokens: {total_tokens}")
    print(f"吞吐量: {total_tokens / total_time:.2f} tokens/s")
    print(f"延迟 P50: {statistics.median(latencies)*1000:.0f}ms")
    print(f"延迟 P90: {sorted(latencies)[int(len(latencies)*0.9)]*1000:.0f}ms")
    print(f"延迟 P99: {sorted(latencies)[int(len(latencies)*0.99)]*1000:.0f}ms")

# asyncio.run(benchmark(concurrency=20, num_requests=200))
```

---

## Docker 容器化部署

### vLLM Docker 部署

```dockerfile
# Dockerfile.vllm
FROM vllm/vllm-openai:latest

# 预下载模型（可选，也可挂载卷）
# RUN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('meta-llama/Llama-3.1-8B-Instruct')"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", \
     "--port", "8000", \
     "--gpu-memory-utilization", "0.9"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model meta-llama/Llama-3.1-8B-Instruct
      --port 8000
      --gpu-memory-utilization 0.9
      --max-model-len 8192
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

```bash
# 启动
docker compose up -d

# 查看日志
docker compose logs -f vllm

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 性能调优实战

### GPU 利用率优化

```python
"""
GPU 利用率诊断与优化策略

常见问题：
1. GPU 利用率低（< 50%）→ batch size 太小，或者 decode 阶段 memory-bound
2. 显存 OOM → max-model-len 或 max-num-seqs 过大
3. 延迟高 → 检查是否有 CPU offload、检查网络延迟
"""

# 监控 GPU 状态
import subprocess
import json

def get_gpu_stats():
    """获取 GPU 使用情况"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "utilization_pct": int(parts[2]),
            "memory_used_mb": int(parts[3]),
            "memory_total_mb": int(parts[4]),
            "temperature_c": int(parts[5]),
        })
    return gpus

def print_gpu_status():
    for gpu in get_gpu_stats():
        mem_pct = gpu["memory_used_mb"] / gpu["memory_total_mb"] * 100
        print(f"GPU {gpu['index']} ({gpu['name']})")
        print(f"  利用率: {gpu['utilization_pct']}%")
        print(f"  显存: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB ({mem_pct:.1f}%)")
        print(f"  温度: {gpu['temperature_c']}°C")

# print_gpu_status()
```

### 调优参数速查表

| 参数 | 影响 | 调优方向 |
|------|------|---------|
| `gpu-memory-utilization` | 显存使用比例 | 提高到 0.95 可增加 KV Cache 空间 |
| `max-num-seqs` | 最大并发数 | 增大可提高吞吐，但增加延迟 |
| `max-model-len` | 最大上下文 | 减小可腾出更多 KV Cache 空间 |
| `tensor-parallel-size` | TP 并行度 | 模型太大放不下单卡时增加 |
| `enforce-eager` | 禁用 CUDA Graph | 调试用，生产关闭 |
| `enable-prefix-caching` | 前缀缓存 | 相同前缀的请求可复用 KV Cache |

### 成本估算

```python
def estimate_deployment_cost(
    model_params_b: float,         # 模型参数量（B）
    precision: str = "fp16",       # fp16, int8, int4
    expected_qps: float = 10,      # 预期 QPS
    avg_output_tokens: int = 200,  # 平均输出 token 数
    cloud_provider: str = "aws",   # aws, azure, gcp
):
    """估算部署成本"""
    bytes_per_param = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    model_size_gb = model_params_b * bytes_per_param[precision]
    
    # KV Cache 额外开销（粗略估计 20-50%）
    total_memory_gb = model_size_gb * 1.4
    
    # GPU 选型
    gpu_options = {
        "A100_40GB": {"memory": 40, "price_per_hour": 3.67,  "bandwidth_gbs": 1555},
        "A100_80GB": {"memory": 80, "price_per_hour": 5.12,  "bandwidth_gbs": 2039},
        "H100_80GB": {"memory": 80, "price_per_hour": 8.50,  "bandwidth_gbs": 3350},
        "L4_24GB":   {"memory": 24, "price_per_hour": 0.81,  "bandwidth_gbs": 300},
        "T4_16GB":   {"memory": 16, "price_per_hour": 0.53,  "bandwidth_gbs": 300},
    }
    
    print(f"模型: {model_params_b}B ({precision})")
    print(f"模型大小: {model_size_gb:.1f} GB")
    print(f"预估总显存需求: {total_memory_gb:.1f} GB")
    print(f"预期 QPS: {expected_qps}")
    print()
    
    for gpu_name, spec in gpu_options.items():
        num_gpus = max(1, int(total_memory_gb / spec["memory"]) + 1)
        
        # 粗略吞吐估计
        tokens_per_sec = spec["bandwidth_gbs"] / model_size_gb * num_gpus
        achievable_qps = tokens_per_sec / avg_output_tokens
        
        # 需要多少组来满足 QPS
        num_replicas = max(1, int(expected_qps / achievable_qps) + 1)
        total_gpus = num_gpus * num_replicas
        
        monthly_cost = total_gpus * spec["price_per_hour"] * 24 * 30
        cost_per_1k_tokens = (spec["price_per_hour"] * num_gpus) / (achievable_qps * avg_output_tokens * 3.6)
        
        print(f"{gpu_name}:")
        print(f"  每组 GPU 数: {num_gpus}, 副本数: {num_replicas}, 总 GPU: {total_gpus}")
        print(f"  单组吞吐: ~{tokens_per_sec:.0f} tok/s ({achievable_qps:.1f} QPS)")
        print(f"  月费用: ${monthly_cost:,.0f}")
        print()

# estimate_deployment_cost(model_params_b=7, precision="int4", expected_qps=10)
```

---

## 监控指标

部署上线后，需要监控以下关键指标：

| 指标 | 说明 | 健康范围 |
|------|------|---------|
| **TTFT** (Time To First Token) | 首 token 延迟 | < 500ms |
| **TPOT** (Time Per Output Token) | 每 token 生成时间 | < 50ms |
| **E2E Latency** | 端到端延迟 | 取决于输出长度 |
| **Throughput** (tokens/s) | 系统总吞吐 | 越高越好 |
| **QPS** | 每秒请求数 | 取决于业务需求 |
| **GPU Utilization** | GPU 计算利用率 | > 70% |
| **GPU Memory** | 显存使用率 | 80-95% |
| **Queue Depth** | 等待队列长度 | < max_num_seqs × 2 |
| **Error Rate** | 请求错误率 | < 1% |

```python
"""Prometheus 指标采集示例（配合 vLLM metrics 端点）"""
import requests
import time

def collect_vllm_metrics(base_url: str = "http://localhost:8000"):
    """采集 vLLM 暴露的 Prometheus 指标"""
    resp = requests.get(f"{base_url}/metrics")
    
    metrics = {}
    for line in resp.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            metrics[parts[0]] = float(parts[1])
    
    # 提取关键指标
    key_metrics = {
        "num_requests_running": metrics.get("vllm:num_requests_running", 0),
        "num_requests_waiting": metrics.get("vllm:num_requests_waiting", 0),
        "gpu_cache_usage_pct": metrics.get("vllm:gpu_cache_usage_perc", 0),
        "avg_generation_throughput": metrics.get("vllm:avg_generation_throughput_toks_per_s", 0),
    }
    
    return key_metrics

# 简单的监控循环
def monitor_loop(interval: int = 10):
    while True:
        try:
            m = collect_vllm_metrics()
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"Running: {m['num_requests_running']:.0f} | "
                  f"Waiting: {m['num_requests_waiting']:.0f} | "
                  f"Cache: {m['gpu_cache_usage_pct']:.1%} | "
                  f"Throughput: {m['avg_generation_throughput']:.0f} tok/s")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(interval)

# monitor_loop()
```

---

## 苏格拉底时刻

1. **vLLM 的 PagedAttention 借鉴了操作系统的虚拟内存分页机制。为什么 KV Cache 的管理问题和内存管理问题如此相似？这种跨领域的类比还能应用到 LLM 系统的哪些方面？**

2. **llama.cpp 用 C/C++ 重写了整个推理流程而不是用 PyTorch。为什么纯 C++ 实现能在 CPU 上比 PyTorch 快这么多？PyTorch 的开销具体在哪里？**

3. **INT4 量化会损失精度，但实际使用中用户往往感知不到差异。这是否说明 LLM 的大部分参数是"冗余"的？如果是，为什么训练时需要这么多参数？**

4. **Tensor Parallel 和 Pipeline Parallel 分别适合什么场景？如果你有 4 张 A100，部署一个 70B 模型，你会选择哪种并行策略？为什么？**

---

## 常见问题 & 面试考点

**Q: vLLM 和 TensorRT-LLM 各自的优势是什么？如何选择？**

A: vLLM 优势在于易用性和 PagedAttention 带来的高显存利用率，适合快速部署和中等规模服务；TensorRT-LLM 优势在于 NVIDIA 深度优化的算子融合和 FP8 支持，极致性能但部署复杂度高。选择标准：如果团队有 NVIDIA 工程支持且追求极致性能，选 TensorRT-LLM；否则选 vLLM。

**Q: 如何估算一个模型需要多少 GPU 显存？**

A: 公式：显存 ≈ 参数量 × 每参数字节 × 1.2~1.5（KV Cache 系数）。例如 70B FP16：70 × 2 × 1.3 ≈ 182 GB，需要 3 张 A100 80GB 或 2 张 H100。KV Cache 系数取决于上下文长度和并发数。

**Q: Continuous Batching 相比 Static Batching 的优势是什么？**

A: Static Batching 中，一个 batch 内所有请求必须等最长的那个完成才能返回，短请求被"拖累"。Continuous Batching 允许已完成的请求立即返回，新请求立即加入，GPU 利用率可以提升 2-10 倍。

**Q: 什么时候应该用 Ollama 而不是 vLLM？**

A: Ollama 适合本地开发测试、快速体验、单用户场景；vLLM 适合多用户高并发的线上服务。如果你的需求是"在笔记本上跑一个模型聊天"，用 Ollama；如果是"部署一个 API 服务给 100 个用户同时用"，用 vLLM。

---

## 推荐资源

- [vLLM 官方文档](https://docs.vllm.ai/) — 高并发推理框架
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) — CPU/边缘推理的基石
- [Ollama 官方文档](https://ollama.ai/) — 最简单的本地部署方案
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA 官方高性能推理框架
- [Efficient Inference on a Single GPU (HuggingFace)](https://huggingface.co/docs/transformers/perf_infer_gpu_one) — 单 GPU 推理优化指南
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180) — vLLM 的核心创新
