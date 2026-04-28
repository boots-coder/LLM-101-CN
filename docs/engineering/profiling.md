---
title: "GPU 性能分析与调试"
description: "显存计算、torch.profiler 实战、性能瓶颈定位与优化"
topics: [profiling, GPU-memory, torch-profiler, CUDA, bottleneck, memory-estimation, roofline]
prereqs: [engineering/inference, engineering/distributed]
---
# GPU 性能分析与调试

> **一句话总结:** 不会 profiling 的工程师只能靠猜来优化——掌握 GPU 显存计算、性能瓶颈定位和 profiling 工具，才能把"感觉慢"变成"知道哪里慢、为什么慢、怎么修"。

## 在大模型体系中的位置

```
Engineering
  ├── Inference Optimization  → 你知道有哪些优化技术
  ├── Distributed Training    → 你知道怎么并行
  ├── Profiling               ◄── 你在这里：你知道怎么找到瓶颈
  └── Quantization            → 你知道怎么压缩
```

推理优化和分布式训练告诉你"有哪些技术"，但 profiling 告诉你"该用哪个"。不做性能分析就上优化技术，相当于闭眼开车。

---

## GPU 显存计算

### 模型参数显存

模型参数的显存占用取决于**参数量**和**数据类型**：

$$
\text{Memory}_{\text{params}} = N_{\text{params}} \times \text{bytes\_per\_param}
$$

| 数据类型 | 每参数字节数 | 7B 模型占用 |
|----------|-------------|------------|
| FP32 | 4 bytes | 28 GB |
| FP16 / BF16 | 2 bytes | 14 GB |
| INT8 | 1 byte | 7 GB |
| INT4 | 0.5 bytes | 3.5 GB |

**快速估算公式：** 参数量（B）× 每参数字节数 = 显存（GB）。7B FP16 模型 ≈ 7 × 2 = 14 GB。

### 训练时的显存组成

训练显存远大于推理，因为需要存储梯度和优化器状态：

$$
\text{Memory}_{\text{train}} = \text{Model} + \text{Gradients} + \text{Optimizer} + \text{Activations}
$$

以 AdamW + FP16 混合精度训练为例：

| 组件 | 每参数字节数 | 7B 模型 |
|------|-------------|---------|
| FP16 模型参数 | 2 | 14 GB |
| FP16 梯度 | 2 | 14 GB |
| FP32 主参数（Adam） | 4 | 28 GB |
| FP32 一阶矩 m | 4 | 28 GB |
| FP32 二阶矩 v | 4 | 28 GB |
| **合计（不含激活）** | **16** | **112 GB** |

::: warning 常见误区
很多人以为 7B FP16 模型训练只需要 14GB 显存。实际上 AdamW 混合精度训练需要约 16 字节/参数 = **112 GB**，这还不包括激活值和 batch 数据！
:::

### 激活值显存

激活值是前向传播过程中每层的中间结果，需要保存用于反向传播。对于 Transformer：

$$
\text{Memory}_{\text{act}} \approx 2 \times \text{seq\_len} \times \text{batch\_size} \times \text{hidden\_dim} \times \text{n\_layers} \times \text{bytes}
$$

**激活检查点（Activation Checkpointing）** 可以用计算换显存：不保存所有激活，只保存部分检查点，反向传播时重新计算中间层。显存从 $O(n)$ 降到 $O(\sqrt{n})$，但计算量增加约 33%。

```python
# PyTorch 激活检查点
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # 不使用检查点：保存所有中间激活
        # x = self.attention(x)
        # x = self.ffn(x)
        
        # 使用检查点：前向传播不保存激活，反向传播时重算
        x = checkpoint(self.attention, x, use_reentrant=False)
        x = checkpoint(self.ffn, x, use_reentrant=False)
        return x
```

### KV Cache 显存

推理时 KV Cache 的显存占用：

$$
\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch} \times \text{bytes}
$$

以 Llama-2-7B（32 层，32 头，head_dim=128）为例，FP16 下：
- 单条序列 4K 长度：$2 \times 32 \times 32 \times 128 \times 4096 \times 2 = 2$ GB
- batch_size=8：**16 GB**，比模型参数还大！

这就是为什么长序列推理的显存瓶颈往往在 KV Cache 而非模型参数。

### 实用：快速估算脚本

```python
def estimate_memory(
    n_params_b: float,      # 参数量（B）
    dtype_bytes: int = 2,   # 数据类型字节数
    mode: str = "inference", # "inference" or "train"
    optimizer: str = "adamw",
    seq_len: int = 4096,
    batch_size: int = 1,
    n_layers: int = 32,
    n_kv_heads: int = 32,
    head_dim: int = 128,
):
    """快速估算 GPU 显存需求（GB）"""
    param_mem = n_params_b * dtype_bytes  # GB
    
    if mode == "inference":
        kv_cache = (2 * n_layers * n_kv_heads * head_dim 
                    * seq_len * batch_size * dtype_bytes) / 1e9
        total = param_mem + kv_cache
        print(f"=== Inference Memory ({n_params_b}B, {dtype_bytes}B/param) ===")
        print(f"  Model params:  {param_mem:.1f} GB")
        print(f"  KV Cache:      {kv_cache:.1f} GB")
        print(f"  Total:         {total:.1f} GB")
    else:
        grad_mem = n_params_b * dtype_bytes
        if optimizer == "adamw":
            opt_mem = n_params_b * (4 + 4 + 4)  # master + m + v, all FP32
        else:
            opt_mem = 0
        total = param_mem + grad_mem + opt_mem
        print(f"=== Training Memory ({n_params_b}B, AdamW mixed-precision) ===")
        print(f"  Model params (FP16): {param_mem:.1f} GB")
        print(f"  Gradients (FP16):    {grad_mem:.1f} GB")
        print(f"  Optimizer (FP32):    {opt_mem:.1f} GB")
        print(f"  Total (excl. act):   {total:.1f} GB")
    return total

# 示例
estimate_memory(7, dtype_bytes=2, mode="inference", seq_len=4096)
# Model params: 14.0 GB, KV Cache: 2.0 GB, Total: 16.0 GB

estimate_memory(7, dtype_bytes=2, mode="train")
# Model params: 14.0 GB, Gradients: 14.0 GB, Optimizer: 84.0 GB, Total: 112.0 GB
```

---

## torch.profiler 实战

### 基础用法

`torch.profiler` 是 PyTorch 官方的性能分析工具，可以捕获 CPU/GPU 的算子执行时间、显存分配、CUDA 核心启动等信息。

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

model = ...  # 你的模型
input_data = ...  # 输入数据

# 基础 profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,       # 记录 tensor shape
    profile_memory=True,      # 记录显存分配
    with_stack=True,          # 记录 Python 调用栈
) as prof:
    with record_function("model_inference"):
        output = model(input_data)

# 打印耗时最高的 20 个算子
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))
```

输出示例：

```
-----------------------------------  -------  -------  -------  
                              Name  CPU Time CUDA Time  # Calls
-----------------------------------  -------  -------  -------  
                    model_inference   52.3ms   48.1ms        1
                        aten::mm    18.2ms   17.8ms       64
                   aten::softmax     3.1ms    2.9ms       32
               aten::layer_norm_     2.8ms    2.6ms       32
                    aten::linear    12.1ms   11.8ms       96
-----------------------------------  -------  -------  -------  
```

### 带 Warmup 的 Schedule Profiling

实际使用中，应跳过前几次迭代（GPU 预热），只对稳态进行 profiling：

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,      # 跳过第 1 次迭代
        warmup=2,    # 预热 2 次（不记录）
        active=3,    # 记录 3 次
        repeat=1,    # 重复 1 轮
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiling_logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 6:  # 1 + 2 + 3
            break
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()  # 通知 profiler 一次迭代结束
```

然后用 TensorBoard 查看：

```bash
tensorboard --logdir=./profiling_logs
# 打开浏览器访问 http://localhost:6006/#pytorch_profiler
```

### 自定义标注

用 `record_function` 给代码段打标签，在 profiling 结果中精确定位：

```python
class MyTransformer(nn.Module):
    def forward(self, x):
        with record_function("attention"):
            attn_out = self.attention(x)
        with record_function("ffn"):
            ffn_out = self.ffn(attn_out)
        return ffn_out
```

---

## 常见性能瓶颈与优化

### 1. CPU-GPU 同步阻塞

**症状**：GPU 利用率忽高忽低，`nvidia-smi` 显示 GPU 使用率波动剧烈。

**原因**：某些操作（如 `.item()`、`print(tensor)`、`tensor.cpu()`）会触发 CPU 等待 GPU 完成所有计算，造成流水线气泡。

```python
# BAD: 每步都触发同步
for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    print(f"Step {step}, loss = {loss.item()}")  # .item() 触发同步！

# GOOD: 减少同步频率
for step, batch in enumerate(dataloader):
    loss = model(batch).loss
    loss.backward()
    optimizer.step()
    if step % 100 == 0:  # 每 100 步才同步一次
        print(f"Step {step}, loss = {loss.item()}")
```

### 2. 显存碎片化

**症状**：`torch.cuda.memory_allocated()` 远小于 `torch.cuda.memory_reserved()`，OOM 时实际已分配显存不到 GPU 总显存的 80%。

**解决方案**：

```python
import torch

# 查看显存使用情况
def print_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, "
          f"Fragmentation: {(reserved - allocated) / reserved * 100:.1f}%")

# 方法 1：设置显存分配策略
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 方法 2：定期清理缓存（谨慎使用，有性能开销）
torch.cuda.empty_cache()
```

### 3. 数据加载成为瓶颈

**症状**：GPU 利用率很低，大量时间花在等待数据。

```python
# 检查数据加载是否是瓶颈
import time

for batch in dataloader:
    t0 = time.time()
    batch = batch.to(device)
    t_load = time.time() - t0
    
    t0 = time.time()
    output = model(batch)
    loss.backward()
    torch.cuda.synchronize()
    t_compute = time.time() - t0
    
    # 如果 t_load >> t_compute，数据加载是瓶颈
    print(f"Load: {t_load*1000:.1f}ms, Compute: {t_compute*1000:.1f}ms")
```

**优化方法**：
- 增加 `num_workers`（DataLoader 并行加载）
- 使用 `pin_memory=True`（加速 CPU→GPU 传输）
- 预处理数据到二进制格式（如 Arrow/Parquet）

### 4. 矩阵维度不对齐

**症状**：matmul 速度比预期慢很多。

GPU 的 Tensor Core 要求矩阵维度是 8（FP16）或 16（INT8）的倍数。不对齐时 GPU 需要额外 padding，浪费计算资源。

```python
# BAD: hidden_dim=1000，不是 8 的倍数
nn.Linear(1000, 1000)  # Tensor Core 效率低

# GOOD: hidden_dim=1024，是 8 的倍数
nn.Linear(1024, 1024)  # Tensor Core 满效率

# 这就是为什么 Llama 的 hidden_dim = 4096, 5120, 8192... 都是 128 的倍数
```

---

## 实战：Profiling 一个 Transformer

以下是一个完整的 profiling 工作流，用于定位 Transformer 推理的瓶颈：

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# 模拟一个简单的 Transformer 层
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=1024, nhead=16, num_layers=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
    
    def forward(self, x):
        return self.encoder(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer().to(device).half()  # FP16
x = torch.randn(4, 512, 1024, device=device, dtype=torch.float16)

# Warmup
for _ in range(3):
    _ = model(x)
torch.cuda.synchronize()

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    for _ in range(5):
        with record_function("forward"):
            output = model(x)
        torch.cuda.synchronize()

# 分析结果
print("=== Top 10 CUDA Operations ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("\n=== Memory Usage ===")
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
```

**分析技巧**：
1. 如果 `aten::mm`（矩阵乘）占大头 → 正常，这是计算密集型
2. 如果 `aten::copy_` 占比高 → 可能存在不必要的 CPU-GPU 数据搬运
3. 如果 `cudaStreamSynchronize` 占比高 → 有同步阻塞
4. 如果某个算子的 CUDA 时间远大于 CPU 时间 → GPU 是瓶颈（计算密集）
5. 如果 CPU 时间远大于 CUDA 时间 → CPU 是瓶颈（数据预处理/IO）

---

## Roofline 模型：判断计算密集 vs 访存密集

**Roofline 模型**是理解 GPU 性能的核心框架。它将算子分为两类：

- **计算密集（Compute-bound）**：瓶颈在 GPU 算力（FLOPS）。如大矩阵乘法。
- **访存密集（Memory-bound）**：瓶颈在显存带宽（GB/s）。如 LayerNorm、Softmax、逐元素操作。

关键指标是**算术强度（Arithmetic Intensity）**：

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}
$$

| 操作 | 算术强度 | 类型 | 优化方向 |
|------|---------|------|---------|
| 大矩阵乘法 (GEMM) | 高 | 计算密集 | 提升 GPU 算力利用率 |
| Softmax | 低 | 访存密集 | 减少显存访问（Flash Attention） |
| LayerNorm | 低 | 访存密集 | 算子融合（Kernel Fusion） |
| 逐元素操作 | 极低 | 访存密集 | 算子融合 |
| Attention (无 Flash) | 中 | 访存密集 | Flash Attention |

**Flash Attention 为什么快？** 传统 Attention 需要将 $QK^T$ 矩阵（$O(n^2)$ 大小）写入 HBM 再读回来做 Softmax。Flash Attention 通过分块计算（tiling），让整个 Attention 在 SRAM 中完成，避免了 HBM 读写，把访存密集的操作变成了计算密集的操作。

---

## 苏格拉底时刻

1. 一个 13B 参数的模型，用 FP16 推理需要多少显存？用 AdamW 混合精度训练呢？一张 80GB A100 够吗？
2. 为什么 KV Cache 的显存可以超过模型参数本身？在什么条件下会发生？
3. Activation Checkpointing 用计算换显存，额外计算量约 33%——这个数字是怎么来的？
4. 如果 profiling 发现 `aten::copy_` 操作耗时很高，最可能的原因是什么？如何优化？
5. 为什么模型的 hidden_dim 通常设为 128 的倍数？如果设为 1000 会怎样？

---

## 常见问题 & 面试考点

| 问题 | 要点 |
|------|------|
| 7B 模型 FP16 推理需要多少显存？ | 约 14 GB 模型 + KV Cache（取决于 seq_len 和 batch_size） |
| 训练时显存的四大组成？ | 参数、梯度、优化器状态、激活值 |
| AdamW 每参数多少字节？ | 混合精度训练：2(FP16 参数) + 2(FP16 梯度) + 12(FP32 主参数+m+v) = 16 字节 |
| 什么是 Activation Checkpointing？ | 不保存中间激活，反向传播时重算。显存 $O(\sqrt{n})$，计算 +33% |
| 计算密集 vs 访存密集怎么判断？ | 看算术强度。GEMM 是计算密集，Softmax/LayerNorm 是访存密集 |
| Flash Attention 快在哪？ | 避免 $O(n^2)$ 的 Attention 矩阵写入 HBM，在 SRAM 中分块完成 |

---

## 推荐资源

- **PyTorch Profiler 官方文档** — `torch.profiler` 的完整 API 和教程
- **NVIDIA Nsight Systems** — GPU 级别的性能分析工具，可视化 CUDA kernel 执行时间线
- **NVIDIA Nsight Compute** — 单个 CUDA kernel 的深度分析（occupancy、memory throughput）
- **Efficient Large Language Model Training（Stas Bekman）** — 大模型训练的显存和性能优化实战指南
- **FlashAttention 论文（Dao et al.）** — 理解 IO-aware 算法设计的经典论文
- **Roofline Model 原始论文（Williams et al.）** — 理解计算密集 vs 访存密集的理论框架
