---
title: "分布式训练"
description: "数据并行、张量并行、流水线并行、ZeRO、FSDP、Context Parallelism、Expert Parallelism"
topics: [distributed-training, data-parallelism, tensor-parallelism, pipeline-parallelism, ZeRO, FSDP, context-parallelism, expert-parallelism, ring-attention]
prereqs: [training/pretraining]
---
# 分布式训练

::: info 一句话总结
当一张 GPU 装不下一个大模型时，分布式训练通过**数据并行、张量并行、流水线并行**及其组合，将计算和显存需求分摊到多张 GPU 甚至多台机器上，使得百亿乃至万亿参数模型的训练成为可能。
:::


## 在大模型体系中的位置

```
┌────────────────────────────────────────────────────────────────────┐
│                    LLM Engineering Overview                        │
│                                                                    │
│  Data Prep → Model Design → [Distributed Training] → Inference → Eval │
│                                      ↑                             │
│                                 You are here                       │
│                                                                    │
│  Distributed training bridges "designing a model" and              │
│  "actually training it". Models >10B params cannot be              │
│  trained in reasonable time without distributed training.          │
└────────────────────────────────────────────────────────────────────┘
```

分布式训练不仅仅是"把模型放到多张卡上"这么简单。它涉及到：

- **显存管理**：如何将参数、梯度、优化器状态、激活值分布到多个设备
- **通信优化**：如何最小化设备间数据传输的开销
- **计算效率**：如何最大化 GPU 利用率，减少空闲等待
- **系统设计**：如何设计拓扑结构、选择并行策略、处理容错

---

## 为什么需要分布式？

### 一个具体的例子：训练 Llama 70B

让我们用**具体数字**来算一下训练 Llama 70B 到底需要多少显存。

**模型参数**：70B（700 亿）参数

以混合精度训练（BF16/FP16）为例，我们需要存储以下内容：

| 存储项目 | 数据类型 | 每参数字节数 | 总显存 |
|---------|---------|------------|-------|
| 模型参数（FP16） | float16 | 2 bytes | $70 \times 10^9 \times 2 = 140$ GB |
| 模型参数（FP32 master copy） | float32 | 4 bytes | $70 \times 10^9 \times 4 = 280$ GB |
| 梯度（FP16） | float16 | 2 bytes | $70 \times 10^9 \times 2 = 140$ GB |
| Adam 一阶动量 (m) | float32 | 4 bytes | $70 \times 10^9 \times 4 = 280$ GB |
| Adam 二阶动量 (v) | float32 | 4 bytes | $70 \times 10^9 \times 4 = 280$ GB |
| **合计（不含激活值）** | | | **~1120 GB** |

::: warning 这还没算激活值！
训练时的中间激活值（activation）也需要大量显存。对于 Llama 70B，一个 batch 的激活值可能需要数十到上百 GB，取决于序列长度和 batch size。
:::


**单张 A100 80GB 的显存**：80 GB

$$
\frac{1120 \text{ GB}}{80 \text{ GB/卡}} = 14 \text{ 张卡}
$$

即使只算参数+梯度+优化器状态，至少需要 **14 张 A100 80GB**，而且还没有给激活值留空间！

### 显存占用的"四大金刚"

对于使用 Adam 优化器的混合精度训练，每个参数 $\Phi$ 的显存占用为：

$$
\text{总显存} = \underbrace{2\Phi}_{\text{FP16 参数}} + \underbrace{2\Phi}_{\text{FP16 梯度}} + \underbrace{4\Phi}_{\text{FP32 master}} + \underbrace{4\Phi}_{\text{Adam m}} + \underbrace{4\Phi}_{\text{Adam v}} = 16\Phi \text{ bytes}
$$

也就是说，**每个参数需要 16 字节的显存**。

| 模型规模 | 参数量 $\Phi$ | 最少显存需求 (16$\Phi$) | 需要 A100 80GB 数量 |
|---------|-------------|----------------------|-------------------|
| GPT-2 | 1.5B | 24 GB | 1 |
| Llama 7B | 7B | 112 GB | 2 |
| Llama 13B | 13B | 208 GB | 3 |
| Llama 70B | 70B | 1120 GB | 14 |
| GPT-4（传闻） | ~1.8T | ~28.8 TB | 360+ |

### 计算需求也很惊人

训练一个 70B 模型，典型训练量为 2T tokens：

$$
\text{FLOPs} \approx 6 \times \Phi \times T = 6 \times 70 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{23}
$$

单张 A100 的 BF16 算力约为 312 TFLOPS（$3.12 \times 10^{14}$ FLOPS），假设 50% 的 MFU（Model FLOPs Utilization）：

$$
\text{训练时间} = \frac{8.4 \times 10^{23}}{3.12 \times 10^{14} \times 0.5} \approx 5.4 \times 10^{9} \text{ 秒} \approx 171 \text{ 年}
$$

用 **1024 张 A100** 并行：

$$
\text{训练时间} = \frac{171 \text{ 年}}{1024} \approx 61 \text{ 天}
$$

所以分布式训练不仅是"不得不做"，而且规模必须足够大。

---

## 数据并行 (Data Parallelism)

### 基本原理

数据并行是最简单、最直观的并行方式。核心思想：

1. 每张 GPU 持有模型的**完整副本**
2. 将一个大 batch 均分到 $N$ 张 GPU
3. 每张 GPU 独立做**前向传播**和**反向传播**
4. 对所有 GPU 的梯度做 **AllReduce**（求平均）
5. 每张 GPU 用相同的平均梯度更新参数

```
         ┌──────────────┐
         │ Large Batch  │
         │ (Global BS)  │
         └──────┬───────┘
                │ Split
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│ GPU 0  │ │ GPU 1  │ │ GPU 2  │
│  Full  │ │  Full  │ │  Full  │
│ Model  │ │ Model  │ │ Model  │
│batch 0 │ │batch 1 │ │batch 2 │
└──┬─────┘ └──┬─────┘ └──┬─────┘
   │ grad_0   │ grad_1   │ grad_2
   └──────────┼──────────┘
              │ AllReduce (average)
   ┌──────────┼──────────┐
   ▼          ▼          ▼
 avg_grad   avg_grad   avg_grad
   │          │          │
   ▼          ▼          ▼
 Update     Update     Update
 Params     Params     Params
```

**数学等价性**：数据并行在数学上等价于使用更大 batch size 的单卡训练。假设全局 batch size 为 $B$，使用 $N$ 张卡，每张卡的 mini-batch 为 $B/N$：

$$
\bar{g} = \frac{1}{N} \sum_{i=1}^{N} g_i = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{B/N}\sum_{j \in \mathcal{D}_i} \nabla_\theta L(x_j) = \frac{1}{B}\sum_{j=1}^{B} \nabla_\theta L(x_j)
$$

### DDP 实现

PyTorch 的 **DistributedDataParallel (DDP)** 是数据并行的标准实现：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 初始化进程组
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 创建模型并包装为 DDP
model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 使用 DistributedSampler 确保每张卡拿到不同的数据
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=per_gpu_batch_size, sampler=sampler)

# 训练循环（与单卡基本一致）
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 确保每个 epoch 的 shuffle 不同
    for batch in dataloader:
        loss = model(batch)
        loss.backward()       # DDP 自动在 backward 中做 AllReduce
        optimizer.step()
        optimizer.zero_grad()
```

**DDP 的关键优化——Gradient Bucketing**：

DDP 并不是等所有梯度计算完再做一次 AllReduce，而是将梯度分成多个 **bucket**，当一个 bucket 的梯度计算完毕就立即开始通信，实现**计算和通信的重叠**：

```
反向传播时间线：
Layer N  →  Layer N-1  →  ...  →  Layer 1  →  Layer 0
  │           │                     │           │
  └─Bucket 3──┘        └──Bucket 2──┘   └─Bucket 1──┘  └─Bucket 0
      ↓ 立即开始                ↓ 立即开始        ↓
    AllReduce              AllReduce         AllReduce
```

这样通信时间被隐藏在计算时间之下，大幅减少了端到端的训练时间。

### 数据并行的瓶颈

数据并行的最大问题是**显存浪费**：

$$
\text{每张卡的显存} = 16\Phi + \text{激活值}
$$

每张卡都存储了完整的模型参数 ($2\Phi$)、完整的梯度 ($2\Phi$)、完整的优化器状态 ($12\Phi$)。对于 70B 模型，即使有 64 张卡，每张卡仍然需要 1120 GB——远超单卡容量。

**数据并行无法训练超出单卡显存的模型。** 这就是我们需要 ZeRO、张量并行、流水线并行等技术的原因。

---

## DeepSpeed ZeRO

ZeRO（**Z**ero **R**edundancy **O**ptimizer）的核心洞察：在数据并行中，每张 GPU 都存了完整的模型状态（参数、梯度、优化器状态），但每张 GPU 每次只需要其中一部分来计算。能不能**分片存储**？

### ZeRO Stage 1：优化器状态分片

**核心思想**：将优化器状态（Adam 的 m 和 v，以及 FP32 master copy）均匀分到 $N$ 张 GPU 上，每张 GPU 只存 $1/N$。

**显存分析**（以 $N=8$ 为例）：

| 存储项目 | 无 ZeRO | ZeRO-1 |
|---------|---------|--------|
| FP16 参数 | $2\Phi$ | $2\Phi$ |
| FP16 梯度 | $2\Phi$ | $2\Phi$ |
| FP32 master copy | $4\Phi$ | $4\Phi / N$ |
| Adam m | $4\Phi$ | $4\Phi / N$ |
| Adam v | $4\Phi$ | $4\Phi / N$ |
| **总计** | $16\Phi$ | $4\Phi + 12\Phi/N$ |

当 $N = 8$ 时：

$$
\text{ZeRO-1 显存} = 4\Phi + \frac{12\Phi}{8} = 4\Phi + 1.5\Phi = 5.5\Phi
$$

相比原来的 $16\Phi$，节省了约 **3x**。

**工作流程**：

1. 前向传播和反向传播正常进行（每张卡有完整参数和梯度）
2. 反向传播后，每张 GPU 只更新自己负责的那 $1/N$ 参数对应的优化器状态
3. 更新完成后，通过 **AllGather** 收集所有 GPU 更新后的参数

### ZeRO Stage 2：梯度分片

在 Stage 1 的基础上，梯度也做分片。

**核心思想**：每张 GPU 只需要自己负责更新的那 $1/N$ 参数对应的梯度，其他梯度计算完后可以立即释放。

| 存储项目 | ZeRO-1 | ZeRO-2 |
|---------|--------|--------|
| FP16 参数 | $2\Phi$ | $2\Phi$ |
| FP16 梯度 | $2\Phi$ | $2\Phi / N$ |
| 优化器状态 | $12\Phi / N$ | $12\Phi / N$ |
| **总计** | $4\Phi + 12\Phi/N$ | $2\Phi + 14\Phi/N$ |

当 $N = 8$ 时：

$$
\text{ZeRO-2 显存} = 2\Phi + \frac{14\Phi}{8} = 2\Phi + 1.75\Phi = 3.75\Phi
$$

**通信方式变化**：ZeRO-2 将原来的 AllReduce 替换为 **ReduceScatter**。每张 GPU 只接收并保留自己负责的那部分梯度的汇总结果。

### ZeRO Stage 3：参数分片

最激进的方案：**参数、梯度、优化器状态全部分片**。

| 存储项目 | ZeRO-2 | ZeRO-3 |
|---------|--------|--------|
| FP16 参数 | $2\Phi$ | $2\Phi / N$ |
| FP16 梯度 | $2\Phi / N$ | $2\Phi / N$ |
| 优化器状态 | $12\Phi / N$ | $12\Phi / N$ |
| **总计** | $2\Phi + 14\Phi/N$ | $16\Phi / N$ |

当 $N = 8$ 时：

$$
\text{ZeRO-3 显存} = \frac{16\Phi}{8} = 2\Phi
$$

相比原始的 $16\Phi$，节省了 **8x**（等于 GPU 数量）！

**完整的三阶段显存对比**：

| 阶段 | 每卡显存 | N=8 时 (70B 模型) | 节省比例 |
|------|---------|------------------|---------|
| 无 ZeRO | $16\Phi$ | 1120 GB | - |
| ZeRO-1 | $4\Phi + 12\Phi/N$ | 385 GB | 2.9x |
| ZeRO-2 | $2\Phi + 14\Phi/N$ | 262.5 GB | 4.3x |
| ZeRO-3 | $16\Phi/N$ | 140 GB | 8x |

**ZeRO-3 的工作流程**：

```
前向传播某一层时：
1. AllGather：收集该层的完整参数（从所有 GPU 获取各自的分片）
2. 计算前向传播
3. 释放非本 GPU 负责的参数分片（只保留 1/N）

反向传播某一层时：
1. AllGather：再次收集该层的完整参数
2. 计算梯度
3. ReduceScatter：将梯度汇总并分发（每个 GPU 只保留 1/N 的梯度）
4. 释放非本 GPU 负责的参数分片

参数更新：
5. 每个 GPU 用本地的 1/N 梯度更新本地的 1/N 优化器状态和参数
```

**代价**：通信量增加。ZeRO-3 的通信量约为普通数据并行的 **1.5 倍**：

$$
\text{普通 DP 通信量} = 2\Phi \quad (\text{AllReduce})
$$

$$
\text{ZeRO-3 通信量} = 3\Phi \quad (\text{前向 AllGather} + \text{反向 AllGather} + \text{ReduceScatter})
$$

### ZeRO-Offload & ZeRO-Infinity

当 GPU 显存仍然不够时，可以利用 **CPU 内存** 和 **NVMe SSD** 来扩展存储：

| 方案 | 卸载目标 | 卸载内容 | 带宽瓶颈 |
|------|---------|---------|---------|
| ZeRO-Offload | CPU 内存 | 优化器状态 + 部分计算 | PCIe 4.0: ~32 GB/s |
| ZeRO-Infinity | NVMe SSD | 参数 + 梯度 + 优化器状态 | NVMe: ~5-7 GB/s |

**策略**：

- 将**计算密集型**操作（前向/反向传播）保留在 GPU 上
- 将**内存密集型**操作（优化器状态更新）卸载到 CPU
- 使用**预取（prefetch）**来隐藏数据传输延迟

**代价**：虽然可以训练更大的模型，但由于 PCIe/NVMe 带宽远低于 GPU 显存带宽（HBM: ~2 TB/s），训练速度会显著下降。

### DeepSpeed 配置实战

以下是一个用于训练 Llama 70B 的完整 DeepSpeed ZeRO-3 配置：

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "none"
        }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false,
    "steps_per_print": 100
}
```

**关键参数解释**：

| 参数 | 作用 |
|------|------|
| `stage` | ZeRO 阶段：1/2/3 |
| `overlap_comm` | 通信与计算重叠，隐藏通信开销 |
| `contiguous_gradients` | 梯度连续存储，减少内存碎片 |
| `reduce_bucket_size` | 梯度 AllReduce 的 bucket 大小 |
| `stage3_prefetch_bucket_size` | 参数预取的 bucket 大小 |
| `stage3_param_persistence_threshold` | 小于此阈值的参数不做分片（减少通信） |
| `offload_optimizer.device` | 优化器卸载到 cpu 还是 nvme |
| `pin_memory` | 使用 page-locked 内存加速 CPU-GPU 传输 |

---

## 张量并行 (Tensor Parallelism)

张量并行将**单层的计算**拆分到多张 GPU 上。这是 Megatron-LM 提出的核心技术。

### Megatron-LM 的列并行与行并行

考虑一个线性层 $Y = XW$，其中 $X \in \mathbb{R}^{b \times d}$，$W \in \mathbb{R}^{d \times k}$。

#### 列并行 (Column Parallel)

将权重矩阵 $W$ **按列**拆分到 $N$ 张 GPU 上：

$$
W = [W_1 | W_2 | \cdots | W_N], \quad W_i \in \mathbb{R}^{d \times k/N}
$$

每张 GPU 计算：

$$
Y_i = X W_i, \quad Y_i \in \mathbb{R}^{b \times k/N}
$$

最终结果通过拼接得到：

$$
Y = [Y_1 | Y_2 | \cdots | Y_N]
$$

**特点**：

- 输入 $X$ 在所有 GPU 上**相同**（需要广播或复制）
- 输出 $Y_i$ 在每张 GPU 上是**部分结果**
- 最终需要 **AllGather** 拼接完整输出

#### 行并行 (Row Parallel)

将权重矩阵 $W$ **按行**拆分到 $N$ 张 GPU 上：

$$
W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{d/N \times k}
$$

相应地，输入 $X$ 也需要**按列**拆分：

$$
X = [X_1 | X_2 | \cdots | X_N], \quad X_i \in \mathbb{R}^{b \times d/N}
$$

每张 GPU 计算：

$$
Y_i = X_i W_i, \quad Y_i \in \mathbb{R}^{b \times k}
$$

最终结果通过求和得到：

$$
Y = \sum_{i=1}^{N} Y_i = X_1 W_1 + X_2 W_2 + \cdots + X_N W_N
$$

**特点**：

- 输入需要**按列拆分**分发
- 输出 $Y_i$ 形状相同，需要 **AllReduce** 求和

#### 前向和反向的通信算子：$f$ 和 $g$

Megatron-LM 定义了两个关键的通信算子：

- **$f$ 算子**：前向传播中是恒等操作（identity），反向传播中执行 AllReduce
- **$g$ 算子**：前向传播中执行 AllReduce，反向传播中是恒等操作

```
列并行线性层：              行并行线性层：
  X ──f──> X (identity)     X_i ──计算──> Y_i ──g──> Y (AllReduce)
  │                                              │
  ▼                                              ▼
  X @ W_i = Y_i             反向传播时 g 是 identity
  │
  ▼ (反向传播时 f 做 AllReduce)
```

### Self-Attention 的张量并行

Transformer 的自注意力天然适合张量并行——**多头注意力本身就是按头独立计算的**。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

假设有 $h$ 个头，使用 $N$ 张 GPU（$N$ 整除 $h$）：

```
          输入 X
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
  GPU 0   GPU 1   GPU 2     ← 每张 GPU 负责 h/N 个头
  Q_0     Q_1     Q_2       ← Q = XW_Q, W_Q 按列拆分 (每张卡 h/N 列)
  K_0     K_1     K_2       ← K = XW_K, W_K 按列拆分
  V_0     V_1     V_2       ← V = XW_V, W_V 按列拆分
    │       │       │
  Attn_0  Attn_1  Attn_2    ← 各自独立计算 attention
    │       │       │
  O_0     O_1     O_2       ← 各自的输出
    │       │       │
    └───────┼───────┘
            │ AllReduce (通过行并行的输出投影 W_O)
            ▼
         最终输出
```

$W_Q, W_K, W_V$ 使用**列并行**（按头拆分），$W_O$（输出投影）使用**行并行**。这样一个 Attention 层只需要一次 AllReduce。

### MLP 的张量并行

Transformer 的 MLP 通常为：

$$
\text{MLP}(x) = \text{dropout}(\text{GeLU}(xA) \cdot B)
$$

其中 $A \in \mathbb{R}^{d \times 4d}$，$B \in \mathbb{R}^{4d \times d}$。

**策略**：第一层 $A$ 用**列并行**，第二层 $B$ 用**行并行**。

为什么这样搭配？

1. **$A$ 用列并行**：$A = [A_1 | A_2]$，每张 GPU 计算 $\text{GeLU}(xA_i)$，因为 GeLU 是逐元素操作，可以在拆分后的结果上直接做
2. **$B$ 用行并行**：$B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$，每张 GPU 的输入恰好是列并行的输出（已经按列拆分），最后 AllReduce 求和

```
    x ──(f: identity)──> x
    │                    │
    ▼                    ▼
  GPU 0: x @ A_1       GPU 1: x @ A_2       ← 列并行
    │                    │
    ▼                    ▼
  GeLU(xA_1)           GeLU(xA_2)           ← 各自独立做 GeLU
    │                    │
    ▼                    ▼
  GeLU(xA_1) @ B_1     GeLU(xA_2) @ B_2    ← 行并行
    │                    │
    └────────┬───────────┘
             │ AllReduce (g)
             ▼
           输出 = GeLU(xA_1)B_1 + GeLU(xA_2)B_2 = GeLU(xA)B
```

**关键优势**：列并行 + 行并行的组合，使得 MLP 层只需要**一次前向 AllReduce + 一次反向 AllReduce**。

### 张量并行核心代码

以下代码展示了行并行和列并行的完整计算过程：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 准备数据：y = xw
bs = 2
row = dim = 4
col = out_dim = dim * 2  # out_dim = 8

x = torch.arange(bs * dim, dtype=torch.float32).reshape(bs, dim)
w = torch.arange(dim * out_dim, dtype=torch.float32, requires_grad=True).reshape(dim, out_dim)

y_label = torch.randn(bs, out_dim)
```

**手动梯度计算验证**：

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial Y}\frac{\partial Y}{\partial W} = X^T \cdot \Delta Y / \text{out\_dim}
$$

```python
# PyTorch 自动求导
mse_loss = nn.MSELoss(reduction='mean')
y_pred = x @ w
loss_torch = mse_loss(y_pred, y_label)
loss_torch.backward()
print(f"PyTorch 自动求导梯度: {w.grad.shape}")  # [4, 8]
```

**行并行 (Row Parallel) 实现**：

```python
# 权重按行拆分：W[4,8] → W_1[2,8] + W_2[2,8]
row_dim = dim // 2  # = 2
w_row_1 = w_row[:row_dim, :]   # GPU 0 上的分片，shape: [2, 8]
w_row_2 = w_row[row_dim:, :]   # GPU 1 上的分片，shape: [2, 8]

# 输入也需要按列拆分
x_col_1 = x[:, :row_dim]       # GPU 0 的输入，shape: [2, 2]
x_col_2 = x[:, row_dim:]       # GPU 1 的输入，shape: [2, 2]

# 各 GPU 独立计算
y_1 = x_col_1 @ w_row_1        # GPU 0: [2,2] @ [2,8] = [2,8]
y_2 = x_col_2 @ w_row_2        # GPU 1: [2,2] @ [2,8] = [2,8]

# AllReduce 求和得到最终结果
y = y_1 + y_2                  # shape: [2, 8]，等价于 x @ w

# 反向传播：各 GPU 独立计算各自分片的梯度
delta_y = (y_1 + y_2 - y_label)
grad_row_1 = x_col_1.t() @ delta_y   # GPU 0 的梯度，shape: [2, 8]
grad_row_2 = x_col_2.t() @ delta_y   # GPU 1 的梯度，shape: [2, 8]

# 拼接得到完整梯度
grad_row = torch.cat((grad_row_1, grad_row_2), dim=0) / out_dim
# shape: [4, 8]，与 PyTorch 自动求导结果一致
```

**列并行 (Column Parallel) 实现**：

```python
# 权重按列拆分：W[4,8] → W_1[4,4] + W_2[4,4]
col_dim = out_dim // 2  # = 4
w_col_1 = w_col[:, :col_dim]   # GPU 0 上的分片，shape: [4, 4]
w_col_2 = w_col[:, col_dim:]   # GPU 1 上的分片，shape: [4, 4]

# 输入相同（广播/复制到每张 GPU）
y_1 = x @ w_col_1              # GPU 0: [2,4] @ [4,4] = [2,4]
y_2 = x @ w_col_2              # GPU 1: [2,4] @ [4,4] = [2,4]

# 拼接得到完整输出 (AllGather)
y = torch.cat((y_1, y_2), dim=1)  # [2, 8]

# 反向传播：各 GPU 用各自的 delta_y 分片独立计算
y_1_delta = y_1 - y_label[:, :col_dim]   # GPU 0 的误差
y_2_delta = y_2 - y_label[:, col_dim:]   # GPU 1 的误差

grad_col_1 = x.t() @ y_1_delta           # GPU 0 的梯度，shape: [4, 4]
grad_col_2 = x.t() @ y_2_delta           # GPU 1 的梯度，shape: [4, 4]

# 拼接得到完整梯度
grad_col = torch.cat((grad_col_1, grad_col_2), dim=1) / out_dim
# shape: [4, 8]
```

**关键观察**：

- 行并行中，前向传播需要 **AllReduce**（求和），梯度各自独立计算
- 列并行中，前向传播需要 **AllGather**（拼接），梯度各自独立计算
- 两者在反向传播中梯度计算都是**各 GPU 独立完成**的，只需要知道误差 $\Delta Y$

---

## 流水线并行 (Pipeline Parallelism)

流水线并行将模型的**不同层**放到不同 GPU 上。

### 朴素流水线的气泡问题

最简单的做法：将模型的 $L$ 层均匀分到 $N$ 张 GPU 上。

```
GPU 0: Layer 0~7     (Llama 70B 有 80 层，4 GPU 则每个 GPU 20 层)
GPU 1: Layer 8~15
GPU 2: Layer 16~23
GPU 3: Layer 24~31
```

**问题：严重的 GPU 空闲**

```
时间 →  ─────────────────────────────────────────────
GPU 0: [  Forward  ][ idle ][ idle ][ idle ][ Backward ]
GPU 1: [   idle    ][ Fwd  ][ idle ][ idle ][  idle   ][ Bwd ]
GPU 2: [   idle    ][ idle ][ Fwd  ][ idle ][  idle   ][ idle ][ Bwd ]
GPU 3: [   idle    ][ idle ][ idle ][ Fwd  ][  idle   ][ idle ][ idle ][ Bwd ]
```

**气泡率（Bubble Ratio）计算**：

假设每个 GPU 上前向传播时间为 $t_f$，反向传播时间为 $t_b \approx 2t_f$，使用 $p$ 个 GPU：

$$
\text{总时间} = (p-1) \cdot t_f + t_f + (p-1) \cdot t_b + t_b = p \cdot (t_f + t_b) + (p-1)(t_f + t_b)
$$

$$
\text{理想时间（无气泡）} = t_f + t_b
$$

$$
\text{气泡率} = \frac{(p-1)(t_f + t_b)}{p(t_f + t_b)} = \frac{p-1}{p}
$$

当 $p = 4$ 时，气泡率 = **75%**！也就是说，75% 的时间 GPU 在空闲。

### GPipe：微批次流水线

**核心思想**：将一个 mini-batch 拆分为 $m$ 个 **micro-batch**，让多个 micro-batch 在流水线中同时执行。

```
时间 →  ─────────────────────────────────────────────
GPU 0: [F1][F2][F3][F4][          ][B4][B3][B2][B1]
GPU 1: [  ][F1][F2][F3][F4][      ][B4][B3][B2][B1]
GPU 2: [  ][  ][F1][F2][F3][F4][  ][B4][B3][B2][B1]
GPU 3: [  ][  ][  ][F1][F2][F3][F4][B4][B3][B2][B1]
                                 ↑
                          这里有一个同步点：
                     等所有前向完成才开始反向
```

GPipe 的气泡率：

$$
\text{气泡率} = \frac{(p-1)}{m + p - 1} \cdot \frac{t_f + t_b}{t_f + t_b} = \frac{p-1}{m + p - 1}
$$

当 $p = 4, m = 16$ 时，气泡率 = $\frac{3}{19}$ = **15.8%**，比朴素流水线的 75% 好了很多。

**缺点**：需要同时存储所有 micro-batch 的激活值，显存开销大。

### 1F1B 调度策略

**1F1B（One Forward One Backward）** 交错执行前向和反向传播，在稳态阶段每做一个前向就做一个反向：

```
时间 →  ──────────────────────────────────────────────────────────
GPU 0: [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][  ][B5][B6][B7]
GPU 1: [  ][F1][F2][F3][  ][B1][F4][B2][F5][B3][F6][B4][B5][B6][B7]
GPU 2: [  ][  ][F1][F2][  ][  ][B1][F3][B2][F4][B3][F5][B4][B5][B6]
GPU 3: [  ][  ][  ][F1][  ][  ][  ][B1][F2][B2][F3][B3][F4][B4][B5]

       ├─ warmup ─┤├───── 稳态 (1F1B) ─────┤├── cooldown ──┤
```

**1F1B 的优势**：

- 稳态阶段每个 GPU 最多同时保留 $p$ 个 micro-batch 的激活值（而非 GPipe 的 $m$ 个）
- **显存峰值大幅降低**

气泡率与 GPipe 相同：$\frac{p-1}{m+p-1}$，但显存更优。

### Interleaved 1F1B

**核心思想**：每张 GPU 不是放连续的层，而是放**多组非连续的层**。

例如，4 张 GPU、16 层，普通分法 vs Interleaved：

```
普通：
GPU 0: Layer 0,1,2,3
GPU 1: Layer 4,5,6,7
GPU 2: Layer 8,9,10,11
GPU 3: Layer 12,13,14,15

Interleaved (v=2 virtual stages per GPU):
GPU 0: Layer 0,1  +  Layer 8,9
GPU 1: Layer 2,3  +  Layer 10,11
GPU 2: Layer 4,5  +  Layer 12,13
GPU 3: Layer 6,7  +  Layer 14,15
```

每张 GPU 上有 $v$ 个"虚拟 stage"。气泡率降低为：

$$
\text{气泡率} = \frac{p-1}{v \cdot m + p - 1}
$$

当 $v = 2$ 时，气泡率减半。代价是通信量增加（因为非连续层之间需要更多通信）。

---

## 3D 并行

### DP + TP + PP 如何组合

实际训练超大模型时，需要同时使用三种并行：

```
┌───────────────────────────────────────────────────────────┐
│                      3D Parallelism                       │
│                                                           │
│  Data Parallel (DP=8): 8 full model replicas              │
│  +-- Pipeline Parallel (PP=4): each split into 4 stages   │
│  |   +-- Tensor Parallel (TP=2): 2 GPUs per stage        │
│  |   |   +-- GPU 0                                       │
│  |   |   +-- GPU 1                                       │
│  |   +-- Tensor Parallel (TP=2)                          │
│  |   |   +-- GPU 2                                       │
│  |   |   +-- GPU 3                                       │
│  |   +-- Tensor Parallel (TP=2)                          │
│  |   |   +-- GPU 4                                       │
│  |   |   +-- GPU 5                                       │
│  |   +-- Tensor Parallel (TP=2)                          │
│  |       +-- GPU 6                                       │
│  |       +-- GPU 7                                       │
│  +-- ... (7 more identical PP groups)                     │
│                                                           │
│  Total GPUs = DP x PP x TP = 8 x 4 x 2 = 64             │
└───────────────────────────────────────────────────────────┘
```

**Llama 70B 在 64 卡上的实际配置示例**：

| 参数 | 值 | 理由 |
|-----|-----|------|
| TP (张量并行) | 8 | 70B 单层参数大，需要更多卡做张量拆分 |
| PP (流水线并行) | 4 | 80 层分成 4 段，每段 20 层 |
| DP (数据并行) | 2 | 剩余维度用于数据并行加速 |
| 总 GPU 数 | 8 × 4 × 2 = 64 | |

或者使用 ZeRO 替代数据并行：

| 参数 | 值 | 理由 |
|-----|-----|------|
| TP | 8 | 节点内 8 卡 NVLink |
| PP | 2 | 跨 2 个节点 |
| ZeRO-1 | 4 组 | 替代纯 DP，进一步节省显存 |
| 总 GPU 数 | 8 × 2 × 4 = 64 | |

### 通信拓扑设计原则

```
┌─────────────── 节点 0 ────────────────┐   ┌─── 节点 1 ───┐
│ GPU0 ←NVLink→ GPU1 ←NVLink→ ... GPU7  │ ← IB/RoCE → │ GPU0 ...    │
│ ├───── TP（张量并行）组 ──────┤         │              │             │
│         高带宽、低延迟                  │              │             │
└────────────────────────────────────────┘   └─────────────┘
                    │                                │
                    └──────── PP（流水线并行）──────────┘
                              通信量小，可跨节点

          DP（数据并行）/ ZeRO：全局范围
```

**黄金法则**：

1. **TP 放在节点内**：张量并行通信频繁且对延迟敏感，必须利用 NVLink 的高带宽（600+ GB/s）
2. **PP 可以跨节点**：流水线并行通信量小（只传激活值和梯度），对延迟容忍度高
3. **DP / ZeRO 全局**：梯度同步可以通过 gradient accumulation 降低通信频率

### Context Parallelism / Ring Attention

当序列长度极长（如 128K、1M token）时，即使模型参数能放下，**单条序列的激活值**也可能超出单 GPU 显存。此时需要将序列本身拆分到多个 GPU 上——这就是 **Context Parallelism（CP）**。

#### 长序列并行的需求

以 Llama 3 128K 为例，单条序列的注意力矩阵大小为 $128K \times 128K = 16G$ 个元素。即使用 BF16，单个注意力头就需要 $16G \times 2 = 32$ GB 显存。多头（64 头）更是完全不可能放在单卡上。

传统的张量并行（TP）拆分的是**头维度**，无法解决单个头内序列过长的问题。Context Parallelism 拆分的是**序列维度**。

#### Ring Attention 原理

Ring Attention 是 Context Parallelism 最主流的实现方式，核心思想是：

1. 将序列均匀分成 $P$ 段，分配到 $P$ 个 GPU
2. 每个 GPU 持有本地的 Q 块（固定不动）
3. K、V 块在 GPU 之间**环形传递**
4. 每一步，各 GPU 用本地 Q 与当前收到的 KV 块计算部分注意力
5. 利用 Online Softmax 在线合并各步结果

```
Ring Attention 执行流程（4 GPU，序列分 4 段）：

Step 0:                         Step 1:
GPU 0: Q0 × K0,V0  ─K0,V0→     GPU 0: Q0 × K3,V3  ─K3,V3→
GPU 1: Q1 × K1,V1  ─K1,V1→     GPU 1: Q1 × K0,V0  ─K0,V0→
GPU 2: Q2 × K2,V2  ─K2,V2→     GPU 2: Q2 × K1,V1  ─K1,V1→
GPU 3: Q3 × K3,V3  ─K3,V3→     GPU 3: Q3 × K2,V2  ─K2,V2→
       ↑___________环形传递_↓           ↑___________环形传递_↓

Step 2:                         Step 3:
GPU 0: Q0 × K2,V2              GPU 0: Q0 × K1,V1
...（继续环形传递）              ...（所有 KV 块都被每个 GPU 看到一次）

每步完成后用 Online Softmax 合并: O_new = rescale(O_old) + P_block @ V_block
```

#### 通信与计算重叠

Ring Attention 的精妙之处在于**通信可以完全被计算掩盖**：

- 当 GPU $i$ 正在用 $Q_i$ 和 $K_j, V_j$ 计算注意力时
- 同时将 $K_j, V_j$ 发送给 GPU $(i+1) \% P$
- 并从 GPU $(i-1) \% P$ 接收下一个 KV 块

只要单步计算时间 > 单块 KV 的传输时间，通信就完全被隐藏。

```python
# Ring Attention 伪代码
def ring_attention(Q_local, K_local, V_local, ring_group):
    """
    Q_local: 本 GPU 的 Q 块 (seq_len/P, d)
    K_local, V_local: 本 GPU 的初始 KV 块
    ring_group: 通信组
    """
    P = ring_group.size()
    O = torch.zeros_like(Q_local)
    l = torch.zeros(Q_local.shape[0], 1)   # softmax 分母
    m = torch.full((Q_local.shape[0], 1), float('-inf'))  # max

    K_recv, V_recv = K_local, V_local

    for step in range(P):
        # 异步发送当前 KV 给下一个 GPU，同时接收上一个 GPU 的 KV
        if step < P - 1:
            send_op = ring_send_async(K_recv, V_recv, ring_group)
            K_next, V_next = ring_recv_async(ring_group)

        # 计算本步注意力（与通信重叠）
        S = Q_local @ K_recv.T / math.sqrt(d)
        m_new = torch.maximum(m, S.max(dim=-1, keepdim=True).values)
        P_block = torch.exp(S - m_new)
        l_new = torch.exp(m - m_new) * l + P_block.sum(dim=-1, keepdim=True)
        O = torch.exp(m - m_new) * O + P_block @ V_recv
        l, m = l_new, m_new

        if step < P - 1:
            send_op.wait()
            K_recv, V_recv = K_next, V_next

    return O / l
```

#### 适用场景和限制

**适用场景**：
- 超长序列训练（128K+），如文档级理解、视频处理
- 与 TP/PP/DP 正交，可自由组合为 4D/5D 并行

**限制**：
- 需要 $P-1$ 步环形通信，通信轮次多
- Causal mask 下，部分 GPU 的 KV 块与 Q 块无交互（被 mask 掉），导致负载不均衡
- 通常 CP 的 GPU 数量 $\leq 8$，超大 CP 组效率下降

### Expert Parallelism

MoE（Mixture of Experts）模型引入了**专家并行**这一独特的并行维度。当模型有数百甚至上千个专家时（如 DeepSeek-V3 有 256 个 routed experts），单 GPU 无法容纳所有专家参数。

#### MoE 并行的核心挑战

MoE 的前向传播中，Router 为每个 token 选择 top-k 个专家。这意味着：
- 不同 token 被路由到不同 GPU 上的不同专家
- 需要**All-to-All 通信**：每个 GPU 将 token 发送到其被路由的专家所在的 GPU

```
Expert Parallelism 通信模式（4 GPU，8 个专家，每 GPU 2 个专家）：

Router 输出:
  Token A → Expert 0 (GPU 0), Expert 5 (GPU 2)
  Token B → Expert 3 (GPU 1), Expert 7 (GPU 3)
  Token C → Expert 1 (GPU 0), Expert 4 (GPU 2)

All-to-All 通信（dispatch）:
  GPU 0 → GPU 0: Token A, Token C（去 Expert 0, 1）
  GPU 0 → GPU 1: Token B（去 Expert 3）
  GPU 0 → GPU 2: Token A, Token C（去 Expert 5, 4）
  GPU 0 → GPU 3: Token B（去 Expert 7）

各 GPU 计算本地专家:
  GPU 0: Expert 0(Token A), Expert 1(Token C)
  GPU 1: Expert 2(-), Expert 3(Token B)
  GPU 2: Expert 4(Token C), Expert 5(Token A)
  GPU 3: Expert 6(-), Expert 7(Token B)

All-to-All 通信（combine）: 将结果发回各 token 的来源 GPU
```

#### All-to-All 通信模式

All-to-All 是 Expert Parallelism 的核心通信原语。与 AllReduce 不同，All-to-All 是**非对称的**——每个 GPU 向每个其他 GPU 发送不同量的数据。

```python
import torch.distributed as dist

def expert_parallel_forward(hidden_states, router_logits, experts, ep_group):
    """
    Expert Parallelism 前向传播
    hidden_states: (batch * seq_len, d)
    router_logits: (batch * seq_len, num_experts)
    """
    # Step 1: Router 决策
    topk_weights, topk_indices = router_logits.topk(k=2, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # Step 2: 按目标专家所在 GPU 分组 token
    tokens_per_expert = count_tokens_per_expert(topk_indices, num_experts)

    # Step 3: All-to-All dispatch（将 token 发送到专家所在的 GPU）
    dispatched = all_to_all(hidden_states, tokens_per_expert, ep_group)

    # Step 4: 本地专家计算
    expert_outputs = []
    for i, expert in enumerate(local_experts):
        mask = (local_expert_indices == i)
        if mask.any():
            expert_outputs.append(expert(dispatched[mask]))

    # Step 5: All-to-All combine（将结果发回来源 GPU）
    combined = all_to_all_reverse(expert_outputs, tokens_per_expert, ep_group)

    # Step 6: 加权合并
    output = topk_weights.unsqueeze(-1) * combined
    return output.sum(dim=1)  # 对 top-k 个专家的结果加权求和
```

#### 专家放置策略（Expert Placement）

专家如何分配到 GPU 上，直接影响通信量和负载均衡：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **均匀分配** | 每个 GPU 放相同数量的专家 | 简单 | 可能通信不均衡 |
| **热度感知** | 高频使用的专家复制到多个 GPU | 减少通信热点 | 增加参数冗余 |
| **亲和性放置** | 经常被同时激活的专家放同一 GPU | 减少 All-to-All 通信 | 需要预分析路由模式 |

DeepSeek-V3 的做法：
- 256 个 routed experts + 1 个 shared expert
- Shared expert 在所有 GPU 上复制（不需要通信）
- Routed experts 均匀分配，配合**辅助 loss** 鼓励负载均衡

#### 与 Data Parallel / Tensor Parallel 的结合

Expert Parallelism (EP) 可以与 DP、TP 自由组合：

```
EP + DP 组合（最常见）：
  - 非专家层（Attention、Shared FFN）：使用数据并行
  - 专家层：使用 Expert Parallelism
  - 例如 8 GPU：EP=8（8 个专家组），非 MoE 层 DP=8

EP + TP + DP 组合：
  - 节点内：TP=2 (Attention 张量并行) + EP=4 (4 个专家组)
  - 节点间：DP=N
```

### 3D/4D/5D 并行实践

随着模型规模和序列长度的增长，并行策略从 3D（DP + TP + PP）扩展到了 4D 甚至 5D。

#### 并行维度一览

| 维度 | 拆分对象 | 通信原语 | 典型放置 |
|------|---------|---------|---------|
| **DP** | batch | AllReduce / ReduceScatter | 全局 |
| **TP** | 层内权重（头维度） | AllReduce / AllGather | 节点内 |
| **PP** | 层间（不同层） | P2P Send/Recv | 跨节点 |
| **CP** | 序列维度 | Ring Send/Recv | 节点内/跨节点 |
| **EP** | 专家 | All-to-All | 节点内/跨节点 |

$$\text{总 GPU 数} = \text{DP} \times \text{TP} \times \text{PP} \times \text{CP} \times \text{EP}$$

#### Megatron-LM 的并行策略组合

Megatron-LM 是 NVIDIA 开发的大规模训练框架，支持 5D 并行：

```bash
# Megatron-LM 5D 并行配置示例
python pretrain_gpt.py \
    --tensor-model-parallel-size 4 \        # TP=4
    --pipeline-model-parallel-size 4 \       # PP=4
    --context-parallel-size 2 \              # CP=2
    --expert-model-parallel-size 8 \         # EP=8（MoE 模型）
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 131072 \
    --micro-batch-size 1 \
    --global-batch-size 2048
    # DP 自动计算: total_gpus / (TP * PP * CP) = 2048 / (4*4*2) = 64
```

**通信拓扑最佳实践**：

```
单节点（8 GPU，NVLink 互连）:
  ├── TP=4: GPU 0-3 为一组，GPU 4-7 为一组
  └── CP=2: 每个 TP 组内再分 2 个 CP 组

跨节点:
  ├── PP=4: 4 个节点串成流水线
  └── DP=N: 所有流水线副本做数据并行
```

#### DeepSeek-V3 的训练并行策略

DeepSeek-V3 是一个 671B 参数的 MoE 模型（37B 激活参数），其训练并行策略极具参考价值：

| 并行维度 | 配置 | 说明 |
|---------|------|------|
| **EP** | 64 | 256 个 routed experts 分到 64 个 GPU |
| **TP** | 1 | 由于使用了 MLA（低秩注意力），单头计算量小，不需要 TP |
| **PP** | 16 | 61 层 Transformer 分成 16 段 |
| **DP** | 按需 | ZeRO-1 优化器分片 |
| **总 GPU** | 2048 H800 | 约 2.79M GPU-hours |

**关键设计决策**：

1. **不用 TP**：DeepSeek-V3 的 MLA 机制将 KV 维度压缩到很小，单层计算量可以放在单 GPU 上，省去了 TP 的高频通信
2. **大 EP 组**：256 个专家需要足够多的 GPU 来分散，EP=64 意味着每个 GPU 放 4 个专家
3. **PP=16**：61 层较深的模型，流水线并行分 16 段，每段约 4 层
4. **FP8 训练**：首次在如此大规模上成功使用 FP8 混合精度，进一步提升吞吐

#### DualPipe：双向微批次调度与通信完全隐藏

> 论文：DeepSeek-V3 Technical Report，arXiv: 2412.19437，Section 3.2.1（page 12-13）

DeepSeek-V3 把跨节点专家并行的 1:1 计算-通信比作为"原罪"——常规 PP 调度下 All-to-All 会主导 step time。**DualPipe** 的关键思路是把每个 chunk 拆成 4 段：`attention` / `all-to-all dispatch` / `MLP` / `all-to-all combine`，反向再借鉴 ZeroBubble 把 `backward` 拆成 `backward for input` 与 `backward for weights`，然后**手工调度**让前向算子与反向通信互相覆盖（论文 Figure 4），最终实现 "both all-to-all and PP communication can be fully hidden"。

**Bubble 公式对比（论文 Table 2，page 13）**：

| Method | Bubble | Parameter | Activation |
|--------|--------|-----------|------------|
| 1F1B | $(PP-1)(F+B)$ | $1\times$ | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | $1\times$ | $PP$ |
| **DualPipe (Ours)** | $(\frac{PP}{2}-1)(F\&B + B - 3W)$ | $2\times$ | $PP+1$ |

其中 $F$ = forward chunk 时间，$B$ = full backward chunk，$W$ = "backward for weights" chunk，$F\&B$ = 一对完全重叠的 forward/backward chunk 的时间。代价是参数要存两份（双向馈送）、激活峰值多 $\frac{1}{PP}$ 倍——论文论证：因为 EP 已经很大、激活只占小头，这点开销可以忽略。

**双向调度示意**（论文 Figure 5 简化）：

```
Device 0: F0 F1 F2 ... ──→  ←── B0 B1 B2 ...
Device 1:    F0 F1 ...      ...    B0 B1
   ...
Device 7:        F0 ...           ...   B0
                ↑                    ↑
         前向从一端进入           反向从另一端进入
         同一时刻两端 micro-batch 各占一半 device，重叠互补
```

DualPipe 仅要求 PP stage 数和 micro-batch 数能被 2 整除，不要求 micro-batch 必须是 PP 的倍数；并且 bubble 与 activation 不随 micro-batch 数增长（与 1F1B 保持一致的好性质）。

#### Cross-Node All-to-All 通信内核

> 论文 Section 3.2.2（page 13-14）

DeepSeek-V3 的集群结构：节点内 NVLink 160 GB/s、节点间 IB 50 GB/s（NVLink ≈ 3.2× IB）。为了让 All-to-All 不成为瓶颈，作者**手写 PTX 内核**，做了三件事：

1. **拓扑感知路由**：限制每个 token 最多 dispatch 到 **4 个节点**，先经 IB 把 token 发送到目标节点上"同 in-node index"的 GPU，再由 NVLink 转发到具体专家所在 GPU。这样 IB 流量被显著降低，IB 与 NVLink 真正流水起来，"each token can efficiently select an average of 3.2 experts per node without incurring additional overhead from NVLink"。
2. **Warp specialization**：dispatch 阶段把 **20 个 SM 划成 10 个 channel**，每个 channel 用不同 warp 分别负责 (1) IB-send、(2) IB-to-NVLink forwarding、(3) NVLink receiving；combine 阶段对应做 NVLink-send / NVLink-to-IB / IB-receive。warp 数按实际负载动态调整。
3. **定制 PTX + auto-tune chunk size**：减少 L2 cache 占用与对其他 SM 计算 kernel 的干扰。

**这是"算法–框架–硬件协同设计"的典型案例**：路由策略的 top-K 选择、kernel 的 SM 划分、网络的 IB/NVLink 带宽比，三者共同决定 4 节点上限和 20 SM 这两个魔数。

#### FP8 混合精度训练

> 论文 Section 3.3（page 14-18）；这是首个在 600B+ MoE 上验证 FP8 的工作，BF16 baseline 的相对 loss 误差 < 0.25%（page 14 末段）。

V3 把绝大多数 GEMM（`Fprop` / `Dgrad` / `Wgrad`，论文 Figure 6）都跑在 FP8 上，相比 BF16 理论提速 2×；同时对 embedding、output head、MoE gating、normalization、attention 这些"对低精度敏感 / 计算量小"的算子保持 BF16/FP32（page 15）。难点不是"用 FP8"，而是**怎么让 FP8 数值稳定**——下面四个机制都要从 PDF 抄。

**1）Fine-Grained 量化（page 16, Figure 7a）**：

| 张量 | 量化粒度 | 含义 |
|------|----------|------|
| Activation | **1 × $N_C$ tile**（per-token, 每 128 channel 一组） | 每 token 的 128 个 channel 共享一个 scale |
| Weight | **$N_C$ × $N_C$ block**（128 × 128，per-input × per-output channel） | 128×128 权重块共享一个 scale |
| NVIDIA 标准 | per-tensor | 整张量一个 scale，对 outlier 极敏感 |

更细的粒度让 scale 跟着局部分布走，outlier 只污染一个 tile/block 而不是整张量。论文指出这与 NVIDIA 下一代 Blackwell 的 microscaling 方向一致（page 17）。

**2）CUDA Core 上的高精度累加（page 17, Figure 7b）**：

H800 的 Tensor Core FP8 GEMM 内部累加只保留约 14 bit 精度（远低于 FP32），$K=4096$ 时相对误差能到 ~2%。V3 的修复：MMA 在 Tensor Core 上以低精度累加，**每 $N_C = 128$ 个元素**（≈ 4 个 WGMMA）就把部分和 promote 到 CUDA Core 的 FP32 寄存器再累加一次。$N_C=128$ 是论文实测的最小值——再小会显著降低 WGMMA issue rate。利用 H800 上"两个 WGMMA 并发"的特性，promotion 与新 MMA 可以重叠，Tensor Core 利用率几乎不掉。

**3）Mantissa over Exponents（page 17）**：NVIDIA H100 推荐做法是 `Fprop` 用 E4M3、`Dgrad/Wgrad` 用 E5M2（更宽指数）来扛梯度的大动态范围。V3 反其道而行——**所有张量统一用 E4M3**，理由是 fine-grained 量化已经让每个 tile/block 共享指数位，等效"动态范围"够用，全 E4M3 反而尾数多 1 bit，精度更高。

**4）低精度存储与通信（page 18）**：

- **优化器状态**：AdamW 的一阶/二阶矩用 **BF16**（master weight 与 gradient 仍 FP32）。
- **激活缓存**：`Wgrad` 路径上的激活直接以 FP8 缓存；attention 后输入用 **E5M6**（自定义 5 指数 6 尾数）以保留更高精度；scale factor 强制取 2 的整数幂，避免反量化引入额外误差。
- **MoE 通信**：MoE up-projection 前的激活 dispatch 用 FP8；combine 路径回 BF16，保住关键精度。

**与既有 FP8 框架的关键差异**：NVIDIA Transformer Engine 默认 per-tensor + delayed quantization + Tensor-Core-only 累加，V3 全换成 fine-grained + online quantization + CUDA-Core promotion——这三处改动叠加才把 FP8 训练误差压到 0.25% 以内。

#### Sequence-Wise Auxiliary Loss（aux-loss-free 的"防极端"补丁）

> 论文 Section 2.1.2 "Complementary Sequence-Wise Auxiliary Loss"（page 9, 公式 17-20）

V3 的负载均衡主力是 **auxiliary-loss-free** 策略：给每个专家一个偏置 $b_i$ 加到亲和分数上做 top-K 路由，每步根据该专家在 batch 上是否过载，按速率 $\gamma$ 升/降 $b_i$；这样既能均衡又不像传统 aux loss 那样把梯度"污染"进模型。但 batch 级均衡不阻止"单条 sequence 内 token 全挤到几个专家"，所以再补一条**序列级**的小权重均衡损失：

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i P_i, \quad
f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}\!\left(s_{i,t} \in \text{Topk}\right), \quad
P_i = \frac{1}{T} \sum_{t=1}^{T} s'_{i,t}
$$

其中 $T$ 是序列长度，$s'_{i,t}$ 是归一化后的亲和分数，$\alpha$ "an extremely small value"。这个损失只在序列内统计——它不和 aux-loss-free 抢"全局均衡"的活，只防一种极端：长序列连续若干 token 路由到同一专家导致 expert parallelism 短板效应。两者是 batch 级偏置 + 序列级软约束 的互补关系。

#### 实际训练集群配置示例

**场景 A：训练 Llama 3 70B（Dense 模型，128K 上下文）**

```
集群: 512 × H100 (64 节点，每节点 8 卡)
配置:
  TP = 8  (节点内 NVLink)
  CP = 4  (4 个节点的 GPU 组成一个 CP ring)
  PP = 4  (4 段流水线)
  DP = 4  (4 个数据并行副本)
  总: 8 × 4 × 4 × 4 = 512 GPU

通信带宽需求:
  TP: ~600 GB/s (NVLink)
  CP: ~50 GB/s (IB, ring 通信可与计算重叠)
  PP: ~10 GB/s (IB, 只传激活值)
  DP: ~10 GB/s (IB, gradient accumulation 降低频率)
```

**场景 B：训练 MoE 模型（256 experts，1T 激活参数）**

```
集群: 2048 × H800
配置:
  EP = 64  (每 GPU 4 个专家)
  PP = 8   (8 段流水线)
  DP = 4   (4 个数据并行副本)
  总: 64 × 8 × 4 = 2048 GPU

关键瓶颈: All-to-All 通信
  每个 token 需发送到 top-k 个专家所在的 GPU
  通信量与 batch_size × seq_len × hidden_dim × top_k 成正比
  解决: 使用 hierarchical All-to-All（节点内 NVLink + 节点间 IB）
```

---

## FSDP (Fully Sharded Data Parallel)

### FSDP 简介

FSDP 是 PyTorch **原生的** ZeRO-3 实现，从 PyTorch 1.11 开始提供，与 PyTorch 生态深度集成。

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

# 定义混合精度策略
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# 包装模型
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价于 ZeRO-3
    mixed_precision=mp_policy,
    auto_wrap_policy=size_based_auto_wrap_policy,    # 自动按大小拆分
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,                          # 限制预取量，控制显存
)

# 训练循环
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### FSDP vs DeepSpeed ZeRO Stage 3

| 对比维度 | FSDP | DeepSpeed ZeRO-3 |
|---------|------|-------------------|
| **集成度** | PyTorch 原生，无需额外库 | 独立库，需要 `pip install deepspeed` |
| **配置方式** | Python API，代码内配置 | JSON 配置文件 + 少量代码 |
| **sharding 单元** | 按 `FlatParameter` 分片 | 按参数分片 |
| **CPU Offload** | 支持 | 支持，且支持 NVMe |
| **流水线并行** | 不直接支持 | 原生支持 |
| **张量并行** | 需配合 `DTensor` | 需配合 Megatron-LM |
| **调试友好** | 更好，PyTorch 原生 stack trace | 自定义引擎，调试较复杂 |
| **HuggingFace** | Trainer + Accelerate 支持 | Trainer + Accelerate 支持 |
| **大规模验证** | Meta 内部大规模使用 | 微软、多家公司大规模使用 |
| **Activation Checkpointing** | `checkpoint_wrapper` | `deepspeed.checkpointing` |

**选择建议**：

- 如果你主要用 PyTorch 生态（HuggingFace 等），优先考虑 **FSDP**
- 如果需要 PP + TP + ZeRO 的完整 3D 并行，考虑 **DeepSpeed + Megatron**
- 如果显存极度紧张需要 NVMe offload，选择 **DeepSpeed ZeRO-Infinity**

---

## 分布式训练中的通信原语

### AllReduce, AllGather, ReduceScatter

这三个是分布式训练中最核心的集合通信操作：

#### AllReduce

**作用**：所有 GPU 上的张量求和（或平均），结果广播到所有 GPU。

```
  Before:                 After (AllReduce SUM):
  GPU 0: [1, 2]          GPU 0: [6, 8]
  GPU 1: [2, 3]    →     GPU 1: [6, 8]
  GPU 2: [3, 3]          GPU 2: [6, 8]
```

**用途**：数据并行中的梯度同步

**通信量**：$2(N-1)/N \cdot D \approx 2D$（$D$ 为数据大小，$N$ 为 GPU 数）

#### AllGather

**作用**：收集所有 GPU 上的张量碎片，拼接成完整张量，广播到所有 GPU。

```
  Before:                 After (AllGather):
  GPU 0: [A]              GPU 0: [A, B, C]
  GPU 1: [B]        →     GPU 1: [A, B, C]
  GPU 2: [C]              GPU 2: [A, B, C]
```

**用途**：ZeRO-3 前向传播时收集完整参数

**通信量**：$(N-1)/N \cdot D \approx D$

#### ReduceScatter

**作用**：先 Reduce（求和），再 Scatter（分发），每个 GPU 只得到结果的一部分。

```
  Before:                 After (ReduceScatter SUM):
  GPU 0: [1, 2, 3]       GPU 0: [6]       (= 1+2+3)
  GPU 1: [2, 3, 4]  →    GPU 1: [8]       (= 2+3+3)
  GPU 2: [3, 3, 3]       GPU 2: [10]      (= 3+4+3)
```

**用途**：ZeRO-2/3 反向传播时的梯度汇总分发

**通信量**：$(N-1)/N \cdot D \approx D$

**关键关系**：

$$
\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}
$$

### Ring AllReduce 算法详解

Ring AllReduce 是目前最常用的 AllReduce 实现，由百度在 2017 年推广。

**核心思想**：将 $N$ 个 GPU 排成一个环形，分两个阶段完成：

**阶段 1：Reduce-Scatter 阶段**

将每个 GPU 的数据分成 $N$ 份，沿环形依次传递并累加：

```
初始状态（4 GPU，每个有 4 个数据块）：
GPU 0: [a0, a1, a2, a3]
GPU 1: [b0, b1, b2, b3]
GPU 2: [c0, c1, c2, c3]
GPU 3: [d0, d1, d2, d3]

Step 1: 每个 GPU 发送一块给下一个 GPU，并接收上一个 GPU 的数据（累加）
GPU 0: [a0,     a1,     a2,     a3+d3 ]
GPU 1: [b0+a0,  b1,     b2,     b3    ]
GPU 2: [c0,     c1+b1,  c2,     c3    ]
GPU 3: [d0,     d1,     d2+c2,  d3    ]

Step 2: 继续...
Step 3: 继续...

N-1 步后，每个 GPU 上有一个完整的 reduce 结果块。
```

**阶段 2：AllGather 阶段**

再经过 $N-1$ 步环形传递，将每个 GPU 上的完整块广播到所有 GPU。

**通信量分析**：

- 每步每个 GPU 发送 $D/N$ 数据
- Reduce-Scatter 需要 $N-1$ 步
- AllGather 需要 $N-1$ 步
- 总通信量：$2(N-1) \times D/N \approx 2D$（与 GPU 数量 $N$ 无关！）

**这就是 Ring AllReduce 的精妙之处——通信量不随 GPU 数量增长。**

### Ray 上的 AllReduce 示意

下面展示如何用 Ray 的 `@ray.remote` task 把 GPU 上的张量收集起来求平均，作为 AllReduce 语义的最小演示。Ray 官方文档：[Ray Core / Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)。

```python
import ray
import torch

ray.init()

@ray.remote(num_gpus=0.5)
def gather_and_mean(refs):
    """把若干 worker 上的 tensor 收回来求均值，AllReduce 的最朴素等价形式。"""
    tensors = ray.get(refs)
    stacked = torch.stack([t.to("cuda") for t in tensors], dim=0)
    return stacked.mean(dim=0)

# 模拟 8 张 GPU 的数据
data_list = [torch.randn(3, 4, device="cpu") for _ in range(8)]
refs = [ray.put(data) for data in data_list]
result = ray.get(gather_and_mean.remote(refs)).to("cpu")
```

::: tip 生产做法
真正的 AllReduce 走的是 NCCL/Gloo，由 `torch.distributed` 或更上层的 `DeepSpeed`/`FSDP` 调度。Ray 在大规模训练里更多承担调度与编排（见 [Ray Train](https://docs.ray.io/en/latest/train/train.html)），不是替代 NCCL 集合通信。
:::

使用 `torch.distributed` 进行通信（更接近生产环境）：

```python
import torch.distributed as dist

# 在每个 worker 中初始化
dist.init_process_group(
    backend="nccl",        # NVIDIA GPU 专用高性能后端
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

# AllReduce 示例
tensor = torch.randn(128, 256, device='cuda')
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 就地操作
tensor /= world_size  # 求平均
```

---

## 分布式训练系统实战：verl 的 Single-Controller 架构

在实际的 RLHF 训练系统里，需要同时编排 **rollout（推理生成）**、**reward 打分**、**reference 模型 logp 计算**、**critic 估值**、**actor 策略更新** 五种角色。verl（参考 [volcengine/verl](https://github.com/volcengine/verl) `single_controller/ray/base.py` 与 `trainer/ppo/ray_trainer.py`）的 **Single-Controller + Multi-Worker** 范式把所有编排集中到一个 Driver 进程，分布式通信由框架自动处理：

```python
# Driver 端：用 RayWorkerGroup 把"一组同构 Worker"包成一个对象，
# 直接像调单机方法一样调它

from verl.single_controller.ray import (
    RayResourcePool,
    RayClassWithInitArgs,
    RayWorkerGroup,
)

# 1. 声明资源池：8 卡放在同节点
resource_pool = RayResourcePool(process_on_nodes=[8], use_gpu=True)

# 2. 给每个角色声明 Worker 类（延迟实例化）
actor_rollout_cls = RayClassWithInitArgs(ActorRolloutRefWorker, config=actor_cfg)
critic_cls        = RayClassWithInitArgs(CriticWorker,           config=critic_cfg)
rm_cls            = RayClassWithInitArgs(RewardModelWorker,      config=rm_cfg)

# 3. 在资源池上启动各 Worker Group
actor_rollout_wg = RayWorkerGroup(resource_pool, actor_rollout_cls)
critic_wg        = RayWorkerGroup(resource_pool, critic_cls)
rm_wg            = RayWorkerGroup(resource_pool, rm_cls)
```

**Worker 端用 `@register` 装饰器声明分布式行为**：

```python
from verl.single_controller.base.decorator import register, Dispatch

class ActorRolloutRefWorker(Worker):

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts):
        # Driver 调 wg.generate_sequences(big_batch) 时：
        # 1. dispatch_fn 自动按 batch 维切片，分发到各 rank
        # 2. 在所有 worker 上并行执行此方法
        # 3. collect_fn 自动把结果拼回完整 batch
        return self.rollout.generate(prompts)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, batch):
        return self.actor.update(batch)
```

**Driver 端的 PPO 主循环看起来就像单机代码**（参考 `verl/trainer/ppo/ray_trainer.py` `RayPPOTrainer.fit()`）：

```python
def fit(self):
    for batch in self.train_dataloader:
        # 1. Rollout 生成 response
        gen_batch = self.actor_rollout_wg.generate_sequences(batch)

        # 2. 老 log_prob + 参考 log_prob
        old_logp = self.actor_rollout_wg.compute_log_prob(gen_batch)
        ref_logp = self.actor_rollout_wg.compute_ref_log_prob(gen_batch)

        # 3. 价值估计 + 奖励打分
        values  = self.critic_wg.compute_values(gen_batch)
        rewards = self.rm_wg.compute_rm_score(gen_batch)

        # 4. GAE 在 Driver CPU 算（轻量、无需分布式）
        batch = compute_advantage(batch, values, rewards, ref_logp)

        # 5. 各 Worker Group 各自 PPO 更新
        self.actor_rollout_wg.update_actor(batch)
        self.critic_wg.update_critic(batch)
```

这个架构体现了大规模 RLHF 训练的核心设计模式：

- **Single Controller**：所有编排集中到 Driver，Worker 只关心计算——避免了"每个 Worker 都需要知道全局拓扑"的复杂度
- **Resource Pool**：声明式分配 GPU，不需要手算 rank
- **Dispatch 装饰器**：把"分布式通信策略"内聚到 Worker 类的方法签名里，调用端无感
- **Hybrid Engine**：通过 `create_colocated_worker_cls` 把 actor / rollout / reference 三个角色挤进同一张 GPU，节省 3× 资源（详见 [Ray 框架](./ray-framework.md#_5-verl-的-single-controller-multi-worker-架构) 一章）

---

## 实战：常见配置方案

### 不同规模模型的推荐并行策略

| 模型规模 | GPU 数量 | 推荐策略 | 说明 |
|---------|---------|---------|------|
| **< 1B** | 1~4 | DDP | 单卡即可放下，数据并行加速 |
| **1B~7B** | 2~8 | DDP + ZeRO-2 | 优化器分片节省显存 |
| **7B~13B** | 4~16 | FSDP / ZeRO-3 | 参数也需要分片 |
| **13B~70B** | 16~64 | ZeRO-3 + TP | 张量并行放节点内 |
| **70B~200B** | 64~256 | TP + PP + ZeRO-1 | 完整 3D 并行 |
| **200B+** | 256~2048+ | TP + PP + ZeRO-1 + Expert Parallel | 3D 并行 + MoE 并行 |

### 常见框架选择

| 框架 | 支持的并行策略 | 适用场景 |
|-----|-------------|---------|
| PyTorch DDP | DP | 中小模型，最简单 |
| PyTorch FSDP | DP + ZeRO-3 | 中大模型，PyTorch 生态 |
| DeepSpeed | DP + ZeRO + PP | 大模型，灵活配置 |
| Megatron-LM | TP + PP + DP | 超大模型，极致性能 |
| Megatron-DeepSpeed | TP + PP + ZeRO | 超大模型，最全面 |
| ColossalAI | TP + PP + ZeRO + Sequence Parallel | 全能型，易用性好 |

### 实用配置示例

**场景 1：8 卡 A100 训练 Llama 7B**

```bash
# 使用 FSDP (最简单)
torchrun --nproc_per_node=8 train.py \
    --model_name meta-llama/Llama-2-7b \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config fsdp_config.json \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

**场景 2：32 卡训练 Llama 70B（DeepSpeed ZeRO-3）**

```bash
deepspeed --num_gpus 32 train.py \
    --model_name meta-llama/Llama-2-70b \
    --deepspeed ds_config_zero3.json \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True
```

**场景 3：64 卡训练 Llama 70B（Megatron-LM 3D 并行）**

```bash
# TP=8 (节点内), PP=4 (跨节点), DP=2 (全局)
python -m torch.distributed.launch \
    --nproc_per_node 8 \
    --nnodes 8 \
    pretrain_llama.py \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --bf16
```

---

## 苏格拉底时刻

::: details 1. 数据并行中，为什么需要 AllReduce 梯度？如果不同步会怎样？
AllReduce 保证所有 GPU 使用相同的平均梯度更新模型，维持模型副本的一致性。如果不同步，各张 GPU 的模型参数会逐渐分化，等价于独立训练多个不同的模型，训练会不收敛，无法利用大 batch 的统计优势。

数学上，AllReduce 保证了 $\theta_{t+1}^{(i)} = \theta_{t+1}^{(j)}, \forall i,j$，即所有 GPU 上参数始终保持同步。
:::


::: details 2. ZeRO-3 相比朴素数据并行节省了多少显存？代价是什么？
ZeRO-3 将每卡显存从 $16\Phi$ 降低到 $16\Phi/N$（$N$ 为 GPU 数量），实现了线性的显存缩减。代价是：

- **通信量增加 50%**：从 $2\Phi$ 增加到 $3\Phi$（每次前向/反向都需要 AllGather 完整参数）
- **通信频率更高**：每一层都需要通信，而非仅在反向传播结束后
- **计算延迟**：需要等待 AllGather 完成才能开始计算（可通过 prefetch 缓解）
:::


::: details "3. 张量并行中列并行和行并行的通信模式有什么不同？为什么 MLP 要"列+行"搭配？"
- **列并行**：前向需要将输入广播/复制到所有 GPU（$f$：identity），反向需要 AllReduce 汇总梯度
- **行并行**：前向需要 AllReduce 汇总部分结果（$g$：AllReduce），反向是 identity

MLP 的"列+行"搭配巧妙之处在于：列并行的输出天然是按列拆分的，恰好作为行并行的输入（行并行需要输入按列拆分）。这样**两层之间不需要额外通信**，整个 MLP 只在最后做一次 AllReduce。如果两层都用列并行，中间就需要一次额外的 AllGather。
:::


::: details "4. 流水线并行中的"气泡"是什么？为什么 1F1B 调度能减少气泡？"
"气泡"是指 GPU 空闲等待的时间。朴素流水线中，前向传播按顺序经过各 GPU，气泡率高达 $(p-1)/p$。

1F1B 通过两个关键优化减少气泡：

1. **微批次拆分**：将 mini-batch 拆成 $m$ 个 micro-batch，使流水线可以同时处理多个 micro-batch
2. **交错调度**：在稳态阶段每做一个前向就做一个反向，GPU 不再长时间空闲

气泡率从 $(p-1)/p$ 降到 $(p-1)/(m+p-1)$。当 $m \gg p$ 时，气泡率趋近于 0。
:::


::: details 5. 为什么张量并行必须放在节点内？能不能跨节点做张量并行？
技术上可以跨节点做张量并行，但性能极差。原因：

- 张量并行在**每一层**的前向和反向都需要通信（AllReduce 或 AllGather）
- 一个 80 层的模型，每个训练 step 需要 $80 \times 2 = 160$ 次张量并行通信
- 节点内 NVLink 带宽 ~600 GB/s，延迟 ~1 us
- 跨节点 InfiniBand 带宽 ~50 GB/s，延迟 ~5 us
- 带宽差 12x，延迟差 5x，160 次通信会导致巨大的性能损失

相比之下，流水线并行每层只传递一次激活值（一个较大的张量），通信次数少，适合跨节点。
:::


::: details 6. Ring AllReduce 的通信量为什么与 GPU 数量无关？
因为 Ring AllReduce 将数据分成 $N$ 份，每步每个 GPU 只发送 $D/N$ 的数据。总共需要 $2(N-1)$ 步：

$$
\text{每 GPU 总发送量} = 2(N-1) \times \frac{D}{N} = 2D \cdot \frac{N-1}{N} \approx 2D
$$

虽然步数增加了，但每步发送的数据量减小了，两者相互抵消。这意味着无论用 8 卡还是 1024 卡，每个 GPU 的通信量都是约 $2D$。

但注意：虽然通信量不变，**延迟**会随 GPU 数量线性增长（$2(N-1)$ 步），这是 Ring AllReduce 在超大规模集群上的瓶颈，此时需要层级化（Hierarchical）AllReduce。
:::


---

## 常见问题 & 面试考点

### 高频面试题

**Q1：混合精度训练中，为什么需要 FP32 master copy？**

> FP16 的精度有限（最小正数约 $6 \times 10^{-8}$），当学习率很小时，参数更新量 $\eta \cdot g$ 可能小于 FP16 的精度，导致更新被"吞掉"（underflow）。FP32 master copy 保证了参数更新的精度。

**Q2：Gradient Accumulation 和数据并行有什么区别？**

> 两者都能增大有效 batch size：
>
> - **Gradient Accumulation**：时间维度扩展，多个 step 的梯度累加后才更新一次，不需要额外 GPU
> - **数据并行**：空间维度扩展，多个 GPU 同时处理不同 batch，需要更多 GPU 但速度更快
>
> 实际中常结合使用：`global_batch_size = num_gpus × per_gpu_batch_size × gradient_accumulation_steps`

**Q3：Activation Checkpointing（梯度检查点）是什么？**

> 训练时不保存所有层的激活值，只保存部分"检查点"层的激活值。反向传播需要某层激活值时，从最近的检查点重新前向计算。
>
> - **显存节省**：从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 为层数）
> - **代价**：约增加 33% 的计算时间（需要重新计算前向）
> - **在大模型训练中几乎必用**

**Q4：NCCL 是什么？为什么 GPU 通信用 NCCL 而不用 MPI？**

> NCCL（NVIDIA Collective Communications Library）是 NVIDIA 专门为 GPU 优化的集合通信库。相比 MPI：
>
> - NCCL 直接利用 NVLink、NVSwitch、PCIe 等 GPU 互连硬件
> - NCCL 支持 GPU-Direct RDMA（GPU 间直接传输，不经过 CPU）
> - NCCL 的 AllReduce 实现比 MPI 在 GPU 场景下快数倍

**Q5：Sequence Parallelism（序列并行）是什么？**

> 序列并行是 Megatron-LM v3 提出的技术，将 Transformer 中 **非张量并行**的操作（LayerNorm、Dropout）也做分片。在张量并行中，这些操作在每个 GPU 上是冗余计算的，序列并行将序列维度拆分，进一步节省显存和计算。

---

## 推荐资源

### 论文

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - ZeRO 原始论文
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) - 张量并行
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) - GPipe
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473) - Megatron-LM 3D 并行
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) - FSDP 论文

### 开源框架

- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 微软的分布式训练框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA 的大模型训练框架
- [ColossalAI](https://github.com/hpcaitech/ColossalAI) - 易用的大模型并行框架
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) - PyTorch 原生全分片数据并行

### 教程与博客

- [Lillian Weng: Large Transformer Model Training](https://lilianweng.github.io/posts/2021-09-25-train-large/) - 经典综述
- [HuggingFace: Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many) - 实战指南
- [DeepSpeed Documentation](https://www.deepspeed.ai/getting-started/) - 官方文档
