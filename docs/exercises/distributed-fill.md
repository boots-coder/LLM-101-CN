---
title: "分布式训练填空"
description: "Level 2-3 填空：AllReduce、DDP、Tensor Parallelism、Pipeline Parallelism"
topics: [fill-in, distributed, DDP, tensor-parallelism, pipeline-parallelism, allreduce]
---
# 分布式训练代码填空 (Level 2-3)

> 本练习覆盖分布式训练的核心技术：Ring AllReduce、DDP 训练循环、Tensor Parallelism（列并行与行并行）、Pipeline Parallelism 微批次调度。
> 代码基于纯 Python / PyTorch 实现，用 `_____` 标记需要填写的部分。

---

## 练习 1: Ring AllReduce 模拟（Level 2）

### 背景

AllReduce 是分布式训练中最基础的通信原语，它的目标是让所有 GPU 上的梯度向量最终变为全局平均值（或求和值）。Ring AllReduce 是一种高效的实现方式，将 N 个 GPU 排成环状，分两个阶段完成：

1. **Scatter-Reduce 阶段**: 每个 GPU 将自己的数据分成 N 个 chunk，经过 N-1 轮通信，每个 GPU 上都积累出一个 chunk 的全局求和结果。
2. **AllGather 阶段**: 再经过 N-1 轮通信，将求和完毕的 chunk 广播到所有 GPU。

本练习用纯 Python list 模拟这个过程，不需要真实多 GPU 环境。

### 任务

```python
def ring_allreduce(gpu_grads, op="mean"):
    """
    模拟 Ring AllReduce。
    
    参数:
        gpu_grads: list of list，gpu_grads[i] 是第 i 个 GPU 上的梯度向量
                   例如 [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
        op: "mean" 或 "sum"
    
    返回:
        list of list，每个 GPU 上的梯度都变为全局平均值（或求和值）
    """
    import copy
    N = len(gpu_grads)
    chunk_size = len(gpu_grads[0]) // N
    
    # 将每个 GPU 的梯度切成 N 个 chunk
    # chunks[i][j] 表示 GPU i 的第 j 个 chunk
    chunks = []
    for i in range(N):
        gpu_chunks = []
        for j in range(N):
            start = j * chunk_size
            end = start + chunk_size
            gpu_chunks.append(list(gpu_grads[i][start:end]))
        chunks.append(gpu_chunks)
    
    # ========== Scatter-Reduce 阶段 ==========
    # 经过 N-1 轮，每轮：GPU i 将自己的某个 chunk 发送给 GPU (i+1)%N
    # 接收方将收到的 chunk 与自己对应位置的 chunk 做逐元素相加
    for step in range(N - 1):
        new_chunks = copy.deepcopy(chunks)
        for i in range(N):
            # ===== 填空 1: 确定发送的 chunk 索引 =====
            # 第 step 轮，GPU i 发送的 chunk 索引
            send_chunk_idx = _____  # 提示: (i - step) % N
            
            # ===== 填空 2: 确定接收方 GPU 编号 =====
            recv_gpu = _____  # 提示: 环形拓扑，下一个 GPU
            
            # ===== 填空 3: 接收方对应 chunk 做逐元素相加 =====
            for k in range(chunk_size):
                new_chunks[recv_gpu][send_chunk_idx][k] = _____
                # 提示: 接收方原有值 + 发送方的值
        chunks = new_chunks
    
    # ========== AllGather 阶段 ==========
    # 经过 N-1 轮，每轮：GPU i 将自己已经完成 reduce 的 chunk 发给下一个 GPU
    # 接收方直接覆盖（不是相加，而是替换）
    for step in range(N - 1):
        new_chunks = copy.deepcopy(chunks)
        for i in range(N):
            # ===== 填空 4: 确定本轮发送的 chunk 索引 =====
            send_chunk_idx = _____  # 提示: (i - step + 1) % N
            
            recv_gpu = (i + 1) % N
            
            # ===== 填空 5: 接收方直接覆盖对应 chunk =====
            new_chunks[recv_gpu][send_chunk_idx] = _____
            # 提示: 直接用发送方的 chunk 替换
        chunks = new_chunks
    
    # 将 chunks 重新拼成完整梯度
    result = []
    for i in range(N):
        grad = []
        for j in range(N):
            if op == "mean":
                grad.extend([v / N for v in chunks[i][j]])
            else:
                grad.extend(chunks[i][j])
        result.append(grad)
    
    return result
```

### 提示

- Scatter-Reduce 阶段的关键：每轮每个 GPU 发送一个 chunk 给下一个 GPU，接收方将收到的 chunk **累加**到自己对应位置
- AllGather 阶段的关键：每轮每个 GPU 发送一个已完成 reduce 的 chunk 给下一个 GPU，接收方**直接替换**
- chunk 索引的规律：每轮移动一位，使得 N-1 轮后所有 chunk 都被处理

<details>
<summary>参考答案</summary>

```python
# 填空 1: 发送的 chunk 索引
send_chunk_idx = (i - step) % N

# 填空 2: 接收方 GPU 编号（环形拓扑的下一个）
recv_gpu = (i + 1) % N

# 填空 3: 逐元素相加
new_chunks[recv_gpu][send_chunk_idx][k] = (
    chunks[recv_gpu][send_chunk_idx][k] + chunks[i][send_chunk_idx][k]
)

# 填空 4: AllGather 阶段发送的 chunk 索引
send_chunk_idx = (i - step + 1) % N

# 填空 5: 直接覆盖
new_chunks[recv_gpu][send_chunk_idx] = list(chunks[i][send_chunk_idx])
```

**验证:**
```python
gpu_grads = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]
result = ring_allreduce(gpu_grads, op="mean")

# 每个 GPU 上的梯度应该相同，且等于全局平均值
expected = [7.0, 8.0, 9.0, 10.0]  # (1+5+9+13)/4, (2+6+10+14)/4, ...
for i, grad in enumerate(result):
    print(f"GPU {i}: {grad}")
    assert grad == expected, f"GPU {i} 结果不正确"
print("所有 GPU 梯度一致，Ring AllReduce 正确!")
```

</details>

---

## 练习 2: 手写 DDP 训练循环（Level 2-3）

### 背景

PyTorch DistributedDataParallel (DDP) 是最常用的数据并行方案。其核心思路：

1. 每个 GPU（进程）持有模型的**完整副本**
2. 每个 GPU 处理不同的数据子集（通过 `DistributedSampler` 划分）
3. Forward 各自独立计算
4. Backward 之后，DDP 自动对梯度做 AllReduce，确保所有 GPU 上的梯度一致
5. 各 GPU 独立执行 `optimizer.step()`，由于初始参数相同 + 梯度相同，参数保持同步

### 任务

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # ===== 填空 1: 初始化进程组 =====
    dist.init_process_group(
        backend=_____,   # 提示: GPU 训练用哪个后端?
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs=3):
    setup(rank, world_size)
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    # 创建模型并移动到对应 GPU
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    
    # ===== 填空 2: 用 DDP 包装模型 =====
    model = _____  # 提示: DDP(model, ...)，需要指定 device_ids
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 模拟数据集
    dataset = TensorDataset(
        torch.randn(1000, 784),
        torch.randint(0, 10, (1000,)),
    )
    
    # ===== 填空 3: 创建分布式采样器 =====
    sampler = _____  # 提示: DistributedSampler，确保每个 GPU 拿到不同数据子集
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        # ===== 填空 4: 使用 sampler 时不能 shuffle =====
        shuffle=_____,  # 提示: sampler 和 shuffle 互斥
    )
    
    for epoch in range(epochs):
        # ===== 填空 5: 每个 epoch 设置 sampler 的 epoch =====
        # 提示: 确保每个 epoch 的数据划分不同
        _____
        
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            # 注意: DDP 会在 backward 过程中自动对梯度做 AllReduce
            # 不需要手动同步梯度
            optimizer.step()
            
            total_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    cleanup()

# 启动方式 (命令行):
# torchrun --nproc_per_node=4 ddp_train.py
# 或者用 mp.spawn:
# import torch.multiprocessing as mp
# mp.spawn(train, args=(world_size,), nprocs=world_size)
```

### 提示

- GPU 后端使用 `"nccl"`，CPU 可以用 `"gloo"`
- `DDP` 包装时需要 `device_ids=[rank]`，告诉 DDP 模型在哪个 GPU 上
- `DistributedSampler` 自动根据 `rank` 和 `world_size` 划分数据
- 使用 `sampler` 时必须设置 `shuffle=False`，因为打乱由 sampler 控制
- `sampler.set_epoch(epoch)` 确保每个 epoch 的 shuffle 种子不同

<details>
<summary>参考答案</summary>

```python
# 填空 1: 初始化进程组
dist.init_process_group(
    backend="nccl",
    rank=rank,
    world_size=world_size,
)

# 填空 2: DDP 包装
model = DDP(model, device_ids=[rank])

# 填空 3: 分布式采样器
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

# 填空 4: 使用 sampler 时 shuffle 必须为 False
shuffle = False

# 填空 5: 设置 epoch
sampler.set_epoch(epoch)
```

**预期行为:**
```
# 使用 4 个 GPU 运行时:
# - 每个 GPU 处理 1000/4 = 250 条数据
# - 所有 GPU 上的模型参数始终保持一致
# - 每个 GPU 计算的 loss 可能不同（因为数据不同），但梯度经过 AllReduce 后一致
# - 训练速度接近单卡的 4 倍（通信开销除外）

# 验证参数同步（在代码中添加）:
# for name, param in model.named_parameters():
#     gathered = [torch.zeros_like(param) for _ in range(world_size)]
#     dist.all_gather(gathered, param)
#     for g in gathered:
#         assert torch.equal(g, param), "参数不同步!"
```

</details>

---

## 练习 3: Tensor Parallelism -- 列并行线性层（Level 3）

### 背景

Megatron-style 张量并行将单个线性层的权重矩阵切分到多个 GPU 上，从而突破单卡显存限制。**列并行 (Column Parallel)** 将权重矩阵 `W` 按列切分：

对于线性层 `Y = XW + b`，其中 `W` 的形状为 `[d_in, d_out]`：
- 将 `W` 按列切成 N 份：`W = [W_1 | W_2 | ... | W_N]`，每份形状 `[d_in, d_out/N]`
- 每个 GPU 独立计算 `Y_i = X @ W_i`，得到部分输出
- 最后将各 GPU 的输出沿最后一维拼接：`Y = [Y_1 | Y_2 | ... | Y_N]`

这种方式不需要通信就能完成 forward（拼接可以延迟到需要完整输出时）。

### 任务

```python
import torch
import torch.nn as nn

def column_parallel_forward(x, weight, bias, num_gpus):
    """
    模拟列并行线性层的 forward。
    
    参数:
        x:        [batch, d_in]    输入（所有 GPU 共享完整输入）
        weight:   [d_in, d_out]    完整权重矩阵
        bias:     [d_out]          完整偏置向量
        num_gpus: int              模拟的 GPU 数量
    
    返回:
        output:   [batch, d_out]   拼接后的完整输出
    """
    d_in, d_out = weight.shape
    assert d_out % num_gpus == 0
    chunk_size = d_out // num_gpus
    
    gpu_outputs = []
    for i in range(num_gpus):
        # ===== 填空 1: 按列切分权重 =====
        # 提示: 第 i 个 GPU 拿到 W 的第 i 个列块
        w_chunk = _____  # weight[:, start:end]
        
        # ===== 填空 2: 按列切分偏置 =====
        b_chunk = _____  # bias[start:end]
        
        # ===== 填空 3: 各分片独立计算 =====
        y_chunk = _____  # 提示: x @ w_chunk + b_chunk
        
        gpu_outputs.append(y_chunk)
    
    # ===== 填空 4: 拼接所有 GPU 的输出 =====
    output = _____  # 提示: torch.cat, 沿最后一维拼接
    
    return output


def verify_column_parallel():
    """验证列并行结果与标准线性层一致"""
    torch.manual_seed(42)
    d_in, d_out, batch = 512, 256, 4
    num_gpus = 4
    
    linear = nn.Linear(d_in, d_out, bias=True)
    x = torch.randn(batch, d_in)
    
    y_standard = linear(x)
    # nn.Linear 的 weight 形状是 [d_out, d_in]，需要转置为 [d_in, d_out]
    y_parallel = column_parallel_forward(x, linear.weight.T, linear.bias, num_gpus)
    
    print(f"最大误差: {(y_standard - y_parallel).abs().max().item():.2e}")
    assert torch.allclose(y_standard, y_parallel, atol=1e-5)
    print("列并行线性层验证通过!")
```

### 提示

- 列切分：`weight[:, i*chunk_size : (i+1)*chunk_size]`
- 每个分片的计算完全独立，不需要通信
- `torch.cat(tensors, dim=-1)` 沿最后一维拼接
- `nn.Linear` 存储的权重是 `[d_out, d_in]`，与数学公式的 `[d_in, d_out]` 是转置关系

<details>
<summary>参考答案</summary>

```python
# 填空 1: 按列切分权重
w_chunk = weight[:, i * chunk_size : (i + 1) * chunk_size]

# 填空 2: 按列切分偏置
b_chunk = bias[i * chunk_size : (i + 1) * chunk_size]

# 填空 3: 各分片独立计算
y_chunk = x @ w_chunk + b_chunk

# 填空 4: 拼接输出
output = torch.cat(gpu_outputs, dim=-1)
```

**验证:**
```python
verify_column_parallel()
# 输出:
# 标准输出形状: torch.Size([4, 256])
# 并行输出形状: torch.Size([4, 256])
# 最大误差: 0.00e+00
# 列并行线性层验证通过!
```

</details>

---

## 练习 4: Tensor Parallelism -- 行并行线性层（Level 3）

### 背景

行并行 (Row Parallel) 是列并行的对偶。对于线性层 `Y = XW`，将 `W` 按行切分：

- `W` 形状 `[d_in, d_out]`，按行切成 N 份：每份 `W_i` 形状 `[d_in/N, d_out]`
- 输入 `X` 也相应按列切分：每份 `X_i` 形状 `[batch, d_in/N]`
- 每个 GPU 计算 `Y_i = X_i @ W_i`，得到 `[batch, d_out]`
- 最终结果通过 AllReduce（求和）得到：`Y = sum(Y_1, Y_2, ..., Y_N)`

在 Transformer 的 FFN 中，通常使用 **column-parallel + row-parallel** 的配对方式：
- 第一个线性层用列并行：输出自然被切分到各 GPU
- 第二个线性层用行并行：输入正好是切分的，输出通过 AllReduce 汇总

### 任务

```python
import torch
import torch.nn as nn

def row_parallel_forward(x, weight, bias, num_gpus):
    """
    模拟行并行线性层的 forward。
    
    参数:
        x:        [batch, d_in]    完整输入
        weight:   [d_in, d_out]    完整权重矩阵
        bias:     [d_out]          偏置（只在 AllReduce 后加一次）
        num_gpus: int              模拟的 GPU 数量
    
    返回:
        output:   [batch, d_out]   AllReduce 后的完整输出
    """
    d_in, d_out = weight.shape
    assert d_in % num_gpus == 0
    chunk_size = d_in // num_gpus
    
    gpu_outputs = []
    for i in range(num_gpus):
        # ===== 填空 1: 按行切分权重 =====
        # 提示: 第 i 个 GPU 拿到 W 的第 i 个行块
        w_chunk = _____  # weight[start:end, :]
        
        # ===== 填空 2: 对应切分输入 =====
        # 提示: 输入的列与权重的行对应
        x_chunk = _____  # x[:, start:end]
        
        # ===== 填空 3: 各分片独立计算 =====
        y_chunk = _____  # 提示: x_chunk @ w_chunk
        
        gpu_outputs.append(y_chunk)
    
    # ===== 填空 4: AllReduce — 对所有分片结果求和 =====
    output = _____  # 提示: 逐元素求和所有 gpu_outputs
    
    # 偏置只加一次（AllReduce 之后）
    output = output + bias
    
    return output


def verify_row_parallel():
    """验证行并行结果与标准线性层一致"""
    torch.manual_seed(42)
    d_in, d_out, batch = 512, 256, 4
    num_gpus = 4
    
    linear = nn.Linear(d_in, d_out, bias=True)
    x = torch.randn(batch, d_in)
    
    y_standard = linear(x)
    y_parallel = row_parallel_forward(x, linear.weight.T, linear.bias, num_gpus)
    
    print(f"最大误差: {(y_standard - y_parallel).abs().max().item():.2e}")
    assert torch.allclose(y_standard, y_parallel, atol=1e-5)
    print("行并行线性层验证通过!")
```

### 提示

- 行切分：`weight[i*chunk_size:(i+1)*chunk_size, :]`
- 输入切分与权重行切分对应：`x[:, i*chunk_size:(i+1)*chunk_size]`
- AllReduce 求和可以用 `sum()` 或循环累加：`gpu_outputs[0] + gpu_outputs[1] + ...`
- 偏置只在 AllReduce 之后加一次，否则会被重复加 N 次

<details>
<summary>参考答案</summary>

```python
# 填空 1: 按行切分权重
w_chunk = weight[i * chunk_size : (i + 1) * chunk_size, :]

# 填空 2: 对应切分输入
x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]

# 填空 3: 各分片独立计算
y_chunk = x_chunk @ w_chunk

# 填空 4: AllReduce 求和
output = gpu_outputs[0]
for t in gpu_outputs[1:]:
    output = output + t
# 或者: output = torch.stack(gpu_outputs).sum(dim=0)
```

**验证:**
```python
verify_row_parallel()
# 输出:
# 最大误差: 0.00e+00
# 行并行线性层验证通过!
```

**讨论: 为什么 Transformer FFN 用 column-parallel + row-parallel 配对?**

FFN 包含两个线性层: `Y = W2(ReLU(W1(X)))`。

- W1 用列并行，激活值 `ReLU(W1(X))` 自然沿 hidden_dim 切分在各 GPU 上
- W2 用行并行，输入正好是切分好的，直接消费无需通信
- 整个 FFN 只需要在 W2 输出处做一次 AllReduce，通信量最小
- 如果两层都用列并行，中间和结尾各需一次 AllGather，通信量翻倍

</details>

---

## 练习 5: Pipeline Parallelism 微批次调度（Level 3）

### 背景

流水线并行 (Pipeline Parallelism) 将模型按层切分为多个 stage，每个 stage 放在一个 GPU 上。为了减少流水线气泡（bubble），将一个 mini-batch 切成多个 micro-batch，让不同 stage 同时处理不同的 micro-batch。

GPipe 式调度的工作方式：
- 先将所有 micro-batch 按顺序通过所有 stage（forward）
- 再反向通过所有 stage（backward）
- 时间步中存在空闲 stage，即"气泡"

设有 S 个 stage、M 个 micro-batch，总时间步为 `(S - 1) + M`（forward），backward 同理。

### 任务

```python
def gpipe_schedule(num_stages, num_microbatches):
    """
    生成 GPipe 式流水线调度表。
    
    参数:
        num_stages:       int  stage 数量 (S)
        num_microbatches: int  micro-batch 数量 (M)
    
    返回:
        forward_table:  dict, forward_table[(stage, time_step)] = micro_batch_id 或 None
        backward_table: dict, 同上
        bubble_ratio:   float, 气泡占比
    """
    S = num_stages
    M = num_microbatches
    
    # Forward 阶段需要的总时间步数
    total_fwd_steps = (S - 1) + M
    
    # ========== Forward 调度 ==========
    forward_table = {}
    for t in range(total_fwd_steps):
        for s in range(S):
            # ===== 填空 1: 计算当前 stage 在时间步 t 处理的 micro-batch 编号 =====
            # 提示: micro-batch m 到达 stage s 的时间步是 m + s
            # 所以在时间步 t，stage s 处理的是 micro-batch (t - s)
            mb = _____  # t - s
            
            # ===== 填空 2: 判断 micro-batch 编号是否有效 =====
            if _____:  # 提示: 0 <= mb < M
                forward_table[(s, t)] = mb
            else:
                forward_table[(s, t)] = None
    
    # ========== Backward 调度 ==========
    # Backward 在所有 forward 完成后开始
    # Backward 的顺序是反向的: 先从最后一个 stage 开始
    total_bwd_steps = (S - 1) + M
    backward_table = {}
    for t in range(total_bwd_steps):
        for s in range(S):
            # ===== 填空 3: Backward 调度 =====
            # 提示: backward 从 stage S-1 开始，方向相反
            # stage s 的 backward 在 forward 的镜像位置
            # micro-batch m 的 backward 到达 stage s 的时间步是 m + (S - 1 - s)
            mb = _____  # t - (S - 1 - s)
            
            if 0 <= mb < M:
                backward_table[(s, t)] = mb
            else:
                backward_table[(s, t)] = None
    
    # ========== 计算 Bubble Ratio ==========
    total_slots = S * (total_fwd_steps + total_bwd_steps)
    
    # ===== 填空 4: 统计有效（非气泡）的 slot 数量 =====
    active_slots = _____
    # 提示: forward 和 backward 中 value 不为 None 的 slot 总数
    
    # ===== 填空 5: 计算 bubble ratio =====
    bubble_ratio = _____  # 提示: (总 slot - 有效 slot) / 总 slot
    
    return forward_table, backward_table, bubble_ratio


def print_gantt(table, label, num_stages, num_steps):
    """打印单个阶段的甘特图"""
    print(f"=== {label} ===")
    print(f"{'Stage':<8}", end="")
    for t in range(num_steps):
        print(f"{'t='+str(t):<6}", end="")
    print()
    for s in range(num_stages):
        print(f"S{s:<7}", end="")
        for t in range(num_steps):
            mb = table.get((s, t))
            print(f"{label[0]+str(mb):<6}" if mb is not None else f"{'.':<6}", end="")
        print()
```

### 提示

- Forward: micro-batch `m` 到达 stage `s` 的时间步是 `m + s`，因为它需要先通过前面的 stage
- Backward: micro-batch `m` 到达 stage `s` 的时间步是 `m + (S-1-s)`，因为 backward 从最后一个 stage 开始
- Bubble ratio 公式: `(S-1) / (S-1+M)` 对于纯 forward 或纯 backward；整体类似
- 减少 bubble 的方法：增大 M（更多 micro-batch），但会增加显存占用

<details>
<summary>参考答案</summary>

```python
# 填空 1: 当前 stage 处理的 micro-batch
mb = t - s

# 填空 2: 判断有效性
if 0 <= mb < M:

# 填空 3: Backward 调度
mb = t - (S - 1 - s)

# 填空 4: 统计有效 slot
active_slots = sum(1 for v in forward_table.values() if v is not None) + \
               sum(1 for v in backward_table.values() if v is not None)

# 填空 5: 计算 bubble ratio
bubble_ratio = (total_slots - active_slots) / total_slots
```

**验证:**
```python
S, M = 4, 6
fwd, bwd, ratio = gpipe_schedule(S, M)
total_steps = (S - 1) + M
print_gantt(fwd, "Forward", S, total_steps)
print_gantt(bwd, "Backward", S, total_steps)
print(f"\nBubble ratio: {ratio:.2%}")

# 预期: Forward 甘特图中 S0 最先开始处理 F0-F5，S3 最晚
#        Backward 甘特图中 S3 最先开始处理 B0-B5，S0 最晚
# Bubble ratio: 33.33%  (理论值: 2*(S-1) / (2*(S-1+M)) = 6/18)

# 增大 M 可以降低 bubble ratio:
for m in [4, 8, 16, 32]:
    _, _, r = gpipe_schedule(S, m)
    print(f"S={S}, M={m}: bubble = {r:.1%}")
# S=4, M=4:  bubble = 42.9%
# S=4, M=8:  bubble = 27.3%
# S=4, M=16: bubble = 15.8%
# S=4, M=32: bubble = 8.6%
```

</details>

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### Ring AllReduce 核心循环

<CodeMasker title="Ring AllReduce: Scatter-Reduce 与 AllGather" :mask-ratio="0.15">
# Scatter-Reduce 阶段
for step in range(N - 1):
    new_chunks = copy.deepcopy(chunks)
    for i in range(N):
        send_chunk_idx = (i - step) % N
        recv_gpu = (i + 1) % N
        for k in range(chunk_size):
            new_chunks[recv_gpu][send_chunk_idx][k] = (
                chunks[recv_gpu][send_chunk_idx][k] + chunks[i][send_chunk_idx][k]
            )
    chunks = new_chunks

# AllGather 阶段
for step in range(N - 1):
    new_chunks = copy.deepcopy(chunks)
    for i in range(N):
        send_chunk_idx = (i - step + 1) % N
        recv_gpu = (i + 1) % N
        new_chunks[recv_gpu][send_chunk_idx] = list(chunks[i][send_chunk_idx])
    chunks = new_chunks
</CodeMasker>

### DDP 训练循环关键步骤

<CodeMasker title="DDP 初始化与训练循环" :mask-ratio="0.15">
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, shuffle=False)

for epoch in range(epochs):
    sampler.set_epoch(epoch)
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x.to(device))
        loss = criterion(output, batch_y.to(device))
        loss.backward()   # DDP 自动 AllReduce 梯度
        optimizer.step()
</CodeMasker>

### Tensor Parallelism: 列并行与行并行

<CodeMasker title="列并行 Forward 与行并行 Forward" :mask-ratio="0.15">
# 列并行: 按列切分权重，拼接输出
for i in range(num_gpus):
    w_chunk = weight[:, i * chunk_size : (i + 1) * chunk_size]
    b_chunk = bias[i * chunk_size : (i + 1) * chunk_size]
    y_chunk = x @ w_chunk + b_chunk
    gpu_outputs.append(y_chunk)
output = torch.cat(gpu_outputs, dim=-1)

# 行并行: 按行切分权重，AllReduce 求和
for i in range(num_gpus):
    w_chunk = weight[i * chunk_size : (i + 1) * chunk_size, :]
    x_chunk = x[:, i * chunk_size : (i + 1) * chunk_size]
    y_chunk = x_chunk @ w_chunk
    gpu_outputs.append(y_chunk)
output = gpu_outputs[0]
for t in gpu_outputs[1:]:
    output = output + t
output = output + bias
</CodeMasker>

### Pipeline Parallelism 微批次调度

<CodeMasker title="GPipe 调度: Forward 与 Backward" :mask-ratio="0.15">
total_fwd_steps = (S - 1) + M

# Forward: micro-batch m 到达 stage s 的时间步 = m + s
for t in range(total_fwd_steps):
    for s in range(S):
        mb = t - s
        if 0 <= mb < M:
            forward_table[(s, t)] = mb

# Backward: micro-batch m 到达 stage s 的时间步 = m + (S-1-s)
for t in range(total_bwd_steps):
    for s in range(S):
        mb = t - (S - 1 - s)
        if 0 <= mb < M:
            backward_table[(s, t)] = mb

# Bubble ratio
active_slots = sum(1 for v in forward_table.values() if v is not None) + \
               sum(1 for v in backward_table.values() if v is not None)
bubble_ratio = (total_slots - active_slots) / total_slots
</CodeMasker>
