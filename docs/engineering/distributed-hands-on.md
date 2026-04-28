---
title: 分布式训练实操代码
description: 从集合通信原语到 ZeRO、张量并行、流水线并行、Ring Attention 的从零实现，每一节都有可运行的 PyTorch 代码。
topics: [Distributed Training, Data Parallelism, Tensor Parallelism, Pipeline Parallelism, ZeRO, Ring Attention]
prereqs: [PyTorch 基础, distributed.md 概念篇, 多进程/多 GPU 基本概念]
references: ["本章代码实现参考了开源分布式训练教程，经过重新设计与改写"]
---

# 分布式训练实操代码

> 概念篇见 [分布式训练](./distributed.md)。本文用**可运行的最小代码**把每种并行策略拆到骨架级别，让你从 `dist.send/recv` 一路写到 Ring Attention。

::: tip 运行环境
所有示例基于 `torch.multiprocessing.spawn` 在单机多进程上模拟多 GPU（backend=`gloo`）。如果你有多卡机器，只需把 backend 换成 `nccl` 并使用 `torchrun` 即可。
:::

---

## 1. 集合通信原语实战

分布式训练的一切，都建立在**集合通信原语**之上。先搞清楚这几个砖块，后面的并行策略就是搭积木。

### 1.1 Broadcast — 一对多广播

将 rank 0 的数据广播给所有 rank，是参数初始化同步的基石。

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def broadcast_demo(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)

    tensor = torch.zeros(2)
    if rank == 0:
        tensor += 100  # 只有 rank 0 持有数据

    print(f"[广播前] Rank {rank}: {tensor}")
    dist.broadcast(tensor, src=0)  # 所有 rank 拿到相同值
    print(f"[广播后] Rank {rank}: {tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(broadcast_demo, args=(4,), nprocs=4)
```

**手动实现版本**：用 P2P 通信模拟 broadcast —— rank 0 逐个 `send`，其他 rank `recv`：

```python
def p2p_broadcast(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)))

    tensor = torch.zeros(2)
    if rank == 0:
        tensor += 100
        for r in range(1, world_size):
            dist.send(tensor, dst=r)
    else:
        dist.recv(tensor, src=0)

    print(f"Rank {rank}: {tensor}")
    dist.destroy_process_group()
```

### 1.2 All-Reduce — 全局规约

每个 rank 贡献一个 tensor，规约后**所有 rank 都拿到相同结果**。这是 DDP 梯度同步的核心。

```python
def allreduce_demo(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)

    tensor = torch.ones(1) * 2 * rank  # rank 0->0, rank 1->2, rank 2->4, rank 3->6
    print(f"[规约前] Rank {rank}: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 0+2+4+6=12
    print(f"[规约后] Rank {rank}: {tensor}")

    dist.destroy_process_group()
```

### 1.3 All-Gather — 全局收集

每个 rank 贡献一小段，收集后**每个 rank 都拥有完整数据**。ZeRO 参数同步的关键。

```python
def allgather_demo(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)

    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]

    dist.all_gather(tensor_list, tensor)
    print(f"Rank {rank}: 本地={tensor}, 收集后={tensor_list}")

    dist.destroy_process_group()
```

### 1.4 Ring All-Reduce — 带宽最优的规约

标准 All-Reduce 的经典实现：先 **Reduce-Scatter**（每个 rank 分到一块 reduce 后的结果），再 **All-Gather**（把各块拼回去）。

```python
def ring_allreduce(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)

    data = torch.zeros(2 * world_size, dtype=torch.int64) + 1
    chunks = list(torch.split(data, 2))
    tmp = torch.zeros(2, dtype=torch.int64)

    # Stage 1: Reduce-Scatter（环形累加）
    for i in range(world_size - 1):
        send_idx = (rank - i) % world_size
        recv_idx = (rank - i - 1) % world_size
        if rank % world_size == 0:
            dist.send(chunks[send_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp, src=(rank - 1) % world_size)
            dist.send(chunks[send_idx], dst=(rank + 1) % world_size)
        chunks[recv_idx] += tmp

    # Stage 2: All-Gather（环形广播）
    for i in range(world_size - 1):
        send_idx = (i + rank + 1) % world_size
        recv_idx = (send_idx + 1) % world_size
        if rank % world_size == 0:
            dist.send(chunks[send_idx], dst=(rank + 1) % world_size)
            dist.recv(tmp, src=(rank - 1) % world_size)
        else:
            dist.recv(tmp, src=(rank - 1) % world_size)
            dist.send(chunks[send_idx], dst=(rank + 1) % world_size)
        chunks[recv_idx] = tmp.clone()

    print(f"Rank {rank}: {chunks}")
    dist.destroy_process_group()
```

::: details 苏格拉底时刻
1. Ring All-Reduce 的通信量为 $2(N-1)/N \times D$，其中 $D$ 是数据量、$N$ 是 rank 数。为什么它是**带宽最优**的？
2. 如果某个 rank 的计算比其他 rank 慢很多，Ring 拓扑会如何表现？对比 Tree-Reduce 有何不同？
3. `dist.all_reduce` 内部在 NCCL 后端实际用的是哪种算法？
:::

---

## 2. DDP 数据并行实战

数据并行的核心思想：**每个 rank 持有完整模型副本，各自处理不同的数据子集，梯度通过 All-Reduce 同步**。

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, classes):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(dim_hidden, classes)

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))

class RandomDataset(Dataset):
    def __init__(self, N, dim, classes):
        self.data = torch.randn(N, dim)
        self.labels = torch.softmax(torch.randn(N, classes), dim=-1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        return {"x": self.data[i], "y": self.labels[i]}

def ddp_train(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:12345",
                            rank=rank, world_size=world_size)

    model = DDP(ToyModel(16, 512, 10))  # 自动在 backward 时插入 All-Reduce
    dataset = RandomDataset(1024, 16, 10)
    sampler = DistributedSampler(dataset)  # 保证每个 rank 拿不同的数据
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)  # 打乱顺序
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch["x"])
            loss = loss_fn(out, batch["y"])
            loss.backward()       # DDP hook: All-Reduce 梯度
            optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}, loss={loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(ddp_train, args=(4,), nprocs=4)
```

::: warning 易错点
- `DistributedSampler` 必须每个 epoch 调用 `set_epoch()`，否则每个 epoch 的数据划分相同。
- DDP 自动在 `backward()` 中做 All-Reduce，**不要**手动再调 `dist.all_reduce` 梯度。
:::

::: details 苏格拉底时刻
1. DDP 在反向传播时用了 **gradient bucketing**（梯度分桶），这比逐参数 All-Reduce 快在哪里？
2. 如果模型有 `BatchNorm`，DDP 的行为会有什么不同？`SyncBatchNorm` 做了什么？
3. DDP 与 `DataParallel` 的核心区别是什么？为什么后者效率更低？
:::

---

## 3. ZeRO 从零实现

ZeRO（Zero Redundancy Optimizer）的核心思想：**不同 rank 各只持有一部分优化器状态/梯度/参数**，用的时候再通过通信收集回来。

### 3.1 ZeRO-1：优化器状态分片

每个 rank 只维护 $1/N$ 的 Adam 状态（M 和 V），更新完后 All-Gather 参数。

```python
class AdamZeRO1:
    """ZeRO Stage 1: 优化器状态分片到各 rank"""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, world_size=1, rank=0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.params = list(params)
        self.world_size, self.rank = world_size, rank
        self.t = 0

        # 每个 rank 只初始化 1/N 的 M, V
        self.M, self.V = [], []
        for p in self.params:
            shard_size = p.data.numel() // world_size
            self.M.append(torch.zeros(shard_size))
            self.V.append(torch.zeros(shard_size))

    def step(self):
        self.t += 1
        for p, M, V in zip(self.params, self.M, self.V):
            # Step 1: All-Reduce 梯度（完整梯度人人都有）
            dist.all_reduce(p.grad, dist.ReduceOp.SUM)
            p.grad /= self.world_size

            # Step 2: 取自己负责的那一片梯度
            shard = p.grad.numel() // self.world_size
            my_grad = p.grad.view(-1)[self.rank * shard:(self.rank + 1) * shard]

            # Step 3: Adam 更新（只更新自己的分片）
            M = self.beta1 * M + (1 - self.beta1) * my_grad
            V = self.beta2 * V + (1 - self.beta2) * my_grad.pow(2)
            m_hat = M / (1 - self.beta1 ** self.t)
            v_hat = V / (1 - self.beta2 ** self.t)

            my_weight = p.data.view(-1)[self.rank * shard:(self.rank + 1) * shard]
            my_weight -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

            # Step 4: All-Gather 同步完整参数
            full = torch.zeros(p.data.numel(), dtype=p.data.dtype)
            dist.all_gather_into_tensor(full, my_weight)
            p.data = full.reshape(p.data.shape)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

::: tip ZeRO-1 通信量分析
与标准 DDP 相比，ZeRO-1 多了一次 All-Gather（参数同步），但优化器状态的显存从 $12\Psi$（Adam fp32）降低到 $12\Psi / N$。对于大模型训练，这笔交易非常划算。
:::

### 3.2 ZeRO-2 与 ZeRO-3 的关键差异

| 级别 | 分片内容 | 额外通信 | 显存节省 |
|:---:|:---:|:---:|:---:|
| ZeRO-1 | 优化器状态 | +1 次 All-Gather | $12\Psi / N$ |
| ZeRO-2 | + 梯度 | 用 Reduce-Scatter 替换 All-Reduce | $+(2\Psi / N)$ |
| ZeRO-3 | + 模型参数 | 前向/反向各需 All-Gather 参数 | $+(2\Psi / N)$ |

**ZeRO-2 核心变化** — 把 All-Reduce 梯度拆成 Reduce-Scatter，每个 rank 只保留自己负责的梯度分片：

```python
def backward_zero2(model, rank, world_size):
    """ZeRO-2: 梯度 Reduce-Scatter，每个 rank 只留 1/N 梯度"""
    for param in model.parameters():
        if param.grad is None:
            continue
        # 先 All-Reduce 拿到完整梯度
        dist.all_reduce(param.grad, dist.ReduceOp.SUM)
        param.grad /= world_size
        full_grad = param.grad.view(-1).clone()

        # Scatter: 每个 rank 只保留自己的分片
        shard = param.grad.numel() // world_size
        param.grad.data = torch.zeros(shard)
        if rank == 0:
            grad_list = list(full_grad.split(shard))
            dist.scatter(param.grad.data, grad_list, src=0)
        else:
            dist.scatter(param.grad.data, [], src=0)
```

**ZeRO-3 核心变化** — 模型参数本身也分片，前向时按需 All-Gather：

```python
def forward_zero3(model, x):
    """ZeRO-3: 前向计算时 All-Gather 参数，计算完再丢弃冗余"""
    # 以 w1 为例：每个 rank 只存 1/N 的权重
    w1_full = torch.zeros(model.w1_full_shape)
    dist.all_gather_into_tensor(w1_full.view(-1), model.w1.weight.data)
    model.w1.weight.data = w1_full  # 临时恢复完整参数
    h = model.w1(x)
    # ... 同理处理其他层
    return h
```

::: details 苏格拉底时刻
1. ZeRO-3 前向时要 All-Gather 参数，反向时也要——这与张量并行 (Tensor Parallelism) 的核心区别是什么？
2. DeepSpeed ZeRO-3 的 `offload` 机制把什么东西放到了 CPU？通信开销如何变化？
3. 为什么 ZeRO-2 用 Reduce-Scatter 比 All-Reduce + 手动切片更高效？
:::

---

## 4. 张量并行从零实现

张量并行把**单层的权重矩阵**切分到多个 GPU 上。核心是两种切分方式：

### 4.1 Column Parallel Linear（列并行）

权重按**列**切分：$W \in \mathbb{R}^{d_{in} \times d_{out}}$ → 每个 rank 持有 $W_i \in \mathbb{R}^{d_{in} \times (d_{out}/N)}$。

- **前向**：输入 $X$ 广播到所有 rank，各自计算 $Y_i = X W_i$（局部输出）
- **反向**：$\frac{\partial L}{\partial X}$ 需要 All-Reduce（因为每个 rank 只看到部分列的梯度）

```python
import torch.autograd as autograd

class _ColParallelFn(autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(w, x)
        return x @ w  # 局部计算

    @staticmethod
    def backward(ctx, grad_out):
        w, x = ctx.saved_tensors
        grad_x = grad_out @ w.t()
        dist.all_reduce(grad_x, dist.ReduceOp.SUM)  # dx 需要 reduce
        grad_w = x.transpose(-2, -1) @ grad_out
        return grad_x, grad_w

class ColParallelLinear(nn.Module):
    def __init__(self, d_in, d_out, rank=0, world_size=1):
        super().__init__()
        # 每个 rank 只持有 d_out/N 列
        self.w = nn.Linear(d_in, d_out // world_size, bias=False)

    def forward(self, x):
        return _ColParallelFn.apply(x, self.w.weight.t())
```

### 4.2 Row Parallel Linear（行并行）

权重按**行**切分：$W \in \mathbb{R}^{d_{in} \times d_{out}}$ → 每个 rank 持有 $W_i \in \mathbb{R}^{(d_{in}/N) \times d_{out}}$。

- **前向**：输入按列切分（每个 rank 拿 $X_i$），局部计算后 **All-Reduce 输出**
- **反向**：$\frac{\partial L}{\partial W_i}$ 无需通信（因为各 rank 只维护自己的分片权重）

```python
class _RowParallelFn(autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(w, x)
        out = x @ w
        dist.all_reduce(out, dist.ReduceOp.SUM)  # 前向 reduce 输出
        return out

    @staticmethod
    def backward(ctx, grad_out):
        w, x = ctx.saved_tensors
        grad_x = grad_out @ w.t()       # 各 rank 只需本地梯度
        grad_w = x.transpose(-2, -1) @ grad_out
        return grad_x, grad_w

class RowParallelLinear(nn.Module):
    def __init__(self, d_in, d_out, rank=0, world_size=1):
        super().__init__()
        self.w = nn.Linear(d_in // world_size, d_out, bias=False)

    def forward(self, x):
        return _RowParallelFn.apply(x, self.w.weight.t())
```

### 4.3 Transformer 中的组合方式

在 Transformer 的 MLP 和 Attention 中，通常**先列并行、后行并行**，这样两次通信刚好"对消"：

```
MLP:  X -> [ColParallel: W1] -> GeLU -> [RowParallel: W2] -> Y
          broadcast X                     All-Reduce Y

Attn: X -> [ColParallel: WQ,WK,WV] -> Attention -> [RowParallel: WO] -> Y
          broadcast X                               All-Reduce Y
```

::: tip Megatron-LM 的设计智慧
列并行 + 行并行的组合使得整个 MLP/Attention 块只需要**一次前向 All-Reduce 和一次反向 All-Reduce**，通信量最小化。
:::

::: details 苏格拉底时刻
1. 列并行为什么在反向时需要 All-Reduce $dX$？画出矩阵乘法的计算图来推导。
2. GQA (Grouped Query Attention) 中 KV head 数量少于 Q head，张量并行如何处理 KV 的分组？
3. 张量并行的 world_size 通常不超过 8（单机 NVLink 内），为什么跨机做张量并行效率很差？
:::

---

## 5. 流水线并行从零实现

流水线并行把**不同层**分配到不同 GPU，数据像流水线一样依次通过各个 stage。

### 5.1 朴素版：逐 Stage 前向 + 反向

```python
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 4, bias=False)
        self.w2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        return self.w2(self.w1(x))

class PipeModel(nn.Module):
    """每个 rank 只持有 num_blocks/world_size 层"""
    def __init__(self, dim, num_blocks):
        super().__init__()
        self.layers = nn.ModuleList([MLP(dim) for _ in range(num_blocks)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        return x
```

朴素 Pipeline 的前向和反向：

```python
def naive_pipeline(rank, world_size):
    # ... init_process_group ...
    model = PipeModel(dim=128, num_blocks=8)  # 每个 rank 执行 2 层

    # Forward: rank 0 -> 1 -> 2 -> 3
    if rank == 0:
        out = model(x)
        dist.send(out, dst=1)
    elif rank == world_size - 1:
        dist.recv(x, src=rank - 1)
        out = model(x)
    else:
        dist.recv(x, src=rank - 1)
        out = model(x)
        dist.send(out, dst=rank + 1)

    # Backward: rank 3 -> 2 -> 1 -> 0
    if rank == world_size - 1:
        loss.backward()
        dist.send(x.grad, dst=rank - 1)
    elif rank == 0:
        dist.recv(grad, src=rank + 1)
        out.backward(gradient=grad)
    else:
        dist.recv(grad, src=rank + 1)
        out.backward(gradient=grad)
        dist.send(x.grad, dst=rank - 1)
```

::: warning Bubble 问题
朴素 Pipeline 的 bubble 率为 $(N-1)/N$，其中 $N$ 是 stage 数——大部分时间 GPU 在空等！
:::

### 5.2 GPipe：Micro-Batch 减少 Bubble

GPipe 的核心改进：把 batch 切成 $M$ 个 micro-batch，前向全部做完再统一反向。

```python
def gpipe_pipeline(rank, world_size):
    # ... init ...
    model = PipeModel(dim=128, num_blocks=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    micro_batches = world_size  # M 个 micro-batch
    x_list = list(torch.chunk(x, micro_batches, dim=0))
    stage_outputs = [None] * micro_batches

    # ---- Forward: 所有 micro-batch 依次流过 ----
    reqs = []
    for i in range(micro_batches):
        if rank != 0:
            dist.recv(x_list[i], src=rank - 1)          # 阻塞接收

        x_list[i].retain_grad()
        stage_outputs[i] = model(x_list[i])

        if rank != world_size - 1:
            req = dist.isend(stage_outputs[i].clone(), dst=rank + 1)  # 异步发送
            reqs.append(req)

    for req in reqs:
        req.wait()
    dist.barrier()

    # ---- Backward: 逆序处理 micro-batch（梯度自动累积） ----
    reqs = []
    for i in range(micro_batches - 1, -1, -1):
        if rank == world_size - 1:
            loss = loss_fn(stage_outputs[i], label_list[i])
            loss /= micro_batches  # 梯度累积需要取平均
            loss.backward()
        else:
            dist.recv(grad_list[i], src=rank + 1)
            stage_outputs[i].backward(gradient=grad_list[i])

        if rank != 0:
            req = dist.isend(x_list[i].grad.clone(), dst=rank - 1)
            reqs.append(req)

    for req in reqs:
        req.wait()
    dist.barrier()

    optimizer.step()  # 累积完所有 micro-batch 的梯度后统一更新
```

::: tip Bubble 率对比
| 方案 | Bubble 率 |
|:---:|:---:|
| 朴素 Pipeline | $(N-1)/N$ |
| GPipe ($M$ micro-batch) | $(N-1)/(M+N-1)$ |
| 1F1B (PipeDream) | 约 $(N-1)/(M+N-1)$ 但显存更优 |
| Zero Bubble | 接近 0 |
:::

::: details 苏格拉底时刻
1. GPipe 为什么要先做完所有 micro-batch 的前向，再统一做反向？如果交错（1F1B）有什么好处？
2. 梯度累积时为什么 loss 要除以 `micro_batches`？不除会怎样？
3. 流水线并行通常与张量并行搭配使用——Megatron-LM 中它们分别沿哪个维度切分？
:::

---

## 6. Ring Attention / 序列并行

当序列长度极长时（128K+），单卡放不下完整的 Attention 矩阵。Ring Attention 的做法：**每个 rank 持有一段 Q，KV 在 ring 上环形传递**。

### 6.1 核心思想

```
Rank 0:  Q0 固定, 依次与 KV0, KV1, KV2, KV3 计算 Attention
Rank 1:  Q1 固定, 依次与 KV1, KV2, KV3, KV0 计算 Attention
...
每一步: KV 沿 ring 发送到下一个 rank
```

关键：使用 **Online Softmax**（Flash Attention V2 的思想），无需一次性看到所有 KV 就能正确计算 Attention。

### 6.2 简化实现

```python
import math

class RingAttention:
    def __init__(self, dim, heads, rank, world_size):
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.rank, self.world_size = rank, world_size

    def block_attention(self, Q, K, V, L, M, O):
        """Flash Attention V2 风格的 block-wise 增量计算"""
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        M_local = S.max(dim=-1, keepdim=True).values
        M_new = torch.maximum(M, M_local)

        L_local = torch.exp(S - M_new).sum(dim=-1, keepdim=True)
        L_new = L * torch.exp(M - M_new) + L_local
        O_new = O * torch.exp(M - M_new) + torch.exp(S - M_new) @ V

        return L_new, M_new, O_new

    def ring_send_recv_kv(self, K, V):
        """环形传输 KV 到下一个 rank"""
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size
        tmp_K, tmp_V = torch.zeros_like(K), torch.zeros_like(V)

        # 奇偶 rank 交替发送/接收，避免死锁
        if self.rank % 2 == 0:
            dist.send(K, dst=next_rank)
            dist.recv(tmp_K, src=prev_rank)
        else:
            dist.recv(tmp_K, src=prev_rank)
            dist.send(K, dst=next_rank)

        if self.rank % 2 == 0:
            dist.send(V, dst=next_rank)
            dist.recv(tmp_V, src=prev_rank)
        else:
            dist.recv(tmp_V, src=prev_rank)
            dist.send(V, dst=next_rank)

        return tmp_K, tmp_V

    def forward(self, Q, K, V):
        """Ring Attention 前向：Q 固定，KV 环形传递"""
        bs, heads, q_len, d = Q.shape

        L = torch.zeros(bs, heads, q_len, 1)
        M = torch.full((bs, heads, q_len, 1), -1e4)
        O = torch.zeros(bs, heads, q_len, d)

        for step in range(self.world_size):
            L, M, O = self.block_attention(Q, K, V, L, M, O)
            if step < self.world_size - 1:  # 最后一步不用传
                K, V = self.ring_send_recv_kv(K, V)

        O = O / L  # 最终 rescale
        return O
```

::: tip 与 Flash Attention 的关系
Ring Attention 的每个 "block" 计算就是一次 Flash Attention V2 的 tile 计算。Online Softmax 的 $(L, M)$ 状态保证了分块计算的数值等价性。
:::

::: details 苏格拉底时刻
1. Ring Attention 的通信与计算可以 overlap 吗？如何用 CUDA stream 实现？
2. 如果引入 RoPE 位置编码，KV 块在环形传递时位置信息需要如何处理？
3. GQA 场景下 KV head 更少，Ring Attention 的通信量会如何变化？
4. Ring Attention 与 Ulysses（序列并行的另一种方案，基于 All-to-All）各有什么优缺点？
:::

---

## 面试高频考点

::: tip 面试清单
1. **All-Reduce vs Reduce-Scatter + All-Gather**：画图说明两者的通信模式和数据流向。
2. **DDP 梯度分桶**：为什么分桶能提升效率？桶大小如何影响 latency/throughput 的 trade-off？
3. **ZeRO 三个 Stage 的显存公式**：给定模型参数量 $\Psi$ 和 GPU 数 $N$，写出各 Stage 的显存占用。
4. **张量并行的通信在哪里**：分别说明列并行和行并行在前向/反向中各需要几次什么通信。
5. **Pipeline Bubble 率**：推导 GPipe 和 1F1B 的 bubble 率公式。
6. **Ring Attention 的正确性**：为什么 Online Softmax 能保证分块计算与全局 Softmax 等价？
7. **混合并行策略**：一个 70B 模型在 64 张 A100 上训练，你会怎么组合 TP/PP/DP/ZeRO？说明理由。
:::

---

## 推荐资源

| 资源 | 说明 |
|:---|:---|
| [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | 官方 DDP 教程 |
| [ZeRO 论文](https://arxiv.org/abs/1910.02054) | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models |
| [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) | 张量并行的奠基性工作 |
| [GPipe 论文](https://arxiv.org/abs/1811.06965) | 微批次流水线并行 |
| [Ring Attention 论文](https://arxiv.org/abs/2310.01889) | Ring Attention with Blockwise Transformers |
| [Flash Attention V2 论文](https://arxiv.org/abs/2307.08691) | Online Softmax + IO-aware Attention |
| [DeepSpeed ZeRO 文档](https://www.deepspeed.ai/tutorials/zero/) | ZeRO 官方实现与调参指南 |
| [Lilian Weng - Large Transformer Model Training](https://lilianweng.github.io/posts/2021-09-25-train-large/) | 综合性博客 |
