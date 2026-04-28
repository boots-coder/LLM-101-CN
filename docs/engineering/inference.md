---
title: "推理优化"
description: "KV Cache、PagedAttention、Continuous Batching、Chunked Prefill、投机采样（EAGLE-3/SpecExit）、PD 分离、vLLM、端侧推理"
topics: [inference, KV-cache, PagedAttention, continuous-batching, chunked-prefill, speculative-decoding, EAGLE-3, SpecExit, vLLM, PD-disaggregation, edge-inference, mobile-inference]
prereqs: [architecture/attention]
---
# 推理优化

::: info 一句话总结
推理优化让大模型从实验室走向生产环境——通过 KV Cache、PagedAttention、Continuous Batching、Chunked Prefill、投机采样、PD 分离等技术，将推理延迟从秒级降到毫秒级，将吞吐量提升数十倍；端侧推理引擎更将 LLM 能力延伸到手机和边缘设备。
:::


## 在大模型体系中的位置

推理优化处于大模型工程化的最后一公里：

```
预训练 → SFT/RLHF → 量化/蒸馏 → 【推理优化】 → 上线服务
                                      ↑
                               你在这里
```

无论模型训练得多好，如果推理效率低下，用户就会面临高延迟、高成本的困境。推理优化的核心目标是：

- **降低延迟（Latency）**：用户等待第一个 token 的时间（TTFT）和每个 token 的生成时间（TPOT）
- **提升吞吐（Throughput）**：单位时间内处理的 token 数量
- **降低成本（Cost）**：最大化 GPU 利用率，减少所需 GPU 数量

推理优化涉及的技术层次：

| 层次 | 技术 | 目标 |
|------|------|------|
| 算法层 | KV Cache、投机采样 | 减少冗余计算 |
| 内存层 | PagedAttention、KV Cache 管理 | 提升显存利用率 |
| 调度层 | Continuous Batching、Chunked Prefill | 提升 GPU 利用率 |
| 架构层 | PD 分离、vLLM | 系统级优化 |
| 内核层 | FlashAttention、融合算子 | 底层计算加速 |

---

## LLM 推理的两个阶段

LLM 的自回归推理天然分为两个性质截然不同的阶段：

### Prefill 阶段（计算密集）

Prefill 阶段处理用户输入的完整 prompt，一次性计算所有输入 token 的 KV Cache。

**特征**：

- 输入：完整 prompt（可能有数千个 token）
- 计算模式：大矩阵乘法，可以高度并行
- 瓶颈：**计算密集（Compute-bound）**，GPU 算力是限制因素
- 类比：一次性读完一篇文章

### Decode 阶段（访存密集）

Decode 阶段逐个生成输出 token，每一步都需要读取全部模型参数和之前所有 token 的 KV Cache。

**特征**：

- 输入：单个 token（上一步的输出）
- 计算模式：向量-矩阵乘法，计算量小但内存读取量大
- 瓶颈：**访存密集（Memory-bound）**，显存带宽是限制因素
- 类比：一个字一个字地写回复，每写一个字都要回头看全部内容

### 用 Roofline Model 分析为什么 Decode 是 Memory-Bound

**Roofline Model** 的核心指标是**算术强度（Arithmetic Intensity）**：

$$\text{算术强度} = \frac{\text{FLOPs（计算量）}}{\text{Bytes（访存量）}}$$

以 **Llama 70B** 为例，分析 Decode 阶段生成单个 token 的情况：

**计算量估算**：

- 模型参数量：70B（700 亿参数）
- 每个参数参与 2 次浮点运算（一次乘法 + 一次加法）
- **FLOPs = 2 x 70B = 140 GFLOPs**

**访存量估算**：

- FP16 精度下，每个参数占 2 Bytes
- 需要读取全部模型参数：70B x 2 = **140 GB**
- 还需要读取 KV Cache（随序列长度增长）

**算术强度**：

$$\text{算术强度} = \frac{140 \text{ GFLOPs}}{140 \text{ GB}} = 1 \text{ FLOP/Byte}$$

而 A100 GPU 的计算/带宽比值：

$$\frac{312 \text{ TFLOPS}}{2 \text{ TB/s}} = 156 \text{ FLOPs/Byte}$$

**结论**：Decode 阶段的算术强度（1）远低于 GPU 的计算带宽比（156），这意味着 GPU 的计算单元大部分时间在等待数据从显存搬运过来，算力严重浪费。这就是为什么 Decode 阶段是 **Memory-bound** 的。

---

## KV Cache

### 为什么需要 KV Cache

在自回归生成中，每个新 token 需要和之前所有 token 做 Attention 计算。如果每次都重新计算所有 token 的 K、V，就会产生大量冗余计算：

```python
# 不使用 KV Cache（每次重新计算）—— O(n^2) 总计算量
for i in range(max_tokens):
    logits = model(all_tokens[:i+1])  # 每次传入全部 token，重复计算前面的 K、V
    next_token = sample(logits)

# 使用 KV Cache（增量计算）—— O(n) 总计算量
kv_cache = None
for i in range(max_tokens):
    logits, kv_cache = model(tokens[i:i+1], past_kv=kv_cache)  # 只传入新 token
    next_token = sample(logits)
```

**没有 KV Cache 时**，生成第 $n$ 个 token 需要对前面所有 token 重新计算 K、V，生成 $N$ 个 token 的总计算量与 $N^2$ 成正比。**有了 KV Cache**，每步只需计算当前 token 的 Q、K、V，然后拼接到缓存中，总计算量与 $N$ 成线性关系。

### KV Cache 的显存计算

KV Cache 的显存公式：

$$\text{KV Cache 显存} = 2 \times L \times H \times d_h \times S \times B \times \text{dtype\_size}$$

其中：

| 符号 | 含义 | Llama 70B 取值 |
|------|------|----------------|
| $2$ | K 和 V 两个矩阵 | 2 |
| $L$ | Transformer 层数 | 80 |
| $H$ | 注意力头数（KV头数，GQA） | 8（GQA） |
| $d_h$ | 每个头的维度 | 128 |
| $S$ | 序列长度 | 4096 |
| $B$ | Batch Size | 1 |
| $\text{dtype\_size}$ | 数据类型字节数（FP16=2） | 2 |

**Llama 70B 在 4K context、batch_size=1 下的 KV Cache**：

$$2 \times 80 \times 8 \times 128 \times 4096 \times 1 \times 2 = 1,073,741,824 \text{ Bytes} \approx 1 \text{ GB}$$

如果使用完整的 64 个注意力头（非 GQA）：

$$2 \times 80 \times 64 \times 128 \times 4096 \times 1 \times 2 \approx 8 \text{ GB}$$

当 batch_size 增大到 32 时，KV Cache 直接膨胀到 **~32 GB**，和模型参数本身（140 GB FP16）相比已经非常可观。这就是为什么 KV Cache 的管理如此重要。

---

## PagedAttention

### 传统 KV Cache 的内存碎片问题

传统方式为每个请求预分配一块**连续的**、**最大长度**的 KV Cache 空间。这带来两个问题：

1. **内部碎片（Internal Fragmentation）**：请求实际生成长度通常远小于预分配的最大长度，大量显存被浪费
2. **外部碎片（External Fragmentation）**：不同请求占用的连续空间大小不一，释放后留下的空隙无法被其他请求利用

实测表明，传统方案的 KV Cache 显存利用率仅为 **20-40%**。

### 虚拟内存的启发

vLLM 团队从操作系统的虚拟内存管理中获得灵感：

| 操作系统概念 | PagedAttention 对应 |
|-------------|---------------------|
| 虚拟页 | 逻辑块（Logical Block） |
| 物理页帧 | 物理块（Physical Block） |
| 页表 | Block Table |
| 按需分页 | 动态分配 KV Cache Block |
| Copy-on-Write | Beam Search 共享前缀 |

### 物理块与逻辑块

核心思想：将 KV Cache 切分为固定大小的 **Block**（如每个 Block 存储 16 个 token 的 KV），通过 Block Table 维护逻辑到物理的映射。

```
逻辑视图（请求看到的连续空间）:
Request 1: [Block 0] [Block 1] [Block 2]
Request 2: [Block 0] [Block 1]

物理存储（实际的 GPU 显存）:
[Page 5] [Page 2] [Page 7] [Page 1] [Page 3] ... [Page N]

Block Table（映射关系）:
Request 1: 逻辑 0→物理 5, 逻辑 1→物理 7, 逻辑 2→物理 2
Request 2: 逻辑 0→物理 1, 逻辑 1→物理 3
```

页分配器的实现：

```python
from collections import deque

class PageAllocator:
    """物理页帧的分配与回收，维护空闲池和链式索引"""

    def __init__(self, page_size: int, total_pages: int):
        self.page_size = page_size
        self.total_pages = total_pages
        self.free_pool = deque(range(total_pages))  # 用 deque 实现 O(1) 双端操作
        self.in_use = set()
        self.slot_offset = [0] * total_pages   # 每页已填充的 token 数
        self.successor = [-1] * total_pages    # 页链表：当前页 → 下一页

    def acquire(self, count: int, prev_page: int = -1) -> list[int]:
        """从空闲池取出 count 个页帧，可选地链接到 prev_page 之后"""
        if len(self.free_pool) < count:
            return []
        grabbed = [self.free_pool.popleft() for _ in range(count)]
        self.in_use.update(grabbed)
        for pid in grabbed:
            self.slot_offset[pid] = 0
            self.successor[pid] = -1
        if prev_page >= 0:
            self.successor[prev_page] = grabbed[0]
        return grabbed

    def release(self, page_ids: list[int]):
        """将页帧归还空闲池，重置元数据"""
        for pid in page_ids:
            if pid in self.in_use:
                self.in_use.discard(pid)
                self.free_pool.append(pid)
                self.slot_offset[pid] = 0
                self.successor[pid] = -1
```

### PagedAttention 内核实现

PagedAttention 的关键在于：**KV Cache 在物理上是非连续的，但 Attention 计算需要正确地跨 Block 聚合结果**。

传统 Attention 的输入形状是 `[batch_size, seq_len, num_heads, head_dim]`，而 PagedAttention 的 KV Cache 输入是 `[num_pages, page_size, num_heads, head_dim]`——按页组织。

**Prefill 阶段的 PageAttention**：

多个请求的 Block 混合排列，通过 `request_num_pages` 记录每个请求占用的页数：

PagedAttention 在 prefill 阶段的概念伪代码（按请求分段循环）：

```python
# 输入张量按 [num_pages, page_size, dim] 组织，每个请求占据若干连续页
# pages_per_request = [3, 2] 表示请求 0 占 3 页，请求 1 占 2 页

def paged_prefill(q_proj, k_proj, v_proj, o_proj,
                  hidden_states, pages_per_request,
                  num_heads, head_dim, attention_backend):
    num_pages, page_size, _ = hidden_states.shape
    q = q_proj(hidden_states).view(num_pages, page_size, num_heads, head_dim)
    k = k_proj(hidden_states).view(num_pages, page_size, num_heads, head_dim)
    v = v_proj(hidden_states).view(num_pages, page_size, num_heads, head_dim)

    outputs = []
    cursor = 0
    for n_pages in pages_per_request:
        q_req = q[cursor : cursor + n_pages]
        k_req = k[cursor : cursor + n_pages]
        v_req = v[cursor : cursor + n_pages]
        outputs.append(attention_backend(q_req, k_req, v_req))
        cursor += n_pages

    out = torch.cat(outputs, dim=0).reshape(num_pages, page_size, num_heads * head_dim)
    return o_proj(out), (k, v)
```

实际生产实现请参阅 [vLLM 源码](https://github.com/vllm-project/vllm)。

**Decode 阶段的 PageAttention（核心难点）**：

Decode 时每个请求只有 1 个新 query token，但需要和分散在多个 Page 上的 KV Cache 做 Attention。核心思路是：

1. 将 query 复制到对应的每个 KV Page 上（dispatch）
2. 在每个 Page 上独立计算局部 Attention
3. 通过 **Online Softmax** 聚合所有 Page 的结果（reduce/combine）

```python
# Decode 阶段：单个 query 对 N 页 KV Cache 的聚合
q = torch.randn(1, 1, d)          # 1 个 query token
K = torch.randn(num_pages, page_size, d)  # KV Cache 分散在多个页上
V = torch.randn(num_pages, page_size, d)

# 每页独立计算局部 Attention
S = q @ K.transpose(1, 2)                          # [num_pages, 1, page_size]
M, _ = torch.max(S, dim=-1, keepdim=True)           # 局部最大值
L = torch.sum(torch.exp(S - M), dim=-1, keepdim=True)  # 局部归一化因子
P = torch.exp(S - M) / L
O = P @ V                                           # 局部输出

# 通过 Online Softmax 聚合多页结果
def combine_result(O, L, M):
    """将多页的局部 Attention 结果聚合为全局正确结果"""
    M_new, _ = torch.max(M, dim=0, keepdim=True)     # 全局最大值
    L_new = torch.exp(M - M_new) * L
    L_new = torch.sum(L_new, dim=0, keepdim=True)     # 全局归一化因子
    O_new = torch.exp(M - M_new) * (L / L_new) * O   # 重新缩放
    return O_new.sum(dim=0, keepdim=True)

O_final = combine_result(O, L, M)
```

这个 Online Softmax 聚合的数学保证是：全局 Softmax 可以分解为局部 Softmax 的加权组合，通过维护全局最大值 $M$ 和归一化因子 $L$ 实现精确（非近似）计算。

### Copy-on-Write 优化（Beam Search 场景）

在 Beam Search 中，多个候选序列共享相同的前缀。传统方式需要为每个候选复制一份完整的 KV Cache，而 PagedAttention 的 Copy-on-Write 机制允许：

- 多个候选序列共享前缀 Block，引用计数 > 1
- 只有当某个候选需要修改某个 Block 时，才复制该 Block
- 显存节省高达 **55%**（vLLM 论文数据）

---

## Continuous Batching

### Static Batching 的低效

传统的 Static Batching 将一批请求打包，等所有请求都生成完毕后才处理下一批：

```
时间 →
请求 A: [====生成 10 tokens====][等待...........][等待...........]
请求 B: [====生成 10 tokens====][====生成 30 tokens====][等待...........]
请求 C: [====生成 10 tokens====][====生成 30 tokens====][====生成 50 tokens====]
                                                         ↑
                                               A 和 B 早已完成，GPU 在空转
```

**问题**：短请求完成后 GPU 在"陪跑"长请求，资源严重浪费。

### Continuous Batching 的调度策略

Continuous Batching（也叫 **Iteration-level Scheduling**）在每次生成一个 token 后重新调度：

```
时间 →
Step 1: [A][B][C]      ← 3 个请求并行生成
Step 2: [A][B][C]
...
Step 10: [A完成][B][C]  ← A 完成，立即释放位置
Step 11: [D加入][B][C]  ← 新请求 D 立即插入
...
Step 30: [D][B完成][C]  ← B 完成
Step 31: [D][E加入][C]  ← 新请求 E 插入
```

核心实现是一个 **prefill / decode 两态调度器**——nano-vllm 用极简的 70 行就把 vLLM 的核心调度思想说清楚了（参考 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) `nanovllm/engine/scheduler.py`，与 vLLM v1 `vllm/v1/core/sched/scheduler.py` 同构）：

```python
from collections import deque

class Scheduler:
    """nano-vllm 风格的 continuous batching 调度器"""

    def __init__(self, config):
        self.max_num_seqs           = config.max_num_seqs           # 同时运行的最大序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 单步最多处理的 token 数
        self.eos = config.eos
        self.block_manager = BlockManager(...)   # 管 KV cache 物理块

        # 两个核心队列：waiting（待 prefill）+ running（已 prefill 在 decode）
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence):
        """新请求进入 waiting"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """每个 step 调用一次，返回这一步要处理的序列 + 是否是 prefill"""

        # ─── 优先调度 prefill：尽量塞满 max_num_batched_tokens ───
        scheduled = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            # 两个准入条件：token 预算够 + KV cache 物理块够
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
                    or not self.block_manager.can_allocate(seq)):
                break
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            self.waiting.popleft()
            self.running.append(seq)
            scheduled.append(seq)
        if scheduled:
            return scheduled, True   # is_prefill = True

        # ─── 否则调度 decode：把 running 里能继续生成的都 batch 起来 ───
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            # 如果新 token 没物理块装下，必要时抢占低优先级 sequence
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())  # 把队尾换出
                else:
                    self.preempt(seq); break          # 自己被换出
            else:
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        # 调度过的 running seq 放回队首，保持公平
        self.running.extendleft(reversed(scheduled))
        return scheduled, False  # is_prefill = False

    def preempt(self, seq):
        """显存不够时把序列踢回 waiting，下次重新 prefill"""
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```

::: tip 三个值得记住的设计选择
1. **prefill 优先**：每次 step 先看 `waiting` 队列能否再塞 prefill；只有塞不进了才退回到 decode。这样能尽快把新请求"接进系统"，对延迟友好。
2. **token 预算 + 物理块预算双约束**：`max_num_batched_tokens` 控制单步显存峰值，`block_manager.can_allocate` 防止 KV cache OOM。
3. **抢占（preempt）而不是排队**：显存吃紧时，新 token 装不下就把 `running` 队尾的序列"踢回 waiting"。被踢回的请求下次重新走 prefill——这是 vLLM "RecomputePreemption" 策略的最简实现。
:::

**KV Cache 管理**（槽位式）：

```python
class KVCacheManager:
    def __init__(self, config):
        self.kv_cache = torch.zeros(
            2, config.num_layers, config.max_batch_size,
            config.max_seq_len, config.num_heads, config.head_dim
        )
        self.slot_to_request = {}
        self.request_to_slot = {}
        self.available_slots = set(range(config.max_batch_size))

    def allocate_slots(self, request_ids):
        """为新请求分配 KV Cache 槽位"""
        allocated = []
        for req_id in request_ids:
            slot = self.available_slots.pop()
            self.slot_to_request[slot] = req_id
            self.request_to_slot[req_id] = slot
            allocated.append(slot)
        return allocated

    def free_slot(self, request_id):
        """释放已完成请求的槽位"""
        slot = self.request_to_slot.pop(request_id)
        del self.slot_to_request[slot]
        self.kv_cache[:, :, slot, :, :, :] = 0  # 清空 KV Cache
        self.available_slots.add(slot)
```

Continuous Batching 将吞吐量提升 **2-8 倍**，是现代推理引擎的标配。

---

## Chunked Prefill

### 长 Prefill 阻塞 Decode 的问题

当系统同时服务多个用户时，新到达的长 prompt（如 4K token）的 Prefill 会长时间独占 GPU，导致已经在 Decode 阶段的请求被阻塞，用户感受到明显的卡顿（TPOT 突增）。

```
不使用 Chunked Prefill:
[Decode R1][Decode R2][========= Prefill R3 (4K tokens) =========][Decode R1][Decode R2]
                       ↑ R1、R2 的 Decode 被阻塞了很久

使用 Chunked Prefill:
[Decode R1][Decode R2][Prefill R3 chunk1][Decode R1][Decode R2][Prefill R3 chunk2][Decode R1]...
                                          ↑ R1、R2 的 Decode 不被打断
```

### 将 Prefill 拆分为 Chunks

核心思想：将长 prompt 按固定大小（如 512 token）切分为多个 chunk，每个 chunk 和当前的 Decode 请求合并成一个 batch 执行。

**Chunked Prefill 的注意力计算**需要处理跨 chunk 的 KV Cache 拼接。下面用最朴素的 PyTorch 写法演示语义，命名沿用 HuggingFace `LlamaAttention` 的 `q_proj/k_proj/v_proj/o_proj`：

```python
# 教学示意：vLLM 真正的实现走的是 PagedAttention CUDA Kernel + KVCacheManager，
# 见 https://github.com/vllm-project/vllm/tree/main/vllm/attention
class ChunkAttention(nn.Module):
    def forward(self, x, kv_cache=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=1)  # 拼接历史 KV 和当前 chunk
            v = torch.cat([v_past, v], dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn), (k, v)  # 返回更新后的 KV Cache
```

**Chunked Prefill 的调度逻辑**：

```python
def chunk_prefill_method(model, x, page_size, dim):
    """将长 prompt 分 chunk 处理"""
    bsz, seq_len = x.shape
    num_chunks = (seq_len + page_size - 1) // page_size
    KVCache = None

    for i in range(num_chunks):
        start = i * page_size
        end = min((i + 1) * page_size, seq_len)
        chunk = x[:, start:end]

        logits, KVCache = model(chunk, KVCache=KVCache)

        if i == num_chunks - 1:  # 最后一个 chunk
            last_token_logits = logits[:, -1, :]
        else:
            last_token_logits = None  # 非最后一个 chunk 的 logits 无效

    return last_token_logits, KVCache
```

**混合 PD 请求的 Chunked Prefill 调度器**——vLLM V1 真正的做法（参考 [vllm-project/vllm](https://github.com/vllm-project/vllm) `vllm/v1/core/sched/scheduler.py` `Scheduler.schedule()`）是 **"以单一 token budget 同时容纳 decode 和 prefill"**，下面是简化骨架：

```python
def schedule(self):
    """vLLM v1 schedule()：一个 token budget 统一管 decode 和 chunked prefill"""
    scheduled_new_tokens = {}                          # req_id -> 本步分配的 token 数
    token_budget = self.max_num_scheduled_tokens       # 单步总预算

    # ── 阶段 1：先把已经在 running 里的请求安排上 ──
    #   - decode 请求消耗 1 token
    #   - chunked prefill 请求按"剩余 prompt"或 long_prefill_threshold 取 min
    for req in self.running:
        if token_budget <= 0:
            break
        num_new = req.num_tokens_with_spec - req.num_computed_tokens
        # 单条 prefill 一次最多 long_prefill_token_threshold（防止一条独吞 budget）
        if 0 < self.long_prefill_threshold < num_new:
            num_new = self.long_prefill_threshold
        num_new = min(num_new, token_budget)

        # KV cache 块够不够？不够就 preempt 队尾让位
        while not self.kv_cache_manager.can_allocate(req, num_new):
            preempted = self.running.pop()
            self.preempt(preempted)
            token_budget += scheduled_new_tokens.pop(preempted.req_id, 0)

        self.kv_cache_manager.allocate_slots(req, num_new)
        scheduled_new_tokens[req.req_id] = num_new
        token_budget -= num_new

    # ── 阶段 2：从 waiting 取新请求加入 running，继续吃 budget ──
    while self.waiting and token_budget > 0:
        req = self.waiting[0]
        num_new = min(len(req.prompt_token_ids), token_budget)
        if not self.kv_cache_manager.can_allocate(req, num_new):
            break
        self.kv_cache_manager.allocate_slots(req, num_new)
        scheduled_new_tokens[req.req_id] = num_new
        token_budget -= num_new
        self.running.append(self.waiting.popleft())

    return SchedulerOutput(scheduled_new_tokens=scheduled_new_tokens)
```

**关键设计原则**——vLLM v1 把 V0 的"prefill_batch / decode_batch 分开管"统一到单 budget：

1. **Token Budget 统一管控**：一个 `max_num_scheduled_tokens` 同时约束 decode（每条 1 token）+ prefill（每条若干 token），不再区分 prefill_batch / decode_batch 两个独立上限
2. **Long Prefill Threshold**：单条请求一次最多 prefill `long_prefill_threshold` 个 token，防止一条 16K prompt 把整个 budget 独吞
3. **preempt-on-OOM**：KV cache 装不下时把 running 队尾踢回 waiting（调度回退而非阻塞），下次 `schedule()` 再重新 prefill

---

## 投机采样 (Speculative Decoding)

### 核心思想：小模型草稿 + 大模型验证

标准的自回归解码每步只能生成 1 个 token，而每步都需要完整的前向传播。投机采样的核心观察是：

> Decode 阶段是 Memory-bound，增加少量计算（验证多个 token）几乎不增加时间。

因此可以用一个**小模型**（Draft Model，如 7B）快速猜测 $K$ 个 token，然后用**大模型**（Target Model，如 70B）一次前向传播同时验证这 $K$ 个 token。

```python
class SPDecoding:
    def __init__(self, model_target, model_draft, spec_n):
        self.model_target = model_target
        self.model_draft = model_draft
        self.spec_n = spec_n  # 每次猜测的 token 数，如 5

    def generate_draft(self, spec_n, x):
        """小模型自回归生成 spec_n 个候选 token"""
        logits_y = []
        for i in range(spec_n):
            with torch.no_grad():
                logits = self.model_draft(x)[:, [-1], :]
                logits_y.append(logits)
                next_token = torch.argmax(logits, dim=-1)
                x = torch.cat([x, next_token], dim=1)
        return x, torch.cat(logits_y, dim=1)

    def generate(self, x, max_new_tokens=30):
        count = 0
        y_new = []
        for i in range(max_new_tokens):
            # Step 1: 小模型猜测
            x_spec, logits_draft = self.generate_draft(self.spec_n, x)
            y_spec = x_spec[:, -self.spec_n:]

            # Step 2: 大模型一次性验证
            logits_target = self.model_target(x_spec)[:, -self.spec_n-1:]
            y_target = torch.argmax(logits_target, dim=-1)

            # Step 3: 从左到右逐个验证
            y_target_verify = y_target[:, :-1]
            verify = y_spec == y_target_verify
            idx1, idx2 = torch.where(verify == False)

            if len(idx2) == 0:
                accept_len = self.spec_n  # 全部接受
            else:
                accept_len = idx2[0]      # 部分接受

            # Step 4: 更新输入
            x = torch.cat((x, y_target[:, :accept_len+1]), dim=1)
            y_new.append(y_target[:, :accept_len+1])
            count += (accept_len + 1)

            if count >= max_new_tokens - 1:
                break
        return torch.cat(y_new, dim=1)
```

### 验证算法的数学保证（为什么输出分布不变）

上面的 Greedy 版本容易理解但实际中多用**采样模式**。Speculative Sampling 的关键在于验证步骤的概率保证：

对于草稿模型的概率分布 $q(x)$ 和目标模型的概率分布 $p(x)$：

1. 以概率 $\min(1, \frac{p(x)}{q(x)})$ **接受**草稿 token $x$
2. 如果拒绝，从修正分布 $p'(x) = \text{norm}(\max(0, p(x) - q(x)))$ 重新采样

```python
class SPSamplingDecoding:
    def generate(self, x, max_new_tokens=30):
        count = 0
        y_new = []
        for i in range(max_new_tokens):
            x_spec, logits_draft = self.generate_draft(self.spec_n, x)
            y_spec = x_spec[:, -self.spec_n:]
            logits_target = self.model_target(x_spec)[:, -self.spec_n-1:]

            accept_len = 0
            next_tokens = []
            for j in range(self.spec_n):
                r = torch.rand(1).item()  # 均匀分布随机数
                token_id = y_spec[0, j]
                q = F.softmax(logits_target[0, j], dim=-1)
                p = F.softmax(logits_draft[0, j], dim=-1)

                # 核心验证：以 min(1, q/p) 的概率接受
                if r < min(1, q[token_id] / p[token_id]):
                    accept_len += 1
                    next_tokens.append(y_spec[0, j])
                else:
                    # 拒绝：从修正分布重新采样
                    q_ = q.clone()
                    idx = torch.where(q < p)
                    q_[idx] = p[idx]  # max(0, p-q) 的近似
                    next_token = torch.multinomial(q_, num_samples=1)
                    next_tokens.append(next_token)
                    break

            if accept_len == self.spec_n:
                # 全部接受，bonus：大模型额外输出 1 个 token
                next_tokens.append(y_target[0, accept_len])

            x = torch.cat((x, torch.tensor([next_tokens])), dim=-1)
            y_new.append(torch.tensor([next_tokens]))
            count += len(next_tokens)
            if count >= max_new_tokens - 1:
                break
        return torch.cat(y_new, dim=1)
```

### 接受率分析

- **最好情况**：Draft Model 和 Target Model 分布完全一致 → 全部 $K$ 个 token 被接受，加速 $K+1$ 倍
- **最差情况**：分布完全不同 → 每次只接受 0 个，退化为标准解码（还多了 Draft 的开销）
- **实际情况**：使用同系列的小模型（如 Llama-7B 作为 Llama-70B 的 Draft），接受率通常在 70-85%，加速 **2-3 倍**

**KL 散度**可以量化两个模型输出分布的差异：

```python
def KL(logits_p, logits_q):
    log_p = F.log_softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    kl = F.kl_div(log_p, q, reduction='sum', log_target=False)
    return kl

# KL 越小，接受率越高
# target vs draft (相近模型): KL ≈ 14.7
# target vs random:          KL ≈ 110.4
```

### EAGLE-3：基于特征层的树状投机采样

上面的基础投机采样有两个关键局限：

1. **Draft Model 是独立的小模型**，与 Target Model 的分布差异大，接受率有上限
2. **线性草稿生成**，每步只猜一条路径

EAGLE-3（腾讯 AngelSlim 开源）的核心创新是：**Draft Model 不再独立，而是直接利用 Target Model 的隐藏状态**，并以**树状结构**并行生成多条候选路径。

#### Draft Model 架构

EAGLE-3 的 Draft Model 非常轻量（Target Model 参数的 1/8 ~ 1/4），其核心是接收 Target Model 中间层的隐藏状态：

```
Target Model (70B)               Draft Model (Light)
┌─────────────┐                  ┌───────────────┐
│  Layer 0-31 │──hidden_states──→│ Combine Layer  │
│  Layer 32   │                  │ (Linear Proj)  │
│  ...        │                  ├───────────────┤
│  Layer N    │                  │ 2-4 Attn Layers│
└─────────────┘                  │ (Shared Embed) │
                                 ├───────────────┤
                                 │  LM Head       │
                                 │ (Vocab Pruned) │
                                 └───────────────┘
```

```python
class Eagle3DraftModel(nn.Module):
    """EAGLE-3 Draft Model 简化实现"""
    def __init__(self, target_hidden_size, draft_hidden_size, 
                 num_layers=2, num_heads=8, draft_vocab_size=8000):
        super().__init__()
        # 从 Target Model 隐藏状态投影到 Draft 空间
        self.hidden_combine = nn.Linear(target_hidden_size, draft_hidden_size)
        
        # 轻量 Transformer 层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=draft_hidden_size,
                nhead=num_heads,
                dim_feedforward=draft_hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 裁剪后的词表头（只保留高频 token）
        self.lm_head = nn.Linear(draft_hidden_size, draft_vocab_size, bias=False)
        
        # 词表映射：draft_vocab → target_vocab
        self.d2t = None  # 在训练时从数据频率统计构建
        self.t2d = None  # target_vocab → draft_vocab (布尔映射)
    
    def forward(self, target_hidden_states, input_ids, attention_mask=None):
        # Step 1: 融合 Target Model 的隐藏状态
        h = self.hidden_combine(target_hidden_states)
        
        # Step 2: 加入 token embedding
        embed = self.embed_tokens(input_ids)
        h = h + embed
        
        # Step 3: 通过轻量 Transformer 层
        for layer in self.layers:
            h = layer(h, memory=None, tgt_mask=attention_mask)
        
        # Step 4: 输出裁剪词表上的 logits
        logits = self.lm_head(h)
        return logits
```

#### 词表裁剪 (Token Pruning)

EAGLE-3 的一个关键优化是**词表裁剪**：Draft Model 不需要预测全部 128K+ 的词表，只需覆盖最常见的 token：

```python
def build_vocab_mapping(dataset, draft_vocab_size=8000, target_vocab_size=128256):
    """从训练数据的 token 频率构建词表映射"""
    from collections import Counter
    
    token_freq = Counter()
    for sample in dataset:
        token_freq.update(sample['input_ids'])
    
    # 取 Top-N 高频 token
    top_tokens = [tok for tok, _ in token_freq.most_common(draft_vocab_size)]
    top_tokens.sort()  # 保持有序
    
    # 覆盖率统计
    total = sum(token_freq.values())
    covered = sum(token_freq[t] for t in top_tokens)
    print(f"词表裁剪: {target_vocab_size} → {draft_vocab_size}")
    print(f"Token 覆盖率: {covered/total:.1%}")  # 通常 > 95%
    
    # 构建双向映射
    d2t = {i: tok for i, tok in enumerate(top_tokens)}    # draft → target
    t2d = {tok: i for i, tok in enumerate(top_tokens)}    # target → draft
    
    return d2t, t2d
```

#### 树状候选生成

与线性猜测不同，EAGLE-3 用**树结构**并行生成多条候选路径：

```
                    [token_0]
                   /    |    \
            [top1]   [top2]   [top3]     ← depth 1: 3 个分支
            / |        |        \
       [t1] [t2]    [t1]      [t1]       ← depth 2: 扩展
        |     |       |
      [t1]  [t1]   [t1]                  ← depth 3: 继续
```

```python
def tree_speculative_decode(target_model, draft_model, input_ids, 
                            tree_depth=5, top_k=10, total_tokens=60):
    """树状投机采样推理循环"""
    
    # Step 1: Prefill - Target Model 计算初始隐藏状态
    with torch.no_grad():
        target_out = target_model(input_ids, output_hidden_states=True)
        target_logits = target_out.logits[:, -1:]
        hidden_states = target_out.hidden_states  # 多层隐藏状态
    
    # Step 2: Draft Model 生成树状候选
    candidates = []  # 所有候选路径
    tree_tokens = []
    tree_positions = []
    
    # 根节点：取 Target 的 top-k 作为第一层
    probs = F.softmax(target_logits[:, -1], dim=-1)
    top_k_tokens = torch.topk(probs, k=top_k).indices[0]
    
    # 广度优先展开树
    queue = [(tok.item(), [tok.item()], 0) for tok in top_k_tokens]
    
    while queue and len(tree_tokens) < total_tokens:
        token, path, depth = queue.pop(0)
        if depth >= tree_depth:
            continue
        
        tree_tokens.append(token)
        tree_positions.append(len(input_ids[0]) + depth)
        
        # Draft Model 预测下一层
        draft_input = torch.tensor([[token]], device=input_ids.device)
        draft_logits = draft_model(hidden_states[-1][:, -1:], draft_input)
        draft_probs = F.softmax(draft_logits[:, -1], dim=-1)
        
        # 选 top-k 子节点
        next_tokens = torch.topk(draft_probs, k=min(3, top_k)).indices[0]
        for next_tok in next_tokens:
            new_path = path + [next_tok.item()]
            queue.append((next_tok.item(), new_path, depth + 1))
            candidates.append(new_path)
    
    # Step 3: Target Model 一次验证所有候选
    all_draft_tokens = torch.tensor([tree_tokens], device=input_ids.device)
    tree_mask = build_tree_attention_mask(tree_positions)  # 因果+树结构
    
    with torch.no_grad():
        verify_logits = target_model(
            all_draft_tokens, 
            attention_mask=tree_mask,
            position_ids=torch.tensor([tree_positions])
        ).logits
    
    # Step 4: 逐路径验证，找到最长接受路径
    best_path, accept_len = evaluate_tree_candidates(
        candidates, verify_logits, target_logits
    )
    
    return best_path, accept_len
```

#### Draft Model 训练流水线

EAGLE-3 的训练分两步：

**Step 1: 生成训练数据** — 用 Target Model 对语料做推理，保存每层隐藏状态：

```bash
# 1. 启动 Target Model 生成对话数据
python generate_data_for_target_model.py \
    --model_path Qwen/Qwen3-32B \
    --data_path ShareGPT_data.jsonl \
    --output_path generated_data/

# 2. 提取隐藏状态（这是关键步骤）
python generate_hidden_for_draft_model.py \
    --model_path Qwen/Qwen3-32B \
    --data_path generated_data/ \
    --output_path hidden_states/ \
    --save_all_layers True
```

**Step 2: 训练 Draft Model** — 用 KL 散度对齐 Draft 和 Target 的输出分布：

```python
def eagle3_training_step(draft_model, batch, training_steps=7):
    """EAGLE-3 训练步骤（简化版）"""
    input_ids = batch['input_ids']
    target_logits = batch['target_logits']     # 预计算的 Target 输出
    hidden_states = batch['hidden_states']      # 预计算的 Target 隐藏状态
    
    total_loss = 0
    losses = []
    
    for step in range(training_steps):
        # Draft Model 前向
        draft_logits = draft_model(hidden_states, input_ids)
        
        # Target 概率（在裁剪词表上）
        target_probs_pruned = F.softmax(
            target_logits[..., draft_model.t2d], dim=-1  # 词表裁剪
        )
        
        # KL 散度损失
        draft_log_probs = F.log_softmax(draft_logits, dim=-1)
        step_loss = -torch.sum(target_probs_pruned * draft_log_probs) / batch_size
        losses.append(step_loss)
        
        # 更新输入（自回归移位）
        input_ids = shift_tokens(input_ids, target_logits)
    
    # 指数衰减加权：越近的预测权重越大
    weights = [0.8 ** i for i in range(len(losses))]
    total_loss = sum(w * l for w, l in zip(weights, losses))
    
    return total_loss
```

#### EAGLE-3 性能数据（来自 AngelSlim 基准测试）

| 模型 | 方法 | GSM8K | Alpaca | HumanEval | MT-bench | 平均加速 |
|------|------|-------|--------|-----------|----------|---------|
| Qwen3-1.7B | 标准推理 | 376 tok/s | 379 tok/s | 378 tok/s | 391 tok/s | 1x |
| Qwen3-1.7B | EAGLE-3 | 617 tok/s | 653 tok/s | 680 tok/s | 621 tok/s | **1.68x** |
| Qwen3-8B | 标准推理 | 150 tok/s | 150 tok/s | 154 tok/s | 154 tok/s | 1x |
| Qwen3-8B | EAGLE-3 | 257 tok/s | 267 tok/s | 245 tok/s | 258 tok/s | **1.70x** |
| Qwen3-32B | 标准推理 | 43.5 tok/s | 43.4 tok/s | 43.2 tok/s | 43.3 tok/s | 1x |
| Qwen3-32B | EAGLE-3 | 80.4 tok/s | 72.5 tok/s | 71.6 tok/s | 74.1 tok/s | **1.71x** |

> 测试条件：vLLM v0.11.2, num_speculative_tokens=2, batch_size=1, output_len=1024

**关键观察**：平均接受长度约 2.5 token/step，模型越大加速比越稳定。

### SpecExit：推理模型的思考早退

对于 DeepSeek-R1、QwQ 等推理模型，生成的 token 中大量是"思考过程"（Reasoning Token），实际有用的答案只占一小部分。SpecExit（腾讯，OSDI 2025 投稿）的核心思想是：

> 用 Draft Model 的隐藏状态**预测何时可以停止思考**，无需等待完整的推理链。

#### 工作原理

```
Target Model 生成推理链:
"<think>首先分析题目... 然后计算... 验证结果...</think>答案是42"
                    ↓
SpecExit 在中间某处判断:
"<think>首先分析题目... 然后计算...</think>答案是42"
                    ↑
        此处已经可以得到正确答案，后面的思考是冗余的
```

#### 信号预测与停止判据

SpecExit 从 Draft Model 的隐藏状态中提取三类信号：

```python
class SpecExitPredictor:
    """SpecExit 早退信号预测器"""
    
    def __init__(self, method="ewma", alpha=0.3, window_size=10):
        self.method = method
        self.alpha = alpha
        self.window_size = window_size
        self.scores = []
    
    def update(self, draft_hidden_state):
        """从 Draft Model 隐藏状态提取停止信号"""
        # 信号类型1: 置信度 (confidence) — 答案 token 的概率
        # 信号类型2: 进度 (progress) — 已完成推理的比例
        # 信号类型3: 剩余量 (remain) — 预计还需要多少 token
        score = self.extract_signal(draft_hidden_state)
        self.scores.append(score)
        return self.predict_next()
    
    def predict_next(self):
        """预测下一步的信号值"""
        if self.method == "ewma":
            # 指数加权移动平均
            ewma = self.scores[0]
            for s in self.scores[1:]:
                ewma = self.alpha * s + (1 - self.alpha) * ewma
            return ewma
        
        elif self.method == "momentum":
            # 动量法：用历史趋势外推
            if len(self.scores) < 2:
                return self.scores[-1]
            deltas = [self.scores[i] - self.scores[i-1] 
                      for i in range(1, len(self.scores))]
            avg_delta = sum(deltas[-self.window_size:]) / len(deltas[-self.window_size:])
            return self.scores[-1] + avg_delta
        
        elif self.method == "mean":
            # 滑动窗口均值
            window = self.scores[-self.window_size:]
            return sum(window) / len(window)
    
    def should_exit(self, predicted_score):
        """判断是否可以提前退出"""
        if "confidence" in self.method:
            return predicted_score > 0.8   # 答案置信度足够高
        elif "progress" in self.method:
            return predicted_score > 0.3   # 推理进度超过 30%
        elif "remain" in self.method:
            return predicted_score < 200   # 预计剩余 token < 200
        return False
```

**SpecExit 效果**（在推理模型上）：
- 生成长度减少 **66%**（省去冗余思考）
- 端到端延迟降低 **2.5x**
- 答案准确率基本不变

### 投机采样方法对比

| 方法 | Draft Model | 接受率 | 加速比 | 特点 |
|------|-------------|--------|--------|------|
| 基础投机采样 | 独立小模型 | 70-85% | 2-3x | 简单，Draft 与 Target 解耦 |
| EAGLE-3 | 特征级轻量模型 | 80-90% | 1.7-1.9x | 利用 Target 隐藏状态，树状搜索 |
| Medusa | 多个 LM Head | 60-80% | 1.5-2x | 无需独立 Draft Model |
| SpecExit | EAGLE-3 + 早退 | - | 2.5x+ | 针对推理模型，减少冗余思考 |
| Self-Speculative | Target 自身浅层 | 50-70% | 1.3-1.5x | 无需额外模型 |

---

## Prefill-Decode 分离 (PD Disaggregation)

### 为什么要分离？

Prefill 是 Compute-bound，Decode 是 Memory-bound，二者对硬件的需求截然不同：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特征 | 大矩阵乘法，高算力需求 | 向量-矩阵乘法，高带宽需求 |
| 理想硬件 | 高 FLOPS 的 GPU (如 H100) | 高带宽的 GPU 或 Memory-centric 架构 |
| Batch Size | 通常较小（几个请求） | 可以很大（数百个请求） |
| 耗时 | 可预测（与 prompt 长度成正比） | 不确定（取决于生成长度） |

将 Prefill 和 Decode 放在不同的物理节点上运行，可以分别优化：

1. **Prefill 节点**：配置高算力 GPU，专注处理 prompt
2. **Decode 节点**：配置高带宽系统，专注逐 token 生成
3. **各自优化 Batch Size**：互不干扰

### 架构设计

PD 分离的基本架构：

```
                    ┌──────────────────────┐
                    │      Scheduler       │
                    │  (Ray Remote Actor)  │
                    └──────┬───────┬───────┘
                           │       │
              ┌────────────▼──┐ ┌──▼────────────┐
              │ Prefill Node  │ │ Decode Node   │
              │  (GPU Group)  │ │  (GPU Group)  │
              └──────┬────────┘ └────┬──────────┘
                     │               │
                     └───────┬───────┘
                             │
                    ┌────────▼────────┐
                    │  KV Cache Store │
                    │  (Distributed)  │
                    └─────────────────┘
```

### 关键组件职责

PD 分离架构中通常有三类 Ray Actor，各自职责清晰：

- **Scheduler**：维护两条请求队列（waiting / running），分别由 Prefill / Decode 节点拉取；负责请求生命周期管理（接收、分发、终止判定）
- **KV Cache Engine**：负责跨节点的 KV Cache 存放与传输，关键映射是 `request_id ↔ batch_id`
- **PD Engine**：调度 Prefill / Decode 两个 actor group 的执行，处理"prefill 完成 → 进入 decode"的状态转移

### KV Cache 传输策略

PD 分离的核心挑战是 **KV Cache 的传输**：Prefill 节点计算完 KV Cache 后需要传输到 Decode 节点。

三种 KV Cache 管理模式：

1. **KV Cache 仅存于 Decode 节点**：Prefill 计算后直接传输过去（实现简单）
2. **PD 各存各的**：Prefill 节点也保留 Cache，用于 Chunked-Prefill
3. **中心化 Cache 服务**：独立的 KV Cache 服务，PD 节点都从中存取

具体的 Ray actor 实现（Scheduler / KV Cache Engine / PD Engine 主循环）较为冗长，本节略去；推荐参考开源 PD 分离实现，如 [Mooncake](https://github.com/kvcache-ai/Mooncake)、[DistServe](https://github.com/LLMServe/DistServe)、以及 [vLLM](https://github.com/vllm-project/vllm) 自身的 disaggregated serving 文档。

**PD 分离 vs PD 融合**：

- PD 分离适合：大规模部署、异构硬件、高吞吐需求
- PD 融合（Chunked Prefill）适合：单节点、低延迟需求
- 实际系统通常是**分离和融合兼并**的，例如 Prefill 节点内部也可以用 Chunked Prefill

---

## vLLM 架构深度解析

vLLM 是目前最流行的开源 LLM 推理引擎，集成了上述几乎所有优化技术。

### vLLM V0 架构

V0 架构采用 **Scheduler + Worker** 的同步执行模式：

```
┌─────────────────────────────────────────────┐
│              vLLM V0 Engine                 │
├─────────────────────────────────────────────┤
│  Scheduler                                  │
│  ├── Waiting Queue（等待 Prefill 的请求）     │
│  ├── Running Queue（正在 Decode 的请求）      │
│  └── Swapped Queue（被换出到 CPU 的请求）     │
├─────────────────────────────────────────────┤
│  Block Manager (PagedAttention 显存管理)     │
│  ├── PageAllocator（逻辑块→物理块映射）        │
│  ├── GPU Block Allocator                    │
│  └── CPU Block Allocator（Swap 用）          │
├─────────────────────────────────────────────┤
│  Worker（模型执行）                           │
│  ├── Model Runner（前向传播）                 │
│  └── Cache Engine（KV Cache 物理存储）        │
└─────────────────────────────────────────────┘
```

V0 的 **KVCachePool** 实现：

```python
class KVCachePool:
    """基于页帧的 KV 缓存存储池"""

    def __init__(self, config):
        self.allocator = PageAllocator(config.page_size, config.num_pages)
        self.page_size = config.page_size

        # KV 的物理存储：按页组织
        self.kv_store = torch.zeros(
            2,                  # K 和 V
            config.num_layers,  # Transformer 层数
            config.num_pages,   # 总页数
            config.page_size,   # 每页 token 数
            config.num_heads,
            config.head_dim,
        )
        self.req_page_map = {}   # request_id -> [page_id, ...]
        self.cached_tokens = {}  # request_id -> 已缓存 token 数

    def write_tokens(self, request_id, KV):
        """将新计算的 KV 追加写入页帧，按需扩展"""
        _, L, T, H, D = KV.shape
        if request_id not in self.cached_tokens:
            # 首次写入：分配起始页
            self.cached_tokens[request_id] = 0
            initial = self.allocator.acquire(count=1)
            self.req_page_map[request_id] = initial

        remaining = T
        while remaining > 0:
            pages = self.req_page_map[request_id]
            capacity = self.page_size * len(pages) - self.cached_tokens[request_id]

            if capacity == 0:
                # 当前页已满，扩展一页
                extra = self.allocator.acquire(count=1, prev_page=pages[-1])[0]
                self.req_page_map[request_id].append(extra)
                capacity = self.page_size

            # 写入本轮数据
            n_write = min(capacity, remaining)
            dst_page = pages[-1]
            offset = self.page_size - capacity
            src_start = T - remaining
            self.kv_store[:, :, dst_page, offset:offset + n_write] = KV[:, :, src_start:src_start + n_write]
            remaining -= n_write
            self.cached_tokens[request_id] += n_write

    def free(self, request_id):
        """释放请求占用的全部页帧"""
        pages = self.req_page_map[request_id]
        self.allocator.release(pages)
        self.kv_store[:, :, pages, :, :, :] = 0
        del self.req_page_map[request_id]
```

### vLLM V1 架构演进

V1 的核心改进是**消除 Prefill/Decode 的概念区分**，统一为 Chunked Prefill：

```
┌──────────────────────────────────────────────────────┐
│                 vLLM V1 EngineCore                   │
├──────────────────────────────────────────────────────┤
│  Scheduler（Chunked Prefill 统一调度）                │
│  ├── Decode 请求优先并入 batch                        │
│  ├── Prefill 请求按 token budget 切分                 │
│  └── 合并为一条序列送入模型                            │
├──────────────────────────────────────────────────────┤
│  KVCachePool（分页式 KV Cache）                       │
│  ├── 按需分配/释放物理页                               │
│  └── 请求级别的页映射管理                              │
├──────────────────────────────────────────────────────┤
│  ModelWrapper（模型执行封装）                          │
│  ├── PageAttention Prefill Kernel                    │
│  └── PageAttention Decoding Kernel                   │
└──────────────────────────────────────────────────────┘
```

**Engine 的 step 函数**（核心主循环）：

下面这段代码以 **nano-vllm**（[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) `nanovllm/engine/llm_engine.py`）为蓝本——它是 vLLM 真实代码的**最小同构版本**（200 行 vs vLLM 几万行），与 vLLM v1 `vllm/v1/engine/llm_engine.py` 的接口完全对齐。读懂 nano-vllm 的 `step()`，就读懂了 vLLM v1 的核心。

```python
class LLMEngine:
    """nano-vllm / vLLM v1 风格的推理引擎主类"""

    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)

        # 多进程拉起 TP workers（rank ≥ 1 在子进程，rank 0 在主进程）
        self.ps, self.events = [], []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, i, event))
            p.start()
            self.ps.append(p); self.events.append(event)

        self.model_runner = ModelRunner(config, rank=0, events=self.events)
        self.tokenizer    = AutoTokenizer.from_pretrained(config.model)
        config.eos        = self.tokenizer.eos_token_id
        self.scheduler    = Scheduler(config)

    def add_request(self, prompt, sampling_params):
        """新请求入队——统一转成 token id 后丢给 scheduler"""
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        一步推理：
          1. scheduler 决定这一步处理哪些 sequence（prefill 或 decode）
          2. ModelRunner 在所有 TP rank 上同步执行前向
          3. scheduler 根据生成的 token 更新 sequence 状态、释放完成的 KV
        """
        seqs, is_prefill = self.scheduler.schedule()

        # 在所有 TP worker 上同步执行（"call" RPC 会广播到子进程）
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        self.scheduler.postprocess(seqs, token_ids)

        # 收集这一步完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]
        # 用 "正/负 token 数" 区分 prefill 吞吐 vs decode 吞吐
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts, sampling_params):
        """对外的高层 API——封装 add_request + step 循环"""
        for p, sp in zip(prompts, sampling_params):
            self.add_request(p, sp)

        outputs = {}
        while not self.is_finished():
            for seq_id, token_ids in self.step()[0]:
                outputs[seq_id] = token_ids

        return [self.tokenizer.decode(outputs[i]) for i in sorted(outputs)]
```

::: tip 核心逻辑只有 5 行
注意 `step()` 的核心其实就是：
```python
seqs, is_prefill = self.scheduler.schedule()       # 1. 选 batch
token_ids = self.model_runner.call("run", seqs, is_prefill)  # 2. 跑模型
self.scheduler.postprocess(seqs, token_ids)        # 3. 更新状态
```
**Scheduler 决定调度策略，ModelRunner 决定执行细节，LLMEngine 只是把它们粘合起来**——这种"用接口区分关注点"的设计是 vLLM 之所以能扩展出 V1 / async / disaggregated 等多种部署模式的关键。
:::

::: tip "正负 num_tokens" 的小巧思
nano-vllm 用 `num_tokens > 0` 表示 prefill 吞吐（按总 token 数）、`< 0` 表示 decode 吞吐（按序列数）。这样上层 `tqdm` 进度条可以用同一个变量名同时报告两种吞吐，省一个分支。
:::

**V1 的模型层**如何处理混合 PD batch：

```python
# 教学示意：真实分发逻辑见 vllm/attention/backends/*.py
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/layer.py
class MixedBatchAttention(nn.Module):
    def forward(self, x, kv_cache, info):
        n_prefill = info.prefill_batch    # Prefill 请求数
        n_decode = info.decoding_batch    # Decode 请求数

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 按请求拆分（每个请求的 chunk_len 不同）
        q_parts = q.split(info.chunk_len, dim=0)
        k_parts = k.split(info.chunk_len, dim=0)
        v_parts = v.split(info.chunk_len, dim=0)

        outputs = []
        # Decode 请求：单 token 走 PagedAttention decode kernel
        if n_decode > 0:
            outputs.append(self.paged_decode(
                q_parts[:n_decode], k_parts[:n_decode], v_parts[:n_decode],
                kv_cache=kv_cache[:n_decode],
            ))
        # Prefill 请求：变长 chunk 走 PagedAttention prefill kernel
        if n_prefill > 0:
            outputs.append(self.paged_prefill(
                q_parts[n_decode:], k_parts[n_decode:], v_parts[n_decode:],
                kv_cache=kv_cache[n_decode:],
            ))

        return self.o_proj(torch.cat(outputs, dim=0))
```

**V0 vs V1 对比**：

| 特性 | V0 | V1 |
|------|----|----|
| Prefill/Decode | 分开处理 | 统一为 Chunked Prefill |
| 调度粒度 | 请求级别 | Token 级别（token budget） |
| 新请求 | 等当前 batch 有空位 | 切 chunk 立即开始 |
| GPU 利用率 | Decode 阶段低 | Prefill 填充空闲算力 |
| 实现复杂度 | 较低 | 较高 |

---

## 端侧推理引擎

前面介绍的 KV Cache、PagedAttention、Continuous Batching、投机采样等技术主要面向云端 GPU 集群。本节聚焦另一条路径——**在手机、笔记本、嵌入式设备上直接运行 LLM**，并以两个开源项目作为锚点剖析其实现思路：

- [llama.cpp](https://github.com/ggml-org/llama.cpp)（C/C++，多后端，社区事实标准）
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)（基于 TVM 的编译式部署，覆盖 iOS/Android/WebGPU）

::: warning 本节定位
端侧 LLM 涉及大量厂商私有 SDK（NPU 驱动、量化工具链、设备端 runtime）。本教程**只讨论开源、可被任意 GitHub 用户复现**的部分。读者在工业项目中遇到具体厂商工具链，请以厂商官方文档为准。
:::

### 端侧推理的约束与机会

::: tip 为什么要在端上跑大模型？
- **隐私**：聊天记录、照片等数据不出设备
- **延迟**：无网络往返，离线也能用
- **成本**：边际推理成本趋近于零
:::

但端侧的硬约束也很硬：

- **统一内存** 8–16 GB（与系统、其他 App 共享），可分配给 LLM 的通常只有 4–8 GB
- **算力上限**远低于 H100 / H200（手机 SoC 单芯片峰值 ~30 TOPS 量级）
- **能耗**不能像云端那样"暴力计算"，单 prompt 回答控制在 1–3 秒内才不烫手
- **后端碎片化**：CPU(Arm/x86)、Apple Metal、Vulkan、OpenCL、CUDA(Jetson)、各家 NPU…… 没有 CUDA 这种事实标准

落到模型上，典型配置是 **0.5B–7B 参数 + INT4 量化**，对应模型文件 ~0.4–4 GB。

### 主流开源端侧引擎

| 引擎 | 主要语言 | 后端覆盖 | 模型格式 | 典型场景 |
|------|---------|---------|---------|---------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ | CPU(Arm/x86)、Metal、CUDA、Vulkan、HIP、OpenCL、SYCL、CANN、Hexagon… | GGUF | 通用、CPU / Apple Silicon 首选 |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | Python + TVM | Vulkan、Metal、CUDA、ROCm、WebGPU、iOS Metal、Android OpenCL | TVM 编译产物 | 跨平台 GPU、Web 浏览器 |
| [Apple Core ML](https://developer.apple.com/documentation/coreml) | Swift / ObjC | ANE / Metal | `.mlmodel` / `.mlpackage` | iOS / macOS 系统集成 |
| [PyTorch ExecuTorch](https://github.com/pytorch/executorch) | Python + C++ | XNNPACK、Vulkan、各家 NPU 后端 | `.pte` | PyTorch 原生工作流 |
| [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/) | C++ | CPU、NNAPI、Core ML、Vulkan | ONNX | ONNX 已有产线 |

> 后端列表参考 [llama.cpp `README.md`](https://github.com/ggml-org/llama.cpp/blob/master/README.md) 的 backend 表（写本节时 `master` 上还在标 *In Progress* 的有 OpenVINO、WebGPU、Hexagon、VirtGPU）。

### llama.cpp 的实现解读

llama.cpp 的核心是把 PyTorch / HuggingFace 模型"压扁"成一个**自包含的二进制文件**（GGUF），然后用纯 C/C++ runtime（GGML）在任意后端上跑起来。

#### GGUF：单文件、可 mmap 的模型容器

GGUF 文件结构（写入器实现见 [`gguf-py/gguf/gguf_writer.py`](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py) 的 `GGUFWriter` 类）：

```
[ Magic + Version ]            ← GGUF_MAGIC, GGUF_VERSION（gguf-py/gguf/constants.py）
[ Tensor Count + KV Count ]
[ Key-Value Metadata ]         ← 词表、超参、chat template、量化方案……
[ Tensor Info × N ]            ← 每个 tensor 的 name / shape / dtype / offset
[ Padding 到对齐边界 ]         ← GGUF_DEFAULT_ALIGNMENT
[ Tensor Data ]                ← 紧凑排布的权重字节流
```

> 写入状态机见 `gguf_writer.py` 中的 `WriterState` 枚举：`NO_FILE → EMPTY → HEADER → KV_DATA → TI_DATA → WEIGHTS`。

为什么这样设计？两个工程目的：

1. **mmap 友好**：tensor data 段对齐排布后，runtime 直接 `mmap()` 映射文件，权重不进进程堆，多进程加载共享同一份页缓存——这是端侧内存紧张时极重要的优化。
2. **元数据自描述**：词表、tokenizer 类型、chat template、RoPE 配置全部塞在 KV 段里，加载时无需额外配置文件。

#### HF → GGUF 转换

入口脚本 [`convert_hf_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)（一万三千多行，每个模型家族对应一个 `*Model` 子类）。命令行用法：

```bash
# 安装依赖后，下载 HF 模型并转换
pip install -r llama.cpp/requirements.txt

python llama.cpp/convert_hf_to_gguf.py \
    /path/to/Qwen2.5-1.5B-Instruct \
    --outtype f16 \
    --outfile qwen2.5-1.5b-f16.gguf
```

`--outtype` 仅支持高精度类型：`f32 / f16 / bf16 / q8_0 / tq1_0 / tq2_0`（见 `convert_hf_to_gguf.py` 的 `parse_args`）。**真正的低比特量化是单独一步**，由 `llama-quantize` 完成（见下文）。

#### GGML 量化族

GGML 的量化类型枚举位于 [`ggml/include/ggml.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h) 的 `enum ggml_type`：

| 类型 | 比特数 | 块大小 | 备注 |
|------|--------|--------|------|
| `GGML_TYPE_F16` / `BF16` | 16 | – | 半精度参考 |
| `GGML_TYPE_Q8_0` | 8 | 32 | 块对称量化，1 个 fp16 scale |
| `GGML_TYPE_Q4_0` / `Q4_1` | 4 | 32 | 经典分块（`_1` 含 zero-point） |
| `GGML_TYPE_Q4_K` / `Q5_K` / `Q6_K` | 4 / 5 / 6 | 256（super-block） | "K-quant"，super-block + sub-block 双层 scale |
| `GGML_TYPE_IQ2_XXS` … `IQ4_XS` | 1.5–4 | 256 | "I-quant"，基于 importance matrix 的码本量化 |
| `GGML_TYPE_MXFP4` / `NVFP4` | 4 | 1 / 4 | OCP MXFP4 / NVIDIA NVFP4 微缩浮点 |

> 完整列表见 `ggml.h` 的 `enum ggml_type`。`Q4_K_M / Q5_K_S` 等 HuggingFace 下载页常见的命名是更高层的"混合策略"：模型不同张量用不同 type（如 `output.weight` 用 Q6_K，attention 用 Q4_K，FFN 用 Q5_K），由 `llama-quantize` 的 ftype 决定。

K-quant 的双层 scale 思想——把一个大块（256 元素）拆成多个子块（如 8 个 32 元素子块），每个子块有自己的小 scale，整大块再有一个统一 scale——是公开提案 [k-quants PR #1684](https://github.com/ggml-org/llama.cpp/pull/1684) 给出的方案。

#### 量化命令

转换得到的 fp16 GGUF 通过 `llama-quantize` 二次量化：

```bash
# 1) 编译 llama.cpp（macOS Apple Silicon 默认带 Metal）
cd llama.cpp && cmake -B build && cmake --build build -j

# 2) 量化为 Q4_K_M
./build/bin/llama-quantize qwen2.5-1.5b-f16.gguf qwen2.5-1.5b-q4_k_m.gguf Q4_K_M

# 3) 跑起来
./build/bin/llama-cli -m qwen2.5-1.5b-q4_k_m.gguf -p "用一句话解释大模型推理优化" -n 128
```

#### 后端调度

llama.cpp 的多后端架构在 `ggml/src/ggml-backend.cpp` 与 `ggml-backend-reg.cpp`：每个后端目录（`ggml-cpu/`、`ggml-metal/`、`ggml-cuda/`、`ggml-vulkan/`、`ggml-opencl/`、`ggml-hexagon/` 等）注册自己支持的 op set，runtime 在加载模型时按 op 兼容性切分子图、把不支持的 op fallback 到 CPU。

对端侧最关键的几点：
- **CPU 后端**自动检测 NEON / SVE / AMX / AVX2 / AVX512 等 SIMD 扩展，所以同一个二进制能在 Pixel 5 和 Pixel 9 上都跑出最佳性能（参考 [`docs/android.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) 的 "hardware acceleration up to SME2 for Arm" 描述）
- **Metal 后端**在 Apple Silicon 上是默认开关；模型权重通过 `mmap` + Metal shared buffer 实现 GPU/CPU 零拷贝
- **OpenCL 后端**主要服务 Adreno（Snapdragon GPU）和 Mali，是 llama.cpp 在普通 Android 设备上跑 GPU 加速的路径

### MLC LLM 的实现解读

MLC LLM 走的是**先编译再部署**的路线（MLC = *Machine Learning Compilation*）：基于 TVM / Relax 把模型 IR 化、做 op fusion 与 schedule auto-tuning，再为每个目标后端生成专用 kernel。

#### 量化配置

[`python/mlc_llm/quantization/`](https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_llm/quantization) 下提供多种方案：

- `group_quantization.py`：经典 group-wise PTQ（`GroupQuantize` 类），支持 int3 / int4 / int8 元素 + uint32 storage
- `awq_quantization.py`：[AWQ](https://arxiv.org/abs/2306.00978)（Activation-aware Weight Quantization）
- `fp8_quantization.py` / `block_scale_quantization.py`：FP8 与块级 scale
- `per_tensor_quantization.py`：per-tensor 静态量化

以 group 量化为例（最常用），其核心字段（参考 [`group_quantization.py`](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/group_quantization.py) 的 `GroupQuantize` dataclass）：

```python
# 摘自 mlc-llm 的 GroupQuantize 配置类
@dataclass
class GroupQuantize:
    name: str
    kind: str                                   # 'group-quant'
    group_size: int                             # 一组多少元素共享一组 scale
    quantize_dtype: Literal["int3", "int4", "int8"]
    storage_dtype: Literal["uint32"]            # 例如 int4 → 8 个塞进一个 uint32
    model_dtype: Literal["float16", "float32", "bfloat16"]
    linear_weight_layout: Literal["KN", "NK"]
```

预设 `q4f16_1` 表示：4-bit 权重（int4 元素，uint32 storage，group_size=32）+ float16 model dtype + KN layout。

#### 三步部署流程

MLC LLM 官方 [Quick Start 文档](https://llm.mlc.ai/docs/get_started/quick_start.html) 给出的命令式流程：

```bash
# 1) 把 HF 权重转换并量化（写入 mlc-chat-config.json + ndarray-cache.json）
mlc_llm convert_weight ./Qwen2.5-1.5B-Instruct/ \
    --quantization q4f16_1 -o ./dist/qwen2_5-1.5b-q4f16_1/

# 2) 生成 mlc-chat-config.json（chat template、采样参数）
mlc_llm gen_config ./Qwen2.5-1.5B-Instruct/ \
    --quantization q4f16_1 --conv-template qwen2 \
    -o ./dist/qwen2_5-1.5b-q4f16_1/

# 3) 编译为目标后端的可执行库（vulkan / metal / cuda / android / iphone / webgpu）
mlc_llm compile ./dist/qwen2_5-1.5b-q4f16_1/mlc-chat-config.json \
    --device metal -o ./dist/qwen2_5-1.5b-q4f16_1/lib.dylib
```

后端覆盖见 [MLC LLM `README.md`](https://github.com/mlc-ai/mlc-llm/blob/main/README.md) 的兼容性表：iOS / iPadOS 走 Metal（A 系列 GPU），Android 走 OpenCL（Adreno 或 Mali）。WebGPU 路径还有兄弟项目 [WebLLM](https://github.com/mlc-ai/web-llm) 可直接在浏览器里跑。

### 移动端实战：Android 跑通最小例子

llama.cpp 官方文档 [`docs/android.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) 给了三条路径，最稳妥的是 NDK 交叉编译：

```bash
# 1) 在主机上交叉编译（需先装好 Android SDK + NDK，并设置 ANDROID_NDK 环境变量）
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_C_FLAGS="-march=armv8.7a" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7a" \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -B build-android

cmake --build build-android --config Release -j8
cmake --install build-android --prefix install-android --config Release

# 2) 推到设备
adb shell "mkdir -p /data/local/tmp/llama.cpp"
adb push install-android /data/local/tmp/llama.cpp/
adb push qwen2.5-1.5b-q4_k_m.gguf /data/local/tmp/llama.cpp/

# 3) 在设备上跑（注意要带 LD_LIBRARY_PATH）
adb shell
$ cd /data/local/tmp/llama.cpp
$ LD_LIBRARY_PATH=lib ./bin/llama-cli \
      -m qwen2.5-1.5b-q4_k_m.gguf \
      -c 4096 -p "你好"
```

> 完整步骤见 [`docs/android.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) 的 "Cross-compile CLI using Android NDK" 章节。`-c 4096` 设上下文长度，太大会让 KV Cache 撑爆设备内存。如果只想快速体验，可在手机上装 [Termux](https://termux.dev/en/) 直接 `apt install` 编译，免 NDK。

如果用 MLC LLM，对应的移动端样板工程在 [`mlc-llm/android/MLCChat`](https://github.com/mlc-ai/mlc-llm/tree/main/android/MLCChat)（Android）和 [`mlc-llm/ios/MLCChat`](https://github.com/mlc-ai/mlc-llm/tree/main/ios/MLCChat)（iOS）。

### 端侧 vs 云端推理对比

| 维度 | 云端 (vLLM / SGLang) | 端侧 (llama.cpp / MLC LLM) |
|------|---------------------|----------------------------|
| 模型规模 | 70B–400B | 0.5B–7B |
| 主流权重精度 | FP16 / BF16 / FP8 | INT4 群量化（Q4_K_M / q4f16_1 …） |
| 模型文件 | 100 GB+ 多 shard | 单文件 0.5–5 GB |
| 加载方式 | TP / PP 分片到多卡 | mmap 单进程 |
| 并发 | Continuous Batching | 单用户 / 极小 batch |
| 主要瓶颈 | 算力 / KV Cache 显存 | 内存带宽 + 设备发热 |
| 部署门槛 | k8s + GPU 池 | 一个 cmake + 一个 .gguf |
| 用户隐私 | 数据离开设备 | 数据本地处理 |

::: tip 端云协同
现实产品里端云通常协同工作：
- 短问答 / 隐私敏感 → 端侧
- 长文档 / 复杂推理 / 工具调用 → 云端
- 智能路由器（router）按请求复杂度做分流，例如 [MLC Router](https://github.com/mlc-ai/mlc-llm/tree/main/python/mlc_llm/router) 给出的就是这个思路
:::

### 选型建议与常见坑

| 场景 | 推荐 | 理由 |
|------|------|------|
| Apple Silicon Mac 本地体验 | llama.cpp + Metal | 一行 cmake 全自动，性能接近 PyTorch + MPS |
| iOS App 集成 | MLC LLM 或 Core ML | MLC 跨平台一份代码；Core ML 与系统集成更深 |
| 普通 Android 手机 | llama.cpp（CPU） | 设备碎片化严重时 CPU 后端最稳 |
| 旗舰 Android（Adreno / Mali） | MLC LLM OpenCL | GPU 加速吞吐显著 |
| 浏览器 / WebGPU | [WebLLM](https://github.com/mlc-ai/web-llm) | 不需要安装、直接跑 |
| PyTorch 已有 pipeline | ExecuTorch | 同一套 `torch.export` 直接走端侧 |

::: warning 端侧常见坑
- **Context length 设太大**：KV Cache 大小 ≈ `2 × n_layer × n_kv_head × head_dim × seq_len × bytes_per_elem`，4096 在 1.5B 模型上已经吃 ~200 MB
- **Q4_0 vs Q4_K_M 别选错**：同样是 4-bit，K-quant 在大多数模型上 perplexity 显著优于 legacy `Q4_0` / `Q4_1`
- **第一次 prefill 慢**：llama.cpp 默认 mmap 是惰性页加载，第一次跑长 prompt 会触发大量缺页中断；可以 `--no-mmap` 强制全量加载（更费内存但首 token 快）
- **设备发热降频**：跑 30 秒以上要监控 SoC 温度，否则 throttle 会让吞吐掉一半
:::

---

## 苏格拉底时刻

在继续之前，尝试回答以下问题来检验你的理解：

::: details 1. 为什么 KV Cache 能加速推理？如果没有 KV Cache，生成第 100 个 token 需要多少次矩阵乘法？
没有 KV Cache 时，生成第 $n$ 个 token 需要对前面 $n-1$ 个 token 重新计算所有层的 K 和 V，总计算量与 $n^2$ 成正比。生成第 100 个 token 需要对 99 个 token 做完整的 Attention 计算。有了 KV Cache，只需要计算当前 token 的 Q、K、V 并拼接，计算量与 $n$ 成线性关系。
:::


::: details 2. PagedAttention 借鉴了操作系统的什么概念？为什么能提升显存利用率？
借鉴了虚拟内存中的分页机制（Paging）。通过将 KV Cache 切分为固定大小的 Block 并按需分配，避免了预分配最大长度导致的内部碎片，以及不同请求长度不一导致的外部碎片。显存利用率从 20-40% 提升到接近 100%。
:::


::: details 3. Continuous Batching 相比 Static Batching 的核心改进是什么？
核心改进是在 iteration 级别而非 request 级别做调度。短请求完成后立即释放资源并插入新请求，避免 GPU 空等长请求完成。吞吐量提升 2-8 倍。
:::


::: details 4. 投机采样为什么是「无损」的？如果小模型猜错了会怎样？
通过精心设计的 accept/reject 机制（以 $\min(1, p(x)/q(x))$ 的概率接受），保证了最终输出的概率分布与大模型一致。小模型猜错时，从修正后的分布 $\max(0, p(x)-q(x))$ 重新采样，确保不引入偏差。最坏情况下退化为普通的大模型推理速度。
:::


::: details 5. Chunked Prefill 的 Token Budget 是什么意思？为什么 Decode 优先？
Token Budget 是每步模型前向传播的总 token 数上限（如 8192）。Decode 优先是因为正在生成的请求需要持续输出 token 以满足用户实时体验——如果被新的 Prefill 阻塞，用户会感受到卡顿。所以先把所有 Decode 请求放入 batch（每个只占 1 token），再用剩余的 budget 给 Prefill 请求分 chunk。
:::


::: details 6. PD 分离的核心挑战是什么？有几种 KV Cache 管理模式？
核心挑战是 Prefill 节点计算完 KV Cache 后需要高效传输到 Decode 节点，传输延迟不能抵消分离带来的收益。三种管理模式：(1) KV Cache 仅存于 Decode 节点 (2) PD 各存各的 (3) 中心化/去中心化 Cache 服务。
:::


::: details 7. 在 PagedAttention 的 Decode 阶段，多个 Page 的 Attention 结果如何正确聚合？
通过 Online Softmax 技巧：每个 Page 独立计算局部的 Attention 输出 $O_i$、最大值 $M_i$ 和归一化因子 $L_i$，然后通过 $O = \sum_i \exp(M_i - M_{global}) \cdot \frac{L_i}{L_{global}} \cdot O_i$ 聚合为全局正确的结果。这是精确计算，非近似。
:::


---

## 常见问题 & 面试考点

### 高频面试题

**Q: LLM 推理的 Prefill 和 Decode 阶段有什么区别？**

A: Prefill 是处理输入 prompt，一次性计算所有 token 的 KV Cache，是 Compute-bound；Decode 是逐 token 生成，每步读取全部参数和 KV Cache，是 Memory-bound。以 Llama 70B 为例，Decode 阶段算术强度仅为 1 FLOP/Byte，远低于 GPU 的计算带宽比 156 FLOP/Byte。

**Q: vLLM 为什么快？核心技术是什么？**

A: vLLM 的核心是 PagedAttention（解决 KV Cache 显存碎片，利用率从 ~30% 提升到 ~100%）+ Continuous Batching（迭代级调度，吞吐提升 2-8x）+ Chunked Prefill（避免长 Prefill 阻塞 Decode）。

**Q: 投机采样的加速比取决于什么因素？**

A: 取决于 Draft Model 和 Target Model 输出分布的一致性（可用 KL 散度度量）。接受率越高加速越大。通常使用同系列的小模型（如 Llama-7B → Llama-70B）可以达到 70-85% 的接受率，实现 2-3x 加速。

**Q: Chunked Prefill 和 PD 分离的关系是什么？**

A: Chunked Prefill 是 PD 融合方案——在同一 GPU 上交错执行 Prefill 和 Decode。PD 分离则是将二者放在不同物理节点上。实际系统通常兼并两种方案：Prefill 节点内部也可以使用 Chunked Prefill。

### 性能指标速查

| 指标 | 含义 | 优化技术 |
|------|------|----------|
| TTFT (Time To First Token) | 首个 token 延迟 | Chunked Prefill, 模型并行 |
| TPOT (Time Per Output Token) | 每个输出 token 延迟 | KV Cache, 投机采样 |
| Throughput (tokens/s) | 吞吐量 | Continuous Batching, PagedAttention |
| GPU Memory Utilization | 显存利用率 | PagedAttention |
| GPU Compute Utilization | 算力利用率 | Chunked Prefill |

---

## 推荐资源

### 必读论文

- **PagedAttention**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（vLLM 核心论文）
- **Speculative Decoding**: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)（Google）
- **EAGLE-3**: [EAGLE-3: Scaling up Speculative Sampling with Feature-Level Draft Model](https://arxiv.org/abs/2602.21233)（腾讯 AngelSlim）
- **SpecExit**: [SpecExit: Accelerating Large Reasoning Models via Speculative Exit](https://arxiv.org/abs/2509.24248)（推理模型思考早退）
- **Orca**: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)（Continuous Batching 的起源）
- **Sarathi-Serve**: [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310)（Chunked Prefill）
- **DistServe**: [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)（PD 分离）

### 代码与工具

- [vLLM GitHub](https://github.com/vllm-project/vllm) — 最流行的开源推理引擎
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA 官方推理优化
- [SGLang](https://github.com/sgl-project/sglang) — 高性能推理框架
- [AngelSlim](https://github.com/Tencent/AngelSlim) — 腾讯大模型压缩工具包（EAGLE-3 投机采样 + 量化 + SpecExit）
- [MLC LLM](https://github.com/mlc-ai/mlc-llm) — 通用 LLM 端侧推理引擎（支持 iOS/Android/GPU/NPU）
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 纯 C/C++ 实现的 LLM 推理，极致兼容性

### 博客与视频

- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)（AnyScale）
- [vLLM 官方文档和 Meetup 系列](https://docs.vllm.ai/)
- [FlashAttention 原理详解](https://zhuanlan.zhihu.com/p/670085985)
