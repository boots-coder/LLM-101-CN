---
title: "vLLM V1 与 PD 分离架构"
description: "从 vLLM V0 到 V1 的架构演进，以及 Prefill-Decode 分离的完整实现"
topics: [vLLM-V1, prefill-decode-disaggregation, scheduler, block-table, chunked-prefill, inference-engine]
prereqs: [engineering/inference, deep-dives/nano-vllm]
---

# vLLM V1 与 PD 分离架构

> 从 Chunked Prefill 到 PD Disaggregation，理解现代推理引擎的两条演进路线。

::: info 前置知识
阅读本章前，建议先完成 [手搓 vLLM 推理引擎](./nano-vllm.md)，掌握 PagedAttention、Continuous Batching、KV Cache 分页管理等基础概念。
:::

## 全景图：V0 到 V1 再到 PD 分离

LLM 推理引擎的演进并非线性的，而是沿着两条互补的路线同时发展：

```
V0 (Continuous Batching + PagedAttention)
 │
 ├──> V1 (Chunked Prefill + 异步调度 + 统一 Kernel)
 │         └── PD 融合路线：同一 GPU 上交错处理 Prefill 和 Decode
 │
 └──> PD Disaggregation
           └── PD 分离路线：不同 GPU 群组分别处理 Prefill 和 Decode
```

**两条路线并不矛盾**——生产级系统通常同时使用：Prefill 节点内部用 Chunked Prefill，而 Prefill/Decode 节点之间做分离。

---

## 第一部分：V0 的局限性

vLLM V0 已经实现了 PagedAttention 和 Continuous Batching，但在高并发场景下暴露出几个问题：

### 1. Prefill 阻塞 Decode

V0 中，一个 step 要么全部做 Prefill，要么全部做 Decode。当一个长 prompt（如 8K tokens）进入时，所有 Decode 请求必须等待这次 Prefill 完成——造成 **Head-of-Line Blocking**。

```
V0 时间线（简化）：
Step 0: [Prefill R1 ████████████████]     ← 长 prompt，耗时久
Step 1: [Decode R0] [Decode R1]            ← R0 被迫等了一整个 step
Step 2: [Decode R0] [Decode R1]
```

### 2. 计算资源浪费

Decode 阶段每个请求只产生 1 个 token 的计算量（`1 x d @ d x d`），而 GPU 算力大量闲置。权重矩阵被反复从 HBM 搬运到 SRAM，**算术强度极低**。

### 3. Kernel 割裂

V0 需要分别维护 Prefill Kernel 和 Decode Kernel：

```python
# V0：两套独立的 attention 路径
if is_prefill:
    output = prefill_attention(Q, K, V, ...)
else:
    output = paged_attention_decode(Q, K_cache, V_cache, ...)
```

这导致代码复杂度高，且无法在同一 batch 内混合 P/D 请求。

---

## 第二部分：V1 核心改进——Chunked Prefill

V1 的核心思想：**将长 Prefill 请求切成小块（chunk），与 Decode 请求交错执行，共享同一个 batch。**

```
V1 时间线（Chunked Prefill）：
Step 0: [Decode R0] [Prefill R1-chunk1 ████]   ← R1 只处理一部分
Step 1: [Decode R0] [Decode R1?] [Prefill R1-chunk2 ████]
Step 2: [Decode R0] [Decode R1] [Prefill R1-chunk3 ██]  ← R1 Prefill 完成
Step 3: [Decode R0] [Decode R1]                ← 全 Decode
```

### 为什么 Decode 能"搭便车"？

在投影计算（Linear 层）中，多个请求的输入可以拼接成一条长序列：

```python
# Prefill 请求（1000 tokens）+ Decode 请求（1 token）
X_merged = torch.cat([X_prefill, x_decode], dim=0)  # [1001, d]
Q_merged = X_merged @ W_Q                            # [1001, d]
# 拆分回各自请求
Q_prefill, q_decode = Q_merged.split([1000, 1], dim=0)
```

Prefill 请求提供了足够的计算量使 GPU 饱和，Decode 请求"捎带"计算，几乎不增加额外开销。

---

## 第三部分：V1 核心组件详解

以下基于教学版 nano-vLLM-V1 源码，逐一剖析核心模块。

### 3.1 KVPageManager：分页资源管理

`KVPageManager` 是最底层的资源管理器，**只负责页面的分配和释放**，不关心 KV Cache 的具体内容。

```python
from collections import deque

class KVPageManager:
    """物理页帧池 —— 维护空闲/占用状态，提供 acquire / release 接口"""

    def __init__(self, slots_per_page: int, total_pages: int):
        self.slots_per_page = slots_per_page
        self.total_pages = total_pages
        # 用 deque 做空闲池，popleft 分配、append 回收
        self.idle_pool: deque[int] = deque(range(total_pages))
        self.in_use: set[int] = set()
        # 每页的 token 填充计数 & 后继指针（构成单向链表）
        self.fill_count = [0] * total_pages
        self.successor = [-1] * total_pages

    def acquire_pages(self, count: int, prev_page: int = -1) -> list[int]:
        """从空闲池中取出 count 个页，可选地把第一页链到 prev_page 之后"""
        if len(self.idle_pool) < count:
            return []  # 资源不足
        grabbed = [self.idle_pool.popleft() for _ in range(count)]
        self.in_use.update(grabbed)
        for pid in grabbed:
            self.fill_count[pid] = 0
            self.successor[pid] = -1
        if prev_page != -1:
            self.successor[prev_page] = grabbed[0]
        return grabbed

    def release_pages(self, page_ids: list[int]):
        """把页归还到空闲池尾部"""
        for pid in page_ids:
            if pid in self.in_use:
                self.in_use.discard(pid)
                self.idle_pool.append(pid)
                self.fill_count[pid] = 0
                self.successor[pid] = -1

    def num_idle(self) -> int:
        """当前可用页数"""
        return len(self.idle_pool)
```

::: tip 设计要点
- `idle_pool` 使用 `deque`，分配时 `popleft`，释放时 `append`，保证先用先还的公平性
- `successor` 实现了页间链表，支持动态扩展——当一个请求的 KV Cache 超出当前页时，自动分配新页并链接
- 与 V0 的 BlockManager 对比：V1 的 KVPageManager 更轻量，只管索引不管数据
:::

### 3.2 PagedKVStore：KV Cache 存储引擎

`PagedKVStore` 基于 `KVPageManager` 构建，管理实际的 KV 张量存储：

```python
class PagedKVStore:
    """将物理页帧映射到 KV 张量，提供按请求读写的接口"""

    def __init__(self, config):
        self.page_mgr = KVPageManager(config.page_size, config.num_pages)
        self.page_size = config.page_size
        # 一次性分配整块 KV 存储
        # 维度: [kv=2, layers, pages, slots_per_page, heads, head_dim]
        self.kv_pool = torch.zeros(
            2, config.num_layers, config.num_pages,
            config.page_size, config.num_heads, config.head_dim
        )
        self.req_pages: dict[int, list[int]] = {}   # req_id -> 页列表
        self.cached_len: dict[int, int] = {}         # req_id -> 已写入 token 数
```

**核心方法——`write_kv`**：写入新 KV 数据时，自动处理跨页写入：

```python
def write_kv(self, req_id: int, kv_data: torch.Tensor):
    """
    把形状为 [2, L, T, H, D] 的新 KV 写入对应请求的页链。
    如果当前尾页剩余 slot 不够，则自动申请新页并继续写入。
    """
    _, num_layers, seq_len, num_heads, head_dim = kv_data.shape

    if req_id not in self.cached_len:
        # 该请求第一次写入，先申请首页
        self.cached_len[req_id] = 0
        first_pages = self.page_mgr.acquire_pages(count=1)
        self.req_pages[req_id] = first_pages

    remaining = seq_len
    while remaining > 0:
        pages = self.req_pages[req_id]
        total_capacity = self.page_size * len(pages)
        free_slots = total_capacity - self.cached_len[req_id]

        if free_slots == 0:
            # 尾页已满 —— 申请新页，链到尾部
            tail = pages[-1] if pages else -1
            fresh = self.page_mgr.acquire_pages(count=1, prev_page=tail)[0]
            self.req_pages[req_id].append(fresh)
            free_slots = self.page_size
        else:
            dst_page = pages[-1]
            col_start = self.page_size - free_slots
            writable = min(free_slots, remaining)
            col_end = col_start + writable
            src_start = seq_len - remaining
            src_end = src_start + writable
            self.kv_pool[:, :, dst_page, col_start:col_end] = \
                kv_data[:, :, src_start:src_end]
            remaining -= writable
            self.cached_len[req_id] += writable

    return self.cached_len[req_id]
```

::: warning 跨页写入
Chunked Prefill 下，一个 chunk 的 KV 可能正好跨越页边界。`write_kv` 的 while 循环确保了这种情况被正确处理——先填满当前页剩余 slot，再分配新页写入剩余部分。
:::

### 3.3 InferenceRequest：请求状态机

每个推理请求的生命周期：

```python
from enum import Enum, auto

class Phase(Enum):
    QUEUED   = auto()
    ACTIVE   = auto()
    DONE     = auto()

class InferenceRequest:
    def __init__(self, rid: int, prompt_ids: list[int], gen_limit: int = 2048):
        self.rid = rid
        self.prompt_ids = prompt_ids              # 原始 prompt token ids
        self.output_ids: list[int] = []           # 已生成的 token
        self.phase = Phase.QUEUED                 # QUEUED -> ACTIVE -> DONE
        self.total_len = len(prompt_ids)
        self.gen_limit = gen_limit
        self.prefilled_len = 0                    # 已完成 prefill 的 token 数

    def append_token(self, tok: int):
        """追加一个新生成的 token，并检查终止条件"""
        self.output_ids.append(tok)
        self.total_len += 1
        if self.reached_end():
            self.phase = Phase.DONE

    def in_decode_stage(self) -> bool:
        """已有输出 token -> 处于 decode 阶段"""
        return len(self.output_ids) > 0

    def reached_end(self) -> bool:
        """达到生成上限或遇到 EOS"""
        return (len(self.output_ids) >= self.gen_limit or
                (self.output_ids and self.output_ids[-1] == EOS_TOKEN))

    def full_sequence(self) -> list[int]:
        """返回完整 token 序列（prompt + 生成部分）"""
        return self.prompt_ids + self.output_ids
```

关键字段 `prefilled_len` 记录了该请求已完成 prefill 的 token 数。在 Chunked Prefill 下，`prefilled_len` 会在多个 step 中逐步增长，直到等于 `len(prompt_ids)`。

### 3.4 BatchPlanner：Chunked Prefill 调度器

调度器是 V1 的大脑，核心逻辑在 `build_batch` 方法中：

```python
class BatchPlanner:
    def build_batch(self, token_budget: int, max_decode_slots: int, max_prefill_slots: int):
        """
        组装一个混合 P/D batch —— Decode 请求优先占用 budget，
        剩余空间留给 Prefill（可能 chunk 化）。
        返回 BatchPlan 供 Engine 执行。
        """
        plan = BatchPlan()

        # ====== 阶段 1：Decode 请求优先入 batch ======
        budget_used = 0
        for rid in self.active_set:
            req = self.all_requests[rid]
            if req.in_decode_stage():
                plan.req_ids.append(rid)
                plan.token_chunks.append([req.output_ids[-1]])
                plan.chunk_lengths.append(1)       # decode 只贡献 1 个 token
                plan.is_decode_flag.append(True)
                budget_used += 1
            if budget_used == max_decode_slots:
                break
        plan.num_decode = budget_used

        # ====== 阶段 2：Prefill 请求填充剩余 budget ======
        for rid in self.active_set:
            req = self.all_requests[rid]
            if req.in_decode_stage():
                continue

            remaining_budget = token_budget - budget_used
            cursor = req.prefilled_len  # 从上次 prefill 停止处继续

            tail = req.prompt_ids[cursor:]
            if len(tail) <= remaining_budget:
                # 剩余 prompt 能一次放完
                chunk = tail
                plan.logit_positions.append(len(chunk) - 1)  # 标记可以出 token
            else:
                # 剩余 prompt 放不下 -> chunked prefill
                chunk = req.prompt_ids[cursor: cursor + remaining_budget]
                plan.logit_positions.append(-1)  # -1 表示本 chunk 不产出 token

            plan.req_ids.append(rid)
            plan.token_chunks.append(chunk)
            plan.chunk_lengths.append(len(chunk))
            plan.is_decode_flag.append(False)
            budget_used += len(chunk)

        # 把所有 chunk 拼成一条扁平序列
        for chunk in plan.token_chunks:
            plan.flat_tokens.extend(chunk)

        return plan
```

::: tip 调度策略的核心思想
1. **Decode 优先**：Decode 请求的延迟敏感度更高（用户在等 token 流式输出），且每个只占 1 token 的 budget
2. **Token Budget 控制**：`token_budget` 限制了单个 step 的总 token 数，确保 GPU 利用率和延迟的平衡
3. **Chunked Prefill**：`logit_positions == -1` 表示这个 chunk 还没处理到 prompt 末尾，不会产出 next token
:::

### 3.5 BatchPlan：Batch 描述符

`BatchPlan` 是调度器和执行引擎之间的桥梁：

```python
@dataclass
class BatchPlan:
    req_ids: List[int]                # 本 batch 的 request id 列表
    token_chunks: List[List[int]]     # 每个请求的 chunk token ids
    chunk_lengths: List[int]          # 每个请求的 chunk 长度
    flat_tokens: List[int]            # 所有 chunk 拼接后的一维序列
    cached_lengths: List[int]         # 每个请求已有的 kv cache 长度
    page_counts: List[int]            # 每个请求的 kv cache 页数
    is_decode_flag: List[bool]        # 标记每个请求是 P 还是 D
    logit_positions: List[int]        # 每个请求在 chunk 中取 logits 的位置
    num_decode: int                   # decode 请求数量
    num_prefill: int                  # prefill 请求数量
```

举一个具体例子说明 flat batch 的结构：

```
假设 token_budget = 12

Decode 请求 R0 (1 token): [t5]
Decode 请求 R1 (1 token): [t8]
Prefill 请求 R2 (chunk, 10 tokens): [p0, p1, p2, ..., p9]

flat_tokens     = [t5, t8, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9]
chunk_lengths   = [1,   1,  10]
is_decode_flag  = [True, True, False]
logit_positions = [0,    0,   9]
```

### 3.6 Attention Kernel：分离的 P/D 计算

V1 的 Attention 层需要对同一 batch 内的 P/D 请求分别处理：

```python
class PagedAttentionLayer(nn.Module):
    def forward(self, X, kv_history, plan: BatchPlan):
        n_prefill = plan.num_prefill
        n_decode  = plan.num_decode
        seq_len, _ = X.shape

        Q, K, V = self.proj_q(X), self.proj_k(X), self.proj_v(X)
        # 投影计算是统一的 —— 这就是 "搭便车"

        Q = Q.reshape(seq_len, H, D)
        K = K.reshape(seq_len, H, D)
        V = V.reshape(seq_len, H, D)
        new_kv = torch.stack((K, V), dim=0)

        # 按 chunk 拆分回各个请求
        Q = Q.split(plan.chunk_lengths, dim=0)
        K = K.split(plan.chunk_lengths, dim=0)
        V = V.split(plan.chunk_lengths, dim=0)

        # Decode 请求：q(1) attend to kv_history(多页)
        if n_decode > 0:
            out_d = self.attend_decode(Q[:n_decode], K[:n_decode], V[:n_decode],
                                       kv_history=..., plan=plan)
        # Prefill 请求：Q(chunk) attend to [kv_history + K_new, V_new]
        if n_prefill > 0:
            out_p = self.attend_prefill(Q[n_decode:], K[n_decode:], V[n_decode:],
                                        kv_history=..., plan=plan)

        # 拼合输出
        out = torch.cat([out_d, out_p], dim=0) if n_decode > 0 and n_prefill > 0 \
              else (out_d if n_decode > 0 else out_p)
        return out, new_kv
```

::: details Prefill Attention 的 Online Softmax 实现
Prefill Kernel 使用了与 Flash Attention 相同的 **Online Softmax** 技巧来处理分页 KV Cache：

```python
def page_attention_prefill_kernel(Q, K, V, mask=None):
    """
    Q: [1, T, H, D]
    K, V: 列表形式 [page_0, page_1, ..., page_n, current_chunk]
    """
    # 将 KV Cache 页 + 当前 chunk 的 KV 组成列表
    K_cache = list(K[0]) + [K[1]]  # 历史页 + 当前 chunk
    V_cache = list(V[0]) + [V[1]]

    O = torch.zeros(H, T, D)
    M = torch.ones(H, T, 1) * -1e5   # running max
    L = torch.zeros(H, T, 1)          # running sum

    for j in range(len(K_cache)):     # 逐页计算
        S_ij = Q_ @ K_cache[j].transpose(1, 2)
        M_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
        M_new = torch.maximum(M_ij, M)
        P_ij = torch.exp(S_ij - M_new)
        L_new = torch.exp(M - M_new) * L + torch.sum(P_ij, dim=-1, keepdim=True)
        O = torch.exp(M - M_new) * O + P_ij @ V_cache[j]
        M, L = M_new, L_new

    return O / L  # 最终归一化
```

这与 Flash Attention 的 tiling 原理一致：逐块计算 softmax，用 running max 避免数值溢出，最终一次性归一化。
:::

### 3.7 InferenceEngine：推理引擎主循环

```python
class InferenceEngine:
    def __init__(self, model, config):
        self.kv_store = PagedKVStore(config)
        self.model_runner = ModelRunner(model, self.kv_store)
        self.planner = BatchPlanner(config.max_seq_len)

    def step(self, config):
        """执行一个推理步骤"""
        if self.planner.count_pending() == 0:
            return

        # 1. 调度：组装混合 P/D batch
        plan = self.planner.build_batch(
            token_budget=config.max_batch_tokens,
            max_prefill_slots=config.max_prefill_batch,
            max_decode_slots=config.max_decoding_batch,
        )

        # 2. 获取 KV Cache
        kv_cache, plan.page_counts = self.kv_store.fetch_kv(plan.req_ids)

        # 3. 构建统一输入
        input_ids = torch.tensor([plan.flat_tokens], dtype=torch.long)

        # 4. 前向计算
        next_token, new_kv = self.run_forward(input_ids, kv_cache, plan)

        # 5. 更新请求状态 + KV Cache
        self.commit(next_token, new_kv, plan)
```

**`commit` 方法的关键逻辑**：根据 `chunk_lengths` 将输出的 KV 拆分回各个请求，然后分别更新：

```python
def commit(self, next_token, new_kv, plan: BatchPlan):
    # 按 chunk 拆分 KV
    per_req_kv = new_kv.split(plan.chunk_lengths, dim=2)

    for rid, kv_slice in zip(plan.req_ids, per_req_kv):
        req = self.planner.all_requests[rid]
        if req.phase == Phase.DONE:
            self.kv_store.evict(rid)        # 请求完成，释放页面
        else:
            self.kv_store.write_kv(rid, kv_slice)  # 写入新 KV

    # 更新 prefilled_len（记录已 prefill 的进度）
    for i, rid in enumerate(plan.req_ids):
        self.planner.all_requests[rid].prefilled_len += plan.chunk_lengths[i]
```

---

## 第四部分：Prefill-Decode 分离 (PD Disaggregation)

### 4.1 为什么要分离？

Chunked Prefill 虽然巧妙地融合了 P/D 计算，但在超大规模部署中存在局限：

| 维度 | Prefill | Decode |
|------|---------|--------|
| 计算特性 | **Compute-bound**（矩阵乘密集） | **Memory-bound**（KV Cache 访存密集） |
| 理想硬件 | 高算力 GPU（如 H100 SXM） | 高带宽 GPU 或专用访存优化硬件 |
| Batch Size | 小（单请求就能填满算力） | 大（需要大 batch 摊薄访存开销） |
| 弹性需求 | 突发流量时需要更多 Prefill 算力 | 相对稳定 |

**PD 分离的核心动机**：让不同特性的计算任务运行在不同的硬件上，各自优化。

```
                    ┌─────────────────────┐
  User Requests ──> │  Prefill Workers     │ ──KV Cache Transfer──>
                    │  (Compute-optimized) │                       
                    └─────────────────────┘                       
                                                                   
                    ┌─────────────────────┐
                    │  Decode Workers      │ ──> Token Stream
                    │  (Memory-optimized)  │
                    └─────────────────────┘
```

### 4.2 架构设计

基于 Ray 实现的 PD 分离系统包含以下组件：

```
┌──────────────────────────────────────────────────────┐
│                  RequestProducer (请求发送)              │
│  - 模拟用户请求，异步投递到 Dispatcher                    │
└───────────────────────┬──────────────────────────────┘
                        │
                        v
┌──────────────────────────────────────────────────────┐
│           DistributedDispatcher (Ray Actor)             │
│  - prefill_queue: 等待 Prefill 的请求                    │
│  - decode_set: 正在 Decode 的请求                        │
│  - 提供 drain_prefill / collect_decode 接口              │
└─────────────┬─────────────────────┬──────────────────┘
              │                     │
              v                     v
┌─────────────────────┐  ┌─────────────────────┐
│  Prefill Executor    │  │  Decode Executor     │
│  - 从 Dispatcher 取   │  │  - 从 Dispatcher 取   │
│    prefill_queue     │  │    decode_set        │
│  - 调用 PrefillWorker │  │  - 调用 DecodeWorker  │
│  - KV 写入 Pool      │  │  - KV 增量写入 Pool    │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           v                        v
┌──────────────────────────────────────────────────────┐
│              SharedKVPool (Ray Actor)                   │
│  - 中心化 KV Cache 存储                                 │
│  - store_prefill(): 批量写入 Prefill 结果                │
│  - store_decode_step(): 单 token 增量写入                │
│  - load_kv(): Decode 节点读取                           │
└──────────────────────────────────────────────────────┘
```

### 4.3 核心组件实现

#### DistributedDispatcher（分布式版本）

PD 分离的 Dispatcher 是一个 **Ray Actor**（独立进程），P/D 两个引擎通过 RPC 调用获取各自的任务：

```python
@dataclass
class DispatchPlan:
    """PD 分离场景下的批次描述（比 V1 的 BatchPlan 更简单，无需 chunk）"""
    req_ids: List[int] = field(default_factory=list)
    token_seqs: List[List[int]] = field(default_factory=list)
    seq_positions: List[int] = field(default_factory=list)
    num_prefill: int = 0
    num_decode: int = 0


@ray.remote
class DistributedDispatcher:
    """Ray Actor —— 跨进程请求分发与状态跟踪"""

    def __init__(self, seq_limit: int = 1024):
        self.all_requests: dict[int, InferenceRequest] = {}
        self.prefill_queue: deque[int] = deque()
        self.decode_set: set[int] = set()
        self.prefill_alive = True
        self.decode_alive = True

    async def submit(self, prompt_ids: list[int], gen_limit: int) -> int:
        """异步提交请求（线程安全，Ray Actor 串行保证）"""
        rid = len(self.all_requests)
        self.all_requests[rid] = InferenceRequest(rid, prompt_ids, gen_limit)
        self.prefill_queue.append(rid)
        return rid

    def drain_prefill_queue(self):
        """Prefill Engine 调用：取出所有待填充请求"""
        if not self.prefill_queue:
            return None
        result = DispatchPlan()
        while self.prefill_queue:
            rid = self.prefill_queue.popleft()
            req = self.all_requests[rid]
            result.req_ids.append(rid)
            result.token_seqs.append(req.prompt_ids)
            result.seq_positions.append(len(req.prompt_ids))
            result.num_prefill += 1
        return result

    def collect_decode_batch(self):
        """Decode Engine 调用：收集所有正在生成的请求"""
        if not self.decode_set:
            return None
        result = DispatchPlan()
        for rid in self.decode_set:
            req = self.all_requests[rid]
            result.req_ids.append(rid)
            result.token_seqs.append([req.output_ids[-1]])
            result.seq_positions.append(len(req.prompt_ids) + len(req.output_ids))
            result.num_decode += 1
        return result

    def report_token(self, rid: int, token: int):
        """P/D 引擎共用：上报一个新 token"""
        req = self.all_requests[rid]
        req.append_token(token)
        if rid not in self.decode_set:
            self.decode_set.add(rid)        # Prefill 完成 -> 进入 decode 集合
        if req.reached_end():
            self.decode_set.discard(rid)    # 生成结束 -> 移除
```

::: tip V1 BatchPlanner vs PD DistributedDispatcher
- V1 BatchPlanner 是 **同进程对象**，在 engine.step() 中同步调用
- PD DistributedDispatcher 是 **Ray Actor**，P/D 引擎通过 `ray.get(dispatcher.method.remote(...))` 远程调用
- PD Dispatcher 不需要 chunk 逻辑（Prefill 一次处理完整 prompt）
:::

#### SharedKVPool（中心化 KV Cache）

PD 分离系统中，KV Cache 需要跨节点共享。最简单的实现是中心化存储：

```python
@ray.remote
class SharedKVPool:
    """Ray Actor —— 中心化 KV 存储，P/D 节点通过 RPC 读写"""

    def __init__(self, config):
        # 预分配连续 KV 张量（非分页，便于教学演示）
        self.kv_tensor = torch.zeros(
            2, config.num_layers, config.kv_cache_batch,
            config.kv_cache_len, config.num_heads, config.head_dim
        )
        self.rid_to_slot: dict[int, int] = {}  # req_id -> slot 索引

    async def store_prefill(self, req_ids: list[int], kv: torch.Tensor):
        """Prefill 节点调用：批量写入完整 KV"""
        base = len(self.rid_to_slot)
        for offset, rid in enumerate(req_ids):
            self.rid_to_slot[rid] = base + offset
        self.kv_tensor[:, :, base: base + len(req_ids)] = kv

    def store_decode_step(self, req_ids: list[int], kv: torch.Tensor,
                          positions: list[int]):
        """Decode 节点调用：逐 token 增量写入"""
        for i, rid in enumerate(req_ids):
            col = positions[i]
            slot = self.rid_to_slot[rid]
            self.kv_tensor[:, :, slot, col] = kv[:, :, i, 0]

    def load_kv(self, req_ids: list[int]) -> torch.Tensor:
        """Decode 节点调用：读取指定请求的 KV"""
        slots = [self.rid_to_slot[r] for r in req_ids]
        return self.kv_tensor[:, :, slots]
```

::: warning KV Cache 传输是 PD 分离的瓶颈
在生产系统中，KV Cache 的传输方式有三种主流方案：

| 方案 | 延迟 | 适用场景 |
|------|------|---------|
| Ray Object Store | 高（序列化开销） | 原型验证 |
| NCCL P2P | 中（GPU 直连） | 同机多卡 |
| RDMA (GPUDirect) | 低（零拷贝） | 跨机部署 |

上面的教学实现使用 Ray Object Store，生产系统（如 [Mooncake](https://github.com/kvcache-ai/Mooncake)、[DistServe](https://arxiv.org/abs/2401.09670)）通常使用 NCCL 或 RDMA。
:::

#### DisaggregatedExecutor（分离引擎）

`DisaggregatedExecutor` 是整个系统的执行主体：

```python
@ray.remote
class DisaggregatedExecutor:
    """管理 Prefill / Decode 两条流水线的执行"""

    def __init__(self, config, prefill_workers, decode_workers,
                 dispatcher, kv_pool):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.dispatcher = dispatcher
        self.kv_pool = kv_pool

    def _run_prefill_once(self):
        """执行一轮 Prefill"""
        plan = ray.get(self.dispatcher.drain_prefill_queue.remote())
        if plan is None:
            return

        batch = self._pack_prefill_input(plan)
        # 调用 Prefill Worker 前向计算
        outputs = ray.get(
            self.prefill_workers.broadcast("forward", x=batch))

        logits = outputs[0][0]
        tokens = torch.argmax(logits[:, -1, :], dim=-1)

        # 写入 KV Pool + 上报 token
        ray.get(self.kv_pool.store_prefill.remote(plan.req_ids, outputs[0][1]))
        for i, rid in enumerate(plan.req_ids):
            ray.get(self.dispatcher.report_token.remote(
                rid, tokens[i].item()))

    def _run_decode_once(self):
        """执行一轮 Decode"""
        plan = ray.get(self.dispatcher.collect_decode_batch.remote())
        if plan is None:
            return

        batch = self._pack_decode_input(plan)
        kv = ray.get(self.kv_pool.load_kv.remote(plan.req_ids))
        # 调用 Decode Worker 前向计算
        result = ray.get(
            self.decode_workers.broadcast(
                "forward", x=batch, kvcaches=kv))

        logits, new_kv = result[0][0], result[0][1]
        # 增量写入 KV Pool
        ray.get(self.kv_pool.store_decode_step.remote(
            plan.req_ids, new_kv, plan.seq_positions))

        tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for i, rid in enumerate(plan.req_ids):
            ray.get(self.dispatcher.report_token.remote(rid, tokens[i]))

    def loop_prefill(self):
        """Prefill 主循环：持续处理直到无等待请求"""
        while not self._prefill_done():
            self._run_prefill_once()

    def loop_decode(self):
        """Decode 主循环：持续处理直到无运行请求"""
        while not self._decode_done():
            self._run_decode_once()
```

#### Worker 体系：Ray 分布式计算

PD 分离的模型计算通过 Ray Actor 封装，支持多 GPU 分布式：

```python
class ModelShard(nn.Module):
    """对底层模型的薄封装"""
    def __init__(self, config, model_cls):
        super().__init__()
        self.net = model_cls(config)

    def forward(self, x, kvcaches=None, seq_pos=None):
        return self.net(x=x, kvcaches=kvcaches, current_length=seq_pos)


@ray.remote(num_cpus=1)
class PrefillWorker(BaseModelWorker):
    """Prefill 节点 Worker"""
    def load_model(self, config, model_cls):
        self.shard = ModelShard(config, model_cls)

    def forward(self, x, kvcaches=None, seq_pos=None):
        return self.shard(x=x, kvcaches=kvcaches, seq_pos=seq_pos)


@ray.remote(num_cpus=1)
class DecodeWorker(BaseModelWorker):
    """Decode 节点 Worker"""
    def load_model(self, config, model_cls):
        self.shard = ModelShard(config, model_cls)

    def forward(self, x, kvcaches=None, seq_pos=None):
        return self.shard(x=x, kvcaches=kvcaches, seq_pos=seq_pos)
```

`WorkerGroup` 管理一组 Worker 的创建和通信：

```python
class WorkerGroup:
    def _spawn_workers(self, pg, gpus_per_worker):
        world_size = self._num_nodes * self._num_gpus_per_node
        # 创建 rank-0 leader
        leader = self.worker_cls.options(
            num_cpus=gpus_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=0
            ),
        ).remote(world_size, 0, None, None)
        self._handles = [leader]

        # 创建其余 worker
        if world_size > 1:
            leader_addr, leader_port = ray.get(
                leader.get_master_addr_port.remote())
            for rank in range(1, world_size):
                w = self.worker_cls.options(...).remote(
                    world_size, rank, leader_addr, leader_port)
                self._handles.append(w)

    def broadcast(self, method_name, *args, **kwargs):
        """在所有 worker 上并行调用同一方法"""
        return [getattr(h, method_name).remote(*args, **kwargs)
                for h in self._handles]
```

### 4.4 完整启动流程

```python
def launch(config):
    ray.init()

    # 创建共享组件
    dispatcher = DistributedDispatcher.remote(config)
    kv_pool = SharedKVPool.remote(config)

    # 创建 Prefill/Decode Worker Groups（各自的 PlacementGroup）
    prefill_group = WorkerGroup(
        num_nodes=1, num_gpus_per_node=config.worker_gpus,
        worker_cls=PrefillWorker, pg=pg_prefill, ...)
    decode_group = WorkerGroup(
        num_nodes=1, num_gpus_per_node=config.worker_gpus,
        worker_cls=DecodeWorker, pg=pg_decode, ...)

    # 创建 P/D Executor
    prefill_exec = DisaggregatedExecutor.remote(
        config, prefill_group, None, dispatcher, kv_pool)
    decode_exec = DisaggregatedExecutor.remote(
        config, None, decode_group, dispatcher, kv_pool)

    # 初始化模型
    ray.get(prefill_group.init_models(config, ToyModel))
    ray.get(decode_group.init_models(config, ToyModel))

    # 启动三个并行任务
    fut_send   = request_producer.remote(config, dispatcher, ...)  # 发请求
    fut_prefill = prefill_exec.loop_prefill.remote()               # Prefill 循环
    fut_decode  = decode_exec.loop_decode.remote()                 # Decode 循环

    ray.get([fut_prefill, fut_decode, fut_send])
```

---

## 第五部分：Chunked Prefill 深度分析

### 5.1 Chunked Prefill 的注意力计算

Chunked Prefill 引入了一种介于 Prefill 和 Decode 之间的计算模式：

| 模式 | 输入 | KV Cache |
|------|------|----------|
| 标准 Prefill | 完整 prompt `[L, d]` | 无 |
| 标准 Decode | 单 token `[1, d]` | 完整历史 `[T, d]` |
| **Chunked Prefill** | chunk `[C, d]` | 之前 chunks 的 cache `[T_prev, d]` |

```
Attention 矩阵示意（chunk_size=2, 总长度=4）:

Step 0 (chunk 0):        Step 1 (chunk 1):
    k1  k2                   k1  k2  k3  k4
q1  x   x                q3  *   *   x   x
q2  x   x                q4  *   *   x   x

x = 当前 chunk 内计算     * = 从 KV Cache 加载
```

第二个 chunk 计算时，需要同时 attend to：
1. **KV Cache 中的历史页**（k1, k2）
2. **当前 chunk 产生的新 KV**（k3, k4）

### 5.2 Chunked Prefill 调度优化

实际部署中，调度策略会进一步优化：

1. **短请求优先**：将 `prompt_len` 短的请求优先调度，快速完成 Prefill 进入 Decode，增加 Decode batch size
2. **长请求分块**：长 prompt 逐 step 进行 Chunked Prefill，每个 step 都能带上 Decode 请求
3. **Budget 动态调整**：根据 GPU 利用率动态调整 `max_batch_tokens`

```
优化调度示例：

Step 0: [Decode(0)] [Prefill R_short(完整)]   ← 短请求一步完成
Step 1: [Decode(0)] [Decode R_short] [Prefill R_long-chunk1]
Step 2: [Decode(0)] [Decode R_short] [Prefill R_long-chunk2]
Step 3: [Decode(0)] [Decode R_short] [Decode R_long]  ← 全 Decode
```

---

## 第六部分：性能对比与选型指南

### V0 vs V1 vs PD 分离

| 维度 | V0 | V1 (Chunked Prefill) | PD 分离 |
|------|-----|---------------------|---------|
| Prefill 阻塞 | 严重 | 无（分 chunk） | 无（独立节点） |
| GPU 利用率 | Decode 时低 | 高（P/D 融合） | 各节点独立最优 |
| 系统复杂度 | 低 | 中 | 高（需要 KV 传输） |
| 硬件异构 | 不支持 | 不支持 | 支持 |
| 弹性扩缩容 | 整体扩缩 | 整体扩缩 | P/D 独立扩缩 |
| KV Cache 传输 | 无 | 无 | **关键瓶颈** |
| 适用规模 | 单卡/小集群 | 中等规模 | 大规模部署 |

### 选型建议

```
单卡/少量 GPU + 低并发 → V0 足够
中等并发 + 长短 prompt 混合 → V1 (Chunked Prefill)
大规模部署 + 高并发 + 硬件异构 → PD 分离
超大规模 → PD 分离 + 每个 P/D 节点内部用 Chunked Prefill
```

---

## 苏格拉底时刻

在继续之前，尝试回答以下问题：

1. **Chunked Prefill 中 `logit_positions == -1` 意味着什么？** 为什么这个 chunk 不产出 next token？
2. **PD 分离系统中，如果 Prefill 速度远快于 Decode，会发生什么？** 系统该如何应对？
3. **为什么 Decode Kernel 需要 Online Softmax 的 reduce 操作，而 Prefill Kernel 不需要？** 提示：考虑 Q 的数量差异。
4. **在 PD 分离架构中，KV Cache 传输能否与计算 overlap？** 如果可以，需要什么条件？
5. **如果一个 Prefill 请求恰好在页边界处被 chunk 切分，`write_kv` 会如何处理？**

::: details 参考思路
1. `logit_positions == -1` 表示该 chunk 不是 prompt 的最后一段，后面还有 chunk。由于 next token 预测依赖完整 prompt 的最后一个位置的 logits，中间 chunk 的 "最后一个位置" 并不是真正的 prompt 末尾，所以不该产出 token。

2. Prefill 过快会导致大量请求堆积在 running_requests 中等待 Decode，消耗大量 KV Cache 存储。应对策略：(a) 限制 Prefill batch size (b) Prefill 节点主动限流 (c) 动态调整 P/D 节点数量比例。

3. Decode 时每个 request 只有 1 个 q，需要对所有 KV Cache 页的 attention 结果做 reduce（Online Softmax combine）。Prefill 时有多个 q，直接在循环内逐页累积即可（也是 Online Softmax，但维度不同）。

4. 可以 overlap。使用 CUDA Stream + 异步 NCCL 通信，在 Decode 节点计算当前 batch 时，同时接收下一批 KV Cache。需要双缓冲（ping-pong buffer）机制。

5. `write_kv` 的 while 循环会检测 `free_slots == 0`，触发新页分配。如果 chunk 的 KV 数据需要跨页写入（一部分填满旧页，剩余写入新页），循环会自然处理。
:::

---

## 面试考点

### 高频问题

1. **vLLM V0 和 V1 的核心区别是什么？**
   - V0：P/D 分 step 执行，长 Prefill 阻塞 Decode
   - V1：Chunked Prefill 将 P/D 融合到同一 batch，Token Budget 控制计算量

2. **Chunked Prefill 为什么能提高吞吐？**
   - Decode 请求"搭便车"：投影计算（Linear 层）中，多个请求拼成大矩阵，GPU 利用率更高
   - 消除 HOL Blocking：长 Prefill 被切片，不再独占 GPU

3. **PD 分离的 KV Cache 传输有哪些方案？各自优劣？**
   - CPU 序列化传输（简单但慢）
   - NCCL GPU-to-GPU（高效但需同一通信域）
   - RDMA / GPUDirect（最快但硬件要求高）

4. **PD 分离与 PD 融合的关系？**
   - 不矛盾，超大规模系统通常两者兼用
   - Prefill 节点内部可以用 Chunked Prefill 处理多个请求
   - 关键在于根据负载特性选择合适的粒度

### 进阶问题

5. **如何设计 Token Budget 的大小？**
   - 太小：GPU 利用率低，Prefill chunk 太碎
   - 太大：单 step 延迟高，Decode 请求等待久
   - 实践：通常设为 2048~8192，根据模型大小和 GPU 算力调整

6. **PD 分离系统中，Dispatcher 为什么用 Ray Actor 而非普通对象？**
   - P/D 引擎运行在不同进程/节点，需要通过 RPC 访问共享状态
   - Ray Actor 保证了并发安全（单线程执行模型）

7. **比较 Continuous Batching、Chunked Prefill、PD Disaggregation 三者的关系。**
   - Continuous Batching 是基础：请求完成即退出，新请求随时加入
   - Chunked Prefill 是在此基础上的 P/D 融合优化
   - PD Disaggregation 是系统级的 P/D 分离部署方案
   - 三者可以组合使用

---

## 推荐资源

### 论文

| 论文 | 关键贡献 |
|------|---------|
| [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369) | Chunked Prefill 的开创性工作 |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670) | PD 分离的系统化设计 |
| [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677) | PD 分离 + 硬件异构 |
| [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | KV Cache 中心化存储架构 |

### 开源项目

| 项目 | 说明 |
|------|------|
| [vLLM](https://github.com/vllm-project/vllm) | V1 架构的生产级实现 |
| [SGLang](https://github.com/sgl-project/sglang) | 支持 PD 分离的推理引擎 |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | KV Cache 分离存储系统 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 本教程 PD 分离代码的 Actor 架构参考 |

### 技术博客

| 资源 | 内容 |
|------|------|
| [vLLM V1 技术介绍 (Google Slides)](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/) | vLLM 官方 V1 架构讲解 |
| [深入 Inference: Continue Batching](https://zhuanlan.zhihu.com/p/1974105325897544853) | Continuous Batching 原理与实现 |
