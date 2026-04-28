---
title: "手搓 vLLM 推理引擎"
description: "从零实现 vLLM 核心：PagedAttention、Continuous Batching、KV Cache 管理、CUDA Graph——1200 行代码达到 vLLM 性能"
topics: [nano-vllm, vLLM, PagedAttention, KV-cache, continuous-batching, scheduler, block-manager, CUDA-graph, Triton]
prereqs: [engineering/inference, architecture/attention]
---

# 手搓 vLLM 推理引擎

## 为什么要手搓 vLLM？

vLLM 是目前最主流的 LLM 推理引擎，但它的源码已经膨胀到 **10 万+ 行**——直接读源码，不现实。

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 用 **~1200 行 Python** 重新实现了 vLLM 的核心功能，而且性能不降反升：

| 推理引擎 | 输出 Tokens | 耗时 (s) | 吞吐 (tokens/s) |
|----------|------------|----------|-----------------|
| vLLM | 133,966 | 98.37 | 1361.84 |
| nano-vllm | 133,966 | 93.41 | **1434.13** |

> 测试环境：RTX 4070 Laptop (8GB)，Qwen3-0.6B，256 条请求，输入/输出长度 100~1024 tokens 随机采样。

学完这篇教程，你将理解推理引擎的**每一个核心组件**：

- **PagedAttention**：像操作系统管理内存一样管理 KV Cache
- **Continuous Batching**：动态调度，不等最慢的请求
- **Prefix Caching**：相同前缀的请求共享 KV Cache
- **CUDA Graph**：消除 kernel launch 开销
- **Tensor Parallelism**：多 GPU 并行推理

## 整体架构

```
LLM.generate(prompts)
    |
LLMEngine
    |-- Scheduler (调度器)
    |   |-- BlockManager (KV Cache 块管理)
    |   +-- Sequence (请求状态)
    |-- ModelRunner (模型执行)
    |   |-- prepare_prefill / prepare_decode
    |   |-- run_model (forward + CUDA Graph)
    |   +-- Sampler
    +-- Tokenizer
```

nano-vllm 的文件结构一目了然：

| 文件 | 行数 | 职责 |
|------|------|------|
| `engine/sequence.py` | 83 | 单个请求的状态管理 |
| `engine/block_manager.py` | 112 | KV Cache 物理块分配 + Prefix Caching |
| `engine/scheduler.py` | 71 | Continuous Batching 调度 + 抢占 |
| `engine/model_runner.py` | 251 | 模型执行、KV Cache 分配、CUDA Graph |
| `engine/llm_engine.py` | 93 | 顶层引擎编排 |
| `layers/attention.py` | 75 | PagedAttention (Triton + Flash Attention) |
| `layers/linear.py` | 153 | 张量并行 Linear 层 |
| `layers/rotary_embedding.py` | 61 | RoPE 位置编码 |
| `layers/embed_head.py` | 66 | Embedding + LM Head |
| `layers/sampler.py` | 15 | 采样器 |
| `layers/activation.py` | 14 | SiLU 激活函数 |
| `layers/layernorm.py` | 50 | RMS LayerNorm |
| `models/qwen3.py` | ~215 | Qwen3 模型实现 |
| `utils/context.py` | 28 | 全局 Attention 上下文 |
| `utils/loader.py` | 29 | Safetensors 权重加载 |
| `config.py` | 26 | 配置 |
| `sampling_params.py` | 11 | 采样参数 |
| `llm.py` | 5 | 公共 API |

一共不到 1200 行，我们逐个击破。

## Step 1: Sequence — 请求的生命周期

每一条推理请求在引擎内部被封装为一个 `Sequence` 对象。先看状态机：

```
WAITING ──allocate──> RUNNING ──EOS/max_tokens──> FINISHED
   ^                    |
   +----preempt--------+
```

::: tip 关键概念
一个 Sequence 可以被**抢占** (preempt)——当 GPU 显存不足时，调度器会把正在运行的请求踢回 WAITING 队列，释放它占用的 KV Cache 块。
:::

完整代码：

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 等待分配 KV Cache
    RUNNING = auto()    # 正在推理
    FINISHED = auto()   # 推理完成

class Sequence:
    block_size = 256    # 每个 KV Cache 块容纳 256 个 token

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)     # 全局唯一 ID
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)         # 当前所有 token（prompt + completion）
        self.last_token = token_ids[-1]          # 最后一个 token（decode 阶段只需要这个）
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0               # prefix cache 命中的 token 数
        self.block_table = []                    # 物理块 ID 列表（页表）
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

几个关键属性：

```python
@property
def num_blocks(self):
    # 当前 token 总数需要多少个块（向上取整）
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self):
    # 最后一个块里有多少个 token
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i):
    # 取第 i 个块对应的 token_ids（用于计算 hash，做 prefix caching）
    return self.token_ids[i * self.block_size : (i+1) * self.block_size]
```

::: details 为什么需要 block_table？
`block_table` 就是**页表**——它记录了这个 Sequence 的 KV Cache 分散存储在哪些物理块上。比如 `block_table = [5, 12, 3]` 表示第 0~255 个 token 的 KV 存在物理块 5，第 256~511 的存在物理块 12，以此类推。这正是 PagedAttention 的核心思想：**像操作系统的虚拟内存一样管理 KV Cache**。
:::

还有一个精巧的序列化设计——通过 `__getstate__` / `__setstate__` 实现跨进程通信时的最小化传输：

```python
def __getstate__(self):
    # 如果还没开始生成，传完整 token_ids（需要计算 hash）
    # 如果已经在 decode，只传 last_token（省带宽）
    return (self.num_tokens, self.num_prompt_tokens,
            self.num_cached_tokens, self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)
```

## Step 2: Block Manager — PagedAttention 的核心

Block Manager 是整个推理引擎最关键的组件。它管理 GPU 上的 KV Cache 物理块，并实现了 **Prefix Caching**。

### Block 数据结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id    # 物理块 ID
        self.ref_count = 0          # 引用计数（多个 Sequence 可共享同一块）
        self.hash = -1              # 块内容的 hash（用于 prefix caching）
        self.token_ids = []         # 块内 token（用于验证 hash 碰撞）

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

### BlockManager 初始化

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]  # 所有物理块
        self.hash_to_block_id = dict()       # hash -> block_id 映射表
        self.free_block_ids = deque(range(num_blocks))  # 空闲块队列
        self.used_block_ids = set()          # 已使用块集合
```

### Hash 计算——Prefix Caching 的基础

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 把前一个块的 hash 链进来
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

::: tip Hash 链
每个块的 hash 不仅取决于自身的 token，还包含**前一个块的 hash**。这样形成一条 hash 链：只有前缀完全相同的两个请求，才能匹配到同一个缓存块。这是 Prefix Caching 的核心思想。
:::

### allocate()——分配 KV Cache 块

这是 Block Manager 最核心的方法，同时实现了**普通分配**和 **Prefix Caching**：

```python
def allocate(self, seq: Sequence):
    assert not seq.block_table          # 新请求，还没有页表
    h = -1
    cache_miss = False
    for i in range(seq.num_blocks):
        token_ids = seq.block(i)        # 取第 i 个块的 token
        # 只有满块才计算 hash（不满的块不能缓存）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
        block_id = self.hash_to_block_id.get(h, -1)

        # 检查是否 cache hit：hash 匹配 + token 内容一致（防碰撞）
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # Cache miss：分配一个新的空闲块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # Cache hit：复用已有的块，增加引用计数
            seq.num_cached_tokens += self.block_size
            if block_id in self.used_block_ids:
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                block = self._allocate_block(block_id)

        # 记录满块的 hash，方便未来复用
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id
        seq.block_table.append(block_id)
```

::: warning 重要细节
一旦发生 cache miss，后续所有块都必须重新分配——即使后面的块 hash 碰巧匹配。因为 hash 链断了，KV Cache 的值已经不对了。
:::

### deallocate()——引用计数释放

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:    # 没人用了才真正释放
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

逆序释放是个小技巧——最后的块最可能没有被共享（因为不同请求的后缀往往不同）。

### can_append() / may_append()——Decode 阶段的块管理

Decode 阶段每步只产生一个新 token。大多数时候新 token 可以写入当前最后一个块的空闲位置，只有当块写满时才需要分配新块：

```python
def can_append(self, seq: Sequence) -> bool:
    # 只有当 token 数刚好是 block_size 的倍数 + 1 时，需要一个新块
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    if len(seq) % self.block_size == 1:
        # 当前块刚写满，需要分配新块
        assert last_block.hash != -1    # 上一个块已完成 hash
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif len(seq) % self.block_size == 0:
        # 当前块刚好写满，计算 hash 供未来 prefix caching 使用
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id

    else:
        # 普通情况：直接写入当前块，不需要分配
        assert last_block.hash == -1
```

## Step 3: Scheduler — Continuous Batching 调度

调度器决定**每一步哪些请求参与计算**，这是 Continuous Batching 的核心。

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs          # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大 batch token 数
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting = deque()   # 等待队列
        self.running = deque()   # 运行队列
```

### schedule() 方法——调度策略

调度策略的核心思想：**Prefill 优先**。

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # ========== 阶段 1：尝试 Prefill ==========
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]
        # 检查两个条件：
        # 1. batch token 数不超限
        # 2. Block Manager 有足够的空闲块
        if (num_batched_tokens + len(seq) > self.max_num_batched_tokens
                or not self.block_manager.can_allocate(seq)):
            break
        num_seqs += 1
        self.block_manager.allocate(seq)       # 分配 KV Cache 块
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # prefix cache 命中的部分不算
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, True   # is_prefill = True

    # ========== 阶段 2：Decode ==========
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()
        # 检查是否有足够的块给这个序列追加 token
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())   # 抢占：踢掉最后加入的序列
            else:
                self.preempt(seq)                  # 没人可踢，只能踢自己
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False   # is_prefill = False
```

::: tip Prefill 优先的原因
Prefill 处理的是新请求的 prompt。如果一直让 decode 跑而不处理新请求，waiting 队列会越来越长，延迟会不可控。所以只要有新请求能塞进去，就优先 prefill。
:::

### 抢占机制

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放它的 KV Cache
    self.waiting.appendleft(seq)        # 放回 waiting 队列头部（优先重新调度）
```

被抢占的序列下次需要重新做 prefill（因为 KV Cache 已经被释放了）。这是 nano-vllm 的简化策略——vLLM 还支持 swap 到 CPU 内存。

### postprocess()——检查终止条件

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        # 两种终止条件：遇到 EOS 或达到最大生成长度
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

## Step 4: Attention — Triton Kernel + Flash Attention

Attention 层是推理引擎与模型计算的交汇点。nano-vllm 用一个 Triton kernel 写 KV Cache，用 Flash Attention 做注意力计算。

### Triton Kernel：写入 KV Cache

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,        # num_heads * head_dim
):
    idx = tl.program_id(0)  # 第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return    # CUDA Graph padding 的无效 token

    # 从 key/value 张量读取
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 写入 KV Cache 的对应 slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

::: details slot_mapping 是什么？
`slot_mapping` 把每个 token 的逻辑位置映射到 KV Cache 的物理位置。计算公式：`slot = block_table[block_idx] * block_size + offset_in_block`。这就是 PagedAttention 的"页表查询"过程。
:::

### Attention Forward：三条路径

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])  # 由 ModelRunner 注入

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Step 1: 把当前 K, V 写入 KV Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                # 路径 A：Prefix Cache 命中
                # 部分 token 的 KV 已在 cache 中，用 block_table 寻址
                k, v = k_cache, v_cache
            # 路径 B：普通 Prefill（或 Prefix Cache Prefill）
            # 用 variable-length Flash Attention 处理不等长序列
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,  # 每个序列的 query 累积长度
                cu_seqlens_k=context.cu_seqlens_k,  # 每个序列的 key 累积长度
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            # 路径 C：Decode
            # 每个序列只有 1 个新 query，attention 的 KV 全部从 cache 中读取
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),         # [batch, 1, heads, dim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个序列的上下文长度
                block_table=context.block_tables,    # 页表
                softmax_scale=self.scale,
                causal=True,
            )
        return o
```

::: warning Prefill 和 Decode 的关键区别
- **Prefill**：一个序列有很多 query token（整个 prompt），用 `flash_attn_varlen_func` 批量处理
- **Decode**：每个序列只有 1 个 query token（刚生成的），用 `flash_attn_with_kvcache` 直接从 paged KV Cache 读取
- **Prefix Cache**：Prefill 但 `cu_seqlens_k > cu_seqlens_q`，说明部分 KV 已在 cache 中，需要通过 block_table 寻址
:::

## Step 5: Model Runner — 模型执行与 CUDA Graph

Model Runner 是最复杂的组件（251 行），负责：模型初始化、KV Cache 分配、输入准备、CUDA Graph 捕获、多 GPU 通信。

### 初始化流程

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # 1. 初始化分布式通信
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 2. 加载模型
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # 3. 探测 GPU 内存，计算能分配多少 KV Cache 块
        self.warmup_model()
        self.allocate_kv_cache()

        # 4. 捕获 CUDA Graph（可选）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 5. 多 GPU 通信：rank 0 为 master，其他 rank 进入 loop 等待指令
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            else:
                self.shm = SharedMemory(name="nanovllm")
                self.loop()   # 非 master 进程进入事件循环
```

### warmup_model()——探测峰值显存

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # 用最大 batch 跑一次 forward，测量峰值显存
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
    self.run(seqs, True)     # dummy forward
    torch.cuda.empty_cache()
```

::: tip 为什么要 warmup？
因为 PyTorch 的 CUDA 内存分配器会缓存内存。warmup 让分配器预先分配好最大所需内存，之后通过 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` 就能精确知道模型 forward 的峰值内存，从而计算出剩余显存能放多少 KV Cache 块。
:::

### allocate_kv_cache()——计算可用块数

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 每个块的内存：2(K+V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim",
                       hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = (2 * hf_config.num_hidden_layers * self.block_size
                   * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize)

    # 可用显存 = 总显存 * 利用率 - 已用 - (峰值 - 当前)
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization - used - peak + current
    ) // block_bytes

    # 分配 KV Cache 张量：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.empty(
        2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
        self.block_size, num_kv_heads, head_dim
    )

    # 把 KV Cache 的切片注入到每一层 Attention 模块
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

::: details KV Cache 的内存布局
KV Cache 是一个 6 维张量 `[2, L, B, S, H, D]`：
- `2`：K 和 V
- `L`：Transformer 层数
- `B`：物理块数
- `S`：block_size（每块 256 个 slot）
- `H`：KV head 数
- `D`：head 维度

每个 slot 存储一个 token 在一层的 KV 向量。Triton kernel 的 `slot_mapping` 就是定位到 `[B, S]` 维度的展平索引。
:::

### prepare_prefill()——构建 Prefill 输入

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]     # query 累积序列长度（Flash Attention 需要）
    cu_seqlens_k = [0]     # key 累积序列长度
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:
        seqlen = len(seq)
        # Prefix Cache 命中的 token 不需要重新计算
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))

        seqlen_q = seqlen - seq.num_cached_tokens   # 实际要算的 query 长度
        seqlen_k = seqlen                            # key 长度（包含 cache）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        # 计算 slot_mapping：跳过已 cache 的块
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))

    # 如果有 prefix cache，需要传 block_tables 给 Flash Attention
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    # 转 tensor 并上传 GPU（用 pin_memory + non_blocking 加速传输）
    input_ids = torch.tensor(input_ids, dtype=torch.int64,
                             pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64,
                             pin_memory=True).cuda(non_blocking=True)
    # ... 其余类似
    set_context(True, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
    return input_ids, positions
```

### prepare_decode()——Decode 输入更简单

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []
    for seq in seqs:
        input_ids.append(seq.last_token)           # 只需要最后一个 token
        positions.append(len(seq) - 1)             # 位置 = 序列长度 - 1
        context_lens.append(len(seq))              # 上下文长度（Flash Attention 需要）
        # slot = 最后一个块的起始位置 + 块内偏移
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )
    # ... 转 tensor 上传 GPU
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)
    return input_ids, positions
```

### CUDA Graph——消除 Decode 阶段的 Launch 开销

Decode 阶段每个序列只处理 1 个 token，计算量很小，但每次 forward 会启动很多小 CUDA kernel。CUDA Graph 把整个 forward 的 kernel 序列**预先录制**下来，之后每次 replay 只需要一次 launch。

```python
def capture_cudagraph(self):
    max_bs = min(self.config.max_num_seqs, 512)
    # 预分配固定大小的输入/输出 buffer
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 为不同 batch size 各捕获一个 graph
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        self.graphs[bs] = graph
```

::: tip 为什么要多个 batch size？
CUDA Graph 的输入形状在录制时就固定了。但实际 decode 的 batch size 会变化（有请求完成退出，有新请求加入）。所以预先录制 `[1, 2, 4, 8, 16, 32, ..., 512]` 这些 batch size 的 graph，运行时选择最小的够用的那个。
:::

执行时的 graph replay：

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill 或超大 batch：直接 eager 执行
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Decode：用 CUDA Graph
        bs = input_ids.size(0)
        context = get_context()
        # 找到最小的 >= bs 的预录 batch size
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        # 把真实数据拷贝到 graph 的输入 buffer
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)    # -1 表示无效（Triton kernel 会跳过）
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()   # 一次 launch 执行整个 forward
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 多 GPU 通信——SharedMemory

nano-vllm 用一种极简方案做多 GPU 通信：rank 0 通过 `SharedMemory` 把方法名和参数序列化后写入共享内存，其他 rank 通过 `Event` 同步读取并执行：

```python
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()    # 通知所有 worker

def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)  # master 广播指令
    method = getattr(self, method_name, None)
    return method(*args)                     # 本地执行
```

## Step 6: LLM Engine — 把一切串起来

LLM Engine 是最顶层的编排器，把 Scheduler、ModelRunner、Tokenizer 组合在一起。

### 初始化——启动多进程

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **config_kwargs)
        # 启动 worker 进程（rank 1, 2, ...）
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # rank 0 在主进程
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
```

### 主循环——generate()

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    # 1. 把所有请求加入 scheduler
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    # 2. 循环直到所有请求完成
    outputs = {}
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()
        # 统计吞吐
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    # 3. 按 seq_id 排序输出，decode 成文本
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(tids), "token_ids": tids}
               for tids in outputs]
    return outputs
```

### step()——单步执行

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()          # 调度
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # 执行
    self.scheduler.postprocess(seqs, token_ids)           # 后处理
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in seqs if seq.is_finished]
    # num_tokens > 0 表示 prefill，< 0 表示 decode（取负用于区分）
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

::: tip 吞吐统计的巧妙设计
`num_tokens` 的正负号区分了 prefill 和 decode：正数是 prefill 总 token 数，负数的绝对值是 decode 的序列数（每个序列每步生成 1 个 token）。这样外层就能分别统计 prefill tok/s 和 decode tok/s。
:::

## Step 7: 模型层实现

### Tensor Parallel Linear

nano-vllm 实现了完整的张量并行 Linear 层体系：

| 类 | 用途 | 切分方式 |
|---|------|---------|
| `ReplicatedLinear` | 不切分，每个 GPU 有完整副本 | 无 |
| `ColumnParallelLinear` | 按列切分输出维度 | `output_size / tp_size` |
| `RowParallelLinear` | 按行切分输入维度 + all_reduce | `input_size / tp_size` |
| `QKVParallelLinear` | QKV 联合切分 | 按 head 数切分 |
| `MergedColumnParallelLinear` | gate + up 合并切分 | 每个子矩阵独立切分 |

Megatron 风格的 TP：Attention 中 QKV 用 Column Parallel（切分 head），O projection 用 Row Parallel（最后 all_reduce）；MLP 中 gate_up 用 Merged Column Parallel，down 用 Row Parallel。

```python
# RowParallelLinear 的 forward：先乘局部矩阵，再 all_reduce
def forward(self, x):
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)    # 把所有 GPU 的局部结果求和
    return y
```

### RoPE 位置编码

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin    # 旋转公式
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

nano-vllm 预计算所有位置的 `cos`/`sin` 表，推理时只需查表。用 `@torch.compile` 加速。

### RMS LayerNorm

```python
class RMSNorm(nn.Module):
    @torch.compile
    def rms_forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight)

    @torch.compile
    def add_rms_forward(self, x, residual):
        # 融合 residual add + RMSNorm，减少一次显存读写
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.to(orig_dtype).mul_(self.weight), residual
```

`add_rms_forward` 是一个常见优化——把 residual add 和 LayerNorm 融合在一起，减少一次显存读写。

### Sampler——Gumbel-Max 采样

```python
class Sampler(nn.Module):
    @torch.compile
    def forward(self, logits, temperatures):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-Max trick：等价于 multinomial sampling，但更适合 GPU 并行
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
```

::: tip Gumbel-Max Trick
传统的 `torch.multinomial` 在 GPU 上效率不高。Gumbel-Max 技巧把采样转化为 `argmax(log(probs) + gumbel_noise)`，等价于 `argmax(probs / exponential_noise)`，全程都是 element-wise 操作，非常适合 `@torch.compile` 融合优化。
:::

### Qwen3 模型组装

```python
class Qwen3DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        # Pre-Norm: LayerNorm -> Attention -> LayerNorm -> MLP
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

## nano-vllm vs vLLM：简化了什么？

| 特性 | vLLM | nano-vllm |
|------|------|-----------|
| 代码量 | 10 万+ 行 | ~1200 行 |
| Chunked Prefill | 支持 | 不支持 |
| 投机采样 (Speculative Decoding) | 支持 | 不支持 |
| PD 分离 (Disaggregated Prefill/Decode) | 支持 | 不支持 |
| Prefix Caching | 支持 | 支持 |
| CUDA Graph | 支持 | 支持 |
| Tensor Parallelism | 支持 | 支持 |
| Continuous Batching | 支持 | 支持 |
| PagedAttention | 支持 | 支持 |
| 在线服务 (HTTP API) | 支持 | 不支持 |
| Swap 到 CPU | 支持 | 不支持（抢占后 re-prefill） |
| 多模型架构 | 支持 | 仅 Qwen3 |

nano-vllm 的核心取舍：**砍掉了工程复杂度，保留了性能关键路径**。Chunked Prefill、投机采样等都是"锦上添花"的优化，不影响核心架构理解。

## 苏格拉底时刻

1. **PagedAttention 的 block_size 应该设多大？** block_size 太小会导致 block_table 很长（overhead 大），太大会导致内存碎片（最后一个块浪费多）。nano-vllm 用 256，vLLM 默认 16。你觉得哪种更合理？分别在什么场景下有优势？

2. **抢占策略为什么踢最后加入的序列？** `self.running.pop()` 踢的是最近加入 running 的序列。这跟操作系统的哪种页面置换策略类似？如果改成踢最长的序列会怎样？

3. **为什么 Prefix Caching 用 hash 链而不是 Trie 树？** Trie 树也能做前缀匹配。用 hash 链有什么优势和劣势？在什么情况下会出问题？

4. **CUDA Graph 为什么不能用于 Prefill？** 提示：思考 Prefill 的 batch 内序列长度变化情况，以及 CUDA Graph 对输入形状的要求。

5. **如果两个请求的前缀完全相同但 sampling temperature 不同，它们能共享 KV Cache 吗？** 为什么？这揭示了 KV Cache 的什么性质？

## 面试考点

**Q: 什么是 PagedAttention？解决了什么问题？**

A: PagedAttention 借鉴操作系统虚拟内存的思想，把 KV Cache 分成固定大小的物理块，用页表（block_table）管理逻辑块到物理块的映射。解决了两个问题：(1) 内存碎片——不需要为每个请求预分配最大长度的连续内存；(2) 内存共享——相同前缀的请求可以共享物理块（Prefix Caching）。

**Q: Continuous Batching 和 Static Batching 有什么区别？**

A: Static Batching 要等一个 batch 中最慢的请求完成才能处理下一个 batch。Continuous Batching 允许已完成的请求立即退出，新请求立即加入，GPU 利用率更高。在 nano-vllm 中，每次 `schedule()` 都会重新构建当前的活跃序列集合。

**Q: 为什么 Decode 阶段适合用 CUDA Graph 而 Prefill 不适合？**

A: Decode 阶段每个序列只处理 1 个 token，计算量小但 kernel launch 开销相对大。CUDA Graph 把多个 kernel 录制成一个 graph，一次 launch 执行所有 kernel。但 CUDA Graph 要求输入张量的形状固定——Decode 阶段的 batch size 只有少数几种可能，可以预先录制；Prefill 阶段每个序列的 prompt 长度不同，不适合 CUDA Graph。

**Q: Prefix Caching 是如何实现的？**

A: 每个完整的 KV Cache 块计算一个 hash（包含块内 token 和前一个块的 hash，形成 hash 链）。新请求分配块时，先查 hash 表：如果命中且 token 内容一致，就复用已有的物理块（增加引用计数），跳过这些 token 的计算。hash 链保证只有前缀完全相同的请求才能匹配。

**Q: Tensor Parallelism 中 Column Parallel 和 Row Parallel 如何配合？**

A: 遵循 Megatron 的设计：Column Parallel 把权重按列切分到各 GPU，每个 GPU 独立计算局部结果；Row Parallel 把权重按行切分，计算完后 all_reduce 汇总。在 Transformer 中，QKV projection 用 Column Parallel（按 head 切分），O projection 用 Row Parallel（最后一次 all_reduce 就够了）。这样每个 Attention 层只需要一次 all_reduce。

## 推荐资源

- [nano-vllm 源码](https://github.com/GeeeekExplorer/nano-vllm) — 本教程的核心参考
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash Attention 论文
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Tensor Parallelism 原始论文
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) — 学习自定义 GPU kernel
