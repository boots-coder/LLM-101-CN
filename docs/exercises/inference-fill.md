---
title: "推理优化填空"
description: "Level 2-3 填空：KV Cache、PagedAttention、Continuous Batching、投机采样"
topics: [fill-in, KV-cache, PagedAttention, continuous-batching, speculative-decoding]
---
# 推理优化代码填空 (Level 2-3)

> 本练习覆盖 LLM 推理优化的核心技术：KV Cache、PagedAttention、Continuous Batching、投机解码、Chunked Prefill。
> 每道题给出代码框架，用 `_____` 标记需要填写的部分。

---

## 练习 1：KV Cache 更新（Level 2）

### 背景

在 Decoder 模型推理中，decode 阶段每一步只输入一个新 token，但需要与所有历史 token 的 K、V 做注意力计算。KV Cache 将历史 K、V 缓存起来，每步只计算新 token 的 k、v 并拼接到缓存中，避免重复计算。

### 任务

填写 `CachedAttention` 中 decode 阶段 KV Cache 的拼接与使用逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CachedAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.k_cache = None  # [bsz, cached_len, dim]
        self.v_cache = None

    def forward(self, x, mask):
        """
        x: [bsz, seq_len, dim]  (decode 阶段 seq_len=1)
        mask: [max_len, max_len] 下三角 causal mask
        """
        q = self.q_proj(x)
        k_new = self.k_proj(x)
        v_new = self.v_proj(x)

        # ===== 填空 1: KV Cache 更新（首步初始化 / 后续 append）=====
        if self.k_cache is None:
            self.k_cache, self.v_cache = k_new, v_new
        else:
            self.k_cache = _____  # 提示: torch.cat 沿 seq 维度拼接
            self.v_cache = _____

        # ===== 填空 2: 使用缓存的 K, V 计算注意力 =====
        scores = q @ _____.transpose(-2, -1) / math.sqrt(self.dim)

        cached_len = self.k_cache.shape[1]
        if cached_len > 1:
            mask = mask[cached_len - 1, :cached_len].unsqueeze(0).unsqueeze(1)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out = attn @ _____  # 提示: 用缓存的 V
        return self.out_proj(out)
```

### 提示

- KV Cache 的拼接使用 `torch.cat`，沿 `dim=1`（seq_len 维度）
- 计算注意力时，Q 来自当前输入（可能只有 1 个 token），K 和 V 来自缓存（包含所有历史 token）

<details>
<summary>参考答案</summary>

```python
# 填空 1
self.k_cache = torch.cat([self.k_cache, k_new], dim=1)
self.v_cache = torch.cat([self.v_cache, v_new], dim=1)

# 填空 2
scores = q @ self.k_cache.transpose(-2, -1) / math.sqrt(self.dim)
out = attn @ self.v_cache
```

</details>

---

## 练习 2：PagedAttention 块表管理（Level 2）

### 背景

vLLM 使用 PagedAttention 技术，将 KV Cache 按固定大小的 block（页）进行管理，类似操作系统的虚拟内存。每个请求维护一个 BlockTable，记录该请求使用了哪些物理 block。当请求需要更多空间时，从空闲池中分配新 block。

### 任务

实现 BlockTable 的 `allocate`（分配初始 blocks）和 `append_token`（追加 token 时按需分配新 block）逻辑。

```python
from typing import List, Set

class BlockAllocator:
    """物理块分配器，管理空闲块池"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks: Set[int] = set(range(num_blocks))

    def allocate(self) -> int:
        """分配一个空闲块，返回块 ID"""
        if not self.free_blocks:
            raise RuntimeError("没有空闲块可分配")
        return self.free_blocks.pop()

    def free(self, block_id: int):
        self.free_blocks.add(block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


class BlockTable:
    """单个请求的块表"""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_ids: List[int] = []
        self.num_tokens: int = 0

    def allocate_for_prompt(self, prompt_len: int):
        """
        为 prompt 分配所需的 blocks
        例如: block_size=4, prompt_len=10 -> 需要 3 个 block
        """
        # ===== 填空 1: 计算需要的 block 数量 =====
        num_blocks_needed = _____  # 提示: 向上取整

        # ===== 填空 2: 逐个分配 block =====
        for _ in range(num_blocks_needed):
            block_id = _____  # 提示: 从 allocator 获取
            self.block_ids.append(block_id)

        self.num_tokens = prompt_len

    def append_token(self):
        """
        追加一个 token，如果当前最后一个 block 已满，需要分配新 block
        """
        # ===== 填空 3: 判断是否需要新 block =====
        if _____:  # 提示: 当前 token 数是 block_size 的整数倍时需要新 block
            new_block_id = self.allocator.allocate()
            self.block_ids.append(new_block_id)

        self.num_tokens += 1

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)
```

### 提示

- 向上取整可用 `(a + b - 1) // b` 或 `math.ceil(a / b)`
- 当 `num_tokens % block_size == 0` 时，说明当前所有 block 都已填满，需要分配新 block

<details>
<summary>参考答案</summary>

```python
# 填空 1: 向上取整
num_blocks_needed = (prompt_len + self.allocator.block_size - 1) // self.allocator.block_size

# 填空 2: 从 allocator 分配
block_id = self.allocator.allocate()

# 填空 3: 判断当前 block 是否已满
if self.num_tokens % self.allocator.block_size == 0:
```

**验证:**
```python
allocator = BlockAllocator(num_blocks=1024, block_size=4)
bt = BlockTable(allocator)
bt.allocate_for_prompt(prompt_len=10)
print(bt.num_blocks)    # 3
print(bt.num_tokens)    # 10

bt.append_token()
print(bt.num_blocks)    # 3 (第3个block还有空间: 10%4=2, 容量4)
print(bt.num_tokens)    # 11

bt.append_token()       # 12%4==0，需要新 block
print(bt.num_blocks)    # 4
```

</details>

---

## 练习 3：Continuous Batching 调度器（Level 3）

### 背景

Continuous Batching 的核心是调度器的 `step()` 函数。每一步中：
1. 先对正在运行的请求执行 decode（生成下一个 token）
2. 检查是否有空闲 slot，若有则从等待队列中取出请求做 prefill
3. 已完成的请求释放 slot，新请求可以随时加入

### 任务

实现 `ContinueBatchingEngine` 的 `step()` 函数。

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class CBConfig:
    max_batch_size: int = 4
    max_seq_len: int = 32
    dim: int = 16
    num_heads: int = 2
    num_layers: int = 3
    vocab_size: int = 20

class Request:
    def __init__(self, request_id: int, prompt: List[int], max_len: int):
        self.request_id = request_id
        self.prompt = prompt
        self.generated_tokens: List[int] = []
        self.max_len = max_len

    def add_token(self, token: int):
        self.generated_tokens.append(token)

    def is_finished(self) -> bool:
        return len(self.prompt) + len(self.generated_tokens) >= self.max_len

    def get_full_sequence(self) -> List[int]:
        return self.prompt + self.generated_tokens


class ContinueBatchingEngine:
    def __init__(self, config):
        self.config = config
        self.pending_requests: List[Request] = []   # 等待队列
        self.running_requests: dict = {}             # slot_id -> Request
        self.next_request_id = 0
        self.max_batch_size = config.max_batch_size

    def add_request(self, prompt: List[int], max_len: int) -> int:
        req = Request(self.next_request_id, prompt, max_len)
        self.pending_requests.append(req)
        self.next_request_id += 1
        return req.request_id

    def step(self):
        """
        一次调度步骤:
        1. 对所有 running 请求做 decode
        2. 移除已完成的请求
        3. 从 pending 中取出请求做 prefill，填满空闲 slot
        """
        finished_ids = []

        # ===== 填空 1: Decode 阶段 — 为每个 running 请求生成一个 token =====
        for slot_id, req in self.running_requests.items():
            # 模拟生成一个 token (实际应调用模型)
            next_token = len(req.get_full_sequence()) % self.config.vocab_size
            _____  # 提示: 将 token 加入请求

            # ===== 填空 2: 检查请求是否完成 =====
            if _____:
                finished_ids.append(slot_id)

        # ===== 填空 3: 释放已完成请求的 slot =====
        for slot_id in finished_ids:
            print(f"[完成] Request {self.running_requests[slot_id].request_id}")
            _____  # 提示: 从 running_requests 中删除

        # ===== 填空 4: Prefill 阶段 — 将 pending 请求填入空闲 slot =====
        available_slots = _____  # 提示: 计算当前空闲 slot 数量
        num_to_prefill = min(available_slots, len(self.pending_requests))

        for _ in range(num_to_prefill):
            req = self.pending_requests.pop(0)
            # 找一个空闲 slot
            used_slots = set(self.running_requests.keys())
            free_slot = None
            for s in range(self.max_batch_size):
                if s not in used_slots:
                    free_slot = s
                    break
            # ===== 填空 5: 将请求放入 running =====
            _____  # 提示: 在 running_requests 中注册
            print(f"[Prefill] Request {req.request_id} -> slot {free_slot}")

    def get_status(self):
        return (f"pending:{len(self.pending_requests)}"
                f"/running:{len(self.running_requests)}")
```

### 提示

- `add_token` 是 `Request` 的方法
- `is_finished()` 返回布尔值
- `del dict[key]` 或 `dict.pop(key)` 可删除字典元素
- 空闲 slot 数 = `max_batch_size - len(running_requests)`

<details>
<summary>参考答案</summary>

```python
# 填空 1: 将 token 加入请求
req.add_token(next_token)

# 填空 2: 检查请求是否完成
if req.is_finished():

# 填空 3: 释放 slot
del self.running_requests[slot_id]

# 填空 4: 计算空闲 slot
available_slots = self.max_batch_size - len(self.running_requests)

# 填空 5: 注册到 running
self.running_requests[free_slot] = req
```

**验证:**
```python
config = CBConfig()
engine = ContinueBatchingEngine(config)

engine.add_request([1, 2, 3], max_len=6)
engine.add_request([4, 5], max_len=5)
engine.add_request([6, 7, 8, 9], max_len=7)

for step in range(10):
    print(f"\n--- Step {step} --- {engine.get_status()}")
    engine.step()
```

</details>

---

## 练习 4：投机采样验证（Level 3）

### 背景

Speculative Decoding 使用一个小的 draft 模型快速猜测 N 个 token，然后用大的 target 模型一次性验证。验证时，逐个比较 draft 和 target 的预测：
- 若一致，则接受
- 若不一致，以 target 的预测替换，后续 draft token 全部丢弃

高级版本（Speculative Sampling）使用概率比较：以概率 `min(1, q/p)` 接受 token，其中 `q` 是 target 概率，`p` 是 draft 概率。

### 任务

实现投机采样中 draft token 的接受/拒绝判断逻辑。

```python
import torch
import torch.nn.functional as F

def speculative_sampling_verify(
    draft_tokens: torch.Tensor,      # [spec_n] draft 模型猜测的 token ids
    draft_logits: torch.Tensor,      # [spec_n, vocab_size] draft 的 logits
    target_logits: torch.Tensor,     # [spec_n+1, vocab_size] target 的 logits
    spec_n: int,
):
    """
    返回: accepted_tokens (List[int]), 接受的 token 列表
    """
    accepted_tokens = []

    for j in range(spec_n):
        # ===== 填空 1: 计算 target 和 draft 在该位置的概率分布 =====
        q = _____  # 提示: target_logits[j] 做 softmax
        p = _____  # 提示: draft_logits[j] 做 softmax

        token_id = draft_tokens[j].item()

        # ===== 填空 2: 采样随机数并判断是否接受 =====
        r = torch.rand(1).item()
        accept_prob = _____  # 提示: min(1, q[token_id] / p[token_id])

        if r < accept_prob:
            # ===== 填空 3: 接受该 token =====
            _____
        else:
            # 拒绝: 从修正分布中重新采样
            q_adjusted = q.clone()
            # ===== 填空 4: 修正分布 max(0, q - p) =====
            q_adjusted = _____
            q_adjusted = q_adjusted / q_adjusted.sum()  # 归一化

            new_token = torch.multinomial(q_adjusted, num_samples=1).item()
            accepted_tokens.append(new_token)
            break  # 拒绝后停止验证

    # ===== 填空 5: 如果全部接受，额外采样一个 bonus token =====
    if len(accepted_tokens) == spec_n:
        bonus_probs = _____  # 提示: target_logits 最后一个位置的 softmax
        bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
        accepted_tokens.append(bonus_token)

    return accepted_tokens
```

### 提示

- `F.softmax(logits, dim=-1)` 将 logits 转为概率
- 接受概率为 `min(1, q[token_id] / p[token_id])`
- 拒绝时的修正分布为 `max(0, q - p)`，即 `torch.clamp(q - p, min=0)`
- 全部接受时，target 模型多算了一个位置（`spec_n+1`），可直接采样 bonus token

<details>
<summary>参考答案</summary>

```python
# 填空 1
q = F.softmax(target_logits[j], dim=-1)
p = F.softmax(draft_logits[j], dim=-1)

# 填空 2
accept_prob = min(1, (q[token_id] / p[token_id]).item())

# 填空 3
accepted_tokens.append(token_id)

# 填空 4
q_adjusted = torch.clamp(q - p, min=0)

# 填空 5
bonus_probs = F.softmax(target_logits[-1], dim=-1)
```

**验证:**
```python
vocab_size = 100
spec_n = 5

draft_tokens = torch.tensor([19, 30, 62, 70, 20])
draft_logits = torch.randn(spec_n, vocab_size)
target_logits = torch.randn(spec_n + 1, vocab_size)

result = speculative_sampling_verify(draft_tokens, draft_logits, target_logits, spec_n)
print(f"接受 {len(result)} 个 token: {result}")
# 每次运行结果不同（随机采样），但 len(result) >= 1
```

</details>

---

## 练习 5：Chunked Prefill 调度（Level 3）

### 背景

当 prompt 很长时，一次性 prefill 会占用大量计算资源，导致同 batch 的 decode 请求被阻塞。Chunked Prefill 将长 prompt 拆分为固定大小的 chunk，每个调度步骤只处理一个 chunk，从而与 decode 请求交替执行，降低延迟。

### 任务

实现 Chunked Prefill 的调度逻辑：将长 prefill 拆分为 chunk，并与 decode 请求混合调度。

```python
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class ChunkedRequest:
    request_id: int
    prompt_tokens: List[int]
    chunk_size: int
    prefilled_len: int = 0      # 已 prefill 的 token 数
    generated_tokens: List[int] = field(default_factory=list)
    max_gen_len: int = 20

    @property
    def is_prefill_done(self) -> bool:
        """prompt 是否已全部 prefill"""
        return self.prefilled_len >= len(self.prompt_tokens)

    @property
    def is_finished(self) -> bool:
        return (self.is_prefill_done and
                len(self.generated_tokens) >= self.max_gen_len)


class ChunkedPrefillScheduler:
    def __init__(self, chunk_size: int = 128, max_batch_tokens: int = 512):
        self.chunk_size = chunk_size
        self.max_batch_tokens = max_batch_tokens  # 每步最大 token 预算
        self.prefilling: List[ChunkedRequest] = []  # 正在分块 prefill 的请求
        self.decoding: List[ChunkedRequest] = []    # decode 阶段的请求
        self.pending: List[ChunkedRequest] = []     # 等待队列

    def add_request(self, request_id: int, prompt: List[int]):
        req = ChunkedRequest(
            request_id=request_id,
            prompt_tokens=prompt,
            chunk_size=self.chunk_size,
        )
        self.pending.append(req)

    def schedule_step(self):
        """
        调度一步:
        1. decode 请求各占 1 token 预算
        2. 剩余预算分配给 prefill chunk
        返回: (decode_requests, prefill_chunks)
              prefill_chunks: List[(request, start, end)]
        """
        budget = self.max_batch_tokens
        decode_requests = []
        prefill_chunks = []

        # ===== 填空 1: decode 请求占预算，每个请求 1 token =====
        for req in self.decoding:
            if budget >= 1:
                decode_requests.append(req)
                _____  # 提示: 减少预算

        # 将 pending 移入 prefilling
        while self.pending and budget >= self.chunk_size:
            self.prefilling.append(self.pending.pop(0))

        # ===== 填空 2: 为 prefilling 请求分配 chunk =====
        still_prefilling = []
        for req in self.prefilling:
            if budget <= 0:
                still_prefilling.append(req)
                continue

            # 计算本次 chunk 的起止位置
            start = req.prefilled_len
            # ===== 填空 3: 计算 chunk 结束位置 =====
            end = _____  # 提示: min(三个值: start+chunk_size, prompt总长, start+剩余预算)

            prefill_chunks.append((req, start, end))

            # ===== 填空 4: 更新已 prefill 长度和预算 =====
            chunk_len = end - start
            _____  # 提示: 更新 req.prefilled_len
            _____  # 提示: 更新 budget

            # ===== 填空 5: 如果 prefill 完成，转入 decoding =====
            if req.is_prefill_done:
                _____  # 提示: 加入 self.decoding
            else:
                still_prefilling.append(req)

        self.prefilling = still_prefilling

        return decode_requests, prefill_chunks
```

### 提示

- decode 每个请求消耗 1 token 预算（因为 decode 阶段每步只生成 1 个 token）
- chunk 的 end 不能超过 prompt 总长度，也不能超过 `start + chunk_size`，还受限于剩余预算
- prefill 完成后，请求从 `prefilling` 列表转移到 `decoding` 列表

<details>
<summary>参考答案</summary>

```python
# 填空 1
budget -= 1

# 填空 2 (结构性，无需填)

# 填空 3
end = min(start + self.chunk_size, len(req.prompt_tokens), start + budget)

# 填空 4
req.prefilled_len = end
budget -= chunk_len

# 填空 5
self.decoding.append(req)
```

**验证:**
```python
scheduler = ChunkedPrefillScheduler(chunk_size=128, max_batch_tokens=512)

# 添加一个长 prompt (500 tokens) 和一个短 prompt (50 tokens)
scheduler.add_request(0, list(range(500)))
scheduler.add_request(1, list(range(50)))

for step in range(6):
    decode_reqs, prefill_chunks = scheduler.schedule_step()
    print(f"\nStep {step}:")
    print(f"  Decode: {[r.request_id for r in decode_reqs]}")
    for req, s, e in prefill_chunks:
        print(f"  Prefill: req={req.request_id}, tokens[{s}:{e}] ({e-s} tokens)")
    print(f"  Status: prefilling={len(scheduler.prefilling)}, "
          f"decoding={len(scheduler.decoding)}, "
          f"pending={len(scheduler.pending)}")

# 预期输出:
# Step 0: Prefill req=0 tokens[0:128], Prefill req=1 tokens[0:50]
# Step 1: Prefill req=0 tokens[128:256], decode req=1
# Step 2: Prefill req=0 tokens[256:384], decode req=1
# Step 3: Prefill req=0 tokens[384:500], decode req=1
# Step 4: decode req=0, req=1
```

</details>

---

## 总结

| 练习 | 难度 | 核心知识点 |
|------|------|-----------|
| KV Cache 更新 | Level 2 | decode 阶段 KV 拼接，Q 与缓存 KV 的注意力计算 |
| PagedAttention 块表 | Level 2 | 物理块分配、向上取整、按需扩展 |
| Continuous Batching | Level 3 | prefill/decode 混合调度、slot 管理、请求生命周期 |
| 投机采样验证 | Level 3 | 概率比较、接受/拒绝、修正分布重采样 |
| Chunked Prefill | Level 3 | token 预算、chunk 拆分、prefill-decode 混合调度 |

---

## MLM 代码训练模式

完成上面的固定填空后，试试随机挖空模式 -- 每次点击「刷新」会随机遮盖不同的代码片段，帮你彻底记住每一行。

### KV Cache 实现

<CodeMasker title="KV Cache：decode 阶段拼接与注意力计算" :mask-ratio="0.15">
def forward(self, x, mask):
    q = self.q_proj(x)
    k_new, v_new = self.k_proj(x), self.v_proj(x)

    if self.k_cache is None:
        self.k_cache, self.v_cache = k_new, v_new
    else:
        self.k_cache = torch.cat([self.k_cache, k_new], dim=1)
        self.v_cache = torch.cat([self.v_cache, v_new], dim=1)

    scores = q @ self.k_cache.transpose(-2, -1) / math.sqrt(self.dim)
    scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    out = attn @ self.v_cache
    return self.out_proj(out)
</CodeMasker>

### 投机采样验证

<CodeMasker title="Speculative Decoding：draft 验证与修正分布重采样" :mask-ratio="0.15">
def speculative_sampling_verify(draft_tokens, draft_logits, target_logits, spec_n):
    accepted_tokens = []

    for j in range(spec_n):
        q = F.softmax(target_logits[j], dim=-1)
        p = F.softmax(draft_logits[j], dim=-1)
        token_id = draft_tokens[j].item()

        r = torch.rand(1).item()
        accept_prob = min(1, (q[token_id] / p[token_id]).item())

        if r < accept_prob:
            accepted_tokens.append(token_id)
        else:
            q_adjusted = torch.clamp(q - p, min=0)
            q_adjusted = q_adjusted / q_adjusted.sum()
            new_token = torch.multinomial(q_adjusted, num_samples=1).item()
            accepted_tokens.append(new_token)
            break

    if len(accepted_tokens) == spec_n:
        bonus_probs = F.softmax(target_logits[-1], dim=-1)
        bonus_token = torch.multinomial(bonus_probs, num_samples=1).item()
        accepted_tokens.append(bonus_token)

    return accepted_tokens
</CodeMasker>

### Continuous Batching 调度

<CodeMasker title="Continuous Batching：decode + prefill 混合调度 step" :mask-ratio="0.15">
def step(self):
    finished_ids = []

    for slot_id, req in self.running_requests.items():
        next_token = len(req.get_full_sequence()) % self.config.vocab_size
        req.add_token(next_token)
        if req.is_finished():
            finished_ids.append(slot_id)

    for slot_id in finished_ids:
        del self.running_requests[slot_id]

    available_slots = self.max_batch_size - len(self.running_requests)
    num_to_prefill = min(available_slots, len(self.pending_requests))

    for _ in range(num_to_prefill):
        req = self.pending_requests.pop(0)
        used_slots = set(self.running_requests.keys())
        for s in range(self.max_batch_size):
            if s not in used_slots:
                free_slot = s
                break
        self.running_requests[free_slot] = req
</CodeMasker>
