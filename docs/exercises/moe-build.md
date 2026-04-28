---
title: MoE 实现挑战
description: Level 4 完整实现：Top-K Router、Expert FFN、Load Balance Loss、容量因子、Sparse MoE Transformer
topics: [build, MoE, mixture-of-experts, top-k-router, load-balance, capacity-factor, expert-parallelism]
---
# MoE 实现挑战 (Level 4)

> **难度:** 困难 | **前置知识:** Transformer 全部内容、SwiGLU、PyTorch 熟练使用、读过 [MoE 代码填空](./moe-fill.md) | **预计时间:** 4-6 小时

## 挑战目标

从零实现一个**可训练**、**稀疏激活**的 mini Sparse-MoE Transformer 模型。不使用任何 MoE 框架（如 `tutel` / `megablocks` / `fairscale.MoE`），所有 Router、Expert、Dispatch、Combine、Load Balance Loss 全部手写。

完成后，你的模型应该能在莎士比亚（或唐诗）字符级数据集上训练 1-2 小时，loss 正常下降，且各 Expert 的负载均衡指标（max/min token 数比值）保持在合理范围。

::: tip 这份挑战与 [GPT 实现挑战](./gpt-build.md) / [Llama 实现挑战](./llama-build.md) 的关系
GPT-build 教你"Decoder Block"，Llama-build 教你"现代化 Decoder Block"，**MoE-build 教你把 FFN 拆成 N 个 Expert + Router**。底层 Attention 你可以直接复用 GPT 或 Llama 的实现，本练习只重构 FFN 部分。
:::

---

## 热身练习

在挑战完整模型之前，先完成以下四个小练习，确保你掌握了核心组件。

### 热身 1：Top-K Router（softmax + topk + 归一化）

Router 是 MoE 的"调度中心"。给定一个 token 特征向量 $x \in \mathbb{R}^d$，路由网络输出一个 logits $g(x) \in \mathbb{R}^N$，其中 $N$ 是 expert 数。我们对 logits 做 softmax，再选 Top-K 个，最后**重新归一化**让选中的权重和为 1。

数学上：

$$
P_i(x) = \frac{\exp(g_i(x))}{\sum_j \exp(g_j(x))}, \quad
\mathcal{T} = \text{TopK}(P, K), \quad
w_i = \frac{P_i}{\sum_{j \in \mathcal{T}} P_j} \text{ for } i \in \mathcal{T}
$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, top_k=2):
        """
        参数:
            dim: token 特征维度
            num_experts: expert 总数 N
            top_k: 每个 token 选 K 个 expert（默认 K=2，对齐 Mixtral）
        """
        super().__init__()
        # TODO: 定义 gate 线性层 (dim -> num_experts)，无 bias
        # 提示: nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):
        """
        参数:
            x: (N, dim) 已展平的 token，N = batch * seq_len
        返回:
            topk_weight: (N, top_k) 归一化后的 top-k 权重，每行和为 1
            topk_idx:    (N, top_k) 选中的 expert id
            gate_logits: (N, num_experts) 原始 logits（auxiliary loss 要用）
        """
        # TODO: 1. 计算 gate logits
        # TODO: 2. softmax 得到概率
        # TODO: 3. torch.topk 选 K 个
        # TODO: 4. 在 top-k 维度归一化（除以 sum，使每行和 = 1）
        pass


# ======== 单元测试 ========
torch.manual_seed(0)
router = TopKRouter(dim=64, num_experts=8, top_k=2)
x = torch.randn(12, 64)  # 12 个 token

w, idx, logits = router(x)

assert w.shape == (12, 2), f"topk_weight shape 错误: {w.shape}"
assert idx.shape == (12, 2), f"topk_idx shape 错误: {idx.shape}"
assert logits.shape == (12, 8), f"gate_logits shape 错误: {logits.shape}"

# 权重归一化校验：每行和为 1
assert torch.allclose(w.sum(dim=-1), torch.ones(12), atol=1e-5), \
    "Top-K 权重未正确归一化（每行和应为 1）"

# expert id 必须在 [0, num_experts) 范围内
assert (idx >= 0).all() and (idx < 8).all(), "expert id 越界"

# 同一 token 不应选两次同一 expert
for i in range(12):
    assert idx[i, 0] != idx[i, 1], f"token {i} 选了同一个 expert 两次"

print("热身 1（Top-K Router）测试通过")
```

::: details 提示（卡住了再看）
- `gate_logits = self.gate(x)` -> `(N, num_experts)`
- `probs = F.softmax(gate_logits, dim=-1)`
- `topk_weight, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)`
- 归一化：`topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)`
:::

---

### 热身 2：Load Balance Loss（auxiliary loss）

稀疏 MoE 训练最容易踩的坑是**专家坍塌**：router 越训越偏，最后所有 token 都涌向同 1-2 个 expert，其他 expert 因为永远得不到梯度而废弃。Switch Transformer / GShard 提出 auxiliary loss 来惩罚这种不均衡：

$$
L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

- $N$：expert 总数
- $f_i$：分发到 expert $i$ 的 token 比例（**离散指示**，对 router 不可导）
- $P_i$：router 给 expert $i$ 的平均 softmax 概率（**连续**，可导）
- $\alpha$：辅助损失系数，常用 `0.01`（Switch）或 `0.001`（GShard）

理想均匀情况下 $f_i = K/N$，$P_i = 1/N$，则 $L_{\text{aux}} = \alpha \cdot K$（与 expert 数无关）。

```python
def load_balance_loss(gate_logits, topk_idx, num_experts):
    """
    Switch Transformer 风格的 auxiliary load balance loss

    参数:
        gate_logits: (N, num_experts) router 原始 logits
        topk_idx:    (N, K)           每个 token 选中的 expert id
        num_experts: int              expert 总数

    返回:
        loss: 标量，对 gate 参数可导
    """
    # TODO: 1. 计算 P_i = mean over tokens of softmax(logits)
    # 提示: F.softmax(gate_logits, dim=-1).mean(dim=0)，shape (num_experts,)

    # TODO: 2. 用 one-hot 累加构造 expert_mask (N, num_experts)，
    #         每个 token 在它选中的 K 个位置上置 1
    # 提示: 可以用 F.one_hot(topk_idx, num_experts).sum(dim=1)

    # TODO: 3. f_i = expert_mask.mean(dim=0)，shape (num_experts,)

    # TODO: 4. loss = num_experts * (f * P).sum()
    pass


# ======== 单元测试 ========
torch.manual_seed(42)
N, num_experts, K = 256, 8, 2

# Case 1: 完全均匀（理想）
# 构造均匀 logits 和均匀分配的 idx
gate_logits = torch.zeros(N, num_experts, requires_grad=True)
topk_idx = torch.stack([
    torch.arange(N) % num_experts,
    (torch.arange(N) + 1) % num_experts,
], dim=-1)
loss_balanced = load_balance_loss(gate_logits, topk_idx, num_experts)
print(f"均匀分配 loss: {loss_balanced.item():.4f}")
# 理论值: N * K/N * 1/N * num_experts = K = 2.0
assert abs(loss_balanced.item() - 2.0) < 1e-4, \
    f"均匀情况 loss 应为 K=2.0，得到 {loss_balanced.item()}"

# Case 2: 全部 token 都涌到 expert 0（最坏情况）
bad_idx = torch.zeros(N, K, dtype=torch.long)
bad_idx[:, 1] = 1  # 退而求其次选 expert 1
loss_bad = load_balance_loss(gate_logits, bad_idx, num_experts)
print(f"坍塌情况 loss: {loss_bad.item():.4f}")
assert loss_bad.item() > loss_balanced.item(), \
    "坍塌情况 loss 应该大于均匀情况"

# 可导性检查
loss_balanced.backward()
assert gate_logits.grad is not None, "loss 必须对 gate_logits 可导"

print("热身 2（Load Balance Loss）测试通过")
```

::: warning 为什么 loss 是 $f \cdot P$ 而不是单纯惩罚 $\text{Var}(f)$？
$f$ 是离散计数，对 router 参数不可导。Switch Transformer 的巧思在于：把不可导的 $f$（指示载荷不均衡的程度）和可导的 $P$（router 的连续输出）相乘，这样梯度只通过 $P$ 回传，而 $f$ 起到"动态权重"的作用——哪个 expert 现在被选得多，就放大对该 expert 概率的惩罚。
:::

---

### 热身 3：Expert FFN（一个 SwiGLU MLP）

每个 expert 就是一个独立的 FFN。现代 MoE（Mixtral / DeepSeek-MoE）的 expert 内部用 SwiGLU。如果你已经在 [Llama 实现挑战](./llama-build.md) 中写过 SwiGLU，这里直接复用。

```python
class Expert(nn.Module):
    """单个 Expert = 一个 SwiGLU FFN"""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 8 // 3  # 与标准 FFN(4d) 参数量近似对齐
        # TODO: 定义 w1, w_gate, w2 三个 Linear（无 bias）
        # w1, w_gate: dim -> hidden_dim
        # w2: hidden_dim -> dim

    def forward(self, x):
        # TODO: SwiGLU = F.silu(w_gate(x)) * w1(x) -> w2
        pass


# ======== 单元测试 ========
expert = Expert(dim=64)
x = torch.randn(16, 64)
y = expert(x)
assert y.shape == (16, 64), f"Expert 输出 shape 错误: {y.shape}"

# 参数量校验：SwiGLU(d=64, h=64*8//3=170) 应有 3*64*170 = 32,640 参数
n_params = sum(p.numel() for p in expert.parameters())
print(f"单个 expert 参数量: {n_params}")

print("热身 3（Expert FFN）测试通过")
```

::: tip 为什么 expert 一定要做成 FFN，而不是 attention？
Attention 层需要在 token 之间交互，token 必须共用同一组 KV 才能计算自注意力；而 FFN 是 **token-wise** 操作，每个 token 独立做变换，天然适合按 token 路由到不同的 expert。这也解释了为什么"MoE 把 FFN 替换成 N 个 expert"，而 attention 部分保持稠密。
:::

---

### 热身 4（可选）：容量因子 + Token Dropping

::: warning 这个热身只在你打算实现"严格容量限制"的工业级 MoE 时才需要
入门版 MoE 可以让 expert 接受任意多 token（动态形状），但训练时为了**张量化（tensorization）**和**通信均衡**，工业实现会强制每个 expert 最多处理 `capacity` 个 token，超出的部分直接丢弃（走 residual）。
:::

容量公式：

$$
\text{capacity} = \lceil \text{capacity\_factor} \cdot \frac{N \cdot K}{E} \rceil
$$

其中 $N$ 是总 token 数，$K$ 是 top-k，$E$ 是 expert 数，`capacity_factor` 通常取 1.0 ~ 1.25。

```python
def assign_with_capacity(topk_idx, num_experts, capacity):
    """
    给每个 token 在它选中的 expert 中分配一个槽位（slot）。
    超出 capacity 的 token 被 drop（slot = -1）。

    参数:
        topk_idx: (N, K) 每个 token 选中的 expert id
        num_experts: int
        capacity: int  每个 expert 最多接受多少 token

    返回:
        slot: (N, K) 每个 token 在对应 expert 内的位置（0..capacity-1），
              -1 表示被 drop
    """
    N, K = topk_idx.shape
    device = topk_idx.device
    slot = torch.full((N, K), -1, dtype=torch.long, device=device)

    # TODO: 1. 把 topk_idx 展平成 (N*K,) 的列表，按"先到先得"顺序累计
    # TODO: 2. 对每个 (token_i, k) pair，计算它在所选 expert 中的累计编号
    # 提示: 可以用 cumsum + scatter 实现，或者按 expert 分组排序
    # TODO: 3. 累计编号 < capacity 的保留，否则置 -1

    # 简化版（O(N*K) Python 循环，可读性优先）：
    counters = [0] * num_experts
    for n in range(N):
        for k in range(K):
            e = topk_idx[n, k].item()
            if counters[e] < capacity:
                slot[n, k] = counters[e]
                counters[e] += 1
    return slot


# ======== 单元测试 ========
N, num_experts, K, capacity = 10, 4, 2, 3

# 构造一个所有 token 都选 expert 0 的极端情况
topk_idx = torch.zeros(N, K, dtype=torch.long)
topk_idx[:, 1] = 1
slot = assign_with_capacity(topk_idx, num_experts, capacity)

# 前 3 个 token 在 expert 0 拿到 slot 0/1/2，后续 7 个被 drop
print("Slots:\n", slot)
assert (slot[:3, 0] >= 0).all(), "前 3 个 token 在 expert 0 应该有 slot"
assert (slot[3:, 0] == -1).all(), "第 4 个 token 起在 expert 0 应该被 drop"

# Drop ratio
dropped = (slot == -1).float().mean().item()
print(f"Drop ratio: {dropped:.2%}")

print("热身 4（容量因子）测试通过")
```

---

## 完整挑战：实现 MoE Transformer

### 模型配置

```python
from dataclasses import dataclass

@dataclass
class MoEConfig:
    # 词表与序列
    vocab_size: int = 5000
    max_seq_len: int = 256

    # Transformer 主干
    dim: int = 384
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.1

    # MoE 专属
    num_experts: int = 8        # expert 总数 N
    top_k: int = 2              # 每个 token 选 K 个 expert
    expert_hidden_dim: int = None  # None -> dim * 8 // 3（SwiGLU 默认）
    aux_loss_alpha: float = 0.01   # auxiliary loss 系数
    capacity_factor: float = 1.25  # 容量因子（None -> 不限制容量）
    use_capacity: bool = False     # 入门版可关掉，留给进阶
```

### 子任务 1：MoE Layer（router + N experts + dispatch + combine）

这是整个挑战的**核心**。你需要把 [moe-fill.md 练习 2](./moe-fill.md) 中的"dispatch-compute-combine"思路完整实现，并接入 Top-K Router 和 Load Balance Loss。

```python
class MoELayer(nn.Module):
    """
    Sparse MoE FFN：替换 Transformer Block 中的标准 FFN

    输入: x (B, T, dim)
    输出: (y, aux_loss)
        y: (B, T, dim)
        aux_loss: 标量，需要加到主 loss 上
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # TODO: 1. 实例化 TopKRouter
        # TODO: 2. 实例化 N 个 Expert（用 nn.ModuleList）
        # 提示: hidden_dim = config.expert_hidden_dim or config.dim * 8 // 3

    def forward(self, x):
        """
        参数: x (B, T, dim)
        返回: y (B, T, dim), aux_loss (scalar)
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # ====== Step 1: Routing ======
        # TODO: 调用 router 得到 topk_weight, topk_idx, gate_logits
        # topk_weight: (N, K), topk_idx: (N, K), gate_logits: (N, num_experts)

        # ====== Step 2: Dispatch + Compute ======
        # 思路: 对每个 expert i:
        #   找出所有选中它的 (token, k) pair
        #   把这些 token 的特征 gather 出来，过 expert_i
        # 注意: torch.where(topk_idx == i) 返回 (token_indices, k_positions)
        y_flat = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            # TODO: 找出选中 expert i 的 (token_ids, k_pos)
            # TODO: 跳过没人选的 expert（continue）
            # TODO: 取出 token 特征 -> 过 expert -> 乘以对应 top-k 权重
            # TODO: scatter_add 回 y_flat
            pass

        # ====== Step 3: Auxiliary Load Balance Loss ======
        # TODO: 调用 load_balance_loss
        # 注意: 乘以 aux_loss_alpha

        y = y_flat.reshape(B, T, D)
        return y, aux_loss


# ======== 单元测试 ========
config = MoEConfig(dim=64, num_experts=4, top_k=2)
moe = MoELayer(config)
x = torch.randn(2, 8, 64)

y, aux = moe(x)
assert y.shape == (2, 8, 64), f"MoE 输出 shape 错误: {y.shape}"
assert aux.dim() == 0, "aux_loss 应该是标量"
assert aux.item() > 0, "aux_loss 在初始化时应 > 0"

# 反向传播：aux_loss 必须对 router 可导
total = y.mean() + aux
total.backward()
assert moe.router.gate.weight.grad is not None, "Router 应能收到梯度"

print("子任务 1（MoE Layer）测试通过")
```

::: warning 为什么用 `scatter_add` 或循环累加，而不是直接索引赋值？
当 `top_k > 1` 时，**同一个 token 会被多个 expert 分别处理**，最后要把这些 expert 的输出按权重相加。如果你写 `y_flat[token_ids] = expert_out`，第二个 expert 会**覆盖**第一个的结果，导致 top-2 退化成 top-1。正确做法是 `y_flat[token_ids] += expert_out * weight`。
:::

::: details Hint：dispatch 的高效实现
朴素实现是 `for i in range(num_experts)` Python 循环，可读性好但 GPU 利用率低。工业实现（如 megablocks）会把 token 按 expert 排序成连续段，用一次 batched matmul 算完所有 expert。本练习只要求循环版本能跑通，进阶版本留给推荐资源。
:::

---

### 子任务 2：MoE Transformer Block

把 MoELayer 接到一个标准的 Pre-Norm Transformer Block 中，**替换原来的 FFN**。

```python
class CausalSelfAttention(nn.Module):
    """因果自注意力（直接复用 GPT-build 的实现，此处略）"""
    def __init__(self, config: MoEConfig):
        super().__init__()
        # ... 见 gpt-build.md
        pass

    def forward(self, x):
        # ... 见 gpt-build.md
        pass


class MoETransformerBlock(nn.Module):
    """
    Pre-Norm + MoE FFN

    forward 流程:
        x = x + Attention(LayerNorm(x))
        h, aux = MoELayer(LayerNorm(x))
        x = x + h
        return x, aux
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        # TODO: 定义 ln1, attn, ln2, moe（用 MoELayer）

    def forward(self, x):
        """
        参数: x (B, T, D)
        返回: x (B, T, D), aux_loss (标量)
        """
        # TODO: 1. attention 残差
        # TODO: 2. MoE 残差（同时收集 aux_loss）
        pass


# ======== 单元测试 ========
config = MoEConfig(dim=64, n_heads=4, num_experts=4, top_k=2)
block = MoETransformerBlock(config)
x = torch.randn(2, 16, 64)
y, aux = block(x)
assert y.shape == x.shape
assert aux.requires_grad

print("子任务 2（MoE Transformer Block）测试通过")
```

---

### 子任务 3：完整 MoE 模型 + 训练 loop

```python
class MoEModel(nn.Module):
    """
    完整 Sparse MoE Transformer (Decoder-Only)

    forward(idx, targets=None):
        返回 logits (B, T, vocab) 和 total_loss (CE loss + aux losses 之和)
    """

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        # TODO: tok_emb (nn.Embedding)
        # TODO: pos_emb (nn.Embedding) 或 RoPE
        # TODO: dropout
        # TODO: blocks = nn.ModuleList([MoETransformerBlock(config) for _ in range(n_layers)])
        # TODO: ln_f (final LayerNorm)
        # TODO: lm_head (Linear, 可与 tok_emb weight tying)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # TODO: token + position embedding
        # TODO: 依次过 N 个 MoETransformerBlock，**累加每层的 aux_loss**
        # 提示:
        #   total_aux = 0
        #   for blk in self.blocks:
        #       x, aux = blk(x)
        #       total_aux = total_aux + aux

        # TODO: ln_f -> lm_head -> logits

        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            # 关键: 总 loss = 主 loss + 平均的 aux loss（每层一份，求平均更稳定）
            loss = ce_loss + total_aux / self.config.n_layers

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """与 GPT 的 generate 完全一致；MoE 推理时仍然走 router，只是不计算 aux_loss"""
        # TODO: 复用 gpt-build.md 的 generate
        pass
```

### 训练脚本骨架

```python
def train():
    # 1. 数据准备（莎士比亚字符级）
    #    text = open('shakespeare.txt').read()
    #    chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}
    #    data = torch.tensor([stoi[c] for c in text])
    #    train, val = data[:int(0.9*len(data))], data[int(0.9*len(data)):]

    # 2. 模型与优化器
    #    config = MoEConfig(vocab_size=len(chars), num_experts=8, top_k=2)
    #    model = MoEModel(config)
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # 3. 训练循环
    #    for step in range(max_steps):
    #        x, y = get_batch(train, config.max_seq_len, batch_size)
    #        logits, loss = model(x, y)
    #        loss.backward()
    #        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #        optimizer.step(); optimizer.zero_grad()
    #        if step % 100 == 0:
    #            print(f"step {step}: loss {loss.item():.4f}")
    #            # 监控 expert 负载（详见"测试与验证"章节）

    # 4. 生成
    #    prompt = torch.tensor([[stoi['T']]])
    #    out = model.generate(prompt, max_new_tokens=200)
    #    print(''.join(itos[i] for i in out[0].tolist()))
    pass
```

---

## 测试与验证

### 测试 1：Shape 测试（必过）

```python
config = MoEConfig(
    vocab_size=100, max_seq_len=32, dim=64,
    n_heads=4, n_layers=2, num_experts=4, top_k=2,
)
model = MoEModel(config)
x = torch.randint(100, (2, 32))
targets = torch.randint(100, (2, 32))

logits, loss = model(x, targets)
assert logits.shape == (2, 32, 100), f"logits shape 错误: {logits.shape}"
assert loss.dim() == 0, "loss 应是标量"

# 参数量统计
n_total = sum(p.numel() for p in model.parameters())
n_expert = sum(p.numel() for p in model.blocks[0].moe.experts.parameters())
print(f"总参数量: {n_total/1e6:.2f}M")
print(f"单层 expert 参数量: {n_expert/1e6:.2f}M")
print(f"激活参数比: ~{config.top_k}/{config.num_experts}")
```

### 测试 2：Load Balance 数值测试（关键）

刚初始化时 router 是随机的，每个 expert 被选中的频率应该**接近均匀**。如果某个 expert 拿到 0% 或 100% 的 token，说明 router 实现有问题。

```python
@torch.no_grad()
def check_expert_load(model, x, layer_idx=0):
    """统计第 layer_idx 层各 expert 接收的 token 数"""
    model.eval()
    # 取出该层的 router
    moe_layer = model.blocks[layer_idx].moe

    # 手动跑一遍前向到该层
    B, T = x.shape
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T))
    for i in range(layer_idx):
        h, _ = model.blocks[i](h)
    h_norm = model.blocks[layer_idx].ln2(
        h + model.blocks[layer_idx].attn(model.blocks[layer_idx].ln1(h))
    )

    h_flat = h_norm.reshape(B*T, -1)
    _, idx, _ = moe_layer.router(h_flat)

    # 统计每个 expert 拿到几个 token（注意：top_k 个槽位都要计）
    counts = torch.zeros(moe_layer.num_experts)
    for k in range(moe_layer.top_k):
        for e in idx[:, k]:
            counts[e] += 1
    return counts


config = MoEConfig(num_experts=8, top_k=2, vocab_size=100, n_layers=2,
                   max_seq_len=64, dim=64, n_heads=4)
model = MoEModel(config)
x = torch.randint(100, (4, 64))  # 256 tokens, 每个选 2 expert -> 512 槽位
counts = check_expert_load(model, x, layer_idx=0)

print(f"各 expert 接收 token 数: {counts.tolist()}")
print(f"max/min ratio: {counts.max().item() / max(counts.min().item(), 1):.2f}")

# 初始化时应大致均匀，比值通常 < 3.0
assert counts.max().item() / max(counts.min().item(), 1) < 5.0, \
    "初始化时 expert 负载严重不均，检查 router 初始化"
```

### 测试 3：训练 loss 应正常下降

```python
config = MoEConfig(vocab_size=65, max_seq_len=64, dim=128, n_heads=4,
                   n_layers=4, num_experts=4, top_k=2)
model = MoEModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

losses = []
for step in range(200):
    x = torch.randint(65, (8, 64))
    y = torch.randint(65, (8, 64))
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 200 步后 loss 应明显下降
print(f"step 0: {losses[0]:.4f}, step 199: {losses[-1]:.4f}")
assert losses[-1] < losses[0] * 0.95, "loss 没有下降，检查实现"
print("测试 3（训练收敛）通过")
```

### 测试 4：Aux loss 在均衡时应趋近理论值

如果路由完全均匀，$L_{\text{aux}} = \alpha \cdot K$。训练几百步后实际值应该接近这个下限（而不是远高于）。

```python
# 在训练循环中额外打印 aux_loss
# 完美均衡: 0.01 * 2 = 0.02
# 实际训练时通常在 0.02 ~ 0.05 之间
```

---

## 进阶：DeepSeek MoE 风格的 fine-grained + shared expert

::: tip 此节为提示性内容，**不强求实现**。完成主挑战后可作为延伸课题。
:::

DeepSeek-MoE / DeepSeek-V2 在标准 MoE 上做了两个改造：

1. **Fine-Grained Expert Segmentation**：把每个 expert 的 hidden_dim 切小，但 expert 数量翻倍（如从 8 个 4d 改成 32 个 1d）。每个 token 选更多更细的 expert（如 top_k=6），可以表达更多专家组合，且总参数量与计算量不变。

2. **Shared Expert Isolation**：保留 1-2 个 **shared expert**（每个 token 都过），专门负责通用能力；剩下的 routed expert 负责专精能力。这避免 routed expert 既要学通识又要学专精，提高了样本效率。

```python
class DeepSeekMoELayer(nn.Module):
    """伪代码：fine-grained + shared expert
    forward(x):
        shared_out = sum(shared_expert_i(x) for i in shared_experts)  # 不路由
        routed_out, aux = standard_moe_forward(x)                     # 走 router
        return shared_out + routed_out, aux
    """
    pass
```

参考：[DeepSeek-MoE 论文](https://arxiv.org/abs/2401.06066) / [DeepSeek-V2 报告](https://arxiv.org/abs/2405.04434)。

---

## 评分标准

| 检查项 | 要求 | 必/选 |
|--------|------|------|
| TopKRouter | softmax + topk + 归一化，权重每行和为 1 | 必 |
| Load Balance Loss | 正确实现 $L_{\text{aux}} = \alpha N \sum f_i P_i$，对 router 可导 | 必 |
| MoELayer | dispatch-compute-combine 三阶段，top_k>1 时不互相覆盖 | 必 |
| MoETransformerBlock | Pre-Norm 残差结构正确，aux_loss 正确传出 | 必 |
| MoEModel | 多层 aux_loss 累加，最终 loss = CE + aux | 必 |
| Shape 测试 | 全部通过 | 必 |
| Load 均衡测试 | 训练后各 expert max/min 比值 < 5 | 必 |
| 训练收敛 | loss 在 1k step 内明显下降 | 必 |
| 容量因子 + token drop | 实现热身 4，并在 MoELayer 中开启 | 选 |
| Shared Expert | 实现 DeepSeek 风格 shared expert | 选 |

---

## 常见陷阱

::: warning 调试 MoE 时这五个坑最常见
1. **Top-K 权重覆盖**：用 `y[idx] = expert_out` 而非 `y[idx] += expert_out * w`，导致 top-2 退化为 top-1。
2. **Aux loss 不可导**：忘记 $P_i$ 必须从 `gate_logits` 经 softmax 算出（保留计算图），如果改用 `topk_weight.mean()` 会丢失归一化前的梯度。
3. **expert mask 维度错**：`F.one_hot(topk_idx, num_experts)` 的 shape 是 `(N, K, num_experts)`，要对 K 维 sum 才能得到 `(N, num_experts)`。
4. **aux_loss 没乘 alpha**：默认 0.01。如果忘乘，aux loss 会主导主 loss，模型只学路由不学语言建模。
5. **Router 初始化太大**：`nn.Linear` 默认初始化对小 dim 偏大，会让初始路由极度不均（softmax 接近 one-hot）。建议把 gate 权重的 std 缩小到 0.02 或更小。
:::

---

## 参考时间分配

| 阶段 | 内容 | 建议时间 |
|------|------|---------|
| 0 | 完成四个热身练习 | 60 分钟 |
| 1 | 实现 `MoELayer`（dispatch-compute-combine） | 60-90 分钟 |
| 2 | 实现 `MoETransformerBlock` + `MoEModel` | 45 分钟 |
| 3 | 训练脚本 + 调试到 loss 下降 | 60-90 分钟 |
| 4 | Load balance 监控与可视化 | 30 分钟 |
| 5 | 进阶（容量因子 / shared expert） | 60+ 分钟 |

---

## 推荐资源

完成挑战后建议对照下列开源实现，理解工业级 MoE 的优化点：

- [karpathy/minMoE](https://github.com/karpathy/llm.c/tree/master/dev/cuda)（Karpathy 的极简风格，最易读）
- [mistralai/mistral-inference](https://github.com/mistralai/mistral-inference)（Mixtral-8x7B 的官方推理代码，关注 `moe.py` 中 router + expert 的组织方式）
- [XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE)（开源 MoE 训练参考，包含负载均衡、router z-loss 等技巧）
- [databricks/megablocks](https://github.com/databricks/megablocks)（dropless MoE + Block-Sparse MatMul，工业级吞吐）
- [deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)（fine-grained + shared expert 的开源实现）
- [google/flaxformer](https://github.com/google/flaxformer)（Switch Transformer / GShard 在 JAX 下的原始实现）

::: tip 与本站其他练习的关系
- 想先打 router / dispatch 基本功？回看 [MoE 代码填空 (L2-L3)](./moe-fill.md)。
- 想先把 Decoder Block 写顺？先做 [GPT 实现挑战](./gpt-build.md) 或 [Llama 实现挑战](./llama-build.md)。
- 想理解 MoE 的位置和体系？读 [DeepSeek 架构剖析](/architecture/deepseek)。
:::

祝你实现顺利！第一次把 aux_loss 调对、看到八个 expert 的 token 数分布稳定在 12% ± 3% 的那一刻，比 GPT loss 下降还有成就感。
