---
title: "注意力机制"
description: "从 Scaled Dot-Product 到 MHA/MQA/GQA/MLA，再到 Flash Attention 深度实现"
topics: [scaled-dot-product, MHA, MQA, GQA, MLA, TPA, flash-attention, online-softmax]
prereqs: [architecture/transformer]
---
# 注意力机制

> **一句话总结:** 注意力机制是 Transformer 的灵魂——从 Scaled Dot-Product Attention 的数学本质，到 MHA/MQA/GQA/MLA 的架构演进，再到 Flash Attention 的工程极致优化，每一步都在平衡**建模能力**与**计算效率**。

## 在大模型体系中的位置

```
Input Token → Embedding + Positional Encoding
                ↓
        ┌────────────────────────┐
        │  Attention (this page) │  ← Core: lets each token "see" other tokens
        └────────────────────────┘
                ↓
           FFN / MoE                ← Per-token nonlinear transform
                ↓
           LayerNorm                ← Stabilize training
                ↓
          x N layers
                ↓
           Output logits
```

注意力层决定了"**信息如何在序列内流动**"。模型的上下文理解能力、长距离依赖建模、推理速度和显存消耗，都与注意力机制的设计直接相关。

---

## Scaled Dot-Product Attention

### 核心公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{n \times d_k}$，$V \in \mathbb{R}^{n \times d_v}$，分别由输入 $X$ 经线性投影得到：

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

### 为什么要除以 $\sqrt{d_k}$？——方差证明

> 以下方差推导来自 Vaswani et al. (2017) "Attention Is All You Need" 原始论文的脚注 4。

假设 $q_i, k_i$ 均为独立随机变量，服从标准正态分布 $\mathcal{N}(0, 1)$。对于点积：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

**逐元素分析：**

$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

$$
\text{Var}(q_i k_i) = (\text{Var}(q_i) + \mathbb{E}[q_i]^2)(\text{Var}(k_i) + \mathbb{E}[k_i]^2) - \mathbb{E}[q_i]^2\mathbb{E}[k_i]^2 = 1
$$

**对整个向量求和：**

$$
\mathbb{E}\!\left[\sum_{i=1}^{d_k} q_i k_i\right] = 0, \quad \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k
$$

点积的方差随维度线性增长！当 $d_k = 1024$ 时，点积分布在 $[-100, 100]$ 量级，softmax 输出会极度尖锐（趋近 one-hot），**梯度几乎消失**。

除以 $\sqrt{d_k}$ 后，利用 $\text{Var}(cX) = c^2\text{Var}(X)$：

$$
\text{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \cdot d_k = 1
$$

注意力分数回到标准正态分布，softmax 输出分布温和，梯度稳定。

> **为什么不除以 $d_k$？** 除以 $d_k$ 会导致方差为 $1/d_k$，分布过于集中，softmax 趋近均匀分布，注意力失去区分能力。

### 代码实现

下面的实现对照 Vaswani et al. (2017) §3.2.1 公式与 PyTorch 官方 [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 的命名约定：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """单头 Scaled Dot-Product Attention（教学版）。

    Args:
        d_model:  输入与输出的隐藏维度
        d_k:      Q/K 的投影维度（缩放分母用 sqrt(d_k)；默认与 d_model 相同）
        d_v:      V 的投影维度（默认与 d_k 相同）
    """

    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None):
        super().__init__()
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else self.d_k
        self.w_q = nn.Linear(d_model, self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.d_v, bias=False)
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, d_model]
        q = self.w_q(x)                                      # [B, T, d_k]
        k = self.w_k(x)                                      # [B, T, d_k]
        v = self.w_v(x)                                      # [B, T, d_v]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # 注意：是 sqrt(d_k)，不是 sqrt(d_model)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)                  # [B, T, T]
        context = weights @ v                                # [B, T, d_v]
        return self.w_o(context)                             # [B, T, d_model]
```

::: tip 几个容易踩的坑
- **缩放分母是 `sqrt(d_k)` 而不是 `sqrt(d_model)`**：在多头实现中 `d_k = d_model / n_heads`，把整个 `d_model` 代进去会让方差归一化失效。
- **mask 用 `-inf` 而不是相加一个负数**：`masked_fill(mask == 0, -inf)` 经过 softmax 后被屏蔽位置严格为 0，加常数会带来数值漂移。
- **`w_o` 的输入维度是 `d_v`，输出是 `d_model`**：当 `d_v ≠ d_model` 时这是恢复隐藏维度的唯一通道，写成 `Linear(d_model, d_model)` 会在 `d_v ≠ d_model` 时挂掉。

工业级实现可直接用 PyTorch ≥ 2.0 的 `F.scaled_dot_product_attention`（自动选择 FlashAttention / Memory-Efficient / Math 后端），或参考 HuggingFace [`transformers/models/llama/modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) 的 `LlamaAttention`。
:::

---

## Multi-Head Attention (MHA)

### 核心思想

单头注意力只能在一个子空间中捕捉关系。多头注意力将 Q、K、V **拆分到 $h$ 个头**，每个头独立计算注意力，最后拼接：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

其中每个头的维度 $d_k = d_{\text{model}} / h$，总参数量不变。

### 完整过程：拆分 → 并行计算 → 拼接

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attn_mask=None):
        q = self._split_heads(self.q_proj(hidden_states))
        k = self._split_heads(self.k_proj(hidden_states))
        v = self._split_heads(self.v_proj(hidden_states))

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores, dim=-1)
        ctx = (weights @ v).transpose(1, 2).contiguous()
        b, n = ctx.shape[0], ctx.shape[1]
        return self.o_proj(ctx.view(b, n, -1))
```

---

## Multi-Query Attention (MQA) 与 Grouped-Query Attention (GQA)

### 演进动机

推理阶段需要缓存历史 K、V（KV Cache），其大小为 `[2, bsz, seq_len, n_heads, head_dim]`。当模型有 64/128 个头时，KV Cache 占用巨大，限制了 batch size 和序列长度。

**核心问题：** 多头的 K、V 是否存在冗余？能否在减少头数的同时保持精度？

### MQA：所有 Q 头共享 1 组 KV

```python
class MultiQueryAttention(nn.Module):
    """所有 Query 头共享同一组 K 和 V（n_kv_heads = 1）"""
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim)  # 单头 K
        self.v_proj = nn.Linear(hidden_size, self.head_dim)  # 单头 V
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, attn_mask=None):
        b, n, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).unsqueeze(1)  # (b, 1, n, head_dim) 广播
        v = self.v_proj(hidden_states).unsqueeze(1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        ctx = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous()
        return self.o_proj(ctx.view(b, n, -1))
```

### GQA：分组共享 KV（Llama 2/3 采用）

GQA 是 MHA 与 MQA 的折中——将 $h$ 个 Q 头分成 $g$ 组，每组共享一套 KV：

下面是 HuggingFace `transformers` 在 `modeling_llama.py` 中使用的 `repeat_kv` 帮助函数（精简自[官方源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)，Apache-2.0）——核心是 `expand` + `reshape` 把分组 KV 复制到与 Q 相同的头数：

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(b, num_kv_heads, n, d) -> (b, num_kv_heads * n_rep, n, d)"""
    b, num_kv_heads, n, d = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, num_kv_heads, n_rep, n, d)
    return hidden_states.reshape(b, num_kv_heads * n_rep, n, d)


class GroupedQueryAttention(nn.Module):
    """对照 HF LlamaAttention：q/k/v_proj 输出维度差异化、每组 KV 复制 num_kv_groups 次"""
    def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=2):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attn_mask=None):
        b, n, _ = hidden_states.shape
        shape = (b, n, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        ctx = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous()
        return self.o_proj(ctx.view(b, n, -1))
```

> **GQA 的本质：** 当 `num_kv_heads == num_heads` 时退化为 MHA；当 `num_kv_heads == 1` 时退化为 MQA。Llama 2 70B 使用 `num_kv_heads = 8`，在质量和效率间取得了极佳平衡。
>
> **官方完整实现：** [transformers/.../modeling_llama.py `LlamaAttention`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)（含 RoPE、KV Cache、多种 backend 调度）。

---

## Multi-Latent Attention (MLA)

### DeepSeek 的创新思路

MQA/GQA 通过减少 KV 头数来压缩 KV Cache，但这本质上是一种"特征丢弃"。DeepSeek-V2 提出 MLA，换了一个思路：**用低秩压缩代替头数削减**。

**核心思想：** 先将输入压缩到一个低维 latent 向量 $c$，再通过 up-projection 恢复完整的多头 KV。KV Cache 只需存储低维的 $c$。

$$
c^{KV} = W^{\text{down}}_{KV} \cdot X, \quad W^{\text{down}}_{KV} \in \mathbb{R}^{d \times d_c}
$$

$$
K = W^{\text{up}}_K \cdot c^{KV}, \quad V = W^{\text{up}}_V \cdot c^{KV}
$$

传统 MHA 的 KV Cache 大小为 $2 \times n_h \times d_h \times l$，MLA 只需存储 $d_c \times l$（$d_c \ll d$），压缩比可达 **16 倍以上**。

### 代码实现（对照 DeepSeek-V3 官方）

下面的最小实现遵循 [deepseek-ai/DeepSeek-V3 `inference/model.py` 的 `MLA`](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py)（MIT），保留官方的命名约定 `wq_a/wq_b/wkv_a/wkv_b`、`q_lora_rank/kv_lora_rank`，并省略了张量并行、量化、KV cache 注册等工程细节，只展示数学骨架：

```python
class MLA(nn.Module):
    """Multi-Head Latent Attention（DeepSeek-V2/V3）— teaching-skeleton 版"""
    def __init__(self, dim, n_heads,
                 q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # Q 路：先压到 q_lora_rank，再升回 n_heads * (nope + rope) 维
        self.wq_a = nn.Linear(dim, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.wq_b = nn.Linear(q_lora_rank, n_heads * self.qk_head_dim, bias=False)

        # KV 路：压到 kv_lora_rank（+ qk_rope_head_dim 给 K_rope 用），再升回 K_nope + V
        self.wkv_a = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim), bias=False)

        self.wo = nn.Linear(n_heads * v_head_dim, dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape

        # ---- Q 路 ----
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # ---- KV 路：拆出共享的 K_rope 和压缩潜变量 ----
        kv = self.wkv_a(x)
        c_kv, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)  # 所有头共享 K_rope

        # 升维成完整 K_nope + V
        kv = self.wkv_b(self.kv_norm(c_kv))
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 把共享的 K_rope 广播给每个 head，再和 K_nope 拼起来
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        attn = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        out = torch.einsum("bsht,bthd->bshd", attn, v)
        return self.wo(out.flatten(2))
```

::: tip 解耦 RoPE 是 MLA 的关键
注意 `q_pe / k_pe` 走的是独立的 `qk_rope_head_dim` 维度通道，**只有这一段 K 会被旋转**；`k_nope` 走压缩 + 升维路径。这样矩阵吸收（下一节）就只需要在 `nope` 部分进行。
:::

### 矩阵吸收技巧（推理优化）

训练完成后，由于 `K_nope = wkv_b(kv_norm(c_kv))` 是 latent → head 的线性映射，可以把 `wkv_b` 提前"吸收"进 `q_nope`，KV Cache 只需保存压缩后的 `c_kv` 和共享的 `k_pe`，不再展开成完整的 `k`、`v`：

$$
\text{score}_{nope} = q_{nope} \cdot W^{\text{up}}_{KV} \cdot c_{KV} = (q_{nope} W^{\text{up}}_{KV}) \cdot c_{KV}
$$

DeepSeek-V3 官方代码用 `einsum` 实现该路径（`attn_impl != "naive"` 分支，参见上方源码链接）。压缩比可达 **16 倍以上**。

---

## Flash Attention

### GPU 内存层次：SRAM vs HBM

| 存储层级 | 容量 | 带宽 | 特点 |
|---------|------|------|------|
| **SRAM**（片上缓存） | ~20 MB | ~19 TB/s | 极快，但容量很小 |
| **HBM**（显存） | 40-80 GB | ~1.5 TB/s | 容量大，但带宽是瓶颈 |

### 标准 Attention 的 IO 瓶颈

标准 attention 的计算流程：

1. 从 HBM 读取 Q、K，计算 $S = QK^T$，**写回 HBM**（$O(n^2)$ 中间矩阵！）
2. 从 HBM 读取 $S$，计算 $P = \text{softmax}(S)$，**写回 HBM**
3. 从 HBM 读取 $P$、$V$，计算 $O = PV$，**写回 HBM**

$n^2$ 大小的中间矩阵反复在 HBM 上读写，**IO 成为瓶颈**，而非计算本身。

### Flash Attention 的分块策略 + Online Softmax

**核心思想：** 将 Q、K、V 分成小块，每块放进 SRAM 中完成全部计算，避免将 $n^2$ 中间矩阵写回 HBM。

难点在于：softmax 需要全局 max 和 sum，分块后怎么办？答案是 **Online Softmax**。

#### Online Softmax 原理

对于向量 $X = [x_1, \dots, x_n]$，标准 softmax 需要两遍扫描（求 max + 求 sum）。Online Softmax 可以**单遍扫描**，通过递推更新：

$$
m^{(t)} = \max(m^{(t-1)},\; x_t)
$$

$$
l^{(t)} = l^{(t-1)} \cdot e^{m^{(t-1)} - m^{(t)}} + e^{x_t - m^{(t)}}
$$

**分块版本：** 每块内部独立算 max 和 sum，块间通过上述递推公式合并。

#### Flash Attention v1 实现（每个 KV 切片驱动一次完整的 Q 扫描）

下面这个版本把 v1 算法封装成一个独立函数，shape 用断言固化，数学等价于完整 softmax，但**只在分块层面操作**——任何时刻 SRAM 里只需要装得下当前的 Q、K、V 三个小切片以及对应的输出/统计量切片。

::: tip 几个开源参考实现
- 论文：Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)](https://arxiv.org/abs/2205.14135)
- 官方 CUDA 内核（Tri Dao 维护）：<https://github.com/Dao-AILab/flash-attention>
- Triton 教学版（与本节 Python 模拟同等阅读价值）：<https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html>
- 在 HuggingFace transformers 中切换 SDPA / FA-2 后端：<https://github.com/huggingface/transformers>
:::

```python
import torch

def flash_attention_v1(Q, K, V, q_block_size: int, kv_block_size: int):
    """
    Flash Attention v1 的纯 PyTorch 教学复刻。
    迭代顺序：每拿到一片 K/V，就让所有 Q 切片都和它"对齐一次"，
    边对齐边以 online softmax 更新各 Q 切片的输出与统计量。
    """
    # 形状契约：(batch, heads, seq, dim)
    assert Q.dim() == 4 and K.shape == V.shape, "Q/K/V 必须是 (B,H,N,d) 且 K、V 同形状"
    assert Q.shape[-1] == K.shape[-1], "head_dim 必须一致"

    # 直接用 -inf 作为初始 running max；softmax 分母里再用一个极小常数兜底防 0
    out          = torch.zeros_like(Q)
    denom        = torch.zeros(*Q.shape[:-1], 1, device=Q.device, dtype=Q.dtype)
    running_max  = torch.full(
        (*Q.shape[:-1], 1), -torch.inf, device=Q.device, dtype=Q.dtype,
    )

    # 沿序列维切片；后四个量随 Q 切片同步切，方便就地写回
    q_tiles            = list(torch.split(Q,           q_block_size,  dim=2))
    k_tiles            = list(torch.split(K,           kv_block_size, dim=2))
    v_tiles            = list(torch.split(V,           kv_block_size, dim=2))
    out_tiles          = list(torch.split(out,         q_block_size,  dim=2))
    denom_tiles        = list(torch.split(denom,       q_block_size,  dim=2))
    running_max_tiles  = list(torch.split(running_max, q_block_size,  dim=2))

    # 外层枚举 K/V 切片：每抓一对 (k_tile, v_tile)，就把所有 Q 切片"扫一遍"
    for k_tile, v_tile in zip(k_tiles, v_tiles):
        for i, (q_tile, prev_out, prev_denom, prev_max) in enumerate(
            zip(q_tiles, out_tiles, denom_tiles, running_max_tiles)
        ):
            # 1) 在小块上算原始注意力分数（实际 CUDA kernel 里此结果只活在 SRAM）
            scores_tile = q_tile @ k_tile.transpose(-2, -1)

            # 2) 该切片自身的 softmax 统计量——先减最大值再 exp，避免溢出
            tile_max  = scores_tile.amax(dim=-1, keepdim=True)
            probs_tile = torch.exp(scores_tile - tile_max)
            tile_denom = probs_tile.sum(dim=-1, keepdim=True)

            # 3) Online softmax 合并：把"之前累计的统计量"和"当前切片的统计量"对齐到同一个新 max
            merged_max   = torch.maximum(prev_max, tile_max)
            scale_prev   = torch.exp(prev_max - merged_max)   # 旧累积的折算因子
            scale_tile   = torch.exp(tile_max - merged_max)   # 当前切片的折算因子
            merged_denom = scale_prev * prev_denom + scale_tile * tile_denom + 1e-12

            # 4) 输出在线累加（数学等价，但表达拆成两步以贴近 GPU 寄存器使用）：
            #    新分子 = scale_prev · 旧分子 + scale_tile · (probs_tile @ v_tile)
            #    再除以新分母 → 当前的"近似输出"
            new_numer = scale_prev * (prev_out * prev_denom) + scale_tile * (probs_tile @ v_tile)
            out_tiles[i]         = new_numer / merged_denom
            denom_tiles[i]       = merged_denom
            running_max_tiles[i] = merged_max

    # 沿序列维拼回完整 O
    return torch.cat(out_tiles, dim=2)
```

外层之所以走 KV 而不是 Q，是 v1 的设计选择：每片 K/V 只从 HBM 读一次，但代价是每片 Q 的输出 / 统计量被反复写回——这正是 v2 接下来要倒过来的地方。

#### Flash Attention v2 改进（先 Q 后 KV）

v2 将外层改为遍历 Q 块、内层遍历 KV 块，**减少 O 的读写次数**，并将 scale 操作推迟到最后：

```python
# Flash Attention v2 关键改动：外层换成 Q，内层才是 KV；O 累加时不再每次除分母
for i, (q_tile, k_iter_seed) in enumerate(zip(q_tiles, [None] * len(q_tiles))):
    cur_out   = torch.zeros_like(q_tile @ v_tiles[0])  # 累加未归一化的分子
    cur_denom = torch.zeros(*q_tile.shape[:-1], 1, device=q_tile.device, dtype=q_tile.dtype)
    cur_max   = torch.full_like(cur_denom, -torch.inf)

    for k_tile, v_tile in zip(k_tiles, v_tiles):
        scores_tile  = q_tile @ k_tile.transpose(-2, -1)
        tile_max     = scores_tile.amax(dim=-1, keepdim=True)
        merged_max   = torch.maximum(cur_max, tile_max)
        probs_tile   = torch.exp(scores_tile - merged_max)
        scale_prev   = torch.exp(cur_max - merged_max)            # 旧累积折算到新 max

        cur_denom = scale_prev * cur_denom + probs_tile.sum(dim=-1, keepdim=True)
        cur_out   = scale_prev * cur_out   + probs_tile @ v_tile  # 注意：此处不除分母
        cur_max   = merged_max

    out_tiles[i] = cur_out / (cur_denom + 1e-12)                  # 整个 Q 切片处理完才做一次归一
```

---

## Flash Attention 深度实现

### 核心算法：Online Softmax 与分块计算

Flash Attention 的关键挑战在于：softmax 是一个**全局操作**，需要知道整个序列的 max 和 sum。分块计算时，每个 block 只能看到部分数据，如何保证结果的精确性？

答案是 **Online Softmax** 的分块递推公式。假设我们已经处理了前 $j-1$ 个 KV 块，当前要合并第 $j$ 个块：

$$
m^{(j)} = \max\!\left(m^{(j-1)},\; \max(\mathbf{S}_{ij})\right)
$$

$$
l^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot l^{(j-1)} + \text{rowsum}\!\left(e^{\mathbf{S}_{ij} - m^{(j)}}\right)
$$

$$
\mathbf{O}^{(j)} = e^{m^{(j-1)} - m^{(j)}} \cdot \mathbf{O}^{(j-1)} + e^{\mathbf{S}_{ij} - m^{(j)}} \cdot V_j
$$

最终输出为 $\mathbf{O} = \mathbf{O}^{(T_c)} / l^{(T_c)}$。这个递推保证了**数学上与标准 Attention 完全等价**。

### 前向传播伪代码

```
算法: Flash Attention 前向传播
输入: Q, K, V ∈ R^{N×d}, 块大小 B_r, B_c
输出: O ∈ R^{N×d}

1. 将 Q 分成 T_r = ⌈N/B_r⌉ 块, K/V 分成 T_c = ⌈N/B_c⌉ 块
2. 初始化 O = 0, l = 0, m = -∞  (均为 R^{N} 向量)
3. for j = 1 to T_c:                    # 外层遍历 KV 块
4.     从 HBM 加载 K_j, V_j 到 SRAM
5.     for i = 1 to T_r:                # 内层遍历 Q 块
6.         从 HBM 加载 Q_i, O_i, l_i, m_i 到 SRAM
7.         计算 S_ij = Q_i @ K_j^T ∈ R^{B_r × B_c}    (在 SRAM 中)
8.         计算 m_ij = rowmax(S_ij)
9.         计算 P_ij = exp(S_ij - m_ij)
10.        计算 l_ij = rowsum(P_ij)
11.        更新 m_new = max(m_i, m_ij)
12.        更新 l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
13.        更新 O_i = exp(m_i - m_new) * O_i + exp(m_ij - m_new) * P_ij @ V_j
14.        将 O_i, l_new, m_new 写回 HBM
15. 返回 O = O / l   (逐行 scale)
```

关键点：**$S_{ij}$ 矩阵从未写入 HBM**，它在 SRAM 中计算、使用、然后丢弃。这就是为什么显存从 $O(N^2)$ 降到了 $O(N)$。

### PyTorch 实现

```python
import torch
import math

def flash_attention_forward(Q, K, V, block_size: int = 64):
    """
    Flash Attention 前向传播的纯 PyTorch 模拟（仅用于教学）。
    真正的 Flash Attention 在 CUDA kernel 中让中间矩阵全程驻留 SRAM，
    这里用张量切片把数据流和数学等价性显式画出来。

    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        block_size: 单个切片在 SRAM 上能放下的行数
    Returns:
        O: (batch, heads, seq_len, head_dim)
    """
    batch, heads, seq_len, head_dim = Q.shape
    q_block_size  = min(block_size, seq_len)
    kv_block_size = min(block_size, seq_len)
    num_q_tiles   = math.ceil(seq_len / q_block_size)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    out_buffer   = torch.zeros_like(Q)
    denom_buffer = torch.zeros(batch, heads, seq_len, 1, device=Q.device, dtype=Q.dtype)
    rmax_buffer  = torch.full(
        (batch, heads, seq_len, 1), -torch.inf, device=Q.device, dtype=Q.dtype,
    )

    q_tiles    = list(Q.split(q_block_size, dim=2))
    k_tiles    = list(K.split(kv_block_size, dim=2))
    v_tiles    = list(V.split(kv_block_size, dim=2))
    out_tiles  = list(out_buffer.split(q_block_size, dim=2))
    den_tiles  = list(denom_buffer.split(q_block_size, dim=2))
    rmax_tiles = list(rmax_buffer.split(q_block_size, dim=2))

    # 外层走 KV：每片 K/V 只从 HBM 读一次（v1 的关键约束）
    for k_tile, v_tile in zip(k_tiles, v_tiles):
        for idx in range(num_q_tiles):
            q_tile     = q_tiles[idx]
            prev_out   = out_tiles[idx]
            prev_denom = den_tiles[idx]
            prev_rmax  = rmax_tiles[idx]

            # (1) 注意力分数：在 SRAM 算完即丢，永不落盘
            scores_tile = (q_tile @ k_tile.transpose(-2, -1)) * softmax_scale

            # (2) 当前切片的 softmax 统计量（先减最大值再 exp）
            tile_max  = scores_tile.amax(dim=-1, keepdim=True)
            probs     = torch.exp(scores_tile - tile_max)
            tile_den  = probs.sum(dim=-1, keepdim=True)

            # (3) 把"历史 max / 分母"和"当前切片 max / 分母"对齐到统一 max
            merged_max = torch.maximum(prev_rmax, tile_max)
            scale_old  = torch.exp(prev_rmax - merged_max)
            scale_cur  = torch.exp(tile_max  - merged_max)

            # (4) 在线累积——为了贴近 v2 的"先分子后归一"，这里仍维持 v1 的"边累积边归一"
            new_denom = scale_old * prev_denom + scale_cur * tile_den
            out_tiles[idx]  = scale_old * prev_out + scale_cur * (probs @ v_tile)
            den_tiles[idx]  = new_denom
            rmax_tiles[idx] = merged_max

    # 这里才把累计分子真正除以总分母——和 v1 论文等价（论文是边除边累，数学上等同）
    out_normalized = [out_tiles[i] / den_tiles[i] for i in range(num_q_tiles)]
    return torch.cat(out_normalized, dim=2)

# 验证与标准 Attention 等价
torch.manual_seed(42)
B, H, N, d = 2, 4, 128, 64
Q = torch.randn(B, H, N, d)
K = torch.randn(B, H, N, d)
V = torch.randn(B, H, N, d)

# 标准 Attention
scale = 1.0 / math.sqrt(d)
S = (Q @ K.transpose(-2, -1)) * scale
P = torch.softmax(S, dim=-1)
O_standard = P @ V

# Flash Attention
O_flash = flash_attention_forward(Q, K, V, block_size=32)

print(f"最大误差: {(O_standard - O_flash).abs().max():.2e}")  # ~1e-6，浮点精度误差
```

### 反向传播：重计算 vs 存储

标准 Attention 的反向传播需要注意力矩阵 $P \in \mathbb{R}^{N \times N}$，这正是我们想避免存储的。Flash Attention 的解决方案是**重计算（recomputation）**：

1. **前向传播**：只保存 $O, l, m$（输出和 softmax 统计量），不保存 $S$ 和 $P$
2. **反向传播**：利用保存的 $l, m$，在 SRAM 中重新计算 $S$ 和 $P$ 的每个块
3. **额外计算量**：反向传播多做了一次分块矩阵乘法，但由于 IO 大幅减少，总体仍然更快

```
反向传播关键步骤：
1. 从 HBM 加载 Q_i, K_j, V_j, O_i, l_i, m_i, dO_i
2. 在 SRAM 中重计算: S_ij = Q_i @ K_j^T, P_ij = softmax(S_ij)  ← 利用 l_i, m_i
3. 计算 dV_j += P_ij^T @ dO_i
4. 计算 dP_ij = dO_i @ V_j^T
5. 计算 dS_ij = P_ij ⊙ (dP_ij - rowsum(dP_ij ⊙ P_ij))    ← softmax 反向
6. 计算 dQ_i += dS_ij @ K_j, dK_j += dS_ij^T @ Q_i
7. 写回 dQ_i, dK_j, dV_j 到 HBM
```

**为什么重计算反而更快？** 因为重计算的代价是 $O(N^2 d)$ 的额外 FLOPs（与前向相同），但节省了 $O(N^2)$ 的 HBM 读写。在现代 GPU 上，IO 是瓶颈（HBM 带宽远低于计算吞吐），用计算换 IO 是划算的。

### Flash Attention 2 的优化

Flash Attention 2 在 v1 基础上做了三项关键优化，将速度进一步提升 ~2x：

**优化 1：减少非矩阵乘法 FLOPs**

v1 中大量时间花在 rescaling 操作（乘以 $e^{m_{old} - m_{new}}$）上。v2 将 rescaling 推迟到内层循环结束后一次性做：

```python
# v1: 每个块都做完整 rescaling
O_i = (l_old / l_new) * exp(m_old - m_new) * O_i + (exp(m_block - m_new) / l_new) * PV

# v2: 延迟 rescaling，最后一步才除以 l
O_i = exp(m_old - m_new) * O_i + P_tilde @ Vj  # 不除以 l
# ... 内层循环结束后 ...
O_i = O_i / l_final  # 一次性 scale
```

在 GPU 上，矩阵乘法（GEMM）由 Tensor Core 加速，而逐元素操作（rescaling）只能用普通 CUDA Core。减少非 matmul FLOPs 能显著提升 Tensor Core 利用率。

**优化 2：更好的并行——外层遍历 Q**

v1 外层遍历 KV、内层遍历 Q，导致每个 Q 块的输出需要反复读写 HBM。v2 反转循环顺序：

```
v1: for KV_block → for Q_block   # O 被反复读写
v2: for Q_block → for KV_block   # 每个 Q 块的 O 只写一次 HBM
```

这使得每个 thread block 独立处理一个 Q 块，不同 Q 块之间无需通信，**在 GPU SM 之间实现了完美并行**。

**优化 3：序列长度维度的并行**

v1 只在 batch 和 head 维度做并行。当 batch size 较小时（如推理），SM 利用率不高。v2 额外在序列长度维度做并行（将 Q 的不同块分配到不同 SM），大幅提升了小 batch 场景的效率。

### IO 复杂度分析

| 算法 | FLOPs | HBM 读写量 | IO 复杂度 |
|------|-------|-----------|----------|
| 标准 Attention | $O(N^2 d)$ | $O(N^2 + Nd)$ | 受 $N^2$ 中间矩阵 IO 限制 |
| Flash Attention | $O(N^2 d)$ | $O(N^2 d^2 / M)$ | $M$ = SRAM 大小 |

**推导**：Flash Attention 的外层有 $T_c = N/B_c$ 次迭代，内层有 $T_r = N/B_r$ 次迭代。每次内层迭代从 HBM 读取 $Q_i$（$B_r \times d$）、$K_j$、$V_j$（$B_c \times d$）。选择最优块大小 $B_c = \Theta(M/d)$、$B_r = \Theta(\min(M/d, d))$（确保块能放入 SRAM），总 HBM 访问量为：

$$
\text{IO} = \Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

当 $M = \Theta(Nd)$ 时（SRAM 足够大），IO 降为 $O(Nd)$，等价于只读写 Q、K、V 各一次。

**数值示例**：$N = 4096$, $d = 128$, $M = 20\text{MB}$（A100 SRAM）

- 标准 Attention IO: $N^2 \approx 16M$ 读写（注意力矩阵 $S$ 和 $P$）
- Flash Attention IO: $N^2 d^2 / M \approx 16M \times 16K / 20M \approx 13M$

看似差不多，但 Flash Attention 避免了写 $S, P$ 到 HBM 再读回的**两次**完整 IO，且块大小经过优化后实际加速更显著。

### 面试考点：为什么 Flash Attention 更快但 FLOPs 相同？

这是一个非常经典的问题，核心答案是**Roofline Model**：

1. **FLOPs 不变**：Flash Attention 计算的数学结果与标准 Attention 完全相同，矩阵乘法的次数一样
2. **IO 大幅减少**：标准 Attention 需要将 $N^2$ 大小的中间矩阵 $S$ 和 $P$ 写入 HBM 再读回，而 Flash Attention 将这些中间结果保持在 SRAM 中
3. **瓶颈转移**：标准 Attention 是 **IO-bound**（显存带宽是瓶颈），Flash Attention 通过减少 IO 将瓶颈转移到 **compute-bound**，从而真正利用上 GPU 的算力
4. **重计算的"免费午餐"**：反向传播多做的那次前向重计算，其 FLOPs 增加约 33%，但 IO 节省远大于此——在 A100 上净加速 2-4 倍

> **一句话总结：** Flash Attention 不是"算得更快"，而是"搬数据搬得更少"。在 GPU 上，SRAM 带宽是 HBM 的 ~10 倍，减少 HBM 访问就是最大的加速。

### Online Softmax：从两遍扫描到一遍扫描

标准 Softmax 需要**两遍扫描**才能完成计算：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j
$$

- **第一遍**：扫描所有元素，求全局最大值 $m$（数值稳定性所需）和求和 $\ell = \sum_j e^{x_j - m}$
- **第二遍**：再扫描一次，计算每个元素的 softmax 值

这意味着整个向量必须在内存中被访问两次。对于 Flash Attention 的分块计算来说，我们无法一次看到完整的行——每次只能看到一个 block。

::: tip Online Softmax 的核心思想
利用指数函数的性质，当新 block 带来更大的 max 值 $m_{\text{new}}$ 时，只需对历史累积量乘一个**修正因子** $e^{m_{\text{old}} - m_{\text{new}}}$ 即可：

$$
\ell_{\text{new}} = \ell_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_{j \in \text{block}} e^{x_j - m_{\text{new}}}
$$

这样只需**一遍扫描**就能得到全局正确的 softmax 分母。
:::

**逐元素 Online Softmax 实现：**

```python
import torch

X = torch.tensor([1.0, 1.5, 1.8, 2.0, 1.4, 2.1])

# ---- 标准 safe softmax（两遍扫描）----
X_max = X.max()
X_safe_softmax = torch.exp(X - X_max) / torch.exp(X - X_max).sum()

# ---- Online Softmax（一遍扫描）----
m_cur = torch.tensor(float('-inf'))
l_cur = torch.tensor(0.0)

for i in range(len(X)):
    m_new = torch.max(m_cur, X[i])
    # 修正历史 sum + 加入新元素
    l_cur = l_cur * torch.exp(m_cur - m_new) + torch.exp(X[i] - m_new)
    m_cur = m_new

X_online_softmax = torch.exp(X - m_cur) / l_cur
print(torch.allclose(X_safe_softmax, X_online_softmax))  # True
```

**分块 Online Softmax**（Flash Attention 实际使用的形式）：

```python
BLOCK = 3
X_blocks = X.split(BLOCK)

m_cur = torch.tensor(float('-inf'))
l_cur = torch.tensor(0.0)

for blk in X_blocks:
    m_blk = blk.max()
    m_new = torch.max(m_cur, m_blk)
    l_cur = l_cur * torch.exp(m_cur - m_new) \
          + torch.exp(blk - m_new).sum()
    m_cur = m_new

X_block_online_softmax = torch.exp(X - m_cur) / l_cur
print(torch.allclose(X_safe_softmax, X_block_online_softmax))  # True
```

> 分块 Online Softmax 是 Flash Attention 能在 SRAM 中分块完成 Softmax 的数学基础——**看到新 block 时修正旧统计量，而非回头重算**。

### Flash Attention 反向传播

Flash Attention 反向传播的关键挑战是：前向传播**没有保存** $N \times N$ 的注意力矩阵 $P$（否则就失去了省显存的意义），但反向传播需要 $P$ 来计算 $dQ, dK, dV$。

::: info 核心策略：重计算 (Recomputation)
前向时只保存 $Q, K, V, O$ 和每行的 **logsumexp** $L_i = m_i + \log(\ell_i)$。反向时在 SRAM 中**重新计算**每个 block 的 $S_{ij}$ 和 $P_{ij}$，避免从 HBM 读取 $N^2$ 大小的矩阵。
:::

**梯度推导**

给定上游梯度 $dO$，标准 Attention 的梯度为：

$$
\begin{aligned}
dV &= P^\top \, dO \\
dP &= dO \, V^\top \\
dS &= P \odot (dP - D), \quad D_i = \sum_j P_{ij} \, dP_{ij} = \text{rowsum}(O \odot dO) \\
dQ &= dS \, K \, / \sqrt{d} \\
dK &= dS^\top \, Q \, / \sqrt{d}
\end{aligned}
$$

其中 $D_i = \sum_j P_{ij} \, dP_{ij}$ 可以用 $O$ 和 $dO$ 直接算出（$D = \text{rowsum}(O \odot dO)$），不需要存储 $P$。

::: details 为什么 $D_i = \text{rowsum}(O \odot dO)$？
因为 $O_i = \sum_j P_{ij} V_j$，所以 $\sum_j P_{ij} (dO_i \cdot V_j) = dO_i \cdot \sum_j P_{ij} V_j = dO_i \cdot O_i$。这正是逐行求和 $\text{rowsum}(O \odot dO)$ 的第 $i$ 个元素。
:::

**分块反向传播实现：**

> 以下实现参考了 Flash Attention 2 论文 (Dao, 2023) 的算法描述，代码经过教学化改写。

```python
import torch
import math

torch.manual_seed(42)
n, dim, nb = 12, 8, 4  # 序列长度, 头维度, block 大小
block = n // nb

Q = torch.randn(n, dim, requires_grad=True)
K = torch.randn(n, dim, requires_grad=True)
V = torch.randn(n, dim, requires_grad=True)

# ---- Flash Attention 前向（保存 O 和 L）----
def flash_attention_forward(Q, K, V):
    O = torch.zeros_like(Q)
    L = torch.zeros(n, 1)
    for tq in range(block):
        q = Q[tq*nb:(tq+1)*nb, :]
        o_old = torch.zeros_like(q)
        l_old = m_old = torch.zeros(nb, 1)
        for tk in range(block):
            k = K[tk*nb:(tk+1)*nb, :]
            v = V[tk*nb:(tk+1)*nb, :]
            s = q @ k.t() / math.sqrt(dim)
            m = s.max(dim=1, keepdim=True).values
            m_new = torch.maximum(m, m_old)
            l = torch.exp(s - m_new).sum(dim=1, keepdim=True)
            l_new = l_old * torch.exp(m_old - m_new) + l
            o_old = l_old * o_old * torch.exp(m_old - m_new) \
                  + torch.exp(s - m_new) @ v
            o_old = o_old / l_new
            l_old, m_old = l_new, m_new
        O[tq*nb:(tq+1)*nb, :] = o_old
        L[tq*nb:(tq+1)*nb, :] = m_old + l_old.log()
    return O, L

O, L = flash_attention_forward(Q, K, V)

# ---- Flash Attention 反向（分块重计算）----
dO = torch.randn_like(O)  # 模拟上游梯度

def flash_attention_backward(Q, K, V, O, dO, L):
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    # D_i = rowsum(O * dO)，不需要 P
    D = (O * dO).sum(dim=1, keepdim=True)

    for tk in range(block):          # 外层遍历 KV block
        k = K[tk*nb:(tk+1)*nb, :]
        v = V[tk*nb:(tk+1)*nb, :]
        for tq in range(block):      # 内层遍历 Q block
            q  = Q[tq*nb:(tq+1)*nb, :]
            o  = O[tq*nb:(tq+1)*nb, :]
            do = dO[tq*nb:(tq+1)*nb, :]
            l  = L[tq*nb:(tq+1)*nb, :]
            d  = D[tq*nb:(tq+1)*nb, :]

            # ---- 重计算 attention（无需从 HBM 读 P）----
            s = q @ k.t() / math.sqrt(dim)
            p = torch.exp(s - l)     # 利用 L = m + log(l) 还原 softmax

            # ---- 梯度计算 ----
            dv = p.t() @ do
            dp = do @ v.t()
            ds = p * (dp - d)        # softmax 反向的紧凑形式
            dq = ds @ k / math.sqrt(dim)
            dk = ds.t() @ q / math.sqrt(dim)

            # ---- 累加到全局梯度 ----
            dV[tk*nb:(tk+1)*nb, :] += dv
            dQ[tq*nb:(tq+1)*nb, :] += dq
            dK[tk*nb:(tk+1)*nb, :] += dk

    return dQ, dK, dV

dQ_flash, dK_flash, dV_flash = flash_attention_backward(
    Q, K, V, O, dO, L
)
```

### 用 PyTorch Autograd 验证正确性

我们实现的反向传播是否正确？我们用 PyTorch 自动微分作为 ground truth 进行对比：

```python
# ---- PyTorch 标准 attention + autograd ----
S = Q @ K.t() / math.sqrt(dim)
P = torch.softmax(S, dim=-1)
O_ref = P @ V

# 用相同的 dO 计算 autograd 梯度
O_ref.backward(dO)

print("dQ allclose:", torch.allclose(dQ_flash, Q.grad, atol=1e-5))
print("dK allclose:", torch.allclose(dK_flash, K.grad, atol=1e-5))
print("dV allclose:", torch.allclose(dV_flash, V.grad, atol=1e-5))
# 输出：全部 True
```

::: warning 反向传播的 IO 分析
标准反向传播需要从 HBM 读取 $P$（大小 $N^2$），IO 复杂度 $O(N^2)$。Flash Attention 反向通过**重计算** $P$，将 IO 降至 $O(N^2 d / M)$（与前向相同），代价是多做了一次 $O(N^2 d)$ 的矩阵乘法——但在 GPU 上这是 **compute-bound** 操作，远快于 HBM 读写。
:::

---

## Tensor Product Attention (TPA)

### 核心思想

标准注意力中，K 和 V 通过单个线性投影 $K = XW^K$ 得到，每个 token 独立生成 KV。**TPA（Tensor Product Attention）** 引入了一种新的分解方式：将 K、V 的投影分解为两个低秩矩阵的**张量积**，让模型在更低的参数量和 KV Cache 开销下保持表达能力。

### 数学公式

TPA 将 K、V 的计算分解为两个低秩分量 $A$（token 级）和 $B$（token 级）的乘积：

$$
A_k = XW_A^k \in \mathbb{R}^{n \times h \times r}, \quad B_k = XW_B^k \in \mathbb{R}^{n \times r \times d_h}
$$

$$
K = \frac{1}{r} A_k \cdot B_k \in \mathbb{R}^{n \times h \times d_h}
$$

V 的计算方式完全类似。其中 $r$ 是分解的秩（rank），远小于 $d_h$。Q 仍然使用标准的线性投影。

**关键洞察：** 这本质上是对 KV 投影矩阵做了 CP 分解（Canonical Polyadic Decomposition），每个 token 的 K/V 由两个低秩因子按位相乘得到，兼顾表达能力与压缩效率。

### 简化版代码实现

以下是简化版 TPA 实现：

```python
import torch
import torch.nn as nn

class TPAProjection(nn.Module):
    """Tensor Product Attention 的 QKV 投影"""
    def __init__(self, d_model=512, n_head=8, head_dim=64, rank=4):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank

        # Q 使用标准投影
        self.W_q = nn.Linear(d_model, n_head * head_dim, bias=False)
        # K, V 各用两个低秩投影 (CP 分解)
        self.W_A_k = nn.Linear(d_model, n_head * rank, bias=False)
        self.W_B_k = nn.Linear(d_model, rank * head_dim, bias=False)
        self.W_A_v = nn.Linear(d_model, n_head * rank, bias=False)
        self.W_B_v = nn.Linear(d_model, rank * head_dim, bias=False)

    def forward(self, x):
        bs, seq_len, _ = x.size()
        q = self.W_q(x).view(bs, seq_len, self.n_head, self.head_dim)

        # K = (1/r) * A_k @ B_k
        A_k = self.W_A_k(x).view(bs * seq_len, self.n_head, self.rank)
        B_k = self.W_B_k(x).view(bs * seq_len, self.rank, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank)
        k = k.view(bs, seq_len, self.n_head, self.head_dim)

        # V = (1/r) * A_v @ B_v
        A_v = self.W_A_v(x).view(bs * seq_len, self.n_head, self.rank)
        B_v = self.W_B_v(x).view(bs * seq_len, self.rank, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank)
        v = v.view(bs, seq_len, self.n_head, self.head_dim)

        return q, k, v

# 使用示例
tpa = TPAProjection(d_model=512, n_head=8, head_dim=64, rank=4)
x = torch.randn(2, 16, 512)
q, k, v = tpa(x)
print(q.shape, k.shape, v.shape)
# torch.Size([2, 16, 8, 64]) torch.Size([2, 16, 8, 64]) torch.Size([2, 16, 8, 64])
```

### TPA vs 标准 Attention 对比

| 特性 | 标准 MHA | TPA |
|------|----------|-----|
| **KV 投影参数** | $2 \times d \times hd_h$ | $2 \times d \times (hr + rd_h)$ |
| **参数压缩比** | 基准 | 当 $r \ll d_h$ 时显著减少 |
| **KV Cache** | 标准 | 可只缓存 $A$、$B$ 因子 |
| **表达能力** | 基准 | rank 足够时接近 MHA |
| **适用场景** | 通用 | KV Cache 受限的长序列推理 |

---

## 注意力变体对比

| 特性 | MHA | MQA | GQA | MLA | TPA |
|------|-----|-----|-----|-----|-----|
| **Q 头数** | $h$ | $h$ | $h$ | $h$（低秩分解） | $h$ |
| **KV 头数** | $h$ | 1 | $g$（$1 < g < h$） | 全头（从 latent 恢复） | $h$（低秩因子） |
| **KV Cache 大小** | $2hd_h l$ | $2d_h l$ | $2gd_h l$ | $d_c l$（$d_c \ll hd_h$） | $2h(r + rd_h/h)l$ |
| **KV 参数量** | $2 \times d \times d$ | $2 \times d \times d_h$ | $2 \times d \times gd_h$ | $d \times d_c + 2 \times d_c \times d$ | $2 \times d \times (hr + rd_h)$ |
| **精度保持** | 基准 | 略有下降 | 接近 MHA | 接近 MHA | 接近 MHA |
| **代表模型** | GPT-3, BERT | PaLM | Llama 2/3 | DeepSeek-V2/V3 | Tensor Product Attention |
| **核心思想** | 多头并行 | KV 共享 | 分组 KV 共享 | 低秩 KV 压缩 | KV 张量积分解 |

> **关键洞察：** MQA/GQA 是在"头数维度"上压缩；MLA 是在"特征维度"上压缩（类似 LoRA 思想）；TPA 是在"投影矩阵"上做 CP 分解——三者从不同角度减少 KV 开销。

---

## 苏格拉底时刻

1. **为什么 MHA 不用一个大头？** 多头让模型在不同子空间并行捕捉不同类型的关系（语法、语义、位置等）。单头只能在一个空间学习，表达能力受限。

2. **Flash Attention 在数学上完全等价，加速从何而来？** 减少了 HBM 读写次数。标准 attention 的 IO 复杂度为 $O(n^2)$（存储中间矩阵 $S, P$），Flash Attention 通过分块将 IO 降至 $O(n^2 d / M)$，其中 $M$ 为 SRAM 大小。

3. **GQA 的 `repeat_interleave` 是否增加了计算量？** 注意力分数的计算量不变（仍为 $n^2 hd_h$）。减少的是投影参数和 KV Cache 的存储/传输开销。在 GPU SRAM 中 repeat 操作几乎免费。

4. **MLA 的"矩阵吸收"为什么能 work？** 基于矩阵乘法结合律：$W^{\text{up}}(W^{\text{down}} \cdot X) = (W^{\text{up}} W^{\text{down}}) \cdot X$。训练时分两步省显存；推理时合并为一步保精度。

5. **Online Softmax 为什么能单遍扫描？** 利用指数函数的性质：$e^{x-m_{\text{new}}} = e^{x-m_{\text{old}}} \cdot e^{m_{\text{old}}-m_{\text{new}}}$，当发现新的 max 时，只需对历史累积量乘一个修正因子。

---

## 常见问题 & 面试考点

**Q1: 注意力的计算复杂度是多少？**
时间复杂度 $O(n^2 d)$，空间复杂度 $O(n^2)$（存储注意力矩阵）。这是长序列建模的主要瓶颈。

**Q2: Causal Mask 在训练和推理中的作用？**
训练时：对注意力矩阵的上三角填充 $-\infty$，防止 token 看到未来信息，保证自回归训练的正确性。推理时（KV Cache 模式）：每步只计算新 token 对所有历史 token 的注意力，mask 隐式生效。

**Q3: KV Cache 为什么只缓存 K 和 V，不缓存 Q？**
自回归推理时，每步只有一个新 token 生成新的 Q（长度为 1），无需缓存。而 K、V 需要保留所有历史 token 的结果。

**Q4: MQA 为什么效果只略有下降？**
高维特征存在冗余，多个头的 KV 投影高度相关。实验表明 KV 的多样性对模型质量影响远小于 Q 的多样性。

**Q5: Flash Attention 能用于训练吗？**
能。Flash Attention 同时优化了前向和反向传播的 IO，训练速度提升 2-4 倍，显存减少 5-20 倍。

---

## 推荐资源

### 论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Flash Attention 论文
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — v2 改进
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — GQA 论文
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) — MLA 原始论文
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867) — Online Softmax 原始论文

### 博客与可视化

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar — 注意力可视化详解
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng — 注意力机制综述博客（前 OpenAI 应用研究负责人）。从 Bahdanau 2015 加性注意力起步，统一梳理出 6 种 score 函数（content-base / additive / location-base / general / dot-product / scaled dot-product）和 3 种结构维度（self / global / local，soft / hard）。下半部分把同一框架延伸到 Neural Turing Machine（content + location 寻址）、Pointer Network（attention 直接挑输入位置）、Transformer（K/V/Q + multi-head + 因果 mask）和 SAGAN（attention 进入 GAN）。读这篇的最大价值：**理解 Transformer 的 scaled dot-product 不是凭空冒出来的，而是 Bahdanau additive → Luong dot-product → 加 1/√n 的自然演化**——把今天工业界的 attention 实现锚定在历史脉络里。建议作为 attention 学习的第一篇综述。

### 代码参考

- **LLMs-from-scratch / ch03**（Sebastian Raschka，Manning 出版书《Build a Large Language Model (From Scratch)》ch03 配套代码）：[multihead-attention.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/01_main-chapter-code/multihead-attention.ipynb) 在同一个 notebook 里平行给出三段实现，正好把本页"为什么 reshape 比拼接快"这个工程直觉补全：

  1. `CausalSelfAttention` —— 单头因果注意力，用 `register_buffer('mask', triu(...))` + `masked_fill_(-inf)` 实现因果遮挡，是理解 mask 写法的最小骨架。
  2. `MultiHeadAttentionWrapper` —— 用 `nn.ModuleList([CausalSelfAttention(...) for _ in range(num_heads)])` 串起 N 个独立头，前向时 `torch.cat([head(x) for head in self.heads], dim=-1)` 做拼接。直观但慢。
  3. `MultiHeadAttention`（Variant B，reshape 版）—— 一次性 `nn.Linear(d_in, d_out)` 投影后用 `view(b, n, num_heads, head_dim).transpose(1, 2)` 隐式拆头，对应本页"完整过程：拆分 → 并行计算 → 拼接"小节的工程实现。

  ```python
  # 节选自 Variant B：一次大矩阵投影，再 reshape 出 num_heads 维
  keys = self.W_key(x)                                          # (b, n, d_out)
  keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
  keys = keys.transpose(1, 2)                                   # (b, h, n, head_dim)
  ```

  对照阅读 Wrapper 版与 reshape 版能看到一个清晰的事实：**数学上等价的 MHA，工程实现差一个 `view+transpose` 就能省掉 N 次小矩阵乘法**——这是后续 Flash Attention、KV Cache、张量并行所有优化的前提。
