---
title: "Llama 架构"
description: "RoPE 旋转位置编码、RMSNorm、GQA、SwiGLU 及长上下文扩展（YaRN）"
topics: [Llama, RoPE, RMSNorm, GQA, SwiGLU, NTK-aware-RoPE, YaRN, position-interpolation]
prereqs: [architecture/gpt]
---
# Llama 架构

> **一句话总结:** Llama 在 GPT 的 Decoder-Only 基础上引入了 RoPE 旋转位置编码、RMSNorm、GQA 分组查询注意力和 SwiGLU 门控激活四大改进，成为开源 LLM 的架构标杆，几乎所有后续开源模型都采用了这套"Llama-style"配方。

## 在大模型体系中的位置

Meta 于 2023 年发布的 Llama 系列模型，在 GPT 的架构基础上融合了近年来被验证有效的多项改进。Llama 架构的意义不仅在于单个模型的性能，更在于它为开源社区提供了一套经过大规模验证的"最佳实践组合"。此后的 Qwen、DeepSeek、Mistral、Yi 等主流开源模型几乎都采用了相同或高度相似的架构设计。

```
GPT 架构                    Llama 架构改进
─────────                   ──────────────
绝对位置编码            →    RoPE 旋转位置编码（更好的外推能力）
LayerNorm              →    RMSNorm（更快，效果相当）
MHA 多头注意力          →    GQA 分组查询注意力（KV Cache 更小）
ReLU / GELU FFN        →    SwiGLU 门控激活（更强的表达能力）
```

## 核心概念

### RoPE 旋转位置编码

位置编码让 Transformer "知道" Token 的位置。原始 Transformer 使用固定的正弦位置编码，GPT 使用可学习的绝对位置编码。Llama 采用了 RoPE（Rotary Position Embedding），这是目前最主流的位置编码方案。

**为什么需要 RoPE？** 绝对位置编码 (abs PE) 的问题在于，两个向量的注意力分数受绝对位置影响：

$$(E_m+P_m)(E_n+P_n)^T = E_mE_n^T + E_mP_n^T + P_mE_n^T + P_mP_n^T$$

其中 $P_mP_n^T$ 可以表示相对位置，但其余项包含绝对位置信息。这意味着在 1~512 位置学好的特征，在 10000~10512 位置可能失效。

**核心思想：** 寻找一种理想的位置变换 $f$，使得 $f(m)f(n)^T = f(m-n)$。RoPE 通过**旋转**来实现这一点——将 Query 和 Key 向量每两个维度做一组二维旋转：

$$\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_k & -\sin m\theta_k \\ \sin m\theta_k & \cos m\theta_k \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}$$

其中频率 $\theta_k = \text{base}^{-2k/d}$（默认 base=10000）。对于位置 $m$ 和 $n$ 的注意力分数：

$$S_{mn,i} = Q_{m,(i)} R(m\theta_i) R^T(n\theta_i) K^T_{n,(i)} = Q_{m,(i)} R((m-n)\theta_i) K^T_{n,(i)}$$

这样注意力分数只取决于相对位置 $m-n$，不受绝对位置影响。

**RoPE 实现代码：**

```python
class RotaryEmbedding(nn.Module):
    """旋转位置编码：预计算角度表，推理时对 Q/K 做旋转"""
    def __init__(self, head_dim: int = 64, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # 频率：theta_k = 1 / base^(2k/d)，k = 0,1,...,d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, inv_freq)          # [max_seq_len, d/2]

        # 用 torch.polar 生成复数旋转因子，再拆成实部/虚部
        emb = torch.polar(torch.ones_like(angles), angles)  # e^{i * angle}
        cos_table = emb.real.repeat(1, 2)                    # [max_seq_len, d]
        sin_table = emb.imag.repeat(1, 2)                    # [max_seq_len, d]
        self.register_buffer("cos_table", cos_table)
        self.register_buffer("sin_table", sin_table)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, n_heads, seq_len, head_dim]"""
        seq_len = x.size(2)

        # 将相邻偶/奇维度交换并取负，实现 (-x1,x0,-x3,x2,...)
        x_paired = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1)
        x_rotated = x_paired.reshape(x.shape)  # 交错排列

        cos = self.cos_table[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos * x + sin * x_rotated
```

在实际 Llama 模型中，RoPE 的角度表在模型顶层缓存一份（`register_buffer`），传参给每一层 decoder block，避免重复存储。注意 RoPE 的维度是 `head_dim` 而非 `dim`，多头共享一份 RoPE 参数。

**RoPE 的关键优势：**

1. **相对位置感知：** 两个位置 $m$ 和 $n$ 的注意力得分只取决于相对距离 $m - n$
2. **更好的长度外推：** 相比绝对位置编码，RoPE 在超过训练长度时退化更缓慢
3. **计算高效：** 只需对 Q、K 做旋转操作，不增加额外参数

**RoPE 的长度外推扩展：** 当需要将上下文窗口从 4K 扩展到 128K 时，常用方法包括 NTK-aware 插值（调整频率基 $\theta$）、YaRN（结合 NTK 和注意力缩放）、ABF（Llama 3 采用，直接增大 base frequency）等。

### 长上下文扩展：从 4K 到 128K+

RoPE 虽然编码相对位置，但它并非天然支持任意长度。模型在训练长度（如 4096）之外的位置上，RoPE 产生的角度是训练时从未见过的，注意力分数会出现严重退化。这就是**长度外推问题**。

#### 外推失败的根本原因

RoPE 的频率为 $\theta_k = \text{base}^{-2k/d}$。对于低频分量（$k$ 较大），$\theta_k$ 很小，位置 $m$ 对应的角度 $m \cdot \theta_k$ 在训练范围内只旋转了很小的角度。当 $m$ 超出训练长度时，这些低频分量的角度进入了训练时未覆盖的区域，模型对这些角度值没有泛化能力。

高频分量（$k$ 较小）问题不大，因为它们在训练范围内已经旋转了多圈，相当于已经"见过"了各种角度。

#### 位置插值（Position Interpolation, PI）

**核心思想**：将超出训练长度的位置**线性压缩**回训练范围内，而不是外推到未见过的区域。

$$m' = m \cdot \frac{L_{\text{train}}}{L_{\text{target}}}$$

例如，训练长度 4096，目标长度 32768，则所有位置除以 $32768/4096 = 8$。位置 32768 被映射到位置 4096，回到训练分布内。

```python
def position_interpolation_rope(x, seq_len, head_dim, base=10000.0,
                                  train_len=4096, target_len=32768):
    """位置插值 RoPE：线性缩放位置索引"""
    scale = train_len / target_len  # 缩放因子 = 4096/32768 = 0.125

    positions = torch.arange(seq_len, dtype=torch.float32)
    positions = positions * scale  # 关键：缩放位置

    freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(positions, freqs)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin
```

**优点**：简单有效，只需少量微调（~1000 步）即可适应新长度。

**缺点**：所有频率被均匀压缩，高频分量的分辨率下降，导致短距离位置区分能力变弱。直觉上，原来位置 1 和位置 2 之间的角度差变成了原来的 1/8。

#### NTK-aware RoPE

**核心思想**：不缩放位置 $m$，而是修改频率基 $\text{base}$，让低频分量被压缩更多，高频分量保持不变。

$$\text{base}' = \text{base} \cdot \alpha^{d/(d-2)}$$

其中 $\alpha = L_{\text{target}} / L_{\text{train}}$ 是扩展倍数。

**为什么叫 NTK-aware？** 这个方法的灵感来自 Neural Tangent Kernel (NTK) 理论。NTK 理论指出，网络对高频信号和低频信号的学习速度不同。修改 base 等效于在频率空间做非均匀插值——高频保持（保留短距离分辨率），低频压缩（支持长距离）。

```python
def ntk_aware_rope(x, seq_len, head_dim, base=10000.0, alpha=8.0):
    """
    NTK-aware RoPE：修改 base 频率实现长度扩展
    alpha = target_len / train_len
    """
    # 关键：修改 base 而非缩放位置
    base_new = base * alpha ** (head_dim / (head_dim - 2))

    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = base_new ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(positions, freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

# 示例：从 4K 扩展到 32K
# alpha = 32768 / 4096 = 8
cos, sin = ntk_aware_rope(None, seq_len=32768, head_dim=128, alpha=8.0)
```

**优点**：无需微调即可直接使用（zero-shot），短距离位置区分能力保持良好。

**缺点**：长距离效果不如 PI + 微调，因为低频分量的改变对模型来说仍是分布外的。

#### YaRN：NTK-by-parts + Attention Scaling

YaRN（Yet another RoPE extensioN）是目前效果最好的长度扩展方法之一，结合了三个技巧：

**技巧 1：NTK-by-parts 插值**

将 RoPE 的频率维度分为三组，分别处理：

- **高频维度**（波长 < 训练长度）：不做任何修改，保留原始频率
- **低频维度**（波长 > 训练长度 × $\beta$）：做线性插值（类似 PI）
- **中间维度**：在不修改和线性插值之间平滑过渡

```python
def yarn_rope(seq_len, head_dim, base=10000.0, train_len=4096,
              target_len=32768, beta_fast=32, beta_slow=1):
    """YaRN: NTK-by-parts + attention scaling"""
    alpha = target_len / train_len

    freqs = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    wavelengths = 2 * math.pi / freqs  # 每个频率对应的波长

    # 分区：根据波长与训练长度的关系决定插值程度
    low = train_len / beta_fast   # 高频边界
    high = train_len / beta_slow  # 低频边界

    # 计算每个维度的插值比例 (0=不插值, 1=完全插值)
    ramp = (wavelengths - low) / (high - low)
    ramp = ramp.clamp(0, 1)

    # 混合频率：高频不变，低频做插值
    freqs_interpolated = freqs / alpha  # PI 风格的插值频率
    freqs_yarn = freqs * (1 - ramp) + freqs_interpolated * ramp

    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs_yarn)

    return torch.cos(angles), torch.sin(angles)
```

**技巧 2：Attention Scaling**

YaRN 发现长序列的注意力 logits 的熵会增大（因为 token 数更多），导致注意力分布过于平坦。解决方案是对注意力 logits 乘以一个温度因子：

$$S = \frac{QK^T}{\sqrt{d}} \cdot \frac{1}{\sqrt{t}}$$

其中 $t = 0.1 \cdot \ln(\alpha) + 1$，$\alpha$ 是扩展倍数。

**技巧 3：少量微调**

YaRN 只需约 400 步微调（~0.1% 的预训练量），即可在 128K 长度上达到良好效果。

#### Dynamic NTK

**核心思想**：根据输入序列的实际长度动态调整 base，而非使用固定的扩展倍数。

```python
def dynamic_ntk_rope(seq_len, head_dim, base=10000.0, train_len=4096):
    """Dynamic NTK: 根据输入长度自适应调整"""
    if seq_len <= train_len:
        # 未超出训练长度，使用原始 RoPE
        alpha = 1.0
    else:
        # 动态计算 alpha
        alpha = seq_len / train_len

    base_new = base * alpha ** (head_dim / (head_dim - 2))
    freqs = base_new ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)
```

**优点**：完全无需微调，推理时自适应。**缺点**：效果不如 YaRN + 微调。

#### 各方法效果对比

| 方法 | 扩展倍数 | 是否需要微调 | 短距离保持 | 长距离效果 | 代表应用 |
|------|---------|------------|-----------|-----------|---------|
| **PI** | 8-16x | 需要（~1000步） | 略有下降 | 好 | Code Llama 16K |
| **NTK-aware** | 4-8x | 不需要 | 好 | 中等 | 开源社区 |
| **YaRN** | 16-64x | 少量（~400步） | 好 | 很好 | Mistral, Yi |
| **Dynamic NTK** | 自适应 | 不需要 | 好 | 中等 | Qwen 早期版本 |
| **ABF (Llama 3)** | 8x | 随预训练完成 | 好 | 好 | Llama 3 (base=500000) |

> **Llama 3 的做法**：直接将 base 从 10000 增大到 500000，并在预训练过程中逐步增加序列长度（从 8K 到 128K）。这相当于把 NTK-aware 的思路融入了预训练阶段，省去了后续的长度扩展步骤。

### RMSNorm（对比 LayerNorm）

RMSNorm 是 LayerNorm 的简化版本，去掉了均值中心化，只保留缩放操作。

**LayerNorm 的缺陷：**

1. re-centred 操作（$x - \mu$）与 ReLU-Like 激活函数的非对称分布冲突，导致特征分布反复中心化/去中心化
2. $\gamma$ 和 $\beta$ 参数容易学到预训练语料的偏置
3. 需计算均值和方差，计算较耗时

**RMS 统计量：** Root-Mean-Square 是一种归一化统计量，变换后数据的 RMS 值恒为 1：

$$RMS(x) = \sqrt{\frac{1}{N} \sum_i x^2_i}, \quad \tilde{x} = \frac{x}{RMS(x)} \implies RMS(\tilde{x}) = 1$$

**RMSNorm 实现代码：**

```python
class LlamaRMSNorm(nn.Module):
    """RMSNorm：仅用均方根做归一化，无均值中心化"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 RMS 并归一化（一步完成，避免中间变量）
        rms_inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x * rms_inv)
```

**RMSNorm 的尺度不变性：** 输入缩放 $s > 0$ 倍，输出不变：

$$\frac{x}{RMS(x)} = \frac{sx}{RMS(sx)} \quad \text{（因为 } RMS(sx) = s \cdot RMS(x) \text{）}$$

| 对比项 | LayerNorm | RMSNorm |
|--------|-----------|---------|
| 均值中心化 | 有（$x - \mu$） | 无 |
| 方差归一化 | 用 $\sigma$ | 用 RMS 代替 |
| 可学习参数 | $\gamma$ 和 $\beta$ | 仅 weight（缩放因子） |
| 计算速度 | 较慢（需计算均值和方差） | 更快（少一次 reduce 操作） |
| 特征分布 | 会偏移分布 | 更易保留特征分布 |

### GQA 分组查询注意力

GQA（Grouped-Query Attention）是 MHA 和 MQA 之间的折中方案。核心问题：多头注意力中 K/V 是否存在冗余？KV-Cache 能否在 `n_heads` 维度上减少？

**三种注意力方案对比：**

```
MHA: Q/K/V 各有 n_heads 头，一一对应
MQA: Q 有 n_heads 头，K/V 仅 1 头（共享给所有 Q）
GQA: Q 有 n_heads 头，K/V 有 n_kv_heads 组（每组共享给一部分 Q）
```

**GQA 实现代码（核心是将 KV 头扩展到 Q 头数量）：**

```python
class GQAAttention(nn.Module):
    """分组查询注意力：Q 头数 > KV 头数，KV 通过 expand 共享给多个 Q 头"""
    def __init__(self, hidden_dim: int = 512, num_q_heads: int = 8, num_kv_heads: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.repeats = num_q_heads // num_kv_heads  # 每个 KV 头服务几个 Q 头

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_heads)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 用 expand 代替 repeat_interleave，避免实际拷贝内存
        k = k[:, :, None, :, :].expand(B, self.num_kv_heads, self.repeats, L, self.head_dim)
        k = k.reshape(B, self.num_q_heads, L, self.head_dim)
        v = v[:, :, None, :, :].expand(B, self.num_kv_heads, self.repeats, L, self.head_dim)
        v = v.reshape(B, self.num_q_heads, L, self.head_dim)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.hidden_dim)
        return self.o_proj(out)
```

**GQA 的 KV-Cache 节省：** 如果有 $H$ 个 Q 头和 $G$ 个 KV 组，KV Cache 缩小为原来的 $G/H$。Llama 2 70B 中 $H=64, G=8$，节省 87.5%。

**GQA 与 RoPE 的配合：** 在实际 Llama 模型中，GQA 在分头后先对 Q/K 应用 RoPE，再做注意力计算。KV Cache 存储的是 `apply_rotary_pos_emb` 后的 KV：

```python
class GQAWithRotaryEmbedding(nn.Module):
    """GQA + RoPE 完整实现，对应 Llama 中的注意力层"""
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.dim
        self.num_q_heads = config.n_heads
        self.num_kv_heads = config.n_kv_heads
        self.head_dim = self.hidden_dim // self.num_q_heads
        self.repeats = self.num_q_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.head_dim * self.num_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.head_dim * self.num_kv_heads, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def forward(self, x, mask=None, rope=None):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 扩展 KV 头到 Q 头数量
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)

        # 对 Q、K 施加旋转位置编码
        q = rope.rotate(q)
        k = rope.rotate(k)

        # 注意：KV Cache 应存储旋转后的 K/V
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.hidden_dim)
        return self.o_proj(out)
```

### SwiGLU 门控激活

SwiGLU 由 Swish 激活函数和 GLU（Gated Linear Unit）结合而成，是一种 token-wise 的特征学习器：

$$h = \text{Swish}(W_\text{gate}(x)) \otimes W_\text{up}(x), \quad y = W_\text{down}(h)$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$ 是 ReLU 的平滑版本，$\otimes$ 为逐元素相乘。

**GLU 门控的本质是特征选择：** gate 是"因材施教"的，不同 token 有独立的门控权重。这比简单的非线性激活提供了更强的表达能力。

**SwiGLU 实现代码：**

```python
class SwiGLU(nn.Module):
    """SwiGLU FFN：等参数量设计 hidden = 8d/3"""
    def __init__(self, dim: int):
        super().__init__()
        hidden = dim * 8 // 3
        # 采用 HuggingFace 命名惯例：gate / up / down
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj   = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish 门控 ⊙ 线性上投影，再降维
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**等参数量设计：** 由于多了一个 $W_\text{gate}$，为保持与标准 FFN 相同的参数量，隐藏维度从 $4d$ 调整为 $\frac{8}{3}d$：

$$3(d \cdot h) = 2(d \cdot 4d) \implies h = \frac{8}{3}d$$

## 完整 Llama Block

将 LlamaRMSNorm、GQA（含 RoPE）、SwiGLU 组装成完整的 Llama 解码块：

```python
class LlamaBlock(nn.Module):
    """单个 Llama Decoder Block：Pre-Norm + GQA + SwiGLU + 残差"""
    def __init__(self, config):
        super().__init__()
        self.attn_norm = LlamaRMSNorm(config.dim)
        self.attention = GQAWithRotaryEmbedding(config)
        self.ffn_norm  = LlamaRMSNorm(config.dim)
        self.feed_forward = SwiGLU(config.dim)

    def forward(self, hidden_states, mask=None, rope=None):
        # 注意力子层 + 残差
        residual = hidden_states
        hidden_states = self.attention(self.attn_norm(hidden_states), mask, rope)
        hidden_states = residual + hidden_states
        # FFN 子层 + 残差
        residual = hidden_states
        hidden_states = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states
```

完整的 Llama 模型将多个 Block 堆叠，顶层加上 Embedding、LlamaRMSNorm 和 LM Head：

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)  # 无绝对位置编码
        self.layers = nn.ModuleList(
            [LlamaBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = LlamaRMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 旋转编码在模型层面实例化，所有 block 共享
        self.rope = RotaryEmbedding(
            head_dim=config.dim // config.n_heads,
            max_seq_len=config.max_len,
        )

    def forward(self, input_ids):
        B, L = input_ids.shape
        h = self.tok_emb(input_ids)
        for layer in self.layers:
            h = layer(h, rope=self.rope)
        h = self.final_norm(h)
        return self.lm_head(h)  # [B, L, vocab_size]
```

**关键设计要点：**

1. **输入层没有位置编码** —— 位置信息完全由 RoPE 在注意力层引入
2. **Pre-Norm 结构** —— LlamaRMSNorm 在注意力/FFN 之前，与 GPT-2 的 Post-Norm 不同
3. **所有 Linear 层不使用 bias** —— 减少参数，避免学习预训练偏置
4. **KV Cache 存储的是旋转编码后的 KV** —— 推理时无需重复计算 RoPE

## 苏格拉底时刻

请停下来思考以下问题，不急于查看答案：

1. RoPE 通过旋转来编码位置——为什么旋转操作能让注意力得分反映相对位置？试从 $R(m\theta)R^T(n\theta) = R((m-n)\theta)$ 理解。
2. GQA 的分组数 $G$ 如何选择？$G=1$ 就是 MQA，$G=H$ 就是 MHA——Llama 2 的 7B/13B 用 MHA 而 70B 用 GQA，这说明了什么？
3. RMSNorm 去掉了均值中心化，但保留了尺度不变性。这种取舍的数学本质是什么？
4. SwiGLU 的 gate 是 token-wise 独立的——不同 token 的门控有什么语义含义？与 LayerNorm 的特征选择有何不同？
5. Llama 的四项改进各自独立还是互相配合？如果只采用其中一两项，效果如何？

## 常见问题 & 面试考点

- **Q: Llama 和 GPT 的架构区别有哪些？** 四项主要改进：RoPE 替代可学习位置编码、RMSNorm 替代 LayerNorm、GQA 替代 MHA（大模型）、SwiGLU 替代 GELU FFN。此外 Llama 去掉了所有 bias。
- **Q: RoPE 如何实现长度外推？** RoPE 本身编码相对位置，天然支持一定程度的外推。更长的上下文可通过调整频率基（NTK 插值）或结合注意力缩放（YaRN）实现。
- **Q: GQA 相比 MHA 能节省多少 KV Cache？** KV Cache 缩小为原来的 $G/H$。Llama 2 70B 中 $H=64, G=8$，节省 87.5%。
- **Q: Llama 1/2/3 之间的架构差异？** Llama 1 和 2 架构基本相同（2 的大模型加了 GQA），Llama 3 的主要变化是扩大词表到 128K、增大 base frequency 以支持更长上下文。

## 推荐资源

### 论文

- **Touvron et al.《LLaMA: Open and Efficient Foundation Language Models》** — Llama 1 原始论文
- **Touvron et al.《Llama 2: Open Foundation and Fine-Tuned Chat Models》** — Llama 2 论文
- **Su et al.《RoFormer: Enhanced Transformer with Rotary Position Embedding》** — RoPE 原始论文
- **Ainslie et al.《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》** — GQA 原始论文
- **Shazeer《GLU Variants Improve Transformer》** — SwiGLU 等门控激活函数的对比实验
- **Zhang & Sennrich《Root Mean Square Layer Normalization》** — RMSNorm 原始论文

### 代码参考

- **[llama3-from-scratch](https://github.com/naklecha/llama3-from-scratch)** — naklecha 用单个 Jupyter Notebook（92 个 cell、约 500 行代码）从 Meta 官方 `consolidated.00.pth` 权重直接手撕出 Llama 3 8B 的整条推理路径，每一步都用张量形状 + 数学注释解释，是把 llama.md 里所有公式落地为可运行代码的最佳读物。它的核心特色是**不用 `nn.Module` 黑盒**，而是逐 tensor 取出 `wq / wk / wv / wo / w1 / w2 / w3 / norm`，让你看清 GQA 中 `n_heads=32, n_kv_heads=8`（每 4 个 Q 头共享一组 KV）到底是怎么对齐的。
- 关键 cell 概念锚点（按推理顺序）：
  - **cell 02-09 tokenizer 与 prompt → tokens**：`tiktoken` + `<\|begin_of_text\|>`，prompt 编码成 17 个 token（`[17]` 形状基线）
  - **cell 11 token embedding**：`tok_embeddings.weight` 查表，`[17] → [17, 4096]`，整个 notebook 唯一用到的 `nn.Module`
  - **cell 13 RMSNorm**：5 行手写 `tensor * rsqrt(mean(x^2) + eps) * weight`，与正文里 `LlamaRMSNorm` 完全对应
  - **cell 19-23 query 拆头**：`wq [4096, 4096]` reshape 为 `[32, 128, 4096]`，先取第 0 头 `[128, 4096]` 跑通 `q = x @ wq^T`
  - **cell 28-37 RoPE**：`freqs = 1 / rope_theta^(2k/d)` → `freqs_cis = polar(1, m·θ)` → `view_as_complex` 把 Q 拆成 64 对二维点 → 复数乘法实现旋转 → `view_as_real` 还原，是理解 RoPE 数学最直观的实现
  - **cell 39-45 K 的 GQA 共享**：`wk [1024, 4096]` reshape 为 `[8, 128, 4096]`，比 Q 少 4 倍，对应 `n_kv_heads=8`；K 也走同一套 RoPE
  - **cell 48-54 attention 打分 + 因果掩码**：`Q @ K^T / sqrt(128)` → `triu(-inf)` 掩码 → `softmax`，并用 `imshow` 画出 `[17, 17]` 注意力热力图
  - **cell 56-62 V 与单头输出**：V 同样 `[8, 128, 4096]`，`attn @ V` 得到单头 `[17, 128]`
  - **cell 64-70 32 头循环 + `wo` 投影**：`head//4` 索引拿到 GQA 共享的 KV 头，拼回 `[17, 4096]` 后过 `wo` 得到 attention 残差
  - **cell 74-78 SwiGLU FFN**：`w2 @ (silu(x @ w1^T) * (x @ w3^T))`，对应 Llama 命名 `w1=gate, w3=up, w2=down`
  - **cell 80 god, everything all at once**：把上面所有步骤塞进 `for layer in range(32)` 的双重循环，一次性跑完全部 32 层
  - **cell 82-90 final norm + lm_head + argmax**：`norm.weight` 做最终 RMSNorm → `output.weight [128256, 4096]` 投影 → 取最后一个 token 的 logits → `argmax` 解码出 `42`，验证整条链路正确

::: tip 配合本章使用
读到 RoPE / GQA / SwiGLU 时，对照打开 `llama3-from-scratch.ipynb` 同名 cell，用张量形状反向校对正文公式。本仓库强调"不用 `nn.Module`"，因此特别适合面试前快速过一遍权重布局与 reshape 路径。
:::

- **[minimind · `precompute_freqs_cis`（YaRN 实现）](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py#L62-L78)** — jingyaogong 在 `model/model_minimind.py` 第 62-78 行用 17 行代码完整实现了上文 YaRN 章节的 NTK-by-parts 长上下文扩展，是把伪代码落地为生产级 RoPE 的最小参考实现。它把 `rope_scaling` 当成一个可选 dict（`factor / beta_fast / beta_slow / attention_factor / original_max_position_embeddings`），与 HuggingFace `config.json` 的 YaRN 字段一一对应，可直接对接 Llama 3.1 / Qwen 2.5 等模型的权重。

```python
# minimind/model/model_minimind.py L62-L78
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
```

- 关键代码锚点（按 YaRN 三技巧对照）：
  - **L64 注释 `f'(i) = f(i)((1-γ) + γ/s)`**：一行公式给出 YaRN 频率混合的本质——γ 是分段 ramp 系数，s 是扩展倍数 `factor`，与正文公式 `freqs * (1 - ramp) + freqs/alpha * ramp` 完全等价（数学上 `1 - ramp + ramp/s = (1-γ) + γ/s`）
  - **L70 `inv_dim`**：把波长边界 `beta_fast=32 / beta_slow=1`（论文默认值）反解为频率维度索引，对应正文 "高频/低频维度分区" 的反函数实现
  - **L71-L72 `low, high` + `ramp`**：用 `clamp(0, 1)` 实现高频不动（ramp=0）/ 低频全插（ramp=1）/ 中间线性过渡的 NTK-by-parts 三段式
  - **L73 `freqs * (1 - ramp + ramp / factor)`**：单行完成正文 `freqs_yarn = freqs*(1-ramp) + freqs/alpha*ramp` 的合并形式
  - **L76-L77 `attn_factor` 乘到 cos/sin**：把 YaRN 技巧 2（Attention Scaling，温度因子 `t = 0.1·ln(α)+1`）合并进 RoPE 频率表，避免改动 attention 主路径，是工程上比"在 softmax 前再乘"更优雅的做法

::: tip 与正文对照
读完正文 165-214 行 YaRN 三技巧的伪代码后，跳到 minimind L62-L78 这 17 行真实实现，可同时验证：（1）NTK-by-parts 的 ramp 计算细节、（2）`rope_scaling` dict 字段如何对接 HuggingFace 权重、（3）attention scaling 如何合并进频率表。
:::
