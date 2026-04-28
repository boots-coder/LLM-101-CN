---
title: "Transformer 架构"
description: "完整的 Encoder-Decoder Transformer 解读，含自注意力、多头注意力、FFN"
topics: [transformer, self-attention, multi-head-attention, positional-encoding, FFN, encoder-decoder]
prereqs: [fundamentals/nlp-basics]
---
# Transformer 架构全面解读

> **一句话总结:** Transformer 是现代所有大语言模型的基础架构，它通过自注意力机制取代了循环结构，实现了真正的并行计算和长距离依赖建模。

---

## 在大模型体系中的位置

Transformer 发表于 2017 年的论文 *"Attention Is All You Need"*，是大模型时代的奠基之作。后续所有主流语言模型——BERT、GPT、Llama、DeepSeek——都是在 Transformer 基础上的变体和改进：

- **BERT**：仅使用 Encoder 部分，双向注意力，擅长理解任务
- **GPT 系列**：仅使用 Decoder 部分，单向（因果）注意力，擅长生成任务
- **T5、BART**：使用完整的 Encoder-Decoder 结构，适合翻译、摘要等 Seq2Seq 任务

理解 Transformer 是理解整个大模型技术栈的必经之路。本章将从宏观结构到每个子模块的数学原理与代码实现，逐层拆解 Transformer。

---

## 从宏观到微观：Transformer 的整体结构

Transformer 的原始设计是一个 **Encoder-Decoder** 架构，数据流如下：

```
输入序列 (src_ids)
    ↓
[Embedding + Positional Encoding]  ← 输入层：词嵌入 + 位置编码
    ↓
[Encoder × N 层]                   ← 每层包含：多头自注意力 + FFN + 残差 + LayerNorm
    ↓
编码表征 (X_src)
    ↓                              ↓
[Decoder × N 层]                   ← 每层包含：掩码自注意力 + 交叉注意力 + FFN
    ↓
[Output Layer (Linear + Softmax)]  ← 输出层：映射到目标词表
    ↓
输出概率分布
```

**Encoder** 接收源序列，通过 N 层堆叠的 EncoderBlock 提取上下文表征。每个 EncoderBlock 包含两个子层：(1) 多头自注意力，(2) 前馈神经网络，每个子层都有残差连接和 LayerNorm。

**Decoder** 接收目标序列，同样堆叠 N 层 DecoderBlock。每个 DecoderBlock 包含三个子层：(1) 带因果掩码的自注意力（防止看到未来信息），(2) 交叉注意力（关注 Encoder 的输出），(3) 前馈神经网络。

---

## 自注意力机制（Self-Attention）

### 从直觉出发

在处理自然语言时，我们需要理解每个词在上下文中的含义。例如 "苹果" 在 "吃苹果" 和 "苹果公司" 中语义完全不同。自注意力机制让每个 token 能够 **"看到"序列中的所有其他 token**，并根据相关性动态分配注意力权重。

核心思想：对于序列中的每个 token，我们问三个问题：

- **Query (Q)**：我在找什么信息？
- **Key (K)**：我有什么信息可以被别人找到？
- **Value (V)**：如果被选中，我能提供什么内容？

### 数学推导

给定输入矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个 token，每个 $d$ 维），通过三个可学习的权重矩阵进行线性投影：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$。

注意力计算公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**逐步拆解：**

1. **计算注意力分数**：$S = QK^T \in \mathbb{R}^{n \times n}$，$S_{ij}$ 表示第 $i$ 个 token 对第 $j$ 个 token 的关注程度
2. **缩放**：除以 $\sqrt{d_k}$，防止点积值过大
3. **归一化**：对每一行做 softmax，得到概率分布 $P = \text{softmax}(S / \sqrt{d_k})$
4. **加权求和**：$Z = PV$，用注意力权重对 Value 进行加权

### 为什么要除以 $\sqrt{d_k}$？——方差分析

假设 $Q$ 和 $K$ 的每个元素都是独立的、均值为 0、方差为 1 的随机变量。那么点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的：

- **均值**：$E[q \cdot k] = 0$
- **方差**：$\text{Var}(q \cdot k) = d_k$

当 $d_k$ 较大时（如 512），点积的方差会非常大，导致 softmax 的输入值分布在极端位置，梯度接近于零（softmax 饱和区）。除以 $\sqrt{d_k}$ 后，方差归一化为 1，使 softmax 工作在梯度较好的区间。

### 注意力实现代码

以下是完整的缩放点积注意力实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)  # Query 投影矩阵
        self.k_proj = nn.Linear(d_model, d_model)  # Key 投影矩阵
        self.v_proj = nn.Linear(d_model, d_model)  # Value 投影矩阵
        self.out_proj = nn.Linear(d_model, d_model) # 输出投影矩阵

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 计算注意力分数，除以 sqrt(d) 进行缩放
        scores = q @ k.transpose(-2, -1) / math.sqrt(D)

        # 掩码处理：将需要屏蔽的位置设为 -inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)  # 按行做 softmax，得到注意力权重
        out = weights @ v                          # 用注意力权重对 Value 加权求和
        return self.out_proj(out)                  # 输出投影
```

**手写 softmax 加深理解：**

```python
def softmax(X):
    """数值稳定的 softmax 实现"""
    m = torch.max(X)        # 减去最大值，防止 exp 溢出
    X_exp = torch.exp(X - m)
    L = torch.sum(X_exp)
    P = X_exp / L
    return P
```

---

## 多头注意力（Multi-Head Attention）

### 为什么需要多头？

单头注意力只能学习一种 "关注模式"。但语言中的依赖关系是多维度的：语法关系、语义关系、指代关系等。多头注意力将向量空间拆分为多个子空间，每个 "头" 独立学习不同的注意力模式，最后再合并。

### 多头如何工作？

给定 $h$ 个头，将 $d$ 维向量拆分为 $h$ 个 $d_k = d / h$ 维的子空间：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中每个头独立计算注意力：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 多头注意力实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # 每个头的维度
        self.q_proj = nn.Linear(d_model, d_model)  # 统一投影，后续拆分
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model) # 多头合并后的输出投影

    def forward(self, x_q, x_k, x_v, mask=None):
        B, L_q, D = x_q.shape
        L_k = x_k.size(1)
        L_v = x_v.size(1)

        q = self.q_proj(x_q)  # [B, L_q, D]
        k = self.k_proj(x_k)
        v = self.v_proj(x_v)

        # 拆分为多个头: [B, L, D] -> [B, num_heads, L, d_head]
        q = q.view(B, L_q, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L_k, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L_v, self.num_heads, self.d_head).transpose(1, 2)

        # 每个头独立计算注意力，注意缩放用的是 d_head
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        # 掩码处理：用 masked_fill 将屏蔽位置设为 -inf
        if mask is not None:
            # mask: [B, 1, L_q, L_k] 或 [B, L_q, L_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)          # 广播到所有头
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)  # [B, num_heads, L_q, L_k]
        out = weights @ v                          # [B, num_heads, L_q, d_head]

        # 合并多头: [B, num_heads, L_q, d_head] -> [B, L_q, D]
        out = out.transpose(1, 2).reshape(B, L_q, D)
        return self.out_proj(out)  # 最终线性投影
```

**关键细节：** 注意缩放因子使用的是 `head_dim`（单个头的维度）而非 `dim`（总维度）。因为每个头在自己的子空间中独立计算注意力，点积的维度是 `head_dim`。

---

## 前馈神经网络（Feed-Forward Network）

每个 Transformer 层中，注意力子层之后紧跟一个 **逐位置的前馈网络**（Position-wise FFN）。它对每个 token 的表征独立做相同的非线性变换。

### 结构

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d \times 4d}$：上投影（升维 4 倍）
- $W_2 \in \mathbb{R}^{4d \times d}$：下投影（降维回原始维度）

**为什么升维 4 倍？** 升维提供了更大的非线性变换空间。研究表明 FFN 层可以看作 "知识存储"，更大的中间维度意味着更强的记忆能力。FFN 的参数量通常占模型总参数的 **三分之二**。

### 实现

```python
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)    # 升维：d -> 4d
        self.act = nn.ReLU()                     # 激活函数
        self.fc2 = nn.Linear(d_ff, d_model)     # 降维：4d -> d

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))  # 升维 → 激活 → 降维
```

---

## 残差连接与 Layer Normalization

### 为什么需要残差连接？

深层网络面临 **梯度消失** 问题——误差信号在反向传播中逐层衰减。残差连接提供了一条 "捷径"，让梯度可以直接跳过中间层回传，使得训练数十甚至上百层的网络成为可能：

$$\text{output} = x + \text{SubLayer}(x)$$

### LayerNorm vs BatchNorm

**BatchNorm** 在 batch 维度上归一化，其可学习参数 $\gamma, \beta$ 的形状与特征空间 `[W x H]` 对应。对于语言模型，相当于 `[bs, seq_len]`，当序列长度变化时，BatchNorm 难以适应。

**LayerNorm** 在特征维度上归一化，对每个 token 独立计算均值和方差，天然兼容变长序列：

$$\mu_i = \frac{1}{d}\sum_{j=1}^d x_{ij}, \quad \sigma_i^2 = \frac{1}{d}\sum_{j=1}^d (x_{ij} - \mu_i)^2$$

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad \tilde{x}_{ij} = \gamma_j \hat{x}_{ij} + \beta_j$$

其中 $\gamma, \beta \in \mathbb{R}^d$ 是可学习的缩放因子和偏移量，作用于 **特征维度**（feature level）。

### LayerNorm 实现

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))   # 缩放因子，初始化为 1
        self.bias = nn.Parameter(torch.zeros(d_model))     # 偏移量，初始化为 0
        self.eps = eps

    def forward(self, x):
        # 在最后一维（特征维度）上计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化：减均值，除标准差
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 可学习的缩放和偏移（feature level）
        return self.weight * x_norm + self.bias
```

**直觉理解：** 经过 ReLU 激活函数后，数据均值非零（全为正值），LayerNorm 将每个 token 的特征重新拉回均值为 0、方差为 1 的分布，稳定训练过程。训练过程中，$\gamma$ 和 $\beta$ 会学到不同维度上的最优缩放和偏移。

### Pre-Norm vs Post-Norm

- **Post-Norm**（原始论文）：$\text{output} = \text{LayerNorm}(x + \text{SubLayer}(x))$
- **Pre-Norm**（GPT 等后续模型）：$\text{output} = x + \text{SubLayer}(\text{LayerNorm}(x))$

Pre-Norm 训练更稳定，但有研究认为 Post-Norm 的最终性能上限可能更高。

---

## 位置编码（Positional Encoding）

### 为什么需要位置编码？

自注意力机制具有 **置换不变性**：交换输入 token 的顺序，每个 token 的注意力特征也只是相应交换，序列的整体语义表示（如 sum pooling）完全不变。但语言是有序的——"我吃鱼" 和 "鱼吃我" 词相同但语义截然相反（同例见 [Transformer Quiz · 第 3 题](/exercises/transformer-quiz)）。

RNN 通过隐状态沿时间维顺序更新，把"先后关系"天然写进了递推结构里；而自注意力把整段序列一次性投到一个无序集合上算两两相似度——位置信息在这一步就丢了，必须从外面显式补回来。

### 正弦余弦编码

Transformer 原始论文使用固定的正弦余弦函数生成位置编码（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) §3.5）。对位置 $\text{pos}$、维度索引 $i \in [0, d/2)$：

$$
\begin{aligned}
PE(\text{pos}, 2i)   &= \sin\!\left(\text{pos} \cdot \theta_i\right) \\
PE(\text{pos}, 2i+1) &= \cos\!\left(\text{pos} \cdot \theta_i\right) \\
\end{aligned}
\qquad \text{其中} \quad \theta_i = \frac{1}{10000^{2i/d}}
$$

即偶数维放 sin、奇数维放 cos，相邻一对维度共享同一频率 $\theta_i$。

**设计直觉：**

- **不同维度使用不同频率**：低维频率高（变化快），高维频率低（变化慢），类似二进制编码的低位和高位
- **值域有界**：$\sin, \cos \in [-1, 1]$，不会随位置增大而爆炸
- **相对位置信息可线性恢复**：考察相邻一对维度 $(2i, 2i+1)$，把位置 $m$ 与位置 $n$ 的编码做内积，由积化和差恒等式 $\cos A \cos B + \sin A \sin B = \cos(A - B)$ 直接得到

  $$\sin(m\theta_i)\sin(n\theta_i) + \cos(m\theta_i)\cos(n\theta_i) = \cos\!\left((m - n)\theta_i\right)$$

  结果只依赖位置差 $m - n$，与绝对位置无关——这正是"相对距离"信息
- **远程衰减性**：位置越远，所有维度的 $\cos((m-n)\theta_i)$ 振荡相消，整体内积得分降低，符合语言中近距离依赖更强的直觉

### 位置编码实现

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 100, base: float = 10000.0):
        super().__init__()
        # 维度索引 [0, 1, ..., d_model/2 - 1]
        dim_idx = torch.arange(d_model // 2, dtype=torch.float)
        # 角频率: 低维变化快，高维变化慢
        freqs = 1.0 / (base ** (2 * dim_idx / d_model))
        # 位置索引
        positions = torch.arange(max_len, dtype=torch.float)
        # 外积: [max_len, d_model/2]
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        # 拼接 sin 和 cos，构造 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = angles.sin()  # 偶数维用 sin
        pe[:, 1::2] = angles.cos()  # 奇数维用 cos
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.shape[0]
        return self.pe[:seq_len, :]  # 截取所需长度
```

### 位置编码的使用

位置编码通过 **加法** 注入到词嵌入中：

$$X = \text{Embedding}(\text{input\_ids}) + PE$$

```python
class InputEmbedding(nn.Module):
    """词嵌入 + 正弦位置编码"""
    def __init__(self, vocab_size: int = 100, d_model: int = 512,
                 max_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # 预计算位置编码（与上面 SinusoidalPositionalEncoding 等价）
        pos_indices = torch.arange(max_len, dtype=torch.float)
        dim_indices = torch.arange(0, d_model, 2, dtype=torch.float)
        angles = pos_indices.unsqueeze(1) / (base ** (dim_indices / d_model))
        pos_enc = torch.zeros(max_len, d_model)
        pos_enc[:, 0::2] = angles.sin()  # 偶数维: sin
        pos_enc[:, 1::2] = angles.cos()  # 奇数维: cos
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, token_ids):
        B, L = token_ids.shape
        emb = self.tok_emb(token_ids)
        return emb + self.pos_enc[:L, :]  # 词嵌入 + 位置编码
```

---

## 完整的 Transformer 实现

将以上所有模块组装起来，以下是完整 Transformer 的核心架构代码：

### Encoder Block

```python
class EncoderLayer(nn.Module):
    """单个编码器层：多头自注意力 + FFN，均带残差连接和 LayerNorm"""
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # 子层 1：多头自注意力 + 残差连接
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        x = x + self.norm1(attn_out)
        # 子层 2：前馈网络 + 残差连接
        ff_out = self.ffn(x)
        x = x + self.norm2(ff_out)
        return x
```

### Decoder Block

```python
class DecoderLayer(nn.Module):
    """单个解码器层：掩码自注意力 + 交叉注意力 + FFN"""
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, cross_mask=None):
        # 子层 1：带因果掩码的自注意力（只能看到已生成的 token）
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = x + self.norm1(attn_out)
        # 子层 2：交叉注意力（Query 来自 Decoder，Key/Value 来自 Encoder）
        attn_out = self.cross_attn(x, enc_out, enc_out, cross_mask)
        x = x + self.norm2(attn_out)
        # 子层 3：前馈网络
        ff_out = self.ffn(x)
        x = x + self.norm3(ff_out)
        return x
```

### 完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512,
                 n_layers: int = 6, num_heads: int = 8, max_len: int = 512):
        super().__init__()
        # 输入层：词嵌入 + 位置编码（源语言和目标语言各一个）
        self.src_embed = InputEmbedding(src_vocab, d_model, max_len)
        self.tgt_embed = InputEmbedding(tgt_vocab, d_model, max_len)
        # 编码器：N 层 EncoderLayer 堆叠
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(n_layers)]
        )
        # 解码器：N 层 DecoderLayer 堆叠
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads) for _ in range(n_layers)]
        )
        # 输出层：映射到目标词表大小
        self.output_proj = nn.Linear(d_model, tgt_vocab)

    def forward(self, src_ids, tgt_ids, src_mask=None,
                tgt_mask=None, cross_mask=None):
        # === Encoder 阶段 ===
        enc = self.src_embed(src_ids)          # 词嵌入 + 位置编码
        for layer in self.enc_layers:
            enc = layer(enc, src_mask)          # 逐层编码

        # === Decoder 阶段 ===
        dec = self.tgt_embed(tgt_ids)          # 目标序列嵌入
        for layer in self.dec_layers:
            dec = layer(dec, enc, tgt_mask, cross_mask)

        # === 输出层 ===
        logits = self.output_proj(dec)         # [B, tgt_len, tgt_vocab]
        return logits
```

**核心设计要点：**

- Encoder 的自注意力是 **全局的**（每个 token 可以看到所有 token）
- Decoder 的自注意力是 **因果的**（通过 `trg_mask` 使用下三角矩阵掩码，防止看到未来 token）
- Decoder 的交叉注意力中，**Query 来自 Decoder、Key/Value 来自 Encoder**，这是信息从源语言流向目标语言的关键通道

---

## 苏格拉底时刻

1. **复杂度瓶颈**：自注意力的计算复杂度是 $O(n^2 d)$，其中 $n$ 是序列长度。这对处理超长文本有什么限制？FlashAttention、稀疏注意力等方法如何缓解？

2. **缩放的必要性**：如果不做 $\sqrt{d_k}$ 缩放，softmax 会趋向 one-hot 分布（赢者通吃），梯度几乎为零。能否用其他方式替代缩放？（提示：加性注意力）

3. **位置编码与置换**：如果去掉位置编码，输入 `[3, 8, 4]` 和 `[8, 3, 4]` 的注意力输出之和完全相同——即 Attention 对序列顺序"视而不见"。这在实际任务中意味着什么？

4. **Decoder-Only 的信息流**：在 GPT 这样的 Decoder-Only 模型中，没有 Encoder 和交叉注意力，模型如何实现"理解"和"生成"的统一？

5. **多头注意力的几何意义**：8 个头意味着 8 个独立的注意力矩阵。在训练后，不同的头是否真的学到了不同的模式？（提示：查阅注意力可视化研究）

6. **LayerNorm 的几何视角**：LayerNorm 将 $d$ 维向量投影到 $d-1$ 维的超球面上（均值为 0 的约束降了一维）。这对模型的表达能力有什么影响？

---

## 常见问题 & 面试考点

**Q1: Transformer 和 RNN 的本质区别是什么？**
> Transformer 通过自注意力在一步内建立全局依赖，而 RNN 需要逐步传递隐状态。Transformer 可以并行计算，训练速度大幅提升；但推理时 Decoder 仍是自回归的。

**Q2: 为什么注意力分数要除以 $\sqrt{d_k}$？**
> 因为 $Q$ 和 $K$ 的点积均值为 0、方差为 $d_k$。当 $d_k$ 很大时，softmax 输入值过大会导致梯度消失。除以 $\sqrt{d_k}$ 将方差归一化为 1。

**Q3: 多头注意力 vs 单头注意力，参数量是否增加？**
> 不增加。总维度 $d$ 被均分到 $h$ 个头，每头维度 $d_k = d/h$。投影矩阵的总参数量保持不变，但多头能捕获多种注意力模式。

**Q4: Pre-Norm 和 Post-Norm 有什么区别？**
> Post-Norm 将 LayerNorm 放在残差之后；Pre-Norm 放在子层之前。Pre-Norm 训练更稳定，收敛更快，是 GPT 等现代模型的标准选择。

**Q5: Encoder 和 Decoder 的掩码有什么不同？**
> Encoder 使用 padding mask（屏蔽 PAD token）；Decoder 同时使用 padding mask 和因果 mask（下三角矩阵，防止看到未来 token）。

**Q6: FFN 层的作用是什么？能不能去掉？**
> FFN 提供逐位置的非线性变换，可以看作 "知识存储"。实验表明去掉 FFN 会显著降低性能。FFN 的 4 倍升维提供了更大的表达空间。

**Q7: 为什么 Transformer 需要位置编码？**
> 自注意力具有置换不变性，无法区分 token 顺序。位置编码显式注入位置信息。正弦余弦编码的优势是：值域有界、编码相对距离（$\cos((m-n)\theta)$）、可外推到训练中未见的序列长度。

**Q8: Cross-Attention 中的 Q、K、V 分别来自哪里？**
> Query 来自 Decoder 的上一子层输出，Key 和 Value 来自 Encoder 的最终输出。这是 Encoder 信息流向 Decoder 的唯一通道。

**Q9: Transformer 的参数量如何估算？**
> 每个 Encoder/Decoder Block 的参数量约为 $12d^2$（4 个注意力矩阵各 $d^2$，FFN 两个矩阵 $4d^2 + 4d^2$）。$N$ 层总参数约 $12Nd^2$，加上 Embedding 层 $V \times d$。

**Q10: 可学习位置编码 vs 固定正弦余弦编码，哪个更好？**
> BERT 使用可学习位置编码（`nn.Embedding(max_len, d_model)`），GPT-2 也是。正弦余弦编码的理论外推性更好，但在实际的固定长度训练中，二者性能相近。现代模型（如 Llama）普遍采用 RoPE（旋转位置编码），兼具两者优势。

---

## 推荐资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始论文，必读
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar 的经典图解
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP 逐行注释实现
- [Transformer 论文逐段精读](https://www.bilibili.com/video/BV1pu411o7BE) — 李沐精读系列
- [On the Expressivity Role of LayerNorm in Transformers' Attention](https://arxiv.org/abs/2305.02582) — LayerNorm 的几何特性分析
- [Geometry and Dynamics of LayerNorm](https://arxiv.org/abs/2405.04134) — LayerNorm 变换的深入研究
